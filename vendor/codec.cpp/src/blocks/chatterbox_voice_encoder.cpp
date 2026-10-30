#include "chatterbox_voice_encoder.h"

#include "../ops/audio_frontend.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

struct ve_layout {
    int32_t n_frames = 0;    // mel frames (center=True)
    int32_t frame_step = 0;  // mel frames between partial starts
    int32_t n_partials = 0;
};

ve_layout ve_layout_for(const codec_voice_encoder_params & p, int64_t n_samples) {
    ve_layout l;
    if (n_samples <= p.n_fft / 2 || p.partial_frames <= 0 || p.hop <= 0) return l;
    l.n_frames = codec_mel_spectrogram_frames(n_samples, p.n_fft, p.hop, p.n_fft / 2);
    l.frame_step = p.rate > 0.0f
        ? (int32_t) std::lround((double) p.sample_rate / (double) p.rate / (double) p.partial_frames)
        : (int32_t) std::lround((double) p.partial_frames * (1.0 - (double) p.overlap));
    if (l.frame_step <= 0 || l.frame_step > p.partial_frames) return l;
    const int32_t numer = std::max(l.n_frames - p.partial_frames + l.frame_step, 0);
    int32_t n_wins = numer / l.frame_step;
    const int32_t remainder = numer % l.frame_step;
    if (n_wins == 0 ||
        (double) (remainder + (p.partial_frames - l.frame_step)) / (double) p.partial_frames >= (double) p.min_coverage) {
        n_wins += 1;
    }
    l.n_partials = n_wins;
    return l;
}

// F.normalize over ne[0] (≡ x / sqrt(sum x² + 1e-12)).
ggml_tensor * l2_normalize(ggml_context * ctx, ggml_tensor * x) {
    const float n = (float) x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, 1e-12f / n), 1.0f / std::sqrt(n));
}

}  // namespace

int32_t codec_voice_encoder_n_partials(const codec_voice_encoder_params & p, int64_t n_samples) {
    return ve_layout_for(p, n_samples).n_partials;
}

ggml_tensor * codec_block_chatterbox_voice_encoder(
        ggml_context * ctx, const codec_model * model, const char * prefix,
        const codec_voice_encoder_params & p, ggml_tensor * pcm) {
    if (ctx == nullptr || model == nullptr || prefix == nullptr || pcm == nullptr) return nullptr;
    const ve_layout lay = ve_layout_for(p, ggml_nelements(pcm));
    if (lay.n_partials <= 0) return nullptr;
    const std::string pre = std::string(prefix) + ".";
    bool missing = false;
    auto W = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = codec_graph_weight(ctx, model, pre + n);
        if (t == nullptr) missing = true;
        return t;
    };

    // Power mel [n_mels, frames], zero-padded / trimmed to whole partials.
    std::vector<float> window;
    ggml_tensor * win_t = codec_model_get_tensor(model, pre + "window");
    if (win_t == nullptr || !codec_tensor_as_vec_f32(win_t, &window)) return nullptr;
    const std::string lut = "lut." + std::string(prefix) + ".stft";
    ggml_tensor * mel = codec_op_mel_spectrogram(ctx, pcm, W("mel_basis"), window, p.n_fft, p.hop, p.n_fft / 2,
                                                 /*magnitude=*/false, 0.0f, lut.c_str());
    if (missing || mel == nullptr || mel->ne[0] != p.n_mels) return nullptr;
    const int32_t target = p.partial_frames + lay.frame_step * (lay.n_partials - 1);
    if (target > lay.n_frames) {
        mel = ggml_pad(ctx, mel, 0, target - lay.n_frames, 0, 0);
    }

    // Partials [n_mels, B, T]: partial b, frame t is mel frame
    // b * frame_step + t. Time outermost so each LSTM step reads a contiguous
    // [n_mels, B] slab. (Overlapping windows cannot be one ggml view.)
    const int32_t B = lay.n_partials;
    const int32_t T = p.partial_frames;
    ggml_tensor * x = nullptr;
    for (int32_t b = 0; b < B; ++b) {
        ggml_tensor * part = ggml_view_3d(ctx, mel, p.n_mels, 1, T, mel->nb[1], mel->nb[1],
                                          (size_t) b * lay.frame_step * mel->nb[1]);
        x = x == nullptr ? ggml_cont(ctx, part) : ggml_concat(ctx, x, part, 1);
    }

    // LSTM (PyTorch gate order i, f, g, o), unrolled over T and the layers.
    // Layer 0 projects all inputs at once; deeper layers per step.
    const int32_t H = p.hidden;
    std::vector<ggml_tensor *> h_seq((size_t) T, nullptr);
    for (int32_t l = 0; l < p.n_layers; ++l) {
        const std::string lp = "lstm_" + std::to_string(l) + ".";
        ggml_tensor * w_ih = codec_graph_cast_f32(ctx, W(lp + "W_ih"));
        ggml_tensor * w_hh = codec_graph_cast_f32(ctx, W(lp + "W_hh"));
        ggml_tensor * b_ih = W(lp + "b_ih");
        ggml_tensor * b_hh = W(lp + "b_hh");
        if (missing) return nullptr;
        ggml_tensor * bias = ggml_add(ctx, codec_graph_cast_f32(ctx, b_ih), codec_graph_cast_f32(ctx, b_hh));
        ggml_tensor * wx_all = nullptr;
        if (l == 0) {
            wx_all = codec_mul_mat(ctx, w_ih, ggml_reshape_2d(ctx, x, p.n_mels, (int64_t) B * T));   // [4H, B * T]
            wx_all = ggml_add(ctx, wx_all, bias);
        }
        ggml_tensor * h = nullptr;
        ggml_tensor * c = nullptr;
        for (int32_t t = 0; t < T; ++t) {
            ggml_tensor * gates = wx_all != nullptr
                ? ggml_view_2d(ctx, wx_all, 4 * H, B, wx_all->nb[1], (size_t) t * B * wx_all->nb[1])
                : ggml_add(ctx, codec_mul_mat(ctx, w_ih, h_seq[(size_t) t]), bias);
            if (h != nullptr) gates = ggml_add(ctx, gates, codec_mul_mat(ctx, w_hh, h));
            ggml_tensor * sg = ggml_sigmoid(ctx, gates);
            ggml_tensor * tg = ggml_tanh(ctx, gates);
            auto gate = [&](ggml_tensor * a, int k) {
                return ggml_view_2d(ctx, a, H, B, a->nb[1], (size_t) k * H * ggml_element_size(a));
            };
            ggml_tensor * ig = ggml_mul(ctx, gate(sg, 0), gate(tg, 2));
            c = c != nullptr ? ggml_add(ctx, ggml_mul(ctx, gate(sg, 1), c), ig) : ggml_cont(ctx, ig);
            h = ggml_mul(ctx, gate(sg, 3), ggml_tanh(ctx, c));                               // [H, B]
            h_seq[(size_t) t] = h;
        }
    }

    // Final hidden state → Linear (+ ReLU) → L2 per partial → mean → L2.
    ggml_tensor * e = codec_mul_mat(ctx, codec_graph_cast_f32(ctx, W("proj.weight")), h_seq.back());
    if (missing) return nullptr;
    e = ggml_add(ctx, e, codec_graph_cast_f32(ctx, W("proj.bias")));                      // [E, B]
    if (p.final_relu) e = ggml_relu(ctx, e);
    e = l2_normalize(ctx, e);
    e = ggml_mean(ctx, ggml_cont(ctx, ggml_transpose(ctx, e)));                           // [1, E]
    return l2_normalize(ctx, ggml_reshape_1d(ctx, e, p.embed));
}
