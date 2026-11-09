#include "ecapa_tdnn.h"

#include "../ops/audio_frontend.h"
#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <string>
#include <vector>

namespace {

struct ecapa_builder {
    ggml_context * ctx;
    const codec_model * model;
    std::string pre;
    bool missing = false;

    ggml_tensor * weight(const std::string & n) {
        ggml_tensor * t = codec_graph_weight(ctx, model, pre + n);
        if (t == nullptr) missing = true;
        return t;
    }

    // Conv1d over x [t, c] with reflect "same" padding. Weights are cast to
    // F32 so activations are not rounded to F16 on the way in.
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int32_t dilation) {
        ggml_tensor * w = weight(name + ".weight");
        ggml_tensor * b = weight(name + ".bias");
        if (missing || x == nullptr) return nullptr;
        const int32_t pad = (int32_t) ((codec_conv1d_kernel_size(w, x->ne[1]) - 1) * dilation) / 2;
        if (pad > 0) x = codec_op_pad_reflect_tc(ctx, x, pad, pad);
        return codec_conv1d(ctx, x, codec_graph_cast_f32(ctx, w), b, 1, dilation, 0);
    }

    ggml_tensor * tdnn(ggml_tensor * x, const std::string & name, int32_t dilation) {
        ggml_tensor * y = conv(x, name, dilation);
        return y != nullptr ? ggml_relu(ctx, y) : nullptr;
    }

    // Res2Net: split channels into `scale` chunks; chunk i > 0 goes through
    // its TDNN after adding the previous chunk's output (from i = 2 on).
    ggml_tensor * res2net(ggml_tensor * x, const std::string & bp, int32_t scale, int32_t dilation) {
        const int64_t t = x->ne[0];
        const int64_t chunk = x->ne[1] / scale;
        ggml_tensor * out = nullptr;
        ggml_tensor * prev = nullptr;
        for (int32_t i = 0; i < scale; ++i) {
            ggml_tensor * xi = ggml_cont(ctx, ggml_view_2d(ctx, x, t, chunk, x->nb[1], (size_t) i * chunk * x->nb[1]));
            ggml_tensor * yi = xi;
            if (i > 0) {
                yi = tdnn(i > 1 ? ggml_add(ctx, xi, prev) : xi, bp + ".res2net." + std::to_string(i - 1) + ".conv", dilation);
                if (yi == nullptr) return nullptr;
                prev = yi;
            }
            out = out == nullptr ? yi : ggml_concat(ctx, out, yi, 1);
        }
        return out;
    }

    // Squeeze-excitation: channel gate from the time mean.
    ggml_tensor * se(ggml_tensor * x, const std::string & bp) {
        ggml_tensor * g = tdnn(ggml_mean(ctx, x), bp + ".se.conv1", 1);                 // [1, se_ch]
        g = conv(g, bp + ".se.conv2", 1);
        return g != nullptr ? ggml_mul(ctx, x, ggml_sigmoid(ctx, g)) : nullptr;
    }
};

// mean / std over time (ne[0]) of x [t, c], weighted by `w` ([t, c], rows
// summing to 1) or uniform when w is null. std = sqrt(max(var, 1e-12)).
void stats(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor ** mean, ggml_tensor ** std_) {
    ggml_tensor * m = w != nullptr ? ggml_sum_rows(ctx, ggml_mul(ctx, x, w)) : ggml_mean(ctx, x);   // [1, c]
    ggml_tensor * d2 = ggml_sqr(ctx, ggml_sub(ctx, x, m));
    ggml_tensor * v = w != nullptr ? ggml_sum_rows(ctx, ggml_mul(ctx, d2, w)) : ggml_mean(ctx, d2);
    *mean = m;
    *std_ = ggml_sqrt(ctx, codec_op_clamp_min(ctx, v, 1e-12f));
}

}  // namespace

int32_t codec_ecapa_n_frames(const codec_ecapa_params & p, int64_t n_samples) {
    const int32_t pad = (p.n_fft - p.hop) / 2;
    if (n_samples <= pad) return 0;
    return codec_mel_spectrogram_frames(n_samples, p.n_fft, p.hop, pad);
}

ggml_tensor * codec_block_ecapa_tdnn(
        ggml_context * ctx, const codec_model * model, const char * prefix,
        const codec_ecapa_params & p, ggml_tensor * pcm) {
    if (ctx == nullptr || model == nullptr || prefix == nullptr || pcm == nullptr ||
        p.n_blocks < 3 || p.n_blocks > kCodecEcapaMaxBlocks || codec_ecapa_n_frames(p, ggml_nelements(pcm)) < 2) {
        return nullptr;
    }
    ecapa_builder b{ctx, model, std::string(prefix) + "."};

    // Log magnitude mel [t, n_mels].
    std::vector<float> window;
    ggml_tensor * win_t = codec_model_get_tensor(model, b.pre + "window");
    if (win_t == nullptr || !codec_tensor_as_vec_f32(win_t, &window)) return nullptr;
    const std::string lut = "lut." + std::string(prefix) + ".stft";
    ggml_tensor * mel = codec_op_mel_spectrogram(ctx, pcm, b.weight("mel_basis"), window, p.n_fft, p.hop,
                                                 (p.n_fft - p.hop) / 2, /*magnitude=*/true, 0.0f, lut.c_str());
    if (b.missing || mel == nullptr || mel->ne[0] != p.n_mels) return nullptr;
    mel = ggml_log(ctx, codec_op_clamp_min(ctx, mel, 1e-5f));
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, mel));

    // TDNN, then SE-Res2Net blocks (residual when the width is unchanged).
    x = b.tdnn(x, "blocks.0.conv", p.dilations[0]);
    ggml_tensor * mfa_in = nullptr;
    for (int32_t i = 1; i < p.n_blocks - 1 && x != nullptr; ++i) {
        const std::string bp = "blocks." + std::to_string(i);
        ggml_tensor * y = b.tdnn(x, bp + ".tdnn1.conv", 1);
        if (y != nullptr) y = b.res2net(y, bp, p.res2net_scale, p.dilations[i]);
        if (y != nullptr) y = b.tdnn(y, bp + ".tdnn2.conv", 1);
        if (y != nullptr) y = b.se(y, bp);
        if (y == nullptr) return nullptr;
        x = y->ne[1] == x->ne[1] ? ggml_add(ctx, y, x) : y;
        mfa_in = mfa_in == nullptr ? x : ggml_concat(ctx, mfa_in, x, 1);
    }
    x = b.tdnn(mfa_in, "mfa.conv", 1);                                              // [t, C]
    if (x == nullptr) return nullptr;

    // Attentive statistics pooling: attention over [x, mean, std] (global
    // stats broadcast over time), softmax over time per channel.
    ggml_tensor * gm = nullptr;
    ggml_tensor * gs = nullptr;
    stats(ctx, x, nullptr, &gm, &gs);
    ggml_tensor * attn_in = ggml_concat(ctx, ggml_concat(ctx, x, ggml_repeat(ctx, gm, x), 1), ggml_repeat(ctx, gs, x), 1);
    ggml_tensor * a = b.tdnn(attn_in, "asp.tdnn.conv", 1);
    if (a == nullptr) return nullptr;
    a = b.conv(ggml_tanh(ctx, a), "asp.conv", 1);
    if (a == nullptr) return nullptr;
    a = ggml_soft_max(ctx, a);                                                      // over time
    ggml_tensor * pm = nullptr;
    ggml_tensor * ps = nullptr;
    stats(ctx, x, a, &pm, &ps);
    ggml_tensor * pooled = ggml_concat(ctx, pm, ps, 1);                             // [1, 2C]

    ggml_tensor * e = b.conv(pooled, "fc", 1);                                      // [1, enc_dim]
    return e != nullptr ? ggml_reshape_1d(ctx, ggml_cont(ctx, e), e->ne[1]) : nullptr;
}
