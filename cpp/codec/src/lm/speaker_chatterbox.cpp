#include "speaker_chatterbox.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"
#include "../runtime/audio_dsp.h"
#include "../ops/lm_ggml_ops.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// Chatterbox speaker encoder runtime.
//
// Hybrid CPU + ggml-graph pipeline:
//
//   ref_pcm (16 kHz mono)
//     │  CPU: codec_runtime_chatterbox_ve_mel_partials
//     ▼
//   mel_partials  (n_partials, 160, 40)
//     │  CPU: 3-layer LSTM unroll (batched over partials)
//     │  CPU: per-partial projection + ReLU + L2 norm
//     │  CPU: mean over partials + L2 norm
//     ▼
//   spk_emb_raw  (256,)
//     │  GRAPH: cond_enc + perceiver + concat
//     │      spkr_enc(spk_emb_raw)                       → cond_spkr      (1,    H=1024)
//     │      perceiver(speech_emb(ref_tokens) + pos)     → cond_speech    (32,   H=1024)
//     │      emotion_adv_fc(emotion_scalar)              → cond_emotion   (1,    H=1024)
//     │      concat([cond_spkr, cond_speech, cond_emot]) → cond_emb       (34,   H=1024)
//     ▼
//   cond_emb  (34, 1024)  →  user-provided out buffer
//
// The CPU front-end is intentionally not in a ggml graph: the LSTM
// would need 480+ unrolled timesteps × 3 layers (160 timesteps per
// partial), the mel front-end is a one-shot CPU-bound DFT, and this
// whole pipeline runs ONCE per `prepare_conditionals` (i.e. once per
// ref-audio clip).  The cond_enc + perceiver side is small but warrants
// the graph cache because callers may re-encode with different
// `emotion` values (Chatterbox upstream re-runs cond_enc every
// `generate(exaggeration=…)` call) and we want the heavy projections
// cached.
// =====================================================================

namespace {

struct cbx_impl {
    // VE shape constants (from GGUF metadata).
    int32_t n_mels        = 40;
    int32_t hidden_size   = 256;
    int32_t num_layers    = 3;
    int32_t embed_size    = 256;
    int32_t n_fft         = 400;
    int32_t hop_size      = 160;
    int32_t win_size      = 400;
    int32_t partial_frames = 160;
    int32_t sample_rate   = 16000;
    float   overlap       = 0.5f;
    float   rate          = 1.3f;
    float   min_coverage  = 0.8f;
    bool    final_relu    = true;

    int32_t hidden_dim    = 1024;   // backbone hidden (= cond_emb row width)
    int32_t n_rows        = 34;
    int32_t speaker_embed_dim = 256;

    // Mel basis + window (F32 host buffers, dequantised at init).
    std::vector<float> mel_basis;   // (n_freq * n_mels)
    std::vector<float> window;      // (n_fft)
    int32_t n_freq = 0;

    // VE LSTM weights laid out row-major (4H × in_dim), gate order
    // [i, f, g, o] per PyTorch nn.LSTM convention.  Float32 throughout
    // (5.5 MB total — trivial).
    struct lstm_layer {
        std::vector<float> W_ih;     // 4H × in_dim
        std::vector<float> W_hh;     // 4H × H
        std::vector<float> b_ih;     // 4H
        std::vector<float> b_hh;     // 4H
        int32_t            in_dim = 0;
    };
    std::vector<lstm_layer> lstm;

    std::vector<float> proj_W;       // E × H
    std::vector<float> proj_b;       // E

    // cond_enc graph tensors (handles into model->weights).
    lm_ggml_tensor * spkr_enc_W       = nullptr; // (E_speaker=256, H=1024) F16/F32
    lm_ggml_tensor * spkr_enc_b       = nullptr; // (H,)
    lm_ggml_tensor * emotion_adv_W    = nullptr; // (1, H)
    lm_ggml_tensor * speech_emb       = nullptr; // (vocab, H)
    lm_ggml_tensor * speech_pos_emb   = nullptr; // (max_speech_tokens, H)
    lm_ggml_tensor * perceiver_q      = nullptr; // (1, 32, H) — 32 learned queries
    lm_ggml_tensor * perc_norm_W      = nullptr; // (H,)
    lm_ggml_tensor * perc_norm_b      = nullptr; // (H,)
    lm_ggml_tensor * perc_to_q_W      = nullptr; // (H, H)
    lm_ggml_tensor * perc_to_q_b      = nullptr; // (H,)
    lm_ggml_tensor * perc_to_k_W      = nullptr; // (H, H)
    lm_ggml_tensor * perc_to_k_b      = nullptr; // (H,)
    lm_ggml_tensor * perc_to_v_W      = nullptr; // (H, H)
    lm_ggml_tensor * perc_to_v_b      = nullptr; // (H,)
    lm_ggml_tensor * perc_proj_out_W  = nullptr; // (H, H)
    lm_ggml_tensor * perc_proj_out_b  = nullptr; // (H,)

    // Lazily-allocated context for the cond_enc graph (one per lm, owned).
    codec_context * cond_ctx = nullptr;
};

lm_ggml_tensor * find_required(codec_lm * lm, const char * name) {
    lm_ggml_tensor * t = lm_ggml_get_tensor(lm->codec->weights, name);
    if (t == nullptr) {
        lm->last_error = std::string("speaker(chatterbox): missing tensor: ") + name;
    }
    return t;
}

bool load_lstm_layer_weights(
        codec_lm * lm, int32_t l, cbx_impl::lstm_layer & w) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "speaker.voice_encoder.lstm_%d.W_ih", l);
    lm_ggml_tensor * w_ih = find_required(lm, buf); if (!w_ih) return false;
    std::snprintf(buf, sizeof(buf), "speaker.voice_encoder.lstm_%d.W_hh", l);
    lm_ggml_tensor * w_hh = find_required(lm, buf); if (!w_hh) return false;
    std::snprintf(buf, sizeof(buf), "speaker.voice_encoder.lstm_%d.b_ih", l);
    lm_ggml_tensor * b_ih = find_required(lm, buf); if (!b_ih) return false;
    std::snprintf(buf, sizeof(buf), "speaker.voice_encoder.lstm_%d.b_hh", l);
    lm_ggml_tensor * b_hh = find_required(lm, buf); if (!b_hh) return false;

    if (!codec_tensor_as_vec_f32(w_ih, &w.W_ih) ||
        !codec_tensor_as_vec_f32(w_hh, &w.W_hh) ||
        !codec_tensor_as_vec_f32(b_ih, &w.b_ih) ||
        !codec_tensor_as_vec_f32(b_hh, &w.b_hh)) {
        lm->last_error = std::string("speaker(chatterbox): dequant failed for lstm_") + std::to_string(l);
        return false;
    }
    w.in_dim = (int32_t) w_ih->ne[0];
    return true;
}

}  // namespace

// ---------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------

bool chatterbox_speaker_init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr) return false;
    if (lm->codec->gguf == nullptr || lm->codec->weights == nullptr) {
        lm->last_error = "speaker(chatterbox): codec has no gguf/weights";
        return false;
    }
    lm_gguf_context * gf = lm->codec->gguf;

    cbx_impl * impl = new (std::nothrow) cbx_impl();
    if (impl == nullptr) {
        lm->last_error = "speaker(chatterbox): out of memory";
        return false;
    }

    impl->n_mels         = codec_read_i32_kv(gf, "codec.speaker.ve.num_mels", 40);
    impl->hidden_size    = codec_read_i32_kv(gf, "codec.speaker.ve.hidden_size", 256);
    impl->num_layers     = codec_read_i32_kv(gf, "codec.speaker.ve.num_layers", 3);
    impl->embed_size     = codec_read_i32_kv(gf, "codec.speaker.ve.speaker_embed_dim", 256);
    impl->n_fft          = codec_read_i32_kv(gf, "codec.speaker.ve.n_fft", 400);
    impl->hop_size       = codec_read_i32_kv(gf, "codec.speaker.ve.hop_size", 160);
    impl->win_size       = codec_read_i32_kv(gf, "codec.speaker.ve.win_size", 400);
    impl->partial_frames = codec_read_i32_kv(gf, "codec.speaker.ve.partial_frames", 160);
    impl->sample_rate    = lm->speaker_info.ref_sample_rate;
    impl->overlap        = codec_read_f32_kv(gf, "codec.speaker.ve.overlap", 0.5f);
    impl->rate           = codec_read_f32_kv(gf, "codec.speaker.ve.rate", 1.3f);
    impl->min_coverage   = codec_read_f32_kv(gf, "codec.speaker.ve.min_coverage", 0.8f);
    impl->final_relu     = codec_read_bool_kv(gf, "codec.speaker.ve.final_relu", true);
    impl->n_freq         = impl->n_fft / 2 + 1;
    impl->hidden_dim     = lm->speaker_info.hidden_dim;
    impl->n_rows         = lm->speaker_info.n_rows;
    impl->speaker_embed_dim = impl->embed_size;

    // VE LSTM layers.
    impl->lstm.resize((size_t) impl->num_layers);
    for (int32_t l = 0; l < impl->num_layers; ++l) {
        if (!load_lstm_layer_weights(lm, l, impl->lstm[(size_t) l])) {
            delete impl; return false;
        }
    }

    // VE projection (256 → 256).
    lm_ggml_tensor * pw = find_required(lm, "speaker.voice_encoder.proj.weight");
    lm_ggml_tensor * pb = find_required(lm, "speaker.voice_encoder.proj.bias");
    if (!pw || !pb) { delete impl; return false; }
    if (!codec_tensor_as_vec_f32(pw, &impl->proj_W) ||
        !codec_tensor_as_vec_f32(pb, &impl->proj_b)) {
        lm->last_error = "speaker(chatterbox): proj dequant failed";
        delete impl; return false;
    }

    // Baked mel basis + Hann window — host-side buffers used by the
    // CPU mel front-end.
    lm_ggml_tensor * mb_t = find_required(lm, "speaker.voice_encoder.mel_basis");
    lm_ggml_tensor * wn_t = find_required(lm, "speaker.voice_encoder.window");
    if (!mb_t || !wn_t) { delete impl; return false; }
    if (!codec_tensor_as_vec_f32(mb_t, &impl->mel_basis) ||
        !codec_tensor_as_vec_f32(wn_t, &impl->window)) {
        lm->last_error = "speaker(chatterbox): mel/window dequant failed";
        delete impl; return false;
    }

    // cond_enc / perceiver weights (live under `lm.chatterbox.cond.*` —
    // bundled by the LM adaptor section, so this is only ever called for
    // GGUFs that also have the LM adaptor; the speaker section is
    // gated on `lm_source` at convert time).
    impl->spkr_enc_W      = find_required(lm, "lm.chatterbox.cond.spkr_enc.weight");
    impl->spkr_enc_b      = find_required(lm, "lm.chatterbox.cond.spkr_enc.bias");
    impl->emotion_adv_W   = find_required(lm, "lm.chatterbox.cond.emotion_adv_fc.weight");
    impl->speech_emb      = find_required(lm, "lm.audio_embd_0.weight");
    impl->speech_pos_emb  = find_required(lm, "lm.chatterbox.speech_pos_emb.weight");
    impl->perceiver_q     = find_required(lm, "lm.chatterbox.cond.perceiver.queries");
    impl->perc_norm_W     = find_required(lm, "lm.chatterbox.cond.perceiver.norm.weight");
    impl->perc_norm_b     = find_required(lm, "lm.chatterbox.cond.perceiver.norm.bias");
    impl->perc_to_q_W     = find_required(lm, "lm.chatterbox.cond.perceiver.to_q.weight");
    impl->perc_to_q_b     = find_required(lm, "lm.chatterbox.cond.perceiver.to_q.bias");
    impl->perc_to_k_W     = find_required(lm, "lm.chatterbox.cond.perceiver.to_k.weight");
    impl->perc_to_k_b     = find_required(lm, "lm.chatterbox.cond.perceiver.to_k.bias");
    impl->perc_to_v_W     = find_required(lm, "lm.chatterbox.cond.perceiver.to_v.weight");
    impl->perc_to_v_b     = find_required(lm, "lm.chatterbox.cond.perceiver.to_v.bias");
    impl->perc_proj_out_W = find_required(lm, "lm.chatterbox.cond.perceiver.proj_out.weight");
    impl->perc_proj_out_b = find_required(lm, "lm.chatterbox.cond.perceiver.proj_out.bias");
    if (!impl->spkr_enc_W || !impl->spkr_enc_b || !impl->emotion_adv_W ||
        !impl->speech_emb || !impl->speech_pos_emb || !impl->perceiver_q ||
        !impl->perc_norm_W || !impl->perc_norm_b ||
        !impl->perc_to_q_W || !impl->perc_to_q_b ||
        !impl->perc_to_k_W || !impl->perc_to_k_b ||
        !impl->perc_to_v_W || !impl->perc_to_v_b ||
        !impl->perc_proj_out_W || !impl->perc_proj_out_b) {
        delete impl; return false;
    }

    lm->speaker_impl = impl;
    return true;
}

void chatterbox_speaker_free(codec_lm * lm) {
    if (lm == nullptr || lm->speaker_impl == nullptr) return;
    cbx_impl * impl = static_cast<cbx_impl *>(lm->speaker_impl);
    if (impl->cond_ctx != nullptr) {
        codec_runtime_free(impl->cond_ctx);
        delete impl->cond_ctx;
        impl->cond_ctx = nullptr;
    }
    delete impl;
    lm->speaker_impl = nullptr;
}

// ---------------------------------------------------------------------
// CPU: LSTM forward (3 layers, batched over partials)
// ---------------------------------------------------------------------
//
// PyTorch nn.LSTM convention (gate order in the concatenated 4H slice):
//   gates = [i, f, g, o]   (input, forget, candidate, output)
//   c_t = f * c_{t-1} + i * g
//   h_t = o * tanh(c_t)
//   where i, f, o use sigmoid and g uses tanh.

namespace {

inline float sigmoidf(float x) {
    // Stable two-piece sigmoid.
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = std::exp(x);
        return z / (1.0f + z);
    }
}

// y[B, 4H] = X[B, in] @ W^T + b   ; B = n_partials.  X laid out
// (B, in) row-major; W is (4H, in) row-major (PyTorch convention).
void linear_4H_T(
        const float * X, int32_t B, int32_t in_dim,
        const float * W, const float * b, int32_t H4,
        float * Y) {
    for (int32_t bi = 0; bi < B; ++bi) {
        for (int32_t r = 0; r < H4; ++r) {
            const float * w_row = W + (size_t) r * (size_t) in_dim;
            float acc = b[r];
            const float * x_row = X + (size_t) bi * (size_t) in_dim;
            for (int32_t k = 0; k < in_dim; ++k) {
                acc += w_row[k] * x_row[k];
            }
            Y[(size_t) bi * (size_t) H4 + (size_t) r] = acc;
        }
    }
}

// Run one LSTM layer over T timesteps for a B-sized batch.  Input
// `X_in` is (T, B, in_dim) row-major; output `H_out` is (T, B, H).
void lstm_layer_forward(
        const cbx_impl::lstm_layer & w,
        const float * X_in,
        int32_t T, int32_t B, int32_t H,
        float * H_out) {
    const int32_t H4 = 4 * H;
    const int32_t in_dim = w.in_dim;

    // Per-timestep scratch buffers.
    std::vector<float> Wx((size_t) B * (size_t) H4, 0.0f);
    std::vector<float> Wh((size_t) B * (size_t) H4, 0.0f);
    std::vector<float> h ((size_t) B * (size_t) H,  0.0f);
    std::vector<float> c ((size_t) B * (size_t) H,  0.0f);
    std::vector<float> h_prev((size_t) B * (size_t) H, 0.0f);

    for (int32_t t = 0; t < T; ++t) {
        const float * x_t = X_in + (size_t) t * (size_t) B * (size_t) in_dim;
        linear_4H_T(x_t,       B, in_dim, w.W_ih.data(), w.b_ih.data(), H4, Wx.data());
        linear_4H_T(h_prev.data(), B, H,  w.W_hh.data(), w.b_hh.data(), H4, Wh.data());

        for (int32_t bi = 0; bi < B; ++bi) {
            const float * gx = Wx.data() + (size_t) bi * (size_t) H4;
            const float * gh = Wh.data() + (size_t) bi * (size_t) H4;
            float * c_b = c.data() + (size_t) bi * (size_t) H;
            float * h_b = h.data() + (size_t) bi * (size_t) H;
            for (int32_t k = 0; k < H; ++k) {
                const float i_pre = gx[k]       + gh[k];
                const float f_pre = gx[H + k]   + gh[H + k];
                const float g_pre = gx[2*H + k] + gh[2*H + k];
                const float o_pre = gx[3*H + k] + gh[3*H + k];

                const float i_g = sigmoidf(i_pre);
                const float f_g = sigmoidf(f_pre);
                const float g_v = std::tanh(g_pre);
                const float o_g = sigmoidf(o_pre);

                c_b[k] = f_g * c_b[k] + i_g * g_v;
                h_b[k] = o_g * std::tanh(c_b[k]);
            }
        }
        // Copy h → H_out and h_prev.
        std::memcpy(
            H_out + (size_t) t * (size_t) B * (size_t) H,
            h.data(),
            (size_t) B * (size_t) H * sizeof(float));
        std::memcpy(h_prev.data(), h.data(),
                    (size_t) B * (size_t) H * sizeof(float));
    }
}

void l2_normalize_rows(float * X, int32_t B, int32_t D) {
    for (int32_t bi = 0; bi < B; ++bi) {
        float * row = X + (size_t) bi * (size_t) D;
        double sq = 0.0;
        for (int32_t k = 0; k < D; ++k) sq += (double) row[k] * (double) row[k];
        const float inv = (float) (1.0 / std::sqrt(sq + 1e-12));
        for (int32_t k = 0; k < D; ++k) row[k] *= inv;
    }
}

}  // namespace

// ---------------------------------------------------------------------
// cond_enc + perceiver graph
// ---------------------------------------------------------------------

namespace {

struct cbx_cond_build_data {
    cbx_impl * impl;
    int32_t    T_speech;
};

// Multi-head non-causal attention block matching upstream
// `AttentionBlock2.forward(x1, x2)`:
//
//   x1_norm = LayerNorm(x1) ; x2_norm = LayerNorm(x2)
//   q = to_q(x1_norm); k = to_k(x2_norm); v = to_v(x2_norm)
//   split into n_heads × head_dim
//   sim = q · k^T * scale  ;  attn = softmax(sim) ; out = attn · v
//   merge heads
//   h = proj_out(out)
//   return x1 + h
//
// Shapes (T_q = T_k for self-attn, T_q = 32, T_k = T_speech for cross):
//   x1 : (T_q, H)
//   x2 : (T_k, H)
//
// `n_heads = 4`, `head_dim = H / n_heads = 256` (Perceiver default).
//
// scale = head_dim^-0.5 (upstream default when scale=None).
lm_ggml_tensor * perceiver_attn_block(
        lm_ggml_context * ctx,
        cbx_impl * impl,
        lm_ggml_tensor * x1,        // (H, T_q)
        lm_ggml_tensor * x2,        // (H, T_k)
        int32_t n_heads) {
    const int64_t H   = x1->ne[0];
    const int64_t T_q = x1->ne[1];
    const int64_t T_k = x2->ne[1];
    const int64_t head_dim = H / n_heads;
    const float   scale    = 1.0f / std::sqrt((float) head_dim);

    lm_ggml_tensor * norm_W = codec_graph_cast_f32(ctx, impl->perc_norm_W);
    lm_ggml_tensor * norm_b = codec_graph_cast_f32(ctx, impl->perc_norm_b);

    lm_ggml_tensor * x1_norm = lm_ggml_norm(ctx, x1, 1e-5f);
    x1_norm = lm_ggml_add(ctx, lm_ggml_mul(ctx, x1_norm, lm_ggml_repeat(ctx,
                lm_ggml_reshape_2d(ctx, norm_W, H, 1), x1_norm)),
                lm_ggml_repeat(ctx, lm_ggml_reshape_2d(ctx, norm_b, H, 1), x1_norm));
    lm_ggml_tensor * x2_norm = lm_ggml_norm(ctx, x2, 1e-5f);
    x2_norm = lm_ggml_add(ctx, lm_ggml_mul(ctx, x2_norm, lm_ggml_repeat(ctx,
                lm_ggml_reshape_2d(ctx, norm_W, H, 1), x2_norm)),
                lm_ggml_repeat(ctx, lm_ggml_reshape_2d(ctx, norm_b, H, 1), x2_norm));

    // Linear projections: q/k/v.  Weights are (in=H, out=H) stored as
    // (out, in) row-major → ggml ne[0]=H_in, ne[1]=H_out — matches the
    // mul_mat src[0] convention.
    auto linear_HH = [&](lm_ggml_tensor * W, lm_ggml_tensor * b, lm_ggml_tensor * x_2d) {
        lm_ggml_tensor * W_lhs = codec_graph_mat_lhs(ctx, W);
        lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, W_lhs, x_2d);   // (H_out, T)
        lm_ggml_tensor * b_f32 = codec_graph_cast_f32(ctx, b);
        lm_ggml_tensor * b_2d  = lm_ggml_reshape_2d(ctx, b_f32, H, 1);
        y = lm_ggml_add(ctx, y, lm_ggml_repeat(ctx, b_2d, y));
        return y;
    };
    lm_ggml_tensor * q = linear_HH(impl->perc_to_q_W, impl->perc_to_q_b, x1_norm);  // (H, T_q)
    lm_ggml_tensor * k = linear_HH(impl->perc_to_k_W, impl->perc_to_k_b, x2_norm);  // (H, T_k)
    lm_ggml_tensor * v = linear_HH(impl->perc_to_v_W, impl->perc_to_v_b, x2_norm);  // (H, T_k)

    // Split into heads: (H, T) → (head_dim, n_heads, T) → permute to
    // (head_dim, T, n_heads) for attention math (lm_ggml_mul_mat on ne[0]).
    auto split_heads = [&](lm_ggml_tensor * t, int64_t T) {
        lm_ggml_tensor * t3 = lm_ggml_reshape_3d(ctx, t, head_dim, n_heads, T);
        return lm_ggml_cont(ctx, lm_ggml_permute(ctx, t3, 0, 2, 1, 3));  // (head_dim, T, n_heads)
    };
    lm_ggml_tensor * q_p = split_heads(q, T_q);   // (head_dim, T_q, n_heads)
    lm_ggml_tensor * k_p = split_heads(k, T_k);
    lm_ggml_tensor * v_p = split_heads(v, T_k);

    // sim = k_p^T @ q_p : lm_ggml_mul_mat(a=k_p, b=q_p) contracts on ne[0]=head_dim
    //   → (T_k, T_q, n_heads)
    lm_ggml_tensor * sim = lm_ggml_mul_mat(ctx, k_p, q_p);
    sim = lm_ggml_scale(ctx, sim, scale);
    lm_ggml_tensor * attn = lm_ggml_soft_max(ctx, sim);              // softmax over ne[0]=T_k

    // out = v_p^T @ attn : but we want (head_dim, T_q, n_heads).
    // v_p is (head_dim, T_k, n_heads), attn is (T_k, T_q, n_heads).
    // lm_ggml_mul_mat needs ne[0]s to match — permute v_p to (T_k, head_dim, n_heads)
    // so contraction is on T_k, result (head_dim, T_q, n_heads).
    lm_ggml_tensor * v_pT = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_p, 1, 0, 2, 3));
    lm_ggml_tensor * out  = lm_ggml_mul_mat(ctx, v_pT, attn);          // (head_dim, T_q, n_heads)

    // Merge heads: (head_dim, T_q, n_heads) → (head_dim, n_heads, T_q) → (H, T_q).
    lm_ggml_tensor * out_merged = lm_ggml_cont(ctx, lm_ggml_permute(ctx, out, 0, 2, 1, 3));
    out_merged = lm_ggml_reshape_2d(ctx, out_merged, H, T_q);

    // proj_out + residual.
    lm_ggml_tensor * h = linear_HH(impl->perc_proj_out_W, impl->perc_proj_out_b, out_merged);
    return lm_ggml_add(ctx, x1, h);
}

bool build_cond_graph(lm_ggml_context * ctx_eval, void * ud, lm_ggml_tensor ** out) {
    auto * b = static_cast<cbx_cond_build_data *>(ud);
    if (ctx_eval == nullptr || b == nullptr || b->impl == nullptr || out == nullptr) return false;
    cbx_impl * impl = b->impl;
    const int64_t H  = impl->hidden_dim;
    const int64_t T_speech = b->T_speech;

    // ---- Inputs ---------------------------------------------------
    lm_ggml_tensor * t_spk     = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, impl->speaker_embed_dim);
    lm_ggml_set_name(t_spk, "spk.in.spkr_emb");
    lm_ggml_tensor * t_tokens  = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, T_speech);
    lm_ggml_set_name(t_tokens, "spk.in.tokens");
    lm_ggml_tensor * t_emotion = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, 1);
    lm_ggml_set_name(t_emotion, "spk.in.emotion");

    // ---- cond_spkr = spkr_enc(spk_emb_raw) ----------------------
    //   spkr_enc.weight is (256, 1024) — ggml ne[0]=256, ne[1]=1024.
    //   y = W^T @ x : lm_ggml_mul_mat(W, x_256) gives (1024,) (contracts on ne[0]).
    lm_ggml_tensor * spk_W = codec_graph_mat_lhs(ctx_eval, impl->spkr_enc_W);
    lm_ggml_tensor * spk_y = lm_ggml_mul_mat(ctx_eval, spk_W, t_spk);   // (1024,)
    lm_ggml_tensor * spk_b = codec_graph_cast_f32(ctx_eval, impl->spkr_enc_b);
    spk_y = lm_ggml_add(ctx_eval, spk_y, spk_b);
    lm_ggml_tensor * cond_spkr = lm_ggml_reshape_2d(ctx_eval, spk_y, H, 1);   // (H, 1)

    // ---- cond_emotion = emotion_adv_fc(emotion) --------------------
    //   emotion_adv_fc.weight is (1, 1024) — ggml ne[0]=1, ne[1]=1024.
    //   y = W @ scalar : lm_ggml_mul_mat(W_2d, scalar_1d).  Result (1024,).
    lm_ggml_tensor * emo_W = codec_graph_mat_lhs(ctx_eval, impl->emotion_adv_W);
    lm_ggml_tensor * emo_y = lm_ggml_mul_mat(ctx_eval, emo_W, t_emotion);   // (1024,)
    lm_ggml_tensor * cond_emotion = lm_ggml_reshape_2d(ctx_eval, emo_y, H, 1);

    // ---- speech_emb lookup + speech_pos_emb add → seq (H, T_speech) -
    lm_ggml_tensor * emb_rows = lm_ggml_get_rows(ctx_eval, impl->speech_emb, t_tokens);   // (H, T_speech)
    emb_rows = codec_graph_cast_f32(ctx_eval, emb_rows);

    // pos_ids = [0, 1, ..., T_speech-1]
    lm_ggml_tensor * pos_ids = lm_ggml_arange(ctx_eval, 0.0f, (float) T_speech, 1.0f);
    pos_ids = lm_ggml_cast(ctx_eval, pos_ids, LM_GGML_TYPE_I32);
    lm_ggml_tensor * pos_rows = lm_ggml_get_rows(ctx_eval, impl->speech_pos_emb, pos_ids);
    pos_rows = codec_graph_cast_f32(ctx_eval, pos_rows);

    lm_ggml_tensor * seq = lm_ggml_add(ctx_eval, emb_rows, pos_rows);     // (H, T_speech)

    // ---- Perceiver: 32 learned queries cross-attend over seq, then self-attend.
    // perceiver_q stored as (1, 32, H) — flatten to (H, 32).
    lm_ggml_tensor * pq_f32 = codec_graph_cast_f32(ctx_eval, impl->perceiver_q);
    lm_ggml_tensor * queries = lm_ggml_reshape_2d(ctx_eval, pq_f32, H, 32);   // (H, 32)
    queries = lm_ggml_cont(ctx_eval, queries);

    lm_ggml_tensor * pre_att = perceiver_attn_block(ctx_eval, impl, queries, seq, /*n_heads=*/4);
    lm_ggml_tensor * att     = perceiver_attn_block(ctx_eval, impl, pre_att, pre_att, /*n_heads=*/4);
    // att shape: (H, 32)

    // ---- Concat: [cond_spkr (1) | att (32) | cond_emotion (1)] → (H, 34)
    lm_ggml_tensor * cat0 = lm_ggml_concat(ctx_eval, cond_spkr, att, 1);      // (H, 33)
    lm_ggml_tensor * cond = lm_ggml_concat(ctx_eval, cat0, cond_emotion, 1);  // (H, 34)
    lm_ggml_set_name(cond, "spk.out.cond_emb");
    *out = cond;
    return true;
}

}  // namespace

// ---------------------------------------------------------------------
// Top-level encode
// ---------------------------------------------------------------------

enum codec_status chatterbox_speaker_encode(
        codec_lm * lm,
        const struct codec_audio * ref_pcm,
        const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
        float emotion,
        float * out, int32_t out_n_elems) {
    if (lm == nullptr || lm->speaker_impl == nullptr) {
        return CODEC_STATUS_INVALID_STATE;
    }
    cbx_impl * impl = static_cast<cbx_impl *>(lm->speaker_impl);

    // ---- 1. PCM → mel partials ------------------------------------
    if (ref_pcm == nullptr || ref_pcm->data == nullptr || ref_pcm->n_samples <= 0) {
        lm->last_error = "chatterbox speaker_encode: ref_pcm missing";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->pcm_type != CODEC_PCM_TYPE_F32) {
        lm->last_error = "chatterbox speaker_encode: ref_pcm must be PCM_TYPE_F32";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->sample_rate != impl->sample_rate) {
        lm->last_error =
            "chatterbox speaker_encode: ref_pcm sample_rate doesn't match the encoder's "
            "ref_sample_rate (16000) — caller must resample first";
        return CODEC_STATUS_INVALID_ARG;
    }

    const float * pcm_ptr = static_cast<const float *>(ref_pcm->data);
    std::vector<float> pcm(pcm_ptr, pcm_ptr + (size_t) ref_pcm->n_samples * (size_t) ref_pcm->n_channels);
    if (ref_pcm->n_channels > 1) {
        // Downmix to mono — VE works on mono.  Simple mean.
        std::vector<float> mono((size_t) ref_pcm->n_samples, 0.0f);
        for (int32_t i = 0; i < ref_pcm->n_samples; ++i) {
            float acc = 0.0f;
            for (int32_t c = 0; c < ref_pcm->n_channels; ++c) {
                acc += pcm[(size_t) i * (size_t) ref_pcm->n_channels + (size_t) c];
            }
            mono[(size_t) i] = acc / (float) ref_pcm->n_channels;
        }
        pcm.swap(mono);
    }

    std::vector<float> partials;
    int32_t n_partials = 0;
    std::string err;
    if (!codec_runtime_chatterbox_ve_mel_partials(
            pcm, impl->sample_rate, impl->mel_basis, impl->n_freq, impl->n_mels,
            impl->window, impl->n_fft, impl->hop_size,
            impl->partial_frames, impl->overlap, impl->rate, impl->min_coverage,
            &partials, &n_partials, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (n_partials <= 0) {
        lm->last_error = "chatterbox speaker_encode: no mel partials";
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // ---- 2. LSTM forward over batched partials --------------------
    // Reshape partials (n_partials, T=160, n_mels) → input to layer 0 in
    // shape (T, B, in_dim) — for each timestep, gather the B partials' rows.
    const int32_t T = impl->partial_frames;
    const int32_t B = n_partials;
    std::vector<float> X((size_t) T * (size_t) B * (size_t) impl->n_mels, 0.0f);
    for (int32_t t = 0; t < T; ++t) {
        for (int32_t bi = 0; bi < B; ++bi) {
            std::memcpy(
                X.data() + ((size_t) t * (size_t) B + (size_t) bi) * (size_t) impl->n_mels,
                partials.data() + ((size_t) bi * (size_t) T + (size_t) t) * (size_t) impl->n_mels,
                (size_t) impl->n_mels * sizeof(float));
        }
    }
    std::vector<float> H_curr((size_t) T * (size_t) B * (size_t) impl->hidden_size, 0.0f);
    lstm_layer_forward(impl->lstm[0], X.data(), T, B, impl->hidden_size, H_curr.data());
    std::vector<float> H_next((size_t) T * (size_t) B * (size_t) impl->hidden_size, 0.0f);
    for (int32_t l = 1; l < impl->num_layers; ++l) {
        lstm_layer_forward(impl->lstm[(size_t) l], H_curr.data(),
                           T, B, impl->hidden_size, H_next.data());
        H_curr.swap(H_next);
    }
    // Final hidden state per partial = H_curr[T-1, :, :]
    std::vector<float> raw_embeds((size_t) B * (size_t) impl->hidden_size, 0.0f);
    std::memcpy(raw_embeds.data(),
                H_curr.data() + (size_t) (T - 1) * (size_t) B * (size_t) impl->hidden_size,
                (size_t) B * (size_t) impl->hidden_size * sizeof(float));

    // ---- 3. proj + (optional) relu + L2 normalize per partial ------
    std::vector<float> proj_embeds((size_t) B * (size_t) impl->embed_size, 0.0f);
    for (int32_t bi = 0; bi < B; ++bi) {
        for (int32_t r = 0; r < impl->embed_size; ++r) {
            float acc = impl->proj_b[(size_t) r];
            const float * w_row = impl->proj_W.data() + (size_t) r * (size_t) impl->hidden_size;
            const float * x_row = raw_embeds.data() + (size_t) bi * (size_t) impl->hidden_size;
            for (int32_t k = 0; k < impl->hidden_size; ++k) acc += w_row[k] * x_row[k];
            if (impl->final_relu && acc < 0.0f) acc = 0.0f;
            proj_embeds[(size_t) bi * (size_t) impl->embed_size + (size_t) r] = acc;
        }
    }
    l2_normalize_rows(proj_embeds.data(), B, impl->embed_size);

    // ---- 4. mean over partials + L2 normalize → spk_emb_raw --------
    std::vector<float> spk_emb_raw((size_t) impl->embed_size, 0.0f);
    for (int32_t bi = 0; bi < B; ++bi) {
        for (int32_t k = 0; k < impl->embed_size; ++k) {
            spk_emb_raw[(size_t) k] += proj_embeds[(size_t) bi * (size_t) impl->embed_size + (size_t) k];
        }
    }
    const float inv_B = 1.0f / (float) B;
    for (int32_t k = 0; k < impl->embed_size; ++k) spk_emb_raw[(size_t) k] *= inv_B;
    l2_normalize_rows(spk_emb_raw.data(), 1, impl->embed_size);

    // ---- 5. cond_enc + perceiver graph ----------------------------
    return chatterbox_speaker_encode_from_emb(
        lm, spk_emb_raw.data(),
        ref_speech_tokens, n_ref_speech_tokens, emotion,
        out, out_n_elems);
}

enum codec_status chatterbox_speaker_encode_from_emb(
        codec_lm * lm,
        const float * speaker_emb,
        const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
        float emotion,
        float * out, int32_t out_n_elems) {
    if (lm == nullptr || lm->speaker_impl == nullptr || speaker_emb == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    cbx_impl * impl = static_cast<cbx_impl *>(lm->speaker_impl);
    if (n_ref_speech_tokens <= 0) {
        lm->last_error = "chatterbox speaker_encode: ref_speech_tokens required";
        return CODEC_STATUS_INVALID_ARG;
    }
    std::string err;
    if (impl->cond_ctx == nullptr) {
        codec_context * cctx = new (std::nothrow) codec_context();
        if (cctx == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
        cctx->model   = lm->codec;
        cctx->backend = lm->codec->backend;
        cctx->params  = codec_context_default_params();
        std::string ierr;
        if (!codec_runtime_init(cctx, &ierr)) {
            delete cctx;
            lm->last_error = "chatterbox speaker_encode: " + ierr;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        impl->cond_ctx = cctx;
    }

    cbx_cond_build_data build = { impl, n_ref_speech_tokens };
    codec_graph_eval_guard guard(impl->cond_ctx);
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind     = CODEC_GRAPH_LM_SPEAKER_CHATTERBOX;
    key.n_frames = n_ref_speech_tokens;
    if (!codec_graph_cache_get_or_build(
            impl->cond_ctx, key, build_cond_graph, &build, sizeof(build),
            &entry, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_spk     = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.in.spkr_emb");
    lm_ggml_tensor * t_tokens  = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.in.tokens");
    lm_ggml_tensor * t_emotion = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.in.emotion");
    lm_ggml_tensor * t_out     = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.out.cond_emb");
    if (!t_spk || !t_tokens || !t_emotion || !t_out) {
        lm->last_error = "chatterbox speaker_encode: cond graph missing tensors";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(impl->cond_ctx, entry, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_spk, speaker_emb,
                                    (size_t) impl->speaker_embed_dim * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_tokens, ref_speech_tokens,
                                    (size_t) n_ref_speech_tokens * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_emotion, &emotion, sizeof(float), &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t nt = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
    if (!codec_graph_compute(impl->cond_ctx, entry, nt, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t need = impl->n_rows * impl->hidden_dim;
    if (out_n_elems < need) {
        lm->last_error = "chatterbox speaker_encode: out buffer too small";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (!codec_runtime_read_tensor(t_out, out,
                                   (size_t) need * sizeof(float), &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}
