#include "speaker_qwen3_tts.h"

#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"
#include "../runtime/audio_dsp.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// Qwen3-TTS ECAPA-TDNN speaker encoder.  Pure CPU forward — at ~5 GFLOPs
// per ref-audio clip it's bounded one-shot work; pulling out to a ggml
// graph isn't worth the complexity (and ECAPA-TDNN has no big
// matmul that dominates the perf budget).
//
// Forward graph (matches `Qwen3TTSSpeakerEncoder.forward` upstream):
//
//   pcm (24 kHz mono)
//     │ mel front-end (reflect pad (n_fft-hop)/2, Hann, |FFT|, slaney mel, log-clip 1e-5)
//     ▼
//   mel  (M=128, T)
//     │ blocks[0]  : TDNN(M=128, C=512, k=5, dil=1)            → x_1  (C, T)
//     │ blocks[1..N-2] : SE-Res2Net(C=512, C=512, scale=8, k, dil)
//     │                  emits x_2, x_3 (each (C, T))
//     │ multi-layer aggregation: cat([x_1, x_2, x_3], dim=channels)
//     │ mfa : TDNN(3C, 1536, k=1, dil=1)                       → mfa_out (1536, T)
//     │ asp : AttentiveStatisticsPooling(1536, attn_ch=128)    → pooled (3072, 1)
//     │ fc  : Conv1d(3072 → 1024, k=1)                         → spk_emb_raw (1024,)
//     ▼
//   (1, 1024) cond_emb  →  caller's out buffer
//
// All convs use reflect padding to preserve sequence length.
// =====================================================================

namespace {

// 1D conv weight stored row-major as (out_ch, in_ch, K).  Index helper:
inline size_t conv_w_off(int32_t oc, int32_t ic, int32_t k,
                          int32_t in_ch, int32_t K) {
    return (size_t) oc * (size_t) in_ch * (size_t) K +
           (size_t) ic * (size_t) K +
           (size_t) k;
}

struct conv1d_w {
    std::vector<float> W;
    std::vector<float> b;
    int32_t out_ch = 0;
    int32_t in_ch  = 0;
    int32_t K      = 0;
    int32_t dilation = 1;
};

bool load_conv(codec_lm * lm, const std::string & prefix, conv1d_w & w,
                int32_t dilation = 1) {
    lm_ggml_tensor * Wt = lm_ggml_get_tensor(lm->codec->weights, (prefix + ".weight").c_str());
    lm_ggml_tensor * bt = lm_ggml_get_tensor(lm->codec->weights, (prefix + ".bias").c_str());
    if (Wt == nullptr || bt == nullptr) {
        lm->last_error = "speaker(qwen3_tts): missing " + prefix + ".{weight,bias}";
        return false;
    }
    if (!codec_tensor_as_vec_f32(Wt, &w.W) || !codec_tensor_as_vec_f32(bt, &w.b)) {
        lm->last_error = "speaker(qwen3_tts): dequant failed for " + prefix;
        return false;
    }
    // ggml ne ordering vs PyTorch Conv1d weight (out_ch, in_ch, K):
    //   PyTorch row-major flat = [oc=0, ic=0..in-1 × k=0..K-1], …
    //   gguf preserves the numpy buffer; the conv runtime walks the
    //   flattened buffer in PyTorch order.
    w.out_ch   = (int32_t) Wt->ne[2];
    w.in_ch    = (int32_t) Wt->ne[1];
    w.K        = (int32_t) Wt->ne[0];
    w.dilation = dilation;
    return true;
}

struct se_block_w {
    conv1d_w conv1;
    conv1d_w conv2;
};

struct ecapa_block_w {
    // Block 0: just the initial TDNN.  Other fields unused.
    conv1d_w tdnn_init;

    // Blocks 1..N-2: SE-Res2Net.
    conv1d_w tdnn1;
    conv1d_w tdnn2;
    std::vector<conv1d_w> res2net;   // scale-1 sub-blocks
    se_block_w se;
    int32_t   res2net_scale = 0;
    int32_t   res2net_k     = 0;
    int32_t   res2net_dil   = 0;

    bool      is_initial = false;
};

struct qwen3_tts_speaker_impl {
    // ECAPA constants (from GGUF metadata).
    int32_t mel_dim       = 128;
    int32_t enc_dim       = 1024;
    int32_t attn_ch       = 128;
    int32_t res2net_scale = 8;
    int32_t se_ch         = 128;
    std::vector<int32_t> enc_channels;
    std::vector<int32_t> enc_kernels;
    std::vector<int32_t> enc_dilations;

    // Mel constants.
    int32_t n_fft   = 1024;
    int32_t hop     = 256;
    int32_t win     = 1024;
    int32_t n_freq  = 0;
    int32_t sample_rate = 24000;

    std::vector<float> mel_basis;    // (n_mels * n_freq)
    std::vector<float> window;

    // Block weights.  blocks[0] = initial TDNN; blocks[1..N-2] = SE-Res2Net.
    std::vector<ecapa_block_w> blocks;

    // MFA TDNN, ASP, final fc.
    conv1d_w mfa_conv;
    conv1d_w asp_tdnn;
    conv1d_w asp_conv;
    conv1d_w fc;

    int32_t n_rows     = 1;
    int32_t hidden_dim = 1024;
};

// ---------------------------------------------------------------------
// 1D conv + ReLU helpers
// ---------------------------------------------------------------------

// Apply a 1D conv (in_ch, in_T) → (out_ch, in_T) with "same" reflect
// padding (ECAPA-TDNN convention).
//
//   y[oc, t] = b[oc] + sum_{ic, k} W[oc, ic, k] * pad(x)[ic, t + k * dil - center]
//   center = ((K - 1) * dil) / 2
//
// Reflect padding: x[-i] = x[i] (excluding boundary).
void conv1d_reflect(
        const float * x, int32_t in_ch, int32_t T,
        const conv1d_w & w,
        float * y) {
    const int32_t K   = w.K;
    const int32_t dil = w.dilation;
    const int32_t Keff = (K - 1) * dil + 1;
    const int32_t center = Keff / 2;

    auto reflect = [T](int32_t i) -> int32_t {
        if (T <= 1) return 0;
        // Reflect repeatedly until inside [0, T-1].
        while (i < 0 || i >= T) {
            if (i < 0)         i = -i;
            else /* i >= T */  i = 2 * (T - 1) - i;
        }
        return i;
    };

    for (int32_t oc = 0; oc < w.out_ch; ++oc) {
        const float bias = w.b[(size_t) oc];
        for (int32_t t = 0; t < T; ++t) {
            float acc = bias;
            for (int32_t ic = 0; ic < in_ch; ++ic) {
                for (int32_t k = 0; k < K; ++k) {
                    const int32_t src_t = reflect(t + k * dil - center);
                    acc += w.W[conv_w_off(oc, ic, k, in_ch, K)] *
                           x[(size_t) ic * (size_t) T + (size_t) src_t];
                }
            }
            y[(size_t) oc * (size_t) T + (size_t) t] = acc;
        }
    }
}

inline void relu_inplace(float * x, size_t n) {
    for (size_t i = 0; i < n; ++i) if (x[i] < 0.0f) x[i] = 0.0f;
}

inline float sigmoidf(float x) {
    if (x >= 0.0f) { const float z = std::exp(-x); return 1.0f / (1.0f + z); }
    const float z = std::exp(x); return z / (1.0f + z);
}

void se_block_forward(
        const float * x, int32_t C, int32_t T,
        const se_block_w & se,
        float * y) {
    // Channel mean over time.
    std::vector<float> m((size_t) C, 0.0f);
    const float invT = 1.0f / (float) T;
    for (int32_t c = 0; c < C; ++c) {
        double s = 0.0;
        for (int32_t t = 0; t < T; ++t) s += x[(size_t) c * (size_t) T + (size_t) t];
        m[(size_t) c] = (float) (s * invT);
    }
    // conv1 (k=1) over a single time step: (in=C, T=1) → (se_ch, T=1)
    std::vector<float> z((size_t) se.conv1.out_ch, 0.0f);
    for (int32_t oc = 0; oc < se.conv1.out_ch; ++oc) {
        float acc = se.conv1.b[(size_t) oc];
        for (int32_t ic = 0; ic < se.conv1.in_ch; ++ic) {
            acc += se.conv1.W[conv_w_off(oc, ic, 0, se.conv1.in_ch, 1)] * m[(size_t) ic];
        }
        z[(size_t) oc] = acc > 0.0f ? acc : 0.0f;        // ReLU
    }
    // conv2 (k=1) → (C, 1) → sigmoid → gate.
    std::vector<float> g((size_t) C, 0.0f);
    for (int32_t oc = 0; oc < C; ++oc) {
        float acc = se.conv2.b[(size_t) oc];
        for (int32_t ic = 0; ic < se.conv2.in_ch; ++ic) {
            acc += se.conv2.W[conv_w_off(oc, ic, 0, se.conv2.in_ch, 1)] * z[(size_t) ic];
        }
        g[(size_t) oc] = sigmoidf(acc);
    }
    for (int32_t c = 0; c < C; ++c) {
        const float gc = g[(size_t) c];
        for (int32_t t = 0; t < T; ++t) {
            y[(size_t) c * (size_t) T + (size_t) t] =
                x[(size_t) c * (size_t) T + (size_t) t] * gc;
        }
    }
}

// Res2NetBlock: chunk the channel axis into `scale` parts, chain
// residuals through `scale-1` TDNN sub-blocks, re-concatenate.
// Matches `Res2NetBlock.forward` exactly.
void res2net_forward(
        const float * x, int32_t C, int32_t T,
        const std::vector<conv1d_w> & subs, int32_t scale,
        float * y) {
    const int32_t chunk = C / scale;
    std::vector<float> prev((size_t) chunk * (size_t) T, 0.0f);
    std::vector<float> add ((size_t) chunk * (size_t) T, 0.0f);
    for (int32_t i = 0; i < scale; ++i) {
        const float * src = x + (size_t) i * (size_t) chunk * (size_t) T;
        float * dst = y + (size_t) i * (size_t) chunk * (size_t) T;
        if (i == 0) {
            std::memcpy(dst, src, (size_t) chunk * (size_t) T * sizeof(float));
        } else if (i == 1) {
            conv1d_reflect(src, chunk, T, subs[(size_t) (i - 1)], dst);
            relu_inplace(dst, (size_t) chunk * (size_t) T);
            std::memcpy(prev.data(), dst, prev.size() * sizeof(float));
        } else {
            for (size_t k = 0; k < add.size(); ++k) add[k] = src[k] + prev[k];
            conv1d_reflect(add.data(), chunk, T, subs[(size_t) (i - 1)], dst);
            relu_inplace(dst, (size_t) chunk * (size_t) T);
            std::memcpy(prev.data(), dst, prev.size() * sizeof(float));
        }
    }
}

// SE-Res2Net block forward.  Returns `(out_ch, T)` in `y`.
void se_res2net_forward(
        const float * x, int32_t in_ch, int32_t T,
        const ecapa_block_w & b,
        std::vector<float> & scratch1,
        std::vector<float> & scratch2,
        float * y) {
    const int32_t out_ch = b.tdnn1.out_ch;

    scratch1.assign((size_t) out_ch * (size_t) T, 0.0f);
    conv1d_reflect(x, in_ch, T, b.tdnn1, scratch1.data());
    relu_inplace(scratch1.data(), scratch1.size());

    scratch2.assign((size_t) out_ch * (size_t) T, 0.0f);
    res2net_forward(scratch1.data(), out_ch, T, b.res2net, b.res2net_scale, scratch2.data());

    scratch1.assign((size_t) out_ch * (size_t) T, 0.0f);
    conv1d_reflect(scratch2.data(), out_ch, T, b.tdnn2, scratch1.data());
    relu_inplace(scratch1.data(), scratch1.size());

    se_block_forward(scratch1.data(), out_ch, T, b.se, y);
}

}  // namespace

// ---------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------

bool qwen3_tts_speaker_init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr ||
        lm->codec->gguf == nullptr || lm->codec->weights == nullptr) {
        if (lm) lm->last_error = "speaker(qwen3_tts): codec has no gguf/weights";
        return false;
    }
    lm_gguf_context * gf = lm->codec->gguf;

    auto * impl = new (std::nothrow) qwen3_tts_speaker_impl();
    if (impl == nullptr) {
        lm->last_error = "speaker(qwen3_tts): out of memory";
        return false;
    }

    impl->mel_dim       = codec_read_i32_kv(gf, "codec.speaker.ecapa.mel_dim", 128);
    impl->enc_dim       = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_dim", 1024);
    impl->attn_ch       = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_attention_channels", 128);
    impl->res2net_scale = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_res2net_scale", 8);
    impl->se_ch         = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_se_channels", 128);
    impl->n_fft         = codec_read_i32_kv(gf, "codec.speaker.ecapa.n_fft", 1024);
    impl->hop           = codec_read_i32_kv(gf, "codec.speaker.ecapa.hop_size", 256);
    impl->win           = codec_read_i32_kv(gf, "codec.speaker.ecapa.win_size", 1024);
    impl->sample_rate   = lm->speaker_info.ref_sample_rate;
    impl->n_freq        = impl->n_fft / 2 + 1;
    impl->n_rows        = lm->speaker_info.n_rows;
    impl->hidden_dim    = lm->speaker_info.hidden_dim;

    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_channels",     &impl->enc_channels);
    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_kernel_sizes", &impl->enc_kernels);
    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_dilations",    &impl->enc_dilations);
    if (impl->enc_channels.empty() ||
        impl->enc_channels.size() != impl->enc_kernels.size() ||
        impl->enc_channels.size() != impl->enc_dilations.size()) {
        lm->last_error = "speaker(qwen3_tts): enc_channels/kernels/dilations missing or mismatched";
        delete impl; return false;
    }

    // Mel basis + window (host buffers).
    lm_ggml_tensor * mb_t = lm_ggml_get_tensor(lm->codec->weights, "speaker.qwen3_tts.mel_basis");
    lm_ggml_tensor * wn_t = lm_ggml_get_tensor(lm->codec->weights, "speaker.qwen3_tts.window");
    if (!mb_t || !wn_t) {
        lm->last_error = "speaker(qwen3_tts): missing mel_basis / window";
        delete impl; return false;
    }
    if (!codec_tensor_as_vec_f32(mb_t, &impl->mel_basis) ||
        !codec_tensor_as_vec_f32(wn_t, &impl->window)) {
        lm->last_error = "speaker(qwen3_tts): mel/window dequant failed";
        delete impl; return false;
    }

    // ---- Block weights ----
    const int32_t n_blocks = (int32_t) impl->enc_channels.size();
    impl->blocks.resize((size_t) (n_blocks - 1));  // exclude the last (which is MFA-only)

    // Block 0: initial TDNN(mel_dim → enc_channels[0], k=enc_kernels[0], dil=enc_dilations[0])
    impl->blocks[0].is_initial = true;
    if (!load_conv(lm, "speaker.qwen3_tts.blocks.0.conv", impl->blocks[0].tdnn_init,
                   impl->enc_dilations[0])) {
        delete impl; return false;
    }

    // Blocks 1..N-2: SE-Res2Net.
    for (int32_t bi = 1; bi < n_blocks - 1; ++bi) {
        auto & b = impl->blocks[(size_t) bi];
        b.is_initial = false;
        const std::string p = "speaker.qwen3_tts.blocks." + std::to_string(bi);
        if (!load_conv(lm, p + ".tdnn1.conv", b.tdnn1, /*dil=*/1) ||
            !load_conv(lm, p + ".tdnn2.conv", b.tdnn2, /*dil=*/1) ||
            !load_conv(lm, p + ".se.conv1",   b.se.conv1, /*dil=*/1) ||
            !load_conv(lm, p + ".se.conv2",   b.se.conv2, /*dil=*/1)) {
            delete impl; return false;
        }
        b.res2net_scale = impl->res2net_scale;
        b.res2net_k     = impl->enc_kernels[(size_t) bi];
        b.res2net_dil   = impl->enc_dilations[(size_t) bi];
        b.res2net.resize((size_t) (impl->res2net_scale - 1));
        for (int32_t ri = 0; ri < impl->res2net_scale - 1; ++ri) {
            if (!load_conv(lm, p + ".res2net." + std::to_string(ri) + ".conv",
                            b.res2net[(size_t) ri], b.res2net_dil)) {
                delete impl; return false;
            }
        }
    }

    if (!load_conv(lm, "speaker.qwen3_tts.mfa.conv",      impl->mfa_conv,   /*dil=*/1) ||
        !load_conv(lm, "speaker.qwen3_tts.asp.tdnn.conv", impl->asp_tdnn,   /*dil=*/1) ||
        !load_conv(lm, "speaker.qwen3_tts.asp.conv",      impl->asp_conv,   /*dil=*/1) ||
        !load_conv(lm, "speaker.qwen3_tts.fc",            impl->fc,         /*dil=*/1)) {
        delete impl; return false;
    }

    lm->speaker_impl = impl;
    return true;
}

void qwen3_tts_speaker_free(codec_lm * lm) {
    if (lm == nullptr || lm->speaker_impl == nullptr) return;
    delete static_cast<qwen3_tts_speaker_impl *>(lm->speaker_impl);
    lm->speaker_impl = nullptr;
}

// ---------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------

enum codec_status qwen3_tts_speaker_encode(
        codec_lm * lm,
        const struct codec_audio * ref_pcm,
        const int32_t * /*ref_speech_tokens*/, int32_t /*n_ref_speech_tokens*/,
        float /*emotion*/,
        float * out, int32_t out_n_elems) {
    if (lm == nullptr || lm->speaker_impl == nullptr) {
        return CODEC_STATUS_INVALID_STATE;
    }
    auto * impl = static_cast<qwen3_tts_speaker_impl *>(lm->speaker_impl);
    if (ref_pcm == nullptr || ref_pcm->data == nullptr || ref_pcm->n_samples <= 0) {
        lm->last_error = "qwen3_tts speaker_encode: ref_pcm missing";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->pcm_type != CODEC_PCM_TYPE_F32) {
        lm->last_error = "qwen3_tts speaker_encode: ref_pcm must be PCM_TYPE_F32";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->sample_rate != impl->sample_rate) {
        lm->last_error =
            "qwen3_tts speaker_encode: sample_rate mismatch (caller must "
            "resample to ref_sample_rate)";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (out == nullptr || out_n_elems < impl->n_rows * impl->hidden_dim) {
        lm->last_error = "qwen3_tts speaker_encode: out buffer too small";
        return CODEC_STATUS_INVALID_ARG;
    }

    // ---- 1. PCM → mel ------------------------------------------------
    const float * pcm_ptr = static_cast<const float *>(ref_pcm->data);
    std::vector<float> pcm(pcm_ptr, pcm_ptr + (size_t) ref_pcm->n_samples * (size_t) ref_pcm->n_channels);
    if (ref_pcm->n_channels > 1) {
        std::vector<float> mono((size_t) ref_pcm->n_samples, 0.0f);
        for (int32_t i = 0; i < ref_pcm->n_samples; ++i) {
            float a = 0.0f;
            for (int32_t c = 0; c < ref_pcm->n_channels; ++c) {
                a += pcm[(size_t) i * (size_t) ref_pcm->n_channels + (size_t) c];
            }
            mono[(size_t) i] = a / (float) ref_pcm->n_channels;
        }
        pcm.swap(mono);
    }

    std::vector<float> mel;
    int32_t T = 0;
    std::string err;
    if (!codec_runtime_qwen3_tts_speaker_mel(
            pcm, impl->mel_basis, impl->n_freq, impl->mel_dim,
            impl->window, impl->n_fft, impl->hop,
            &mel, &T, &err)) {
        lm->last_error = "qwen3_tts speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (T < 2) {
        lm->last_error = "qwen3_tts speaker_encode: too few mel frames "
                          "(ref audio shorter than (n_fft + hop) at 24 kHz)";
        return CODEC_STATUS_INVALID_ARG;
    }

    // ---- 2. Block 0: initial TDNN + ReLU -----------------------------
    const ecapa_block_w & blk0 = impl->blocks[0];
    const int32_t C0 = blk0.tdnn_init.out_ch;
    std::vector<float> x1((size_t) C0 * (size_t) T, 0.0f);
    conv1d_reflect(mel.data(), impl->mel_dim, T, blk0.tdnn_init, x1.data());
    relu_inplace(x1.data(), x1.size());

    // ---- 3. SE-Res2Net blocks ---------------------------------------
    // outputs[] holds blocks[1..N-2] outputs (each (C, T) at C = enc_channels[bi]).
    std::vector<std::vector<float>> outputs;
    outputs.reserve(impl->blocks.size() - 1);
    std::vector<float> scratch_a, scratch_b;
    const float * prev = x1.data();
    int32_t prev_C = C0;
    for (size_t bi = 1; bi < impl->blocks.size(); ++bi) {
        const auto & b = impl->blocks[bi];
        const int32_t C = b.tdnn1.out_ch;
        std::vector<float> out_b((size_t) C * (size_t) T, 0.0f);
        se_res2net_forward(prev, prev_C, T, b, scratch_a, scratch_b, out_b.data());
        // Skip connection: SE-Res2Net adds the input back.
        // Matches `SqueezeExcitationRes2NetBlock.forward`:
        //   return self.se_block(...) + residual  (when in==out channels)
        if (prev_C == C) {
            for (size_t k = 0; k < out_b.size(); ++k) out_b[k] += prev[k];
        }
        outputs.emplace_back(std::move(out_b));
        prev = outputs.back().data();
        prev_C = C;
    }

    // ---- 4. Multi-layer feature aggregation -------------------------
    // Cat outputs[0..end] along channel dim → (sum_C, T).
    int32_t mfa_in_C = 0;
    for (const auto & v : outputs) mfa_in_C += (int32_t) (v.size() / (size_t) T);
    std::vector<float> cat((size_t) mfa_in_C * (size_t) T, 0.0f);
    {
        int32_t off = 0;
        for (const auto & v : outputs) {
            std::memcpy(cat.data() + (size_t) off * (size_t) T, v.data(),
                        v.size() * sizeof(float));
            off += (int32_t) (v.size() / (size_t) T);
        }
    }
    std::vector<float> mfa_out((size_t) impl->mfa_conv.out_ch * (size_t) T, 0.0f);
    conv1d_reflect(cat.data(), mfa_in_C, T, impl->mfa_conv, mfa_out.data());
    relu_inplace(mfa_out.data(), mfa_out.size());
    const int32_t Cmfa = impl->mfa_conv.out_ch;

    // ---- 5. Attentive Statistical Pooling ---------------------------
    // Per-channel mean / std over the full sequence (length-T mask is all 1).
    // attention input = cat([x, mean.broadcast(T), std.broadcast(T)], dim=channels)
    //                   shape (3C, T)
    // tdnn(3C → attn_ch) + tanh + conv(attn_ch → C) → softmax over T → (C, T)
    // weighted (mean, std) over time → cat → (2C, 1)
    std::vector<float> ch_mean((size_t) Cmfa, 0.0f);
    std::vector<float> ch_std ((size_t) Cmfa, 0.0f);
    const float invT = 1.0f / (float) T;
    for (int32_t c = 0; c < Cmfa; ++c) {
        double s = 0.0;
        for (int32_t t = 0; t < T; ++t) s += mfa_out[(size_t) c * (size_t) T + (size_t) t];
        ch_mean[(size_t) c] = (float) (s * invT);
        double sq = 0.0;
        for (int32_t t = 0; t < T; ++t) {
            const double d = (double) mfa_out[(size_t) c * (size_t) T + (size_t) t] - ch_mean[(size_t) c];
            sq += d * d;
        }
        ch_std[(size_t) c] = (float) std::sqrt(std::max(1e-12, sq * (double) invT));
    }
    // Build (3C, T) cat.
    std::vector<float> asp_in((size_t) (3 * Cmfa) * (size_t) T, 0.0f);
    for (int32_t c = 0; c < Cmfa; ++c) {
        std::memcpy(asp_in.data() + (size_t) c * (size_t) T,
                    mfa_out.data() + (size_t) c * (size_t) T,
                    (size_t) T * sizeof(float));
    }
    for (int32_t c = 0; c < Cmfa; ++c) {
        for (int32_t t = 0; t < T; ++t) {
            asp_in[(size_t) (c + Cmfa) * (size_t) T + (size_t) t] = ch_mean[(size_t) c];
            asp_in[(size_t) (c + 2 * Cmfa) * (size_t) T + (size_t) t] = ch_std[(size_t) c];
        }
    }
    // tdnn(3C → attn_ch) + tanh.  The TDNN block has ReLU inside its
    // forward (Conv1d → ReLU), so the pipeline is conv → ReLU → tanh,
    // not conv → tanh.
    std::vector<float> a1((size_t) impl->asp_tdnn.out_ch * (size_t) T, 0.0f);
    conv1d_reflect(asp_in.data(), 3 * Cmfa, T, impl->asp_tdnn, a1.data());
    relu_inplace(a1.data(), a1.size());
    for (size_t i = 0; i < a1.size(); ++i) a1[i] = std::tanh(a1[i]);
    // conv(attn_ch → Cmfa)
    std::vector<float> a2((size_t) Cmfa * (size_t) T, 0.0f);
    conv1d_reflect(a1.data(), impl->asp_tdnn.out_ch, T, impl->asp_conv, a2.data());
    // Softmax over T per channel.
    for (int32_t c = 0; c < Cmfa; ++c) {
        float * row = a2.data() + (size_t) c * (size_t) T;
        float mx = row[0];
        for (int32_t t = 1; t < T; ++t) if (row[(size_t) t] > mx) mx = row[(size_t) t];
        double s = 0.0;
        for (int32_t t = 0; t < T; ++t) {
            row[(size_t) t] = std::exp(row[(size_t) t] - mx);
            s += row[(size_t) t];
        }
        const float inv = (float) (1.0 / std::max(s, 1e-12));
        for (int32_t t = 0; t < T; ++t) row[(size_t) t] *= inv;
    }
    // weighted mean / std
    std::vector<float> pooled_mean((size_t) Cmfa, 0.0f);
    std::vector<float> pooled_std ((size_t) Cmfa, 0.0f);
    for (int32_t c = 0; c < Cmfa; ++c) {
        double m = 0.0;
        for (int32_t t = 0; t < T; ++t) {
            m += (double) a2[(size_t) c * (size_t) T + (size_t) t] *
                 (double) mfa_out[(size_t) c * (size_t) T + (size_t) t];
        }
        pooled_mean[(size_t) c] = (float) m;
        double v = 0.0;
        for (int32_t t = 0; t < T; ++t) {
            const double d = (double) mfa_out[(size_t) c * (size_t) T + (size_t) t] - m;
            v += (double) a2[(size_t) c * (size_t) T + (size_t) t] * d * d;
        }
        pooled_std[(size_t) c] = (float) std::sqrt(std::max(1e-12, v));
    }
    // Cat mean + std → (2C, 1)
    std::vector<float> pooled((size_t) (2 * Cmfa), 0.0f);
    std::memcpy(pooled.data(),                 pooled_mean.data(), (size_t) Cmfa * sizeof(float));
    std::memcpy(pooled.data() + (size_t) Cmfa, pooled_std .data(), (size_t) Cmfa * sizeof(float));

    // ---- 6. Final fc Conv1d (k=1) → (1024,) -------------------------
    std::vector<float> spk_emb((size_t) impl->fc.out_ch, 0.0f);
    for (int32_t oc = 0; oc < impl->fc.out_ch; ++oc) {
        float acc = impl->fc.b[(size_t) oc];
        for (int32_t ic = 0; ic < impl->fc.in_ch; ++ic) {
            acc += impl->fc.W[conv_w_off(oc, ic, 0, impl->fc.in_ch, 1)] * pooled[(size_t) ic];
        }
        spk_emb[(size_t) oc] = acc;
    }

    // Output is (1, hidden_dim).  Hidden matches enc_dim by design.
    std::memcpy(out, spk_emb.data(), (size_t) impl->hidden_dim * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}
