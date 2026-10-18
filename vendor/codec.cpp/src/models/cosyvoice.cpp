#include "cosyvoice.h"

#include "../blocks/campplus.h"
#include "../blocks/dit.h"
#include "../blocks/s3_tokenizer.h"
#include "../ops/audio_frontend.h"
#include "../ops/conv1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

// CosyVoice token2wav, one graph per decode: flow encoder → length regulation
// → unrolled CFM (Euler × CFG) → HiFT. Reference: FunAudioLLM/CosyVoice
// cosyvoice/flow/flow.py (MaskedDiffWithXvec, CausalMaskedDiffWithXvec,
// CausalMaskedDiffWithDiT) and cosyvoice/hifigan/generator.py; BreezyVoice's
// fork (mtkresearch/BreezyVoice) for the v1 joint-nearest regulator.

namespace {

constexpr int32_t kMel = 80;
constexpr int32_t kSpkDim = 192;
constexpr int32_t kTokSampleRate = 16000;       // speech tokenizer + CAMPPlus
constexpr int32_t kRefMaxSeconds = 30;          // frontend asserts <= 30 s
constexpr int32_t kMaxTimesteps = 64;
constexpr int32_t kSplitEdgeTokens = 20;        // regulator head/tail (token_overlap_len)

const codec_cosyvoice & cv_of(const codec_model * m) {
    return *static_cast<const codec_cosyvoice *>(m->impl);
}

// v1 mel frames for `n` tokens at 50 Hz: int(n / 50 * 22050 / 256).
int32_t v1_mel_len(int32_t n, int32_t token_rate, int32_t sample_rate, bool f32) {
    if (f32) {   // BreezyVoice: (token_len / 50 * 22050 / 256).int() on a float32 tensor
        float x = (float) n / (float) token_rate;
        x = x * (float) sample_rate;
        x = x / 256.0f;
        return (int32_t) x;
    }
    return (int32_t) ((double) n / (double) token_rate * (double) sample_rate / 256.0);
}

struct cv_lengths {
    int32_t n_prompt = 0;   // prompt speech tokens
    int32_t n_target = 0;   // generated tokens
    int32_t mel_len1 = 0;   // prompt mel frames (cond prefix, trimmed from the output)
    int32_t mel_total = 0;  // flow output frames
};

cv_lengths cv_lengths_for(const codec_cosyvoice & cv, int32_t n_prompt, int32_t n_target, int32_t prompt_feat) {
    cv_lengths l;
    l.n_prompt = n_prompt;
    l.n_target = n_target;
    l.mel_len1 = prompt_feat;
    if (cv.version != 1) {
        l.mel_total = (n_prompt + n_target) * cv.token_mel_ratio;
    } else if (cv.length_regulator == "joint_nearest") {
        l.mel_total = v1_mel_len(n_prompt + n_target, cv.token_rate, cv.sample_rate, true);
    } else {
        l.mel_total = prompt_feat + v1_mel_len(n_target, cv.token_rate, cv.sample_rate, false);
    }
    return l;
}

// Mel frames after the speed change: int(mel / speed) (cli/model.py token2wav).
int32_t speed_mel_len(const codec_cosyvoice & cv, int32_t mel) {
    return cv.speed == 1.0f ? mel : (int32_t) ((double) mel / (double) cv.speed);
}

ggml_tensor * time_slice(ggml_context * ctx, ggml_tensor * x, int64_t t0, int64_t n) {
    return ggml_cont(ctx, ggml_view_2d(ctx, x, n, x->ne[1], x->nb[1], (size_t) t0 * x->nb[0]));
}

// F.interpolate(size=n_out, align_corners=False) along time of x [T, c],
// mirroring PyTorch's float32 source index: `nearest` (legacy floor(dst * in /
// out)) is an exact row gather, `linear` a matmul with a constant [T, n_out]
// weight matrix.
ggml_tensor * interp_matrix(ggml_context * ctx, ggml_tensor * x, int32_t n_out, bool nearest, const std::string & name) {
    const int32_t n_in = (int32_t) x->ne[0];
    const float scale = (float) n_in / (float) n_out;
    if (nearest) {
        std::vector<int32_t> idx((size_t) n_out);
        for (int32_t o = 0; o < n_out; ++o) idx[(size_t) o] = std::min((int32_t) std::floor((float) o * scale), n_in - 1);
        const int64_t ne[1] = {n_out};
        ggml_tensor * t_idx = codec_graph_const(ctx, name.c_str(), GGML_TYPE_I32, 1, ne, idx.data());
        ggml_tensor * rows = ggml_get_rows(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)), t_idx);   // [c, n_out]
        return ggml_cont(ctx, ggml_transpose(ctx, rows));
    }
    std::vector<float> m((size_t) n_in * n_out, 0.0f);
    for (int32_t o = 0; o < n_out; ++o) {
        float * row = m.data() + (size_t) o * n_in;
        float src = ((float) o + 0.5f) * scale - 0.5f;
        if (src < 0.0f) src = 0.0f;
        const int32_t i0 = std::min((int32_t) src, n_in - 1);
        const int32_t i1 = std::min(i0 + 1, n_in - 1);
        const float lam = src - (float) i0;
        row[i0] += 1.0f - lam;
        row[i1] += lam;
    }
    ggml_tensor * w = codec_graph_const_f32(ctx, name.c_str(), m, n_in, n_out);
    return codec_mul_mat(ctx, w, ggml_cont(ctx, x));   // contract over time: [n_out, c]
}

ggml_tensor * interp_linear(ggml_context * ctx, ggml_tensor * x, int32_t n_out, const char * tag) {
    return interp_matrix(ctx, x, n_out, false, std::string("lut.cv.lr_") + tag);
}

// InterpolateRegulator conv stack: [Conv(k3) → GroupNorm(1) → Mish] × k → Conv(1×1).
ggml_tensor * lr_convs(ggml_context * ctx, const codec_model * model, const codec_cosyvoice & cv, ggml_tensor * x) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, "cv.flow.lr." + n); };
    for (int32_t i = 0; i < cv.lr_blocks && x != nullptr; ++i) {
        const std::string s = std::to_string(i);
        x = codec_conv1d(ctx, x, W(s + ".cv.w"), W(s + ".cv.b"), 1, 1, 1);
        if (x == nullptr) return nullptr;
        x = codec_op_group_norm(ctx, x, 1, 1e-5f, W(s + ".gn.w"), W(s + ".gn.b"));
        if (x == nullptr) return nullptr;
        x = codec_op_unary(ctx, x, CODEC_UNARY_MISH);
    }
    return x == nullptr ? nullptr : codec_conv1d(ctx, x, W("out.w"), W("out.b"), 1, 1, 0);
}

// Flow encoder → mu [mel_total, 80].
ggml_tensor * build_mu(ggml_context * ctx, const codec_model * model, const codec_cosyvoice & cv,
                       const cv_lengths & l, ggml_tensor * tokens) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, n); };
    ggml_tensor * emb = W("cv.flow.input_emb.w");
    if (emb == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_get_rows(ctx, emb, tokens);   // [hidden, T]
    const int32_t T = l.n_prompt + l.n_target;

    if (cv.version == 3) {
        // PreLookahead on the 80-d token embedding, then repeat_interleave(2).
        ggml_tensor * h = codec_block_pre_lookahead(ctx, model, "cv.flow.pre",
                                                    ggml_cont(ctx, ggml_transpose(ctx, x_ct)));
        if (h == nullptr) return nullptr;
        ggml_tensor * h3 = ggml_reshape_3d(ctx, h, 1, T, kMel);
        h3 = ggml_repeat(ctx, h3, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cv.token_mel_ratio, T, kMel));
        return ggml_reshape_2d(ctx, ggml_cont(ctx, h3), cv.token_mel_ratio * T, kMel);
    }

    ggml_tensor * h = codec_block_conformer_embed(ctx, model, "cv.flow.enc.embed", cv.enc, x_ct);
    if (cv.version == 2) h = codec_block_pre_lookahead(ctx, model, "cv.flow.enc.pre", h);
    h = codec_block_conformer_stack(ctx, model, "cv.flow.enc.blk", cv.enc, h, cv.enc_layers);
    if (h == nullptr) return nullptr;
    if (cv.version == 2) {
        h = codec_block_upsample_conv(ctx, model, "cv.flow.enc.up", h, cv.token_mel_ratio);
        if (h == nullptr) return nullptr;
        h = codec_block_conformer_embed(ctx, model, "cv.flow.enc.up_embed", cv.enc,
                                        ggml_cont(ctx, ggml_transpose(ctx, h)));
        h = codec_block_conformer_stack(ctx, model, "cv.flow.enc.up_blk", cv.enc, h, cv.enc_up_layers);
        if (h == nullptr) return nullptr;
    }
    h = codec_op_layer_norm_tc(ctx, h, cv.enc.embed_eps, W("cv.flow.enc.after_norm.w"), W("cv.flow.enc.after_norm.b"));
    if (h == nullptr) return nullptr;
    ggml_tensor * mu_ct = codec_op_linear(ctx, ggml_cont(ctx, ggml_transpose(ctx, h)), W("cv.flow.proj.w"),
                                          W("cv.flow.proj.b"));
    if (mu_ct == nullptr) return nullptr;
    ggml_tensor * mu = ggml_cont(ctx, ggml_transpose(ctx, mu_ct));   // [T or 2T, 80]
    if (cv.version == 2) return mu;

    // v1 InterpolateRegulator.
    if (cv.length_regulator == "joint_nearest") {
        mu = interp_matrix(ctx, mu, l.mel_total, true, "lut.cv.lr_nearest");
    } else {
        const int32_t mel_len2 = l.mel_total - l.mel_len1;
        ggml_tensor * x2 = time_slice(ctx, mu, l.n_prompt, l.n_target);
        if (l.n_target > 2 * kSplitEdgeTokens) {
            const int32_t edge = v1_mel_len(kSplitEdgeTokens, cv.token_rate, cv.sample_rate, false);
            ggml_tensor * hd = interp_linear(ctx, time_slice(ctx, x2, 0, kSplitEdgeTokens), edge, "head");
            ggml_tensor * md = interp_linear(ctx, time_slice(ctx, x2, kSplitEdgeTokens, l.n_target - 2 * kSplitEdgeTokens),
                                             mel_len2 - 2 * edge, "mid");
            ggml_tensor * tl = interp_linear(ctx, time_slice(ctx, x2, l.n_target - kSplitEdgeTokens, kSplitEdgeTokens), edge, "tail");
            x2 = ggml_concat(ctx, ggml_concat(ctx, hd, md, 0), tl, 0);
        } else {
            x2 = interp_linear(ctx, x2, mel_len2, "target");
        }
        if (l.n_prompt > 0) {
            ggml_tensor * x1 = interp_linear(ctx, time_slice(ctx, mu, 0, l.n_prompt), l.mel_len1, "prompt");
            x2 = ggml_concat(ctx, x1, x2, 0);
        }
        mu = x2;
    }
    return lr_convs(ctx, model, cv, mu);
}

struct cv_decode_build {
    cv_lengths len;
    const codec_model * model = nullptr;
};

bool build_decode(ggml_context * ctx, void * user_data, ggml_tensor ** out) {
    auto * p = static_cast<cv_decode_build *>(user_data);
    if (ctx == nullptr || p == nullptr || p->model == nullptr || out == nullptr) return false;
    const codec_model * model = p->model;
    const codec_cosyvoice & cv = cv_of(model);
    const cv_lengths & l = p->len;
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, n); };
    const int32_t T_tok = l.n_prompt + l.n_target;
    const int32_t T_mel = l.mel_total;

    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_tok);
    ggml_set_name(t_tok, "cv.tokens");
    ggml_set_input(t_tok);
    ggml_tensor * mu = build_mu(ctx, model, cv, l, t_tok);
    if (mu == nullptr || mu->ne[0] != T_mel) return false;

    // spks = Linear(F.normalize(x-vector)).
    ggml_tensor * t_spk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kSpkDim, 1);
    ggml_set_name(t_spk, "cv.spk_embedding");
    ggml_set_input(t_spk);
    ggml_tensor * spk_n = ggml_scale(ctx, ggml_rms_norm(ctx, t_spk, 1e-12f), 1.0f / std::sqrt((float) kSpkDim));
    ggml_tensor * spks = codec_op_linear(ctx, spk_n, W("cv.flow.spk_aff.w"), W("cv.flow.spk_aff.b"));
    if (spks == nullptr) return false;
    spks = ggml_reshape_1d(ctx, spks, kMel);

    // cond = [prompt_feat, zeros].
    ggml_tensor * cond = nullptr;
    if (l.mel_len1 > 0) {
        ggml_tensor * t_pf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kMel, l.mel_len1);
        ggml_set_name(t_pf, "cv.prompt_feat");
        ggml_set_input(t_pf);
        cond = ggml_pad(ctx, ggml_cont(ctx, ggml_transpose(ctx, t_pf)), T_mel - l.mel_len1, 0, 0, 0);
    } else {
        cond = ggml_scale(ctx, mu, 0.0f);
    }

    ggml_tensor * t_z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, kMel);
    ggml_set_name(t_z, "cv.noise_z");
    ggml_set_input(t_z);

    codec_cfm_solver_params sp;
    sp.n_steps = cv.n_timesteps;
    sp.cfg_rate = cv.cfg_rate;
    codec_cfm_estimator_fn estimator;
    if (cv.version == 3) {
        codec_dit_params dp;
        dp.depth = cv.dit_depth;
        dp.dim = cv.dit_dim;
        dp.n_heads = cv.dit_heads;
        dp.rope_dim = cv.dit_rope_dim;
        estimator = [&, dp](ggml_tensor * x, ggml_tensor * m, ggml_tensor * s, ggml_tensor * c, float t, float) {
            return codec_block_dit(ctx, model, "cv.dit", dp, x, m, s, c, t);
        };
    } else {
        estimator = [&](ggml_tensor * x, ggml_tensor * m, ggml_tensor * s, ggml_tensor * c, float t, float) -> ggml_tensor * {
            ggml_tensor * te = codec_block_cfm_time_mlp(ctx, model, "cv.cfm", t, cv.unet.in_channels, cv.unet.time_scale);
            return te == nullptr ? nullptr : codec_block_cfm_unet(ctx, model, "cv.cfm", cv.unet, x, m, s, c, te);
        };
    }
    ggml_tensor * x = codec_block_cfm_solve(ctx, sp, t_z, mu, spks, cond, estimator);
    if (x == nullptr) return false;
    ggml_tensor * mel = time_slice(ctx, ggml_cont(ctx, x), l.mel_len1, T_mel - l.mel_len1);
    const int32_t n_mel = speed_mel_len(cv, T_mel - l.mel_len1);
    if (n_mel != T_mel - l.mel_len1) mel = interp_matrix(ctx, mel, n_mel, false, "lut.cv.speed");
    ggml_set_name(mel, "cv.mel");

    const int32_t T_pcm = n_mel * codec_hift_upsample_factor(cv.hift);
    ggml_tensor * t_phase = nullptr;   // SineGen2 has no per-harmonic initial phase input
    if (cv.hift.sinegen == 1) {
        t_phase = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cv.hift.nb_harmonics + 1);
        ggml_set_name(t_phase, "cv.nsf_phase");
        ggml_set_input(t_phase);
    }
    ggml_tensor * t_noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_pcm, cv.hift.nb_harmonics + 1);
    ggml_set_name(t_noise, "cv.nsf_noise");
    ggml_set_input(t_noise);
    ggml_tensor * pcm = codec_block_hift(ctx, model, "cv.hift", cv.hift, mel, t_phase, t_noise);
    if (pcm == nullptr) return false;
    ggml_set_name(pcm, "cv.pcm");
    *out = pcm;
    return true;
}

// ------------------------------------------------------------ speakers ----
// Slots: prompt_token I32 [T], prompt_feat F32 [80, F], embedding F32 [192].
// SFT speakers (spk2info) carry only `embedding`.

bool speaker_validate(const codec_model * model, const codec_speaker * spk, std::string * err) {
    const codec_cosyvoice & cv = cv_of(model);
    const codec_speaker_slot * emb = spk->slot("embedding");
    const codec_speaker_slot * tok = spk->slot("prompt_token");
    const codec_speaker_slot * feat = spk->slot("prompt_feat");
    if (emb == nullptr || emb->type != GGML_TYPE_F32 || emb->nelements() != kSpkDim) {
        *err = "CosyVoice speaker needs an F32 [192] embedding slot";
        return false;
    }
    if ((tok == nullptr) != (feat == nullptr)) {
        *err = "prompt_token and prompt_feat must be given together";
        return false;
    }
    if (tok == nullptr) return true;
    if (tok->type != GGML_TYPE_I32 || tok->nelements() != tok->ne[0]) {
        *err = "prompt_token must be an I32 vector";
        return false;
    }
    const auto * v = reinterpret_cast<const int32_t *>(tok->data.data());
    for (int64_t i = 0; i < tok->ne[0]; ++i) {
        if (v[i] < 0 || v[i] >= cv.codebook_size) {
            *err = "prompt_token id out of range";
            return false;
        }
    }
    if (feat->type != GGML_TYPE_F32 || feat->ne[0] != kMel || feat->nelements() != feat->ne[0] * feat->ne[1]) {
        *err = "prompt_feat must be F32 [80, frames]";
        return false;
    }
    if (cv.version != 1 && feat->ne[1] != cv.token_mel_ratio * tok->ne[0]) {
        *err = "prompt_feat frames must be token_mel_ratio * prompt_token";
        return false;
    }
    return true;
}

struct cv_speaker_build {
    int32_t n_samples = 0;
    int32_t sample_rate = 0;
    const codec_model * model = nullptr;
};

int64_t ref_len16(int64_t n, int32_t sr) {
    return codec_resample_out_len(n, sr, kTokSampleRate);
}

// CosyVoiceFrontEnd as one graph: PCM → 16 kHz (load_wav) → speech tokens +
// CAMPPlus x-vector; 16 kHz → sample_rate → matcha log-mel (prompt_feat).
bool build_speaker(ggml_context * ctx, void * user_data, ggml_tensor ** out) {
    const auto * p = static_cast<const cv_speaker_build *>(user_data);
    if (ctx == nullptr || p == nullptr || p->model == nullptr || out == nullptr) return false;
    const codec_cosyvoice & cv = cv_of(p->model);

    ggml_tensor * pcm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p->n_samples);
    ggml_set_name(pcm, "cv.spk.pcm");
    ggml_set_input(pcm);
    ggml_tensor * wav16 = codec_op_resample_sinc_hann(ctx, pcm, p->sample_rate, kTokSampleRate);
    ggml_tensor * mel_fb = codec_graph_weight(ctx, p->model, "cv.spk.mel_fb");
    if (wav16 == nullptr || mel_fb == nullptr) return false;
    ggml_tensor * wav_sr = codec_op_resample_sinc_hann(ctx, wav16, kTokSampleRate, cv.sample_rate);
    if (wav_sr == nullptr) return false;

    ggml_tensor * tokens = nullptr;
    if (cv.version == 1) {
        tokens = codec_block_s3_tokenizer_v1(ctx, p->model, cv.tokenizer_layers, wav16);
    } else {
        codec_s3_tokenizer_params tp;
        tp.n_layers = cv.tokenizer_layers;
        tp.pad_multiple = 1;
        tokens = codec_block_s3_tokenizer(ctx, p->model, tp, wav16);
    }
    ggml_tensor * feat = codec_op_matcha_log_mel(ctx, wav_sr, mel_fb, cv.mel_n_fft, cv.mel_hop);   // [80, F]
    ggml_tensor * emb = codec_block_campplus(ctx, p->model, "cv.cp", wav16, /*onnx_seg_pooling=*/true);
    if (tokens == nullptr || feat == nullptr || emb == nullptr) return false;
    ggml_set_name(tokens, "cv.spk.prompt_token");
    ggml_set_name(feat, "cv.spk.prompt_feat");
    ggml_set_name(emb, "cv.spk.embedding");
    ggml_set_output(tokens);
    ggml_set_output(feat);
    *out = emb;
    return true;
}

enum codec_status speaker_from_audio(codec_context * ctx, const std::vector<float> & pcm, int32_t sample_rate,
                                     codec_speaker * out) {
    const codec_cosyvoice & cv = cv_of(ctx->model);
    if (pcm.empty() || sample_rate <= 0 || pcm.size() > (size_t) INT32_MAX) {
        codec_context_set_error(ctx, "invalid reference audio");
        return CODEC_STATUS_INVALID_ARG;
    }
    if ((int64_t) pcm.size() > (int64_t) kRefMaxSeconds * sample_rate) {
        codec_context_set_error(ctx, "reference audio longer than 30 s (CosyVoice limit)");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int64_t n16 = ref_len16((int64_t) pcm.size(), sample_rate);
    const int64_t n_sr = codec_resample_out_len(n16, kTokSampleRate, cv.sample_rate);
    const int32_t n_tok = cv.version == 1 ? codec_s3_tokenizer_v1_n_tokens(n16) : codec_s3_tokenizer_n_tokens(n16, 1);
    if (n_tok <= 0 || !codec_campplus_accepts(n16) || n_sr <= (cv.mel_n_fft - cv.mel_hop) / 2) {
        codec_context_set_error(ctx, "reference audio too short for the speaker encoders");
        return CODEC_STATUS_INVALID_ARG;
    }

    cv_speaker_build build;
    build.n_samples = (int32_t) pcm.size();
    build.sample_rate = sample_rate;
    build.model = ctx->model;
    std::string err;
    codec_graph_eval_guard guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(ctx, {CODEC_GRAPH_COSYVOICE_SPEAKER, build.n_samples, 0, sample_rate, 0, 0},
                                        build_speaker, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "cv.spk.pcm");
    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "cv.spk.prompt_token");
    ggml_tensor * t_feat = codec_graph_get_tensor(ctx, entry, "cv.spk.prompt_feat");
    ggml_tensor * t_emb = codec_graph_get_tensor(ctx, entry, "cv.spk.embedding");
    if (t_pcm == nullptr || t_tok == nullptr || t_feat == nullptr || t_emb == nullptr) {
        codec_context_set_error(ctx, "invalid speaker graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_graph_compute(ctx, entry, ctx->n_threads, &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> tokens((size_t) ggml_nelements(t_tok));
    std::vector<float> feat((size_t) ggml_nelements(t_feat));
    std::vector<float> emb((size_t) ggml_nelements(t_emb));
    if (!codec_runtime_read_tensor(t_tok, tokens.data(), tokens.size() * sizeof(int32_t), &err) ||
        !codec_runtime_read_tensor(t_feat, feat.data(), feat.size() * sizeof(float), &err) ||
        !codec_runtime_read_tensor(t_emb, emb.data(), emb.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    int64_t n_feat = t_feat->ne[1];
    if (cv.version != 1) {
        // 24 kHz frontends align mel and tokens: token_len = min(F / 2, T).
        const int64_t tl = std::min<int64_t>(n_feat / cv.token_mel_ratio, (int64_t) tokens.size());
        tokens.resize((size_t) tl);
        n_feat = tl * cv.token_mel_ratio;
        feat.resize((size_t) (kMel * n_feat));
    }
    const int64_t ne_tok[1] = {(int64_t) tokens.size()};
    const int64_t ne_feat[2] = {kMel, n_feat};
    const int64_t ne_emb[1] = {kSpkDim};
    codec_speaker_add_slot(out, "prompt_token", GGML_TYPE_I32, ne_tok, 1, tokens.data());
    codec_speaker_add_slot(out, "prompt_feat", GGML_TYPE_F32, ne_feat, 2, feat.data());
    codec_speaker_add_slot(out, "embedding", GGML_TYPE_F32, ne_emb, 1, emb.data());
    return CODEC_STATUS_SUCCESS;
}

// ------------------------------------------------------------- decode ----

enum codec_status decode(codec_context * ctx, const codec_token_buffer * tokens, codec_pcm_buffer * out_pcm,
                         codec_decode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || tokens == nullptr || tokens->data == nullptr ||
        tokens->n_tokens <= 0 || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    const codec_cosyvoice & cv = cv_of(ctx->model);
    const codec_speaker * spk = params.speaker != nullptr ? params.speaker : codec_get_speaker(ctx->model, nullptr);
    if (spk == nullptr) {
        codec_context_set_error(ctx, "CosyVoice decode needs a speaker (pass one, or use a model with builtin speakers)");
        return CODEC_STATUS_INVALID_ARG;
    }
    std::string err;
    if (!speaker_validate(ctx->model, spk, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INVALID_ARG;
    }
    const codec_speaker_slot * s_tok = spk->slot("prompt_token");
    const codec_speaker_slot * s_feat = spk->slot("prompt_feat");
    const codec_speaker_slot * s_emb = spk->slot("embedding");

    std::vector<int32_t> seq;
    const int32_t n_prompt = s_tok != nullptr ? (int32_t) s_tok->ne[0] : 0;
    if (n_prompt > 0) {
        const auto * pt = reinterpret_cast<const int32_t *>(s_tok->data.data());
        seq.assign(pt, pt + n_prompt);
    }
    for (int32_t i = 0; i < tokens->n_tokens; ++i) {
        const int32_t v = tokens->data[i];
        if (v >= 0 && v < cv.codebook_size) seq.push_back(v);
    }
    const int32_t n_target = (int32_t) seq.size() - n_prompt;
    if (n_target <= 0) {
        codec_context_set_error(ctx, "no speech tokens to decode");
        return CODEC_STATUS_INVALID_ARG;
    }
    cv_decode_build build;
    build.len = cv_lengths_for(cv, n_prompt, n_target, s_feat != nullptr ? (int32_t) s_feat->ne[1] : 0);
    build.model = ctx->model;
    const cv_lengths & l = build.len;
    if (l.mel_len1 >= l.mel_total || speed_mel_len(cv, l.mel_total - l.mel_len1) < 1) {
        codec_context_set_error(ctx, "prompt mel longer than the flow output");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t T_pcm = speed_mel_len(cv, l.mel_total - l.mel_len1) * codec_hift_upsample_factor(cv.hift);

    codec_graph_eval_guard guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(ctx, {CODEC_GRAPH_COSYVOICE_DECODE, n_target, n_prompt, l.mel_len1, l.mel_total, 0},
                                        build_decode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "cv.tokens");
    ggml_tensor * t_spk = codec_graph_get_tensor(ctx, entry, "cv.spk_embedding");
    ggml_tensor * t_pf = l.mel_len1 > 0 ? codec_graph_get_tensor(ctx, entry, "cv.prompt_feat") : nullptr;
    ggml_tensor * t_z = codec_graph_get_tensor(ctx, entry, "cv.noise_z");
    ggml_tensor * t_phase = cv.hift.sinegen == 1 ? codec_graph_get_tensor(ctx, entry, "cv.nsf_phase") : nullptr;
    ggml_tensor * t_noise = codec_graph_get_tensor(ctx, entry, "cv.nsf_noise");
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "cv.pcm");
    if (t_tok == nullptr || t_spk == nullptr || (l.mel_len1 > 0 && t_pf == nullptr) || t_z == nullptr ||
        (cv.hift.sinegen == 1 && t_phase == nullptr) || t_noise == nullptr || t_pcm == nullptr) {
        codec_context_set_error(ctx, "invalid decode graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> noise_z((size_t) l.mel_total * kMel);
    std::vector<float> nsf_phase, nsf_noise;
    {
        std::mt19937_64 rng(0);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (float & v : noise_z) v = gauss(rng);
        codec_hift_fill_random(cv.hift, T_pcm, rng, &nsf_phase, &nsf_noise);
    }
    if (!codec_runtime_write_tensor(t_tok, seq.data(), seq.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_spk, s_emb->data.data(), s_emb->data.size(), &err) ||
        (t_pf != nullptr && !codec_runtime_write_tensor(t_pf, s_feat->data.data(), (size_t) kMel * l.mel_len1 * sizeof(float), &err)) ||
        !codec_runtime_write_tensor(t_z, noise_z.data(), noise_z.size() * sizeof(float), &err) ||
        (t_phase != nullptr && !codec_runtime_write_tensor(t_phase, nsf_phase.data(), nsf_phase.size() * sizeof(float), &err)) ||
        !codec_runtime_write_tensor(t_noise, nsf_noise.data(), nsf_noise.size() * sizeof(float), &err) ||
        !codec_graph_compute(ctx, entry, ctx->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n = (int32_t) t_pcm->ne[0];
    float * data = static_cast<float *>(std::malloc((size_t) n * sizeof(float)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_read_tensor(t_pcm, data, (size_t) n * sizeof(float), &err)) {
        std::free(data);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    out_pcm->data = data;
    out_pcm->n_samples = n;
    out_pcm->sample_rate = cv.sample_rate;
    out_pcm->n_channels = 1;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------- lifecycle ----

enum codec_status init(codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) return CODEC_STATUS_INVALID_ARG;
    auto & cv = *static_cast<codec_cosyvoice *>(model->impl);
    gguf_context * g = model->gguf;
    cv.version = codec_read_i32_kv(g, "cosyvoice.version", 2);
    cv.sample_rate = codec_read_i32_kv(g, "codec.sample_rate", cv.version == 1 ? 22050 : 24000);
    cv.hop_size = codec_read_i32_kv(g, "codec.hop_size", cv.version == 1 ? 441 : 960);
    cv.token_rate = codec_read_i32_kv(g, "codec.token_rate_hz", cv.version == 1 ? 50 : 25);
    cv.codebook_size = codec_read_i32_kv(g, "codec.codebook_size", cv.version == 1 ? 4096 : 6561);
    cv.n_timesteps = codec_read_i32_kv(g, "cosyvoice.flow.n_timesteps", 10);
    cv.cfg_rate = codec_read_f32_kv(g, "cosyvoice.flow.cfg_rate", 0.7f);
    cv.speed = codec_read_f32_kv(g, "cosyvoice.flow.speed", 1.0f);
    cv.token_mel_ratio = codec_read_i32_kv(g, "cosyvoice.flow.token_mel_ratio", 2);
    cv.length_regulator = codec_read_str_kv(g, "cosyvoice.flow.length_regulator", "split_linear");
    cv.lr_blocks = codec_read_i32_kv(g, "cosyvoice.flow.lr_blocks", 4);
    cv.enc.hidden = codec_read_i32_kv(g, "cosyvoice.flow.enc.hidden", 512);
    cv.enc.n_heads = codec_read_i32_kv(g, "cosyvoice.flow.enc.heads", 8);
    cv.enc.block_eps = codec_read_f32_kv(g, "cosyvoice.flow.enc.block_eps", 1e-12f);
    cv.enc_layers = codec_read_i32_kv(g, "cosyvoice.flow.enc.layers", 6);
    cv.enc_up_layers = codec_read_i32_kv(g, "cosyvoice.flow.enc.up_layers", 0);
    cv.unet.n_levels = codec_read_i32_kv(g, "cosyvoice.cfm.n_levels", 1);
    cv.unet.n_mid = codec_read_i32_kv(g, "cosyvoice.cfm.n_mid", 12);
    cv.unet.n_transformers = codec_read_i32_kv(g, "cosyvoice.cfm.n_transformers", 4);
    cv.unet.n_heads = codec_read_i32_kv(g, "cosyvoice.cfm.n_heads", 8);
    cv.unet.head_dim = codec_read_i32_kv(g, "cosyvoice.cfm.head_dim", 64);
    cv.unet.causal = codec_read_bool_kv(g, "cosyvoice.cfm.causal", true);
    cv.dit_depth = codec_read_i32_kv(g, "cosyvoice.dit.depth", 22);
    cv.dit_dim = codec_read_i32_kv(g, "cosyvoice.dit.dim", 1024);
    cv.dit_heads = codec_read_i32_kv(g, "cosyvoice.dit.heads", 16);
    cv.dit_rope_dim = codec_read_i32_kv(g, "cosyvoice.dit.rope_dim", 64);

    cv.hift.sample_rate = cv.sample_rate;
    codec_read_i32_array_kv_vec(g, "cosyvoice.hift.upsample_rates", &cv.hift.upsample_rates);
    codec_read_i32_array_kv_vec(g, "cosyvoice.hift.upsample_kernels", &cv.hift.upsample_kernels);
    codec_read_i32_array_kv_vec(g, "cosyvoice.hift.resblock_kernels", &cv.hift.resblock_kernels);
    codec_read_i32_array_kv_vec(g, "cosyvoice.hift.source_resblock_kernels", &cv.hift.source_resblock_kernels);
    cv.hift.sinegen = codec_read_i32_kv(g, "cosyvoice.hift.sinegen", cv.sample_rate == 22050 ? 1 : 2);
    cv.hift_causal = codec_read_bool_kv(g, "cosyvoice.hift.causal", false);
    cv.hift.causal = cv.hift_causal;

    cv.has_speaker_encoder = codec_read_bool_kv(g, "cosyvoice.spk.has_encoder", false);
    cv.mel_n_fft = codec_read_i32_kv(g, "cosyvoice.spk.mel_n_fft", 1920);
    cv.mel_hop = codec_read_i32_kv(g, "cosyvoice.spk.mel_hop", 480);
    cv.tokenizer_layers = codec_read_i32_kv(g, "cosyvoice.spk.tokenizer_layers", 6);

    if (cv.version < 1 || cv.version > 3 || cv.n_timesteps < 1 || cv.n_timesteps > kMaxTimesteps ||
        !(cv.cfg_rate >= 0.0f) || !(cv.speed > 0.0f && cv.speed < 100.0f) || cv.hift.upsample_rates.empty() ||
        cv.hift.upsample_rates.size() != cv.hift.upsample_kernels.size() ||
        cv.hift.upsample_rates.size() != cv.hift.source_resblock_kernels.size() ||
        (cv.version == 1 && cv.length_regulator != "joint_nearest" && cv.length_regulator != "split_linear")) {
        return CODEC_STATUS_INVALID_ARG;
    }
    model->sample_rate = cv.sample_rate;
    model->encode_sample_rate = 0;
    model->has_encoder = false;
    model->has_decoder = true;
    model->hop_size = cv.hop_size;
    model->n_q = 1;
    model->codebook_size = cv.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = -1;
    model->can_speaker_from_audio = cv.has_speaker_encoder;
    return CODEC_STATUS_SUCCESS;
}

void * create_impl() { return new (std::nothrow) codec_cosyvoice(); }
void destroy_impl(void * p) { delete static_cast<codec_cosyvoice *>(p); }

}  // namespace

const struct codec_model_vtable * codec_cosyvoice_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_COSYVOICE,
        "CosyVoice",
        create_impl,
        destroy_impl,
        init,
        codec_graph_size_exact,
        nullptr,
        decode,
        nullptr,
        speaker_validate,
        speaker_from_audio,
    };
    return &vtable;
}
