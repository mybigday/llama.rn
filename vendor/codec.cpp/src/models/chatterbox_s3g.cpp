#include "chatterbox_s3g.h"

#include "../blocks/campplus.h"
#include "../blocks/cfm.h"
#include "../blocks/conformer.h"
#include "../blocks/hift.h"
#include "../blocks/s3_tokenizer.h"
#include "../blocks/xvector_sincnet.h"
#include "../ops/audio_frontend.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

// Architectural constants for Chatterbox S3Gen (S3Token2Wav in
// chatterbox/s3gen/s3gen.py): CosyVoice2's token2wav, see src/blocks/.
namespace {

constexpr int32_t kFlowSpkEmbedDim = 192;
constexpr int32_t kFlowVocabSize = 6561;
constexpr int32_t kFlowEncoderHidden = 512;
constexpr int32_t kFlowEncoderLayers = 6;
constexpr int32_t kFlowEncoderUpLayers = 4;

// Tunable params for flow decode (matches CFM_PARAMS in chatterbox/s3gen/configs.py).
constexpr int32_t kFlowDefaultNTimesteps = 10;
constexpr int32_t kFlowMeanflowNTimesteps = 2;
constexpr int32_t kFlowMaxNTimesteps = 64;
constexpr float   kFlowCfgRate = 0.7f;

constexpr int32_t kCfmInChannels = 320;        // packed [x, mu, spks, cond] = 80*4
constexpr int32_t kCfmOutChannels = 80;
constexpr int32_t kCfmTimeEmbedDim = 1024;
constexpr float   kCfmTimeEmbedScale = 1000.0f;
constexpr int32_t kS3gSampleRate = 24000;   // S3GEN_SR
constexpr int32_t kS3SampleRate = 16000;    // S3_SR (tokenizer + CAMPPlus)
constexpr int32_t kSpkRefMaxSeconds = 10;   // DEC_COND_LEN
constexpr int32_t kSpkMelNFft = 1920;
constexpr int32_t kSpkMelHop = 480;

codec_hift_params s3g_hift_params(int32_t sample_rate) {
    codec_hift_params p;   // defaults are the S3Gen 24 kHz HiFT
    p.sample_rate = sample_rate;
    return p;
}

}  // namespace

struct codec_s3g_flow_build {
    int32_t t_in = 0;        // total token sequence length (prompt + speech)
    int32_t mel_len1 = 0;    // speaker prompt_feat frame count (2 * prompt_token_len, +1 when unpadded)
    int32_t sample_rate = 0; // S3GenSR (24000) for NSF harmonic frequencies
    const codec_model * model = nullptr;
};

static bool codec_s3g_build_flow(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out) {
    auto * p = static_cast<codec_s3g_flow_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || p->model == nullptr || out == nullptr || p->t_in <= 0) {
        return false;
    }
    const codec_model * model = p->model;
    const codec_chatterbox_s3g & s3g = *static_cast<const codec_chatterbox_s3g *>(model->impl);
    auto W = [&](const std::string & n) -> ggml_tensor * { return codec_graph_weight(ctx_eval, model, n); };

    const int32_t T = p->t_in;
    const int32_t T_total = T * 2;
    const int32_t mel_len1 = p->mel_len1;

    // ---- UpsampleConformerEncoder ----
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, T);
    ggml_set_name(t_tok, "s3g.flow.tokens");
    ggml_tensor * emb_table = W("s3g.flow.input_emb.w");
    if (emb_table == nullptr) return false;
    codec_conformer_params cp;
    cp.hidden = kFlowEncoderHidden;
    cp.n_heads = 8;
    ggml_tensor * h_tc = codec_block_conformer_embed(ctx_eval, model, "s3g.flow.enc.embed", cp,
                                                     ggml_get_rows(ctx_eval, emb_table, t_tok));
    h_tc = codec_block_pre_lookahead(ctx_eval, model, "s3g.flow.enc.pre", h_tc);
    h_tc = codec_block_conformer_stack(ctx_eval, model, "s3g.flow.enc.blk", cp, h_tc, kFlowEncoderLayers);
    h_tc = codec_block_upsample_conv(ctx_eval, model, "s3g.flow.enc.up", h_tc, 2);
    if (h_tc == nullptr) return false;
    h_tc = codec_block_conformer_embed(ctx_eval, model, "s3g.flow.enc.up_embed", cp,
                                       ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_tc)));
    h_tc = codec_block_conformer_stack(ctx_eval, model, "s3g.flow.enc.up_blk", cp, h_tc, kFlowEncoderUpLayers);
    if (h_tc == nullptr) return false;
    h_tc = codec_op_layer_norm_tc(ctx_eval, h_tc, cp.embed_eps, W("s3g.flow.enc.after_norm.w"),
                                  W("s3g.flow.enc.after_norm.b"));
    if (h_tc == nullptr) return false;
    ggml_tensor * mu_ct = codec_op_linear(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, h_tc)),
                                          W("s3g.flow.proj.w"), W("s3g.flow.proj.b"));   // [80, T_total]
    if (mu_ct == nullptr) return false;
    ggml_tensor * mu_tc = ggml_cont(ctx_eval, ggml_transpose(ctx_eval, mu_ct));
    ggml_set_name(mu_tc, "s3g.flow.mu");

    // Speaker embedding: F.normalize(embedding) (= rms_norm / sqrt(N)) → Linear(192, 80).
    ggml_tensor * sp_emb = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, kFlowSpkEmbedDim, 1);
    ggml_set_name(sp_emb, "s3g.flow.spk_embedding");
    ggml_tensor * sp_normed = ggml_scale(ctx_eval, ggml_rms_norm(ctx_eval, sp_emb, 1e-12f),
                                         1.0f / std::sqrt((float) kFlowSpkEmbedDim));
    ggml_tensor * sp_proj = codec_op_linear(ctx_eval, sp_normed, W("s3g.flow.spk_aff.w"), W("s3g.flow.spk_aff.b"));
    if (sp_proj == nullptr) return false;
    ggml_tensor * spks = ggml_reshape_1d(ctx_eval, sp_proj, kCfmOutChannels);

    // Cond [T_total, 80]: prompt_feat (mel_len1 frames) then zeros.
    ggml_tensor * pf_2d = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, kCfmOutChannels, mel_len1);
    ggml_set_name(pf_2d, "s3g.flow.prompt_feat");
    ggml_tensor * cond = ggml_pad(ctx_eval, ggml_cont(ctx_eval, ggml_transpose(ctx_eval, pf_2d)),
                                  T_total - mel_len1, 0, 0, 0);

    // ---- CFM (U-Net estimator, unrolled Euler + CFG) ----
    // The noise z is a graph input (ggml lacks a portable cross-backend RNG).
    ggml_tensor * t_z = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_total, kCfmOutChannels);
    ggml_set_name(t_z, "s3g.flow.noise_z");
    codec_cfm_unet_params up;
    codec_cfm_solver_params sp;
    sp.n_steps = s3g.n_timesteps;
    sp.cfg_rate = s3g.cfg_rate;
    sp.meanflow = s3g.meanflow;
    auto estimator = [&](ggml_tensor * x, ggml_tensor * mu, ggml_tensor * sk, ggml_tensor * cd,
                         float t, float r) -> ggml_tensor * {
        ggml_tensor * t_emb = codec_block_cfm_time_mlp(ctx_eval, model, "s3g.cfm", t, kCfmInChannels, kCfmTimeEmbedScale);
        if (t_emb == nullptr) return nullptr;
        if (sp.meanflow) {
            ggml_tensor * r_emb = codec_block_cfm_time_mlp(ctx_eval, model, "s3g.cfm", r, kCfmInChannels, kCfmTimeEmbedScale);
            ggml_tensor * mixer = W("s3g.cfm.t_mix.w");
            if (r_emb == nullptr || mixer == nullptr) return nullptr;
            t_emb = codec_mul_mat(ctx_eval, mixer, ggml_concat(ctx_eval, t_emb, r_emb, 0));
        }
        return codec_block_cfm_unet(ctx_eval, model, "s3g.cfm", up, x, mu, sk, cd, t_emb);
    };
    ggml_tensor * x = codec_block_cfm_solve(ctx_eval, sp, t_z, mu_tc, spks, cond, estimator);
    if (x == nullptr) return false;

    // Trim the prompt prefix: mel = x[mel_len1:T_total].
    const int32_t T_speech = T_total - mel_len1;
    ggml_tensor * mel_full = ggml_cont(ctx_eval, x);
    ggml_tensor * mel = ggml_cont(ctx_eval, ggml_view_2d(ctx_eval, mel_full, T_speech, kCfmOutChannels,
                                                         mel_full->nb[1], (size_t) mel_len1 * mel_full->nb[0]));
    ggml_set_name(mel, "s3g.flow.mel");

    // ---- HiFT ----
    const codec_hift_params hp = s3g_hift_params(p->sample_rate);
    const int32_t T_pcm = T_speech * codec_hift_upsample_factor(hp);
    ggml_tensor * t_phase = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, hp.nb_harmonics + 1);
    ggml_tensor * t_noise = ggml_new_tensor_2d(ctx_eval, GGML_TYPE_F32, T_pcm, hp.nb_harmonics + 1);
    ggml_set_name(t_phase, "s3g.hift.nsf_phase");
    ggml_set_name(t_noise, "s3g.hift.nsf_noise");
    ggml_tensor * pcm = codec_block_hift(ctx_eval, model, "s3g.hift", hp, mel, t_phase, t_noise);
    if (pcm == nullptr) return false;
    ggml_set_name(pcm, "s3g.flow.pcm");
    *out = pcm;
    return true;
}

// ---------------- speakers ---------------------
// Slots: prompt_token I32 [T], prompt_feat F32 [80, F] (24 kHz log-mel,
// F = 2T or 2T+1), embedding F32 [192] (CAMPPlus x-vector).

static bool codec_s3g_speaker_validate(
        const codec_model * model, const codec_speaker * spk, std::string * err) {
    const auto & s3g = *static_cast<const codec_chatterbox_s3g *>(model->impl);
    const codec_speaker_slot * tok = spk->slot("prompt_token");
    const codec_speaker_slot * feat = spk->slot("prompt_feat");
    const codec_speaker_slot * emb = spk->slot("embedding");
    if (tok == nullptr || feat == nullptr || emb == nullptr) {
        *err = "Chatterbox-S3G speaker needs prompt_token, prompt_feat and embedding slots";
        return false;
    }
    if (tok->type != GGML_TYPE_I32 || tok->ne[0] <= 0 || tok->nelements() != tok->ne[0]) {
        *err = "prompt_token must be a non-empty I32 vector";
        return false;
    }
    if (feat->type != GGML_TYPE_F32 || feat->ne[0] != kCfmOutChannels || feat->ne[1] <= 0 ||
            feat->nelements() != feat->ne[0] * feat->ne[1]) {
        *err = "prompt_feat must be F32 [80, frames]";
        return false;
    }
    if (feat->ne[1] > 2 * tok->ne[0] + 1) {
        *err = "prompt_feat has more frames than 2 * prompt_token + 1";
        return false;
    }
    if (emb->type != GGML_TYPE_F32 || emb->nelements() != kFlowSpkEmbedDim) {
        *err = "embedding must be F32 [192]";
        return false;
    }
    auto tokens_in_range = [](const codec_speaker_slot * t) {
        const auto * v = reinterpret_cast<const int32_t *>(t->data.data());
        for (int64_t i = 0; i < t->nelements(); ++i) {
            if (v[i] < 0 || v[i] >= kFlowVocabSize) return false;
        }
        return true;
    };
    if (!tokens_in_range(tok)) {
        *err = "prompt_token id out of range";
        return false;
    }
    if (!s3g.lm_encoder.empty()) {
        const codec_speaker_slot * lm = spk->slot("lm_cond");
        if (lm == nullptr || lm->type != GGML_TYPE_F32 || lm->nelements() != s3g.lm_cond_dim) {
            *err = "speaker needs an F32 lm_cond slot of " + std::to_string(s3g.lm_cond_dim) + " values";
            return false;
        }
        const codec_speaker_slot * ref = spk->slot("ref_tokens");
        if (ref != nullptr && (ref->type != GGML_TYPE_I32 || !tokens_in_range(ref))) {
            *err = "ref_tokens must be I32 S3 token ids";
            return false;
        }
    }
    return true;
}

// S3Gen.embed_ref as one graph: reference PCM (any rate) → resample to 24 kHz
// and truncate to the 10 s S3Gen conditions on → matcha log-mel (prompt_feat)
// and, at 16 kHz, S3 tokens (prompt_token) + CAMPPlus (embedding). Models with
// `spk.lm_encoder = xvector_sincnet` (KittenTTS 2) add the LM conditioning
// from the whole clip at 16 kHz: spk_proj(normalize(x-vector)) and the S3
// tokens of the in-context reference.
struct codec_s3g_speaker_build {
    int32_t n_samples = 0;
    int32_t sample_rate = 0;
    int32_t with_lm = 0;
    const codec_model * model = nullptr;
};

struct codec_s3g_speaker_lengths {
    int64_t n24 = 0;    // 24 kHz samples after truncation
    int64_t n16 = 0;    // 16 kHz samples of the truncated clip
    int64_t full16 = 0; // 16 kHz samples of the whole clip (LM conditioning)
};

static codec_s3g_speaker_lengths codec_s3g_speaker_lengths_for(int64_t n, int32_t sample_rate) {
    codec_s3g_speaker_lengths l;
    l.n24 = std::min<int64_t>(codec_resample_out_len(n, sample_rate, kS3gSampleRate),
                              (int64_t) kSpkRefMaxSeconds * kS3gSampleRate);
    l.n16 = codec_resample_out_len(l.n24, kS3gSampleRate, kS3SampleRate);
    l.full16 = codec_resample_out_len(n, sample_rate, kS3SampleRate);
    return l;
}

static bool codec_s3g_build_speaker(ggml_context * ctx, void * user_data, ggml_tensor ** out) {
    const auto * p = static_cast<const codec_s3g_speaker_build *>(user_data);
    if (ctx == nullptr || p == nullptr || p->model == nullptr || out == nullptr || p->n_samples <= 0) {
        return false;
    }
    const codec_s3g_speaker_lengths len = codec_s3g_speaker_lengths_for(p->n_samples, p->sample_rate);
    const codec_s3_tokenizer_params s3t;

    ggml_tensor * pcm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p->n_samples);
    ggml_set_name(pcm, "s3g.spk.pcm");

    ggml_tensor * wav24 = codec_op_resample_sinc_hann(ctx, pcm, p->sample_rate, kS3gSampleRate);
    if (wav24 == nullptr) return false;
    if (ggml_nelements(wav24) > len.n24) {
        wav24 = ggml_cont(ctx, ggml_view_1d(ctx, wav24, len.n24, 0));
    }
    ggml_tensor * wav16 = codec_op_resample_sinc_hann(ctx, wav24, kS3gSampleRate, kS3SampleRate);
    ggml_tensor * mel_fb = codec_graph_weight(ctx, p->model, "s3g.spk.mel_fb");
    if (wav16 == nullptr || mel_fb == nullptr) return false;

    ggml_tensor * feat = codec_op_matcha_log_mel(ctx, wav24, mel_fb, kSpkMelNFft, kSpkMelHop);   // [80, F]
    ggml_tensor * tokens = codec_block_s3_tokenizer(ctx, p->model, s3t, wav16);
    ggml_tensor * emb = codec_block_campplus(ctx, p->model, "s3g.cp", wav16);
    if (feat == nullptr || tokens == nullptr || emb == nullptr) return false;
    ggml_set_name(feat, "s3g.spk.prompt_feat");
    ggml_set_name(tokens, "s3g.spk.prompt_token");
    ggml_set_name(emb, "s3g.spk.embedding");
    ggml_set_output(feat);
    ggml_set_output(tokens);

    if (p->with_lm) {
        ggml_tensor * full16 = codec_op_resample_sinc_hann(ctx, pcm, p->sample_rate, kS3SampleRate);
        if (full16 == nullptr) return false;
        ggml_tensor * lm_cond = codec_block_xvector_sincnet(ctx, p->model, "kt.xv", "kt.spk_proj", full16);
        ggml_tensor * ref_tokens = codec_block_s3_tokenizer(ctx, p->model, s3t, full16);
        if (lm_cond == nullptr || ref_tokens == nullptr) return false;
        ggml_set_name(lm_cond, "s3g.spk.lm_cond");
        ggml_set_name(ref_tokens, "s3g.spk.ref_tokens");
        ggml_set_output(lm_cond);
        ggml_set_output(ref_tokens);
    }
    *out = emb;
    return true;
}

static enum codec_status codec_s3g_speaker_from_audio(
        codec_context * ctx, const std::vector<float> & pcm, int32_t sample_rate, codec_speaker * out) {
    const auto & s3g = *static_cast<const codec_chatterbox_s3g *>(ctx->model->impl);
    if (pcm.empty() || sample_rate <= 0 || pcm.size() > (size_t) INT32_MAX) {
        codec_context_set_error(ctx, "invalid reference audio");
        return CODEC_STATUS_INVALID_ARG;
    }
    const bool with_lm = s3g.lm_encoder == "xvector_sincnet";
    const codec_s3g_speaker_lengths len = codec_s3g_speaker_lengths_for((int64_t) pcm.size(), sample_rate);
    if (len.n24 <= (kSpkMelNFft - kSpkMelHop) / 2 || !codec_campplus_accepts(len.n16) ||
        (with_lm && !codec_xvector_sincnet_accepts(len.full16))) {
        codec_context_set_error(ctx, "reference audio too short for the speaker encoders");
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_s3g_speaker_build build;
    build.n_samples = (int32_t) pcm.size();
    build.sample_rate = sample_rate;
    build.with_lm = with_lm ? 1 : 0;
    build.model = ctx->model;

    std::string err;
    codec_graph_eval_guard guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx, {CODEC_GRAPH_CHATTERBOX_S3G_SPEAKER, build.n_samples, build.with_lm, sample_rate, 0, 0},
            codec_s3g_build_speaker, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "s3g.spk.pcm");
    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "s3g.spk.prompt_token");
    ggml_tensor * t_feat = codec_graph_get_tensor(ctx, entry, "s3g.spk.prompt_feat");
    ggml_tensor * t_emb = codec_graph_get_tensor(ctx, entry, "s3g.spk.embedding");
    ggml_tensor * t_lm = with_lm ? codec_graph_get_tensor(ctx, entry, "s3g.spk.lm_cond") : nullptr;
    ggml_tensor * t_ref = with_lm ? codec_graph_get_tensor(ctx, entry, "s3g.spk.ref_tokens") : nullptr;
    if (t_pcm == nullptr || t_tok == nullptr || t_feat == nullptr || t_emb == nullptr ||
        (with_lm && (t_lm == nullptr || t_ref == nullptr))) {
        codec_context_set_error(ctx, "invalid speaker graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_graph_compute(ctx, entry, std::max(1, ctx->model->n_threads), &err)) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> tokens((size_t) ggml_nelements(t_tok));
    std::vector<float> feat((size_t) ggml_nelements(t_feat));
    std::vector<float> embedding((size_t) ggml_nelements(t_emb));
    std::vector<float> lm_cond(t_lm ? (size_t) ggml_nelements(t_lm) : 0);
    std::vector<int32_t> ref_tokens(t_ref ? (size_t) ggml_nelements(t_ref) : 0);
    if (!codec_runtime_read_tensor(t_tok, tokens.data(), tokens.size() * sizeof(int32_t), &err) ||
        !codec_runtime_read_tensor(t_feat, feat.data(), feat.size() * sizeof(float), &err) ||
        !codec_runtime_read_tensor(t_emb, embedding.data(), embedding.size() * sizeof(float), &err) ||
        (t_lm && !codec_runtime_read_tensor(t_lm, lm_cond.data(), lm_cond.size() * sizeof(float), &err)) ||
        (t_ref && !codec_runtime_read_tensor(t_ref, ref_tokens.data(), ref_tokens.size() * sizeof(int32_t), &err))) {
        codec_context_set_error(ctx, "speaker graph: " + err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // embed_ref keeps mel_len == 2 * token_len by dropping trailing tokens.
    const int64_t n_feat = t_feat->ne[1];
    tokens.resize((size_t) std::min<int64_t>((int64_t) tokens.size(), n_feat / 2));

    const int64_t ne_tok[1] = {(int64_t) tokens.size()};
    const int64_t ne_feat[2] = {kCfmOutChannels, n_feat};
    const int64_t ne_emb[1] = {kFlowSpkEmbedDim};
    codec_speaker_add_slot(out, "prompt_token", GGML_TYPE_I32, ne_tok, 1, tokens.data());
    codec_speaker_add_slot(out, "prompt_feat", GGML_TYPE_F32, ne_feat, 2, feat.data());
    codec_speaker_add_slot(out, "embedding", GGML_TYPE_F32, ne_emb, 1, embedding.data());
    if (with_lm) {
        const int64_t ne_lm[1] = {(int64_t) lm_cond.size()};
        const int64_t ne_ref[1] = {(int64_t) ref_tokens.size()};
        codec_speaker_add_slot(out, "lm_cond", GGML_TYPE_F32, ne_lm, 1, lm_cond.data());
        codec_speaker_add_slot(out, "ref_tokens", GGML_TYPE_I32, ne_ref, 1, ref_tokens.data());
    }
    return CODEC_STATUS_SUCCESS;
}

// GGUFs converted before the speaker format carry one builtin voice as
// `chatterbox_s3g.cond.*` metadata + `s3g.cond.*` tensors; expose it as the
// speaker "default" so they keep decoding unchanged.
static bool codec_s3g_add_legacy_speaker(codec_model * model) {
    if (!codec_read_bool_kv(model->gguf, "chatterbox_s3g.has_builtin_conditioning", false)) {
        return true;
    }
    std::vector<int32_t> prompt_token;
    codec_read_i32_array_kv_vec(model->gguf, "chatterbox_s3g.cond.prompt_token", &prompt_token);
    const int32_t tok_len = codec_read_i32_kv(model->gguf, "chatterbox_s3g.cond.prompt_token_len", 0);
    if (tok_len <= 0 || tok_len > (int32_t) prompt_token.size()) return false;
    prompt_token.resize((size_t) tok_len);

    ggml_tensor * feat = ggml_get_tensor(model->weights, "s3g.cond.prompt_feat");
    ggml_tensor * emb = ggml_get_tensor(model->weights, "s3g.cond.embedding");
    if (feat == nullptr || emb == nullptr || feat->type != GGML_TYPE_F32 || emb->type != GGML_TYPE_F32 ||
            feat->ne[0] != kCfmOutChannels || ggml_nelements(emb) != kFlowSpkEmbedDim) {
        return false;
    }
    std::vector<float> feat_data((size_t) ggml_nelements(feat));
    std::vector<float> emb_data((size_t) ggml_nelements(emb));
    ggml_backend_tensor_get(feat, feat_data.data(), 0, ggml_nbytes(feat));
    ggml_backend_tensor_get(emb, emb_data.data(), 0, ggml_nbytes(emb));

    codec_speaker spk;
    spk.name = "default";
    const int64_t ne_tok[1] = {tok_len};
    const int64_t ne_feat[2] = {kCfmOutChannels, ggml_nelements(feat) / kCfmOutChannels};
    const int64_t ne_emb[1] = {kFlowSpkEmbedDim};
    codec_speaker_add_slot(&spk, "prompt_token", GGML_TYPE_I32, ne_tok, 1, prompt_token.data());
    codec_speaker_add_slot(&spk, "prompt_feat", GGML_TYPE_F32, ne_feat, 2, feat_data.data());
    codec_speaker_add_slot(&spk, "embedding", GGML_TYPE_F32, ne_emb, 1, emb_data.data());
    model->speakers.push_back(std::move(spk));
    return true;
}

// ---------------- model lifecycle / vtable ---------------------

enum codec_status codec_chatterbox_s3g_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3g & s3g = *static_cast<codec_chatterbox_s3g *>(model->impl);
    s3g.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", s3g.sample_rate);
    s3g.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", s3g.hop_size);
    s3g.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", s3g.n_q);
    s3g.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", s3g.codebook_size);
    s3g.meanflow = codec_read_bool_kv(model->gguf, "chatterbox_s3g.meanflow", s3g.meanflow);
    s3g.n_timesteps = codec_read_i32_kv(model->gguf, "chatterbox_s3g.flow.n_timesteps",
        s3g.meanflow ? kFlowMeanflowNTimesteps : kFlowDefaultNTimesteps);
    s3g.cfg_rate = codec_read_f32_kv(model->gguf, "chatterbox_s3g.flow.cfg_rate", kFlowCfgRate);
    s3g.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", s3g.has_encoder);
    s3g.has_speaker_encoder = codec_read_bool_kv(model->gguf, "chatterbox_s3g.spk.has_encoder", false);
    model->can_speaker_from_audio = s3g.has_speaker_encoder;
    s3g.lm_encoder = codec_read_str_kv(model->gguf, "chatterbox_s3g.spk.lm_encoder", "");
    s3g.lm_cond_dim = codec_read_i32_kv(model->gguf, "chatterbox_s3g.spk.lm_cond_dim", 0);
    if (!s3g.lm_encoder.empty() && (s3g.lm_encoder != "xvector_sincnet" || s3g.lm_cond_dim <= 0)) {
        return CODEC_STATUS_INVALID_ARG;
    }
    s3g.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", s3g.has_decoder);

    model->sample_rate = s3g.sample_rate;
    model->encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", 0);
    model->has_encoder = s3g.has_encoder;
    model->has_decoder = s3g.has_decoder;
    model->hop_size = s3g.hop_size;
    model->n_q = s3g.n_q;
    model->codebook_size = s3g.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = -1;

    if (s3g.n_q != 1 || s3g.codebook_size != kFlowVocabSize || s3g.sample_rate <= 0 || s3g.hop_size <= 0) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (s3g.n_timesteps < 1 || s3g.n_timesteps > kFlowMaxNTimesteps || !(s3g.cfg_rate >= 0.0f)) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (s3g.meanflow) {
        auto * mixer = ggml_get_tensor(model->weights, "s3g.cfm.t_mix.w");
        if (mixer == nullptr || mixer->ne[0] != 2 * kCfmTimeEmbedDim ||
                mixer->ne[1] != kCfmTimeEmbedDim || mixer->ne[2] != 1 || mixer->ne[3] != 1) {
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    if (!codec_s3g_add_legacy_speaker(model)) {
        return CODEC_STATUS_INVALID_ARG;
    }

    return CODEC_STATUS_SUCCESS;
}

// Public decode entry. Tokens → PCM via a single unified graph that runs the
// flow encoder, the unrolled CFM ODE (10 Euler steps × CFG cond/uncond) and
// the HiFT vocoder (f0_predictor + NSF source + STFT + main + iSTFT) all in
// one ggml graph evaluation. Only RNG sampling and the trim-fade tail remain
// CPU-side: random noise is fed in as input tensors, and trim-fade applies
// after reading the PCM out (it modifies the leading 40 ms only).
enum codec_status codec_chatterbox_s3g_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || ctx->model->impl == nullptr ||
        tokens == nullptr || tokens->data == nullptr || tokens->n_tokens <= 0 ||
        out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    auto * s3g = static_cast<codec_chatterbox_s3g *>(ctx->model->impl);
    if (!s3g->has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    const codec_speaker * spk = params.speaker != nullptr
        ? params.speaker
        : codec_get_speaker(ctx->model, nullptr);
    if (spk == nullptr) {
        codec_context_set_error(ctx, "Chatterbox-S3G decode needs a speaker (model has no default speaker)");
        return CODEC_STATUS_INVALID_ARG;
    }
    std::string err;
    if (params.speaker != nullptr && !codec_s3g_speaker_validate(ctx->model, spk, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INVALID_ARG;
    }
    const codec_speaker_slot * spk_tok = spk->slot("prompt_token");
    const codec_speaker_slot * spk_feat = spk->slot("prompt_feat");
    const codec_speaker_slot * spk_emb = spk->slot("embedding");

    // Build the full token list: prompt_token (speaker) + speech_token (filtered).
    std::vector<int32_t> tok_seq;
    const int32_t n_prompt = (int32_t) spk_tok->ne[0];
    const auto * prompt = reinterpret_cast<const int32_t *>(spk_tok->data.data());
    tok_seq.reserve((size_t) n_prompt + (size_t) tokens->n_tokens);
    tok_seq.insert(tok_seq.end(), prompt, prompt + n_prompt);
    for (int32_t i = 0; i < tokens->n_tokens; ++i) {
        const int32_t v = tokens->data[i];
        if (v >= 0 && v < s3g->codebook_size) tok_seq.push_back(v);
    }
    const int32_t T_tok = (int32_t) tok_seq.size();
    if (T_tok <= 0) {
        codec_context_set_error(ctx, "empty token sequence");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t mel_len1 = (int32_t) spk_feat->ne[1];
    const int32_t T_total = T_tok * 2;
    if (mel_len1 > T_total) {
        codec_context_set_error(ctx, "prompt_feat_frames exceeds encoder output length");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t T_speech = T_total - mel_len1;
    const codec_hift_params hp = s3g_hift_params(s3g->sample_rate);
    const int32_t T_pcm = T_speech * codec_hift_upsample_factor(hp);

    // Build/cache the unified graph.
    codec_s3g_flow_build build = {};
    build.t_in = T_tok;
    build.mel_len1 = mel_len1;
    build.sample_rate = s3g->sample_rate;
    build.model = ctx->model;

    codec_graph_eval_guard guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            {CODEC_GRAPH_CHATTERBOX_S3G_DECODE, T_tok, 0, 0, mel_len1, kFlowEncoderHidden},
            codec_s3g_build_flow,
            &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (entry == nullptr) {
        codec_context_set_error(ctx, "decode graph build returned no entry");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Required graph inputs.
    ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "s3g.flow.tokens");
    ggml_tensor * t_z = codec_graph_get_tensor(ctx, entry, "s3g.flow.noise_z");
    ggml_tensor * t_phase = codec_graph_get_tensor(ctx, entry, "s3g.hift.nsf_phase");
    ggml_tensor * t_nsf_noise = codec_graph_get_tensor(ctx, entry, "s3g.hift.nsf_noise");
    ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "s3g.flow.pcm");
    ggml_tensor * t_spk_emb = codec_graph_get_tensor(ctx, entry, "s3g.flow.spk_embedding");
    ggml_tensor * t_prompt_feat = codec_graph_get_tensor(ctx, entry, "s3g.flow.prompt_feat");
    if (t_tok == nullptr || t_z == nullptr || t_spk_emb == nullptr || t_prompt_feat == nullptr ||
        t_phase == nullptr || t_nsf_noise == nullptr || t_pcm == nullptr) {
        codec_context_set_error(ctx, "invalid decode graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // CFM noise + NSF random inputs.
    std::vector<float> noise_z((size_t) T_total * (size_t) kCfmOutChannels, 0.0f);
    std::vector<float> nsf_phase, nsf_noise;
    {
        std::mt19937_64 rng(/*seed=*/0);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t i = 0; i < noise_z.size(); ++i) noise_z[i] = gauss(rng);
        codec_hift_fill_random(hp, T_pcm, rng, &nsf_phase, &nsf_noise);
    }

    if (!codec_runtime_write_tensor(t_tok, tok_seq.data(), tok_seq.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_spk_emb, spk_emb->data.data(), spk_emb->data.size(), &err) ||
        !codec_runtime_write_tensor(t_prompt_feat, spk_feat->data.data(), spk_feat->data.size(), &err) ||
        !codec_runtime_write_tensor(t_z, noise_z.data(), noise_z.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_phase, nsf_phase.data(), nsf_phase.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_nsf_noise, nsf_noise.data(), nsf_noise.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_compute(ctx, entry, std::max(1, ctx->model->n_threads), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_pcm = (int32_t) t_pcm->ne[0];
    std::vector<float> pcm((size_t) n_pcm, 0.0f);
    if (!codec_runtime_read_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // trim_fade: silence the first 20 ms then cosine-fade in the next 20 ms.
    const int32_t n_trim = s3g->sample_rate / 50;
    if (n_trim > 0) {
        for (int32_t i = 0; i < n_trim && i < (int32_t) pcm.size(); ++i) pcm[(size_t) i] = 0.0f;
        for (int32_t i = 0; i < n_trim && (n_trim + i) < (int32_t) pcm.size(); ++i) {
            const float angle = (float) M_PI * (1.0f - (float) i / (float) n_trim);
            pcm[(size_t) (n_trim + i)] *= 0.5f * (std::cos(angle) + 1.0f);
        }
    }

    float * data = static_cast<float *>(std::malloc(pcm.size() * sizeof(float)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, pcm.data(), pcm.size() * sizeof(float));

    out_pcm->data = data;
    out_pcm->n_samples = (int32_t) pcm.size();
    out_pcm->sample_rate = s3g->sample_rate;
    out_pcm->n_channels = 1;
    (void) params;
    return CODEC_STATUS_SUCCESS;
}

// Single-graph token → mel decode. The whole flow encoder + CFM ODE (10 Euler
// steps × CFG cond/uncond passes) is unrolled inside `codec_s3g_build_flow`;
// here we just feed token IDs and a noise tensor and read out the mel.

static void * codec_chatterbox_s3g_create_impl() {
    return new (std::nothrow) codec_chatterbox_s3g();
}

static void codec_chatterbox_s3g_destroy_impl(void * ptr) {
    delete static_cast<codec_chatterbox_s3g *>(ptr);
}

const struct codec_model_vtable * codec_chatterbox_s3g_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_CHATTERBOX_S3G,
        "Chatterbox-S3G",
        codec_chatterbox_s3g_create_impl,
        codec_chatterbox_s3g_destroy_impl,
        codec_chatterbox_s3g_init,
        codec_graph_size_exact,
        nullptr,
        codec_chatterbox_s3g_decode,
        nullptr,
        codec_s3g_speaker_validate,
        codec_s3g_speaker_from_audio,
    };
    return &vtable;
}
