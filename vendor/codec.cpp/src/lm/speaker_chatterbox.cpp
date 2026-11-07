#include "speaker_chatterbox.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"
#include "../ops/ggml_ops.h"
#include "../blocks/chatterbox_voice_encoder.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// Chatterbox speaker encoder runtime: one cached graph per call.
//
//   ref_pcm (16 kHz mono)
//     │  VoiceEncoder block (src/blocks/chatterbox_voice_encoder): power mel
//     │  → 160-frame partials → 3-layer LSTM → proj → L2 → mean → L2
//     ▼
//   spk_emb_raw  (256,)            ← or the caller's embedding (from_emb)
//     │  cond_enc + perceiver + concat
//     │      spkr_enc(spk_emb_raw)                       → cond_spkr      (1,    H=1024)
//     │      perceiver(speech_emb(ref_tokens) + pos)     → cond_speech    (32,   H=1024)
//     │      emotion_adv_fc(emotion_scalar)              → cond_emotion   (1,    H=1024)
//     │      concat([cond_spkr, cond_speech, cond_emot]) → cond_emb       (34,   H=1024)
//     ▼
//   cond_emb  (34, 1024)  →  user-provided out buffer
// =====================================================================

namespace {

struct cbx_impl {
    codec_voice_encoder_params ve;

    int32_t hidden_dim    = 1024;   // backbone hidden (= cond_emb row width)
    int32_t n_rows        = 34;
    int32_t speaker_embed_dim = 256;

    // cond_enc layout.  Turbo drops the perceiver (raw speech_emb rows of the
    // cond prompt), the emotion row and the learned speech PE.
    bool use_perceiver   = true;
    bool has_emotion     = true;
    bool learned_pos_emb = true;

    // cond_enc graph tensors (handles into model->weights).
    ggml_tensor * spkr_enc_W       = nullptr; // (E_speaker=256, H=1024) F16/F32
    ggml_tensor * spkr_enc_b       = nullptr; // (H,)
    ggml_tensor * emotion_adv_W    = nullptr; // (1, H)
    ggml_tensor * speech_emb       = nullptr; // (vocab, H)
    ggml_tensor * speech_pos_emb   = nullptr; // (max_speech_tokens, H)
    ggml_tensor * perceiver_q      = nullptr; // (1, 32, H) — 32 learned queries
    ggml_tensor * perc_norm_W      = nullptr; // (H,)
    ggml_tensor * perc_norm_b      = nullptr; // (H,)
    ggml_tensor * perc_to_q_W      = nullptr; // (H, H)
    ggml_tensor * perc_to_q_b      = nullptr; // (H,)
    ggml_tensor * perc_to_k_W      = nullptr; // (H, H)
    ggml_tensor * perc_to_k_b      = nullptr; // (H,)
    ggml_tensor * perc_to_v_W      = nullptr; // (H, H)
    ggml_tensor * perc_to_v_b      = nullptr; // (H,)
    ggml_tensor * perc_proj_out_W  = nullptr; // (H, H)
    ggml_tensor * perc_proj_out_b  = nullptr; // (H,)

    // Lazily-allocated context for the cond_enc graph (one per lm, owned).
    codec_context * cond_ctx = nullptr;
};

ggml_tensor * find_required(codec_lm * lm, const char * name) {
    ggml_tensor * t = ggml_get_tensor(lm->codec->weights, name);
    if (t == nullptr) {
        lm->last_error = std::string("speaker(chatterbox): missing tensor: ") + name;
    }
    return t;
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
    gguf_context * gf = lm->codec->gguf;

    cbx_impl * impl = new (std::nothrow) cbx_impl();
    if (impl == nullptr) {
        lm->last_error = "speaker(chatterbox): out of memory";
        return false;
    }

    codec_voice_encoder_params & ve = impl->ve;
    ve.n_mels         = codec_read_i32_kv(gf, "codec.speaker.ve.num_mels", ve.n_mels);
    ve.hidden         = codec_read_i32_kv(gf, "codec.speaker.ve.hidden_size", ve.hidden);
    ve.n_layers       = codec_read_i32_kv(gf, "codec.speaker.ve.num_layers", ve.n_layers);
    ve.embed          = codec_read_i32_kv(gf, "codec.speaker.ve.speaker_embed_dim", ve.embed);
    ve.n_fft          = codec_read_i32_kv(gf, "codec.speaker.ve.n_fft", ve.n_fft);
    ve.hop            = codec_read_i32_kv(gf, "codec.speaker.ve.hop_size", ve.hop);
    ve.partial_frames = codec_read_i32_kv(gf, "codec.speaker.ve.partial_frames", ve.partial_frames);
    ve.sample_rate    = lm->speaker_info.ref_sample_rate;
    ve.overlap        = codec_read_f32_kv(gf, "codec.speaker.ve.overlap", ve.overlap);
    ve.rate           = codec_read_f32_kv(gf, "codec.speaker.ve.rate", ve.rate);
    ve.min_coverage   = codec_read_f32_kv(gf, "codec.speaker.ve.min_coverage", ve.min_coverage);
    ve.final_relu     = codec_read_bool_kv(gf, "codec.speaker.ve.final_relu", ve.final_relu);
    impl->hidden_dim     = lm->speaker_info.hidden_dim;
    impl->n_rows         = lm->speaker_info.n_rows;
    impl->speaker_embed_dim = ve.embed;
    for (const char * t : {"speaker.voice_encoder.mel_basis", "speaker.voice_encoder.window",
                           "speaker.voice_encoder.proj.weight", "speaker.voice_encoder.proj.bias"}) {
        if (!find_required(lm, t)) { delete impl; return false; }
    }
    for (int32_t l = 0; l < ve.n_layers; ++l) {
        for (const char * part : {"W_ih", "W_hh", "b_ih", "b_hh"}) {
            const std::string n = "speaker.voice_encoder.lstm_" + std::to_string(l) + "." + part;
            if (!find_required(lm, n.c_str())) { delete impl; return false; }
        }
    }

    // cond_enc / perceiver weights (live under `lm.chatterbox.cond.*` —
    // bundled by the LM adaptor section, so this is only ever called for
    // GGUFs that also have the LM adaptor; the speaker section is
    // gated on `lm_source` at convert time).
    impl->use_perceiver   = codec_read_bool_kv(gf, "codec.lm.chatterbox.use_perceiver", true);
    impl->has_emotion     = codec_read_bool_kv(gf, "codec.lm.chatterbox.has_emotion_cond", true);
    impl->learned_pos_emb = codec_read_bool_kv(gf, "codec.lm.chatterbox.learned_pos_emb", true);
    impl->spkr_enc_W      = find_required(lm, "lm.chatterbox.cond.spkr_enc.weight");
    impl->spkr_enc_b      = find_required(lm, "lm.chatterbox.cond.spkr_enc.bias");
    impl->speech_emb      = find_required(lm, "lm.audio_embd_0.weight");
    if (!impl->spkr_enc_W || !impl->spkr_enc_b || !impl->speech_emb) {
        delete impl; return false;
    }
    if (impl->has_emotion) {
        impl->emotion_adv_W = find_required(lm, "lm.chatterbox.cond.emotion_adv_fc.weight");
        if (!impl->emotion_adv_W) { delete impl; return false; }
    }
    if (impl->learned_pos_emb) {
        impl->speech_pos_emb = find_required(lm, "lm.chatterbox.speech_pos_emb.weight");
        if (!impl->speech_pos_emb) { delete impl; return false; }
    }
    if (!impl->use_perceiver) {
        lm->speaker_impl = impl;
        return true;
    }
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
    if (!impl->perceiver_q ||
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
// cond_enc + perceiver graph
// ---------------------------------------------------------------------

namespace {

struct cbx_cond_build_data {
    cbx_impl * impl;
    int32_t    T_speech;
    int32_t    n_samples;   // > 0: run the VoiceEncoder on "spk.in.pcm"
    const codec_model * model;
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
ggml_tensor * perceiver_attn_block(
        ggml_context * ctx,
        cbx_impl * impl,
        ggml_tensor * x1,        // (H, T_q)
        ggml_tensor * x2,        // (H, T_k)
        int32_t n_heads) {
    const int64_t H   = x1->ne[0];
    const int64_t T_q = x1->ne[1];
    const int64_t T_k = x2->ne[1];
    const int64_t head_dim = H / n_heads;
    const float   scale    = 1.0f / std::sqrt((float) head_dim);

    ggml_tensor * norm_W = codec_graph_cast_f32(ctx, impl->perc_norm_W);
    ggml_tensor * norm_b = codec_graph_cast_f32(ctx, impl->perc_norm_b);

    ggml_tensor * x1_norm = ggml_norm(ctx, x1, 1e-5f);
    x1_norm = ggml_add(ctx, ggml_mul(ctx, x1_norm, ggml_repeat(ctx,
                ggml_reshape_2d(ctx, norm_W, H, 1), x1_norm)),
                ggml_repeat(ctx, ggml_reshape_2d(ctx, norm_b, H, 1), x1_norm));
    ggml_tensor * x2_norm = ggml_norm(ctx, x2, 1e-5f);
    x2_norm = ggml_add(ctx, ggml_mul(ctx, x2_norm, ggml_repeat(ctx,
                ggml_reshape_2d(ctx, norm_W, H, 1), x2_norm)),
                ggml_repeat(ctx, ggml_reshape_2d(ctx, norm_b, H, 1), x2_norm));

    // Linear projections: q/k/v.  Weights are (in=H, out=H) stored as
    // (out, in) row-major → ggml ne[0]=H_in, ne[1]=H_out — matches the
    // mul_mat src[0] convention.
    auto linear_HH = [&](ggml_tensor * W, ggml_tensor * b, ggml_tensor * x_2d) {
        ggml_tensor * W_lhs = codec_graph_mat_lhs(ctx, W);
        ggml_tensor * y = codec_mul_mat(ctx, W_lhs, x_2d);   // (H_out, T)
        ggml_tensor * b_f32 = codec_graph_cast_f32(ctx, b);
        ggml_tensor * b_2d  = ggml_reshape_2d(ctx, b_f32, H, 1);
        y = ggml_add(ctx, y, b_2d);
        return y;
    };
    ggml_tensor * q = linear_HH(impl->perc_to_q_W, impl->perc_to_q_b, x1_norm);  // (H, T_q)
    ggml_tensor * k = linear_HH(impl->perc_to_k_W, impl->perc_to_k_b, x2_norm);  // (H, T_k)
    ggml_tensor * v = linear_HH(impl->perc_to_v_W, impl->perc_to_v_b, x2_norm);  // (H, T_k)

    // Split into heads: (H, T) → (head_dim, n_heads, T) → permute to
    // (head_dim, T, n_heads) for attention math (ggml_mul_mat on ne[0]).
    auto split_heads = [&](ggml_tensor * t, int64_t T) {
        ggml_tensor * t3 = ggml_reshape_3d(ctx, t, head_dim, n_heads, T);
        return ggml_cont(ctx, ggml_permute(ctx, t3, 0, 2, 1, 3));  // (head_dim, T, n_heads)
    };
    ggml_tensor * q_p = split_heads(q, T_q);   // (head_dim, T_q, n_heads)
    ggml_tensor * k_p = split_heads(k, T_k);
    ggml_tensor * v_p = split_heads(v, T_k);

    // sim = k_p^T @ q_p : ggml_mul_mat(a=k_p, b=q_p) contracts on ne[0]=head_dim
    //   → (T_k, T_q, n_heads)
    ggml_tensor * sim = codec_mul_mat(ctx, k_p, q_p);
    sim = ggml_scale(ctx, sim, scale);
    ggml_tensor * attn = ggml_soft_max(ctx, sim);              // softmax over ne[0]=T_k

    // out = v_p^T @ attn : but we want (head_dim, T_q, n_heads).
    // v_p is (head_dim, T_k, n_heads), attn is (T_k, T_q, n_heads).
    // ggml_mul_mat needs ne[0]s to match — permute v_p to (T_k, head_dim, n_heads)
    // so contraction is on T_k, result (head_dim, T_q, n_heads).
    ggml_tensor * v_pT = ggml_cont(ctx, ggml_permute(ctx, v_p, 1, 0, 2, 3));
    ggml_tensor * out  = codec_mul_mat(ctx, v_pT, attn);          // (head_dim, T_q, n_heads)

    // Merge heads: (head_dim, T_q, n_heads) → (head_dim, n_heads, T_q) → (H, T_q).
    ggml_tensor * out_merged = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out_merged = ggml_reshape_2d(ctx, out_merged, H, T_q);

    // proj_out + residual.
    ggml_tensor * h = linear_HH(impl->perc_proj_out_W, impl->perc_proj_out_b, out_merged);
    return ggml_add(ctx, x1, h);
}

bool build_cond_graph(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    auto * b = static_cast<cbx_cond_build_data *>(ud);
    if (ctx_eval == nullptr || b == nullptr || b->impl == nullptr || out == nullptr) return false;
    cbx_impl * impl = b->impl;
    const int64_t H  = impl->hidden_dim;
    const int64_t T_speech = b->T_speech;

    // ---- Inputs ---------------------------------------------------
    ggml_tensor * t_spk = nullptr;
    if (b->n_samples > 0) {
        ggml_tensor * pcm = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, b->n_samples);
        ggml_set_name(pcm, "spk.in.pcm");
        t_spk = codec_block_chatterbox_voice_encoder(ctx_eval, b->model, "speaker.voice_encoder", impl->ve, pcm);
        if (t_spk == nullptr) return false;
    } else {
        t_spk = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, impl->speaker_embed_dim);
        ggml_set_name(t_spk, "spk.in.spkr_emb");
    }
    ggml_tensor * t_tokens  = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_I32, T_speech);
    ggml_set_name(t_tokens, "spk.in.tokens");

    // ---- cond_spkr = spkr_enc(spk_emb_raw) ----------------------
    //   spkr_enc.weight is (256, 1024) — ggml ne[0]=256, ne[1]=1024.
    //   y = W^T @ x : ggml_mul_mat(W, x_256) gives (1024,) (contracts on ne[0]).
    ggml_tensor * spk_W = codec_graph_mat_lhs(ctx_eval, impl->spkr_enc_W);
    ggml_tensor * spk_y = codec_mul_mat(ctx_eval, spk_W, t_spk);   // (1024,)
    ggml_tensor * spk_b = codec_graph_cast_f32(ctx_eval, impl->spkr_enc_b);
    spk_y = ggml_add(ctx_eval, spk_y, spk_b);
    ggml_tensor * cond_spkr = ggml_reshape_2d(ctx_eval, spk_y, H, 1);   // (H, 1)

    // ---- speech_emb lookup (+ speech_pos_emb) → seq (H, T_speech) ----
    ggml_tensor * emb_rows = ggml_get_rows(ctx_eval, impl->speech_emb, t_tokens);   // (H, T_speech)
    ggml_tensor * seq = codec_graph_cast_f32(ctx_eval, emb_rows);
    if (impl->learned_pos_emb) {
        // pos_ids = [0, 1, ..., T_speech-1]
        ggml_tensor * pos_ids = codec_op_arange(ctx_eval, 0.0f, (float) T_speech, 1.0f);
        pos_ids = ggml_cast(ctx_eval, pos_ids, GGML_TYPE_I32);
        ggml_tensor * pos_rows = ggml_get_rows(ctx_eval, impl->speech_pos_emb, pos_ids);
        seq = ggml_add(ctx_eval, seq, codec_graph_cast_f32(ctx_eval, pos_rows));
    }

    if (!impl->use_perceiver) {
        // Turbo: [cond_spkr (1) | seq (T_speech)] (no emotion row).
        ggml_tensor * cond = ggml_concat(ctx_eval, cond_spkr, seq, 1);
        ggml_set_name(cond, "spk.out.cond_emb");
        *out = cond;
        return true;
    }

    // ---- cond_emotion = emotion_adv_fc(emotion) --------------------
    //   emotion_adv_fc.weight is (1, 1024) — ggml ne[0]=1, ne[1]=1024.
    //   y = W @ scalar : ggml_mul_mat(W_2d, scalar_1d).  Result (1024,).
    ggml_tensor * t_emotion = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, 1);
    ggml_set_name(t_emotion, "spk.in.emotion");
    ggml_tensor * emo_W = codec_graph_mat_lhs(ctx_eval, impl->emotion_adv_W);
    ggml_tensor * emo_y = codec_mul_mat(ctx_eval, emo_W, t_emotion);   // (1024,)
    ggml_tensor * cond_emotion = ggml_reshape_2d(ctx_eval, emo_y, H, 1);

    // ---- Perceiver: 32 learned queries cross-attend over seq, then self-attend.
    // perceiver_q stored as (1, 32, H) — flatten to (H, 32).
    ggml_tensor * pq_f32 = codec_graph_cast_f32(ctx_eval, impl->perceiver_q);
    ggml_tensor * queries = ggml_reshape_2d(ctx_eval, pq_f32, H, 32);   // (H, 32)
    queries = ggml_cont(ctx_eval, queries);

    ggml_tensor * pre_att = perceiver_attn_block(ctx_eval, impl, queries, seq, /*n_heads=*/4);
    ggml_tensor * att     = perceiver_attn_block(ctx_eval, impl, pre_att, pre_att, /*n_heads=*/4);
    // att shape: (H, 32)

    // ---- Concat: [cond_spkr (1) | att (32) | cond_emotion (1)] → (H, 34)
    ggml_tensor * cat0 = ggml_concat(ctx_eval, cond_spkr, att, 1);      // (H, 33)
    ggml_tensor * cond = ggml_concat(ctx_eval, cat0, cond_emotion, 1);  // (H, 34)
    ggml_set_name(cond, "spk.out.cond_emb");
    *out = cond;
    return true;
}

}  // namespace

// One cond graph call: `pcm` (VoiceEncoder in-graph) or `speaker_emb`.
static enum codec_status cbx_run_cond_graph(
        codec_lm * lm, cbx_impl * impl,
        const float * pcm, int32_t n_samples, const float * speaker_emb,
        const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
        float emotion, float * out, int32_t out_n_elems) {
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

    cbx_cond_build_data build = { impl, n_ref_speech_tokens, pcm != nullptr ? n_samples : 0, lm->codec };
    codec_graph_eval_guard guard(impl->cond_ctx);
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind     = CODEC_GRAPH_LM_SPEAKER_CHATTERBOX;
    key.n_frames = n_ref_speech_tokens;
    key.n_in     = build.n_samples;
    if (!codec_graph_cache_get_or_build(
            impl->cond_ctx, key, build_cond_graph, &build, sizeof(build),
            &entry, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_in      = codec_graph_get_tensor(impl->cond_ctx, entry, pcm != nullptr ? "spk.in.pcm" : "spk.in.spkr_emb");
    ggml_tensor * t_tokens  = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.in.tokens");
    ggml_tensor * t_emotion = impl->use_perceiver
        ? codec_graph_get_tensor(impl->cond_ctx, entry, "spk.in.emotion") : nullptr;
    ggml_tensor * t_out     = codec_graph_get_tensor(impl->cond_ctx, entry, "spk.out.cond_emb");
    if (!t_in || !t_tokens || (impl->use_perceiver && !t_emotion) || !t_out) {
        lm->last_error = "chatterbox speaker_encode: cond graph missing tensors";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(impl->cond_ctx, entry, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const size_t in_bytes = pcm != nullptr ? (size_t) n_samples * sizeof(float)
                                           : (size_t) impl->speaker_embed_dim * sizeof(float);
    if (!codec_runtime_write_tensor(t_in, pcm != nullptr ? pcm : speaker_emb, in_bytes, &err) ||
        !codec_runtime_write_tensor(t_tokens, ref_speech_tokens,
                                    (size_t) n_ref_speech_tokens * sizeof(int32_t), &err) ||
        (t_emotion && !codec_runtime_write_tensor(t_emotion, &emotion, sizeof(float), &err))) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t nt = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
    if (!codec_graph_compute(impl->cond_ctx, entry, nt, &err)) {
        lm->last_error = "chatterbox speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t rows = impl->use_perceiver ? impl->n_rows : 1 + n_ref_speech_tokens;
    const int32_t need = rows * impl->hidden_dim;
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

    if (ref_pcm == nullptr || ref_pcm->data == nullptr || ref_pcm->n_samples <= 0) {
        lm->last_error = "chatterbox speaker_encode: ref_pcm missing";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->pcm_type != CODEC_PCM_TYPE_F32) {
        lm->last_error = "chatterbox speaker_encode: ref_pcm must be PCM_TYPE_F32";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->sample_rate != impl->ve.sample_rate) {
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

    if (codec_voice_encoder_n_partials(impl->ve, (int64_t) pcm.size()) <= 0) {
        lm->last_error = "chatterbox speaker_encode: ref_pcm too short";
        return CODEC_STATUS_INVALID_ARG;
    }
    return cbx_run_cond_graph(lm, impl, pcm.data(), (int32_t) pcm.size(), nullptr,
                              ref_speech_tokens, n_ref_speech_tokens, emotion, out, out_n_elems);
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
    return cbx_run_cond_graph(lm, impl, nullptr, 0, speaker_emb,
                              ref_speech_tokens, n_ref_speech_tokens, emotion, out, out_n_elems);
}
