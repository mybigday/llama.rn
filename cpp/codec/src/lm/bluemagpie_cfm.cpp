#include "lm_internal.h"

#include "../models/bluemagpie_blocks.h"
#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// codec_lm kind: continuous_latent_cfm  (VoxCPM / BlueMagpie)
//
// The backbone (Barbet, in llama.cpp) hands in a hidden state per AR step.
// This adaptor runs the whole continuous-latent generation step inside ONE
// cached ggml graph:
//
//   h_in ─ tslm_adapter ─ FSQ ─ lm_hidden ┐
//          fusion_concat_proj(lm_hidden, prev_feedback_lm) ─ RALM(causal) ─ residual_hidden
//   mu = [lm_to_dit(lm_hidden) ; res_to_dit(residual_hidden)]
//   patch = LocDiT/CFM(mu, cond=prev_patch, z)         (Euler × CFG, unrolled)
//   stop  = stop_head(lm_hidden)
//   LocEnc(patch) ─ enc_to_tslm_proj ─ feedback (→ next backbone step)
//                 ─ enc_to_lm_proj   ─ feedback_lm (→ next RALM input)
//
// RALM uses full-sequence recompute over the accumulated inputs each step
// (O(N^2) but correct + reuses the parity-verified block; incremental KV is a
// later optimisation — see residual_depth_ar's KV variant).
// =====================================================================

namespace {

struct cfm_impl {
    int32_t h_barbet, h_vox, h_enc, h_dit;
    int32_t latent_dim, patch_size, n_mu;
    int32_t n_locenc, n_locdit, n_ralm;
    int32_t n_heads, n_kv, head_dim;
    int32_t fsq_scale;
    float   eps;
};

struct cfm_state {
    int32_t            kv_pos = 0;       // RALM positions cached so far (= T)
    std::vector<float> prev_patch;      // [latent_dim*patch_size] cond
    std::vector<float> prev_feedback_lm;// [h_vox]
    std::vector<float> feedback_tslm;   // [h_barbet]  (for step_feedback_embd)

    // Persistent per-layer RALM K/V cache (incremental decode, llama.cpp /
    // residual_depth_ar style).  Each step computes Q/K/V for the 1 new token,
    // attends over cache[0..kv_pos] + new, and writes the new K/V back — so
    // RALM is O(kv_pos) per step instead of an O(T) full recompute (O(N^2)
    // total instead of O(N^3)).
    lm_ggml_context *             ctx_kv = nullptr;
    lm_ggml_backend_buffer_t      buf_kv = nullptr;
    std::vector<lm_ggml_tensor *> k_cache;   // [n_ralm] (head_dim, n_kv, max_T)
    std::vector<lm_ggml_tensor *> v_cache;   // [n_ralm]
    int32_t                    max_T = 0;
};

lm_ggml_tensor * lin(lm_ggml_context * c, lm_ggml_tensor * w, lm_ggml_tensor * x, lm_ggml_tensor * b) {
    lm_ggml_tensor * y = lm_ggml_mul_mat(c, w, x);
    return b ? lm_ggml_add(c, y, codec_graph_cast_f32(c, b)) : y;
}

// One RALM layer, incremental KV (1 new token).  Causal, no rope, no qk_norm.
// Attends over k/v_cache[0..kv_pos] + the new token; writes new K/V to slot
// kv_pos (side-effect roots).  O(kv_pos) instead of an O(T) full recompute.
lm_ggml_tensor * bm_ralm_kv_step(lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix,
    const codec_model * model, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * k_cache_l, lm_ggml_tensor * v_cache_l, int32_t kv_pos) {

    auto W = [&](const char * s) { return codec_graph_weight(ctx, model, prefix + s); };
    const int32_t q_dim = n_heads * head_dim;

    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln1.w"));
    lm_ggml_tensor * q  = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, W(".attn_q.w"), h), head_dim, n_heads, 1);
    lm_ggml_tensor * kk = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, W(".attn_k.w"), h), head_dim, n_kv, 1);
    lm_ggml_tensor * vv = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, W(".attn_v.w"), h), head_dim, n_kv, 1);

    lm_ggml_tensor * k_all = kk, * v_all = vv;
    if (kv_pos > 0) {
        lm_ggml_tensor * k_old = lm_ggml_view_3d(ctx, k_cache_l, head_dim, n_kv, kv_pos, k_cache_l->nb[1], k_cache_l->nb[2], 0);
        lm_ggml_tensor * v_old = lm_ggml_view_3d(ctx, v_cache_l, head_dim, n_kv, kv_pos, v_cache_l->nb[1], v_cache_l->nb[2], 0);
        k_all = lm_ggml_concat(ctx, k_old, kk, 2);
        v_all = lm_ggml_concat(ctx, v_old, vv, 2);
    }
    lm_ggml_tensor * q_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q, 0, 2, 1, 3));
    lm_ggml_tensor * k_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k_all, 0, 2, 1, 3));
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_p, q_p);
    scores = lm_ggml_scale(ctx, scores, 1.0f / std::sqrt((float) head_dim));
    scores = lm_ggml_diag_mask_inf(ctx, scores, kv_pos);
    scores = lm_ggml_soft_max(ctx, scores);
    lm_ggml_tensor * v_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_all, 1, 2, 0, 3));
    lm_ggml_tensor * attn = lm_ggml_cont(ctx, lm_ggml_permute(ctx, lm_ggml_mul_mat(ctx, v_p, scores), 0, 2, 1, 3));
    attn = lm_ggml_reshape_2d(ctx, attn, q_dim, 1);
    x_ht = lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, W(".attn_o.w"), attn));

    // write new K/V into the persistent cache at slot kv_pos (side-effect roots)
    lm_ggml_tensor * k_dst = lm_ggml_view_3d(ctx, k_cache_l, head_dim, n_kv, 1, k_cache_l->nb[1], k_cache_l->nb[2], (size_t) kv_pos * k_cache_l->nb[2]);
    lm_ggml_tensor * v_dst = lm_ggml_view_3d(ctx, v_cache_l, head_dim, n_kv, 1, v_cache_l->nb[1], v_cache_l->nb[2], (size_t) kv_pos * v_cache_l->nb[2]);
    lm_ggml_set_output(lm_ggml_cpy(ctx, kk, k_dst));
    lm_ggml_set_output(lm_ggml_cpy(ctx, vv, v_dst));

    h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln2.w"));
    lm_ggml_tensor * g = lm_ggml_silu(ctx, lm_ggml_mul_mat(ctx, W(".gate.w"), h));
    lm_ggml_tensor * u = lm_ggml_mul_mat(ctx, W(".up.w"), h);
    return lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, W(".down.w"), lm_ggml_mul(ctx, g, u)));
}

// Build the whole per-step graph.
struct cfm_build {
    cfm_impl  imp;
    int32_t   kv_pos;       // RALM positions cached so far
    int32_t   n_real;       // CFM real (non-zero-init) steps
    float     cfg_value;
    float     dt[64];
    lm_ggml_tensor * k_cache[32];
    lm_ggml_tensor * v_cache[32];
    const codec_model * model;
};

bool build_step(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    cfm_build * p = static_cast<cfm_build *>(ud);
    const cfm_impl & I = p->imp;
    const codec_model * model = p->model;
    auto W = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    const int32_t P = I.patch_size, D = I.latent_dim;

    lm_ggml_tensor * h_in   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_barbet, 1);  lm_ggml_set_name(h_in, "bm.cfm.h_in");
    lm_ggml_tensor * pfb_lm = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_vox, 1);      lm_ggml_set_name(pfb_lm, "bm.cfm.pfb_lm");
    lm_ggml_tensor * cond   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, P);            lm_ggml_set_name(cond, "bm.cfm.cond");
    lm_ggml_tensor * z      = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, P);            lm_ggml_set_name(z, "bm.cfm.z");
    lm_ggml_tensor * tsin   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_dit, p->n_real); lm_ggml_set_name(tsin, "bm.cfm.tsin");
    lm_ggml_tensor * dtsin  = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_dit, 1);       lm_ggml_set_name(dtsin, "bm.cfm.dtsin");

    // ---- tslm_adapter(h_in) → h_vox ----
    lm_ggml_tensor * a = codec_op_rms_norm_ct(ctx, h_in, I.eps, W("lm.tslm_adapter.norm.w"));
    a = lin(ctx, W("lm.tslm_adapter.proj.w"), a, W("lm.tslm_adapter.proj.b"));   // (h_vox,1)
    {   // residual SwiGLU block
        lm_ggml_tensor * bn = codec_op_rms_norm_ct(ctx, a, I.eps, W("lm.tslm_adapter.blk0.ln.w"));
        lm_ggml_tensor * g = lm_ggml_silu(ctx, lin(ctx, W("lm.tslm_adapter.blk0.gate.w"), bn, nullptr));
        lm_ggml_tensor * u = lin(ctx, W("lm.tslm_adapter.blk0.up.w"), bn, nullptr);
        a = lm_ggml_add(ctx, a, lin(ctx, W("lm.tslm_adapter.blk0.down.w"), lm_ggml_mul(ctx, g, u), nullptr));
    }
    // ---- FSQ: round(tanh(in_proj(a))*s)/s then out_proj ----
    lm_ggml_tensor * q = lm_ggml_tanh(ctx, lin(ctx, W("lm.fsq.in_proj.w"), a, W("lm.fsq.in_proj.b")));
    q = lm_ggml_scale(ctx, lm_ggml_round(ctx, lm_ggml_scale(ctx, q, (float) I.fsq_scale)), 1.0f / (float) I.fsq_scale);
    lm_ggml_tensor * lm_hidden = lin(ctx, W("lm.fsq.out_proj.w"), q, W("lm.fsq.out_proj.b"));   // (h_vox,1)

    // ---- RALM input = fusion_concat_proj([lm_hidden ; prev_feedback_lm]) ----
    lm_ggml_tensor * fus = lm_ggml_concat(ctx, lm_hidden, pfb_lm, 0);                                // (2*h_vox,1)
    lm_ggml_tensor * ralm_new = lin(ctx, W("lm.proj.fusion_concat.w"), fus, W("lm.proj.fusion_concat.b")); // (h_vox,1)

    // ---- RALM: one incremental step over the persistent KV cache ----
    lm_ggml_tensor * rh = ralm_new;   // 1 new token
    for (int32_t i = 0; i < I.n_ralm; ++i) {
        rh = bm_ralm_kv_step(ctx, rh, "lm.ralm.layers." + std::to_string(i), model,
                             I.n_heads, I.n_kv, I.head_dim, I.eps,
                             p->k_cache[i], p->v_cache[i], p->kv_pos);
    }
    lm_ggml_tensor * residual_hidden = codec_op_rms_norm_ct(ctx, rh, I.eps, W("lm.ralm.norm.w"));  // (h_vox,1)

    // ---- mu = [lm_to_dit(lm_hidden) ; res_to_dit(residual_hidden)] → (h_dit, n_mu) ----
    lm_ggml_tensor * mu1 = lin(ctx, W("lm.proj.lm_to_dit.w"), lm_hidden, W("lm.proj.lm_to_dit.b"));
    lm_ggml_tensor * mu2 = lin(ctx, W("lm.proj.res_to_dit.w"), residual_hidden, W("lm.proj.res_to_dit.b"));
    lm_ggml_tensor * mu = lm_ggml_concat(ctx, mu1, mu2, 1);   // (h_dit, 2)

    // ---- CFM Euler solver (unrolled, cfg_zero_star) ----
    lm_ggml_tensor * cond_h = lin(ctx, W("lm.locdit.cond_proj.w"), cond, W("lm.locdit.cond_proj.b"));  // (h_dit,P)
    lm_ggml_tensor * mu_zero = lm_ggml_scale(ctx, mu, 0.0f);
    auto time_mlp = [&](const char * pfx, lm_ggml_tensor * s) {
        lm_ggml_tensor * h = lm_ggml_silu(ctx, lin(ctx, W((std::string(pfx) + ".l1.w").c_str()), s, W((std::string(pfx) + ".l1.b").c_str())));
        return lin(ctx, W((std::string(pfx) + ".l2.w").c_str()), h, W((std::string(pfx) + ".l2.b").c_str()));
    };
    lm_ggml_tensor * dt_emb = time_mlp("lm.locdit.dtime_mlp", dtsin);
    const int64_t T = (int64_t) I.n_mu + 1 + 2 * P;
    lm_ggml_tensor * cos_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.cos"), I.head_dim, T, W("lm.rope.cos")->nb[1], 0));
    lm_ggml_tensor * sin_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.sin"), I.head_dim, T, W("lm.rope.sin")->nb[1], 0));

    lm_ggml_tensor * x = z;
    for (int32_t s = 0; s < p->n_real; ++s) {
        lm_ggml_tensor * x_h = lin(ctx, W("lm.locdit.in_proj.w"), x, W("lm.locdit.in_proj.b"));
        lm_ggml_tensor * tsin_s = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, tsin, I.h_dit, 1, tsin->nb[1], (size_t) s * tsin->nb[1]));
        lm_ggml_tensor * t_h = lm_ggml_add(ctx, time_mlp("lm.locdit.time_mlp", tsin_s), dt_emb);
        lm_ggml_tensor * pos = bm_locdit_core(ctx, model, x_h, cond_h, mu,      t_h, cos_t, sin_t,
                                           I.n_locdit, I.n_heads, I.n_kv, I.head_dim, I.eps, P, I.h_dit, I.n_mu);
        lm_ggml_tensor * neg = bm_locdit_core(ctx, model, x_h, cond_h, mu_zero, t_h, cos_t, sin_t,
                                           I.n_locdit, I.n_heads, I.n_kv, I.head_dim, I.eps, P, I.h_dit, I.n_mu);
        lm_ggml_tensor * dot = lm_ggml_sum(ctx, lm_ggml_mul(ctx, pos, neg));
        lm_ggml_tensor * nn  = lm_ggml_sum(ctx, lm_ggml_mul(ctx, neg, neg));
        lm_ggml_tensor * st  = lm_ggml_div(ctx, dot, lm_ggml_scale_bias(ctx, nn, 1.0f, 1e-8f));
        lm_ggml_tensor * neg_st = lm_ggml_mul(ctx, neg, st);
        lm_ggml_tensor * dphi = lm_ggml_add(ctx, neg_st, lm_ggml_scale(ctx, lm_ggml_sub(ctx, pos, neg_st), p->cfg_value));
        x = lm_ggml_sub(ctx, x, lm_ggml_scale(ctx, dphi, p->dt[s]));
    }
    lm_ggml_set_name(x, "bm.cfm.patch");
    lm_ggml_set_output(x);

    // ---- stop head ----
    lm_ggml_tensor * sp = lm_ggml_silu(ctx, lin(ctx, W("lm.stop.proj.w"), lm_hidden, W("lm.stop.proj.b")));
    lm_ggml_tensor * stop_logit = lin(ctx, W("lm.stop.head.w"), sp, nullptr);   // (2,1)
    lm_ggml_set_name(stop_logit, "bm.cfm.stop");
    lm_ggml_set_output(stop_logit);

    // ---- LocEnc(patch) → feedback ----
    lm_ggml_tensor * le = lin(ctx, W("lm.locenc.in_proj.w"), x, W("lm.locenc.in_proj.b"));   // (h_enc, P)
    lm_ggml_tensor * sptok = lm_ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, W("lm.locenc.special_token")), I.h_enc, 1);
    le = lm_ggml_concat(ctx, sptok, le, 1);                                                   // (h_enc, P+1)
    const int64_t Te = le->ne[1];
    lm_ggml_tensor * ecos = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.cos"), I.head_dim, Te, W("lm.rope.cos")->nb[1], 0));
    lm_ggml_tensor * esin = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.sin"), I.head_dim, Te, W("lm.rope.sin")->nb[1], 0));
    for (int32_t i = 0; i < I.n_locenc; ++i) {
        le = codec_bm_minicpm_block_ht(ctx, le, "lm.locenc.layers." + std::to_string(i), model,
                                       I.n_heads, I.n_kv, I.head_dim, I.eps, ecos, esin, /*causal=*/false);
    }
    le = codec_op_rms_norm_ct(ctx, le, I.eps, W("lm.locenc.norm.w"));
    lm_ggml_tensor * cls = lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, le, I.h_enc, 0));                 // (h_enc,)
    lm_ggml_tensor * fb_tslm = lin(ctx, W("lm.proj.enc_to_tslm.w"), cls, W("lm.proj.enc_to_tslm.b"));  // (h_barbet,)
    lm_ggml_set_name(fb_tslm, "bm.cfm.fb_tslm");
    lm_ggml_set_output(fb_tslm);
    lm_ggml_tensor * fb_lm = lin(ctx, W("lm.proj.enc_to_lm.w"), cls, W("lm.proj.enc_to_lm.b"));        // (h_vox,)
    lm_ggml_set_name(fb_lm, "bm.cfm.fb_lm");
    lm_ggml_set_output(fb_lm);

    *out = x;
    return true;
}

// ---------------------------------------------------------------------

bool init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr) return false;
    lm_gguf_context * gf = lm->codec->gguf;
    cfm_impl * I = new (std::nothrow) cfm_impl();
    if (!I) { lm->last_error = "oom"; return false; }
    I->h_barbet  = lm->info.hidden_dim;
    I->h_vox     = codec_read_i32_kv(gf, "codec.lm.h_vox", 2048);
    I->h_enc     = codec_read_i32_kv(gf, "codec.lm.h_enc", 1024);
    I->h_dit     = codec_read_i32_kv(gf, "codec.lm.h_dit", 1024);
    I->latent_dim = lm->info.latent_dim;
    I->patch_size = lm->info.patch_size;
    I->n_mu      = 2;
    I->n_locenc  = codec_read_i32_kv(gf, "codec.lm.n_locenc", 12);
    I->n_locdit  = codec_read_i32_kv(gf, "codec.lm.n_locdit", 12);
    I->n_ralm    = codec_read_i32_kv(gf, "codec.lm.n_ralm", 8);
    I->n_heads   = codec_read_i32_kv(gf, "codec.lm.n_heads", 16);
    I->n_kv      = codec_read_i32_kv(gf, "codec.lm.n_kv", 2);
    I->head_dim  = codec_read_i32_kv(gf, "codec.lm.head_dim", 128);
    I->fsq_scale = codec_read_i32_kv(gf, "codec.lm.fsq_scale", 9);
    I->eps       = codec_read_f32_kv(gf, "codec.lm.rms_eps", 1e-5f);
    lm->impl = I;
    return true;
}

void free_lm(codec_lm * lm) {
    if (lm && lm->impl) { delete static_cast<cfm_impl *>(lm->impl); lm->impl = nullptr; }
}

bool state_init(codec_lm_state * st) {
    cfm_impl * I = static_cast<cfm_impl *>(st->lm->impl);
    cfm_state * s = new (std::nothrow) cfm_state();
    if (!s) return false;
    s->prev_patch.assign((size_t) I->latent_dim * I->patch_size, 0.0f);
    s->prev_feedback_lm.assign((size_t) I->h_vox, 0.0f);
    s->feedback_tslm.assign((size_t) I->h_barbet, 0.0f);
    st->impl = s;
    return true;
}
void state_free(codec_lm_state * st) {
    if (st && st->impl) {
        cfm_state * s = static_cast<cfm_state *>(st->impl);
        if (s->buf_kv) lm_ggml_backend_buffer_free(s->buf_kv);
        if (s->ctx_kv) lm_ggml_free(s->ctx_kv);
        delete s;
        st->impl = nullptr;
    }
}
void state_reset(codec_lm_state * st) {
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    s->kv_pos = 0;   // discard cached RALM K/V (positions before 0 are never read)
    std::fill(s->prev_patch.begin(), s->prev_patch.end(), 0.0f);
    std::fill(s->prev_feedback_lm.begin(), s->prev_feedback_lm.end(), 0.0f);
    std::fill(s->feedback_tslm.begin(), s->feedback_tslm.end(), 0.0f);
}

// Lazily allocate the persistent RALM K/V cache in the codec backend.
bool alloc_kv(cfm_state * s, const cfm_impl * I, codec_model * codec) {
    if (s->ctx_kv != nullptr) return true;
    if (codec == nullptr || codec->backend == nullptr) return false;
    s->max_T = 4096;
    const size_t hdr = (size_t) I->n_ralm * 2 * lm_ggml_tensor_overhead() + lm_ggml_tensor_overhead() * 8;
    lm_ggml_init_params ip = { hdr, nullptr, true };
    s->ctx_kv = lm_ggml_init(ip);
    if (!s->ctx_kv) return false;
    s->k_cache.assign((size_t) I->n_ralm, nullptr);
    s->v_cache.assign((size_t) I->n_ralm, nullptr);
    for (int32_t l = 0; l < I->n_ralm; ++l) {
        s->k_cache[(size_t) l] = lm_ggml_new_tensor_3d(s->ctx_kv, LM_GGML_TYPE_F32, I->head_dim, I->n_kv, s->max_T);
        s->v_cache[(size_t) l] = lm_ggml_new_tensor_3d(s->ctx_kv, LM_GGML_TYPE_F32, I->head_dim, I->n_kv, s->max_T);
        if (!s->k_cache[(size_t) l] || !s->v_cache[(size_t) l]) return false;
    }
    s->buf_kv = lm_ggml_backend_alloc_ctx_tensors(s->ctx_kv, codec->backend);
    return s->buf_kv != nullptr;
}

void sinusoidal(double val, int32_t dim, float * out) {
    const int32_t half = dim / 2;
    const double step = std::log(10000.0) / (double) (half - 1);
    for (int32_t i = 0; i < half; ++i) {
        const double e = 1000.0 * val * std::exp((double) i * -step);
        out[i] = (float) std::sin(e);
        out[half + i] = (float) std::cos(e);
    }
}

enum codec_status step_generate(codec_lm_state * st, const float * h_in, float cfg_value,
                                int32_t n_timesteps, const float * noise, float * out_patch, int32_t * out_stop) {
    cfm_impl * I = static_cast<cfm_impl *>(st->lm->impl);
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    const int32_t P = I->patch_size, D = I->latent_dim;
    const size_t dp = (size_t) D * P;

    // CFM schedule (sway t_span, zero-init skip) — host side, like the scaffold.
    const int32_t n = n_timesteps;
    std::vector<double> tspan((size_t) n + 1);
    for (int32_t i = 0; i <= n; ++i) {
        const double ts = 1.0 - (double) i / (double) n;
        tspan[(size_t) i] = ts + 1.0 * (std::cos(M_PI / 2.0 * ts) - 1.0 + ts);
    }
    if (!alloc_kv(s, I, st->lm->codec)) { st->last_error = "RALM KV cache alloc failed"; return CODEC_STATUS_INTERNAL_ERROR; }
    if (s->kv_pos >= s->max_T) { st->last_error = "RALM KV cache full (max_T)"; return CODEC_STATUS_INVALID_STATE; }

    const int32_t zero_init = std::max(1, (int32_t) ((double) (n + 1) * 0.04));
    cfm_build b = {};
    b.imp = *I; b.kv_pos = s->kv_pos; b.cfg_value = cfg_value; b.model = st->lm->codec;
    for (int32_t l = 0; l < I->n_ralm; ++l) { b.k_cache[l] = s->k_cache[(size_t) l]; b.v_cache[l] = s->v_cache[(size_t) l]; }
    std::vector<double> t_real;
    double t = tspan[0], dt = tspan[0] - tspan[1];
    for (int32_t step = 1; step <= n; ++step) {
        if (step > zero_init) { if ((int32_t) t_real.size() < 64) b.dt[t_real.size()] = (float) dt; t_real.push_back(t); }
        t -= dt;
        if (step < n) dt = t - tspan[(size_t) step + 1];
    }
    b.n_real = (int32_t) t_real.size();
    if (b.n_real <= 0 || b.n_real > 64) { st->last_error = "bad n_timesteps"; return CODEC_STATUS_INVALID_ARG; }
    std::vector<float> tsin_all((size_t) I->h_dit * b.n_real), dtsin((size_t) I->h_dit);
    for (int32_t k = 0; k < b.n_real; ++k) sinusoidal(t_real[(size_t) k], I->h_dit, tsin_all.data() + (size_t) k * I->h_dit);
    sinusoidal(0.0, I->h_dit, dtsin.data());
    std::vector<float> zbuf;
    if (noise == nullptr) { zbuf.assign(dp, 0.0f); noise = zbuf.data(); }

    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_BLUEMAGPIE_CFM_STEP;
    key.n_frames = s->kv_pos;     // graph depends on the cached prefix length (KV view + mask offset)
    key.n_q = n_timesteps;
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_step, &b, sizeof(b), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    auto wr = [&](const char * nm, const float * d, size_t n_el) -> bool {
        lm_ggml_tensor * tt = G(nm); return tt && codec_runtime_write_tensor(tt, d, n_el * sizeof(float), &err);
    };
    bool ok = wr("bm.cfm.h_in", h_in, I->h_barbet) && wr("bm.cfm.pfb_lm", s->prev_feedback_lm.data(), I->h_vox)
        && wr("bm.cfm.cond", s->prev_patch.data(), dp) && wr("bm.cfm.z", noise, dp)
        && wr("bm.cfm.tsin", tsin_all.data(), tsin_all.size()) && wr("bm.cfm.dtsin", dtsin.data(), dtsin.size());
    if (!ok) { st->last_error = err.empty() ? "write failed" : err; return CODEC_STATUS_INTERNAL_ERROR; }

    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nth, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    // read outputs
    if (!codec_runtime_read_tensor(G("bm.cfm.patch"), out_patch, dp * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    float stop2[2] = {0, 0};
    codec_runtime_read_tensor(G("bm.cfm.stop"), stop2, sizeof(stop2), &err);
    if (out_stop) *out_stop = (stop2[1] > stop2[0]) ? 1 : 0;
    codec_runtime_read_tensor(G("bm.cfm.fb_tslm"), s->feedback_tslm.data(), (size_t) I->h_barbet * sizeof(float), &err);
    codec_runtime_read_tensor(G("bm.cfm.fb_lm"), s->prev_feedback_lm.data(), (size_t) I->h_vox * sizeof(float), &err);

    // advance state: the RALM K/V for this position was written into the cache
    // by the graph (side-effect roots), so just bump the cached length.
    s->kv_pos += 1;
    std::memcpy(s->prev_patch.data(), out_patch, dp * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}

enum codec_status step_feedback_embd(codec_lm_state * st, float * out_embd) {
    cfm_impl * I = static_cast<cfm_impl *>(st->lm->impl);
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    std::memcpy(out_embd, s->feedback_tslm.data(), (size_t) I->h_barbet * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}

}  // namespace

const codec_lm_kind_vtable codec_lm_vtable_continuous_latent_cfm = {
    /*.kind               =*/ CODEC_LM_KIND_CONTINUOUS_LATENT_CFM,
    /*.name               =*/ "continuous_latent_cfm",
    /*.init               =*/ init,
    /*.free               =*/ free_lm,
    /*.state_init         =*/ state_init,
    /*.state_free         =*/ state_free,
    /*.state_reset        =*/ state_reset,
    /*.step_begin         =*/ nullptr,
    /*.step_pending       =*/ nullptr,
    /*.step_logits        =*/ nullptr,
    /*.step_push_code     =*/ nullptr,
    /*.step_finish        =*/ nullptr,
    /*.audio_embd         =*/ nullptr,
    /*.compose_audio_embd =*/ nullptr,
    /*.compose_next_embd  =*/ nullptr,
    /*.speaker_encode     =*/ nullptr,
    /*.step_generate      =*/ step_generate,
    /*.step_feedback_embd =*/ step_feedback_embd,
};
