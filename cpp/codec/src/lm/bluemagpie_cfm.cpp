#include "lm_internal.h"

#include "../models/bluemagpie_blocks.h"
#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <cstring>
#include <new>
#include <random>
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
    int32_t min_len;   // stop guard: stop honoured only for patch index > min_len
    float   eps;
};

struct cfm_state {
    int32_t            kv_pos = 0;       // RALM positions cached so far (= T)
    int32_t            patch_index = 0;  // 0-based AR patch counter (for min_len guard)
    int32_t            min_len = -1;      // per-state override (-1 = use impl default)
    std::vector<float> prev_patch;      // [latent_dim*patch_size] cond
    std::vector<float> prev_feedback_lm;// [h_vox]
    std::vector<float> feedback_tslm;   // [h_barbet]  (for step_feedback_embd)

    // Prefill priming: after codec_lm_text_prefill, `primed` is set and the
    // last prefix position's (lm_hidden, residual_hidden) are cached here.  The
    // NEXT step_generate consumes these directly (no adapter/FSQ/RALM) — the
    // reference's iteration-0 uses the <|audio_start|> text position, whose
    // lm_hidden is tslm_adapter(h) WITHOUT FSQ, and runs no RALM step.
    bool               primed = false;
    std::vector<float> prefill_lm_hidden;       // [h_vox]
    std::vector<float> prefill_residual_hidden; // [h_vox]

    // Teacher-forcing (parity tests only): when `has_teacher_patch` is set, the
    // NEXT step_generate uses `teacher_patch` (a reference latent patch) as the
    // LocEnc feedback source AND as this step's cond, instead of the codec's own
    // generated patch — so codec replays the reference trajectory exactly and
    // every emitted patch can be compared against the reference at high corr
    // (free-running feedback would otherwise diverge chaotically after a few
    // steps).  The patch the graph *emits* is still codec's own.  Cleared after
    // consumption; the flag is set via codec_lm_set_teacher_patch.
    bool               has_teacher_patch = false;
    std::vector<float> teacher_patch;           // [latent_dim*patch_size]

    // CFM init-noise RNG for the noise==NULL path.  The reference samples
    // z ~ N(0,1) per step (torch.randn); zero-init instead yields degenerate
    // "mean" ODE trajectories - muffled audio whose stop head never crosses.
    // Seeded from the codec context seed at state_init/state_reset so a fresh
    // state replays the same noise sequence (deterministic per utterance).
    std::mt19937       rng;

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

bool alloc_kv(cfm_state * s, const cfm_impl * I, codec_model * codec);   // fwd (defined below)

lm_ggml_tensor * lin(lm_ggml_context * c, lm_ggml_tensor * w, lm_ggml_tensor * x, lm_ggml_tensor * b) {
    lm_ggml_tensor * y = lm_ggml_mul_mat(c, w, x);
    return b ? lm_ggml_add(c, y, codec_graph_cast_f32(c, b)) : y;
}

// Matmul-weight accessor: no dequant CPY for F16/BF16 (see
// codec_graph_weight_mat).  Use for every tensor that lands as the LHS of
// lm_ggml_mul_mat in this per-step graph.

// One RALM layer, incremental KV (1 new token).  Causal, no rope, no qk_norm.
// Attends over a fixed bucket cache[0..B) plus the 1 new token, where B is a
// 64-multiple >= kv_pos+1 so the graph shape is kv_pos-independent within a
// bucket.  The additive `mask` (B, 1) input (0 for valid slots <= kv_pos,
// -inf beyond) zeroes out slots the host hasn't filled yet.  The new K/V are
// NOT written into the cache in-graph — they are surfaced as named outputs
// (kk_out/vv_out) so the host can scatter them into the persistent cache after
// compute, keeping the graph fully static per bucket.
lm_ggml_tensor * bm_ralm_kv_step(lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix,
    const codec_model * model, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * k_cache_l, lm_ggml_tensor * v_cache_l, int32_t bucket, lm_ggml_tensor * mask,
    lm_ggml_tensor * row_idx, lm_ggml_tensor ** kset_out, lm_ggml_tensor ** vset_out) {

    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, prefix + s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, prefix + s); };
    const int32_t q_dim = n_heads * head_dim;

    const int32_t kv_dim = n_kv * head_dim;
    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln1.w"));
    lm_ggml_tensor * q, * kk, * vv;
    lm_ggml_tensor * w_qkv = codec_graph_weight_mat(ctx, model, prefix + ".attn_qkv.w");
    if (w_qkv != nullptr) {
        lm_ggml_tensor * qkv = lm_ggml_mul_mat(ctx, w_qkv, h);   // (q_dim+2*kv_dim, 1)
        q  = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, qkv, q_dim,  0)), head_dim, n_heads, 1);
        kk = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, qkv, kv_dim, (size_t) q_dim * qkv->nb[0])), head_dim, n_kv, 1);
        vv = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, qkv, kv_dim, (size_t) (q_dim + kv_dim) * qkv->nb[0])), head_dim, n_kv, 1);
    } else {
        q  = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_q.w"), h), head_dim, n_heads, 1);
        kk = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_k.w"), h), head_dim, n_kv, 1);
        vv = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_v.w"), h), head_dim, n_kv, 1);
    }
    // In-graph scatter of the new token's K/V into the persistent cache at row
    // `row_idx` (= kv_pos), replacing the Phase-3 host readback + tensor_set.
    // The cache (head_dim, n_kv, max_T) is contiguous, so a 2D view
    // (head_dim*n_kv, max_T) treats each token as one row; lm_ggml_set_rows writes
    // the single flattened K/V row at kv_pos.  INVARIANT: this write targets row
    // kv_pos, but attention above reads only cache[0..bucket) with the additive
    // mask valid on [0..kv_pos) and the *new* token supplied by the in-graph
    // concat (k_all/v_all) — so row kv_pos is NEVER read within this same graph.
    // The write is therefore a pure side-output; ordering vs. the read is
    // irrelevant.  Marked as an output by the caller so graph.cpp roots it.
    {
        const int64_t kw = (int64_t) head_dim * n_kv;
        lm_ggml_tensor * k2 = lm_ggml_view_2d(ctx, k_cache_l, kw, k_cache_l->ne[2], k_cache_l->nb[2], 0);
        lm_ggml_tensor * v2 = lm_ggml_view_2d(ctx, v_cache_l, kw, v_cache_l->ne[2], v_cache_l->nb[2], 0);
        lm_ggml_tensor * kk_row = lm_ggml_reshape_2d(ctx, kk, kw, 1);
        lm_ggml_tensor * vv_row = lm_ggml_reshape_2d(ctx, vv, kw, 1);
        *kset_out = lm_ggml_set_rows(ctx, k2, kk_row, row_idx);
        *vset_out = lm_ggml_set_rows(ctx, v2, vv_row, row_idx);
    }

    // Bucketed key/value: first `bucket` slots from the persistent cache (the
    // new token's slot kv_pos is inside the bucket and gets host-filled after
    // this step, but the additive mask keeps it -inf for the current query, so
    // attention over the cache view + the freshly computed kk/vv is exact).
    lm_ggml_tensor * k_old = lm_ggml_view_3d(ctx, k_cache_l, head_dim, n_kv, bucket, k_cache_l->nb[1], k_cache_l->nb[2], 0);
    lm_ggml_tensor * v_old = lm_ggml_view_3d(ctx, v_cache_l, head_dim, n_kv, bucket, v_cache_l->nb[1], v_cache_l->nb[2], 0);
    lm_ggml_tensor * k_all = lm_ggml_concat(ctx, k_old, kk, 2);   // (head_dim, n_kv, bucket+1)
    lm_ggml_tensor * v_all = lm_ggml_concat(ctx, v_old, vv, 2);

    lm_ggml_tensor * q_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q, 0, 2, 1, 3));       // (head_dim, 1, n_heads)
    lm_ggml_tensor * k_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k_all, 0, 2, 1, 3));   // (head_dim, bucket+1, n_kv)
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_p, q_p);                          // (bucket+1, 1, n_heads)
    // Fused scale + additive mask + softmax.  mask is (bucket+1, 1): valid
    // slots (<= kv_pos and the self slot) are 0, unfilled slots are -inf.
    scores = lm_ggml_soft_max_ext(ctx, scores, mask, 1.0f / std::sqrt((float) head_dim), 0.0f);
    lm_ggml_tensor * v_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_all, 1, 2, 0, 3));
    lm_ggml_tensor * attn = lm_ggml_cont(ctx, lm_ggml_permute(ctx, lm_ggml_mul_mat(ctx, v_p, scores), 0, 2, 1, 3));
    attn = lm_ggml_reshape_2d(ctx, attn, q_dim, 1);
    x_ht = lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, WM(".attn_o.w"), attn));

    h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln2.w"));
    lm_ggml_tensor * mlp;
    lm_ggml_tensor * w_gu = codec_graph_weight_mat(ctx, model, prefix + ".gate_up.w");
    if (w_gu != nullptr) {
        // Fused gate|up: gate is the first half of ne[0] → swiglu (non-swapped).
        lm_ggml_tensor * gu = lm_ggml_mul_mat(ctx, w_gu, h);   // (2*ffn, 1)
        mlp = lm_ggml_swiglu(ctx, gu);
    } else {
        lm_ggml_tensor * g = lm_ggml_mul_mat(ctx, WM(".gate.w"), h);
        lm_ggml_tensor * u = lm_ggml_mul_mat(ctx, WM(".up.w"), h);
        mlp = lm_ggml_swiglu_split(ctx, g, u);
    }
    return lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, WM(".down.w"), mlp));
}

// tslm_adapter(h) → (h_vox, T): RMSNorm + proj + one residual SwiGLU block.
// Shared by the per-step graph and the prefill graph.  x_in is (h_barbet, T).
lm_ggml_tensor * bm_tslm_adapter(lm_ggml_context * ctx, const codec_model * model, lm_ggml_tensor * x_in, float eps) {
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };
    lm_ggml_tensor * a = codec_op_rms_norm_ct(ctx, x_in, eps, W("lm.tslm_adapter.norm.w"));
    a = lin(ctx, WM("lm.tslm_adapter.proj.w"), a, W("lm.tslm_adapter.proj.b"));   // (h_vox,T)
    lm_ggml_tensor * bn = codec_op_rms_norm_ct(ctx, a, eps, W("lm.tslm_adapter.blk0.ln.w"));
    lm_ggml_tensor * g  = lm_ggml_mul_mat(ctx, WM("lm.tslm_adapter.blk0.gate.w"), bn);
    lm_ggml_tensor * u  = lm_ggml_mul_mat(ctx, WM("lm.tslm_adapter.blk0.up.w"), bn);
    lm_ggml_tensor * dn = lm_ggml_mul_mat(ctx, WM("lm.tslm_adapter.blk0.down.w"), lm_ggml_swiglu_split(ctx, g, u));
    return lm_ggml_add(ctx, a, dn);
}

// FSQ over an adapter output a (h_vox, T): out_proj(round(tanh(in_proj(a))*s)/s).
lm_ggml_tensor * bm_fsq(lm_ggml_context * ctx, const codec_model * model, lm_ggml_tensor * a, int32_t fsq_scale) {
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };
    lm_ggml_tensor * q = lm_ggml_tanh(ctx, lin(ctx, WM("lm.fsq.in_proj.w"), a, W("lm.fsq.in_proj.b")));
    q = lm_ggml_scale(ctx, lm_ggml_round(ctx, lm_ggml_scale(ctx, q, (float) fsq_scale)), 1.0f / (float) fsq_scale);
    return lin(ctx, WM("lm.fsq.out_proj.w"), q, W("lm.fsq.out_proj.b"));   // (h_vox,T)
}

// ---------------------------------------------------------------------
// RALM full-prefix prefill graph.
//
// The reference runs the RALM causally over the WHOLE prompt before the AR
// loop, seeding a static KV cache.  For every prefix position (all treated as
// TEXT positions here — zero-shot / "null" speaker):
//   lm_hidden_p = tslm_adapter(h_p)                       (NO FSQ on text)
//   ralm_in_p   = fusion_concat_proj(concat(lm_hidden_p, 0))   (feat_embed_lm=0)
// then RALM(causal, no rope) over the T inputs.  We take the LAST position's
// output (post RALM final norm) as prefill_residual_hidden, and the LAST
// position's lm_hidden as prefill_lm_hidden, and scatter K/V for ALL T positions
// into the persistent cache slots [0..T).
//
// This graph runs ONCE per utterance; it is bucketed on T so the cache reuses
// across utterances of the same padded length.
struct cfm_prefill_build {
    cfm_impl  imp;
    int32_t   n_pos;        // real prefix length T
    lm_ggml_tensor * k_cache[32];
    lm_ggml_tensor * v_cache[32];
    const codec_model * model;
};

// One RALM prefill layer over T tokens: causal self-attention (no rope), and
// scatter this layer's per-token K/V into the persistent cache rows [0..T).
// x_ht (h_vox, T) → (h_vox, T).  kset/vset are surfaced as outputs so graph.cpp
// roots the scatter writes.
lm_ggml_tensor * bm_ralm_prefill_layer(lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix,
    const codec_model * model, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * k_cache_l, lm_ggml_tensor * v_cache_l, lm_ggml_tensor * row_idx, int32_t T,
    lm_ggml_tensor ** kset_out, lm_ggml_tensor ** vset_out) {

    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, prefix + s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, prefix + s); };
    const int32_t q_dim  = n_heads * head_dim;
    const int32_t kv_dim = n_kv * head_dim;

    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln1.w"));   // (h_vox, T)
    lm_ggml_tensor * q, * kk, * vv;
    lm_ggml_tensor * w_qkv = codec_graph_weight_mat(ctx, model, prefix + ".attn_qkv.w");
    if (w_qkv != nullptr) {
        lm_ggml_tensor * qkv = lm_ggml_mul_mat(ctx, w_qkv, h);   // (q_dim+2*kv_dim, T)
        lm_ggml_tensor * qv = lm_ggml_view_2d(ctx, qkv, q_dim,  T, qkv->nb[1], 0);
        lm_ggml_tensor * kv = lm_ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], (size_t) q_dim * qkv->nb[0]);
        lm_ggml_tensor * vvw= lm_ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]);
        q  = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, qv),  head_dim, n_heads, T);
        kk = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, kv),  head_dim, n_kv,    T);
        vv = lm_ggml_reshape_3d(ctx, lm_ggml_cont(ctx, vvw), head_dim, n_kv,    T);
    } else {
        q  = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_q.w"), h), head_dim, n_heads, T);
        kk = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_k.w"), h), head_dim, n_kv,    T);
        vv = lm_ggml_reshape_3d(ctx, lm_ggml_mul_mat(ctx, WM(".attn_v.w"), h), head_dim, n_kv,    T);
    }

    // Scatter K/V for all T tokens into the persistent cache rows [0..T).  The
    // cache (head_dim, n_kv, max_T) is contiguous; a 2D view (head_dim*n_kv,
    // max_T) treats each token as one row; lm_ggml_set_rows writes rows row_idx.
    {
        const int64_t kw = (int64_t) head_dim * n_kv;
        lm_ggml_tensor * k2 = lm_ggml_view_2d(ctx, k_cache_l, kw, k_cache_l->ne[2], k_cache_l->nb[2], 0);
        lm_ggml_tensor * v2 = lm_ggml_view_2d(ctx, v_cache_l, kw, v_cache_l->ne[2], v_cache_l->nb[2], 0);
        lm_ggml_tensor * kk_rows = lm_ggml_reshape_2d(ctx, kk, kw, T);
        lm_ggml_tensor * vv_rows = lm_ggml_reshape_2d(ctx, vv, kw, T);
        *kset_out = lm_ggml_set_rows(ctx, k2, kk_rows, row_idx);
        *vset_out = lm_ggml_set_rows(ctx, v2, vv_rows, row_idx);
    }

    // Causal self-attention over the T tokens (no rope).
    lm_ggml_tensor * q_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q,  0, 2, 1, 3));   // (d, T, n_heads)
    lm_ggml_tensor * k_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, kk, 0, 2, 1, 3));   // (d, T, n_kv)
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_p, q_p);                       // (T_k, T_q, n_heads)
    scores = lm_ggml_scale(ctx, scores, 1.0f / std::sqrt((float) head_dim));
    scores = lm_ggml_diag_mask_inf(ctx, scores, 0);
    scores = lm_ggml_soft_max(ctx, scores);
    lm_ggml_tensor * v_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, vv, 1, 2, 0, 3));   // (T, d, n_kv)
    lm_ggml_tensor * attn = lm_ggml_cont(ctx, lm_ggml_permute(ctx, lm_ggml_mul_mat(ctx, v_p, scores), 0, 2, 1, 3));
    attn = lm_ggml_reshape_2d(ctx, attn, q_dim, T);
    x_ht = lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, WM(".attn_o.w"), attn));

    h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln2.w"));
    lm_ggml_tensor * mlp;
    lm_ggml_tensor * w_gu = codec_graph_weight_mat(ctx, model, prefix + ".gate_up.w");
    if (w_gu != nullptr) {
        mlp = lm_ggml_swiglu(ctx, lm_ggml_mul_mat(ctx, w_gu, h));
    } else {
        lm_ggml_tensor * g = lm_ggml_mul_mat(ctx, WM(".gate.w"), h);
        lm_ggml_tensor * u = lm_ggml_mul_mat(ctx, WM(".up.w"), h);
        mlp = lm_ggml_swiglu_split(ctx, g, u);
    }
    return lm_ggml_add(ctx, x_ht, lm_ggml_mul_mat(ctx, WM(".down.w"), mlp));
}

bool build_prefill(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    cfm_prefill_build * p = static_cast<cfm_prefill_build *>(ud);
    const cfm_impl & I = p->imp;
    const codec_model * model = p->model;
    const int32_t T = p->n_pos;

    // Barbet prefix hiddens (h_barbet, T), position-major.
    lm_ggml_tensor * h_in = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_barbet, T);
    lm_ggml_set_name(h_in, "bm.pf.h_in");
    lm_ggml_set_input(h_in);

    // Row indices [0..T) for the K/V scatter (host-written I32 arange).
    lm_ggml_tensor * row_idx = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, T);
    lm_ggml_set_name(row_idx, "bm.pf.row_idx");
    lm_ggml_set_input(row_idx);

    // lm_hidden = tslm_adapter(h_in) (NO FSQ — every prefix position is text).
    lm_ggml_tensor * a = bm_tslm_adapter(ctx, model, h_in, I.eps);   // (h_vox, T)

    // ralm_in = fusion_concat_proj(concat(a, 0))  (feat_embed_lm half = 0 for text).
    lm_ggml_tensor * zeros = lm_ggml_scale(ctx, a, 0.0f);                        // (h_vox, T) zeros
    lm_ggml_tensor * fus   = lm_ggml_concat(ctx, a, zeros, 0);                   // (2*h_vox, T)
    lm_ggml_tensor * rh    = lin(ctx, codec_graph_weight_mat(ctx, model, "lm.proj.fusion_concat.w"),
                             fus, codec_graph_weight(ctx, model, "lm.proj.fusion_concat.b"));  // (h_vox, T)

    for (int32_t i = 0; i < I.n_ralm; ++i) {
        lm_ggml_tensor * kset_i = nullptr, * vset_i = nullptr;
        rh = bm_ralm_prefill_layer(ctx, rh, "lm.ralm.layers." + std::to_string(i), model,
                                   I.n_heads, I.n_kv, I.head_dim, I.eps,
                                   p->k_cache[i], p->v_cache[i], row_idx, T, &kset_i, &vset_i);
        lm_ggml_set_name(kset_i, ("bm.pf.kset." + std::to_string(i)).c_str());
        lm_ggml_set_output(kset_i);
        lm_ggml_set_name(vset_i, ("bm.pf.vset." + std::to_string(i)).c_str());
        lm_ggml_set_output(vset_i);
    }
    lm_ggml_tensor * rn = codec_op_rms_norm_ct(ctx, rh, I.eps, codec_graph_weight(ctx, model, "lm.ralm.norm.w"));

    // Last-position outputs.
    lm_ggml_tensor * res_last = lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, rn, I.h_vox, (size_t) (T - 1) * rn->nb[1]));
    lm_ggml_set_name(res_last, "bm.pf.res_last");
    lm_ggml_set_output(res_last);
    lm_ggml_tensor * lm_last = lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, a, I.h_vox, (size_t) (T - 1) * a->nb[1]));
    lm_ggml_set_name(lm_last, "bm.pf.lm_last");
    lm_ggml_set_output(lm_last);

    *out = res_last;
    return true;
}

enum codec_status text_prefill(codec_lm_state * st, const float * hiddens, int32_t n_pos, int32_t hidden_dim) {
    cfm_impl * I = static_cast<cfm_impl *>(st->lm->impl);
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    (void) hidden_dim;   // validated by the public wrapper == I->h_barbet

    if (!alloc_kv(s, I, st->lm->codec)) { st->last_error = "RALM KV cache alloc failed"; return CODEC_STATUS_INTERNAL_ERROR; }
    if (n_pos > s->max_T) { st->last_error = "prefix longer than RALM KV cache (max_T)"; return CODEC_STATUS_INVALID_ARG; }

    // Start from a clean cache/state: prefill defines the whole prefix.
    s->kv_pos = 0;
    s->patch_index = 0;

    cfm_prefill_build b = {};
    b.imp = *I; b.n_pos = n_pos; b.model = st->lm->codec;
    for (int32_t l = 0; l < I->n_ralm; ++l) { b.k_cache[l] = s->k_cache[(size_t) l]; b.v_cache[l] = s->v_cache[(size_t) l]; }

    codec_graph_eval_guard guard(st->ctx, /*persist=*/false);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_BLUEMAGPIE_CFM_PREFILL;
    key.n_frames = n_pos;   // graph shape depends on the (unbucketed) prefix length
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_prefill, &b, sizeof(b), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    if (!codec_runtime_write_tensor(G("bm.pf.h_in"), hiddens, (size_t) I->h_barbet * n_pos * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::vector<int32_t> row_idx((size_t) n_pos);
    for (int32_t i = 0; i < n_pos; ++i) row_idx[(size_t) i] = i;
    if (!codec_runtime_write_tensor(G("bm.pf.row_idx"), row_idx.data(), (size_t) n_pos * sizeof(int32_t), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nth, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    if (!codec_runtime_read_tensor(G("bm.pf.res_last"), s->prefill_residual_hidden.data(),
                                   (size_t) I->h_vox * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_read_tensor(G("bm.pf.lm_last"), s->prefill_lm_hidden.data(),
                                   (size_t) I->h_vox * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }

    // K/V for all T positions were scattered into the cache in-graph.
    s->kv_pos = n_pos;
    s->primed = true;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status set_min_len(codec_lm_state * st, int32_t min_len) {
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    s->min_len = min_len;   // <0 restores impl default
    return CODEC_STATUS_SUCCESS;
}

enum codec_status set_teacher_patch(codec_lm_state * st, const float * patch, int32_t n) {
    cfm_impl * I = static_cast<cfm_impl *>(st->lm->impl);
    cfm_state * s = static_cast<cfm_state *>(st->impl);
    const int32_t dp = I->latent_dim * I->patch_size;
    if (patch == nullptr) { s->has_teacher_patch = false; return CODEC_STATUS_SUCCESS; }
    if (n != dp) { st->last_error = "teacher patch length mismatch"; return CODEC_STATUS_INVALID_ARG; }
    s->teacher_patch.assign(patch, patch + dp);
    s->has_teacher_patch = true;
    return CODEC_STATUS_SUCCESS;
}

// Build the whole per-step graph.
//
// IMPORTANT: the graph shape depends ONLY on (bucket, n_real, cfg_value) — NOT
// on kv_pos.  Two steps in the same 64-bucket produce byte-identical
// build_user_data, so the runtime's consecutive-call fast path (see
// codec_graph_cache_get_or_build) can skip the rebuild + galloc re-plan.  The
// per-step variation (which cache slots are valid) is carried entirely by the
// runtime `bm.cfm.ralm_mask` input.
struct cfm_build {
    cfm_impl  imp;
    int32_t   bucket;       // RALM cache view length (round_up(kv_pos+1, 64))
    int32_t   n_real;       // CFM real (non-zero-init) steps
    float     cfg_value;
    int32_t   primed;       // 1 = primed first step: lm_hidden/residual_hidden are
                            // INPUTS (from prefill); skip adapter/FSQ/RALM entirely.
    int32_t   teacher;      // 1 = LocEnc runs on the `bm.cfm.teacher` input patch
                            // (teacher-forced trajectory) instead of the emitted x.
    float     dt[64];
    lm_ggml_tensor * k_cache[32];
    lm_ggml_tensor * v_cache[32];
    const codec_model * model;
};

bool build_step(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    cfm_build * p = static_cast<cfm_build *>(ud);
    const cfm_impl & I = p->imp;
    const codec_model * model = p->model;
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };
    const int32_t P = I.patch_size, D = I.latent_dim;
    // lin() wrapper that fetches the weight via the no-dequant matmul accessor.
    auto linW = [&](const char * wn, lm_ggml_tensor * x, const char * bn) {
        return lin(ctx, WM(wn), x, bn ? W(bn) : nullptr);
    };

    lm_ggml_tensor * cond   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, P);            lm_ggml_set_name(cond, "bm.cfm.cond");
    lm_ggml_tensor * z      = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, P);            lm_ggml_set_name(z, "bm.cfm.z");
    lm_ggml_tensor * tsin   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_dit, p->n_real); lm_ggml_set_name(tsin, "bm.cfm.tsin");
    lm_ggml_tensor * dtsin  = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_dit, 1);       lm_ggml_set_name(dtsin, "bm.cfm.dtsin");

    lm_ggml_tensor * lm_hidden;
    lm_ggml_tensor * residual_hidden;

    if (p->primed) {
        // Primed first step: lm_hidden / residual_hidden come straight from the
        // prefill (last prefix position — a text position, so lm_hidden is the
        // non-FSQ tslm_adapter output).  No adapter / FSQ / RALM step here; the
        // RALM step for the NEXT iteration runs in the next (non-primed) call.
        lm_hidden = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_vox, 1);
        lm_ggml_set_name(lm_hidden, "bm.cfm.lm_hidden_in");   lm_ggml_set_input(lm_hidden);
        residual_hidden = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_vox, 1);
        lm_ggml_set_name(residual_hidden, "bm.cfm.res_hidden_in"); lm_ggml_set_input(residual_hidden);
    } else {
        lm_ggml_tensor * h_in   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_barbet, 1);  lm_ggml_set_name(h_in, "bm.cfm.h_in");
        lm_ggml_tensor * pfb_lm = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.h_vox, 1);      lm_ggml_set_name(pfb_lm, "bm.cfm.pfb_lm");

        // ---- tslm_adapter(h_in) → h_vox ----
        lm_ggml_tensor * a = codec_op_rms_norm_ct(ctx, h_in, I.eps, W("lm.tslm_adapter.norm.w"));
        a = linW("lm.tslm_adapter.proj.w", a, "lm.tslm_adapter.proj.b");   // (h_vox,1)
        {   // residual SwiGLU block
            lm_ggml_tensor * bn = codec_op_rms_norm_ct(ctx, a, I.eps, W("lm.tslm_adapter.blk0.ln.w"));
            lm_ggml_tensor * g = linW("lm.tslm_adapter.blk0.gate.w", bn, nullptr);
            lm_ggml_tensor * u = linW("lm.tslm_adapter.blk0.up.w", bn, nullptr);
            a = lm_ggml_add(ctx, a, linW("lm.tslm_adapter.blk0.down.w", lm_ggml_swiglu_split(ctx, g, u), nullptr));
        }
        // ---- FSQ: round(tanh(in_proj(a))*s)/s then out_proj ----
        lm_ggml_tensor * q = lm_ggml_tanh(ctx, linW("lm.fsq.in_proj.w", a, "lm.fsq.in_proj.b"));
        q = lm_ggml_scale(ctx, lm_ggml_round(ctx, lm_ggml_scale(ctx, q, (float) I.fsq_scale)), 1.0f / (float) I.fsq_scale);
        lm_hidden = linW("lm.fsq.out_proj.w", q, "lm.fsq.out_proj.b");   // (h_vox,1)

        // ---- RALM input = fusion_concat_proj([lm_hidden ; prev_feedback_lm]) ----
        lm_ggml_tensor * fus = lm_ggml_concat(ctx, lm_hidden, pfb_lm, 0);                                // (2*h_vox,1)
        lm_ggml_tensor * ralm_new = linW("lm.proj.fusion_concat.w", fus, "lm.proj.fusion_concat.b"); // (h_vox,1)

        // ---- RALM: one incremental step over the persistent KV cache ----
        // Shared additive attention mask for all RALM layers.  scores has ne0 =
        // bucket+1 (bucket cache slots + the 1 concatenated new token), so the
        // mask is (bucket+1, 1).  Host fills it each step.
        lm_ggml_tensor * ralm_mask = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->bucket + 1, 1);
        lm_ggml_set_name(ralm_mask, "bm.cfm.ralm_mask");
        lm_ggml_set_input(ralm_mask);

        // Row index (= kv_pos) for the in-graph lm_ggml_set_rows K/V scatter.  1-element
        // I32 input, host-written per step; the graph stays static per bucket.
        lm_ggml_tensor * ralm_row = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, 1);
        lm_ggml_set_name(ralm_row, "bm.cfm.ralm_row");
        lm_ggml_set_input(ralm_row);

        lm_ggml_tensor * rh = ralm_new;   // 1 new token
        for (int32_t i = 0; i < I.n_ralm; ++i) {
            lm_ggml_tensor * kset_i = nullptr, * vset_i = nullptr;
            rh = bm_ralm_kv_step(ctx, rh, "lm.ralm.layers." + std::to_string(i), model,
                                 I.n_heads, I.n_kv, I.head_dim, I.eps,
                                 p->k_cache[i], p->v_cache[i], p->bucket, ralm_mask, ralm_row, &kset_i, &vset_i);
            // The in-graph set_rows writes are pure side-outputs; flag them so
            // graph.cpp roots them for execution (they scatter into the persistent
            // cache and are never read back within this graph).
            lm_ggml_set_name(kset_i, ("bm.cfm.ralm_kset." + std::to_string(i)).c_str());
            lm_ggml_set_output(kset_i);
            lm_ggml_set_name(vset_i, ("bm.cfm.ralm_vset." + std::to_string(i)).c_str());
            lm_ggml_set_output(vset_i);
        }
        residual_hidden = codec_op_rms_norm_ct(ctx, rh, I.eps, W("lm.ralm.norm.w"));  // (h_vox,1)
    }

    // ---- mu = [lm_to_dit(lm_hidden) ; res_to_dit(residual_hidden)] → (h_dit, n_mu) ----
    lm_ggml_tensor * mu1 = linW("lm.proj.lm_to_dit.w", lm_hidden, "lm.proj.lm_to_dit.b");
    lm_ggml_tensor * mu2 = linW("lm.proj.res_to_dit.w", residual_hidden, "lm.proj.res_to_dit.b");
    lm_ggml_tensor * mu = lm_ggml_concat(ctx, mu1, mu2, 1);   // (h_dit, 2)

    // ---- CFM Euler solver (unrolled, cfg_zero_star) ----
    lm_ggml_tensor * cond_h = linW("lm.locdit.cond_proj.w", cond, "lm.locdit.cond_proj.b");  // (h_dit,P)
    lm_ggml_tensor * mu_zero = lm_ggml_scale(ctx, mu, 0.0f);
    auto time_mlp = [&](const char * pfx, lm_ggml_tensor * s) {
        lm_ggml_tensor * h = lm_ggml_silu(ctx, lin(ctx, WM((std::string(pfx) + ".l1.w").c_str()), s, W((std::string(pfx) + ".l1.b").c_str())));
        return lin(ctx, WM((std::string(pfx) + ".l2.w").c_str()), h, W((std::string(pfx) + ".l2.b").c_str()));
    };
    lm_ggml_tensor * dt_emb = time_mlp("lm.locdit.dtime_mlp", dtsin);
    const int64_t T = (int64_t) I.n_mu + 1 + 2 * P;
    lm_ggml_tensor * cos_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.cos"), I.head_dim, T, W("lm.rope.cos")->nb[1], 0));
    lm_ggml_tensor * sin_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.sin"), I.head_dim, T, W("lm.rope.sin")->nb[1], 0));

    const bool cfg_one = (p->cfg_value == 1.0f);
    lm_ggml_tensor * x = z;
    for (int32_t s = 0; s < p->n_real; ++s) {
        lm_ggml_tensor * x_h = linW("lm.locdit.in_proj.w", x, "lm.locdit.in_proj.b");
        lm_ggml_tensor * tsin_s = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, tsin, I.h_dit, 1, tsin->nb[1], (size_t) s * tsin->nb[1]));
        lm_ggml_tensor * t_h = lm_ggml_add(ctx, time_mlp("lm.locdit.time_mlp", tsin_s), dt_emb);
        lm_ggml_tensor * dphi;
        if (cfg_one) {
            // cfg == 1: dphi = pos, skip the uncond branch entirely.
            dphi = bm_locdit_core(ctx, model, x_h, cond_h, mu, t_h, cos_t, sin_t,
                                  I.n_locdit, I.n_heads, I.n_kv, I.head_dim, I.eps, P, I.h_dit, I.n_mu);
        } else {
            lm_ggml_tensor *pos = nullptr, *neg = nullptr;
            bm_locdit_core_batched(ctx, model, x_h, cond_h, mu, mu_zero, t_h, cos_t, sin_t,
                                   I.n_locdit, I.n_heads, I.n_kv, I.head_dim, I.eps, P, I.h_dit, I.n_mu,
                                   &pos, &neg);
            lm_ggml_tensor * dot = lm_ggml_sum(ctx, lm_ggml_mul(ctx, pos, neg));
            lm_ggml_tensor * nn  = lm_ggml_sum(ctx, lm_ggml_mul(ctx, neg, neg));
            lm_ggml_tensor * st  = lm_ggml_div(ctx, dot, lm_ggml_scale_bias(ctx, nn, 1.0f, 1e-8f));
            lm_ggml_tensor * neg_st = lm_ggml_mul(ctx, neg, st);
            dphi = lm_ggml_add(ctx, neg_st, lm_ggml_scale(ctx, lm_ggml_sub(ctx, pos, neg_st), p->cfg_value));
        }
        x = lm_ggml_sub(ctx, x, lm_ggml_scale(ctx, dphi, p->dt[s]));
    }
    lm_ggml_set_name(x, "bm.cfm.patch");
    lm_ggml_set_output(x);

    // ---- stop head ----
    lm_ggml_tensor * sp = lm_ggml_silu(ctx, linW("lm.stop.proj.w", lm_hidden, "lm.stop.proj.b"));
    lm_ggml_tensor * stop_logit = linW("lm.stop.head.w", sp, nullptr);   // (2,1)
    lm_ggml_set_name(stop_logit, "bm.cfm.stop");
    lm_ggml_set_output(stop_logit);

    // ---- LocEnc(patch) → feedback ----
    // Teacher-forcing: LocEnc runs on the reference patch input so the feedback
    // (and hence the next step's RALM/backbone drive) tracks the reference
    // trajectory exactly.  The emitted patch `x` is unchanged.
    lm_ggml_tensor * le_src = x;
    if (p->teacher) {
        lm_ggml_tensor * tp = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, P);
        lm_ggml_set_name(tp, "bm.cfm.teacher"); lm_ggml_set_input(tp);
        le_src = tp;
    }
    lm_ggml_tensor * le = linW("lm.locenc.in_proj.w", le_src, "lm.locenc.in_proj.b");   // (h_enc, P)
    lm_ggml_tensor * sptok = lm_ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, W("lm.locenc.special_token")), I.h_enc, 1);
    le = lm_ggml_concat(ctx, sptok, le, 1);                                                   // (h_enc, P+1)
    const int64_t Te = le->ne[1];
    lm_ggml_tensor * ecos = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.cos"), I.head_dim, Te, W("lm.rope.cos")->nb[1], 0));
    lm_ggml_tensor * esin = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.sin"), I.head_dim, Te, W("lm.rope.sin")->nb[1], 0));
    lm_ggml_tensor * le_pos = lm_ggml_cast(ctx, lm_ggml_arange(ctx, 0.0f, (float) Te, 1.0f), LM_GGML_TYPE_I32);
    for (int32_t i = 0; i < I.n_locenc; ++i) {
        le = codec_bm_minicpm_block_ht(ctx, le, "lm.locenc.layers." + std::to_string(i), model,
                                       I.n_heads, I.n_kv, I.head_dim, I.eps, ecos, esin, /*causal=*/false, le_pos);
    }
    le = codec_op_rms_norm_ct(ctx, le, I.eps, W("lm.locenc.norm.w"));
    lm_ggml_tensor * cls = lm_ggml_cont(ctx, lm_ggml_view_1d(ctx, le, I.h_enc, 0));                 // (h_enc,)
    lm_ggml_tensor * fb_tslm = linW("lm.proj.enc_to_tslm.w", cls, "lm.proj.enc_to_tslm.b");  // (h_barbet,)
    lm_ggml_set_name(fb_tslm, "bm.cfm.fb_tslm");
    lm_ggml_set_output(fb_tslm);
    lm_ggml_tensor * fb_lm = linW("lm.proj.enc_to_lm.w", cls, "lm.proj.enc_to_lm.b");        // (h_vox,)
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
    I->min_len   = codec_read_i32_kv(gf, "codec.lm.min_len", 2);
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
    s->prefill_lm_hidden.assign((size_t) I->h_vox, 0.0f);
    s->prefill_residual_hidden.assign((size_t) I->h_vox, 0.0f);
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
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
    s->patch_index = 0;
    s->primed = false;
    s->has_teacher_patch = false;
    std::fill(s->prev_patch.begin(), s->prev_patch.end(), 0.0f);
    std::fill(s->prev_feedback_lm.begin(), s->prev_feedback_lm.end(), 0.0f);
    std::fill(s->feedback_tslm.begin(), s->feedback_tslm.end(), 0.0f);
    std::fill(s->prefill_lm_hidden.begin(), s->prefill_lm_hidden.end(), 0.0f);
    std::fill(s->prefill_residual_hidden.begin(), s->prefill_residual_hidden.end(), 0.0f);
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
    // min_len override is preserved across reset (host set it deliberately).
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
    if (s->buf_kv == nullptr) return false;
    // Zero the cache: bucketed attention views slots [kv_pos..bucket) whose
    // contents are masked to -inf, but uninitialised backend memory could be
    // NaN, and NaN survives the additive mask (NaN + -inf = NaN).  Zeroing
    // guarantees finite scores so masked slots softmax to exactly 0.
    lm_ggml_backend_buffer_clear(s->buf_kv, 0);
    return true;
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

    // Primed first step (after codec_lm_text_prefill): lm_hidden / residual_hidden
    // come from the prefill's last position; no adapter / FSQ / RALM step here.
    // h_in is accepted but UNUSED on this call (the reference's iteration-0 reuses
    // the <|audio_start|> prefill position; the RALM step for the next iteration
    // runs in the next, non-primed call).  kv_pos does NOT advance here.
    const bool primed = s->primed;

    const int32_t zero_init = std::max(1, (int32_t) ((double) (n + 1) * 0.04));
    // Bucket the RALM cache view length so the graph shape is kv_pos-independent
    // within a 64-step window: consecutive steps reuse the same cache entry.
    const int32_t kBucketQuantum = 64;
    const int32_t bucket = ((s->kv_pos + 1 + kBucketQuantum - 1) / kBucketQuantum) * kBucketQuantum;
    cfm_build b = {};
    b.imp = *I; b.bucket = bucket; b.cfg_value = cfg_value; b.model = st->lm->codec;
    b.primed = primed ? 1 : 0;
    b.teacher = s->has_teacher_patch ? 1 : 0;
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
    if (noise == nullptr) {
        // Sample z ~ N(0,1) like the reference (torch.randn) - zero init is NOT
        // a valid fallback, it collapses the CFM to its mean trajectory.
        zbuf.resize(dp);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (size_t i = 0; i < dp; ++i) zbuf[i] = nd(s->rng);
        noise = zbuf.data();
    }

    codec_graph_eval_guard guard(st->ctx, /*persist=*/true);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_BLUEMAGPIE_CFM_STEP;
    key.n_frames = bucket;        // graph shape depends only on the 64-bucketed KV view length
    key.n_q = n_timesteps;
    // hop bit0 = cfg==1 single-branch shape; bit1 = primed (lm/res hidden are
    // inputs, no adapter/FSQ/RALM subgraph); bit2 = teacher-forced LocEnc input.
    key.hop = ((cfg_value == 1.0f) ? 1 : 0) | (primed ? 2 : 0) | (b.teacher ? 4 : 0);
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_step, &b, sizeof(b), &entry, &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
    }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    auto wr = [&](const char * nm, const float * d, size_t n_el) -> bool {
        lm_ggml_tensor * tt = G(nm); return tt && codec_runtime_write_tensor(tt, d, n_el * sizeof(float), &err);
    };
    bool ok = wr("bm.cfm.cond", s->prev_patch.data(), dp) && wr("bm.cfm.z", noise, dp)
        && wr("bm.cfm.tsin", tsin_all.data(), tsin_all.size()) && wr("bm.cfm.dtsin", dtsin.data(), dtsin.size());
    if (b.teacher) {
        ok = ok && wr("bm.cfm.teacher", s->teacher_patch.data(), dp);
    }
    if (primed) {
        // Primed step inputs: cached lm_hidden / residual_hidden from prefill.
        ok = ok && wr("bm.cfm.lm_hidden_in", s->prefill_lm_hidden.data(), I->h_vox)
                && wr("bm.cfm.res_hidden_in", s->prefill_residual_hidden.data(), I->h_vox);
        if (!ok) { st->last_error = err.empty() ? "write failed" : err; return CODEC_STATUS_INTERNAL_ERROR; }
    } else {
        // RALM additive attention mask, shape (bucket+1, 1).  Layout of the
        // concatenated keys: slots [0..bucket) are the cache view, slot [bucket] is
        // the freshly computed new token.  Valid slots: old cache [0..kv_pos) and
        // the self slot [bucket]; unfilled cache slots [kv_pos..bucket) are -inf.
        std::vector<float> ralm_mask((size_t) bucket + 1, 0.0f);
        for (int32_t j = s->kv_pos; j < bucket; ++j) ralm_mask[(size_t) j] = -INFINITY;
        ok = ok && wr("bm.cfm.h_in", h_in, I->h_barbet) && wr("bm.cfm.pfb_lm", s->prev_feedback_lm.data(), I->h_vox)
                && wr("bm.cfm.ralm_mask", ralm_mask.data(), ralm_mask.size());
        if (!ok) { st->last_error = err.empty() ? "write failed" : err; return CODEC_STATUS_INTERNAL_ERROR; }
        // Row index for the in-graph set_rows K/V scatter (= current kv_pos).
        lm_ggml_tensor * tr = G("bm.cfm.ralm_row");
        if (!tr || !codec_runtime_write_tensor(tr, &s->kv_pos, sizeof(int32_t), &err)) {
            st->last_error = err.empty() ? "row write failed" : err; return CODEC_STATUS_INTERNAL_ERROR; }
    }

    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nth, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    // read outputs
    if (!codec_runtime_read_tensor(G("bm.cfm.patch"), out_patch, dp * sizeof(float), &err)) {
        st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    float stop2[2] = {0, 0};
    codec_runtime_read_tensor(G("bm.cfm.stop"), stop2, sizeof(stop2), &err);
    int32_t stop = (stop2[1] > stop2[0]) ? 1 : 0;
    // min_len guard: the reference does `if i > min_len and stop == 1: break`, so
    // the stop flag is IGNORED for patches 0..min_len (0-based patch index).
    const int32_t min_len = (s->min_len >= 0) ? s->min_len : I->min_len;
    if (s->patch_index <= min_len) stop = 0;
    if (out_stop) *out_stop = stop;
    codec_runtime_read_tensor(G("bm.cfm.fb_tslm"), s->feedback_tslm.data(), (size_t) I->h_barbet * sizeof(float), &err);
    codec_runtime_read_tensor(G("bm.cfm.fb_lm"), s->prev_feedback_lm.data(), (size_t) I->h_vox * sizeof(float), &err);

    // The new token's K/V were scattered into the persistent cache in-graph via
    // lm_ggml_set_rows (row = kv_pos), so no host readback + tensor_set is needed.

    // advance state.  On a primed step no RALM step ran, so kv_pos does NOT
    // advance (the RALM step for iteration 1 runs in the next, non-primed call).
    if (!primed) s->kv_pos += 1;
    s->primed = false;   // consumed the prime
    s->patch_index += 1;
    // Next step's cond = this step's patch.  Under teacher-forcing that is the
    // reference patch (so the trajectory stays aligned); else the codec's own.
    if (s->has_teacher_patch) {
        std::memcpy(s->prev_patch.data(), s->teacher_patch.data(), dp * sizeof(float));
        s->has_teacher_patch = false;   // consumed; caller re-arms per step
    } else {
        std::memcpy(s->prev_patch.data(), out_patch, dp * sizeof(float));
    }
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
    /*.text_prefill       =*/ text_prefill,
    /*.set_min_len        =*/ set_min_len,
    /*.set_teacher_patch  =*/ set_teacher_patch,
};
