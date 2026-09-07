// Shared BlueMagpie / VoxCPM ggml building blocks (declared in
// bluemagpie_blocks.h), reused by the codec_lm continuous_latent_cfm kind:
//   - codec_bm_minicpm_block_ht : MiniCPM decoder block (RMSNorm + GQA +
//                                 baked-RoPE + SwiGLU, use_mup=false)
//   - bm_locdit_core            : LocDiT estimator core (one CFM velocity eval)
//
// Plus `codec_bluemagpie_cfm_eval` — a CFM-only evaluation entry used by the
// end-to-end integration test (tests/e2e/bluemagpie_e2e_smoke.py): given the
// per-step (mu, cond, z) the LM side produces, it runs the unrolled CFM Euler
// solver and returns the latent patch.  Each block was parity-verified against
// the PyTorch reference before being chained into the adaptor.

#include "bluemagpie_audiovae.h"

#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

ggml_tensor * bm_linear(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);          // w (in,out) · x (in,T) → (out,T)
    if (bias != nullptr) {
        y = ggml_add(ctx, y, codec_graph_cast_f32(ctx, bias));  // (out,) broadcasts over T
    }
    return y;
}

// rotate_half over ne[0] for x ne=(d, ...): cat(-x[d/2:], x[:d/2]).
// Works for 3D (d, A, T) and 4D (d, A, T, B).
ggml_tensor * bm_rotate_half(ggml_context * ctx, ggml_tensor * x) {
    const int64_t d = x->ne[0];
    const int64_t h = d / 2;
    ggml_tensor * x1 = ggml_cont(ctx, ggml_view_4d(ctx, x, h, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0));
    ggml_tensor * x2 = ggml_cont(ctx, ggml_view_4d(ctx, x, h, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], h * x->nb[0]));
    return ggml_concat(ctx, ggml_neg(ctx, x2), x1, 0);
}

// HF-style RoPE with precomputed cos/sin. x ne=(d, n_heads, T); cos/sin ne=(d, T).
ggml_tensor * bm_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * cos_dt, ggml_tensor * sin_dt) {
    const int64_t d = x->ne[0];
    const int64_t T = x->ne[2];
    ggml_tensor * cos_b = ggml_reshape_3d(ctx, cos_dt, d, 1, T);   // broadcast over heads (ne[1])
    ggml_tensor * sin_b = ggml_reshape_3d(ctx, sin_dt, d, 1, T);
    ggml_tensor * xr = bm_rotate_half(ctx, x);
    return ggml_add(ctx, ggml_mul(ctx, x, cos_b), ggml_mul(ctx, xr, sin_b));
}

// LocDiT time embedding: SiLU MLP over a sinusoidal embedding.
ggml_tensor * bm_time_mlp(ggml_context * ctx, const codec_model * model, const char * pfx, ggml_tensor * sin_emb) {
    auto W  = [&](const std::string & s) { return codec_graph_weight(ctx, model, std::string(pfx) + s); };
    auto WM = [&](const std::string & s) { return codec_graph_weight_mat(ctx, model, std::string(pfx) + s); };
    ggml_tensor * h = bm_linear(ctx, WM(".l1.w"), sin_emb, W(".l1.b"));
    h = ggml_silu(ctx, h);
    return bm_linear(ctx, WM(".l2.w"), h, W(".l2.b"));
}

}  // namespace

// Shared MiniCPM decoder block (use_mup=false → plain residual).
// x_htb ne=(hidden, T, B).  B branches processed jointly: every linear folds
// (hidden, T, B) → (hidden, T*B) so its mul_mat is B× wider (weights streamed
// once); attention keeps branches separate by carrying B in ne[3] so a branch
// never attends across the batch.  cos_dt/sin_dt ne=(head_dim, T) or NULL.
// GQA via mul_mat batch broadcast (n_kv→n_heads over ne[2], B over ne[3]).
ggml_tensor * codec_bm_minicpm_block_htb(
    ggml_context * ctx, ggml_tensor * x_htb, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    ggml_tensor * cos_dt, ggml_tensor * sin_dt, bool causal,
    ggml_tensor * rope_pos) {

    auto W  = [&](const char * s) -> ggml_tensor * { return codec_graph_weight(ctx, model, prefix + s); };
    auto WM = [&](const char * s) -> ggml_tensor * { return codec_graph_weight_mat(ctx, model, prefix + s); };
    const int64_t hidden = x_htb->ne[0];
    const int64_t T = x_htb->ne[1];
    const int64_t B = x_htb->ne[2];
    const int32_t q_dim = n_heads * head_dim;

    // fold batch into the token dim for every matmul; ne[0] (hidden) unchanged.
    auto fold = [&](ggml_tensor * t) {  // (hidden_any, T, B) → (hidden_any, T*B)
        return ggml_reshape_2d(ctx, t, t->ne[0], T * B);
    };

    ggml_tensor * h = codec_op_rms_norm_ct(ctx, fold(x_htb), eps, W(".ln1.w"));   // (hidden, T*B)
    const int64_t TB = T * B;
    const int32_t kv_dim = n_kv * head_dim;
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * w_qkv = codec_graph_weight_mat(ctx, model, prefix + ".attn_qkv.w");
    if (w_qkv != nullptr) {
        // Fused QKV: one (in, q_dim+2*kv_dim) matmul, split the output rows with
        // views (the row = ne[0] split is contiguous per column).
        ggml_tensor * qkv = ggml_mul_mat(ctx, w_qkv, h);   // (q_dim+2*kv_dim, T*B)
        ggml_tensor * qv = ggml_view_2d(ctx, qkv, q_dim,  TB, qkv->nb[1], 0);
        ggml_tensor * kv = ggml_view_2d(ctx, qkv, kv_dim, TB, qkv->nb[1], (size_t) q_dim * qkv->nb[0]);
        ggml_tensor * vv = ggml_view_2d(ctx, qkv, kv_dim, TB, qkv->nb[1], (size_t) (q_dim + kv_dim) * qkv->nb[0]);
        q = ggml_reshape_4d(ctx, ggml_cont(ctx, qv), head_dim, n_heads, T, B);
        k = ggml_reshape_4d(ctx, ggml_cont(ctx, kv), head_dim, n_kv, T, B);
        v = ggml_reshape_4d(ctx, ggml_cont(ctx, vv), head_dim, n_kv, T, B);
    } else {
        q = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, WM(".attn_q.w"), h), head_dim, n_heads, T, B);
        k = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, WM(".attn_k.w"), h), head_dim, n_kv, T, B);
        v = ggml_reshape_4d(ctx, ggml_mul_mat(ctx, WM(".attn_v.w"), h), head_dim, n_kv, T, B);
    }
    if (cos_dt != nullptr) {
        // Prefer native ggml_rope_ext (one node per q/k) over the baked cos/sin
        // rotate_half (~8 nodes).  The GGUF bakes the LongRoPE short_factor branch
        // as freq_factors + theta/attn_factor KV; NEOX mode + ext_factor=0 +
        // freq_scale=1 reproduces the table exactly.  Falls back to the baked
        // cos/sin path if short_factor is absent (older GGUFs).
        ggml_tensor * short_factor = codec_graph_weight_or_null(ctx, model, "lm.rope.short_factor");
        if (short_factor != nullptr) {
            const float theta       = codec_read_f32_kv(model->gguf, "codec.lm.rope_theta", 10000.0f);
            const float attn_factor = codec_read_f32_kv(model->gguf, "codec.lm.rope_attn_factor", 1.0f);
            // Prefer the hoisted shared position vector (built once per graph).
            // A length-T view avoids the per-block ARANGE + CAST nodes.
            ggml_tensor * t_pos = (rope_pos != nullptr && rope_pos->ne[0] >= T)
                ? ((rope_pos->ne[0] == T) ? rope_pos
                   : ggml_view_1d(ctx, rope_pos, T, 0))
                : ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float) T, 1.0f), GGML_TYPE_I32);
            auto rope_ext = [&](ggml_tensor * x) {
                // x (head_dim, n_head, T, B); positions length T broadcast over B.
                return ggml_rope_ext(ctx, x, t_pos, short_factor, head_dim,
                                     GGML_ROPE_TYPE_NEOX, /*n_ctx_orig=*/0, theta,
                                     /*freq_scale=*/1.0f, /*ext_factor=*/0.0f,
                                     attn_factor, /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
            };
            q = rope_ext(q);
            k = rope_ext(k);
        } else {
            // cos/sin (d,T) → (d,1,T,1) broadcasts over heads (ne[1]) and batch (ne[3])
            ggml_tensor * cos_b = ggml_reshape_4d(ctx, cos_dt, head_dim, 1, T, 1);
            ggml_tensor * sin_b = ggml_reshape_4d(ctx, sin_dt, head_dim, 1, T, 1);
            auto rope4 = [&](ggml_tensor * x) {
                ggml_tensor * xr = bm_rotate_half(ctx, x);
                return ggml_add(ctx, ggml_mul(ctx, x, cos_b), ggml_mul(ctx, xr, sin_b));
            };
            q = rope4(q);
            k = rope4(k);
        }
    }
    ggml_tensor * q_p = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // (d, T, n_heads, B)
    ggml_tensor * k_p = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));   // (d, T, n_kv, B)
    ggml_tensor * scores = ggml_mul_mat(ctx, k_p, q_p);                     // (T_k, T_q, n_heads, B)
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);
    if (causal) {
        // causal path: scale, mask, softmax (soft_max_ext has no built-in causal mask)
        scores = ggml_scale(ctx, scores, kq_scale);
        scores = ggml_diag_mask_inf(ctx, scores, 0);
        scores = ggml_soft_max(ctx, scores);
    } else {
        // non-causal (LocDiT/LocEnc): fuse the 1/sqrt(d) scale into soft_max_ext.
        // NOTE: ggml_flash_attn_ext was tried here (Vulkan) but did NOT win — the
        // sequences are tiny (T=11/5), so FA's fixed kernel + F16 KV casts cancel
        // the dispatch-count savings.  Manual QK^T/softmax/(attn·V) stays.
        scores = ggml_soft_max_ext(ctx, scores, nullptr, kq_scale, 0.0f);
    }
    ggml_tensor * v_p = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));   // (T, d, n_kv, B)
    ggml_tensor * attn = ggml_mul_mat(ctx, v_p, scores);                    // (d, T_q, n_heads, B)
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));            // (d, n_heads, T, B)
    attn = ggml_reshape_2d(ctx, attn, q_dim, T * B);
    ggml_tensor * o = ggml_mul_mat(ctx, WM(".attn_o.w"), attn);             // (hidden, T*B)
    ggml_tensor * x_flat = ggml_add(ctx, fold(x_htb), o);

    h = codec_op_rms_norm_ct(ctx, x_flat, eps, W(".ln2.w"));
    ggml_tensor * mlp;
    ggml_tensor * w_gu = codec_graph_weight_mat(ctx, model, prefix + ".gate_up.w");
    if (w_gu != nullptr) {
        // Fused gate|up: one (in, 2*ffn) matmul; SwiGLU over the split output.
        // Converter concats [gate, up] along ne[0], so gate is the FIRST half.
        // ggml_swiglu (non-swapped) silu's the first half → silu(gate) * up.
        ggml_tensor * gu = ggml_mul_mat(ctx, w_gu, h);   // (2*ffn, T*B)
        mlp = ggml_swiglu(ctx, gu);                      // silu(gate) * up
    } else {
        ggml_tensor * gate = ggml_mul_mat(ctx, WM(".gate.w"), h);
        ggml_tensor * up   = ggml_mul_mat(ctx, WM(".up.w"), h);
        mlp = ggml_swiglu_split(ctx, gate, up);          // silu(gate) * up
    }
    ggml_tensor * down = ggml_mul_mat(ctx, WM(".down.w"), mlp);
    ggml_tensor * out = ggml_add(ctx, x_flat, down);                       // (hidden, T*B)
    return ggml_reshape_3d(ctx, out, hidden, T, B);
}

// B=1 wrapper: x_ht ne=(hidden, T).
ggml_tensor * codec_bm_minicpm_block_ht(
    ggml_context * ctx, ggml_tensor * x_ht, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    ggml_tensor * cos_dt, ggml_tensor * sin_dt, bool causal,
    ggml_tensor * rope_pos) {
    ggml_tensor * x_htb = ggml_reshape_3d(ctx, x_ht, x_ht->ne[0], x_ht->ne[1], 1);
    ggml_tensor * y = codec_bm_minicpm_block_htb(ctx, x_htb, prefix, model,
        n_heads, n_kv, head_dim, eps, cos_dt, sin_dt, causal, rope_pos);
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
}

// LocDiT estimator core: pre-projected x_h/cond_h (h_dit,P), mu_h (h_dit,n_mu),
// t_h (h_dit,1) → predicted velocity patch (latent_dim, P).  seq = [mu,t,cond,x],
// bidirectional MiniCPM stack, take the x-tail, out_proj.
ggml_tensor * bm_locdit_core(
    ggml_context * ctx, const codec_model * model,
    ggml_tensor * x_h, ggml_tensor * cond_h, ggml_tensor * mu_h, ggml_tensor * t_h,
    ggml_tensor * cos_t, ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu) {

    ggml_tensor * seq = ggml_concat(ctx, mu_h, t_h, 1);
    seq = ggml_concat(ctx, seq, cond_h, 1);
    seq = ggml_concat(ctx, seq, x_h, 1);
    // Shared rope position vector for all layers (hoisted; kills per-block ARANGE).
    const int64_t T_seq = seq->ne[1];
    ggml_tensor * rope_pos = ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float) T_seq, 1.0f), GGML_TYPE_I32);
    for (int32_t i = 0; i < n_layers; ++i) {
        seq = codec_bm_minicpm_block_ht(ctx, seq, "lm.locdit.layers." + std::to_string(i), model,
                                        n_heads, n_kv, head_dim, eps, cos_t, sin_t, /*causal=*/false, rope_pos);
    }
    seq = codec_op_rms_norm_ct(ctx, seq, eps, codec_graph_weight(ctx, model, "lm.locdit.norm.w"));
    const int64_t start = (int64_t) n_mu + 1 + P;
    ggml_tensor * xt = ggml_cont(ctx, ggml_view_2d(ctx, seq, h_dit, P, seq->nb[1], start * seq->nb[1]));
    return bm_linear(ctx, codec_graph_weight_mat(ctx, model, "lm.locdit.out_proj.w"), xt,
                     codec_graph_weight(ctx, model, "lm.locdit.out_proj.b"));
}

// Batched LocDiT estimator: runs the two CFG branches (pos = mu, neg = mu_zero)
// jointly through the MiniCPM stack so every weight matmul is 2× wide and the
// locdit weights are streamed once per Euler step instead of twice.  x_h/cond_h
// (h_dit,P), t_h (h_dit,1) are shared; mu_h / mu_zero_h (h_dit,n_mu) differ.
// Returns two velocity patches (latent_dim, P) via out params pos_out/neg_out.
void bm_locdit_core_batched(
    ggml_context * ctx, const codec_model * model,
    ggml_tensor * x_h, ggml_tensor * cond_h, ggml_tensor * mu_h, ggml_tensor * mu_zero_h,
    ggml_tensor * t_h, ggml_tensor * cos_t, ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu,
    ggml_tensor ** pos_out, ggml_tensor ** neg_out) {

    const int64_t T = (int64_t) n_mu + 1 + P + P;   // [mu ; t ; cond ; x]
    auto build_seq = [&](ggml_tensor * mu) {
        ggml_tensor * s = ggml_concat(ctx, mu, t_h, 1);
        s = ggml_concat(ctx, s, cond_h, 1);
        return ggml_concat(ctx, s, x_h, 1);         // (h_dit, T)
    };
    ggml_tensor * seq_pos = build_seq(mu_h);
    ggml_tensor * seq_neg = build_seq(mu_zero_h);
    // stack the two branches along ne[2]: (h_dit, T, 2)
    ggml_tensor * seq = ggml_concat(ctx,
        ggml_reshape_3d(ctx, seq_pos, h_dit, T, 1),
        ggml_reshape_3d(ctx, seq_neg, h_dit, T, 1), 2);

    // Shared rope position vector for all layers (hoisted; kills per-block ARANGE).
    ggml_tensor * rope_pos = ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float) T, 1.0f), GGML_TYPE_I32);
    for (int32_t i = 0; i < n_layers; ++i) {
        seq = codec_bm_minicpm_block_htb(ctx, seq, "lm.locdit.layers." + std::to_string(i), model,
                                         n_heads, n_kv, head_dim, eps, cos_t, sin_t, /*causal=*/false, rope_pos);
    }
    // rms_norm folds batch into token dim (per-token op, batch-safe)
    ggml_tensor * seq_flat = ggml_reshape_2d(ctx, seq, h_dit, T * 2);
    seq_flat = codec_op_rms_norm_ct(ctx, seq_flat, eps, codec_graph_weight(ctx, model, "lm.locdit.norm.w"));
    seq = ggml_reshape_3d(ctx, seq_flat, h_dit, T, 2);

    const int64_t start = (int64_t) n_mu + 1 + P;   // x-tail start
    // x-tail per branch: (h_dit, P, 2)
    ggml_tensor * xt = ggml_cont(ctx, ggml_view_3d(ctx, seq, h_dit, P, 2,
                                                   seq->nb[1], seq->nb[2], start * seq->nb[1]));
    // out_proj stays batched (fold 2 into token dim), then split.
    ggml_tensor * xt_flat = ggml_reshape_2d(ctx, xt, h_dit, P * 2);
    ggml_tensor * vel = bm_linear(ctx, codec_graph_weight_mat(ctx, model, "lm.locdit.out_proj.w"), xt_flat,
                                  codec_graph_weight(ctx, model, "lm.locdit.out_proj.b"));  // (latent, P*2)
    const int64_t D = vel->ne[0];
    ggml_tensor * vel3 = ggml_reshape_3d(ctx, vel, D, P, 2);
    *pos_out = ggml_cont(ctx, ggml_view_2d(ctx, vel3, D, P, vel3->nb[1], 0));
    *neg_out = ggml_cont(ctx, ggml_view_2d(ctx, vel3, D, P, vel3->nb[1], vel3->nb[2]));
}

// ---------------------------------------------------------------------
// CFM-only evaluation (one AR step's diffusion), used by the e2e test.
// ---------------------------------------------------------------------

namespace {

struct bm_cfm_build {
    int32_t P, latent_dim, h_dit, n_mu, n_layers, n_heads, n_kv, head_dim;
    float   eps, cfg_value;
    int32_t n_real;
    float   dt[64];
    const codec_model * model;
};

bool bm_build_cfm(ggml_context * ctx, void * ud, ggml_tensor ** out) {
    bm_cfm_build * p = static_cast<bm_cfm_build *>(ud);
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, p->model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, p->model, s); };
    const int32_t D = p->latent_dim;

    ggml_tensor * z    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, p->P);          ggml_set_name(z, "bm.cfm.z");
    ggml_tensor * cond = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, p->P);          ggml_set_name(cond, "bm.cfm.cond");
    ggml_tensor * mu   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->h_dit, p->n_mu); ggml_set_name(mu, "bm.cfm.mu");
    ggml_tensor * tsin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->h_dit, p->n_real); ggml_set_name(tsin, "bm.cfm.tsin");
    ggml_tensor * dtsin= ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->h_dit, 1);       ggml_set_name(dtsin, "bm.cfm.dtsin");

    ggml_tensor * cond_h  = bm_linear(ctx, WM("lm.locdit.cond_proj.w"), cond, W("lm.locdit.cond_proj.b"));
    ggml_tensor * mu_zero = ggml_scale(ctx, mu, 0.0f);
    ggml_tensor * dt_emb  = bm_time_mlp(ctx, p->model, "lm.locdit.dtime_mlp", dtsin);

    const int64_t T = (int64_t) p->n_mu + 1 + 2 * p->P;
    ggml_tensor * cos_t = ggml_cont(ctx, ggml_view_2d(ctx, W("lm.rope.cos"), p->head_dim, T, W("lm.rope.cos")->nb[1], 0));
    ggml_tensor * sin_t = ggml_cont(ctx, ggml_view_2d(ctx, W("lm.rope.sin"), p->head_dim, T, W("lm.rope.sin")->nb[1], 0));

    const bool cfg_one = (p->cfg_value == 1.0f);
    ggml_tensor * x = z;
    for (int32_t s = 0; s < p->n_real; ++s) {
        ggml_tensor * x_h   = bm_linear(ctx, WM("lm.locdit.in_proj.w"), x, W("lm.locdit.in_proj.b"));
        ggml_tensor * tsin_s = ggml_cont(ctx, ggml_view_2d(ctx, tsin, p->h_dit, 1, tsin->nb[1], (size_t) s * tsin->nb[1]));
        ggml_tensor * t_h   = ggml_add(ctx, bm_time_mlp(ctx, p->model, "lm.locdit.time_mlp", tsin_s), dt_emb);
        ggml_tensor * dphi;
        if (cfg_one) {
            // cfg == 1: dphi = pos, skip the uncond branch entirely.
            dphi = bm_locdit_core(ctx, p->model, x_h, cond_h, mu, t_h, cos_t, sin_t,
                                  p->n_layers, p->n_heads, p->n_kv, p->head_dim, p->eps, p->P, p->h_dit, p->n_mu);
        } else {
            ggml_tensor *pos = nullptr, *neg = nullptr;
            bm_locdit_core_batched(ctx, p->model, x_h, cond_h, mu, mu_zero, t_h, cos_t, sin_t,
                                   p->n_layers, p->n_heads, p->n_kv, p->head_dim, p->eps, p->P, p->h_dit, p->n_mu,
                                   &pos, &neg);
            ggml_tensor * dot = ggml_sum(ctx, ggml_mul(ctx, pos, neg));
            ggml_tensor * nn  = ggml_sum(ctx, ggml_mul(ctx, neg, neg));
            ggml_tensor * st  = ggml_div(ctx, dot, ggml_scale_bias(ctx, nn, 1.0f, 1e-8f));
            ggml_tensor * neg_st = ggml_mul(ctx, neg, st);
            dphi = ggml_add(ctx, neg_st, ggml_scale(ctx, ggml_sub(ctx, pos, neg_st), p->cfg_value));
        }
        x = ggml_sub(ctx, x, ggml_scale(ctx, dphi, p->dt[s]));
    }
    ggml_set_name(x, "bm.cfm.out");
    *out = x;
    return true;
}

void bm_sinusoidal(double val, int32_t dim, float * out) {
    const int32_t half = dim / 2;
    const double step = std::log(10000.0) / (double) (half - 1);
    for (int32_t i = 0; i < half; ++i) {
        const double e = 1000.0 * val * std::exp((double) i * -step);
        out[i] = (float) std::sin(e);
        out[half + i] = (float) std::cos(e);
    }
}

}  // namespace

// Evaluate one AR step's CFM diffusion from the LM-side (mu, cond) + init noise z.
// Layouts match the e2e harness: z/cond are [P,D] frame-major (buffer[p*D+d]),
// mu is [n_mu*h_dit] flat; out is [P,D] frame-major (the latent patch).
extern "C" int codec_bluemagpie_cfm_eval(
    struct codec_context * ctx, const float * z, const float * cond, const float * mu,
    int32_t P, int32_t n_timesteps, float cfg_value, float * out, int32_t out_n) {
    if (ctx == nullptr || ctx->model == nullptr) return 1;
    gguf_context * gf = ctx->model->gguf;
    bm_cfm_build b = {};
    b.P = P;
    b.latent_dim = codec_read_i32_kv(gf, "codec.lm.latent_dim", 64);
    b.h_dit      = codec_read_i32_kv(gf, "codec.lm.h_dit", 1024);
    b.n_mu       = 2;
    b.n_layers   = codec_read_i32_kv(gf, "codec.lm.n_locdit", 12);
    b.n_heads    = codec_read_i32_kv(gf, "codec.lm.n_heads", 16);
    b.n_kv       = codec_read_i32_kv(gf, "codec.lm.n_kv", 2);
    b.head_dim   = codec_read_i32_kv(gf, "codec.lm.head_dim", 128);
    b.eps        = codec_read_f32_kv(gf, "codec.lm.rms_eps", 1e-5f);
    b.cfg_value  = cfg_value;
    b.model      = ctx->model;
    if (out_n < b.latent_dim * P) return 2;

    // sway t_span schedule + zero-init skip (host side).
    const int32_t n = n_timesteps;
    std::vector<double> tspan((size_t) n + 1);
    for (int32_t i = 0; i <= n; ++i) {
        const double ts = 1.0 - (double) i / (double) n;
        tspan[(size_t) i] = ts + 1.0 * (std::cos(M_PI / 2.0 * ts) - 1.0 + ts);
    }
    const int32_t zero_init = std::max(1, (int32_t) ((double) (n + 1) * 0.04));
    std::vector<double> t_real;
    double t = tspan[0], dt = tspan[0] - tspan[1];
    for (int32_t step = 1; step <= n; ++step) {
        if (step > zero_init) {
            if ((int32_t) t_real.size() < 64) b.dt[t_real.size()] = (float) dt;
            t_real.push_back(t);
        }
        t = t - dt;
        if (step < n) dt = t - tspan[(size_t) step + 1];
    }
    b.n_real = (int32_t) t_real.size();
    if (b.n_real <= 0 || b.n_real > 64) return 9;

    std::vector<float> tsin_all((size_t) b.h_dit * (size_t) b.n_real);
    for (int32_t s = 0; s < b.n_real; ++s) bm_sinusoidal(t_real[(size_t) s], b.h_dit, tsin_all.data() + (size_t) s * b.h_dit);
    std::vector<float> dtsin((size_t) b.h_dit);
    bm_sinusoidal(0.0, b.h_dit, dtsin.data());

    codec_graph_eval_guard guard(ctx, /*persist=*/true);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_BLUEMAGPIE_CFM;
    key.n_frames = P;
    key.n_q = n_timesteps;
    key.hop = (cfg_value == 1.0f) ? 1 : 0;  // cfg==1 builds a different (single-branch) graph shape
    if (!codec_graph_cache_get_or_build(ctx, key, bm_build_cfm, &b, sizeof(b), &entry, &err)) return 3;
    auto G = [&](const char * nm) { return codec_graph_get_tensor(ctx, entry, nm); };
    ggml_tensor *tz=G("bm.cfm.z"), *tc=G("bm.cfm.cond"), *tm=G("bm.cfm.mu"),
                *tt=G("bm.cfm.tsin"), *td=G("bm.cfm.dtsin"), *to=G("bm.cfm.out");
    if (!tz||!tc||!tm||!tt||!td||!to) return 4;
    if (!codec_graph_prepare_io(ctx, entry, &err)) return 5;
    const size_t dp = (size_t) b.latent_dim * (size_t) P * sizeof(float);
    if (!codec_runtime_write_tensor(tz, z, dp, &err)) return 6;
    if (!codec_runtime_write_tensor(tc, cond, dp, &err)) return 6;
    if (!codec_runtime_write_tensor(tm, mu, (size_t) b.h_dit * (size_t) b.n_mu * sizeof(float), &err)) return 6;
    if (!codec_runtime_write_tensor(tt, tsin_all.data(), tsin_all.size() * sizeof(float), &err)) return 6;
    if (!codec_runtime_write_tensor(td, dtsin.data(), dtsin.size() * sizeof(float), &err)) return 6;
    const int32_t nth = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, nth, &err)) return 7;
    if (!codec_runtime_read_tensor(to, out, dp, &err)) return 8;
    return 0;
}
