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

#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>
#include <vector>

namespace {

lm_ggml_tensor * bm_linear(lm_ggml_context * ctx, lm_ggml_tensor * w, lm_ggml_tensor * x, lm_ggml_tensor * bias) {
    lm_ggml_tensor * y = lm_ggml_mul_mat(ctx, w, x);          // w (in,out) · x (in,T) → (out,T)
    if (bias != nullptr) {
        y = lm_ggml_add(ctx, y, codec_graph_cast_f32(ctx, bias));  // (out,) broadcasts over T
    }
    return y;
}

// rotate_half over ne[0] for x ne=(d, A, T): cat(-x[d/2:], x[:d/2]).
lm_ggml_tensor * bm_rotate_half(lm_ggml_context * ctx, lm_ggml_tensor * x) {
    const int64_t d = x->ne[0];
    const int64_t h = d / 2;
    lm_ggml_tensor * x1 = lm_ggml_cont(ctx, lm_ggml_view_3d(ctx, x, h, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0));
    lm_ggml_tensor * x2 = lm_ggml_cont(ctx, lm_ggml_view_3d(ctx, x, h, x->ne[1], x->ne[2], x->nb[1], x->nb[2], h * x->nb[0]));
    return lm_ggml_concat(ctx, lm_ggml_neg(ctx, x2), x1, 0);
}

// HF-style RoPE with precomputed cos/sin. x ne=(d, n_heads, T); cos/sin ne=(d, T).
lm_ggml_tensor * bm_rope(lm_ggml_context * ctx, lm_ggml_tensor * x, lm_ggml_tensor * cos_dt, lm_ggml_tensor * sin_dt) {
    const int64_t d = x->ne[0];
    const int64_t T = x->ne[2];
    lm_ggml_tensor * cos_b = lm_ggml_reshape_3d(ctx, cos_dt, d, 1, T);   // broadcast over heads (ne[1])
    lm_ggml_tensor * sin_b = lm_ggml_reshape_3d(ctx, sin_dt, d, 1, T);
    lm_ggml_tensor * xr = bm_rotate_half(ctx, x);
    return lm_ggml_add(ctx, lm_ggml_mul(ctx, x, cos_b), lm_ggml_mul(ctx, xr, sin_b));
}

// LocDiT time embedding: SiLU MLP over a sinusoidal embedding.
lm_ggml_tensor * bm_time_mlp(lm_ggml_context * ctx, const codec_model * model, const char * pfx, lm_ggml_tensor * sin_emb) {
    auto W = [&](const std::string & s) { return codec_graph_weight(ctx, model, std::string(pfx) + s); };
    lm_ggml_tensor * h = bm_linear(ctx, W(".l1.w"), sin_emb, W(".l1.b"));
    h = lm_ggml_silu(ctx, h);
    return bm_linear(ctx, W(".l2.w"), h, W(".l2.b"));
}

}  // namespace

// Shared MiniCPM decoder block (use_mup=false → plain residual).  x_ht ne=(hidden, T).
// cos_dt/sin_dt ne=(head_dim, T) or NULL (no_rope).  GQA via mul_mat batch broadcast.
lm_ggml_tensor * codec_bm_minicpm_block_ht(
    lm_ggml_context * ctx, lm_ggml_tensor * x_ht, const std::string & prefix, const codec_model * model,
    int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    lm_ggml_tensor * cos_dt, lm_ggml_tensor * sin_dt, bool causal) {

    auto W = [&](const char * s) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, prefix + s); };
    const int64_t T = x_ht->ne[1];
    const int32_t q_dim = n_heads * head_dim;

    lm_ggml_tensor * h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln1.w"));
    lm_ggml_tensor * q = lm_ggml_mul_mat(ctx, W(".attn_q.w"), h);
    lm_ggml_tensor * k = lm_ggml_mul_mat(ctx, W(".attn_k.w"), h);
    lm_ggml_tensor * v = lm_ggml_mul_mat(ctx, W(".attn_v.w"), h);
    q = lm_ggml_reshape_3d(ctx, q, head_dim, n_heads, T);
    k = lm_ggml_reshape_3d(ctx, k, head_dim, n_kv, T);
    v = lm_ggml_reshape_3d(ctx, v, head_dim, n_kv, T);
    if (cos_dt != nullptr) {
        q = bm_rope(ctx, q, cos_dt, sin_dt);
        k = bm_rope(ctx, k, cos_dt, sin_dt);
    }
    lm_ggml_tensor * q_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q, 0, 2, 1, 3));   // (d, T, n_heads)
    lm_ggml_tensor * k_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k, 0, 2, 1, 3));   // (d, T, n_kv)
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_p, q_p);                     // (T_k, T_q, n_heads)
    scores = lm_ggml_scale(ctx, scores, 1.0f / std::sqrt((float) head_dim));
    if (causal) scores = lm_ggml_diag_mask_inf(ctx, scores, 0);
    scores = lm_ggml_soft_max(ctx, scores);
    lm_ggml_tensor * v_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v, 1, 2, 0, 3));   // (T, d, n_kv)
    lm_ggml_tensor * attn = lm_ggml_mul_mat(ctx, v_p, scores);                    // (d, T_q, n_heads)
    attn = lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn, 0, 2, 1, 3));            // (d, n_heads, T)
    attn = lm_ggml_reshape_2d(ctx, attn, q_dim, T);
    lm_ggml_tensor * o = lm_ggml_mul_mat(ctx, W(".attn_o.w"), attn);
    x_ht = lm_ggml_add(ctx, x_ht, o);

    h = codec_op_rms_norm_ct(ctx, x_ht, eps, W(".ln2.w"));
    lm_ggml_tensor * gate = lm_ggml_mul_mat(ctx, W(".gate.w"), h);
    lm_ggml_tensor * up = lm_ggml_mul_mat(ctx, W(".up.w"), h);
    lm_ggml_tensor * mlp = lm_ggml_mul(ctx, lm_ggml_silu(ctx, gate), up);
    lm_ggml_tensor * down = lm_ggml_mul_mat(ctx, W(".down.w"), mlp);
    return lm_ggml_add(ctx, x_ht, down);
}

// LocDiT estimator core: pre-projected x_h/cond_h (h_dit,P), mu_h (h_dit,n_mu),
// t_h (h_dit,1) → predicted velocity patch (latent_dim, P).  seq = [mu,t,cond,x],
// bidirectional MiniCPM stack, take the x-tail, out_proj.
lm_ggml_tensor * bm_locdit_core(
    lm_ggml_context * ctx, const codec_model * model,
    lm_ggml_tensor * x_h, lm_ggml_tensor * cond_h, lm_ggml_tensor * mu_h, lm_ggml_tensor * t_h,
    lm_ggml_tensor * cos_t, lm_ggml_tensor * sin_t,
    int32_t n_layers, int32_t n_heads, int32_t n_kv, int32_t head_dim, float eps,
    int32_t P, int32_t h_dit, int32_t n_mu) {

    lm_ggml_tensor * seq = lm_ggml_concat(ctx, mu_h, t_h, 1);
    seq = lm_ggml_concat(ctx, seq, cond_h, 1);
    seq = lm_ggml_concat(ctx, seq, x_h, 1);
    for (int32_t i = 0; i < n_layers; ++i) {
        seq = codec_bm_minicpm_block_ht(ctx, seq, "lm.locdit.layers." + std::to_string(i), model,
                                        n_heads, n_kv, head_dim, eps, cos_t, sin_t, /*causal=*/false);
    }
    seq = codec_op_rms_norm_ct(ctx, seq, eps, codec_graph_weight(ctx, model, "lm.locdit.norm.w"));
    const int64_t start = (int64_t) n_mu + 1 + P;
    lm_ggml_tensor * xt = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, seq, h_dit, P, seq->nb[1], start * seq->nb[1]));
    return bm_linear(ctx, codec_graph_weight(ctx, model, "lm.locdit.out_proj.w"), xt,
                     codec_graph_weight(ctx, model, "lm.locdit.out_proj.b"));
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

bool bm_build_cfm(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    bm_cfm_build * p = static_cast<bm_cfm_build *>(ud);
    auto W = [&](const char * s) { return codec_graph_weight(ctx, p->model, s); };
    const int32_t D = p->latent_dim;

    lm_ggml_tensor * z    = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, p->P);          lm_ggml_set_name(z, "bm.cfm.z");
    lm_ggml_tensor * cond = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, D, p->P);          lm_ggml_set_name(cond, "bm.cfm.cond");
    lm_ggml_tensor * mu   = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->h_dit, p->n_mu); lm_ggml_set_name(mu, "bm.cfm.mu");
    lm_ggml_tensor * tsin = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->h_dit, p->n_real); lm_ggml_set_name(tsin, "bm.cfm.tsin");
    lm_ggml_tensor * dtsin= lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->h_dit, 1);       lm_ggml_set_name(dtsin, "bm.cfm.dtsin");

    lm_ggml_tensor * cond_h  = bm_linear(ctx, W("lm.locdit.cond_proj.w"), cond, W("lm.locdit.cond_proj.b"));
    lm_ggml_tensor * mu_zero = lm_ggml_scale(ctx, mu, 0.0f);
    lm_ggml_tensor * dt_emb  = bm_time_mlp(ctx, p->model, "lm.locdit.dtime_mlp", dtsin);

    const int64_t T = (int64_t) p->n_mu + 1 + 2 * p->P;
    lm_ggml_tensor * cos_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.cos"), p->head_dim, T, W("lm.rope.cos")->nb[1], 0));
    lm_ggml_tensor * sin_t = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, W("lm.rope.sin"), p->head_dim, T, W("lm.rope.sin")->nb[1], 0));

    lm_ggml_tensor * x = z;
    for (int32_t s = 0; s < p->n_real; ++s) {
        lm_ggml_tensor * x_h   = bm_linear(ctx, W("lm.locdit.in_proj.w"), x, W("lm.locdit.in_proj.b"));
        lm_ggml_tensor * tsin_s = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, tsin, p->h_dit, 1, tsin->nb[1], (size_t) s * tsin->nb[1]));
        lm_ggml_tensor * t_h   = lm_ggml_add(ctx, bm_time_mlp(ctx, p->model, "lm.locdit.time_mlp", tsin_s), dt_emb);
        lm_ggml_tensor * pos = bm_locdit_core(ctx, p->model, x_h, cond_h, mu,      t_h, cos_t, sin_t,
                                           p->n_layers, p->n_heads, p->n_kv, p->head_dim, p->eps, p->P, p->h_dit, p->n_mu);
        lm_ggml_tensor * neg = bm_locdit_core(ctx, p->model, x_h, cond_h, mu_zero, t_h, cos_t, sin_t,
                                           p->n_layers, p->n_heads, p->n_kv, p->head_dim, p->eps, p->P, p->h_dit, p->n_mu);
        lm_ggml_tensor * dot = lm_ggml_sum(ctx, lm_ggml_mul(ctx, pos, neg));
        lm_ggml_tensor * nn  = lm_ggml_sum(ctx, lm_ggml_mul(ctx, neg, neg));
        lm_ggml_tensor * st  = lm_ggml_div(ctx, dot, lm_ggml_scale_bias(ctx, nn, 1.0f, 1e-8f));
        lm_ggml_tensor * neg_st = lm_ggml_mul(ctx, neg, st);
        lm_ggml_tensor * dphi   = lm_ggml_add(ctx, neg_st, lm_ggml_scale(ctx, lm_ggml_sub(ctx, pos, neg_st), p->cfg_value));
        x = lm_ggml_sub(ctx, x, lm_ggml_scale(ctx, dphi, p->dt[s]));
    }
    lm_ggml_set_name(x, "bm.cfm.out");
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
    lm_gguf_context * gf = ctx->model->gguf;
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

    codec_graph_eval_guard guard(ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_BLUEMAGPIE_CFM;
    key.n_frames = P;
    key.n_q = n_timesteps;
    if (!codec_graph_cache_get_or_build(ctx, key, bm_build_cfm, &b, sizeof(b), &entry, &err)) return 3;
    auto G = [&](const char * nm) { return codec_graph_get_tensor(ctx, entry, nm); };
    lm_ggml_tensor *tz=G("bm.cfm.z"), *tc=G("bm.cfm.cond"), *tm=G("bm.cfm.mu"),
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
