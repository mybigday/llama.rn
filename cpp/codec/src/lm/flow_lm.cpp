#include "lm_internal.h"
#include "spm_unigram.h"

#include "../ops/lm_ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>
#include <gguf.h>

#include <cmath>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

// =====================================================================
// codec_lm kind: flow_lm  (Kyutai Pocket-TTS)
//
// SELF-CONTAINED continuous-latent AR model — no external llama.cpp backbone.
// The AR transformer, text LUT, LSD flow head (SimpleMLPAdaLN) and EOS head all
// live in the codec GGUF under `lm.*`.  Sequence layout:
//   [ text LUT embeds | (bos_before_voice) | voice rows | AR latent embeds ]
// all fed through ONE causal transformer with a shared incremental KV cache.
//
// Per AR frame (codec_lm_flow_step), inside ONE cached graph:
//   in = input_linear( bos_emb if BOS else prev_latent )       (ldim -> d_model)
//   h  = transformer_step(in, KV cache @ pos kv_pos)           (6 layers, NORMAL rope)
//   c  = out_norm(h_last)
//   eos_logit = out_eos(c)
//   latent = LSD_decode(flow_net(c, ·), noise)                 (num_steps unrolled)
//
// RoPE is INTERLEAVED (real=x[2i], imag=x[2i+1]) => ggml NORMAL rope, matching
// the pocket_mimi transformer (verified at parity 1.0).
//
// The transformer LayerNorms (norm1/norm2/out_norm) are standard affine
// LayerNorm (eps 1e-5).  The flow head uses eps-1e-6 LayerNorm (affine in_ln,
// affine-free final) and an UNBIASED (ddof=1) RMSNorm inside the timestep
// embedders — both replicated exactly here.
// =====================================================================

namespace {

// POD config copied into graph build_user_data (must be trivially copyable —
// the graph cache key hashes it byte-wise).  Keeps the non-copyable tokenizer
// out of the build structs.
struct flow_impl {
    int32_t d_model;       // 1024
    int32_t n_layers;      // 6
    int32_t n_heads;       // 16
    int32_t head_dim;      // 64
    int32_t ffn_dim;       // 4096
    int32_t ldim;          // 32
    int32_t flow_dim;      // 512
    int32_t flow_depth;    // 6
    int32_t n_txt_bins;    // 4000
    int32_t insert_bos_before_voice;
    int32_t lsd_decode_steps;
    int32_t frames_after_eos;
    float   max_period;    // 10000
    float   temperature;   // 0.7
    float   eos_threshold; // -4.0
    float   ln_eps;        // 1e-5 (transformer LN)
    float   flow_ln_eps;   // 1e-6 (flow head LN)
    float   flow_rms_eps;  // 1e-5 (time-embed RMSNorm)
};

// Per-model state owned by codec_lm (adds the non-copyable tokenizer).
struct flow_model {
    flow_impl  cfg;
    SpmUnigram spm;
    bool       has_spm = false;
};

struct flow_state {
    int32_t            kv_pos = 0;   // AR positions written to the KV cache so far
    int32_t            frame  = 0;   // 0-based AR frame counter
    bool               primed = false;
    std::mt19937       rng;

    // Persistent per-layer transformer K/V cache (incremental decode).
    lm_ggml_context *             ctx_kv = nullptr;
    lm_ggml_backend_buffer_t      buf_kv = nullptr;
    std::vector<lm_ggml_tensor *> k_cache;   // [n_layers] (head_dim, n_heads, max_T)
    std::vector<lm_ggml_tensor *> v_cache;
    int32_t                    max_T = 0;
};

bool alloc_kv(flow_state * s, const flow_impl * I, codec_model * codec);

lm_ggml_tensor * lin(lm_ggml_context * c, lm_ggml_tensor * w, lm_ggml_tensor * x, lm_ggml_tensor * b) {
    lm_ggml_tensor * y = lm_ggml_mul_mat(c, w, x);
    return b ? lm_ggml_add(c, y, codec_graph_cast_f32(c, b)) : y;
}

// modulate(x, shift, scale) = x*(1+scale) + shift.  x is (dim, N); shift/scale
// are (dim, 1) broadcast over N.
lm_ggml_tensor * modulate(lm_ggml_context * c, lm_ggml_tensor * x, lm_ggml_tensor * shift, lm_ggml_tensor * scale) {
    lm_ggml_tensor * one_plus = lm_ggml_scale_bias(c, scale, 1.0f, 1.0f);   // scale + 1
    return lm_ggml_add(c, lm_ggml_mul(c, x, one_plus), shift);
}

// Unbiased (ddof=1) RMSNorm from pocket_tts.modules.mlp._rms_norm:
//   var = eps + x.var(dim=-1, keepdim=True)   [unbiased over `dim` elements]
//   y   = x * alpha * rsqrt(var)
// x is (dim, N).  alpha is (dim,).
lm_ggml_tensor * rms_norm_unbiased(lm_ggml_context * c, lm_ggml_tensor * x, lm_ggml_tensor * alpha, float eps) {
    const int64_t D = x->ne[0];
    lm_ggml_tensor * mean = lm_ggml_scale(c, lm_ggml_sum_rows(c, x), 1.0f / (float) D);   // (1, N)
    lm_ggml_tensor * xc   = lm_ggml_sub(c, x, mean);
    lm_ggml_tensor * sq   = lm_ggml_mul(c, xc, xc);
    // unbiased variance: sum(sq)/(D-1)
    lm_ggml_tensor * var  = lm_ggml_scale(c, lm_ggml_sum_rows(c, sq), 1.0f / (float) (D - 1));  // (1, N)
    lm_ggml_tensor * denom = lm_ggml_sqrt(c, lm_ggml_scale_bias(c, var, 1.0f, eps));            // sqrt(var+eps), (1,N)
    lm_ggml_tensor * y    = lm_ggml_div(c, x, lm_ggml_repeat(c, denom, x));                     // x / sqrt(var+eps)
    return lm_ggml_mul(c, y, lm_ggml_repeat(c, lm_ggml_reshape_2d(c, codec_graph_cast_f32(c, alpha), D, 1), y));
}

// One transformer layer, incremental KV step (T_new new tokens).  Causal,
// interleaved NORMAL rope with position offset `pos0`, standard affine
// LayerNorm, GELU FFN, no biases.  x_ct is (d_model, T_new).  Attends over
// cache[0..bucket) + new tokens; scatters new K/V into cache rows [pos0..).
lm_ggml_tensor * flow_tf_layer(lm_ggml_context * ctx, lm_ggml_tensor * x_ct, const std::string & pfx,
    const codec_model * model, const flow_impl & I, lm_ggml_tensor * pos_new,
    lm_ggml_tensor * k_cache_l, lm_ggml_tensor * v_cache_l, int32_t bucket, lm_ggml_tensor * mask,
    lm_ggml_tensor * row_idx, lm_ggml_tensor ** kset_out, lm_ggml_tensor ** vset_out) {

    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, pfx + s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, pfx + s); };
    const int32_t H = I.n_heads, D = I.head_dim;
    const int64_t Tn = x_ct->ne[1];

    // ---- self-attention ----
    lm_ggml_tensor * h = codec_op_layer_norm_ct(ctx, x_ct, I.ln_eps, W(".inln.w"), W(".inln.b"));
    lm_ggml_tensor * q = lm_ggml_mul_mat(ctx, WM(".attn.q_proj.w"), h);   // (d_model, Tn)
    lm_ggml_tensor * k = lm_ggml_mul_mat(ctx, WM(".attn.k_proj.w"), h);
    lm_ggml_tensor * v = lm_ggml_mul_mat(ctx, WM(".attn.v_proj.w"), h);

    // (d_model,Tn) -> (head_dim, H, Tn), rope over head_dim (NORMAL/interleaved).
    lm_ggml_tensor * q3 = lm_ggml_reshape_3d(ctx, q, D, H, Tn);
    lm_ggml_tensor * k3 = lm_ggml_reshape_3d(ctx, k, D, H, Tn);
    lm_ggml_tensor * v3 = lm_ggml_reshape_3d(ctx, v, D, H, Tn);
    q3 = lm_ggml_rope_ext(ctx, q3, pos_new, nullptr, D, LM_GGML_ROPE_TYPE_NORMAL, 0,
                       I.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k3 = lm_ggml_rope_ext(ctx, k3, pos_new, nullptr, D, LM_GGML_ROPE_TYPE_NORMAL, 0,
                       I.max_period, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Scatter the new tokens' K/V into cache rows [pos0..pos0+Tn) via set_rows on
    // a flat (head_dim*H, max_T) view.
    {
        const int64_t kw = (int64_t) D * H;
        lm_ggml_tensor * k2 = lm_ggml_view_2d(ctx, k_cache_l, kw, k_cache_l->ne[2], k_cache_l->nb[2], 0);
        lm_ggml_tensor * v2 = lm_ggml_view_2d(ctx, v_cache_l, kw, v_cache_l->ne[2], v_cache_l->nb[2], 0);
        lm_ggml_tensor * kk_rows = lm_ggml_reshape_2d(ctx, lm_ggml_cont(ctx, k3), kw, Tn);
        lm_ggml_tensor * vv_rows = lm_ggml_reshape_2d(ctx, lm_ggml_cont(ctx, v3), kw, Tn);
        *kset_out = lm_ggml_set_rows(ctx, k2, kk_rows, row_idx);
        *vset_out = lm_ggml_set_rows(ctx, v2, vv_rows, row_idx);
    }

    // Keys/values: bucket cache slots + the Tn new tokens.
    lm_ggml_tensor * k_old = lm_ggml_view_3d(ctx, k_cache_l, D, H, bucket, k_cache_l->nb[1], k_cache_l->nb[2], 0);
    lm_ggml_tensor * v_old = lm_ggml_view_3d(ctx, v_cache_l, D, H, bucket, v_cache_l->nb[1], v_cache_l->nb[2], 0);
    lm_ggml_tensor * k_all = lm_ggml_concat(ctx, k_old, k3, 2);   // (D, H, bucket+Tn)
    lm_ggml_tensor * v_all = lm_ggml_concat(ctx, v_old, v3, 2);

    lm_ggml_tensor * q_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q3,    0, 2, 1, 3));   // (D, Tn, H)
    lm_ggml_tensor * k_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k_all, 0, 2, 1, 3));   // (D, bucket+Tn, H)
    lm_ggml_tensor * scores = lm_ggml_mul_mat(ctx, k_p, q_p);                          // (bucket+Tn, Tn, H)
    // Fused scale + additive mask (bucket+Tn, Tn) + softmax.
    scores = lm_ggml_soft_max_ext(ctx, scores, mask, 1.0f / std::sqrt((float) D), 0.0f);
    lm_ggml_tensor * v_p = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_all, 1, 2, 0, 3));    // (bucket+Tn, D, H)
    lm_ggml_tensor * attn = lm_ggml_cont(ctx, lm_ggml_permute(ctx, lm_ggml_mul_mat(ctx, v_p, scores), 0, 2, 1, 3));
    attn = lm_ggml_reshape_2d(ctx, attn, (int64_t) D * H, Tn);
    x_ct = lm_ggml_add(ctx, x_ct, lm_ggml_mul_mat(ctx, WM(".attn.o_proj.w"), attn));

    // ---- FFN (GELU-erf, no bias) ----
    lm_ggml_tensor * f = codec_op_layer_norm_ct(ctx, x_ct, I.ln_eps, W(".paln.w"), W(".paln.b"));
    f = lm_ggml_gelu(ctx, lm_ggml_mul_mat(ctx, WM(".mlp.fc1.w"), f));
    f = lm_ggml_mul_mat(ctx, WM(".mlp.fc2.w"), f);
    return lm_ggml_add(ctx, x_ct, f);
}

// TimestepEmbedder(scalar t) -> (flow_dim,): args = t*freqs; cat[cos,sin]; mlp;
// then unbiased RMSNorm.  `sval` is the scalar time; freqs is (flow_dim/2,).
lm_ggml_tensor * flow_time_embed(lm_ggml_context * ctx, const codec_model * model, const flow_impl & I,
    int32_t idx, float sval) {
    const std::string p = "lm.flow.time_embed." + std::to_string(idx);
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, p + s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, p + s); };
    lm_ggml_tensor * freqs = codec_graph_cast_f32(ctx, W(".freqs"));   // (half,) = freq_embed/2
    const int64_t half = freqs->ne[0];
    lm_ggml_tensor * args  = lm_ggml_scale(ctx, freqs, sval);             // t*freqs
    // frequency embedding = cat([cos(args), sin(args)]) -> (2*half,) = freq_embed_size
    lm_ggml_tensor * emb   = lm_ggml_concat(ctx, lm_ggml_cos(ctx, args), lm_ggml_sin(ctx, args), 0);
    emb = lm_ggml_reshape_2d(ctx, emb, 2 * half, 1);
    lm_ggml_tensor * h = lm_ggml_silu(ctx, lin(ctx, WM(".l1.w"), emb, W(".l1.b")));   // (flow_dim,1)
    h = lin(ctx, WM(".l2.w"), h, W(".l2.b"));
    return rms_norm_unbiased(ctx, h, W(".rms.alpha"), I.flow_rms_eps);   // (flow_dim,1)
}

// SimpleMLPAdaLN flow_net(c, s, t, x): c is (d_model,1) cond, x is (ldim,1)
// input, s/t are scalars.  Returns (ldim,1) flow direction.
lm_ggml_tensor * flow_net(lm_ggml_context * ctx, const codec_model * model, const flow_impl & I,
    lm_ggml_tensor * cond, float sval, float tval, lm_ggml_tensor * x) {
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };

    lm_ggml_tensor * xh = lin(ctx, WM("lm.flow.input_proj.w"), x, W("lm.flow.input_proj.b"));   // (flow_dim,1)
    lm_ggml_tensor * ts = flow_time_embed(ctx, model, I, 0, sval);
    lm_ggml_tensor * tt = flow_time_embed(ctx, model, I, 1, tval);
    lm_ggml_tensor * t_comb = lm_ggml_scale(ctx, lm_ggml_add(ctx, ts, tt), 0.5f);                     // (flow_dim,1)
    lm_ggml_tensor * ce = lin(ctx, WM("lm.flow.cond_embed.w"), cond, W("lm.flow.cond_embed.b"));// (flow_dim,1)
    lm_ggml_tensor * y  = lm_ggml_add(ctx, t_comb, ce);                                            // (flow_dim,1)
    lm_ggml_tensor * sy = lm_ggml_silu(ctx, y);   // SiLU applied inside adaLN Sequential

    for (int32_t b = 0; b < I.flow_depth; ++b) {
        const std::string rp = "lm.flow.res." + std::to_string(b);
        auto RW  = [&](const char * s) { return codec_graph_weight(ctx, model, rp + s); };
        auto RWM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, rp + s); };
        // adaLN(silu(y)) -> chunk(3): shift, scale, gate  (each flow_dim).
        lm_ggml_tensor * mod = lin(ctx, RWM(".adaln.w"), sy, RW(".adaln.b"));   // (3*flow_dim,1)
        lm_ggml_tensor * shift = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], 0));
        lm_ggml_tensor * scale = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], (size_t) I.flow_dim * mod->nb[0]));
        lm_ggml_tensor * gate  = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], (size_t) 2 * I.flow_dim * mod->nb[0]));
        lm_ggml_tensor * hn = codec_op_layer_norm_ct(ctx, xh, I.flow_ln_eps, RW(".in_ln.w"), RW(".in_ln.b"));
        hn = modulate(ctx, hn, shift, scale);
        hn = lin(ctx, RWM(".mlp.l2.w"), lm_ggml_silu(ctx, lin(ctx, RWM(".mlp.l1.w"), hn, RW(".mlp.l1.b"))), RW(".mlp.l2.b"));
        xh = lm_ggml_add(ctx, xh, lm_ggml_mul(ctx, gate, hn));
    }
    // FinalLayer: adaLN(silu(y)) -> chunk(2): shift, scale; norm_final (no
    // affine); modulate; linear.
    lm_ggml_tensor * fmod = lin(ctx, WM("lm.flow.final.adaln.w"), sy, W("lm.flow.final.adaln.b"));  // (2*flow_dim,1)
    lm_ggml_tensor * fshift = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, fmod, I.flow_dim, 1, fmod->nb[1], 0));
    lm_ggml_tensor * fscale = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, fmod, I.flow_dim, 1, fmod->nb[1], (size_t) I.flow_dim * fmod->nb[0]));
    lm_ggml_tensor * xf = lm_ggml_norm(ctx, xh, I.flow_ln_eps);   // affine-free LayerNorm
    xf = modulate(ctx, xf, fshift, fscale);
    return lin(ctx, WM("lm.flow.final.linear.w"), xf, W("lm.flow.final.linear.b"));   // (ldim,1)
}

// ---------------------------------------------------------------------
// Prefill graph: text LUT embeds (+ optional voice rows) -> transformer ->
// fill KV cache for positions [0..T).
// ---------------------------------------------------------------------
struct flow_prefill_build {
    flow_impl imp;
    int32_t   n_tok;
    int32_t   n_voice;
    int32_t   T;            // n_tok + (bos?1:0) + n_voice
    int32_t   use_bos;      // insert_bos_before_voice && n_voice>0
    lm_ggml_tensor * k_cache[32];
    lm_ggml_tensor * v_cache[32];
    const codec_model * model;
};

bool build_prefill(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    flow_prefill_build * p = static_cast<flow_prefill_build *>(ud);
    const flow_impl & I = p->imp;
    const codec_model * model = p->model;
    const int32_t T = p->T;

    // Text embeddings via get_rows on the LUT (n_tok tokens).
    lm_ggml_tensor * tok = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, p->n_tok);
    lm_ggml_set_name(tok, "flow.pf.tok"); lm_ggml_set_input(tok);
    lm_ggml_tensor * embed_w = codec_graph_cast_f32(ctx, codec_graph_weight(ctx, model, "lm.text.embed.w"));  // (d_model, n_bins+1)
    lm_ggml_tensor * text_emb = lm_ggml_get_rows(ctx, embed_w, tok);   // (d_model, n_tok)

    lm_ggml_tensor * seq = text_emb;
    if (p->use_bos) {
        lm_ggml_tensor * bos = lm_ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, codec_graph_weight(ctx, model, "lm.bos_before_voice")), I.d_model, 1);
        seq = lm_ggml_concat(ctx, seq, bos, 1);
    }
    if (p->n_voice > 0) {
        lm_ggml_tensor * voice = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.d_model, p->n_voice);
        lm_ggml_set_name(voice, "flow.pf.voice"); lm_ggml_set_input(voice);
        seq = lm_ggml_concat(ctx, seq, voice, 1);   // (d_model, T)
    }

    // Positions 0..T-1 and cache row indices 0..T-1.
    lm_ggml_tensor * pos = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, T);
    lm_ggml_set_name(pos, "flow.pf.pos"); lm_ggml_set_input(pos);
    lm_ggml_tensor * row_idx = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, T);
    lm_ggml_set_name(row_idx, "flow.pf.row"); lm_ggml_set_input(row_idx);
    // Causal additive mask (T, T): row_idx layout is [cache 0..T) then... here
    // bucket==T and the new tokens are the whole prefix, so keys = cache[0..T)
    // ++ new[0..T); we attend only over the new tokens with a causal mask, and
    // the cache half is fully masked (-inf).  See build below.
    lm_ggml_tensor * mask = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, 2 * T, T);
    lm_ggml_set_name(mask, "flow.pf.mask"); lm_ggml_set_input(mask);

    lm_ggml_tensor * x = seq;
    for (int32_t l = 0; l < I.n_layers; ++l) {
        lm_ggml_tensor * kset = nullptr, * vset = nullptr;
        x = flow_tf_layer(ctx, x, "lm.tf.l" + std::to_string(l), model, I, pos,
                          p->k_cache[l], p->v_cache[l], /*bucket=*/T, mask, row_idx, &kset, &vset);
        lm_ggml_set_name(kset, ("flow.pf.kset." + std::to_string(l)).c_str()); lm_ggml_set_output(kset);
        lm_ggml_set_name(vset, ("flow.pf.vset." + std::to_string(l)).c_str()); lm_ggml_set_output(vset);
    }
    // Prefill produces no consumed output beyond the K/V scatter; root a scalar.
    lm_ggml_tensor * sink = lm_ggml_sum(ctx, x);
    lm_ggml_set_name(sink, "flow.pf.sink"); lm_ggml_set_output(sink);
    *out = sink;
    return true;
}

// ---------------------------------------------------------------------
// Per-step graph: one AR frame.
// ---------------------------------------------------------------------
struct flow_step_build {
    flow_impl imp;
    int32_t   bucket;      // KV cache view length (round_up(kv_pos+1,64))
    int32_t   is_bos;      // 1 => input = input_linear(bos_emb)
    int32_t   n_steps;     // LSD steps (unrolled)
    lm_ggml_tensor * k_cache[32];
    lm_ggml_tensor * v_cache[32];
    const codec_model * model;
};

bool build_step(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    flow_step_build * p = static_cast<flow_step_build *>(ud);
    const flow_impl & I = p->imp;
    const codec_model * model = p->model;
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };

    // ---- input embedding: input_linear(seq) where seq = bos_emb (BOS) or latent.
    lm_ggml_tensor * seq;
    if (p->is_bos) {
        seq = lm_ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, W("lm.bos_emb")), I.ldim, 1);
    } else {
        seq = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.ldim, 1);
        lm_ggml_set_name(seq, "flow.st.latent_in"); lm_ggml_set_input(seq);
    }
    lm_ggml_tensor * in = lm_ggml_mul_mat(ctx, WM("lm.input_linear.w"), seq);   // (d_model,1)

    // ---- transformer step over KV cache (1 new token at pos = kv_pos) ----
    lm_ggml_tensor * pos = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, 1);
    lm_ggml_set_name(pos, "flow.st.pos"); lm_ggml_set_input(pos);
    lm_ggml_tensor * row_idx = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, 1);
    lm_ggml_set_name(row_idx, "flow.st.row"); lm_ggml_set_input(row_idx);
    lm_ggml_tensor * mask = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, p->bucket + 1, 1);
    lm_ggml_set_name(mask, "flow.st.mask"); lm_ggml_set_input(mask);

    lm_ggml_tensor * x = in;
    for (int32_t l = 0; l < I.n_layers; ++l) {
        lm_ggml_tensor * kset = nullptr, * vset = nullptr;
        x = flow_tf_layer(ctx, x, "lm.tf.l" + std::to_string(l), model, I, pos,
                          p->k_cache[l], p->v_cache[l], p->bucket, mask, row_idx, &kset, &vset);
        lm_ggml_set_name(kset, ("flow.st.kset." + std::to_string(l)).c_str()); lm_ggml_set_output(kset);
        lm_ggml_set_name(vset, ("flow.st.vset." + std::to_string(l)).c_str()); lm_ggml_set_output(vset);
    }

    // ---- out_norm + EOS head ----
    lm_ggml_tensor * c = codec_op_layer_norm_ct(ctx, x, I.ln_eps, W("lm.out_norm.w"), W("lm.out_norm.b"));  // (d_model,1)
    lm_ggml_tensor * eos = lin(ctx, WM("lm.out_eos.w"), c, W("lm.out_eos.b"));   // (1,1)
    lm_ggml_set_name(eos, "flow.st.eos"); lm_ggml_set_output(eos);

    // ---- LSD flow (unrolled Euler) ----
    //   current = noise;  for i in [0,n): s=i/n, t=(i+1)/n;
    //     current += flow_net(c, s, t, current) / n
    lm_ggml_tensor * noise = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.ldim, 1);
    lm_ggml_set_name(noise, "flow.st.noise"); lm_ggml_set_input(noise);
    lm_ggml_tensor * cur = noise;
    const int32_t n = p->n_steps;
    for (int32_t i = 0; i < n; ++i) {
        const float sval = (float) i / (float) n;
        const float tval = (float) (i + 1) / (float) n;
        lm_ggml_tensor * dir = flow_net(ctx, model, I, c, sval, tval, cur);
        cur = lm_ggml_add(ctx, cur, lm_ggml_scale(ctx, dir, 1.0f / (float) n));
    }
    lm_ggml_set_name(cur, "flow.st.latent"); lm_ggml_set_output(cur);
    *out = cur;
    return true;
}

// ---------------------------------------------------------------------
// speaker_proj graph: mu [ldim, T] -> [d_model, T] rows (F.linear, no bias).
// ---------------------------------------------------------------------
struct flow_spk_build { flow_impl imp; int32_t T; const codec_model * model; };

bool build_spkproj(lm_ggml_context * ctx, void * ud, lm_ggml_tensor ** out) {
    flow_spk_build * p = static_cast<flow_spk_build *>(ud);
    const flow_impl & I = p->imp;
    lm_ggml_tensor * mu = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, I.ldim, p->T);
    lm_ggml_set_name(mu, "flow.sp.mu"); lm_ggml_set_input(mu);
    lm_ggml_tensor * w = codec_graph_weight_mat(ctx, p->model, "lm.speaker_proj.w");   // (ldim, d_model)
    lm_ggml_tensor * rows = lm_ggml_mul_mat(ctx, w, mu);   // (d_model, T)
    lm_ggml_set_name(rows, "flow.sp.rows"); lm_ggml_set_output(rows);
    *out = rows;
    return true;
}

// =====================================================================
// vtable impl
// =====================================================================

bool init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr) return false;
    lm_gguf_context * gf = lm->codec->gguf;
    flow_model * M = new (std::nothrow) flow_model();
    if (!M) { lm->last_error = "oom"; return false; }
    flow_impl * I = &M->cfg;
    I->d_model    = codec_read_i32_kv(gf, "codec.lm.d_model", 1024);
    I->n_layers   = codec_read_i32_kv(gf, "codec.lm.n_layers", 6);
    I->n_heads    = codec_read_i32_kv(gf, "codec.lm.n_heads", 16);
    I->head_dim   = codec_read_i32_kv(gf, "codec.lm.head_dim", I->d_model / (I->n_heads > 0 ? I->n_heads : 1));
    I->ffn_dim    = codec_read_i32_kv(gf, "codec.lm.ffn_dim", 4 * I->d_model);
    I->ldim       = codec_read_i32_kv(gf, "codec.lm.ldim", 32);
    I->flow_dim   = codec_read_i32_kv(gf, "codec.lm.flow_dim", 512);
    I->flow_depth = codec_read_i32_kv(gf, "codec.lm.flow_depth", 6);
    I->n_txt_bins = codec_read_i32_kv(gf, "codec.lm.lut_n_bins", 4000);
    I->insert_bos_before_voice = codec_read_bool_kv(gf, "codec.lm.insert_bos_before_voice", false) ? 1 : 0;
    I->lsd_decode_steps = codec_read_i32_kv(gf, "codec.lm.lsd_decode_steps", 1);
    I->frames_after_eos = codec_read_i32_kv(gf, "codec.lm.frames_after_eos", -1);
    I->max_period   = codec_read_f32_kv(gf, "codec.lm.max_period", 10000.0f);
    I->temperature  = codec_read_f32_kv(gf, "codec.lm.temperature", 0.7f);
    I->eos_threshold= codec_read_f32_kv(gf, "codec.lm.eos_threshold", -4.0f);
    I->ln_eps       = 1e-5f;
    I->flow_ln_eps  = 1e-6f;
    I->flow_rms_eps = 1e-5f;

    // Load SentencePiece tokenizer from the base64 KV.
    const std::string b64 = codec_lm_read_string_kv(lm->codec, "codec.lm.tokenizer.spm_b64");
    if (!b64.empty()) {
        // base64 decode.
        static const auto dec = [](char c) -> int {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62; if (c == '/') return 63; return -1; };
        std::vector<uint8_t> raw; raw.reserve(b64.size() * 3 / 4);
        int val = 0, bits = -8;
        for (char c : b64) {
            if (c == '=') break;
            int d = dec(c); if (d < 0) continue;
            val = (val << 6) | d; bits += 6;
            if (bits >= 0) { raw.push_back((uint8_t) ((val >> bits) & 0xff)); bits -= 8; }
        }
        if (M->spm.load(raw.data(), raw.size())) M->has_spm = true;
    }

    lm->impl = M;
    return true;
}

void free_lm(codec_lm * lm) {
    if (lm && lm->impl) { delete static_cast<flow_model *>(lm->impl); lm->impl = nullptr; }
}

bool alloc_kv(flow_state * s, const flow_impl * I, codec_model * codec) {
    if (s->ctx_kv != nullptr) return true;
    if (codec == nullptr || codec->backend == nullptr) return false;
    s->max_T = 4096;
    const size_t hdr = (size_t) I->n_layers * 2 * lm_ggml_tensor_overhead() + lm_ggml_tensor_overhead() * 8;
    lm_ggml_init_params ip = { hdr, nullptr, true };
    s->ctx_kv = lm_ggml_init(ip);
    if (!s->ctx_kv) return false;
    s->k_cache.assign((size_t) I->n_layers, nullptr);
    s->v_cache.assign((size_t) I->n_layers, nullptr);
    for (int32_t l = 0; l < I->n_layers; ++l) {
        s->k_cache[(size_t) l] = lm_ggml_new_tensor_3d(s->ctx_kv, LM_GGML_TYPE_F32, I->head_dim, I->n_heads, s->max_T);
        s->v_cache[(size_t) l] = lm_ggml_new_tensor_3d(s->ctx_kv, LM_GGML_TYPE_F32, I->head_dim, I->n_heads, s->max_T);
        if (!s->k_cache[(size_t) l] || !s->v_cache[(size_t) l]) return false;
    }
    s->buf_kv = lm_ggml_backend_alloc_ctx_tensors(s->ctx_kv, codec->backend);
    if (s->buf_kv == nullptr) return false;
    lm_ggml_backend_buffer_clear(s->buf_kv, 0);
    return true;
}

bool state_init(codec_lm_state * st) {
    flow_state * s = new (std::nothrow) flow_state();
    if (!s) return false;
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
    st->impl = s;
    return true;
}
void state_free(codec_lm_state * st) {
    if (st && st->impl) {
        flow_state * s = static_cast<flow_state *>(st->impl);
        if (s->buf_kv) lm_ggml_backend_buffer_free(s->buf_kv);
        if (s->ctx_kv) lm_ggml_free(s->ctx_kv);
        delete s;
        st->impl = nullptr;
    }
}
void state_reset(codec_lm_state * st) {
    flow_state * s = static_cast<flow_state *>(st->impl);
    s->kv_pos = 0;
    s->frame  = 0;
    s->primed = false;
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
}

}  // namespace

// =====================================================================
// Public flow_lm entry points (declared in codec_lm.h).
// =====================================================================

// Cache the info struct in the impl so we can return a stable pointer.
static codec_lm_flow_info g_flow_info;   // single-model process assumption is fine here

const struct codec_lm_flow_info * codec_lm_flow_get_info(struct codec_lm * lm) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || lm->impl == nullptr) return nullptr;
    flow_model * M = static_cast<flow_model *>(lm->impl);
    flow_impl * I = &M->cfg;
    g_flow_info.d_model          = I->d_model;
    g_flow_info.ldim             = I->ldim;
    g_flow_info.n_txt_bins       = I->n_txt_bins;
    g_flow_info.insert_bos_before_voice = I->insert_bos_before_voice;
    g_flow_info.frames_after_eos = I->frames_after_eos;
    g_flow_info.temperature      = I->temperature;
    g_flow_info.eos_threshold    = I->eos_threshold;
    g_flow_info.lsd_decode_steps = I->lsd_decode_steps;
    g_flow_info.has_tokenizer    = M->has_spm ? 1 : 0;
    return &g_flow_info;
}

enum codec_status codec_lm_flow_tokenize(struct codec_lm * lm, const char * text,
        int32_t * out_ids, int32_t cap, int32_t * n_out) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || text == nullptr ||
        out_ids == nullptr || n_out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_model * M = static_cast<flow_model *>(lm->impl);
    if (!M->has_spm) { lm->last_error = "no SentencePiece tokenizer baked in"; return CODEC_STATUS_NOT_SUPPORTED; }
    std::vector<int32_t> ids = M->spm.encode(text);
    if ((int32_t) ids.size() > cap) { lm->last_error = "token buffer too small"; return CODEC_STATUS_INVALID_ARG; }
    std::memcpy(out_ids, ids.data(), ids.size() * sizeof(int32_t));
    *n_out = (int32_t) ids.size();
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_flow_denorm_latent(struct codec_lm * lm, const float * latent, float * out) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || latent == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_impl * I = &static_cast<flow_model *>(lm->impl)->cfg;
    lm_ggml_tensor * std_t  = lm_ggml_get_tensor(lm->codec->weights, "lm.emb_std");
    lm_ggml_tensor * mean_t = lm_ggml_get_tensor(lm->codec->weights, "lm.emb_mean");
    if (!std_t || !mean_t) { lm->last_error = "missing emb_std/emb_mean"; return CODEC_STATUS_INTERNAL_ERROR; }
    const float * es = (const float *) std_t->data;
    const float * em = (const float *) mean_t->data;
    for (int32_t d = 0; d < I->ldim; ++d) out[d] = latent[d] * es[d] + em[d];
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_flow_speaker_rows(struct codec_lm * lm, const float * mu,
        int32_t n_voice, float * out, int32_t out_cap_rows) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || mu == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_impl * I = &static_cast<flow_model *>(lm->impl)->cfg;
    if (lm_ggml_get_tensor(lm->codec->weights, "lm.speaker_proj.w") == nullptr) {
        lm->last_error = "model has no speaker_proj (no voice cloning)"; return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (n_voice <= 0 || out_cap_rows < n_voice) return CODEC_STATUS_INVALID_ARG;
    // Uses a codec_context borrowed from the model for a one-off graph compute.
    // We build a transient context (same pattern as codec_lm_state's ctx).
    codec_context * ctx = new (std::nothrow) codec_context();
    if (!ctx) return CODEC_STATUS_INTERNAL_ERROR;
    ctx->model = lm->codec; ctx->backend = lm->codec->backend; ctx->params = codec_context_default_params();
    std::string err;
    enum codec_status rc = CODEC_STATUS_SUCCESS;
    if (!codec_runtime_init(ctx, &err)) { delete ctx; lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    {
        flow_spk_build b = {}; b.imp = *I; b.T = n_voice; b.model = lm->codec;
        codec_graph_eval_guard guard(ctx, false);
        codec_graph_cache_entry * entry = nullptr;
        codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_SPKPROJ; key.n_frames = n_voice;
        if (!codec_graph_cache_get_or_build(ctx, key, build_spkproj, &b, sizeof(b), &entry, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        if (!codec_graph_prepare_io(ctx, entry, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        if (!codec_runtime_write_tensor(codec_graph_get_tensor(ctx, entry, "flow.sp.mu"), mu, (size_t) I->ldim * n_voice * sizeof(float), &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        {
            const int32_t nth = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
            if (!codec_graph_compute(ctx, entry, nth, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        }
        if (!codec_runtime_read_tensor(codec_graph_get_tensor(ctx, entry, "flow.sp.rows"), out, (size_t) n_voice * I->d_model * sizeof(float), &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
    }
done:
    if (rc != CODEC_STATUS_SUCCESS) lm->last_error = err;
    codec_runtime_free(ctx); delete ctx;
    return rc;
}

enum codec_status codec_lm_flow_prefill(struct codec_lm_state * st, const int32_t * token_ids,
        int32_t n_tok, const float * voice_rows, int32_t n_voice) {
    if (st == nullptr || st->lm == nullptr || st->lm->kind != CODEC_LM_KIND_FLOW_LM ||
        token_ids == nullptr || n_tok <= 0) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_impl * I = &static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_state * s = static_cast<flow_state *>(st->impl);
    if (voice_rows == nullptr) n_voice = 0;
    const int32_t use_bos = (I->insert_bos_before_voice && n_voice > 0) ? 1 : 0;
    const int32_t T = n_tok + use_bos + n_voice;

    if (!alloc_kv(s, I, st->lm->codec)) { st->last_error = "KV cache alloc failed"; return CODEC_STATUS_INTERNAL_ERROR; }
    if (T > s->max_T) { st->last_error = "prefix longer than KV cache (max_T)"; return CODEC_STATUS_INVALID_ARG; }
    s->kv_pos = 0; s->frame = 0;

    flow_prefill_build b = {}; b.imp = *I; b.n_tok = n_tok; b.n_voice = n_voice; b.T = T;
    b.use_bos = use_bos; b.model = st->lm->codec;
    for (int32_t l = 0; l < I->n_layers; ++l) { b.k_cache[l] = s->k_cache[(size_t) l]; b.v_cache[l] = s->v_cache[(size_t) l]; }

    codec_graph_eval_guard guard(st->ctx, false);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_PREFILL; key.n_frames = T; key.n_q = n_voice; key.hop = use_bos;
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_prefill, &b, sizeof(b), &entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    if (!codec_runtime_write_tensor(G("flow.pf.tok"), token_ids, (size_t) n_tok * sizeof(int32_t), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    if (n_voice > 0) {
        if (!codec_runtime_write_tensor(G("flow.pf.voice"), voice_rows, (size_t) n_voice * I->d_model * sizeof(float), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    }
    std::vector<int32_t> pos((size_t) T), row((size_t) T);
    for (int32_t i = 0; i < T; ++i) { pos[(size_t) i] = i; row[(size_t) i] = i; }
    if (!codec_runtime_write_tensor(G("flow.pf.pos"), pos.data(), (size_t) T * sizeof(int32_t), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    if (!codec_runtime_write_tensor(G("flow.pf.row"), row.data(), (size_t) T * sizeof(int32_t), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    // Causal mask (2T, T): keys = cache[0..T) ++ new[0..T).  The cache half is
    // stale (unfilled) so mask it fully; within the new half apply a causal
    // mask (query i attends to new-key j<=i).
    std::vector<float> mask((size_t) 2 * T * T, 0.0f);
    for (int32_t qi = 0; qi < T; ++qi) {
        float * row_m = mask.data() + (size_t) qi * 2 * T;
        for (int32_t k = 0; k < T; ++k) row_m[k] = -INFINITY;          // cache half: all masked
        for (int32_t k = 0; k < T; ++k) row_m[T + k] = (k <= qi) ? 0.0f : -INFINITY;  // new half: causal
    }
    if (!codec_runtime_write_tensor(G("flow.pf.mask"), mask.data(), mask.size() * sizeof(float), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nth, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    s->kv_pos = T; s->primed = true;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_flow_step(struct codec_lm_state * st, const float * noise,
        float * out_latent, float * out_eos_logit, int32_t * out_is_eos) {
    if (st == nullptr || st->lm == nullptr || st->lm->kind != CODEC_LM_KIND_FLOW_LM || out_latent == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_impl * I = &static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_state * s = static_cast<flow_state *>(st->impl);
    if (!alloc_kv(s, I, st->lm->codec)) { st->last_error = "KV cache alloc failed"; return CODEC_STATUS_INTERNAL_ERROR; }
    if (s->kv_pos >= s->max_T) { st->last_error = "KV cache full"; return CODEC_STATUS_INVALID_STATE; }

    const int32_t is_bos = (s->frame == 0) ? 1 : 0;   // first AR frame consumes BOS (NaN->bos_emb)
    const int32_t kBucket = 64;
    const int32_t bucket = ((s->kv_pos + 1 + kBucket - 1) / kBucket) * kBucket;
    const int32_t n_steps = I->lsd_decode_steps > 0 ? I->lsd_decode_steps : 1;

    flow_step_build b = {}; b.imp = *I; b.bucket = bucket; b.is_bos = is_bos; b.n_steps = n_steps; b.model = st->lm->codec;
    for (int32_t l = 0; l < I->n_layers; ++l) { b.k_cache[l] = s->k_cache[(size_t) l]; b.v_cache[l] = s->v_cache[(size_t) l]; }

    // Sample or use provided noise ~ N(0, temp) (std = sqrt(temp)).
    std::vector<float> zbuf;
    const float * z = noise;
    if (z == nullptr) {
        zbuf.resize((size_t) I->ldim);
        const float sd = std::sqrt(I->temperature);
        std::normal_distribution<float> nd(0.0f, sd);
        for (int32_t d = 0; d < I->ldim; ++d) zbuf[(size_t) d] = nd(s->rng);
        z = zbuf.data();
    }

    codec_graph_eval_guard guard(st->ctx, /*persist=*/true);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_STEP; key.n_frames = bucket; key.n_q = n_steps; key.hop = is_bos;
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_step, &b, sizeof(b), &entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    if (!codec_graph_prepare_io(st->ctx, entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    // Inputs.
    if (!is_bos) {
        if (!codec_runtime_write_tensor(G("flow.st.latent_in"), out_latent, (size_t) I->ldim * sizeof(float), &err)) {
            // out_latent must hold the PREVIOUS latent on entry for non-BOS frames.
            st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR;
        }
    }
    if (!codec_runtime_write_tensor(G("flow.st.noise"), z, (size_t) I->ldim * sizeof(float), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    if (!codec_runtime_write_tensor(G("flow.st.pos"), &s->kv_pos, sizeof(int32_t), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    if (!codec_runtime_write_tensor(G("flow.st.row"), &s->kv_pos, sizeof(int32_t), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    std::vector<float> mask((size_t) bucket + 1, 0.0f);
    for (int32_t j = s->kv_pos; j < bucket; ++j) mask[(size_t) j] = -INFINITY;   // unfilled cache slots
    if (!codec_runtime_write_tensor(G("flow.st.mask"), mask.data(), mask.size() * sizeof(float), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, nth, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }

    if (!codec_runtime_read_tensor(G("flow.st.latent"), out_latent, (size_t) I->ldim * sizeof(float), &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    float eos = 0.0f;
    codec_runtime_read_tensor(G("flow.st.eos"), &eos, sizeof(float), &err);
    if (out_eos_logit) *out_eos_logit = eos;
    if (out_is_eos)    *out_is_eos = (eos > I->eos_threshold) ? 1 : 0;

    s->kv_pos += 1;
    s->frame  += 1;
    s->primed = false;
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------

const codec_lm_kind_vtable codec_lm_vtable_flow_lm = {
    /*.kind               =*/ CODEC_LM_KIND_FLOW_LM,
    /*.name               =*/ "flow_lm",
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
    /*.step_generate      =*/ nullptr,
    /*.step_feedback_embd =*/ nullptr,
    /*.text_prefill       =*/ nullptr,
    /*.set_min_len        =*/ nullptr,
    /*.set_teacher_patch  =*/ nullptr,
};
