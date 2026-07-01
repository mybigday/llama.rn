#include "moss_audio.h"

#include "../ops/conv1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/lm_gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// MOSS-Audio-Tokenizer (Nano + full).  Pure-Transformer codec from
// OpenMOSS-Team — alternating PatchedPretransform reshapes + causal
// Transformer blocks (RoPE + LayerScale + GELU FFN) wrapped around a
// 16-level Residual LFQ quantizer.
//
// The runtime reads the per-block schema (patch sizes, dims, layer
// counts, RoPE base, sliding-window context duration) from GGUF metadata
// — the same C++ path therefore handles both the 22 M Nano and the 1.6 B
// full variant; only the converter and the GGUF differ.
// =====================================================================

static const char * codec_moss_name_pcm()       { return "moss.encode.pcm"; }
static const char * codec_moss_name_codes()     { return "moss.encode.codes"; }
static const char * codec_moss_name_dec_codes() { return "moss.decode.codes"; }
static const char * codec_moss_name_dec_audio() { return "moss.decode.audio"; }

namespace {

// PatchedPretransform.encode: (B, D, L) → (B, D*patch, L/patch) by reshaping
// the time axis into (D, patch, L_out) then permuting (D, patch) → (patch, D)
// so the output channels are (D × patch) interleaved.  In TC layout (ne[0]=t,
// ne[1]=c) this becomes:
//     [t, c]   reshape (patch, t/patch, c)
//             permute → (t/patch, patch, c)   ← still TC-major along ne[0]
//             reshape → (t/patch, patch*c)
lm_ggml_tensor * codec_moss_patch_encode(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    int32_t patch) {
    if (ctx == nullptr || x_tc == nullptr || patch <= 1) return x_tc;
    const int64_t t = x_tc->ne[0];
    const int64_t c = x_tc->ne[1];
    if (t % patch != 0) return nullptr;
    const int64_t t_out = t / patch;
    // Reshape so dim0=patch, dim1=t_out, dim2=c.
    lm_ggml_tensor * x3 = lm_ggml_reshape_3d(ctx, x_tc, patch, t_out, c);
    // Permute (patch, t_out, c) → (t_out, patch, c) so contiguous flatten gives
    // [c outer, patch outer-inner, t_out innermost] — which is the desired
    // PyTorch layout `(B, D, patch, L) → (B, D*patch, L)` after the reshape.
    lm_ggml_tensor * x_perm = lm_ggml_cont(ctx, lm_ggml_permute(ctx, x3, 1, 0, 2, 3));
    // Reshape (t_out, patch, c) → (t_out, patch*c).  Memory order: ne[0]=t_out,
    // ne[1]=patch*c.  Matches the (B, D*patch, L_out) transposed layout where
    // channels are laid out as `(d_orig, patch_idx)` flattened.
    return lm_ggml_reshape_2d(ctx, x_perm, t_out, patch * c);
}

// PatchedPretransform.decode (mirror of encode): (B, D*patch, L) → (B, D, L*patch).
// In TC layout: input (t, d_out=d_in/patch, patch_inner_dim) → output (t*patch, d).
lm_ggml_tensor * codec_moss_patch_decode(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    int32_t patch) {
    if (ctx == nullptr || x_tc == nullptr || patch <= 1) return x_tc;
    const int64_t t = x_tc->ne[0];
    const int64_t c = x_tc->ne[1];
    if (c % patch != 0) return nullptr;
    const int64_t c_out = c / patch;
    // Reshape ne=(t, patch, c_out) — channels are (patch, c_out) interleaved.
    lm_ggml_tensor * x3 = lm_ggml_reshape_3d(ctx, x_tc, t, patch, c_out);
    // Permute (t, patch, c_out) → (patch, t, c_out): now ne[0]=patch innermost
    // for each (c_out, t) pair the patch-many slots come back together.
    lm_ggml_tensor * x_perm = lm_ggml_cont(ctx, lm_ggml_permute(ctx, x3, 1, 0, 2, 3));
    return lm_ggml_reshape_2d(ctx, x_perm, patch * t, c_out);
}

// MOSS Transformer layer:
//   y = x + ls1 * self_attn(LN(x))     (causal, sliding-window, RoPE)
//   y = y + ls2 * (Linear → GELU → Linear)(LN(y))
// All linears are bias-free.  norm1/norm2 are full LayerNorm (gamma + beta).
lm_ggml_tensor * codec_moss_transformer_layer_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * n1_w, lm_ggml_tensor * n1_b,
    lm_ggml_tensor * n2_w, lm_ggml_tensor * n2_b,
    lm_ggml_tensor * qkv_w,
    lm_ggml_tensor * out_w,
    lm_ggml_tensor * fc1_w,
    lm_ggml_tensor * fc2_w,
    lm_ggml_tensor * ls1,
    lm_ggml_tensor * ls2,
    int32_t head_dim,
    int32_t n_heads,
    float rope_theta,
    int32_t window,
    int32_t n_valid) {

    if (ctx == nullptr || x_tc == nullptr) return nullptr;
    const int64_t t = x_tc->ne[0];
    const int32_t hidden = head_dim * n_heads;

    // Attention.
    lm_ggml_tensor * res = x_tc;
    lm_ggml_tensor * h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n1_w, n1_b);
    if (h == nullptr) return nullptr;

    // h_tc → h_ct (mul_mat needs ne[0]=in_dim).
    lm_ggml_tensor * h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h));   // [c=hidden, t]
    lm_ggml_tensor * qkv = lm_ggml_mul_mat(ctx, qkv_w, h_ct);            // [3*hidden, t]
    if (qkv == nullptr) return nullptr;
    lm_ggml_tensor * q = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden, t, qkv->nb[1], 0));
    lm_ggml_tensor * k = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden, t, qkv->nb[1], (size_t) hidden * qkv->nb[0]));
    lm_ggml_tensor * v = lm_ggml_cont(ctx, lm_ggml_view_2d(ctx, qkv, hidden, t, qkv->nb[1], (size_t) hidden * qkv->nb[0] * 2));
    lm_ggml_tensor * q_dht = lm_ggml_reshape_3d(ctx, q, head_dim, n_heads, t);    // [d, h, t]
    lm_ggml_tensor * k_dht = lm_ggml_reshape_3d(ctx, k, head_dim, n_heads, t);
    lm_ggml_tensor * v_dht = lm_ggml_reshape_3d(ctx, v, head_dim, n_heads, t);
    // MOSS RoPE: rotate pairs (q[2k], q[2k+1]) along head_dim per token
    // position.  We bypass codec_op_rope (which assumes a (d, t, h) input
    // ordering) and call lm_ggml_rope_ext directly with the natural (d, h, t)
    // layout — that matches ggml's expected (n_embd_per_head, n_head,
    // n_tokens) shape and uses NEOX-mode for interleaved pairs.
    lm_ggml_tensor * t_pos = lm_ggml_cast(ctx, lm_ggml_arange(ctx, 0.0f, (float) t, 1.0f), LM_GGML_TYPE_I32);
    // ggml NEOX = half-split pairs (k, k+D/2); NORMAL = interleaved (2k, 2k+1).
    // MOSS does `q.view(..., D//2, 2)` which is interleaved → NORMAL mode.
    lm_ggml_tensor * q_rope_dht = lm_ggml_rope_ext(ctx, q_dht, t_pos, nullptr,
                                             head_dim, LM_GGML_ROPE_TYPE_NORMAL, 0,
                                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    lm_ggml_tensor * k_rope_dht = lm_ggml_rope_ext(ctx, k_dht, t_pos, nullptr,
                                             head_dim, LM_GGML_ROPE_TYPE_NORMAL, 0,
                                             rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (q_rope_dht == nullptr || k_rope_dht == nullptr) return nullptr;
    // Permute (d, h, t) → (d, t, h) for the attention helper.
    lm_ggml_tensor * q_dth = lm_ggml_cont(ctx, lm_ggml_permute(ctx, q_rope_dht, 0, 2, 1, 3));   // [d, t, h]
    lm_ggml_tensor * k_dth = lm_ggml_cont(ctx, lm_ggml_permute(ctx, k_rope_dht, 0, 2, 1, 3));
    lm_ggml_tensor * v_dth = lm_ggml_cont(ctx, lm_ggml_permute(ctx, v_dht,      0, 2, 1, 3));

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = true;
    attn_p.window = window;
    attn_p.n_valid = n_valid;
    lm_ggml_tensor * attn_dth = codec_op_lm_attn_ctx_dth(ctx, q_dth, k_dth, v_dth, &attn_p);
    if (attn_dth == nullptr) return nullptr;
    // Permute back (d, t, h) → (d, h, t) → reshape (hidden, t).
    lm_ggml_tensor * attn_dht = lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn_dth, 0, 2, 1, 3));
    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(ctx, attn_dht, hidden, t);
    lm_ggml_tensor * out_ct = lm_ggml_mul_mat(ctx, out_w, attn_ct);                            // [hidden, t]
    if (out_ct == nullptr) return nullptr;
    lm_ggml_tensor * out_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, out_ct));                   // [t, hidden]

    // LayerScale_1: per-channel multiply.  Broadcast ls1 (ne=(hidden,)) over
    // the time axis via reshape (1, hidden) + lm_ggml_repeat to match out_tc.
    if (ls1 != nullptr) {
        lm_ggml_tensor * ls1_2d = lm_ggml_reshape_2d(ctx, ls1, 1, hidden);
        lm_ggml_tensor * ls1_rep = lm_ggml_repeat(ctx, ls1_2d, out_tc);
        out_tc = lm_ggml_mul(ctx, out_tc, ls1_rep);
    }
    x_tc = lm_ggml_add(ctx, res, out_tc);

    // FFN.
    res = x_tc;
    h = codec_op_layer_norm_tc(ctx, x_tc, 1e-5f, n2_w, n2_b);
    h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h));            // [hidden, t]
    lm_ggml_tensor * ff = lm_ggml_mul_mat(ctx, fc1_w, h_ct);        // [ffn_dim, t]
    ff = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff));             // [t, ffn_dim] — gelu reads any layout
    ff = lm_ggml_gelu(ctx, ff);
    ff = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff));             // [ffn_dim, t]
    ff = lm_ggml_mul_mat(ctx, fc2_w, ff);                        // [hidden, t]
    if (ff == nullptr) return nullptr;
    lm_ggml_tensor * ff_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff));
    if (ls2 != nullptr) {
        lm_ggml_tensor * ls2_2d = lm_ggml_reshape_2d(ctx, ls2, 1, hidden);
        lm_ggml_tensor * ls2_rep = lm_ggml_repeat(ctx, ls2_2d, ff_tc);
        ff_tc = lm_ggml_mul(ctx, ff_tc, ls2_rep);
    }
    return lm_ggml_add(ctx, res, ff_tc);
}

// ProjectedTransformer: input_proj (Linear, no bias) → causal-windowed
// Transformer stack → output_proj (Linear, no bias).  Input/output are in
// CT layout (ne[0]=channels, ne[1]=time); the Transformer body operates in
// TC layout internally.
lm_ggml_tensor * codec_moss_projected_transformer_ct(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_ct,
    const codec_model * model,
    const std::string & base,
    int32_t in_dim,
    int32_t out_dim,
    int32_t d_model,
    int32_t n_heads,
    int32_t n_layers,
    int32_t window,
    float rope_theta,
    int32_t n_valid) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, name);
    };

    // input_proj/output_proj are `nn.Linear` when `input_dim != d_model` /
    // `d_model != output_dim`, otherwise `nn.Identity()`.  The converter
    // emits the weight only when the source Linear exists; treat absence as
    // identity (in_dim == d_model, etc.).
    lm_ggml_tensor * in_w  = W(base + ".input_proj.w");
    lm_ggml_tensor * out_w = W(base + ".output_proj.w");

    lm_ggml_tensor * h_ct = (in_w != nullptr) ? lm_ggml_mul_mat(ctx, in_w, x_ct) : x_ct;
    if (h_ct == nullptr) return nullptr;
    lm_ggml_tensor * h_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h_ct));   // [t, d_model]
    (void) d_model;

    const int32_t head_dim = d_model / n_heads;
    for (int32_t li = 0; li < n_layers; ++li) {
        const std::string lp = base + ".l" + std::to_string(li);
        h_tc = codec_moss_transformer_layer_tc(
            ctx, h_tc,
            W(lp + ".norm1.w"), W(lp + ".norm1.b"),
            W(lp + ".norm2.w"), W(lp + ".norm2.b"),
            W(lp + ".attn.qkv.w"),
            W(lp + ".attn.out.w"),
            W(lp + ".ffn.fc1.w"),
            W(lp + ".ffn.fc2.w"),
            W(lp + ".ls1"),
            W(lp + ".ls2"),
            head_dim, n_heads, rope_theta, window, n_valid);
        if (h_tc == nullptr) return nullptr;
    }

    h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h_tc));     // [d_model, t]
    if (out_w == nullptr) return h_ct;
    return lm_ggml_mul_mat(ctx, out_w, h_ct);                // [out_dim, t]
    (void) out_dim;
}

}  // namespace

// =====================================================================
// Encode graph
// =====================================================================

struct moss_encode_build {
    int32_t n_in        = 0;          // pcm samples per channel (padded to bottleneck multiple)
    int32_t n_in_valid  = 0;          // pcm samples per channel before padding
    int32_t n_channels  = 1;
    bool    interleave  = true;
    int32_t latent_dim  = 0;
    int32_t rvq_dim     = 0;
    int32_t cb_dim      = 0;
    int32_t n_q         = 0;
    int32_t n_codes     = 0;          // T_audio_total / hop
    const codec_moss_audio * cfg = nullptr;
    const codec_model * model    = nullptr;
};

static bool codec_moss_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    moss_encode_build * p = static_cast<moss_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr || p->cfg == nullptr) return false;
    const codec_moss_audio & cfg = *p->cfg;
    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // Input PCM laid out as a single mono-equivalent stream.  When
    // interleaving is on the prep step has already woven channels together,
    // so the graph sees a length-`n_in * n_channels` flat tensor.
    const int32_t n_total = p->n_in * (p->interleave ? p->n_channels : 1);
    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, n_total, 1);
    lm_ggml_set_name(t_pcm, codec_moss_name_pcm());

    // The encoder works in TC layout (ne[0]=t, ne[1]=c=1).
    lm_ggml_tensor * x_tc = t_pcm;

    // Walk the encoder modules.
    for (int32_t mi = 0; mi < cfg.enc_n_modules; ++mi) {
        if (cfg.enc_module_type[mi] == 0) {
            x_tc = codec_moss_patch_encode(ctx_eval, x_tc, cfg.enc_patch_size[mi]);
            if (x_tc == nullptr) return false;
        } else {
            const std::string base = "moss.enc.b" + std::to_string(mi);
            lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc));
            const int32_t window = (int32_t) std::round(cfg.enc_context_duration[mi] * (float) (p->cfg->sample_rate * (p->cfg->channel_interleave ? p->cfg->number_channels : 1)));
            // The `window` is in tokens at the *current* frame rate; we
            // recompute it from `context_duration * frame_rate`.  At the i-th
            // encoder transformer the frame rate is
            //   frame_rate(i) = sample_rate * n_channels(if interleave) / Π_{j<i}(patch_j).
            // `enc_context_duration[mi]` already encodes the seconds budget;
            // we just need the cumulative downsample to convert to tokens.
            int32_t cum_down = 1;
            for (int32_t j = 0; j < mi; ++j) {
                if (cfg.enc_module_type[j] == 0) cum_down *= cfg.enc_patch_size[j];
            }
            const int32_t fr_num = p->cfg->sample_rate *
                                   (p->cfg->channel_interleave ? p->cfg->number_channels : 1);
            const int32_t win_tokens = (int32_t) std::round(cfg.enc_context_duration[mi] *
                                                            (float) fr_num / (float) cum_down);
            (void) window;
            // Valid (un-padded) frame count at this block's frame rate.  The
            // original mono-eq input length is `p->n_in * channels(if
            // interleave)`; after `cum_down`x downsample it becomes the frame
            // count below.  HF passes this as `input_lengths` to the SDPA
            // mask so queries don't attend to padded-zero keys.
            const int32_t valid_in_mono = p->n_in_valid * (p->interleave ? p->n_channels : 1);
            const int32_t n_valid_block = valid_in_mono / cum_down;
            x_ct = codec_moss_projected_transformer_ct(
                ctx_eval, x_ct, p->model, base,
                cfg.enc_in_dim[mi], cfg.enc_out_dim[mi],
                cfg.enc_d_model[mi], cfg.enc_n_heads[mi], cfg.enc_n_layers[mi],
                win_tokens, cfg.enc_max_period[mi], n_valid_block);
            if (x_ct == nullptr) return false;
            x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));
        }
    }

    // Quantizer: WNConv 1×1 input_proj (latent → rvq_dim).  Use codec_conv1d
    // with k=1 — it dispatches to a pointwise mul_mat.
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W("moss.q.input_proj.w"),
                        W("moss.q.input_proj.b"),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (x_tc == nullptr) return false;

    const int32_t cb_dim = p->cb_dim;
    // Residual LFQ: each level
    //   r = current residual (shape [t, rvq_dim])
    //   r → in_proj → [t, cb_dim]
    //   normalize per-row → cosine-NN argmax against codebook_norm → indices
    //   z_q_e = embedding (codebook[idx]) → out_proj → [t, rvq_dim]
    //   residual -= z_q_e
    lm_ggml_tensor * residual = x_tc;
    lm_ggml_tensor * codes_per_level[64] = { nullptr };
    if (p->n_q > 64) return false;
    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        const std::string base = "moss.q." + std::to_string(qi);

        // in_proj: 1×1 conv (rvq_dim → cb_dim)
        lm_ggml_tensor * z_e = codec_conv1d(ctx_eval, residual,
                                         W(base + ".in_proj.w"),
                                         W(base + ".in_proj.b"),
                                         1, 1, 0);
        if (z_e == nullptr) return false;

        // L2-normalize per t-step (shared helper), then move to CT for the
        // upcoming mul_mat against the codebook.
        lm_ggml_tensor * z_n = codec_op_l2_normalize_tc(ctx_eval, z_e, 1e-12f);          // [t, cb_dim]
        lm_ggml_tensor * z_n_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_n));    // [cb_dim, t]

        // Cosine sims: (cb_size, cb_dim) @ (cb_dim, t) → (cb_size, t).  Use
        // mul_mat which contracts ne[0]=cb_dim.
        lm_ggml_tensor * cb_norm = W(base + ".codebook_norm");
        if (cb_norm == nullptr) return false;
        cb_norm = codec_graph_cast_f32(ctx_eval, cb_norm);
        lm_ggml_tensor * sims = lm_ggml_mul_mat(ctx_eval, cb_norm, z_n_ct);                // [cb_size, t]
        lm_ggml_tensor * idx = lm_ggml_argmax(ctx_eval, sims);                              // ne=(t,) i32
        codes_per_level[qi] = idx;

        // Reconstruction: gather codebook (raw, not normalised) + out_proj.
        lm_ggml_tensor * cb = W(base + ".codebook");
        if (cb == nullptr) return false;
        cb = codec_graph_cast_f32(ctx_eval, cb);
        lm_ggml_tensor * z_q_ct = lm_ggml_get_rows(ctx_eval, cb, idx);                      // [cb_dim, t]
        lm_ggml_tensor * z_q_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_q_ct));
        lm_ggml_tensor * z_q_proj = codec_conv1d(ctx_eval, z_q_tc,
                                              W(base + ".out_proj.w"),
                                              W(base + ".out_proj.b"),
                                              1, 1, 0);
        if (z_q_proj == nullptr) return false;
        residual = lm_ggml_sub(ctx_eval, residual, z_q_proj);
        (void) cb_dim;
    }

    // Stack codes_per_level into a single (n_q, t) tensor by concatenating
    // along ne[1] (each idx is 1D [t] — promote to (t, 1) and concat).
    lm_ggml_tensor * codes_2d = nullptr;
    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        lm_ggml_tensor * idx_2d = lm_ggml_reshape_2d(ctx_eval, codes_per_level[qi], (int) codes_per_level[qi]->ne[0], 1);
        codes_2d = (codes_2d == nullptr) ? idx_2d : lm_ggml_concat(ctx_eval, codes_2d, idx_2d, /*dim=*/1);
    }
    codes_2d = lm_ggml_cont(ctx_eval, codes_2d);
    lm_ggml_set_name(codes_2d, codec_moss_name_codes());
    *out = codes_2d;
    return true;
}

// =====================================================================
// Decode graph
// =====================================================================

struct moss_decode_build {
    int32_t n_codes     = 0;          // = T_audio * channels / hop
    int32_t n_channels  = 1;
    bool    interleave  = true;
    int32_t n_q         = 0;
    int32_t latent_dim  = 0;
    int32_t rvq_dim     = 0;
    int32_t cb_dim      = 0;
    const codec_moss_audio * cfg = nullptr;
    const codec_model * model    = nullptr;
};

static bool codec_moss_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    moss_decode_build * p = static_cast<moss_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr || p->cfg == nullptr) return false;
    const codec_moss_audio & cfg = *p->cfg;
    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // Input codes — laid out as ne=(t, n_q) int32; each row q holds one
    // codebook's per-frame indices.
    lm_ggml_tensor * t_codes = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->n_codes, p->n_q);
    lm_ggml_set_name(t_codes, codec_moss_name_dec_codes());

    // Reverse-quantize: sum out_proj(codebook[code_q]) over q levels.
    lm_ggml_tensor * acc_tc = nullptr;
    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        lm_ggml_tensor * idx = lm_ggml_view_1d(ctx_eval, t_codes, p->n_codes, qi * t_codes->nb[1]);
        const std::string base = "moss.q." + std::to_string(qi);
        lm_ggml_tensor * cb = W(base + ".codebook");
        if (cb == nullptr) return false;
        cb = codec_graph_cast_f32(ctx_eval, cb);
        lm_ggml_tensor * z_p_ct = lm_ggml_get_rows(ctx_eval, cb, idx);                      // [cb_dim, t]
        lm_ggml_tensor * z_p_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_p_ct)); // [t, cb_dim]
        lm_ggml_tensor * z_q = codec_conv1d(ctx_eval, z_p_tc,
                                         W(base + ".out_proj.w"),
                                         W(base + ".out_proj.b"),
                                         1, 1, 0);
        if (z_q == nullptr) return false;
        acc_tc = (acc_tc == nullptr) ? z_q : lm_ggml_add(ctx_eval, acc_tc, z_q);
    }
    // Final output_proj from rvq_dim back to latent_dim.
    acc_tc = codec_conv1d(ctx_eval, acc_tc,
                          W("moss.q.output_proj.w"),
                          W("moss.q.output_proj.b"),
                          1, 1, 0);
    if (acc_tc == nullptr) return false;

    // Walk the decoder modules (mirror of encoder).
    lm_ggml_tensor * x_tc = acc_tc;
    for (int32_t mi = 0; mi < cfg.dec_n_modules; ++mi) {
        if (cfg.dec_module_type[mi] == 0) {
            x_tc = codec_moss_patch_decode(ctx_eval, x_tc, cfg.dec_patch_size[mi]);
            if (x_tc == nullptr) return false;
        } else {
            const std::string base = "moss.dec.b" + std::to_string(mi);
            // For decoder, `cum_down` is relative to the start of the decoder.
            // The model API gives `context_duration` in seconds at the
            // sample_rate; the actual frame_rate at decoder step mi is
            //   sample_rate * channels(if interleave) / Π_{j after mi}(patch_j).
            int32_t cum_remaining_down = 1;
            for (int32_t j = mi + 1; j < cfg.dec_n_modules; ++j) {
                if (cfg.dec_module_type[j] == 0) cum_remaining_down *= cfg.dec_patch_size[j];
            }
            const int32_t fr_num = p->cfg->sample_rate *
                                   (p->cfg->channel_interleave ? p->cfg->number_channels : 1);
            const int32_t win_tokens = (int32_t) std::round(cfg.dec_context_duration[mi] *
                                                            (float) fr_num / (float) cum_remaining_down);
            lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc));
            x_ct = codec_moss_projected_transformer_ct(
                ctx_eval, x_ct, p->model, base,
                cfg.dec_in_dim[mi], cfg.dec_out_dim[mi],
                cfg.dec_d_model[mi], cfg.dec_n_heads[mi], cfg.dec_n_layers[mi],
                win_tokens, cfg.dec_max_period[mi], /*n_valid=*/0);
            if (x_ct == nullptr) return false;
            x_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_ct));
        }
    }

    // Final shape after decoder: ne=(T_audio_total, 1).
    lm_ggml_set_name(x_tc, codec_moss_name_dec_audio());
    *out = x_tc;
    return true;
}

// =====================================================================
// Public entry points
// =====================================================================

static enum codec_status codec_moss_run_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens) {

    codec_moss_audio & cfg = *static_cast<codec_moss_audio *>(ctx->model->impl);
    if (pcm.empty()) {
        codec_context_set_error(ctx, "invalid MOSS-Audio PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }
    // hop_size in the GGUF metadata is the *per-channel* sample-to-codes
    // ratio (= 3840 for the 24 kHz Nano).  Internally the encoder operates
    // on a channel-interleaved stream whose downsample is the product of
    // every PatchedPretransform's patch_size.  Pad PCM per-channel first,
    // then collapse to interleaved.
    const int32_t hop_per_ch = std::max(1, cfg.hop_size);
    const int32_t n_total_in = (int32_t) pcm.size();
    if (n_total_in % cfg.number_channels != 0) {
        codec_context_set_error(ctx, "MOSS-Audio: PCM length must be divisible by number_channels");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t n_per_ch_in = n_total_in / cfg.number_channels;
    const int32_t pad_per_ch = (hop_per_ch - n_per_ch_in % hop_per_ch) % hop_per_ch;
    const int32_t n_per_ch = n_per_ch_in + pad_per_ch;
    const int32_t n_total = n_per_ch * cfg.number_channels;
    std::vector<float> pcm_pad((size_t) n_total, 0.0f);
    // Existing interleaved layout: [s0_c0, s0_c1, …, s_{N-1}_c0, s_{N-1}_c1].
    // We rewrite it with the per-channel padding tail set to zero.
    for (int32_t s = 0; s < n_per_ch_in; ++s) {
        for (int32_t c = 0; c < cfg.number_channels; ++c) {
            pcm_pad[(size_t) s * (size_t) cfg.number_channels + (size_t) c] =
                pcm[(size_t) s * (size_t) cfg.number_channels + (size_t) c];
        }
    }
    const int32_t n_codes = n_per_ch / hop_per_ch;
    if (n_codes <= 0) {
        codec_context_set_error(ctx, "MOSS-Audio input too short");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t n_in_per_ch = n_per_ch;

    moss_encode_build build = {};
    build.n_in       = n_in_per_ch;
    build.n_in_valid = n_per_ch_in;
    build.n_channels = cfg.number_channels;
    build.interleave = cfg.channel_interleave;
    build.latent_dim = cfg.latent_dim;
    build.rvq_dim    = cfg.rvq_dim;
    build.cb_dim     = cfg.codebook_dim;
    build.n_q        = cfg.n_q;
    build.n_codes    = n_codes;
    build.cfg        = &cfg;
    build.model      = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_MOSS_AUDIO_ENCODE,
              /*n_frames=*/n_codes, /*n_q=*/cfg.n_q, /*hop=*/cfg.hop_size,
              /*n_in=*/n_total, /*latent_dim=*/cfg.latent_dim },
            codec_moss_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, codec_moss_name_pcm());
    lm_ggml_tensor * t_codes = codec_graph_get_tensor(ctx, entry, codec_moss_name_codes());
    if (t_pcm == nullptr || t_codes == nullptr) {
        codec_context_set_error(ctx, "cached MOSS-Audio encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_pcm, pcm_pad.data(), pcm_pad.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (t_codes->type != LM_GGML_TYPE_I32 ||
        t_codes->ne[0] != n_codes || t_codes->ne[1] != cfg.n_q) {
        codec_context_set_error(ctx, "unexpected MOSS-Audio token tensor shape");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Read codes as (n_q, t) interleaved from the graph (column-major), then
    // re-pack into the (T, Q) layout that codec_token_buffer requires.
    std::vector<int32_t> codes_qt((size_t) cfg.n_q * (size_t) n_codes);
    if (!codec_runtime_read_tensor(t_codes, codes_qt.data(),
                                   codes_qt.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    int32_t * data = static_cast<int32_t *>(std::malloc(codes_qt.size() * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // codes_qt has ggml ne=(t, n_q) — i.e. ne[0]=t fastest, n_q outer.  In
    // memory: [t0_q0, t1_q0, …, t_{n-1}_q0, t0_q1, …].  Re-interleave into
    // (T, Q): out[t*n_q + q] = codes_qt[q*n_codes + t].
    for (int32_t t = 0; t < n_codes; ++t) {
        for (int32_t q = 0; q < cfg.n_q; ++q) {
            int32_t v = codes_qt[(size_t) q * (size_t) n_codes + (size_t) t];
            if (v < 0) v = 0;
            if (v > cfg.codebook_size - 1) v = cfg.codebook_size - 1;
            data[t * cfg.n_q + q] = v;
        }
    }

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = cfg.n_q * n_codes;
    out_tokens->n_frames = n_codes;
    out_tokens->n_q = cfg.n_q;
    out_tokens->codebook_size = cfg.codebook_size;
    out_tokens->sample_rate = cfg.encode_sample_rate;
    out_tokens->hop_size = cfg.hop_size;
    return CODEC_STATUS_SUCCESS;
}

static enum codec_status codec_moss_run_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm) {

    codec_moss_audio & cfg = *static_cast<codec_moss_audio *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_q != cfg.n_q || tokens->n_frames <= 0) {
        codec_context_set_error(ctx, "invalid MOSS-Audio token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t n_codes = tokens->n_frames;

    moss_decode_build build = {};
    build.n_codes    = n_codes;
    build.n_channels = cfg.number_channels;
    build.interleave = cfg.channel_interleave;
    build.n_q        = cfg.n_q;
    build.latent_dim = cfg.latent_dim;
    build.rvq_dim    = cfg.rvq_dim;
    build.cb_dim     = cfg.codebook_dim;
    build.cfg        = &cfg;
    build.model      = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_MOSS_AUDIO_DECODE,
              /*n_frames=*/n_codes, /*n_q=*/cfg.n_q, /*hop=*/cfg.hop_size,
              /*n_in=*/0, /*latent_dim=*/cfg.latent_dim },
            codec_moss_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_codes = codec_graph_get_tensor(ctx, entry, codec_moss_name_dec_codes());
    lm_ggml_tensor * t_audio = codec_graph_get_tensor(ctx, entry, codec_moss_name_dec_audio());
    if (t_codes == nullptr || t_audio == nullptr) {
        codec_context_set_error(ctx, "cached MOSS-Audio decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // Repack tokens from (T, Q) interleaved to (n_q, T) for the graph's
    // (ne[0]=t, ne[1]=n_q) layout — same memory move as encode but reversed.
    std::vector<int32_t> codes_qt((size_t) cfg.n_q * (size_t) n_codes);
    for (int32_t t = 0; t < n_codes; ++t) {
        for (int32_t q = 0; q < cfg.n_q; ++q) {
            int32_t v = tokens->data[t * cfg.n_q + q];
            if (v < 0) v = 0;
            if (v > cfg.codebook_size - 1) v = cfg.codebook_size - 1;
            codes_qt[(size_t) q * (size_t) n_codes + (size_t) t] = v;
        }
    }
    if (!codec_runtime_write_tensor(t_codes, codes_qt.data(),
                                    codes_qt.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_total = (int32_t) t_audio->ne[0];
    std::vector<float> raw((size_t) n_total, 0.0f);
    if (!codec_runtime_read_tensor(t_audio, raw.data(), raw.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t n_samples_per_ch = n_total;
    int32_t n_channels = 1;
    std::vector<float> out_pcm_v;
    if (cfg.channel_interleave && cfg.number_channels > 1) {
        n_samples_per_ch = n_total / cfg.number_channels;
        n_channels = cfg.number_channels;
        // raw is laid out [s0_c0, s0_c1, …, s0_cN-1, s1_c0, …].  WAV expects
        // interleaved samples too — same layout, no remap needed.
        out_pcm_v = std::move(raw);
    } else {
        out_pcm_v = std::move(raw);
    }

    float * pcm = static_cast<float *>(std::malloc(out_pcm_v.size() * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(pcm, out_pcm_v.data(), out_pcm_v.size() * sizeof(float));

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data        = pcm;
    out_pcm->n_samples   = n_samples_per_ch;
    out_pcm->sample_rate = cfg.sample_rate;
    out_pcm->n_channels  = n_channels;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_moss_audio_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    (void) out_latent;
    (void) params;
    return codec_moss_run_encode(ctx, pcm, out_tokens);
}

enum codec_status codec_moss_audio_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    (void) params;
    return codec_moss_run_decode(ctx, tokens, out_pcm);
}

enum codec_status codec_moss_audio_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_moss_audio & cfg = *static_cast<codec_moss_audio *>(model->impl);

    cfg.sample_rate    = codec_read_i32_kv(model->gguf, "codec.sample_rate", cfg.sample_rate);
    cfg.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", cfg.sample_rate);
    cfg.hop_size       = codec_read_i32_kv(model->gguf, "codec.hop_size", cfg.hop_size);
    cfg.n_q            = codec_read_i32_kv(model->gguf, "codec.n_q", cfg.n_q);
    cfg.codebook_size  = codec_read_i32_kv(model->gguf, "codec.codebook_size", cfg.codebook_size);
    cfg.codebook_dim   = codec_read_i32_kv(model->gguf, "codec.codebook_dim", cfg.codebook_dim);
    cfg.latent_dim     = codec_read_i32_kv(model->gguf, "codec.latent_dim", cfg.latent_dim);
    cfg.has_encoder    = codec_read_bool_kv(model->gguf, "codec.has_encoder", cfg.has_encoder);
    cfg.has_decoder    = codec_read_bool_kv(model->gguf, "codec.has_decoder", cfg.has_decoder);
    cfg.number_channels = codec_read_i32_kv(model->gguf, "moss.number_channels", cfg.number_channels);
    cfg.channel_interleave = codec_read_bool_kv(model->gguf, "moss.channel_interleave", cfg.channel_interleave);
    cfg.rvq_dim        = codec_read_i32_kv(model->gguf, "moss.rvq_dim", cfg.rvq_dim);
    cfg.context_duration = codec_read_f32_kv(model->gguf, "moss.context_duration", cfg.context_duration);

    cfg.enc_n_modules = codec_read_i32_kv(model->gguf, "moss.enc.n_modules", 0);
    cfg.dec_n_modules = codec_read_i32_kv(model->gguf, "moss.dec.n_modules", 0);
    if (cfg.enc_n_modules > CODEC_MOSS_MAX_MODULES || cfg.dec_n_modules > CODEC_MOSS_MAX_MODULES) {
        return CODEC_STATUS_INVALID_STATE;
    }
    auto read_arr_i = [&](const char * key, int32_t * dst, int32_t n) {
        codec_read_i32_array_kv(model->gguf, key, dst, n);
    };
    auto read_arr_f = [&](const char * key, float * dst, int32_t n) {
        codec_read_f32_array_kv(model->gguf, key, dst, n);
    };
    read_arr_i("moss.enc.module_types",       cfg.enc_module_type, cfg.enc_n_modules);
    read_arr_i("moss.enc.patch_sizes",        cfg.enc_patch_size,  cfg.enc_n_modules);
    read_arr_i("moss.enc.in_dims",            cfg.enc_in_dim,      cfg.enc_n_modules);
    read_arr_i("moss.enc.out_dims",           cfg.enc_out_dim,     cfg.enc_n_modules);
    read_arr_i("moss.enc.d_models",           cfg.enc_d_model,     cfg.enc_n_modules);
    read_arr_i("moss.enc.n_heads",            cfg.enc_n_heads,     cfg.enc_n_modules);
    read_arr_i("moss.enc.n_layers",           cfg.enc_n_layers,    cfg.enc_n_modules);
    read_arr_i("moss.enc.ffn_dims",           cfg.enc_ffn_dim,     cfg.enc_n_modules);
    read_arr_f("moss.enc.context_durations",  cfg.enc_context_duration, cfg.enc_n_modules);
    read_arr_f("moss.enc.max_periods",        cfg.enc_max_period,  cfg.enc_n_modules);
    read_arr_f("moss.enc.layer_scales",       cfg.enc_layer_scale, cfg.enc_n_modules);
    read_arr_i("moss.dec.module_types",       cfg.dec_module_type, cfg.dec_n_modules);
    read_arr_i("moss.dec.patch_sizes",        cfg.dec_patch_size,  cfg.dec_n_modules);
    read_arr_i("moss.dec.in_dims",            cfg.dec_in_dim,      cfg.dec_n_modules);
    read_arr_i("moss.dec.out_dims",           cfg.dec_out_dim,     cfg.dec_n_modules);
    read_arr_i("moss.dec.d_models",           cfg.dec_d_model,     cfg.dec_n_modules);
    read_arr_i("moss.dec.n_heads",            cfg.dec_n_heads,     cfg.dec_n_modules);
    read_arr_i("moss.dec.n_layers",           cfg.dec_n_layers,    cfg.dec_n_modules);
    read_arr_i("moss.dec.ffn_dims",           cfg.dec_ffn_dim,     cfg.dec_n_modules);
    read_arr_f("moss.dec.context_durations",  cfg.dec_context_duration, cfg.dec_n_modules);
    read_arr_f("moss.dec.max_periods",        cfg.dec_max_period,  cfg.dec_n_modules);
    read_arr_f("moss.dec.layer_scales",       cfg.dec_layer_scale, cfg.dec_n_modules);

    model->sample_rate        = cfg.sample_rate;
    model->encode_sample_rate = cfg.encode_sample_rate;
    model->has_encoder        = cfg.has_encoder;
    model->has_decoder        = cfg.has_decoder;
    model->hop_size           = cfg.hop_size;
    model->n_q                = cfg.n_q;
    model->codebook_size      = cfg.codebook_size;
    model->latent_dim         = cfg.latent_dim;
    model->expected_channels  = cfg.number_channels;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_moss_create_impl() { return new (std::nothrow) codec_moss_audio(); }
static void codec_moss_destroy_impl(void * ptr) { delete static_cast<codec_moss_audio *>(ptr); }

static enum codec_status codec_moss_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_moss_audio_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_moss_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_moss_audio_encode(ctx, pcm, out_tokens, out_latent, params);
}

const struct codec_model_vtable * codec_moss_audio_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_MOSS_AUDIO,
        "moss_audio_tokenizer",
        codec_moss_create_impl,
        codec_moss_destroy_impl,
        codec_moss_audio_init,
        codec_graph_size_exact,
        codec_moss_encode_wrap,
        codec_moss_decode_wrap,
    };
    return &vtable;
}
