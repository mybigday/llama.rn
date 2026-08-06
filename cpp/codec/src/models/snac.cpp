#include "snac.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/pool1d.h"
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
// SNAC (hubertsiuzdak/snac_24khz)
//
// Encoder:  PCM -> WNConv(1->48) -> 4 EncoderBlocks (strides [2,4,8,8])
//           with depthwise ResidualUnits + Snake1d -> depthwise WNConv.
// Quantizer: 3-level Residual VQ at strides [4,2,1] of the latent.  Each
//            level: avg_pool -> in_proj -> L2-norm -> cosine-NN against
//            (pre-baked) L2-normalized codebook -> indices.  Reconstruction
//            uses out_proj + repeat-interleave.
// Decoder:  depthwise + pointwise (latent->decoder_dim) -> 4 DecoderBlocks
//           (rates [8,8,4,2], each Snake -> ConvTranspose -> [NoiseBlock]
//           -> 3 ResidualUnits) -> Snake -> WNConv -> Tanh.
//
// Codes are exposed as a (n_q=3, T_super) int32 buffer with the SNAC
// "Orpheus" packing: row 0 = coarse codes (each repeated 4×), row 1 =
// medium codes (each repeated 2×), row 2 = fine codes — all aligned to the
// fine-grained T/512 frame rate.  This keeps the API rectangular while
// letting the decoder recover the underlying (T/2048, T/1024, T/512)
// triple by sub-sampling.
//
// NoiseBlock is run as identity (matching the deterministic decode path
// the e2e test uses); the trained `linear` weight is still loaded but the
// noise multiplier is zero, so `x + 0 * linear(x) = x`.
// =====================================================================

static const char * codec_snac_name_pcm()      { return "snac.encode.pcm"; }
static const char * codec_snac_name_codes_0()  { return "snac.encode.codes_0"; }
static const char * codec_snac_name_codes_1()  { return "snac.encode.codes_1"; }
static const char * codec_snac_name_codes_2()  { return "snac.encode.codes_2"; }
static const char * codec_snac_name_dec_in_0() { return "snac.decode.in_codes_0"; }
static const char * codec_snac_name_dec_in_1() { return "snac.decode.in_codes_1"; }
static const char * codec_snac_name_dec_in_2() { return "snac.decode.in_codes_2"; }
static const char * codec_snac_name_dec_out()  { return "snac.decode.audio"; }

// ---------------------------------------------------------------------
// Shared graph-build helpers (encode + decode share the depthwise-Snake
// ResidualUnit pattern + the WNConv1d weight loader).
// ---------------------------------------------------------------------

namespace {

// SNAC Snake1d (no alpha clamp).
//
//   y = x + (1 / (alpha + 1e-9)) * sin(alpha * x)^2
//
// Crucially, alpha can be **negative** in trained SNAC weights — the upstream
// formula `(alpha + 1e-9).reciprocal()` preserves sign.  `codec_op_snake`
// uses `clamp(alpha, eps, FLT_MAX)` (intended for BigVGAN-style monotonic
// snake) which would flip negative-alpha channels into a near-singular
// `1/eps` factor and ruin parity.  This local helper sticks to the SNAC
// formula exactly.
lm_ggml_tensor * codec_snac_snake_tc(lm_ggml_context * ctx, lm_ggml_tensor * x_tc, lm_ggml_tensor * alpha) {
    if (ctx == nullptr || x_tc == nullptr || alpha == nullptr) return nullptr;
    alpha = codec_graph_cast_f32(ctx, alpha);
    lm_ggml_tensor * a_2d = lm_ggml_reshape_2d(ctx, alpha, 1, x_tc->ne[1]);
    lm_ggml_tensor * a_rep = lm_ggml_repeat(ctx, a_2d, x_tc);                         // [t, c]
    lm_ggml_tensor * a_eps = lm_ggml_scale_bias(ctx, a_rep, 1.0f, 1e-9f);             // alpha + 1e-9
    lm_ggml_tensor * ax = lm_ggml_mul(ctx, a_rep, x_tc);                              // alpha * x
    lm_ggml_tensor * s = lm_ggml_sin(ctx, ax);
    lm_ggml_tensor * s2 = lm_ggml_mul(ctx, s, s);
    lm_ggml_tensor * frac = lm_ggml_div(ctx, s2, a_eps);                              // sin^2 / (alpha + eps)
    return lm_ggml_add(ctx, x_tc, frac);
}

// ResidualUnit (depthwise, kernel=7, dilation = d):
//   y = x + Conv1×1(Snake(Conv7-dilated-depthwise(Snake(x))))
// Time may shrink by `(k-1)*dilation` when dilation > 1 / padding < (k-1)*d/2;
// the upstream `ResidualUnit.forward` then center-crops `x` to match `y`.
lm_ggml_tensor * codec_snac_residual_unit_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * a1, lm_ggml_tensor * c1_w, lm_ggml_tensor * c1_b,
    lm_ggml_tensor * a2, lm_ggml_tensor * c2_w, lm_ggml_tensor * c2_b,
    int32_t dilation,
    int32_t kernel = 7) {

    if (ctx == nullptr || x_tc == nullptr) return nullptr;

    const int32_t pad = ((kernel - 1) * dilation) / 2;
    lm_ggml_tensor * h = codec_snac_snake_tc(ctx, x_tc, a1);
    if (h == nullptr) return nullptr;
    h = codec_conv1d_depthwise(ctx, h, c1_w, c1_b, /*stride=*/1, dilation, pad);
    if (h == nullptr) return nullptr;
    h = codec_snac_snake_tc(ctx, h, a2);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c2_w, c2_b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;

    // Center-crop x to match y if needed.
    const int64_t t_x = x_tc->ne[0];
    const int64_t t_y = h->ne[0];
    if (t_x > t_y) {
        const int32_t crop = (int32_t) ((t_x - t_y) / 2);
        x_tc = codec_op_crop_1d(ctx, x_tc, crop, (int32_t) (t_x - t_y - crop));
    }
    return lm_ggml_add(ctx, x_tc, h);
}

// (Shared `codec_op_l2_normalize_tc` lives in src/ops/lm_ggml_ops.{cpp,h}.)

// Repeat-interleave along the time axis for a TC tensor.  Each input frame
// is expanded to `factor` consecutive output frames.  Layout-wise this is
// `reshape(t, 1, c) -> repeat(factor, 1, 1) -> reshape(t*factor, c)` once we
// account for ggml's column-major-on-ne[0] convention.
lm_ggml_tensor * codec_snac_repeat_interleave_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    int32_t factor) {

    if (ctx == nullptr || x_tc == nullptr || factor <= 0) return nullptr;
    if (factor == 1) return x_tc;

    const int64_t t = x_tc->ne[0];
    const int64_t c = x_tc->ne[1];
    // x_tc as 3D: ne=(1, t, c).  lm_ggml_repeat targets ne=(factor, t, c) so each
    // (t, c) entry is replicated `factor` times along the new ne[0] axis.
    lm_ggml_tensor * x_3d = lm_ggml_reshape_3d(ctx, x_tc, 1, t, c);
    lm_ggml_tensor * tmpl = lm_ggml_new_tensor_3d(ctx, LM_GGML_TYPE_F32, factor, t, c);
    lm_ggml_tensor * x_rep = lm_ggml_repeat(ctx, x_3d, tmpl);                          // [factor, t, c]
    // Contiguous flatten: per channel, the sequence is
    //   [x[0], x[0], …, x[0] (factor×), x[1], x[1], …]  → repeat-interleave.
    return lm_ggml_reshape_2d(ctx, lm_ggml_cont(ctx, x_rep), factor * t, c);
}

}  // namespace

// ---------------------------------------------------------------------
// Encode graph
// ---------------------------------------------------------------------

struct snac_encode_build {
    int32_t n_pcm        = 0;       // padded input length (multiple of pad_to)
    int32_t latent_dim   = 0;
    int32_t cb_dim       = 0;
    int32_t cb_size      = 0;
    int32_t hop          = 0;       // np.prod(encoder_rates)
    int32_t encoder_dim  = 0;
    int32_t encoder_rates[4] = { 0, 0, 0, 0 };
    int32_t vq_strides[3]    = { 0, 0, 0 };
    int32_t n_levels     = 3;
    const codec_model * model = nullptr;
};

static lm_ggml_tensor * codec_snac_encoder_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    int32_t bi,
    int32_t stride,
    const codec_model * model) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, model, name);
    };
    const int32_t dilations[3] = { 1, 3, 9 };
    const std::string base = "snac.enc.b" + std::to_string(bi);
    for (int32_t ri = 0; ri < 3; ++ri) {
        const std::string r = base + ".r" + std::to_string(ri);
        x_tc = codec_snac_residual_unit_tc(
            ctx_eval, x_tc,
            W(r + ".act1.alpha"), W(r + ".conv1.w"), W(r + ".conv1.b"),
            W(r + ".act2.alpha"), W(r + ".conv2.w"), W(r + ".conv2.b"),
            dilations[ri]);
        if (x_tc == nullptr) return nullptr;
    }
    x_tc = codec_snac_snake_tc(ctx_eval, x_tc, W(base + ".act.alpha"));
    if (x_tc == nullptr) return nullptr;
    // WNConv1d kernel=2*stride, padding=ceil(stride/2).
    const int32_t kernel = 2 * stride;
    const int32_t padding = (stride + 1) / 2;
    return codec_conv1d(ctx_eval, x_tc, W(base + ".down.w"), W(base + ".down.b"),
                        /*stride=*/stride, /*dilation=*/1, padding);
}

static lm_ggml_tensor * codec_snac_quantize_level(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * residual_tc,        // [t_lat, latent_dim] (current residual)
    int32_t qi,
    int32_t stride,
    int32_t latent_dim,
    int32_t cb_dim,
    int32_t cb_size,
    const codec_model * model,
    lm_ggml_tensor ** out_codes,         // [t_lat / stride] int32
    lm_ggml_tensor ** out_z_q_tc) {       // reconstruction at full latent rate

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, model, name);
    };
    const std::string base = "snac.q." + std::to_string(qi);

    // 1) Avg-pool along time by `stride` (matches torch.nn.functional.avg_pool1d).
    //    In TC layout (ne[0]=t, ne[1]=c), lm_ggml_pool_1d reduces ne[0] which is
    //    exactly the time axis — no transpose needed.
    lm_ggml_tensor * pooled_tc = residual_tc;
    if (stride > 1) {
        pooled_tc = lm_ggml_pool_1d(ctx_eval, residual_tc,
                                 LM_GGML_OP_POOL_AVG, stride, stride, 0);
        if (pooled_tc == nullptr) return nullptr;
        pooled_tc = lm_ggml_cont(ctx_eval, pooled_tc);
    }

    // 2) in_proj (latent_dim -> codebook_dim) via 1×1 conv.
    lm_ggml_tensor * z_e = codec_conv1d(ctx_eval, pooled_tc,
                                     W(base + ".in_proj.w"),
                                     W(base + ".in_proj.b"),
                                     /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (z_e == nullptr) return nullptr;

    // 3) L2-normalize z_e per t-step.
    lm_ggml_tensor * z_n = codec_op_l2_normalize_tc(ctx_eval, z_e, /*eps=*/1e-12f);
    if (z_n == nullptr) return nullptr;

    // 4) Cosine-NN against the pre-baked L2-normalized codebook.  Codebook
    //    shape from converter: (cb_size, cb_dim) row-major; ggml ne =
    //    (cb_dim, cb_size).  z_n shape (t, cb_dim).  The dot product
    //    `z_n @ cb_norm.T` gives (t, cb_size).  In ggml, mul_mat contracts
    //    ne[0]: with z_n permuted to (cb_dim, t) and codebook ne=(cb_dim,
    //    cb_size), `mul_mat(codebook, z_n_ct)` returns (cb_size, t).
    lm_ggml_tensor * cb_norm = W(base + ".codebook_norm");
    if (cb_norm == nullptr) return nullptr;
    cb_norm = codec_graph_cast_f32(ctx_eval, cb_norm);

    lm_ggml_tensor * z_n_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_n));   // [d, t_lat]
    lm_ggml_tensor * sims = lm_ggml_mul_mat(ctx_eval, cb_norm, z_n_ct);                 // [cb_size, t_lat]

    // 5) argmax along ne[0] (= codebook axis) to get indices [t_lat].
    lm_ggml_tensor * idx = lm_ggml_argmax(ctx_eval, sims);                              // ne=(t_lat,) i32
    *out_codes = idx;

    // 6) Reconstruction: gather codebook[idx] (shape (t_lat, cb_dim)) → out_proj.
    lm_ggml_tensor * cb = W(base + ".codebook");
    if (cb == nullptr) return nullptr;
    cb = codec_graph_cast_f32(ctx_eval, cb);
    lm_ggml_tensor * z_q = lm_ggml_get_rows(ctx_eval, cb, idx);                         // [cb_dim, t_lat]
    // out_proj is a 1×1 conv mapping cb_dim -> latent_dim.  Switch back to TC.
    lm_ggml_tensor * z_q_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_q));    // [t_lat, cb_dim]
    z_q_tc = codec_conv1d(ctx_eval, z_q_tc,
                          W(base + ".out_proj.w"),
                          W(base + ".out_proj.b"),
                          /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (z_q_tc == nullptr) return nullptr;

    // 7) Repeat-interleave by stride to get reconstruction at the latent rate.
    if (stride > 1) {
        z_q_tc = codec_snac_repeat_interleave_tc(ctx_eval, z_q_tc, stride);
        if (z_q_tc == nullptr) return nullptr;
    }
    *out_z_q_tc = z_q_tc;
    return z_q_tc;
}

static bool codec_snac_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    snac_encode_build * p = static_cast<snac_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) return false;
    if (p->n_pcm <= 0 || p->n_levels != 3) return false;

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_pcm, 1);
    lm_ggml_set_name(t_pcm, codec_snac_name_pcm());

    // Initial WNConv(1->encoder_dim, k=7, p=3).
    lm_ggml_tensor * x_tc = codec_conv1d(ctx_eval, t_pcm,
                                      W("snac.enc.conv0.w"),
                                      W("snac.enc.conv0.b"),
                                      /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;

    for (int32_t bi = 0; bi < 4; ++bi) {
        x_tc = codec_snac_encoder_block(ctx_eval, x_tc, bi + 1, p->encoder_rates[bi], p->model);
        if (x_tc == nullptr) return false;
    }

    // Final depthwise WNConv(latent, latent, k=7, p=3).
    x_tc = codec_conv1d_depthwise(ctx_eval, x_tc,
                                  W("snac.enc.conv_final.w"),
                                  W("snac.enc.conv_final.b"),
                                  /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;

    // 3-level Residual VQ.  residual starts at z, each level emits codes_i +
    // updates residual.  We don't need the cumulative `z_q` here — only the
    // codes leave the graph.
    lm_ggml_tensor * residual = x_tc;
    lm_ggml_tensor * codes_out[3] = { nullptr, nullptr, nullptr };
    for (int32_t qi = 0; qi < 3; ++qi) {
        lm_ggml_tensor * codes = nullptr;
        lm_ggml_tensor * z_q = nullptr;
        if (codec_snac_quantize_level(
                ctx_eval, residual, qi, p->vq_strides[qi],
                p->latent_dim, p->cb_dim, p->cb_size,
                p->model, &codes, &z_q) == nullptr) {
            return false;
        }
        codes_out[qi] = codes;
        residual = lm_ggml_sub(ctx_eval, residual, z_q);
    }

    lm_ggml_set_name(codes_out[0], codec_snac_name_codes_0());
    lm_ggml_set_name(codes_out[1], codec_snac_name_codes_1());
    lm_ggml_set_name(codes_out[2], codec_snac_name_codes_2());
    // codes_out[2] is the build_fn return value and is auto-flagged as an
    // output by the runtime; the first two have to be flagged here so galloc
    // doesn't reuse their buffers after the residual `sub` consumes them.
    lm_ggml_set_output(codes_out[0]);
    lm_ggml_set_output(codes_out[1]);
    *out = codes_out[2];
    return true;
}

// ---------------------------------------------------------------------
// Decode graph
// ---------------------------------------------------------------------

struct snac_decode_build {
    int32_t n_super_frames = 0;       // = T_lat / vq_strides[0]
    int32_t latent_dim   = 0;
    int32_t cb_dim       = 0;
    int32_t cb_size      = 0;
    int32_t hop          = 0;
    int32_t decoder_dim  = 0;
    int32_t decoder_rates[4] = { 0, 0, 0, 0 };
    int32_t vq_strides[3]    = { 0, 0, 0 };
    bool    apply_noise  = false;     // run NoiseBlock with x noise=0 (identity)
    const codec_model * model = nullptr;
};

static lm_ggml_tensor * codec_snac_decoder_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_tc,
    int32_t bi,
    int32_t stride,
    bool apply_noise,
    const codec_model * model) {

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, model, name);
    };
    const std::string base = "snac.dec.b" + std::to_string(bi);

    x_tc = codec_snac_snake_tc(ctx_eval, x_tc, W(base + ".act.alpha"));
    if (x_tc == nullptr) return nullptr;
    // WNConvTranspose1d with kernel=2*stride, stride, padding=ceil(stride/2),
    // output_padding=stride%2.  In ggml conv_transpose_1d output length is
    // (T_in - 1) * stride + kernel.  PyTorch's effective output is
    // (T_in - 1) * stride + kernel - 2*padding + output_padding.  Match by
    // running the raw convtr1d then symmetrically cropping `2*padding -
    // output_padding`.
    const int32_t kernel = 2 * stride;
    const int32_t padding = (stride + 1) / 2;
    const int32_t output_padding = stride % 2;
    const int32_t total_crop = 2 * padding - output_padding;

    lm_ggml_tensor * y = codec_convtr1d(ctx_eval, x_tc,
                                     W(base + ".convtr.w"),
                                     W(base + ".convtr.b"),
                                     /*stride=*/stride, /*padding=*/0, /*dilation=*/1);
    if (y == nullptr) return nullptr;
    if (total_crop > 0) {
        // PyTorch ConvTranspose1d crops symmetrically by `padding` on each
        // side, then appends `output_padding` zero-padding at the end (we
        // collapse the two as a single asymmetric trim).
        const int32_t crop_left = padding;
        const int32_t crop_right = padding - output_padding;
        y = codec_op_crop_1d(ctx_eval, y, crop_left, crop_right);
        if (y == nullptr) return nullptr;
    }
    x_tc = y;

    if (apply_noise) {
        // NoiseBlock: x = x + linear(x) * noise, noise ~ N(0,1) shared across
        // channels.  For deterministic decode we run with noise=0, which the
        // SNAC reference matches in the parity test.  The `linear` weight is
        // still on disk; we just don't use it (kept skipping is faster than
        // computing then multiplying by zero).
    }

    const int32_t dilations[3] = { 1, 3, 9 };
    for (int32_t ri = 0; ri < 3; ++ri) {
        const std::string r = base + ".r" + std::to_string(ri);
        x_tc = codec_snac_residual_unit_tc(
            ctx_eval, x_tc,
            W(r + ".act1.alpha"), W(r + ".conv1.w"), W(r + ".conv1.b"),
            W(r + ".act2.alpha"), W(r + ".conv2.w"), W(r + ".conv2.b"),
            dilations[ri]);
        if (x_tc == nullptr) return nullptr;
    }
    return x_tc;
}

static bool codec_snac_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    snac_decode_build * p = static_cast<snac_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->model == nullptr) return false;
    if (p->n_super_frames <= 0) return false;

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    // Inputs: 3 compact 1D code tensors of lengths n_super, 2*n_super, 4*n_super.
    const int32_t n_super = p->n_super_frames;
    const int32_t lat_t = n_super * p->vq_strides[0];   // = n_super * 4
    lm_ggml_tensor * codes[3] = { nullptr, nullptr, nullptr };
    const char * names[3] = {
        codec_snac_name_dec_in_0(), codec_snac_name_dec_in_1(), codec_snac_name_dec_in_2(),
    };
    const int32_t lens[3] = {
        n_super,
        n_super * 2,
        n_super * 4,
    };
    for (int32_t qi = 0; qi < 3; ++qi) {
        codes[qi] = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, lens[qi]);
        lm_ggml_set_name(codes[qi], names[qi]);
    }

    // Sum z_q over the 3 levels.  Each level: gather codebook → out_proj →
    // repeat-interleave to length lat_t.
    lm_ggml_tensor * z_q_total = nullptr;
    for (int32_t qi = 0; qi < 3; ++qi) {
        const std::string base = "snac.q." + std::to_string(qi);
        lm_ggml_tensor * cb = W(base + ".codebook");
        if (cb == nullptr) return false;
        cb = codec_graph_cast_f32(ctx_eval, cb);
        lm_ggml_tensor * z_p = lm_ggml_get_rows(ctx_eval, cb, codes[qi]);                 // [cb_dim, t_qi]
        lm_ggml_tensor * z_p_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_p));  // [t_qi, cb_dim]
        lm_ggml_tensor * z_qi = codec_conv1d(ctx_eval, z_p_tc,
                                          W(base + ".out_proj.w"),
                                          W(base + ".out_proj.b"),
                                          /*stride=*/1, /*dilation=*/1, /*padding=*/0);
        if (z_qi == nullptr) return false;
        const int32_t stride = p->vq_strides[qi];
        if (stride > 1) {
            z_qi = codec_snac_repeat_interleave_tc(ctx_eval, z_qi, stride);
            if (z_qi == nullptr) return false;
        }
        z_q_total = (z_q_total == nullptr) ? z_qi : lm_ggml_add(ctx_eval, z_q_total, z_qi);
    }

    // Initial depthwise + pointwise: (latent, T) → (decoder_dim, T).
    lm_ggml_tensor * x_tc = codec_conv1d_depthwise(ctx_eval, z_q_total,
                                                W("snac.dec.conv_in_dw.w"),
                                                W("snac.dec.conv_in_dw.b"),
                                                /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W("snac.dec.conv_in_pw.w"),
                        W("snac.dec.conv_in_pw.b"),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (x_tc == nullptr) return false;

    for (int32_t bi = 0; bi < 4; ++bi) {
        x_tc = codec_snac_decoder_block(ctx_eval, x_tc, bi, p->decoder_rates[bi],
                                        p->apply_noise, p->model);
        if (x_tc == nullptr) return false;
    }

    // Final Snake → WNConv(64→1, k=7, p=3) → tanh.
    x_tc = codec_snac_snake_tc(ctx_eval, x_tc, W("snac.dec.act_final.alpha"));
    if (x_tc == nullptr) return false;
    x_tc = codec_conv1d(ctx_eval, x_tc,
                        W("snac.dec.conv_final.w"),
                        W("snac.dec.conv_final.b"),
                        /*stride=*/1, /*dilation=*/1, /*padding=*/3);
    if (x_tc == nullptr) return false;
    x_tc = lm_ggml_tanh(ctx_eval, x_tc);
    lm_ggml_set_name(x_tc, codec_snac_name_dec_out());
    *out = x_tc;
    (void) lat_t;
    return true;
}

// ---------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------

static enum codec_status codec_snac_run_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens) {

    codec_snac & sn = *static_cast<codec_snac *>(ctx->model->impl);
    if (pcm.empty()) {
        codec_context_set_error(ctx, "invalid SNAC PCM input");
        return CODEC_STATUS_INVALID_ARG;
    }

    // Pad input to a multiple of pad_to (= hop * vq_strides[0]).
    const int32_t pad_to = std::max(1, sn.pad_to);
    const int32_t n_in = (int32_t) pcm.size();
    const int32_t pad_for = ((n_in + pad_to - 1) / pad_to) * pad_to - n_in;
    const int32_t n_pcm = n_in + pad_for;
    std::vector<float> pcm_pad((size_t) n_pcm, 0.0f);
    std::memcpy(pcm_pad.data(), pcm.data(), pcm.size() * sizeof(float));

    const int32_t n_super = n_pcm / pad_to;            // = T/2048
    const int32_t lat_t   = n_pcm / sn.hop_size;       // = T/512
    if (n_super <= 0 || lat_t <= 0) {
        codec_context_set_error(ctx, "SNAC input too short");
        return CODEC_STATUS_INVALID_ARG;
    }

    snac_encode_build build = {};
    build.n_pcm       = n_pcm;
    build.latent_dim  = sn.latent_dim;
    build.cb_dim      = sn.codebook_dim;
    build.cb_size     = sn.codebook_size;
    build.hop         = sn.hop_size;
    build.encoder_dim = sn.encoder_dim;
    for (int i = 0; i < 4; ++i) build.encoder_rates[i] = sn.encoder_rates[i];
    for (int i = 0; i < 3; ++i) build.vq_strides[i]    = sn.vq_strides[i];
    build.n_levels = 3;
    build.model    = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_SNAC_ENCODE,
              /*n_frames=*/n_super, /*n_q=*/3, /*hop=*/sn.hop_size,
              /*n_in=*/n_pcm, /*latent_dim=*/sn.latent_dim },
            codec_snac_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm   = codec_graph_get_tensor(ctx, entry, codec_snac_name_pcm());
    lm_ggml_tensor * t_c0    = codec_graph_get_tensor(ctx, entry, codec_snac_name_codes_0());
    lm_ggml_tensor * t_c1    = codec_graph_get_tensor(ctx, entry, codec_snac_name_codes_1());
    lm_ggml_tensor * t_c2    = codec_graph_get_tensor(ctx, entry, codec_snac_name_codes_2());
    if (t_pcm == nullptr || t_c0 == nullptr || t_c1 == nullptr || t_c2 == nullptr) {
        codec_context_set_error(ctx, "cached SNAC encode graph is invalid");
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

    // Pack codes as (n_q=3, n_super) int32.  Rows are level-0 / 1 / 2 codes
    // expanded by their respective stride so all three rows have the same
    // length.  This lets the decoder reverse the packing by sub-sampling.
    auto read_i32 = [&](lm_ggml_tensor * t, std::vector<int32_t> & out) -> bool {
        const size_t n = (size_t) t->ne[0];
        out.assign(n, 0);
        return codec_runtime_read_tensor(t, out.data(), n * sizeof(int32_t), &err);
    };
    std::vector<int32_t> c0, c1, c2;
    if (!read_i32(t_c0, c0) || !read_i32(t_c1, c1) || !read_i32(t_c2, c2)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if ((int32_t) c0.size() != n_super || (int32_t) c1.size() != n_super * 2 ||
        (int32_t) c2.size() != n_super * 4) {
        codec_context_set_error(ctx, "unexpected SNAC code lengths");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_frames = n_super * sn.vq_strides[0];     // = lat_t
    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) 3 * n_frames * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    // codec_token_buffer uses an interleaved (T, Q) layout —
    // data[t*n_q + q] is the q-th codebook's value at frame t.  Pack each
    // level by stride-expanding into the fine-grained T/512 frame rate so
    // all three rows have the same length.
    const int32_t nq = 3;
    for (int32_t t = 0; t < n_frames; ++t) {
        data[t * nq + 0] = c0[(size_t) (t / 4)];   // coarse
        data[t * nq + 1] = c1[(size_t) (t / 2)];   // medium
        data[t * nq + 2] = c2[(size_t) t];         // fine
    }

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = 3 * n_frames;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = 3;
    out_tokens->codebook_size = sn.codebook_size;
    out_tokens->sample_rate = sn.encode_sample_rate;
    out_tokens->hop_size = sn.hop_size;
    return CODEC_STATUS_SUCCESS;
}

static enum codec_status codec_snac_run_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm) {

    codec_snac & sn = *static_cast<codec_snac *>(ctx->model->impl);
    if (tokens == nullptr || tokens->data == nullptr || tokens->n_q != 3 || tokens->n_frames <= 0) {
        codec_context_set_error(ctx, "invalid SNAC token buffer (expected n_q=3)");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (tokens->n_frames % sn.vq_strides[0] != 0) {
        codec_context_set_error(ctx, "SNAC token n_frames must be a multiple of vq_strides[0]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t n_frames = tokens->n_frames;
    const int32_t n_super = n_frames / sn.vq_strides[0];

    // Recover compact codes from the (n_q=3, n_frames) packing.  The loader
    // (codec_example_load_npy_i32_2d_tq) transposes 2D NPYs to a (T, Q)
    // interleaved layout, so `tokens->data[t*n_q + q]` is the q-th level's
    // value at frame t.  Row 0 (q=0) holds level-0 codes each repeated
    // vq_strides[0]× = 4×; row 1 holds level-1 codes ×2; row 2 holds level-2
    // codes raw.
    const int32_t nq = tokens->n_q;
    auto at = [&](int32_t t, int32_t q) -> int32_t {
        return std::max(0, std::min(sn.codebook_size - 1, tokens->data[t * nq + q]));
    };
    std::vector<int32_t> c0((size_t) n_super);
    std::vector<int32_t> c1((size_t) n_super * 2);
    std::vector<int32_t> c2((size_t) n_frames);
    for (int32_t i = 0; i < n_super; ++i) {
        c0[(size_t) i] = at(i * 4, 0);
    }
    for (int32_t i = 0; i < n_super * 2; ++i) {
        c1[(size_t) i] = at(i * 2, 1);
    }
    for (int32_t i = 0; i < n_frames; ++i) {
        c2[(size_t) i] = at(i, 2);
    }

    snac_decode_build build = {};
    build.n_super_frames = n_super;
    build.latent_dim     = sn.latent_dim;
    build.cb_dim         = sn.codebook_dim;
    build.cb_size        = sn.codebook_size;
    build.hop            = sn.hop_size;
    build.decoder_dim    = sn.decoder_dim;
    for (int i = 0; i < 4; ++i) build.decoder_rates[i] = sn.decoder_rates[i];
    for (int i = 0; i < 3; ++i) build.vq_strides[i]    = sn.vq_strides[i];
    build.apply_noise = false;
    build.model       = ctx->model;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_SNAC_DECODE,
              /*n_frames=*/n_super, /*n_q=*/3, /*hop=*/sn.hop_size,
              /*n_in=*/0, /*latent_dim=*/sn.latent_dim },
            codec_snac_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_c0  = codec_graph_get_tensor(ctx, entry, codec_snac_name_dec_in_0());
    lm_ggml_tensor * t_c1  = codec_graph_get_tensor(ctx, entry, codec_snac_name_dec_in_1());
    lm_ggml_tensor * t_c2  = codec_graph_get_tensor(ctx, entry, codec_snac_name_dec_in_2());
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, codec_snac_name_dec_out());
    if (t_c0 == nullptr || t_c1 == nullptr || t_c2 == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached SNAC decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_c0, c0.data(), c0.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_c1, c1.data(), c1.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_c2, c2.data(), c2.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }


    if (t_out->type != LM_GGML_TYPE_F32 || t_out->ne[1] != 1) {
        codec_context_set_error(ctx, "unexpected SNAC decode output shape/type");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_samples = (int32_t) t_out->ne[0];
    std::vector<float> pcm_v((size_t) n_samples, 0.0f);
    if (!codec_runtime_read_tensor(t_out, pcm_v.data(), pcm_v.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    float * pcm = static_cast<float *>(std::malloc(pcm_v.size() * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(pcm, pcm_v.data(), pcm_v.size() * sizeof(float));

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data        = pcm;
    out_pcm->n_samples   = n_samples;
    out_pcm->sample_rate = sn.sample_rate;
    out_pcm->n_channels  = 1;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_snac_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    (void) out_latent;
    codec_snac & sn = *static_cast<codec_snac *>(ctx->model->impl);
    if (!sn.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (params.n_q != 0 && params.n_q != 3) {
        codec_context_set_error(ctx, "SNAC encode n_q must be 0 or 3");
        return CODEC_STATUS_INVALID_ARG;
    }
    return codec_snac_run_encode(ctx, pcm, out_tokens);
}

enum codec_status codec_snac_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    (void) params;
    codec_snac & sn = *static_cast<codec_snac *>(ctx->model->impl);
    if (!sn.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    return codec_snac_run_decode(ctx, tokens, out_pcm);
}

enum codec_status codec_snac_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    codec_snac & sn = *static_cast<codec_snac *>(model->impl);

    sn.sample_rate    = codec_read_i32_kv(model->gguf, "codec.sample_rate", sn.sample_rate);
    sn.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", sn.sample_rate);
    sn.hop_size       = codec_read_i32_kv(model->gguf, "codec.hop_size", sn.hop_size);
    sn.pad_to         = codec_read_i32_kv(model->gguf, "codec.pad_to", sn.pad_to);
    sn.n_q            = codec_read_i32_kv(model->gguf, "codec.n_q", sn.n_q);
    sn.codebook_size  = codec_read_i32_kv(model->gguf, "codec.codebook_size", sn.codebook_size);
    sn.codebook_dim   = codec_read_i32_kv(model->gguf, "codec.codebook_dim", sn.codebook_dim);
    sn.latent_dim     = codec_read_i32_kv(model->gguf, "codec.latent_dim", sn.latent_dim);
    sn.encoder_dim    = codec_read_i32_kv(model->gguf, "snac.encoder_dim", sn.encoder_dim);
    sn.decoder_dim    = codec_read_i32_kv(model->gguf, "snac.decoder_dim", sn.decoder_dim);
    sn.has_encoder    = codec_read_bool_kv(model->gguf, "codec.has_encoder", sn.has_encoder);
    sn.has_decoder    = codec_read_bool_kv(model->gguf, "codec.has_decoder", sn.has_decoder);
    codec_read_i32_array_kv(model->gguf, "snac.encoder_rates", sn.encoder_rates, 4);
    codec_read_i32_array_kv(model->gguf, "snac.decoder_rates", sn.decoder_rates, 4);
    codec_read_i32_array_kv(model->gguf, "snac.vq_strides",    sn.vq_strides,    3);

    model->sample_rate        = sn.sample_rate;
    model->encode_sample_rate = sn.encode_sample_rate;
    model->has_encoder        = sn.has_encoder;
    model->has_decoder        = sn.has_decoder;
    model->hop_size           = sn.hop_size;
    model->n_q                = sn.n_q;
    model->codebook_size      = sn.codebook_size;
    model->latent_dim         = sn.latent_dim;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_snac_create_impl() { return new (std::nothrow) codec_snac(); }
static void codec_snac_destroy_impl(void * ptr) { delete static_cast<codec_snac *>(ptr); }

static enum codec_status codec_snac_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_snac_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_snac_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_snac_encode(ctx, pcm, out_tokens, out_latent, params);
}

const struct codec_model_vtable * codec_snac_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_SNAC,
        "snac",
        codec_snac_create_impl,
        codec_snac_destroy_impl,
        codec_snac_init,
        codec_graph_size_exact,
        codec_snac_encode_wrap,
        codec_snac_decode_wrap,
    };
    return &vtable;
}
