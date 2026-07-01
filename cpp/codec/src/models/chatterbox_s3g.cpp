#include "chatterbox_s3g.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

// Architectural constants for the Chatterbox S3GenSR HiFTGenerator
// (S3Token2Wav.__init__ in chatterbox/s3gen/s3gen.py).
namespace {

constexpr int32_t kHiftInChannels = 80;
constexpr int32_t kHiftNFft = 16;
constexpr int32_t kHiftHop = 4;
constexpr int32_t kHiftNFftBins = kHiftNFft / 2 + 1;          // 9
constexpr int32_t kHiftHeadOutDim = kHiftNFft + 2;            // 18
constexpr int32_t kHiftNbHarmonics = 8;
constexpr float   kHiftNsfAlpha = 0.1f;
constexpr float   kHiftNsfSigma = 0.003f;
constexpr float   kHiftNsfVoicedThreshold = 10.0f;
constexpr float   kHiftLreluSlope = 0.1f;
constexpr float   kHiftLreluSlopeDefault = 0.01f; // PyTorch F.leaky_relu default
constexpr float   kHiftAudioLimit = 0.99f;
constexpr int32_t kHiftF0NumLayers = 5;
constexpr int32_t kHiftNumUps = 3;
constexpr int32_t kHiftUpsampleRates[kHiftNumUps] = {8, 5, 3};
constexpr int32_t kHiftUpsampleKernels[kHiftNumUps] = {16, 11, 7};
constexpr int32_t kHiftSourceDownStrides[kHiftNumUps] = {15, 3, 1};
constexpr int32_t kHiftSourceDownPads[kHiftNumUps] = {7, 1, 0};
constexpr int32_t kHiftResblockKernels[kHiftNumUps] = {3, 7, 11};

// Per-up-stage source resblock (one per upsample, each has 3 dilations [1, 3, 5])
constexpr int32_t kHiftSourceResblockKernels[kHiftNumUps] = {7, 7, 11};
constexpr int32_t kHiftResblockDilations[3] = {1, 3, 5};

// f0_upsamp scale factor = prod(upsample_rates) * istft_hop = 8*5*3*4 = 480.
constexpr int32_t kHiftSourceUpsample = 480;

constexpr int32_t kFlowSpkEmbedDim = 192;
constexpr int32_t kFlowVocabSize = 6561;
constexpr int32_t kFlowEncoderHidden = 512;
constexpr int32_t kFlowEncoderLayers = 6;
constexpr int32_t kFlowEncoderUpLayers = 4;

// Tunable params for flow decode (matches CFM_PARAMS in chatterbox/s3gen/configs.py).
constexpr int32_t kFlowDefaultNTimesteps = 10;
constexpr float   kFlowCfgRate = 0.7f;
constexpr float   kFlowTSchedulerCosineHalfPi = 0.5f * 3.14159265358979323846f;

// CFM estimator (ConditionalDecoder) constants — match
// chatterbox/s3gen/decoder.py / matcha/decoder.py / matcha/transformer.py.
constexpr int32_t kCfmInChannels = 320;        // packed [x, mu, spks, cond] = 80*4
constexpr int32_t kCfmOutChannels = 80;
constexpr int32_t kCfmChannels = 256;          // channels[0]
constexpr int32_t kCfmTimeEmbedDim = kCfmChannels * 4;  // 1024
constexpr int32_t kCfmNumMidBlocks = 12;
constexpr int32_t kCfmTransformersPerBlock = 4;
constexpr int32_t kCfmAttentionHeadDim = 64;
constexpr int32_t kCfmAttentionHeads = 8;
constexpr int32_t kCfmAttnInner = kCfmAttentionHeadDim * kCfmAttentionHeads;  // 512
constexpr float   kCfmTimeEmbedScale = 1000.0f;

}  // namespace

// ---------------- HiFT in-graph forwards -------------------

// f0_predictor (mel → f0). Mel input is [T_mel, 80] column-major.
static lm_ggml_tensor * codec_s3g_hift_f0_forward(
    lm_ggml_context * ctx, lm_ggml_tensor * mel_tc, const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    lm_ggml_tensor * x = mel_tc;
    for (int32_t i = 0; i < kHiftF0NumLayers; ++i) {
        lm_ggml_tensor * w = W("s3g.hift.f0.cn." + std::to_string(i) + ".w");
        lm_ggml_tensor * b = W("s3g.hift.f0.cn." + std::to_string(i) + ".b");
        if (w == nullptr || b == nullptr) return nullptr;
        x = codec_conv1d(ctx, x, w, b, /*stride=*/1, /*dilation=*/1, /*padding=*/1);
        if (x == nullptr) return nullptr;
        x = codec_op_unary(ctx, x, CODEC_UNARY_ELU);
        if (x == nullptr) return nullptr;
    }
    lm_ggml_tensor * cls_w = W("s3g.hift.f0.cls.w");
    lm_ggml_tensor * cls_b = W("s3g.hift.f0.cls.b");
    if (cls_w == nullptr || cls_b == nullptr) return nullptr;
    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x));
    lm_ggml_tensor * f0 = lm_ggml_mul_mat(ctx, cls_w, x_ct);
    if (f0 == nullptr) return nullptr;
    f0 = lm_ggml_add(ctx, f0, cls_b);
    f0 = lm_ggml_abs(ctx, f0);
    return lm_ggml_reshape_1d(ctx, f0, x->ne[0]);
}

// ---------------- HiFT main path (mel + s_stft → conv_post head) ----------------

// HiFi-GAN ResBlock with 3 dilation branches at the chatterbox dilations.
// Each branch is the shared `codec_op_hifigan_resblock_branch_ct`.
static lm_ggml_tensor * codec_s3g_apply_resblock(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    const codec_model * model,
    const std::string & prefix,
    int32_t kernel_size) {
    for (int32_t idx = 0; idx < 3; ++idx) {
        const int32_t d = kHiftResblockDilations[idx];
        lm_ggml_tensor * a1 = codec_graph_weight(ctx_eval, model, prefix + ".a1." + std::to_string(idx));
        lm_ggml_tensor * a2 = codec_graph_weight(ctx_eval, model, prefix + ".a2." + std::to_string(idx));
        lm_ggml_tensor * c1w = codec_graph_weight(ctx_eval, model, prefix + ".cv1." + std::to_string(idx) + ".w");
        lm_ggml_tensor * c1b = codec_graph_weight(ctx_eval, model, prefix + ".cv1." + std::to_string(idx) + ".b");
        lm_ggml_tensor * c2w = codec_graph_weight(ctx_eval, model, prefix + ".cv2." + std::to_string(idx) + ".w");
        lm_ggml_tensor * c2b = codec_graph_weight(ctx_eval, model, prefix + ".cv2." + std::to_string(idx) + ".b");
        x = codec_op_hifigan_resblock_branch_ct(ctx_eval, x, a1, a2, c1w, c1b, c2w, c2b, kernel_size, d);
        if (x == nullptr) return nullptr;
    }
    return x;
}

// HiFT main forward: (mel_tc, s_stft_tc) → head [n_fft+2, T_head] (column-major: ne[0]=T).
static lm_ggml_tensor * codec_s3g_hift_main_forward(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * t_mel,
    lm_ggml_tensor * t_stft,
    const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx_eval, model, n); };
    if (t_mel == nullptr || t_stft == nullptr) return nullptr;

    lm_ggml_tensor * conv_pre_w = W("s3g.hift.conv_pre.w");
    lm_ggml_tensor * conv_pre_b = W("s3g.hift.conv_pre.b");
    if (conv_pre_w == nullptr || conv_pre_b == nullptr) return nullptr;
    lm_ggml_tensor * x = codec_conv1d(ctx_eval, t_mel, conv_pre_w, conv_pre_b, 1, 1, 3);
    if (x == nullptr) return nullptr;

    for (int32_t i = 0; i < kHiftNumUps; ++i) {
        const int32_t u = kHiftUpsampleRates[i];
        const int32_t k = kHiftUpsampleKernels[i];
        const int32_t up_pad = (k - u) / 2;

        x = lm_ggml_leaky_relu(ctx_eval, x, kHiftLreluSlope, /*inplace=*/false);

        lm_ggml_tensor * up_w = W("s3g.hift.up." + std::to_string(i) + ".w");
        lm_ggml_tensor * up_b = W("s3g.hift.up." + std::to_string(i) + ".b");
        if (up_w == nullptr || up_b == nullptr) return nullptr;
        x = codec_convtr1d(ctx_eval, x, up_w, up_b, /*stride=*/u, /*padding=*/up_pad, /*dilation=*/1);
        if (x == nullptr) return nullptr;

        if (i == kHiftNumUps - 1) {
            // ReflectionPad1d((1, 0)) — left pad 1, right pad 0. With T >= 2 the reflected sample is x[1].
            const int32_t T = (int32_t) x->ne[0];
            lm_ggml_tensor * x_left = lm_ggml_view_2d(
                ctx_eval, x,
                /*ne0=*/1, /*ne1=*/x->ne[1],
                /*nb1=*/x->nb[1],
                /*offset=*/(size_t) 1 * x->nb[0]);
            x_left = lm_ggml_cont(ctx_eval, x_left);
            lm_ggml_tensor * x_full = lm_ggml_view_2d(
                ctx_eval, x,
                /*ne0=*/T, /*ne1=*/x->ne[1],
                /*nb1=*/x->nb[1],
                /*offset=*/0);
            x_full = lm_ggml_cont(ctx_eval, x_full);
            x = lm_ggml_concat(ctx_eval, x_left, x_full, /*dim=*/0);
            if (x == nullptr) return nullptr;
        }

        // Source path: source_downs[i] is plain Conv1d (no weight_norm output reshape needed since converter materialised).
        lm_ggml_tensor * sd_w = W("s3g.hift.src_dn." + std::to_string(i) + ".w");
        lm_ggml_tensor * sd_b = W("s3g.hift.src_dn." + std::to_string(i) + ".b");
        if (sd_w == nullptr || sd_b == nullptr) return nullptr;
        lm_ggml_tensor * si = codec_conv1d(
            ctx_eval, t_stft, sd_w, sd_b,
            /*stride=*/kHiftSourceDownStrides[i],
            /*dilation=*/1,
            /*padding=*/kHiftSourceDownPads[i]);
        if (si == nullptr) return nullptr;

        si = codec_s3g_apply_resblock(
            ctx_eval, si, model,
            "s3g.hift.src_rb." + std::to_string(i),
            kHiftSourceResblockKernels[i]);
        if (si == nullptr) return nullptr;

        // Source x and main x must align in the time dimension. ConvTranspose1d output
        // length = (T_in - 1) * stride + kernel - 2*pad. After the trailing reflection
        // pad on the last stage the +1 sample shows up. The source path produces the
        // matching length for stages 0/1; on stage 2 we right-trim si to x's length.
        if (si->ne[0] != x->ne[0]) {
            const int32_t common = (int32_t) std::min<int64_t>(si->ne[0], x->ne[0]);
            lm_ggml_tensor * si_trim = lm_ggml_cont(ctx_eval, lm_ggml_view_2d(
                ctx_eval, si,
                /*ne0=*/common, /*ne1=*/si->ne[1],
                /*nb1=*/si->nb[1],
                /*offset=*/0));
            lm_ggml_tensor * x_trim = lm_ggml_cont(ctx_eval, lm_ggml_view_2d(
                ctx_eval, x,
                /*ne0=*/common, /*ne1=*/x->ne[1],
                /*nb1=*/x->nb[1],
                /*offset=*/0));
            si = si_trim;
            x = x_trim;
        }
        x = lm_ggml_add(ctx_eval, x, si);

        // 3 parallel resblocks averaged.
        lm_ggml_tensor * xs = nullptr;
        for (int32_t j = 0; j < 3; ++j) {
            lm_ggml_tensor * branch = codec_s3g_apply_resblock(
                ctx_eval, x, model,
                "s3g.hift.rb." + std::to_string(i * 3 + j),
                kHiftResblockKernels[j]);
            if (branch == nullptr) return nullptr;
            xs = (xs == nullptr) ? branch : lm_ggml_add(ctx_eval, xs, branch);
        }
        x = lm_ggml_scale(ctx_eval, xs, 1.0f / 3.0f);
    }

    x = lm_ggml_leaky_relu(ctx_eval, x, kHiftLreluSlopeDefault, /*inplace=*/false);

    lm_ggml_tensor * cp_w = W("s3g.hift.conv_post.w");
    lm_ggml_tensor * cp_b = W("s3g.hift.conv_post.b");
    if (cp_w == nullptr || cp_b == nullptr) return nullptr;
    x = codec_conv1d(ctx_eval, x, cp_w, cp_b, 1, 1, 3);  // [t_pcm/4, 18]
    if (x == nullptr) return nullptr;
    return x;
}

// ---------------- HiFT runtime orchestration -------------------


// ---------------- CFM estimator (ConditionalDecoder) graph ---------------

// CausalResnetBlock1D forward:
//   h = block1(x);  h += mlp(t).unsqueeze(-1);  h = block2(h);
//   return h + res_conv(x)
// where mlp = Mish + Linear(time_embed_dim, dim_out) and res_conv = Conv1d(dim_in, dim_out, 1).
// `t_emb` is [time_embed_dim] (1D); the shared CFM Resnet block handles the
// `mish + Linear` projection and broadcasting along the time axis.
static lm_ggml_tensor * codec_s3g_cfm_causal_resnet(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * t_emb,
    const codec_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    return codec_op_cfm_causal_resnet_block_tc(ctx, x_tc, t_emb,
        W(prefix + ".b1.cv.w"), W(prefix + ".b1.cv.b"),
        W(prefix + ".b1.ln.w"), W(prefix + ".b1.ln.b"),
        W(prefix + ".b2.cv.w"), W(prefix + ".b2.cv.b"),
        W(prefix + ".b2.ln.w"), W(prefix + ".b2.ln.b"),
        W(prefix + ".mlp.w"),   W(prefix + ".mlp.b"),
        W(prefix + ".res.w"),   W(prefix + ".res.b"));
}

// Diffusers BasicTransformerBlock (no cross-attn) — thin model-side wrapper
// that resolves weights by name and delegates to the shared op.
static lm_ggml_tensor * codec_s3g_cfm_basic_transformer(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    return codec_op_basic_transformer_block_tc(ctx, x_tc,
        W(prefix + ".norm1.w"), W(prefix + ".norm1.b"),
        W(prefix + ".attn.q.w"), W(prefix + ".attn.k.w"), W(prefix + ".attn.v.w"),
        W(prefix + ".attn.o.w"), W(prefix + ".attn.o.b"),
        W(prefix + ".norm3.w"), W(prefix + ".norm3.b"),
        W(prefix + ".ff.w1.w"), W(prefix + ".ff.w1.b"),
        W(prefix + ".ff.w2.w"), W(prefix + ".ff.w2.b"),
        kCfmAttentionHeadDim, kCfmAttentionHeads);
}


// Time-MLP-projected diffusion timestep embedding for a compile-time constant
// `t_v`. Composes the shared sinusoidal-time-emb op with the model's TimestepMLP
// (Linear → SiLU → Linear).
static lm_ggml_tensor * codec_s3g_cfm_time_emb(lm_ggml_context * ctx, float t_v, const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    lm_ggml_tensor * t_emb_in = codec_op_sinusoidal_time_emb(ctx, t_v, kCfmInChannels, kCfmTimeEmbedScale);
    if (t_emb_in == nullptr) return nullptr;

    lm_ggml_tensor * t_l1w = W("s3g.cfm.t.l1.w");
    lm_ggml_tensor * t_l1b = W("s3g.cfm.t.l1.b");
    lm_ggml_tensor * t_l2w = W("s3g.cfm.t.l2.w");
    lm_ggml_tensor * t_l2b = W("s3g.cfm.t.l2.b");
    if (t_l1w == nullptr || t_l1b == nullptr || t_l2w == nullptr || t_l2b == nullptr) return nullptr;
    lm_ggml_tensor * te_2d = lm_ggml_reshape_2d(ctx, t_emb_in, kCfmInChannels, 1);
    lm_ggml_tensor * te = codec_op_linear(ctx, te_2d, t_l1w, t_l1b);
    if (te == nullptr) return nullptr;
    te = codec_op_unary(ctx, te, CODEC_UNARY_SILU);
    te = codec_op_linear(ctx, te, t_l2w, t_l2b);
    if (te == nullptr) return nullptr;
    return lm_ggml_reshape_1d(ctx, te, kCfmTimeEmbedDim);
}

// Single estimator forward (ConditionalDecoder.forward): given x, mu, spks,
// cond and an already-projected time embedding, produces dxdt [T, 80].
static lm_ggml_tensor * codec_s3g_cfm_estimator_forward(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_in,    // [T, 80]
    lm_ggml_tensor * mu_in,   // [T, 80]
    lm_ggml_tensor * spks_in, // [80]
    lm_ggml_tensor * cond_in, // [T, 80]
    lm_ggml_tensor * t_emb,   // [time_embed_dim]
    const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    if (x_in == nullptr || mu_in == nullptr || spks_in == nullptr || cond_in == nullptr || t_emb == nullptr) {
        return nullptr;
    }

    // Pack input: concat([x, mu, spks_repeat, cond]) along channel dim → [T, 320].
    lm_ggml_tensor * spks_2d = lm_ggml_reshape_2d(ctx, spks_in, 1, kCfmOutChannels);
    lm_ggml_tensor * spks_rep = lm_ggml_repeat(ctx, spks_2d, x_in);
    lm_ggml_tensor * pack1 = lm_ggml_concat(ctx, x_in, mu_in, /*dim=*/1);
    lm_ggml_tensor * pack2 = lm_ggml_concat(ctx, pack1, spks_rep, /*dim=*/1);
    lm_ggml_tensor * x_tc  = lm_ggml_concat(ctx, pack2, cond_in, /*dim=*/1);
    if (x_tc == nullptr) return nullptr;

    // ---- down block ----
    lm_ggml_tensor * skip = nullptr;
    {
        const std::string p_dn = "s3g.cfm.dn.0";
        x_tc = codec_s3g_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_dn + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = codec_s3g_cfm_basic_transformer(ctx, x_tc, model, p_dn + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
        skip = x_tc;
        lm_ggml_tensor * dnw = W(p_dn + ".x.w");
        lm_ggml_tensor * dnb = W(p_dn + ".x.b");
        if (dnw == nullptr || dnb == nullptr) return nullptr;
        x_tc = codec_conv1d_causal(ctx, x_tc, dnw, dnb, /*stride=*/1, /*dilation=*/1);
        if (x_tc == nullptr) return nullptr;
    }

    // ---- mid blocks ----
    for (int32_t bi = 0; bi < kCfmNumMidBlocks; ++bi) {
        const std::string p_md = "s3g.cfm.md." + std::to_string(bi);
        x_tc = codec_s3g_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_md + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = codec_s3g_cfm_basic_transformer(ctx, x_tc, model, p_md + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
    }

    // ---- up block ----
    {
        const std::string p_up = "s3g.cfm.up.0";
        x_tc = lm_ggml_concat(ctx, x_tc, skip, /*dim=*/1);
        if (x_tc == nullptr) return nullptr;
        x_tc = codec_s3g_cfm_causal_resnet(ctx, x_tc, t_emb, model, p_up + ".r");
        if (x_tc == nullptr) return nullptr;
        for (int32_t ti = 0; ti < kCfmTransformersPerBlock; ++ti) {
            x_tc = codec_s3g_cfm_basic_transformer(ctx, x_tc, model, p_up + ".t." + std::to_string(ti));
            if (x_tc == nullptr) return nullptr;
        }
        lm_ggml_tensor * upw = W(p_up + ".x.w");
        lm_ggml_tensor * upb = W(p_up + ".x.b");
        if (upw == nullptr || upb == nullptr) return nullptr;
        x_tc = codec_conv1d_causal(ctx, x_tc, upw, upb, /*stride=*/1, /*dilation=*/1);
        if (x_tc == nullptr) return nullptr;
    }

    // ---- final block + final_proj ----
    lm_ggml_tensor * fcw = W("s3g.cfm.final.cv.w");
    lm_ggml_tensor * fcb = W("s3g.cfm.final.cv.b");
    lm_ggml_tensor * flw = W("s3g.cfm.final.ln.w");
    lm_ggml_tensor * flb = W("s3g.cfm.final.ln.b");
    lm_ggml_tensor * pw  = W("s3g.cfm.proj.w");
    lm_ggml_tensor * pb  = W("s3g.cfm.proj.b");
    if (fcw == nullptr || fcb == nullptr || flw == nullptr || flb == nullptr || pw == nullptr || pb == nullptr) return nullptr;
    x_tc = codec_op_causal_block1d_tc(ctx, x_tc, fcw, fcb, flw, flb);
    if (x_tc == nullptr) return nullptr;
    return codec_conv1d(ctx, x_tc, pw, pb, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
}

// (Sinusoidal pos emb moved to codec_runtime_sinusoidal_pos_emb in audio_dsp.cpp.)

// ---------------- Flow encoder (UpsampleConformerEncoder) graph -----------

static lm_ggml_tensor * codec_s3g_flow_pre_lookahead(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    lm_ggml_tensor * c1w = W("s3g.flow.enc.pre.cv1.w");
    lm_ggml_tensor * c1b = W("s3g.flow.enc.pre.cv1.b");
    lm_ggml_tensor * c2w = W("s3g.flow.enc.pre.cv2.w");
    lm_ggml_tensor * c2b = W("s3g.flow.enc.pre.cv2.b");
    if (c1w == nullptr || c1b == nullptr || c2w == nullptr || c2b == nullptr) return nullptr;

    // F.pad(0, 3): right-pad on time → conv1d(k=4) → leaky_relu(default 0.01) → F.pad(2, 0): left-pad → conv1d(k=3) → +residual.
    lm_ggml_tensor * h = codec_op_pad_1d(ctx, x_tc, /*pad_left=*/0, /*pad_right=*/3);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c1w, c1b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;
    h = lm_ggml_leaky_relu(ctx, h, 0.01f, /*inplace=*/false);
    h = codec_op_pad_1d(ctx, h, /*pad_left=*/2, /*pad_right=*/0);
    if (h == nullptr) return nullptr;
    h = codec_conv1d(ctx, h, c2w, c2b, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (h == nullptr) return nullptr;
    return lm_ggml_add(ctx, h, x_tc);
}

static lm_ggml_tensor * codec_s3g_flow_up_layer(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    lm_ggml_tensor * uw = W("s3g.flow.enc.up.w");
    lm_ggml_tensor * ub = W("s3g.flow.enc.up.b");
    if (uw == nullptr || ub == nullptr) return nullptr;

    // F.interpolate(scale_factor=2, mode='nearest') along the time axis.
    // lm_ggml_interpolate operates on ne[0..3] dims. For x_tc with ne[0]=t, ne[1]=c:
    // resize to (2t, c, 1, 1) with nearest mode = 0.
    const int64_t t2 = x_tc->ne[0] * 2;
    const int64_t c = x_tc->ne[1];
    lm_ggml_tensor * up = lm_ggml_interpolate(ctx, x_tc, t2, c, x_tc->ne[2], x_tc->ne[3], /*mode=*/LM_GGML_SCALE_MODE_NEAREST);
    if (up == nullptr) return nullptr;

    // F.pad(stride*2=4, 0) on time axis (left=4, right=0).
    lm_ggml_tensor * up_pad = codec_op_pad_1d(ctx, up, /*pad_left=*/4, /*pad_right=*/0);
    if (up_pad == nullptr) return nullptr;
    return codec_conv1d(ctx, up_pad, uw, ub, /*stride=*/1, /*dilation=*/1, /*padding=*/0);
}

// One UpsampleConformerEncoder Conformer block:
//   x = norm_mha(x) → rel-pos self-attn → +x_residual
//   x = norm_ff(x)  → FFN(swish) → +x_residual
// (no macaron_style, no cnn_module, normalize_before=True.)
static lm_ggml_tensor * codec_s3g_flow_conformer_block(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    lm_ggml_tensor * pos_emb,  // [d, pe_len] column-major (d=hidden)
    const codec_model * model,
    const std::string & prefix) {
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx, model, n); };
    lm_ggml_tensor * nmw = W(prefix + ".norm_mha.w");
    lm_ggml_tensor * nmb = W(prefix + ".norm_mha.b");
    lm_ggml_tensor * nfw = W(prefix + ".norm_ff.w");
    lm_ggml_tensor * nfb = W(prefix + ".norm_ff.b");
    lm_ggml_tensor * qw = W(prefix + ".attn.q.w");
    lm_ggml_tensor * qb = W(prefix + ".attn.q.b");
    lm_ggml_tensor * kw = W(prefix + ".attn.k.w");
    lm_ggml_tensor * kb = W(prefix + ".attn.k.b");
    lm_ggml_tensor * vw = W(prefix + ".attn.v.w");
    lm_ggml_tensor * vb = W(prefix + ".attn.v.b");
    lm_ggml_tensor * ow = W(prefix + ".attn.o.w");
    lm_ggml_tensor * ob = W(prefix + ".attn.o.b");
    lm_ggml_tensor * pw = W(prefix + ".attn.pos.w");
    lm_ggml_tensor * pbu = W(prefix + ".attn.pbu");
    lm_ggml_tensor * pbv = W(prefix + ".attn.pbv");
    lm_ggml_tensor * f1w = W(prefix + ".ff.w1.w");
    lm_ggml_tensor * f1b = W(prefix + ".ff.w1.b");
    lm_ggml_tensor * f2w = W(prefix + ".ff.w2.w");
    lm_ggml_tensor * f2b = W(prefix + ".ff.w2.b");
    if (nmw == nullptr || nfw == nullptr || qw == nullptr || pw == nullptr || pbu == nullptr) return nullptr;

    const int32_t T = (int32_t) x_tc->ne[0];

    // norm_mha (LayerNorm over channel axis).
    lm_ggml_tensor * h_tc = codec_op_layer_norm_tc(ctx, x_tc, /*eps=*/1e-12f, nmw, nmb);
    if (h_tc == nullptr) return nullptr;
    lm_ggml_tensor * h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, h_tc));  // [c, t]

    lm_ggml_tensor * q_ct = codec_op_linear(ctx, h_ct, qw, qb);
    lm_ggml_tensor * k_ct = codec_op_linear(ctx, h_ct, kw, kb);
    lm_ggml_tensor * v_ct = codec_op_linear(ctx, h_ct, vw, vb);
    lm_ggml_tensor * p_ct = codec_op_linear(ctx, pos_emb, pw, /*b=*/nullptr);  // [c, pe_len]
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr || p_ct == nullptr) return nullptr;

    auto to_dth = [&](lm_ggml_tensor * x_ct_in, int32_t T_in) -> lm_ggml_tensor * {
        // x_ct_in: [c=heads*d, T_in] → reshape (d, h, T_in) → permute (d, T_in, h).
        lm_ggml_tensor * r = lm_ggml_reshape_3d(ctx, x_ct_in, kCfmAttentionHeadDim, kCfmAttentionHeads, T_in);
        return lm_ggml_cont(ctx, lm_ggml_permute(ctx, r, 0, 2, 1, 3));
    };

    lm_ggml_tensor * q_dth = to_dth(q_ct, T);
    lm_ggml_tensor * k_dth = to_dth(k_ct, T);
    lm_ggml_tensor * v_dth = to_dth(v_ct, T);
    lm_ggml_tensor * p_dth = to_dth(p_ct, 2 * T - 1);
    if (q_dth == nullptr || k_dth == nullptr || v_dth == nullptr || p_dth == nullptr) return nullptr;

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) kCfmAttentionHeadDim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_dth = codec_op_lm_attn_rel_pos_dth(ctx, q_dth, k_dth, v_dth, p_dth, pbu, pbv, &attn_p);
    if (attn_dth == nullptr) return nullptr;

    // (d, t, h) → (d, h, t) → (d*h, t).
    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx,
        lm_ggml_cont(ctx, lm_ggml_permute(ctx, attn_dth, 0, 2, 1, 3)),
        kCfmAttnInner, T);
    lm_ggml_tensor * proj_ct = codec_op_linear(ctx, attn_ct, ow, ob);  // [c, t]
    if (proj_ct == nullptr) return nullptr;
    lm_ggml_tensor * proj_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, proj_ct));
    x_tc = lm_ggml_add(ctx, x_tc, proj_tc);

    // FFN: norm_ff → Linear(c, ff_inner) → SiLU → Linear(ff_inner, c) → +residual.
    lm_ggml_tensor * ff_tc = codec_op_layer_norm_tc(ctx, x_tc, /*eps=*/1e-12f, nfw, nfb);
    if (ff_tc == nullptr) return nullptr;
    lm_ggml_tensor * ff_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff_tc));
    lm_ggml_tensor * ff = codec_op_linear(ctx, ff_ct, f1w, f1b);  // [ff_inner, t]
    if (ff == nullptr) return nullptr;
    ff = codec_op_unary(ctx, ff, CODEC_UNARY_SILU);
    ff = codec_op_linear(ctx, ff, f2w, f2b);  // [c, t]
    if (ff == nullptr) return nullptr;
    lm_ggml_tensor * ff_out_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ff));
    return lm_ggml_add(ctx, x_tc, ff_out_tc);
}

struct codec_s3g_flow_build {
    int32_t t_in = 0;        // total token sequence length (prompt + speech)
    int32_t mel_len1 = 0;    // builtin prompt_feat frame count (== 2 * prompt_token_len)
    int32_t sample_rate = 0; // S3GenSR (24000) for NSF harmonic frequencies
    const codec_model * model = nullptr;
};

static bool codec_s3g_build_flow(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    auto * p = static_cast<codec_s3g_flow_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || p->model == nullptr || out == nullptr || p->t_in <= 0) {
        return false;
    }
    const codec_model * model = p->model;
    auto W = [&](const std::string & n) -> lm_ggml_tensor * { return codec_graph_weight(ctx_eval, model, n); };

    const int32_t T = p->t_in;
    const int32_t T_up = T * 2;
    const int32_t mel_len1 = p->mel_len1;
    const int32_t T_total = T_up;
    const int32_t sample_rate = p->sample_rate;

    // Token IDs are the only graph input on the encoder side; the relative
    // positional encodings for both sequence lengths are built in-graph via
    // `codec_op_espnet_rel_pos_emb`.
    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, T);
    lm_ggml_set_name(t_tok, "s3g.flow.tokens");
    lm_ggml_tensor * t_pe1 = codec_op_espnet_rel_pos_emb(ctx_eval, T, kFlowEncoderHidden);
    lm_ggml_tensor * t_pe2 = codec_op_espnet_rel_pos_emb(ctx_eval, T_up, kFlowEncoderHidden);
    if (t_pe1 == nullptr || t_pe2 == nullptr) return false;

    // Token embedding lookup via lm_ggml_get_rows. Saved table shape (V, hidden) →
    // ggml ne[0]=hidden, ne[1]=V. lm_ggml_get_rows(table, indices) returns
    // [hidden, T]: rows of the embedding for each token id.
    lm_ggml_tensor * emb_table = W("s3g.flow.input_emb.w");
    if (emb_table == nullptr) return false;
    lm_ggml_tensor * x_ct = lm_ggml_get_rows(ctx_eval, emb_table, t_tok);  // [hidden, T]
    if (x_ct == nullptr) return false;

    // LinearNoSubsampling.out = Sequential(Linear(c, c), LayerNorm(c), Dropout) → +PositionalEncoding(scale=sqrt(d)).
    lm_ggml_tensor * el_w = W("s3g.flow.enc.embed.lin.w");
    lm_ggml_tensor * el_b = W("s3g.flow.enc.embed.lin.b");
    lm_ggml_tensor * en_w = W("s3g.flow.enc.embed.ln.w");
    lm_ggml_tensor * en_b = W("s3g.flow.enc.embed.ln.b");
    if (el_w == nullptr || el_b == nullptr || en_w == nullptr || en_b == nullptr) return false;
    lm_ggml_tensor * h_proj = codec_op_linear(ctx_eval, x_ct, el_w, el_b);
    if (h_proj == nullptr) return false;
    lm_ggml_tensor * h_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, h_proj));
    h_tc = codec_op_layer_norm_tc(ctx_eval, h_tc, /*eps=*/1e-5f, en_w, en_b);
    if (h_tc == nullptr) return false;
    h_tc = lm_ggml_scale(ctx_eval, h_tc, std::sqrt((float) kFlowEncoderHidden));

    // pre_lookahead.
    h_tc = codec_s3g_flow_pre_lookahead(ctx_eval, h_tc, model);
    if (h_tc == nullptr) return false;

    // 6 conformer blocks.
    for (int32_t bi = 0; bi < kFlowEncoderLayers; ++bi) {
        h_tc = codec_s3g_flow_conformer_block(ctx_eval, h_tc, t_pe1, model,
                                              "s3g.flow.enc.blk." + std::to_string(bi));
        if (h_tc == nullptr) return false;
    }

    // up_layer: 2× nearest upsample + left-pad 4 + Conv1d(k=5).
    h_tc = codec_s3g_flow_up_layer(ctx_eval, h_tc, model);
    if (h_tc == nullptr) return false;

    // up_embed (same Linear+LN+scale structure as `embed`).
    lm_ggml_tensor * eul_w = W("s3g.flow.enc.up_embed.lin.w");
    lm_ggml_tensor * eul_b = W("s3g.flow.enc.up_embed.lin.b");
    lm_ggml_tensor * eun_w = W("s3g.flow.enc.up_embed.ln.w");
    lm_ggml_tensor * eun_b = W("s3g.flow.enc.up_embed.ln.b");
    if (eul_w == nullptr || eul_b == nullptr || eun_w == nullptr || eun_b == nullptr) return false;
    {
        lm_ggml_tensor * h_ct2 = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, h_tc));
        lm_ggml_tensor * h_proj2 = codec_op_linear(ctx_eval, h_ct2, eul_w, eul_b);
        if (h_proj2 == nullptr) return false;
        h_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, h_proj2));
        h_tc = codec_op_layer_norm_tc(ctx_eval, h_tc, /*eps=*/1e-5f, eun_w, eun_b);
        if (h_tc == nullptr) return false;
        h_tc = lm_ggml_scale(ctx_eval, h_tc, std::sqrt((float) kFlowEncoderHidden));
    }

    // 4 conformer blocks for upsampled stream.
    for (int32_t bi = 0; bi < kFlowEncoderUpLayers; ++bi) {
        h_tc = codec_s3g_flow_conformer_block(ctx_eval, h_tc, t_pe2, model,
                                              "s3g.flow.enc.up_blk." + std::to_string(bi));
        if (h_tc == nullptr) return false;
    }

    // after_norm.
    lm_ggml_tensor * an_w = W("s3g.flow.enc.after_norm.w");
    lm_ggml_tensor * an_b = W("s3g.flow.enc.after_norm.b");
    if (an_w == nullptr || an_b == nullptr) return false;
    h_tc = codec_op_layer_norm_tc(ctx_eval, h_tc, /*eps=*/1e-5f, an_w, an_b);
    if (h_tc == nullptr) return false;

    // encoder_proj: Linear(512, 80) → mu in [T_total, 80] (column-major, ne[0]=T_total).
    lm_ggml_tensor * pj_w = W("s3g.flow.proj.w");
    lm_ggml_tensor * pj_b = W("s3g.flow.proj.b");
    if (pj_w == nullptr || pj_b == nullptr) return false;
    lm_ggml_tensor * h_ct_after = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, h_tc));
    lm_ggml_tensor * mu_ct = codec_op_linear(ctx_eval, h_ct_after, pj_w, pj_b);  // [80, T_total]
    if (mu_ct == nullptr) return false;
    lm_ggml_tensor * mu_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, mu_ct));  // [T_total, 80]
    lm_ggml_set_name(mu_tc, "s3g.flow.mu");

    // Speaker embedding: F.normalize(embedding, dim=1) then spk_aff Linear(192, 80).
    lm_ggml_tensor * sp_emb = W("s3g.cond.embedding");           // ne[0]=192, ne[1]=1
    lm_ggml_tensor * sa_w = W("s3g.flow.spk_aff.w");
    lm_ggml_tensor * sa_b = W("s3g.flow.spk_aff.b");
    if (sp_emb == nullptr || sa_w == nullptr || sa_b == nullptr) return false;
    // L2 normalise along the embedding dim. lm_ggml_norm divides by RMS(x), so
    // multiply by sqrt(N) afterwards to recover the actual L2-norm-scaled vector.
    // lm_ggml_rms_norm computes x / sqrt(mean(x^2) + eps); F.normalize divides by
    // sqrt(sum(x^2)) = sqrt(mean(x^2) * N) = sqrt(N) * sqrt(mean(x^2)).
    // So F.normalize(x) = lm_ggml_rms_norm(x) / sqrt(N).
    lm_ggml_tensor * sp_normed = lm_ggml_rms_norm(ctx_eval, sp_emb, /*eps=*/1e-12f);
    sp_normed = lm_ggml_scale(ctx_eval, sp_normed, 1.0f / std::sqrt((float) kFlowSpkEmbedDim));
    // Linear(192, 80) — sp_normed has ne[0]=192, treat it as [c=192, t=1].
    lm_ggml_tensor * sp_proj = codec_op_linear(ctx_eval, sp_normed, sa_w, sa_b);  // [80, 1]
    lm_ggml_tensor * spks = lm_ggml_reshape_1d(ctx_eval, sp_proj, kCfmOutChannels);
    lm_ggml_set_name(spks, "s3g.flow.spks");

    // Cond [T_total, 80]: prepend prompt_feat (mel_len1 frames), then zeros.
    // prompt_feat ggml shape: ne[0]=80=feat_dim, ne[1]=mel_len1, ne[2]=1.
    lm_ggml_tensor * pf = W("s3g.cond.prompt_feat");
    if (pf == nullptr) return false;
    // View prompt_feat as [80, mel_len1] (drop trailing dim 1) → transpose to [mel_len1, 80].
    lm_ggml_tensor * pf_2d = lm_ggml_reshape_2d(ctx_eval, pf, kCfmOutChannels, mel_len1);
    lm_ggml_tensor * pf_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, pf_2d));  // [mel_len1, 80]
    // Zero tail [T_total - mel_len1, 80].
    const int32_t tail = T_total - mel_len1;
    lm_ggml_tensor * zero_tail = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, tail, kCfmOutChannels);
    zero_tail = lm_ggml_scale(ctx_eval, zero_tail, 0.0f);
    lm_ggml_tensor * cond = lm_ggml_concat(ctx_eval, pf_tc, zero_tail, /*dim=*/0);  // [T_total, 80]
    lm_ggml_set_name(cond, "s3g.flow.cond");

    // ---- CFM ODE unrolled: 10 Euler steps, each with CFG cond/uncond passes.
    // The noise z is fed as a graph input (ggml lacks a portable cross-backend
    // RNG), then x is updated by chained lm_ggml_add nodes; nothing escapes to CPU.
    lm_ggml_tensor * t_z = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, T_total, kCfmOutChannels);
    lm_ggml_set_name(t_z, "s3g.flow.noise_z");
    lm_ggml_tensor * mu_zero   = lm_ggml_scale(ctx_eval, mu_tc, 0.0f);
    lm_ggml_tensor * spks_zero = lm_ggml_scale(ctx_eval, spks, 0.0f);
    lm_ggml_tensor * cond_zero = lm_ggml_scale(ctx_eval, cond, 0.0f);
    lm_ggml_tensor * x = t_z;
    for (int32_t i = 0; i < kFlowDefaultNTimesteps; ++i) {
        const float lin_t = (float) i / (float) kFlowDefaultNTimesteps;
        const float lin_r = (float) (i + 1) / (float) kFlowDefaultNTimesteps;
        const float t_v = 1.0f - std::cos(lin_t * kFlowTSchedulerCosineHalfPi);
        const float r_v = 1.0f - std::cos(lin_r * kFlowTSchedulerCosineHalfPi);
        const float dt = r_v - t_v;

        lm_ggml_tensor * t_emb = codec_s3g_cfm_time_emb(ctx_eval, t_v, model);
        if (t_emb == nullptr) return false;

        lm_ggml_tensor * dxdt_cond   = codec_s3g_cfm_estimator_forward(ctx_eval, x, mu_tc,   spks,      cond,      t_emb, model);
        lm_ggml_tensor * dxdt_uncond = codec_s3g_cfm_estimator_forward(ctx_eval, x, mu_zero, spks_zero, cond_zero, t_emb, model);
        if (dxdt_cond == nullptr || dxdt_uncond == nullptr) return false;

        // (1+cfg) * dxdt_cond - cfg * dxdt_uncond → cfg-corrected velocity.
        lm_ggml_tensor * sub = lm_ggml_sub(ctx_eval,
            lm_ggml_scale(ctx_eval, dxdt_cond,   1.0f + kFlowCfgRate),
            lm_ggml_scale(ctx_eval, dxdt_uncond, kFlowCfgRate));
        x = lm_ggml_add(ctx_eval, x, lm_ggml_scale(ctx_eval, sub, dt));
    }

    // ---- Trim prompt prefix from the final mel: x[mel_len1:T_total, :] ----
    const int32_t T_speech = T_total - mel_len1;
    lm_ggml_tensor * mel_full = lm_ggml_cont(ctx_eval, x);
    lm_ggml_tensor * mel_view = lm_ggml_view_2d(
        ctx_eval, mel_full,
        /*ne0=*/T_speech, /*ne1=*/kCfmOutChannels,
        /*nb1=*/mel_full->nb[1],
        /*offset=*/(size_t) mel_len1 * mel_full->nb[0]);
    lm_ggml_tensor * mel = lm_ggml_cont(ctx_eval, mel_view);
    lm_ggml_set_name(mel, "s3g.flow.mel");

    // ============ HiFT (mel → wav) all in-graph ============
    // f0_predictor.
    lm_ggml_tensor * f0 = codec_s3g_hift_f0_forward(ctx_eval, mel, model);
    if (f0 == nullptr) return false;

    // NSF source + STFT inputs. Random phase/noise are fed CPU-side as graph
    // inputs; STFT / iSTFT bases and the Hann window are constant tables and
    // also fed in (precomputed once on CPU).
    const int32_t T_pcm = T_speech * kHiftSourceUpsample;
    const int32_t T_stft = T_pcm / kHiftHop + 1;
    const int32_t kHarm = kHiftNbHarmonics + 1;
    lm_ggml_tensor * t_phase = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, kHarm);                    // [9]
    lm_ggml_tensor * t_nsf_noise = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, T_pcm, kHarm);         // [T_pcm, 9]
    lm_ggml_tensor * t_basis_re_k = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFft, 1, kHiftNFftBins);  // [n_fft, 1, n_bins] STFT conv kernel
    lm_ggml_tensor * t_basis_im_k = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFft, 1, kHiftNFftBins);
    lm_ggml_tensor * t_istft_re   = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFftBins, kHiftNFft);     // [n_bins, n_fft] iSTFT matmul weights
    lm_ggml_tensor * t_istft_im   = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFftBins, kHiftNFft);
    lm_ggml_tensor * t_hann       = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFft);                   // [n_fft]
    // ggml ConvTranspose1d weight shape: ne = (k, out_c, in_c). Identity kernel has
    // out_c=1 and in_c=n_fft so each input channel scatters to its own kernel slot.
    lm_ggml_tensor * t_ola_w      = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, kHiftNFft, 1, kHiftNFft);
    lm_ggml_set_name(t_phase, "s3g.hift.nsf_phase");
    lm_ggml_set_name(t_nsf_noise, "s3g.hift.nsf_noise");
    lm_ggml_set_name(t_basis_re_k, "s3g.hift.stft_basis_re_k");
    lm_ggml_set_name(t_basis_im_k, "s3g.hift.stft_basis_im_k");
    lm_ggml_set_name(t_istft_re, "s3g.hift.istft_basis_re");
    lm_ggml_set_name(t_istft_im, "s3g.hift.istft_basis_im");
    lm_ggml_set_name(t_hann, "s3g.hift.hann");
    lm_ggml_set_name(t_ola_w, "s3g.hift.ola_w");

    // ----- NSF source generation in-graph -----
    // Upsample f0 [T_speech] → [T_pcm] (nearest neighbour, factor 480).
    lm_ggml_tensor * f0_4d = lm_ggml_reshape_4d(ctx_eval, f0, T_speech, 1, 1, 1);
    lm_ggml_tensor * f0_pcm_4d = lm_ggml_interpolate(ctx_eval, f0_4d, T_pcm, 1, 1, 1, LM_GGML_SCALE_MODE_NEAREST);
    lm_ggml_tensor * f0_pcm = lm_ggml_reshape_2d(ctx_eval, f0_pcm_4d, T_pcm, 1);  // [T_pcm, 1]

    // Per-harmonic frequency scales: (h+1)/sample_rate for h=0..8.
    lm_ggml_tensor * h_idx = lm_ggml_arange(ctx_eval, 1.0f, (float) (kHarm + 1), 1.0f);  // [9] = [1..9]
    lm_ggml_tensor * scales = lm_ggml_scale(ctx_eval, h_idx, 1.0f / (float) sample_rate);
    lm_ggml_tensor * scales_2d = lm_ggml_reshape_2d(ctx_eval, scales, 1, kHarm);
    lm_ggml_tensor * f_harm_template = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, T_pcm, kHarm);
    lm_ggml_tensor * f_harm = lm_ggml_mul(ctx_eval,
        lm_ggml_repeat(ctx_eval, f0_pcm, f_harm_template),
        lm_ggml_repeat(ctx_eval, scales_2d, f_harm_template));

    // theta = 2π · cumsum_T(f_harm) → [T_pcm, 9].
    lm_ggml_tensor * theta = lm_ggml_scale(ctx_eval, lm_ggml_cumsum(ctx_eval, f_harm), 2.0f * (float) M_PI);

    // sine_waves = sine_amp * sin(theta + phase_in)
    lm_ggml_tensor * phase_2d = lm_ggml_reshape_2d(ctx_eval, t_phase, 1, kHarm);
    lm_ggml_tensor * sine_waves = lm_ggml_sin(ctx_eval,
        lm_ggml_add(ctx_eval, theta, lm_ggml_repeat(ctx_eval, phase_2d, f_harm_template)));
    sine_waves = lm_ggml_scale(ctx_eval, sine_waves, kHiftNsfAlpha);

    // uv mask = step(f0_pcm - voiced_threshold) → [T_pcm, 1]; broadcast to [T_pcm, 9].
    lm_ggml_tensor * uv = lm_ggml_step(ctx_eval, lm_ggml_scale_bias(ctx_eval, f0_pcm, 1.0f, -kHiftNsfVoicedThreshold));
    lm_ggml_tensor * uv_bcast = lm_ggml_repeat(ctx_eval, uv, f_harm_template);

    // noise_amp = (sigma - alpha/3) * uv + alpha/3; noise = noise_amp * nsf_noise.
    lm_ggml_tensor * noise_amp = lm_ggml_scale_bias(ctx_eval, uv_bcast,
        kHiftNsfSigma - kHiftNsfAlpha / 3.0f,
        kHiftNsfAlpha / 3.0f);
    lm_ggml_tensor * noise = lm_ggml_mul(ctx_eval, noise_amp, t_nsf_noise);

    // waves = sine_waves * uv + noise; sine_merge = tanh(linear(waves)) along harmonic axis.
    lm_ggml_tensor * waves = lm_ggml_add(ctx_eval, lm_ggml_mul(ctx_eval, sine_waves, uv_bcast), noise);
    lm_ggml_tensor * lin_w = W("s3g.hift.src.lin.w");  // [1, 9] PyTorch → ggml ne[0]=9, ne[1]=1.
    lm_ggml_tensor * lin_b = W("s3g.hift.src.lin.b");
    if (lin_w == nullptr || lin_b == nullptr) return false;
    lm_ggml_tensor * waves_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, waves));  // [9, T_pcm]
    lm_ggml_tensor * sine_merge_ct = codec_op_linear(ctx_eval, waves_ct, lin_w, lin_b);  // [1, T_pcm]
    lm_ggml_tensor * sine_merge_1d = lm_ggml_tanh(ctx_eval, lm_ggml_reshape_1d(ctx_eval, sine_merge_ct, T_pcm));

    // ----- Source STFT in-graph -----
    // Zero-pad sine_merge by n_fft/2 each side, then conv1d against basis kernels.
    lm_ggml_tensor * sm_2d = lm_ggml_reshape_2d(ctx_eval, sine_merge_1d, T_pcm, 1);          // [T_pcm, 1]
    lm_ggml_tensor * sm_padded = codec_op_pad_1d(ctx_eval, sm_2d,
        /*pad_left=*/kHiftNFft / 2, /*pad_right=*/kHiftNFft / 2);                       // [T_pcm + n_fft, 1]
    lm_ggml_tensor * stft_re = codec_conv1d(ctx_eval, sm_padded, t_basis_re_k, /*b=*/nullptr,
        /*stride=*/kHiftHop, /*dilation=*/1, /*padding=*/0);                            // [T_stft, n_bins]
    lm_ggml_tensor * stft_im = codec_conv1d(ctx_eval, sm_padded, t_basis_im_k, /*b=*/nullptr,
        /*stride=*/kHiftHop, /*dilation=*/1, /*padding=*/0);                            // [T_stft, n_bins]
    if (stft_re == nullptr || stft_im == nullptr) return false;
    lm_ggml_tensor * s_stft = lm_ggml_concat(ctx_eval, stft_re, stft_im, /*dim=*/1);          // [T_stft, n_fft+2]
    (void) T_stft;

    // ----- HiFT main path -----
    lm_ggml_tensor * head = codec_s3g_hift_main_forward(ctx_eval, mel, s_stft, model);     // [T_head, n_fft+2]
    if (head == nullptr) return false;

    // ----- iSTFT in-graph -----
    const int32_t T_head = (int32_t) head->ne[0];
    // Split head into log-mag / phase along channel dim (ne[1]).
    lm_ggml_tensor * head_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, head));  // [n_fft+2, T_head]
    lm_ggml_tensor * mag_log = lm_ggml_view_2d(ctx_eval, head_ct, kHiftNFftBins, T_head, head_ct->nb[1], 0);
    lm_ggml_tensor * phase_v = lm_ggml_view_2d(ctx_eval, head_ct, kHiftNFftBins, T_head, head_ct->nb[1],
        (size_t) kHiftNFftBins * head_ct->nb[0]);
    mag_log = lm_ggml_cont(ctx_eval, mag_log);
    phase_v = lm_ggml_cont(ctx_eval, phase_v);

    // mag = exp(clamp_max(mag_log, 1e2)). Approximate via min then exp.
    lm_ggml_tensor * mag = lm_ggml_exp(ctx_eval, lm_ggml_clamp(ctx_eval, mag_log, -1e30f, 1e2f));
    lm_ggml_tensor * phase_sin = lm_ggml_sin(ctx_eval, phase_v);
    lm_ggml_tensor * re_F = lm_ggml_mul(ctx_eval, mag, lm_ggml_cos(ctx_eval, phase_sin));   // [n_bins, T_head]
    lm_ggml_tensor * im_F = lm_ggml_mul(ctx_eval, mag, lm_ggml_sin(ctx_eval, phase_sin));

    // Synthesise frames: frame[n, t] = sum_k (basis_re[n, k] * re_F[k, t] - basis_im[n, k] * im_F[k, t]) / N.
    lm_ggml_tensor * frame_re = codec_op_linear(ctx_eval, re_F, t_istft_re, /*b=*/nullptr); // [n_fft, T_head]
    lm_ggml_tensor * frame_im = codec_op_linear(ctx_eval, im_F, t_istft_im, /*b=*/nullptr);
    if (frame_re == nullptr || frame_im == nullptr) return false;
    lm_ggml_tensor * frame = lm_ggml_scale(ctx_eval, lm_ggml_sub(ctx_eval, frame_re, frame_im), 1.0f / (float) kHiftNFft);

    // Window-multiply: broadcast hann [n_fft] over [n_fft, T_head].
    lm_ggml_tensor * hann_2d = lm_ggml_reshape_2d(ctx_eval, t_hann, kHiftNFft, 1);
    lm_ggml_tensor * windowed = lm_ggml_mul(ctx_eval, frame, lm_ggml_repeat(ctx_eval, hann_2d, frame));

    // OLA via ConvTranspose1d with identity weight (kernel n_fft, in=n_fft, out=1).
    // Input layout [t, c]: codec_convtr1d expects ne[0]=t, ne[1]=in_c. Our `windowed`
    // is [n_fft, T_head] (ne[0]=n_fft, ne[1]=T_head); transpose to [T_head, n_fft].
    lm_ggml_tensor * windowed_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, windowed));
    lm_ggml_tensor * ola_signal = codec_convtr1d(ctx_eval, windowed_tc, t_ola_w, /*b=*/nullptr,
        /*stride=*/kHiftHop, /*padding=*/0, /*dilation=*/1);                       // [T_pcm + n_fft, 1]

    // OLA envelope: same convtr on a constant [T_head, n_fft] tensor of window^2.
    // Build window^2 broadcast: hann_sq = hann * hann.
    lm_ggml_tensor * hann_sq = lm_ggml_mul(ctx_eval, t_hann, t_hann);
    lm_ggml_tensor * hann_sq_2d = lm_ggml_reshape_2d(ctx_eval, hann_sq, kHiftNFft, 1);
    lm_ggml_tensor * env_template = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, T_head, kHiftNFft);
    lm_ggml_tensor * env_in = lm_ggml_repeat(ctx_eval, lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, hann_sq_2d)), env_template);
    lm_ggml_tensor * env = codec_convtr1d(ctx_eval, env_in, t_ola_w, /*b=*/nullptr,
        /*stride=*/kHiftHop, /*padding=*/0, /*dilation=*/1);

    // Divide signal by envelope (clamp envelope above a small epsilon).
    lm_ggml_tensor * env_safe = lm_ggml_clamp(ctx_eval, env, 1e-11f, 1e30f);
    lm_ggml_tensor * signal = lm_ggml_div(ctx_eval, ola_signal, env_safe);                // [out_size, 1]

    // Trim n_fft/2 from each side and squeeze the channel dim.
    const int32_t out_size = (T_head - 1) * kHiftHop + kHiftNFft;
    const int32_t pad = kHiftNFft / 2;
    const int32_t pcm_len = out_size - 2 * pad;
    lm_ggml_tensor * signal_cont = lm_ggml_cont(ctx_eval, signal);
    lm_ggml_tensor * signal_view = lm_ggml_view_2d(ctx_eval, signal_cont, pcm_len, 1,
        signal_cont->nb[1], (size_t) pad * signal_cont->nb[0]);
    lm_ggml_tensor * pcm = lm_ggml_reshape_1d(ctx_eval, lm_ggml_cont(ctx_eval, signal_view), pcm_len);
    pcm = lm_ggml_clamp(ctx_eval, pcm, -kHiftAudioLimit, kHiftAudioLimit);
    lm_ggml_set_name(pcm, "s3g.flow.pcm");

    *out = pcm;
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
    s3g.has_builtin_conditioning = codec_read_bool_kv(
        model->gguf, "chatterbox_s3g.has_builtin_conditioning", s3g.has_builtin_conditioning);
    s3g.builtin_prompt_token_len = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_token_len", s3g.builtin_prompt_token_len);
    s3g.builtin_prompt_feat_frames = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_feat_frames", s3g.builtin_prompt_feat_frames);
    s3g.builtin_prompt_feat_dim = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.prompt_feat_dim", s3g.builtin_prompt_feat_dim);
    s3g.builtin_embedding_dim = codec_read_i32_kv(
        model->gguf, "chatterbox_s3g.cond.embedding_dim", s3g.builtin_embedding_dim);
    codec_read_i32_array_kv_vec(model->gguf, "chatterbox_s3g.cond.prompt_token", &s3g.builtin_prompt_token);
    s3g.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", s3g.has_encoder);
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
    if (s3g.meanflow) {
        // Meanflow inference path is not implemented yet; flag explicitly.
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    if (s3g.has_builtin_conditioning) {
        if (s3g.builtin_prompt_token_len <= 0 ||
            s3g.builtin_prompt_feat_frames <= 0 ||
            s3g.builtin_prompt_feat_dim <= 0 ||
            s3g.builtin_embedding_dim <= 0 ||
            s3g.builtin_prompt_token.empty() ||
            s3g.builtin_prompt_token_len > (int32_t) s3g.builtin_prompt_token.size()) {
            return CODEC_STATUS_INVALID_ARG;
        }

        lm_ggml_tensor * prompt_feat = lm_ggml_get_tensor(model->weights, "s3g.cond.prompt_feat");
        lm_ggml_tensor * embedding = lm_ggml_get_tensor(model->weights, "s3g.cond.embedding");
        if (prompt_feat == nullptr || embedding == nullptr) {
            return CODEC_STATUS_INVALID_ARG;
        }
        if (prompt_feat->type != LM_GGML_TYPE_F32 || embedding->type != LM_GGML_TYPE_F32) {
            return CODEC_STATUS_INVALID_ARG;
        }
        if (prompt_feat->ne[0] != s3g.builtin_prompt_feat_dim ||
            prompt_feat->ne[1] != s3g.builtin_prompt_feat_frames ||
            prompt_feat->ne[2] != 1 ||
            embedding->ne[0] != s3g.builtin_embedding_dim ||
            embedding->ne[1] != 1) {
            return CODEC_STATUS_INVALID_ARG;
        }
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
    if (!s3g->has_builtin_conditioning) {
        codec_context_set_error(ctx, "Chatterbox-S3G decode requires builtin conditioning (no ref_wav path yet)");
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    if (s3g->meanflow) {
        codec_context_set_error(ctx, "Chatterbox-S3G meanflow path not implemented");
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    std::string err;

    // Build the full token list: prompt_token (builtin) + speech_token (filtered).
    std::vector<int32_t> tok_seq;
    tok_seq.reserve(s3g->builtin_prompt_token_len + tokens->n_tokens);
    for (int32_t i = 0; i < s3g->builtin_prompt_token_len; ++i) {
        tok_seq.push_back(s3g->builtin_prompt_token[(size_t) i]);
    }
    for (int32_t i = 0; i < tokens->n_tokens; ++i) {
        const int32_t v = tokens->data[i];
        if (v >= 0 && v < s3g->codebook_size) tok_seq.push_back(v);
    }
    const int32_t T_tok = (int32_t) tok_seq.size();
    if (T_tok <= 0) {
        codec_context_set_error(ctx, "empty token sequence");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t mel_len1 = s3g->builtin_prompt_feat_frames;
    const int32_t T_total = T_tok * 2;
    if (mel_len1 > T_total) {
        codec_context_set_error(ctx, "prompt_feat_frames exceeds encoder output length");
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t T_speech = T_total - mel_len1;
    const int32_t T_pcm = T_speech * kHiftSourceUpsample;

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
    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "s3g.flow.tokens");
    lm_ggml_tensor * t_z = codec_graph_get_tensor(ctx, entry, "s3g.flow.noise_z");
    lm_ggml_tensor * t_phase = codec_graph_get_tensor(ctx, entry, "s3g.hift.nsf_phase");
    lm_ggml_tensor * t_nsf_noise = codec_graph_get_tensor(ctx, entry, "s3g.hift.nsf_noise");
    lm_ggml_tensor * t_basis_re_k = codec_graph_get_tensor(ctx, entry, "s3g.hift.stft_basis_re_k");
    lm_ggml_tensor * t_basis_im_k = codec_graph_get_tensor(ctx, entry, "s3g.hift.stft_basis_im_k");
    lm_ggml_tensor * t_istft_re = codec_graph_get_tensor(ctx, entry, "s3g.hift.istft_basis_re");
    lm_ggml_tensor * t_istft_im = codec_graph_get_tensor(ctx, entry, "s3g.hift.istft_basis_im");
    lm_ggml_tensor * t_hann = codec_graph_get_tensor(ctx, entry, "s3g.hift.hann");
    lm_ggml_tensor * t_ola_w = codec_graph_get_tensor(ctx, entry, "s3g.hift.ola_w");
    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "s3g.flow.pcm");
    if (t_tok == nullptr || t_z == nullptr ||
        t_phase == nullptr || t_nsf_noise == nullptr ||
        t_basis_re_k == nullptr || t_basis_im_k == nullptr ||
        t_istft_re == nullptr || t_istft_im == nullptr ||
        t_hann == nullptr || t_ola_w == nullptr || t_pcm == nullptr) {
        codec_context_set_error(ctx, "invalid decode graph IO");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // STFT/iSTFT bases + Hann window + OLA identity kernel — precomputed once.
    std::vector<float> hann, stft_re_k, stft_im_k, istft_re, istft_im, ola_w;
    codec_runtime_stft_basis_kernels(kHiftNFft, &stft_re_k, &stft_im_k, &hann);
    codec_runtime_istft_synthesis_basis(kHiftNFft, hann, &istft_re, &istft_im);
    codec_runtime_ola_identity_kernel(kHiftNFft, &ola_w);

    // CFM noise + NSF random inputs.
    std::vector<float> noise_z((size_t) T_total * (size_t) kCfmOutChannels, 0.0f);
    std::vector<float> nsf_phase((size_t) (kHiftNbHarmonics + 1), 0.0f);
    std::vector<float> nsf_noise((size_t) T_pcm * (size_t) (kHiftNbHarmonics + 1), 0.0f);
    {
        std::mt19937_64 rng(/*seed=*/0);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::uniform_real_distribution<float> uni_phase(-(float) M_PI, (float) M_PI);
        for (size_t i = 0; i < noise_z.size(); ++i) noise_z[i] = gauss(rng);
        nsf_phase[0] = 0.0f;  // first harmonic phase fixed to 0 per SineGen.
        for (int32_t i = 1; i < (kHiftNbHarmonics + 1); ++i) nsf_phase[(size_t) i] = uni_phase(rng);
        for (size_t i = 0; i < nsf_noise.size(); ++i) nsf_noise[i] = gauss(rng);
    }

    if (!codec_runtime_write_tensor(t_tok, tok_seq.data(), tok_seq.size() * sizeof(int32_t), &err) ||
        !codec_runtime_write_tensor(t_z, noise_z.data(), noise_z.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_phase, nsf_phase.data(), nsf_phase.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_nsf_noise, nsf_noise.data(), nsf_noise.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_basis_re_k, stft_re_k.data(), stft_re_k.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_basis_im_k, stft_im_k.data(), stft_im_k.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_istft_re, istft_re.data(), istft_re.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_istft_im, istft_im.data(), istft_im.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_hann, hann.data(), hann.size() * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_ola_w, ola_w.data(), ola_w.size() * sizeof(float), &err)) {
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
    };
    return &vtable;
}
