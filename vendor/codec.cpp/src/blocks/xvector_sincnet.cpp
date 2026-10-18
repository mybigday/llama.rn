#include "xvector_sincnet.h"

#include "../ops/conv1d.h"
#include "../ops/pool1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <cmath>
#include <string>

namespace {

constexpr int32_t kSincKernel = 251;
constexpr int32_t kSincStride = 10;
constexpr int32_t kPool = 3;
constexpr int32_t kTdnnKernel[5] = {5, 3, 3, 1, 1};
constexpr int32_t kTdnnDilation[5] = {1, 2, 3, 1, 1};
constexpr float kNormEps = 1e-5f;
constexpr float kLeakySlope = 0.01f;

// Frames after the SincNet stages and the (valid, undilated/dilated) TDNNs.
int32_t xvector_frames(int32_t n) {
    int32_t t = (n - kSincKernel) / kSincStride + 1;
    t = t / kPool;
    for (int i = 0; i < 2; ++i) t = (t - 5 + 1) / kPool;
    for (int i = 0; i < 5; ++i) t -= (kTdnnKernel[i] - 1) * kTdnnDilation[i];
    return t;
}

// InstanceNorm1d(affine) on [t, c]: normalise each channel over time.
ggml_tensor * instance_norm_tc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(ctx, x, kNormEps);
    x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, w, 1, w->ne[0]));
    return ggml_add(ctx, x, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
}

}  // namespace

bool codec_xvector_sincnet_accepts(int64_t n_samples) {
    return n_samples <= INT32_MAX && xvector_frames((int32_t) n_samples) >= 2;
}

ggml_tensor * codec_block_xvector_sincnet(
        ggml_context * ctx, const codec_model * model, const char * prefix, const char * proj_prefix,
        ggml_tensor * pcm16k) {
    if (ctx == nullptr || model == nullptr || prefix == nullptr || pcm16k == nullptr ||
        !codec_xvector_sincnet_accepts(ggml_nelements(pcm16k))) {
        return nullptr;
    }
    bool missing = false;
    auto W = [&](const char * pre, const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = codec_graph_weight(ctx, model, std::string(pre) + "." + n);
        if (t == nullptr) missing = true;
        return t;
    };
    const char * pre = prefix;
    ggml_tensor * wav = ggml_reshape_2d(ctx, pcm16k, ggml_nelements(pcm16k), 1);

    // SincNet: wav InstanceNorm → 3 × (conv → [abs] → maxpool(3) → InstanceNorm → LeakyReLU).
    auto sinc_stage = [&](int i, ggml_tensor * x) -> ggml_tensor * {
        const std::string ci = "conv" + std::to_string(i);
        if (i == 0) {
            x = ggml_abs(ctx, codec_conv1d(ctx, x, W(pre, ci + ".w"), nullptr, kSincStride, 1, 0));
        } else {
            x = codec_conv1d(ctx, x, W(pre, ci + ".w"), W(pre, ci + ".b"), 1, 1, 0);
        }
        if (missing || x == nullptr) return nullptr;
        x = codec_op_pool_1d(ctx, x, GGML_OP_POOL_MAX, kPool, kPool, 0);
        const std::string ni = "norm" + std::to_string(i);
        x = instance_norm_tc(ctx, x, W(pre, ni + ".w"), W(pre, ni + ".b"));
        return missing ? nullptr : codec_op_leaky_relu(ctx, x, kLeakySlope);
    };
    ggml_tensor * x = ggml_cont(ctx, instance_norm_tc(ctx, wav, W(pre, "wav_norm.w"), W(pre, "wav_norm.b")));
    {
        // The sinc filterbank runs on CPU: like a DFT, its quiet bands lose
        // precision in offloaded matmuls (see codec_graph_pin_cpu), and the
        // InstanceNorm after it amplifies them.
        codec_graph_cpu_scope cpu(ctx);
        x = sinc_stage(0, x);
    }
    for (int i = 1; i < 3 && x != nullptr; ++i) {
        x = sinc_stage(i, x);
    }
    if (x == nullptr) return nullptr;

    // TDNN stack: conv → LeakyReLU → BatchNorm (inference affine).
    for (int i = 0; i < 5; ++i) {
        const std::string ti = "tdnn" + std::to_string(i);
        x = codec_conv1d(ctx, x, W(pre, ti + ".w"), W(pre, ti + ".b"), 1, kTdnnDilation[i], 0);
        if (missing || x == nullptr) return nullptr;
        x = codec_op_leaky_relu(ctx, x, kLeakySlope);
        ggml_tensor * s = W(pre, ti + ".bn.s");
        ggml_tensor * t = W(pre, ti + ".bn.t");
        if (missing) return nullptr;
        x = ggml_add(ctx, ggml_mul(ctx, x, ggml_reshape_2d(ctx, s, 1, s->ne[0])), ggml_reshape_2d(ctx, t, 1, t->ne[0]));
    }

    // Statistics pooling (mean, unbiased std) → Linear(3000, 512).
    const int64_t t = x->ne[0];
    ggml_tensor * mean = ggml_mean(ctx, x);                                         // [1, c]
    ggml_tensor * var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, ggml_sub(ctx, x, mean))),
                                   1.0f / (float) (t - 1));
    ggml_tensor * stats = ggml_concat(ctx, mean, ggml_sqrt(ctx, var), 1);           // [1, 2c]
    stats = ggml_reshape_1d(ctx, ggml_cont(ctx, stats), stats->ne[1]);
    ggml_tensor * emb = codec_op_linear(ctx, stats, W(pre, "emb.w"), W(pre, "emb.b"));
    if (missing || emb == nullptr) return nullptr;

    if (proj_prefix != nullptr) {
        // L2-normalise (F.normalize ≡ rms_norm / sqrt(N)), Linear, LayerNorm.
        const float n = (float) emb->ne[0];
        emb = ggml_scale(ctx, ggml_rms_norm(ctx, emb, 1e-12f), 1.0f / std::sqrt(n));
        emb = codec_op_linear(ctx, emb, W(proj_prefix, "w"), W(proj_prefix, "b"));
        if (missing || emb == nullptr) return nullptr;
        emb = ggml_norm(ctx, emb, kNormEps);
        emb = ggml_add(ctx, ggml_mul(ctx, emb, W(proj_prefix, "ln.w")), W(proj_prefix, "ln.b"));
        if (missing) return nullptr;
    }
    return ggml_cont(ctx, emb);
}
