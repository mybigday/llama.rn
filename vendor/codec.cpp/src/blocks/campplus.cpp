#include "campplus.h"

#include "../ops/audio_frontend.h"
#include "../ops/conv1d.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <string>
#include <vector>

namespace {

constexpr int32_t kFeatDim = 80;
constexpr int32_t kHeadChannels = 32;
constexpr int32_t kHeadOutH = kFeatDim / 8;       // 80 → 40 → 20 → 10
constexpr int32_t kSegLen = 100;                  // CAMLayer.seg_pooling
constexpr int32_t kNumBlocks = 3;
constexpr int32_t kBlockLayers[kNumBlocks] = {12, 24, 16};
constexpr int32_t kBlockDilation[kNumBlocks] = {1, 2, 2};

int32_t campplus_tdnn_frames(int32_t t) {
    return (t + 2 * 2 - 5) / 2 + 1;   // Conv1d(k=5, stride=2, pad=2)
}

int32_t campplus_n_segments(int32_t t) {
    return (t + kSegLen - 1) / kSegLen;
}

// x [t, c] * scale[c] + shift[c] (inference BatchNorm) then ReLU.
ggml_tensor * bn_relu_tc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * s, ggml_tensor * t) {
    x = ggml_mul(ctx, x, ggml_reshape_2d(ctx, s, 1, s->ne[0]));
    x = ggml_add(ctx, x, ggml_reshape_2d(ctx, t, 1, t->ne[0]));
    return ggml_relu(ctx, x);
}

// 2D conv over [W=t, H=freq, C] with folded-BN bias.
ggml_tensor * conv2d_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                          int32_t stride_h, int32_t pad) {
    ggml_tensor * y = ggml_conv_2d(ctx, w, x, /*s0=*/1, /*s1=*/stride_h, pad, pad, 1, 1);
    return ggml_add(ctx, y, ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]));
}

// Context-aware masking pooling: per-channel mean over time + the mean of each
// 100-frame segment broadcast back over its frames (avg_pool1d, ceil_mode).
ggml_tensor * cam_context_tc(ggml_context * ctx, ggml_tensor * x, ggml_tensor * seg_inv_cnt) {
    const int64_t t = x->ne[0];
    const int64_t c = x->ne[1];
    const int64_t n_seg = seg_inv_cnt->ne[1];
    ggml_tensor * xp = ggml_pad(ctx, x, (int) (n_seg * kSegLen - t), 0, 0, 0);
    ggml_tensor * r = ggml_reshape_3d(ctx, xp, kSegLen, n_seg, c);
    ggml_tensor * seg = ggml_mul(ctx, ggml_sum_rows(ctx, r), seg_inv_cnt);          // [1, n_seg, c]
    seg = ggml_cont(ctx, ggml_repeat(ctx, seg, r));                                 // [100, n_seg, c]
    seg = ggml_reshape_2d(ctx, seg, n_seg * kSegLen, c);
    seg = ggml_cont(ctx, ggml_view_2d(ctx, seg, t, c, seg->nb[1], 0));             // [t, c]
    return ggml_add(ctx, seg, ggml_mean(ctx, x));
}

}  // namespace

bool codec_campplus_accepts(int64_t n_samples) {
    return campplus_tdnn_frames(codec_kaldi_fbank_frames(n_samples)) >= 2;
}

ggml_tensor * codec_block_campplus(
        ggml_context * ctx, const codec_model * model, const char * prefix, ggml_tensor * pcm16k,
        bool onnx_seg_pooling) {
    if (ctx == nullptr || model == nullptr || prefix == nullptr || pcm16k == nullptr ||
        !codec_campplus_accepts(ggml_nelements(pcm16k))) {
        return nullptr;
    }
    const std::string pre = std::string(prefix) + ".";
    bool missing = false;
    auto W = [&](const std::string & n) -> ggml_tensor * {
        ggml_tensor * t = codec_graph_weight(ctx, model, pre + n);
        if (t == nullptr) missing = true;
        return t;
    };

    // Feature map [W=t, H=80, C=1].
    ggml_tensor * fbank = codec_op_kaldi_fbank_cmn(ctx, pcm16k, W("fbank_mel"));
    if (missing || fbank == nullptr) return nullptr;
    const int32_t n_frames = (int32_t) fbank->ne[0];
    fbank = ggml_reshape_4d(ctx, fbank, n_frames, kFeatDim, 1, 1);

    // 1 / (frames in each 100-frame segment) after the stride-2 TDNN.
    const int32_t t2 = campplus_tdnn_frames(n_frames);
    const int32_t n_seg = campplus_n_segments(t2);
    std::vector<float> inv_cnt((size_t) n_seg, 1.0f / (float) kSegLen);
    if (!onnx_seg_pooling) inv_cnt.back() = 1.0f / (float) (t2 - (n_seg - 1) * kSegLen);
    const std::string seg_name = "lut." + std::string(prefix) + ".seg_inv_cnt";
    ggml_tensor * seg_inv_cnt = codec_graph_const_f32(ctx, seg_name.c_str(), inv_cnt, 1, n_seg, 1);
    if (seg_inv_cnt == nullptr) return nullptr;

    // FCM head: conv-bn-relu, 2 × 2 BasicResBlocks (stride 2 on freq), conv-bn-relu.
    ggml_tensor * x = ggml_relu(ctx, conv2d_bias(ctx, fbank, W("head.conv1.w"), W("head.conv1.b"), 1, 1));
    if (missing) return nullptr;
    for (int32_t li = 1; li <= 2; ++li) {
        for (int32_t bi = 0; bi < 2; ++bi) {
            const std::string bp = "head.l" + std::to_string(li) + "." + std::to_string(bi) + ".";
            const int32_t stride = bi == 0 ? 2 : 1;
            ggml_tensor * h = ggml_relu(ctx, conv2d_bias(ctx, x, W(bp + "conv1.w"), W(bp + "conv1.b"), stride, 1));
            if (missing) return nullptr;
            h = conv2d_bias(ctx, h, W(bp + "conv2.w"), W(bp + "conv2.b"), 1, 1);
            ggml_tensor * sc = x;
            if (stride != 1) {
                sc = conv2d_bias(ctx, x, W(bp + "sc.w"), W(bp + "sc.b"), stride, 0);
            }
            if (missing) return nullptr;
            x = ggml_relu(ctx, ggml_add(ctx, h, sc));
        }
    }
    x = ggml_relu(ctx, conv2d_bias(ctx, x, W("head.conv2.w"), W("head.conv2.b"), 2, 1));
    if (missing) return nullptr;
    // [t, 10, 32] → [t, 320] with channel index c * 10 + h (PyTorch reshape order).
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), n_frames, kHeadOutH * kHeadChannels);

    // x-vector: TDNN (k=5, stride 2) then 3 CAM dense blocks with transitions.
    x = ggml_relu(ctx, codec_conv1d(ctx, x, W("tdnn.w"), W("tdnn.b"), 2, 1, 2));
    if (missing || x == nullptr) return nullptr;
    for (int32_t bi = 0; bi < kNumBlocks; ++bi) {
        const int32_t dil = kBlockDilation[bi];
        for (int32_t li = 0; li < kBlockLayers[bi]; ++li) {
            const std::string lp = "b" + std::to_string(bi + 1) + "." + std::to_string(li) + ".";
            ggml_tensor * h = bn_relu_tc(ctx, x, W(lp + "bn.s"), W(lp + "bn.t"));
            if (missing) return nullptr;
            h = ggml_relu(ctx, codec_conv1d(ctx, h, W(lp + "lin1.w"), W(lp + "lin1.b"), 1, 1, 0));
            ggml_tensor * y = codec_conv1d(ctx, h, W(lp + "local.w"), nullptr, 1, dil, dil);
            ggml_tensor * c = cam_context_tc(ctx, h, seg_inv_cnt);
            c = ggml_relu(ctx, codec_conv1d(ctx, c, W(lp + "cl1.w"), W(lp + "cl1.b"), 1, 1, 0));
            ggml_tensor * m = ggml_sigmoid(ctx, codec_conv1d(ctx, c, W(lp + "cl2.w"), W(lp + "cl2.b"), 1, 1, 0));
            if (missing || y == nullptr || m == nullptr) return nullptr;
            x = ggml_concat(ctx, x, ggml_mul(ctx, y, m), 1);
        }
        const std::string tp = "tr" + std::to_string(bi + 1) + ".";
        ggml_tensor * h = bn_relu_tc(ctx, x, W(tp + "bn.s"), W(tp + "bn.t"));
        if (missing) return nullptr;
        x = codec_conv1d(ctx, h, W(tp + "w"), nullptr, 1, 1, 0);
        if (x == nullptr) return nullptr;
    }
    x = bn_relu_tc(ctx, x, W("out_bn.s"), W("out_bn.t"));
    if (missing) return nullptr;

    // Statistics pooling: [mean, unbiased std] over time, then dense + BN (folded).
    const int64_t t = x->ne[0];
    ggml_tensor * mean = ggml_mean(ctx, x);                                         // [1, c]
    ggml_tensor * d = ggml_sub(ctx, x, mean);
    ggml_tensor * var = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, d)), 1.0f / (float) (t - 1));
    ggml_tensor * stats = ggml_concat(ctx, mean, ggml_sqrt(ctx, var), 1);           // [1, 2c]
    ggml_tensor * emb = codec_conv1d(ctx, stats, W("dense.w"), W("dense.b"), 1, 1, 0);  // [1, 192]
    if (missing || emb == nullptr) return nullptr;
    return ggml_reshape_1d(ctx, ggml_cont(ctx, emb), kCodecCampplusEmbedDim);
}
