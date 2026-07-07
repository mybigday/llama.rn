#include "pool1d.h"

#include "lm_ggml_ops.h"

#include <algorithm>

// Length-preserving (stride=1) max / avg pool with constant-zero padding,
// matching PyTorch `nn.MaxPool1d(k, padding=p)` and
// `nn.AvgPool1d(k, padding=p, count_include_pad=True)`.
//
// We can't use `lm_ggml_pool_1d`'s built-in padding directly because:
//   - `LM_GGML_OP_POOL_AVG` divides by the count of *valid* (non-padded) input
//     positions, i.e. `count_include_pad=False`, while every in-tree caller
//     wants the PyTorch default `count_include_pad=True` (divide by kernel).
//   - `LM_GGML_OP_POOL_MAX` skips padded positions entirely, so for input with
//     trained negative values the boundary windows would yield a different
//     answer than PyTorch's "treat padding as zero" semantics.
// Pre-padding the input with explicit zeros and then running pool with `p=0`
// makes both semantics line up: the kernel always covers `k` valid positions
// (so avg divides by `k`) and the explicit zeros participate as 0 in the max
// (matching PyTorch when the data is non-negative; in-tree callers feed
// `x_abs`).
static lm_ggml_tensor * codec_pool1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t pad,
    lm_ggml_op_pool op) {

    if (ctx == nullptr || x == nullptr) return nullptr;
    const int32_t k = std::max(1, kernel);
    const int32_t p = std::max(0, pad);

    lm_ggml_tensor * x_f32 = (x->type == LM_GGML_TYPE_F32) ? x : lm_ggml_cast(ctx, x, LM_GGML_TYPE_F32);
    lm_ggml_tensor * x_pad = (p > 0) ? codec_op_pad_1d(ctx, x_f32, p, p) : x_f32;
    if (x_pad == nullptr) return nullptr;

    lm_ggml_tensor * y = lm_ggml_pool_1d(ctx, x_pad, op, /*k0=*/k, /*s0=*/1, /*p0=*/0);
    return y ? lm_ggml_cont(ctx, y) : nullptr;
}

lm_ggml_tensor * codec_op_max_pool1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {
    return codec_pool1d(ctx, x, kernel, pad, LM_GGML_OP_POOL_MAX);
}

lm_ggml_tensor * codec_op_avg_pool1d(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {
    return codec_pool1d(ctx, x, kernel, pad, LM_GGML_OP_POOL_AVG);
}
