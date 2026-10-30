#include "pool1d.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"

#include <algorithm>

ggml_tensor * codec_op_pool_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    enum ggml_op_pool op,
    int32_t k,
    int32_t s,
    int32_t p) {

    if (ctx == nullptr || x == nullptr || x->ne[3] > 1) return nullptr;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    ggml_tensor * y = ggml_pool_2d(ctx, x4, op, k, 1, s, 1, (float) p, 0.0f);   // [t_out, 1, c, n]
    if (!codec_graph_backend_supports(y) && op == GGML_OP_POOL_AVG && k == s && p == 0) {
        // Non-overlapping average (ggml-opencl has no POOL_*): row sums of
        // [k, t/k, c] blocks, / k.
        const int64_t t_out = x->ne[0] / k;
        ggml_tensor * xs = x->ne[0] == t_out * k ? x : ggml_cont(ctx, ggml_view_3d(ctx, x, t_out * k, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0));
        ggml_tensor * sums = ggml_sum_rows(ctx, ggml_reshape_4d(ctx, xs, k, t_out, x->ne[1], x->ne[2]));   // [1, t/k, c, n]
        return ggml_reshape_3d(ctx, ggml_scale(ctx, sums, 1.0f / (float) k), t_out, x->ne[1], x->ne[2]);
    }
    return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], y->ne[3]);
}

// Length-preserving (stride=1) max / avg pool with constant-zero padding,
// matching PyTorch `nn.MaxPool1d(k, padding=p)` and
// `nn.AvgPool1d(k, padding=p, count_include_pad=True)`.
//
// We can't use `ggml_pool_1d`'s built-in padding directly because:
//   - `GGML_OP_POOL_AVG` divides by the count of *valid* (non-padded) input
//     positions, i.e. `count_include_pad=False`, while every in-tree caller
//     wants the PyTorch default `count_include_pad=True` (divide by kernel).
//   - `GGML_OP_POOL_MAX` skips padded positions entirely, so for input with
//     trained negative values the boundary windows would yield a different
//     answer than PyTorch's "treat padding as zero" semantics.
// Pre-padding the input with explicit zeros and then running pool with `p=0`
// makes both semantics line up: the kernel always covers `k` valid positions
// (so avg divides by `k`) and the explicit zeros participate as 0 in the max
// (matching PyTorch when the data is non-negative; in-tree callers feed
// `x_abs`).
static ggml_tensor * codec_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad,
    ggml_op_pool op) {

    if (ctx == nullptr || x == nullptr) return nullptr;
    const int32_t k = std::max(1, kernel);
    const int32_t p = std::max(0, pad);

    ggml_tensor * x_f32 = (x->type == GGML_TYPE_F32) ? x : ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor * x_pad = (p > 0) ? codec_op_pad_1d(ctx, x_f32, p, p) : x_f32;
    if (x_pad == nullptr) return nullptr;

    ggml_tensor * y = codec_op_pool_1d(ctx, x_pad, op, /*k=*/k, /*s=*/1, /*p=*/0);
    return y ? ggml_cont(ctx, y) : nullptr;
}

ggml_tensor * codec_op_max_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {
    return codec_pool1d(ctx, x, kernel, pad, GGML_OP_POOL_MAX);
}

ggml_tensor * codec_op_avg_pool1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int32_t kernel,
    int32_t pad) {
    return codec_pool1d(ctx, x, kernel, pad, GGML_OP_POOL_AVG);
}
