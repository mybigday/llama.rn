#ifndef CODEC_RUNTIME_TENSOR_UTILS_H
#define CODEC_RUNTIME_TENSOR_UTILS_H

#include "../codec_internal.h"

bool codec_runtime_write_tensor(lm_ggml_tensor * t, const void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor(lm_ggml_tensor * t, void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor_i32_2d_tq(lm_ggml_tensor * t, std::vector<int32_t> * out, std::string * error);

// Returns the raw loaded GGUF weight tensor for `name` (no graph cast). Returns
// nullptr if the tensor is missing or the model is not loaded. Used for runtime
// metadata reads (STFT windows, mel filter banks, codebook lookups by index).
lm_ggml_tensor * codec_model_get_tensor(const codec_model * model, const char * name);
lm_ggml_tensor * codec_model_get_tensor(const codec_model * model, const std::string & name);

// Returns the loaded GGUF weight tensor for `name` from model->weights, with a
// graph-level cast to F32 if the stored type is not F32. The tensor is intended
// to be referenced directly in the graph (no per-call CPU dequantize). Returns
// nullptr if the tensor is missing or the model is not loaded.
lm_ggml_tensor * codec_graph_weight(lm_ggml_context * ctx_eval, const codec_model * model, const char * name);
lm_ggml_tensor * codec_graph_weight(lm_ggml_context * ctx_eval, const codec_model * model, const std::string & name);

// Same as codec_graph_weight, but returns nullptr silently if the tensor is
// absent (used when a weight is optional, e.g. a bias that may be missing).
lm_ggml_tensor * codec_graph_weight_or_null(lm_ggml_context * ctx_eval, const codec_model * model, const char * name);
lm_ggml_tensor * codec_graph_weight_or_null(lm_ggml_context * ctx_eval, const codec_model * model, const std::string & name);

// Cast a tensor to F32 in the graph if it isn't already.
lm_ggml_tensor * codec_graph_cast_f32(lm_ggml_context * ctx_eval, lm_ggml_tensor * t);

// Pass-through wrapper intended for tensors that will land as the LHS
// (src[0], the weight side) of lm_ggml_mul_mat.  lm_ggml_mul_mat handles F32 /
// F16 / BF16 src[0] with an F32 src[1] natively via fused vec_dot
// kernels — wrapping the weight in lm_ggml_cast(.., F32) just bakes an
// extra dequant op into the graph that runs every execution, wasting
// memory bandwidth proportional to the weight size.
//
// Truly quantized types (Q4_K, Q5_K, Q8_0, …) still get cast to F32
// because the on-the-fly per-block re-quant of the F32 activation
// inside their mul_mat kernel adds enough compute that an explicit
// cast (which can be tiled / fused) is the lesser evil.  Most codec_lm
// GGUFs ship F16 weights, so this is the hot path.
lm_ggml_tensor * codec_graph_mat_lhs(lm_ggml_context * ctx_eval, lm_ggml_tensor * t);

#endif
