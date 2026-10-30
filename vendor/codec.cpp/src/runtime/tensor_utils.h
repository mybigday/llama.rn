#ifndef CODEC_RUNTIME_TENSOR_UTILS_H
#define CODEC_RUNTIME_TENSOR_UTILS_H

#include "../codec_internal.h"

bool codec_runtime_write_tensor(ggml_tensor * t, const void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor(ggml_tensor * t, void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor_i32_2d_tq(ggml_tensor * t, std::vector<int32_t> * out, std::string * error);

// Returns the raw loaded GGUF weight tensor for `name` (no graph cast). Returns
// nullptr if the tensor is missing or the model is not loaded. Used for runtime
// metadata reads (STFT windows, mel filter banks, codebook lookups by index).
ggml_tensor * codec_model_get_tensor(const codec_model * model, const char * name);
ggml_tensor * codec_model_get_tensor(const codec_model * model, const std::string & name);

// Returns the loaded GGUF weight tensor for `name` from model->weights, with a
// graph-level cast to F32 if the stored type is not F32, for elementwise use
// (biases, norm scales: a lossless widening).  Matmul / conv ops given this
// tensor recover the stored weight (codec_graph_stored_weight) and run at its
// type.  Returns nullptr if the tensor is missing or the model is not loaded.
ggml_tensor * codec_graph_weight(ggml_context * ctx_eval, const codec_model * model, const char * name);
ggml_tensor * codec_graph_weight(ggml_context * ctx_eval, const codec_model * model, const std::string & name);

// Same as codec_graph_weight, but returns nullptr silently if the tensor is
// absent (used when a weight is optional, e.g. a bias that may be missing).
ggml_tensor * codec_graph_weight_or_null(ggml_context * ctx_eval, const codec_model * model, const char * name);
ggml_tensor * codec_graph_weight_or_null(ggml_context * ctx_eval, const codec_model * model, const std::string & name);

// Cast a tensor to F32 in the graph if it isn't already.
ggml_tensor * codec_graph_cast_f32(ggml_context * ctx_eval, ggml_tensor * t);

// Weights are computed at their stored GGUF type (F32 / F16 / BF16 / block-
// quantized): matmul / conv / codebook operands never get cast.  The helpers below
// hand those ops the stored tensor.
//
// The stored weight behind `t` when `t` is (a reshape of) the F32 cast that
// codec_graph_weight / codec_graph_cast_f32 put on it; `t` itself otherwise.
ggml_tensor * codec_graph_stored_weight(ggml_context * ctx_eval, ggml_tensor * t);

// mul_mat src[0] side: the stored weight (see codec_graph_stored_weight).
ggml_tensor * codec_graph_mat_lhs(ggml_context * ctx_eval, ggml_tensor * t);

// Fetch a matmul weight (src[0] side) at its stored type.
ggml_tensor * codec_graph_weight_mat(ggml_context * ctx_eval, const codec_model * model, const char * name);
ggml_tensor * codec_graph_weight_mat(ggml_context * ctx_eval, const codec_model * model, const std::string & name);

// ggml_mul_mat with `a` at its stored type (codec_graph_stored_weight), F32
// accumulation.  ggml converts `b` to the weight's dot type (F16 for an F16
// weight, Q8 for most quantized ones): that is the weight's precision.
ggml_tensor * codec_mul_mat(ggml_context * ctx_eval, ggml_tensor * a, ggml_tensor * b);

#endif
