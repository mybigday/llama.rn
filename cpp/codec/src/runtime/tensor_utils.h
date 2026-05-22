#ifndef CODEC_RUNTIME_TENSOR_UTILS_H
#define CODEC_RUNTIME_TENSOR_UTILS_H

#include "../codec_internal.h"

bool codec_runtime_write_tensor(lm_ggml_tensor * t, const void * data, size_t n_bytes, std::string * error);
bool codec_runtime_read_tensor(lm_ggml_tensor * t, void * data, size_t n_bytes, std::string * error);

#endif
