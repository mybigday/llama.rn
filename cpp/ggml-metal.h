// An interface allowing to compute lm_ggml_cgraph with Metal
//
// This is a fully functional interface that extends ggml with GPU support for Apple devices.
// A similar interface can be created for other GPU backends (e.g. Vulkan, CUDA, OpenCL, etc.)
//
// How it works?
//
// As long as your program can create and evaluate a lm_ggml_cgraph on the CPU, you can use this
// interface to evaluate the same graph on the GPU. Instead of using lm_ggml_graph_compute(), you
// use lm_ggml_metal_graph_compute() (or lm_ggml_vulkan_graph_compute(), etc.)
//
// You only need to make sure that all memory buffers that you used during the graph creation
// are mapped to the device memory with the lm_ggml_metal_add_buffer() function. This mapping is
// used during the graph evaluation to determine the arguments of the compute kernels.
//
// Synchronization between device and host memory (for example for input and output tensors)
// is done with the lm_ggml_metal_set_tensor() and lm_ggml_metal_get_tensor() functions.
//

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

// max memory buffers that can be mapped to the device
#define LM_GGML_METAL_MAX_BUFFERS 64

struct lm_ggml_tensor;
struct lm_ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// backend API
// user-code should use only these functions
//

LM_GGML_API void lm_ggml_backend_metal_log_set_callback(lm_ggml_log_callback log_callback, void * user_data);

LM_GGML_API lm_ggml_backend_t lm_ggml_backend_metal_init(void);

LM_GGML_API bool lm_ggml_backend_is_metal(lm_ggml_backend_t backend);

LM_GGML_API LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_from_ptr(void * data, size_t size, size_t max_size);

LM_GGML_API void lm_ggml_backend_metal_set_n_cb(lm_ggml_backend_t backend, int n_cb);

LM_GGML_API LM_GGML_CALL lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type(void);

// helper to check if the device supports a specific family
// ideally, the user code should be doing these checks
// ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
LM_GGML_API bool lm_ggml_backend_metal_supports_family(lm_ggml_backend_t backend, int family);

// capture all command buffers committed the next time `lm_ggml_backend_graph_compute` is called
LM_GGML_API void lm_ggml_backend_metal_capture_next_compute(lm_ggml_backend_t backend);

#ifdef __cplusplus
}
#endif

