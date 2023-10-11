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
#define LM_GGML_METAL_MAX_BUFFERS 16
#define LM_GGML_METAL_MAX_COMMAND_BUFFERS 32

struct lm_ggml_tensor;
struct lm_ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

//
// internal API
// temporary exposed to user-code
//

struct lm_ggml_metal_context;

void lm_ggml_metal_log_set_callback(lm_ggml_log_callback log_callback, void * user_data);

// number of command buffers to use
struct lm_ggml_metal_context * lm_ggml_metal_init(int n_cb);
void lm_ggml_metal_free(struct lm_ggml_metal_context * ctx);

void * lm_ggml_metal_host_malloc(size_t n);
void   lm_ggml_metal_host_free  (void * data);

// set the number of command buffers to use
void lm_ggml_metal_set_n_cb(struct lm_ggml_metal_context * ctx, int n_cb);

// creates a mapping between a host memory buffer and a device memory buffer
// - make sure to map all buffers used in the graph before calling lm_ggml_metal_graph_compute
// - the mapping is used during computation to determine the arguments of the compute kernels
// - you don't need to keep the host memory buffer allocated as it is never accessed by Metal
// - max_size specifies the maximum size of a tensor and is used to create shared views such
//   that it is guaranteed that the tensor will fit in at least one of the views
//
bool lm_ggml_metal_add_buffer(
        struct lm_ggml_metal_context * ctx,
                       const char * name,
                             void * data,
                           size_t   size,
                           size_t   max_size);

// set data from host memory into the device
void lm_ggml_metal_set_tensor(struct lm_ggml_metal_context * ctx, struct lm_ggml_tensor * t);

// get data from the device into host memory
void lm_ggml_metal_get_tensor(struct lm_ggml_metal_context * ctx, struct lm_ggml_tensor * t);

// try to find operations that can be run concurrently in the graph
// you should run it again if the topology of your graph changes
void lm_ggml_metal_graph_find_concurrency(struct lm_ggml_metal_context * ctx, struct lm_ggml_cgraph * gf, bool check_mem);

// if the graph has been optimized for concurrently dispatch, return length of the concur_list if optimized
int lm_ggml_metal_if_optimized(struct lm_ggml_metal_context * ctx);

// output the concur_list for lm_ggml_alloc
int * lm_ggml_metal_get_concur_list(struct lm_ggml_metal_context * ctx);

// same as lm_ggml_graph_compute but uses Metal
// creates gf->n_threads command buffers in parallel
void lm_ggml_metal_graph_compute(struct lm_ggml_metal_context * ctx, struct lm_ggml_cgraph * gf);

//
// backend API
// user-code should use only these functions
//

LM_GGML_API lm_ggml_backend_t lm_ggml_backend_metal_init(void);

LM_GGML_API bool lm_ggml_backend_is_metal(lm_ggml_backend_t backend);

LM_GGML_API void lm_ggml_backend_metal_set_n_cb(lm_ggml_backend_t backend, int n_cb);

#ifdef __cplusplus
}
#endif

