#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct lm_ggml_backend;
struct lm_ggml_backend_buffer;
struct lm_ggml_backend_buffer_type;

//
// Legacy API
//

typedef struct lm_ggml_allocr * lm_ggml_allocr_t;

// initialize allocator for use with CPU backend only
LM_GGML_API lm_ggml_allocr_t lm_ggml_allocr_new(void * data, size_t size, size_t alignment);
LM_GGML_API lm_ggml_allocr_t lm_ggml_allocr_new_measure(size_t alignment);

// initialize allocator for use with ggml-backend
LM_GGML_API lm_ggml_allocr_t lm_ggml_allocr_new_from_buffer(struct lm_ggml_backend_buffer * buffer);
LM_GGML_API lm_ggml_allocr_t lm_ggml_allocr_new_from_backend(struct lm_ggml_backend * backend, size_t size); // allocates an owned buffer
LM_GGML_API lm_ggml_allocr_t lm_ggml_allocr_new_measure_from_backend(struct lm_ggml_backend * backend);

LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_allocr_get_buffer(lm_ggml_allocr_t alloc);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
LM_GGML_API void   lm_ggml_allocr_set_parse_seq(lm_ggml_allocr_t alloc, const int * list, int n);

LM_GGML_API void   lm_ggml_allocr_free       (lm_ggml_allocr_t alloc);
LM_GGML_API bool   lm_ggml_allocr_is_measure (lm_ggml_allocr_t alloc);
LM_GGML_API void   lm_ggml_allocr_reset      (lm_ggml_allocr_t alloc);
LM_GGML_API void   lm_ggml_allocr_alloc      (lm_ggml_allocr_t alloc, struct lm_ggml_tensor * tensor);
LM_GGML_API size_t lm_ggml_allocr_max_size   (lm_ggml_allocr_t alloc);

LM_GGML_API size_t lm_ggml_allocr_alloc_graph(lm_ggml_allocr_t alloc, struct lm_ggml_cgraph * graph);

//
// ggml-backend v2 API
//

// Separate tensor and graph allocator objects
// This is necessary for multi-backend allocation because the graph allocator needs to use multiple tensor allocators
// The original API is kept as a wrapper around the new API

// Tensor allocator
typedef struct lm_ggml_tallocr * lm_ggml_tallocr_t;

LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new(void * data, size_t size, size_t alignment);
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_measure(size_t alignment);
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_from_buft(struct lm_ggml_backend_buffer_type * buft, size_t size);
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_from_backend(struct lm_ggml_backend * backend, size_t size); // allocates an owned buffer
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_from_buffer(struct lm_ggml_backend_buffer * buffer);
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_measure_from_buft(struct lm_ggml_backend_buffer_type * buft);
LM_GGML_API lm_ggml_tallocr_t lm_ggml_tallocr_new_measure_from_backend(struct lm_ggml_backend * backend);

LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_tallocr_get_buffer(lm_ggml_tallocr_t talloc);

LM_GGML_API void   lm_ggml_tallocr_free       (lm_ggml_tallocr_t talloc);
LM_GGML_API bool   lm_ggml_tallocr_is_measure (lm_ggml_tallocr_t talloc);
LM_GGML_API void   lm_ggml_tallocr_reset      (lm_ggml_tallocr_t talloc);
LM_GGML_API void   lm_ggml_tallocr_alloc      (lm_ggml_tallocr_t talloc, struct lm_ggml_tensor * tensor);
LM_GGML_API size_t lm_ggml_tallocr_max_size   (lm_ggml_tallocr_t talloc);


// Graph allocator
typedef struct lm_ggml_gallocr * lm_ggml_gallocr_t;

LM_GGML_API lm_ggml_gallocr_t lm_ggml_gallocr_new(void);
LM_GGML_API void   lm_ggml_gallocr_free(lm_ggml_gallocr_t galloc);

LM_GGML_API void   lm_ggml_gallocr_set_parse_seq(lm_ggml_gallocr_t galloc, const int * list, int n);
LM_GGML_API size_t lm_ggml_gallocr_alloc_graph(lm_ggml_gallocr_t galloc, lm_ggml_tallocr_t talloc, struct lm_ggml_cgraph * graph);

// Allocate tensors from the allocators given by the hash table
LM_GGML_API void   lm_ggml_gallocr_alloc_graph_n(
                    lm_ggml_gallocr_t galloc,
                    struct lm_ggml_cgraph * graph,
                    struct lm_ggml_hash_set hash_set,
                    lm_ggml_tallocr_t * hash_node_talloc);


// Utils
// Create a buffer and allocate all the tensors in a lm_ggml_context
LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_backend_alloc_ctx_tensors_from_buft(struct lm_ggml_context * ctx, struct lm_ggml_backend_buffer_type * buft);
LM_GGML_API struct lm_ggml_backend_buffer * lm_ggml_backend_alloc_ctx_tensors(struct lm_ggml_context * ctx, struct lm_ggml_backend * backend);

#ifdef  __cplusplus
}
#endif
