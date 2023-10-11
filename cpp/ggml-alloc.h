#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

struct lm_ggml_backend_buffer;

LM_GGML_API struct lm_ggml_allocr * lm_ggml_allocr_new(void * data, size_t size, size_t alignment);
LM_GGML_API struct lm_ggml_allocr * lm_ggml_allocr_new_measure(size_t alignment);
LM_GGML_API struct lm_ggml_allocr * lm_ggml_allocr_new_from_buffer(struct lm_ggml_backend_buffer * buffer);

// tell the allocator to parse nodes following the order described in the list
// you should call this if your graph are optimized to execute out-of-order
LM_GGML_API void   lm_ggml_allocr_set_parse_seq(struct lm_ggml_allocr * alloc, const int * list, int n);

LM_GGML_API void   lm_ggml_allocr_free       (struct lm_ggml_allocr * alloc);
LM_GGML_API bool   lm_ggml_allocr_is_measure (struct lm_ggml_allocr * alloc);
LM_GGML_API void   lm_ggml_allocr_reset      (struct lm_ggml_allocr * alloc);
LM_GGML_API void   lm_ggml_allocr_alloc      (struct lm_ggml_allocr * alloc, struct lm_ggml_tensor * tensor);
LM_GGML_API size_t lm_ggml_allocr_alloc_graph(struct lm_ggml_allocr * alloc, struct lm_ggml_cgraph * graph);
LM_GGML_API size_t lm_ggml_allocr_max_size   (struct lm_ggml_allocr * alloc);

LM_GGML_API size_t lm_ggml_allocr_alloc_graph_n(
                    struct lm_ggml_allocr * alloc,
                    struct lm_ggml_cgraph ** graphs, int n_graphs,
                    struct lm_ggml_tensor *** inputs, struct lm_ggml_tensor *** outputs);

#ifdef  __cplusplus
}
#endif
