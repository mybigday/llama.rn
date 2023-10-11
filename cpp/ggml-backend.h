#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif
    struct lm_ggml_backend;
    struct lm_ggml_backend_buffer;

    // type-erased backend-specific types / wrappers
    typedef void * lm_ggml_backend_context_t;
    typedef void * lm_ggml_backend_graph_plan_t;
    typedef void * lm_ggml_backend_buffer_context_t;

    // avoid accessing internals of these types
    typedef struct lm_ggml_backend        * lm_ggml_backend_t;
    typedef struct lm_ggml_backend_buffer * lm_ggml_backend_buffer_t;

    //
    // backend buffer
    //

    struct lm_ggml_backend_buffer_i {
        void   (*free_buffer)   (lm_ggml_backend_buffer_t buffer);
        void * (*get_base)      (lm_ggml_backend_buffer_t buffer); // get base pointer
        size_t (*get_alloc_size)(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor); // pre-allocation callback
        void   (*init_tensor)   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor); // post-allocation callback
        void   (*free_tensor)   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor); // pre-free callback
    };

    // TODO: hide behind API
    struct lm_ggml_backend_buffer {
        struct lm_ggml_backend_buffer_i iface;

        lm_ggml_backend_t                backend;
        lm_ggml_backend_buffer_context_t context;

        size_t size;
    };

    // backend buffer functions
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_buffer_init(
            struct lm_ggml_backend                  * backend,
            struct lm_ggml_backend_buffer_i           iface,
                   lm_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    LM_GGML_API void   lm_ggml_backend_buffer_free          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_alignment (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void * lm_ggml_backend_buffer_get_base      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_size      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_alloc_size(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API void   lm_ggml_backend_buffer_init_tensor   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API void   lm_ggml_backend_buffer_free_tensor   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);

    //
    // backend
    //

    struct lm_ggml_backend_i {
        const char * (*get_name)(lm_ggml_backend_t backend);

        void (*free)(lm_ggml_backend_t backend);

        // buffer allocation
        lm_ggml_backend_buffer_t (*alloc_buffer)(lm_ggml_backend_t backend, size_t size);

        // get buffer alignment
        size_t (*get_alignment)(lm_ggml_backend_t backend);

        // tensor data access
        // these functions can be asynchronous, helper functions are provided for synchronous access that automatically call synchronize
        void (*set_tensor_async)(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*synchronize)     (lm_ggml_backend_t backend);

        // (optional) copy tensor between different backends, allow for single-copy tranfers
        void (*cpy_tensor_from)(lm_ggml_backend_t backend, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);
        void (*cpy_tensor_to)  (lm_ggml_backend_t backend, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

        // compute graph with a plan
        lm_ggml_backend_graph_plan_t (*graph_plan_create) (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

        // compute graph without a plan
        void (*graph_compute)(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);
    };

    // TODO: hide behind API
    struct lm_ggml_backend {
        struct lm_ggml_backend_i iface;

        lm_ggml_backend_context_t context;
    };

    // backend helper functions
    LM_GGML_API lm_ggml_backend_t lm_ggml_get_backend(const struct lm_ggml_tensor * tensor);

    LM_GGML_API const char * lm_ggml_backend_name(lm_ggml_backend_t backend);
    LM_GGML_API void         lm_ggml_backend_free(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_alloc_buffer(lm_ggml_backend_t backend, size_t size);

    LM_GGML_API size_t lm_ggml_backend_get_alignment(lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_tensor_set_async(      struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get_async(const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    LM_GGML_API void lm_ggml_backend_tensor_set(      struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get(const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    LM_GGML_API void lm_ggml_backend_synchronize(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_graph_plan_t lm_ggml_backend_graph_plan_create (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

    LM_GGML_API void lm_ggml_backend_graph_plan_free   (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
    LM_GGML_API void lm_ggml_backend_graph_plan_compute(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
    LM_GGML_API void lm_ggml_backend_graph_compute     (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API bool lm_ggml_backend_supports_op       (lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);

    // tensor copy between different backends
    LM_GGML_API void lm_ggml_backend_tensor_copy(struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    //
    // CPU backend
    //

    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_cpu_init(void);

    LM_GGML_API bool lm_ggml_backend_is_cpu(lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_cpu_set_n_threads(lm_ggml_backend_t backend_cpu, int n_threads);

    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_cpu_buffer_from_ptr(lm_ggml_backend_t backend_cpu, void * ptr, size_t size);

#ifdef  __cplusplus
}
#endif
