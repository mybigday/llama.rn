#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    //
    // Backend buffer
    //

    // buffer type
    typedef void * lm_ggml_backend_buffer_type_context_t;

    struct lm_ggml_backend_buffer_type_i {
        const char *          (*get_name)        (lm_ggml_backend_buffer_type_t buft);
        lm_ggml_backend_buffer_t (*alloc_buffer)    (lm_ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*get_alignment)   (lm_ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*get_alloc_size)  (lm_ggml_backend_buffer_type_t buft, const struct lm_ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*supports_backend)(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend); // check if the buffer type is usable by the backend
        // check if tensor data is in host memory
        // should be equivalent to supports_backend(buft, lm_ggml_backend_cpu_init())
        bool                  (*is_host)         (lm_ggml_backend_buffer_type_t buft);
    };

    struct lm_ggml_backend_buffer_type {
        struct lm_ggml_backend_buffer_type_i  iface;
        lm_ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * lm_ggml_backend_buffer_context_t;

    struct lm_ggml_backend_buffer_i {
        const char * (*get_name)   (lm_ggml_backend_buffer_t buffer);
        void         (*free_buffer)(lm_ggml_backend_buffer_t buffer);
        void *       (*get_base)   (lm_ggml_backend_buffer_t buffer);
        void         (*init_tensor)(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
        void         (*set_tensor) (lm_ggml_backend_buffer_t buffer,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor) (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*cpy_tensor) (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*clear)      (lm_ggml_backend_buffer_t buffer, uint8_t value);
        void         (*reset)      (lm_ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct lm_ggml_backend_buffer {
        struct lm_ggml_backend_buffer_i  iface;
        lm_ggml_backend_buffer_type_t    buft;
        lm_ggml_backend_buffer_context_t context;
        size_t size;
        enum lm_ggml_backend_buffer_usage usage;
    };

    lm_ggml_backend_buffer_t lm_ggml_backend_buffer_init(
                   lm_ggml_backend_buffer_type_t      buft,
            struct lm_ggml_backend_buffer_i           iface,
                   lm_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use lm_ggml_backend_tensor_copy instead
    bool lm_ggml_backend_buffer_copy_tensor(const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    //
    // Backend
    //

    typedef void * lm_ggml_backend_context_t;

    struct lm_ggml_backend_i {
        const char * (*get_name)(lm_ggml_backend_t backend);

        void (*free)(lm_ggml_backend_t backend);

        // buffer allocation
        lm_ggml_backend_buffer_type_t (*get_default_buffer_type)(lm_ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*synchronize)(lm_ggml_backend_t backend);

        // compute graph with a plan
        lm_ggml_backend_graph_plan_t (*graph_plan_create) (lm_ggml_backend_t backend, const struct lm_ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
        void                      (*graph_plan_compute)(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

        // compute graph without a plan (async)
        bool (*graph_compute)(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*supports_op)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);
    };

    struct lm_ggml_backend {
        struct lm_ggml_backend_i iface;

        lm_ggml_backend_context_t context;
    };

    //
    // Backend registry
    //

    typedef lm_ggml_backend_t (*lm_ggml_backend_init_fn)(const char * params, void * user_data);

    void lm_ggml_backend_register(const char * name, lm_ggml_backend_init_fn init_fn, lm_ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
