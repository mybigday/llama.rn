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
        const char *          (*LM_GGML_CALL get_name)        (lm_ggml_backend_buffer_type_t buft);
        lm_ggml_backend_buffer_t (*LM_GGML_CALL alloc_buffer)    (lm_ggml_backend_buffer_type_t buft, size_t size);
        size_t                (*LM_GGML_CALL get_alignment)   (lm_ggml_backend_buffer_type_t buft); // tensor alignment
        size_t                (*LM_GGML_CALL get_max_size)    (lm_ggml_backend_buffer_type_t buft); // allocation max size
        size_t                (*LM_GGML_CALL get_alloc_size)  (lm_ggml_backend_buffer_type_t buft, const struct lm_ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
        bool                  (*LM_GGML_CALL supports_backend)(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend); // check if the buffer type is usable by the backend
        // check if tensor data is in host memory
        // should be equivalent to supports_backend(buft, lm_ggml_backend_cpu_init())
        bool                  (*LM_GGML_CALL is_host)         (lm_ggml_backend_buffer_type_t buft);
    };

    struct lm_ggml_backend_buffer_type {
        struct lm_ggml_backend_buffer_type_i  iface;
        lm_ggml_backend_buffer_type_context_t context;
    };

    // buffer
    typedef void * lm_ggml_backend_buffer_context_t;

    struct lm_ggml_backend_buffer_i {
        const char * (*LM_GGML_CALL get_name)   (lm_ggml_backend_buffer_t buffer);
        void         (*LM_GGML_CALL free_buffer)(lm_ggml_backend_buffer_t buffer);
        void *       (*LM_GGML_CALL get_base)   (lm_ggml_backend_buffer_t buffer);
        void         (*LM_GGML_CALL init_tensor)(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
        void         (*LM_GGML_CALL set_tensor) (lm_ggml_backend_buffer_t buffer,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*LM_GGML_CALL get_tensor) (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool         (*LM_GGML_CALL cpy_tensor) (lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
        void         (*LM_GGML_CALL clear)      (lm_ggml_backend_buffer_t buffer, uint8_t value);
        void         (*LM_GGML_CALL reset)      (lm_ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
    };

    struct lm_ggml_backend_buffer {
        struct lm_ggml_backend_buffer_i  iface;
        lm_ggml_backend_buffer_type_t    buft;
        lm_ggml_backend_buffer_context_t context;
        size_t size;
        enum lm_ggml_backend_buffer_usage usage;
    };

    LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_buffer_init(
                   lm_ggml_backend_buffer_type_t      buft,
            struct lm_ggml_backend_buffer_i           iface,
                   lm_ggml_backend_buffer_context_t   context,
                   size_t                          size);

    // do not use directly, use lm_ggml_backend_tensor_copy instead
    bool lm_ggml_backend_buffer_copy_tensor(const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    // buffer that contains a collection of buffers
    LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_multi_buffer_alloc_buffer(lm_ggml_backend_buffer_t * buffers, size_t n_buffers);
    LM_GGML_CALL bool                  lm_ggml_backend_buffer_is_multi_buffer(lm_ggml_backend_buffer_t buffer);
    LM_GGML_CALL void                  lm_ggml_backend_multi_buffer_set_usage(lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage);

    //
    // Backend
    //

    typedef void * lm_ggml_backend_context_t;

    struct lm_ggml_backend_i {
        const char * (*LM_GGML_CALL get_name)(lm_ggml_backend_t backend);

        void (*LM_GGML_CALL free)(lm_ggml_backend_t backend);

        // buffer allocation
        lm_ggml_backend_buffer_type_t (*LM_GGML_CALL get_default_buffer_type)(lm_ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*LM_GGML_CALL set_tensor_async)(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*LM_GGML_CALL get_tensor_async)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*LM_GGML_CALL cpy_tensor_async)(lm_ggml_backend_t backend_src, lm_ggml_backend_t backend_dst, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

        // (optional) complete all pending operations
        void (*LM_GGML_CALL synchronize)(lm_ggml_backend_t backend);

        // compute graph with a plan (not used currently)
        lm_ggml_backend_graph_plan_t (*LM_GGML_CALL graph_plan_create) (lm_ggml_backend_t backend, const struct lm_ggml_cgraph * cgraph);
        void                      (*LM_GGML_CALL graph_plan_free)   (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

        // compute graph with a plan
        enum lm_ggml_status (*LM_GGML_CALL graph_plan_compute)(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
        // compute graph without a plan (async)
        enum lm_ggml_status (*LM_GGML_CALL graph_compute)     (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);

        // check if the backend supports an operation
        bool (*LM_GGML_CALL supports_op)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);

        // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
        // these should be expensive operations with large batch sizes that may benefit from running on this backend
        // even if the weight has to be copied from the CPU temporarily
        bool (*LM_GGML_CALL offload_op)(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);

        // (optional) event synchronization
        lm_ggml_backend_event_t (*LM_GGML_CALL event_new)         (lm_ggml_backend_t backend);
        void                 (*LM_GGML_CALL event_free)        (lm_ggml_backend_event_t event);
        void                 (*LM_GGML_CALL event_record)      (lm_ggml_backend_event_t event);
        void                 (*LM_GGML_CALL event_wait)        (lm_ggml_backend_t backend, lm_ggml_backend_event_t event);
        void                 (*LM_GGML_CALL event_synchronize) (lm_ggml_backend_event_t event);
    };

    struct lm_ggml_backend {
        lm_ggml_guid_t guid;

        struct lm_ggml_backend_i iface;
        lm_ggml_backend_context_t context;
    };

    struct lm_ggml_backend_event {
        lm_ggml_backend_t backend;
        void * context;
    };

    //
    // Backend registry
    //

    typedef lm_ggml_backend_t (*LM_GGML_CALL lm_ggml_backend_init_fn)(const char * params, void * user_data);

    LM_GGML_CALL void lm_ggml_backend_register(const char * name, lm_ggml_backend_init_fn init_fn, lm_ggml_backend_buffer_type_t default_buffer_type, void * user_data);

#ifdef  __cplusplus
}
#endif
