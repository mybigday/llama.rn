#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct lm_ggml_backend_buffer_type * lm_ggml_backend_buffer_type_t;
    typedef struct lm_ggml_backend_buffer * lm_ggml_backend_buffer_t;
    typedef struct lm_ggml_backend_event * lm_ggml_backend_event_t;
    typedef struct lm_ggml_backend * lm_ggml_backend_t;
    typedef void * lm_ggml_backend_graph_plan_t;

    //
    // Backend buffer
    //

    // buffer type
    LM_GGML_API           const char *          lm_ggml_backend_buft_name            (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_buft_alloc_buffer    (lm_ggml_backend_buffer_type_t buft, size_t size);
    LM_GGML_API           size_t                lm_ggml_backend_buft_get_alignment   (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API           size_t                lm_ggml_backend_buft_get_max_size    (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API LM_GGML_CALL size_t                lm_ggml_backend_buft_get_alloc_size  (lm_ggml_backend_buffer_type_t buft, struct lm_ggml_tensor * tensor);
    LM_GGML_API           bool                  lm_ggml_backend_buft_supports_backend(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend);
    LM_GGML_API           bool                  lm_ggml_backend_buft_is_host         (lm_ggml_backend_buffer_type_t buft);

    // buffer
    enum lm_ggml_backend_buffer_usage {
        LM_GGML_BACKEND_BUFFER_USAGE_ANY = 0,
        LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
    };

    LM_GGML_API           const char *               lm_ggml_backend_buffer_name          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           void                       lm_ggml_backend_buffer_free          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           void *                     lm_ggml_backend_buffer_get_base      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           size_t                     lm_ggml_backend_buffer_get_size      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API LM_GGML_CALL void                       lm_ggml_backend_buffer_init_tensor   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API           size_t                     lm_ggml_backend_buffer_get_alignment (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           size_t                     lm_ggml_backend_buffer_get_max_size  (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           size_t                     lm_ggml_backend_buffer_get_alloc_size(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API           void                       lm_ggml_backend_buffer_clear         (lm_ggml_backend_buffer_t buffer, uint8_t value);
    LM_GGML_API           bool                       lm_ggml_backend_buffer_is_host       (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           void                       lm_ggml_backend_buffer_set_usage     (lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage);
    LM_GGML_API           lm_ggml_backend_buffer_type_t lm_ggml_backend_buffer_get_type      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API           void                       lm_ggml_backend_buffer_reset         (lm_ggml_backend_buffer_t buffer);

    //
    // Backend
    //

    LM_GGML_API lm_ggml_guid_t  lm_ggml_backend_guid(lm_ggml_backend_t backend);
    LM_GGML_API const char * lm_ggml_backend_name(lm_ggml_backend_t backend);
    LM_GGML_API void         lm_ggml_backend_free(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_get_default_buffer_type(lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_buffer_t      lm_ggml_backend_alloc_buffer(lm_ggml_backend_t backend, size_t size);
    LM_GGML_API size_t                     lm_ggml_backend_get_alignment(lm_ggml_backend_t backend);
    LM_GGML_API size_t                     lm_ggml_backend_get_max_size(lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_tensor_set_async(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get_async(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    LM_GGML_API LM_GGML_CALL void lm_ggml_backend_tensor_set(      struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API LM_GGML_CALL void lm_ggml_backend_tensor_get(const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

    LM_GGML_API void lm_ggml_backend_synchronize(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_graph_plan_t lm_ggml_backend_graph_plan_create(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API void                      lm_ggml_backend_graph_plan_free  (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);

    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_plan_compute (lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan);
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_compute      (lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API enum lm_ggml_status lm_ggml_backend_graph_compute_async(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph);
    LM_GGML_API bool lm_ggml_backend_supports_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);
    LM_GGML_API bool lm_ggml_backend_offload_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op);

    // tensor copy between different backends
    LM_GGML_API void lm_ggml_backend_tensor_copy(struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    // asynchronous copy
    // the copy is performed after all the currently queued operations in backend_src
    // backend_dst will wait for the copy to complete before performing other operations
    // automatic fallback to sync copy if async is not supported
    LM_GGML_API void lm_ggml_backend_tensor_copy_async(lm_ggml_backend_t backend_src, lm_ggml_backend_t backend_dst, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst);

    // events
    LM_GGML_API lm_ggml_backend_event_t   lm_ggml_backend_event_new        (lm_ggml_backend_t backend);
    LM_GGML_API void                   lm_ggml_backend_event_free       (lm_ggml_backend_event_t event);
    LM_GGML_API void                   lm_ggml_backend_event_record     (lm_ggml_backend_event_t event);
    LM_GGML_API void                   lm_ggml_backend_event_synchronize(lm_ggml_backend_event_t event);
    LM_GGML_API void                   lm_ggml_backend_event_wait       (lm_ggml_backend_t backend, lm_ggml_backend_event_t event); // wait async on event

    //
    // CPU backend
    //

    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_cpu_init(void);

    LM_GGML_API LM_GGML_CALL bool lm_ggml_backend_is_cpu                (lm_ggml_backend_t backend);
    LM_GGML_API           void lm_ggml_backend_cpu_set_n_threads     (lm_ggml_backend_t backend_cpu, int n_threads);
    LM_GGML_API           void lm_ggml_backend_cpu_set_abort_callback(lm_ggml_backend_t backend_cpu, lm_ggml_abort_callback abort_callback, void * abort_callback_data);

    // Create a backend buffer from an existing pointer
    LM_GGML_API LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    LM_GGML_API LM_GGML_CALL lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_buffer_type(void);

#ifdef LM_GGML_USE_CPU_HBM
    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_hbm_buffer_type(void);
#endif

    //
    // Backend registry
    //

    // The backend registry is a registry of all the available backends, and allows initializing backends in a generic way

    LM_GGML_API size_t                     lm_ggml_backend_reg_get_count(void);
    LM_GGML_API size_t                     lm_ggml_backend_reg_find_by_name(const char * name);
    LM_GGML_API lm_ggml_backend_t             lm_ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is name[:params]
    LM_GGML_API const char *               lm_ggml_backend_reg_get_name(size_t i);
    LM_GGML_API lm_ggml_backend_t             lm_ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific
    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_reg_get_default_buffer_type(size_t i);
    LM_GGML_API lm_ggml_backend_buffer_t      lm_ggml_backend_reg_alloc_buffer(size_t i, size_t size);

    //
    // Backend scheduler
    //

    // The backend scheduler allows for multiple backends to be used together
    // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
    // The backends are selected based on:
    // - the backend that supports the operation
    // - the location of the pre-allocated tensors (e.g. the weights)
    /*
      Example usage:

        // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be asigned
        // preferrably to run on the same backend as the buffer
        lm_ggml_backend_buffer_set_usage(buf_weights, LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        sched = lm_ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, LM_GGML_DEFAULT_GRAPH_SIZE, false);

        // initialize buffers from a max size graph (optional)
        reserve_graph = build_graph(sched, max_batch_size);

        // manually assign nodes to a backend (optional, should not be needed in most cases)
        struct lm_ggml_tensor * node = lm_ggml_mul_mat(ctx, ...);
        lm_ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

        lm_ggml_backend_sched_reserve(sched, reserve_graph);

        // compute
        graph = build_graph(sched);
        lm_ggml_backend_sched_graph_compute(sched, graph);

        // if there are graph inputs:
        lm_ggml_backend_sched_reset(sched);
        lm_ggml_backend_sched_alloc_graph(sched, graph);
        lm_ggml_backend_tensor_set(input_tensor, ...);
        lm_ggml_backend_sched_graph_compute(sched, graph);
    }
    */

    struct lm_ggml_backend_sched;
    typedef struct lm_ggml_backend_sched * lm_ggml_backend_sched_t;

    // when ask == true, the scheduler wants to know if the user wants to observe this node
    // this allows the scheduler to batch nodes together in order to evaluate them in a single call
    //
    // when ask == false, the scheduler is passing the node tensor to the user for observation
    // if the user returns false, the scheduler will cancel the graph compute
    //
    typedef bool (*lm_ggml_backend_sched_eval_callback)(struct lm_ggml_tensor * t, bool ask, void * user_data);

    // Initialize a backend scheduler
    LM_GGML_API lm_ggml_backend_sched_t lm_ggml_backend_sched_new(lm_ggml_backend_t * backends, lm_ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel);
    LM_GGML_API void                 lm_ggml_backend_sched_free(lm_ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    LM_GGML_API bool                 lm_ggml_backend_sched_reserve(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * measure_graph);

    // Get the number of splits of the last graph
    LM_GGML_API int                  lm_ggml_backend_sched_get_n_splits(lm_ggml_backend_sched_t sched);
    LM_GGML_API int                  lm_ggml_backend_sched_get_n_copies(lm_ggml_backend_sched_t sched);

    LM_GGML_API size_t               lm_ggml_backend_sched_get_buffer_size(lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend);

    LM_GGML_API void                 lm_ggml_backend_sched_set_tensor_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node, lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_t       lm_ggml_backend_sched_get_tensor_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node);

    // Allocate and compute graph on the backend scheduler
    LM_GGML_API bool                 lm_ggml_backend_sched_alloc_graph(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph);
    LM_GGML_API enum lm_ggml_status     lm_ggml_backend_sched_graph_compute(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph);
    LM_GGML_API enum lm_ggml_status     lm_ggml_backend_sched_graph_compute_async(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph);
    LM_GGML_API void                 lm_ggml_backend_sched_synchronize(lm_ggml_backend_sched_t sched);

    // Reset all assignments and allocators - must be called before changing the node backends
    LM_GGML_API void                 lm_ggml_backend_sched_reset(lm_ggml_backend_sched_t sched);

    // Set a callback to be called for each resulting node during graph compute
    LM_GGML_API void                 lm_ggml_backend_sched_set_eval_callback(lm_ggml_backend_sched_t sched, lm_ggml_backend_sched_eval_callback callback, void * user_data);

    //
    // Utils
    //

    struct lm_ggml_backend_graph_copy {
        lm_ggml_backend_buffer_t buffer;
        struct lm_ggml_context * ctx_allocated;
        struct lm_ggml_context * ctx_unallocated;
        struct lm_ggml_cgraph * graph;
    };

    // Copy a graph to a different backend
    LM_GGML_API struct lm_ggml_backend_graph_copy lm_ggml_backend_graph_copy(lm_ggml_backend_t backend, struct lm_ggml_cgraph * graph);
    LM_GGML_API void                           lm_ggml_backend_graph_copy_free(struct lm_ggml_backend_graph_copy copy);

    typedef bool (*LM_GGML_CALL lm_ggml_backend_eval_callback)(int node_index, struct lm_ggml_tensor * t1, struct lm_ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    LM_GGML_API bool lm_ggml_backend_compare_graph_backend(lm_ggml_backend_t backend1, lm_ggml_backend_t backend2, struct lm_ggml_cgraph * graph, lm_ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    LM_GGML_API void lm_ggml_backend_tensor_alloc(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, void * addr);
    LM_GGML_API void lm_ggml_backend_view_init(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);


#ifdef  __cplusplus
}
#endif
