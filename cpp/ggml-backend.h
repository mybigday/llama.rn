#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

    typedef struct lm_ggml_backend_buffer_type * lm_ggml_backend_buffer_type_t;
    typedef struct lm_ggml_backend_buffer * lm_ggml_backend_buffer_t;
    typedef struct lm_ggml_backend * lm_ggml_backend_t;
    typedef void * lm_ggml_backend_graph_plan_t;

    //
    // Backend buffer
    //

    // buffer type
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_buft_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size);
    LM_GGML_API size_t lm_ggml_backend_buft_get_alignment (lm_ggml_backend_buffer_type_t buft);
    LM_GGML_API size_t lm_ggml_backend_buft_get_alloc_size(lm_ggml_backend_buffer_type_t buft, struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_backend_buft_supports_backend(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend);

    // buffer
    LM_GGML_API void   lm_ggml_backend_buffer_free          (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void * lm_ggml_backend_buffer_get_base      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_size      (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API void   lm_ggml_backend_buffer_init_tensor   (lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_alignment (lm_ggml_backend_buffer_t buffer);
    LM_GGML_API size_t lm_ggml_backend_buffer_get_alloc_size(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);
    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_buffer_type(lm_ggml_backend_buffer_t buffer);

    //
    // Backend
    //


    LM_GGML_API const char * lm_ggml_backend_name(lm_ggml_backend_t backend);
    LM_GGML_API void         lm_ggml_backend_free(lm_ggml_backend_t backend);

    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_get_default_buffer_type(lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_buffer_t      lm_ggml_backend_alloc_buffer(lm_ggml_backend_t backend, size_t size);
    LM_GGML_API size_t                     lm_ggml_backend_get_alignment(lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_tensor_set_async(lm_ggml_backend_t backend,       struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    LM_GGML_API void lm_ggml_backend_tensor_get_async(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor,       void * data, size_t offset, size_t size);

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
    LM_GGML_API void lm_ggml_backend_tensor_copy_async(lm_ggml_backend_t backend, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst); // automatic fallback to sync copy

    //
    // CPU backend
    //

    LM_GGML_API lm_ggml_backend_t lm_ggml_backend_cpu_init(void);

    LM_GGML_API bool lm_ggml_backend_is_cpu(lm_ggml_backend_t backend);
    LM_GGML_API void lm_ggml_backend_cpu_set_n_threads(lm_ggml_backend_t backend_cpu, int n_threads);

    // Create a backend buffer from an existing pointer
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);

    LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_buffer_type(void);

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

        sched = lm_ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, num_backends);
        // sched is initialized with measure allocators and cannot be used until allocated with a measure graph

        // initialize buffers from a measure graph
        measure_graph = build_graph(sched); // use the allocr to allocate inputs as needed

        // in build_graph:
        build_graph(...) {
            // allocating tensors in a specific backend (optional, recommended: pre-allocate inputs in a different buffer)
            alloc_cpu = lm_ggml_backend_sched_get_allocr(sched, backend_cpu);
            lm_ggml_allocr_alloc(alloc_cpu, tensor);

            // manually assigning nodes to a backend (optional, shouldn't be needed in most cases)
            struct lm_ggml_tensor * node = lm_ggml_mul_mat(ctx, ...);
            lm_ggml_backend_sched_set_node_backend(sched, node, backend_gpu);
        }

        // allocate backend buffers from measure graph
        lm_ggml_backend_sched_init_measure(sched, measure_graph);

        // the scheduler is now ready to compute graphs

        // compute
        graph = build_graph(sched);
        lm_ggml_backend_sched_graph_compute(sched, graph);
    */

    struct lm_ggml_backend_sched;
    typedef struct lm_ggml_backend_sched * lm_ggml_backend_sched_t;

    // Initialize a backend scheduler
    LM_GGML_API lm_ggml_backend_sched_t lm_ggml_backend_sched_new(lm_ggml_backend_t * backends, int n_backends);

    LM_GGML_API void lm_ggml_backend_sched_free(lm_ggml_backend_sched_t sched);

    // Initialize backend buffers from a measure graph
    LM_GGML_API void lm_ggml_backend_sched_init_measure(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * measure_graph);

    LM_GGML_API lm_ggml_tallocr_t        lm_ggml_backend_sched_get_tallocr(lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend);
    LM_GGML_API lm_ggml_backend_buffer_t lm_ggml_backend_sched_get_buffer (lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend);

    LM_GGML_API void lm_ggml_backend_sched_set_node_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node, lm_ggml_backend_t backend);

    // Allocate a graph on the backend scheduler
    LM_GGML_API void lm_ggml_backend_sched_graph_compute(
            lm_ggml_backend_sched_t sched,
            struct lm_ggml_cgraph * graph);


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

    typedef bool (*lm_ggml_backend_eval_callback)(int node_index, struct lm_ggml_tensor * t1, struct lm_ggml_tensor * t2, void * user_data);

    // Compare the output of two backends
    LM_GGML_API void lm_ggml_backend_compare_graph_backend(lm_ggml_backend_t backend1, lm_ggml_backend_t backend2, struct lm_ggml_cgraph * graph, lm_ggml_backend_eval_callback callback, void * user_data);

    // Tensor initialization
    LM_GGML_API void lm_ggml_backend_tensor_alloc(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, void * addr);
    LM_GGML_API void lm_ggml_backend_view_init(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor);


#ifdef  __cplusplus
}
#endif
