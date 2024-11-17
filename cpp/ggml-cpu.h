#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    // Scheduling priorities
    enum lm_ggml_sched_priority {
        LM_GGML_SCHED_PRIO_NORMAL,
        LM_GGML_SCHED_PRIO_MEDIUM,
        LM_GGML_SCHED_PRIO_HIGH,
        LM_GGML_SCHED_PRIO_REALTIME
    };

    // Threadpool params
    // Use lm_ggml_threadpool_params_default() or lm_ggml_threadpool_params_init() to populate the defaults
    struct lm_ggml_threadpool_params {
        bool                cpumask[LM_GGML_MAX_N_THREADS]; // mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads;                   // number of threads
        enum lm_ggml_sched_priority prio;                   // thread priority
        uint32_t            poll;                        // polling level (0 - no polling, 100 - aggressive polling)
        bool                strict_cpu;                  // strict cpu placement
        bool                paused;                      // start in paused state
    };

    struct lm_ggml_threadpool;     // forward declaration, see ggml.c

    typedef struct lm_ggml_threadpool * lm_ggml_threadpool_t;

    // the compute plan that needs to be prepared for lm_ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct lm_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `lm_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `lm_ggml_graph_compute()`

        int n_threads;
        struct lm_ggml_threadpool * threadpool;

        // abort lm_ggml_graph_compute when true
        lm_ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // numa strategies
    enum lm_ggml_numa_strategy {
        LM_GGML_NUMA_STRATEGY_DISABLED   = 0,
        LM_GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        LM_GGML_NUMA_STRATEGY_ISOLATE    = 2,
        LM_GGML_NUMA_STRATEGY_NUMACTL    = 3,
        LM_GGML_NUMA_STRATEGY_MIRROR     = 4,
        LM_GGML_NUMA_STRATEGY_COUNT
    };

    LM_GGML_BACKEND_API void    lm_ggml_numa_init(enum lm_ggml_numa_strategy numa); // call once for better performance on NUMA systems
    LM_GGML_BACKEND_API bool    lm_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_new_i32(struct lm_ggml_context * ctx, int32_t value);
    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_new_f32(struct lm_ggml_context * ctx, float value);

    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_set_i32 (struct lm_ggml_tensor * tensor, int32_t value);
    LM_GGML_BACKEND_API struct lm_ggml_tensor * lm_ggml_set_f32 (struct lm_ggml_tensor * tensor, float value);

    LM_GGML_BACKEND_API int32_t lm_ggml_get_i32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_BACKEND_API void    lm_ggml_set_i32_1d(const struct lm_ggml_tensor * tensor, int i, int32_t value);

    LM_GGML_BACKEND_API int32_t lm_ggml_get_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_BACKEND_API void    lm_ggml_set_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    LM_GGML_BACKEND_API float   lm_ggml_get_f32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_BACKEND_API void    lm_ggml_set_f32_1d(const struct lm_ggml_tensor * tensor, int i, float value);

    LM_GGML_BACKEND_API float   lm_ggml_get_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_BACKEND_API void    lm_ggml_set_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    LM_GGML_BACKEND_API struct lm_ggml_threadpool_params lm_ggml_threadpool_params_default(int n_threads);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_params_init   (struct lm_ggml_threadpool_params * p, int n_threads);
    LM_GGML_BACKEND_API bool                          lm_ggml_threadpool_params_match  (const struct lm_ggml_threadpool_params * p0, const struct lm_ggml_threadpool_params * p1);
    LM_GGML_BACKEND_API struct lm_ggml_threadpool *      lm_ggml_threadpool_new          (struct lm_ggml_threadpool_params  * params);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_free         (struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API int                           lm_ggml_threadpool_get_n_threads(struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_pause        (struct lm_ggml_threadpool * threadpool);
    LM_GGML_BACKEND_API void                          lm_ggml_threadpool_resume       (struct lm_ggml_threadpool * threadpool);

    // lm_ggml_graph_plan() has to be called before lm_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    LM_GGML_BACKEND_API struct lm_ggml_cplan lm_ggml_graph_plan(
                  const struct lm_ggml_cgraph * cgraph,
                                       int   n_threads, /* = LM_GGML_DEFAULT_N_THREADS */
                    struct lm_ggml_threadpool * threadpool /* = NULL */ );
    LM_GGML_BACKEND_API enum lm_ggml_status  lm_ggml_graph_compute(struct lm_ggml_cgraph * cgraph, struct lm_ggml_cplan * cplan);

    // same as lm_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    LM_GGML_BACKEND_API enum lm_ggml_status  lm_ggml_graph_compute_with_ctx(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, int n_threads);

    //
    // system info
    //

    // x86
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_sse3       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_ssse3      (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx2       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_f16c       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_fma        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx_vnni   (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512     (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_vbmi(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_vnni(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_avx512_bf16(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_amx_int8   (void);
    // ARM
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_neon       (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_arm_fma    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_fp16_va    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_matmul_int8(void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_sve        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    // other
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_riscv_v    (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_vsx        (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_wasm_simd  (void);
    LM_GGML_BACKEND_API int lm_ggml_cpu_has_llamafile  (void);

    // Internal types and functions exposed for tests and benchmarks

    typedef void (*lm_ggml_from_float_to_mat_t)
                                     (const float * LM_GGML_RESTRICT x, void * LM_GGML_RESTRICT y, int64_t nr, int64_t k, int64_t bs);
    typedef void (*lm_ggml_vec_dot_t)  (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT x, size_t bx,
                                       const void * LM_GGML_RESTRICT y, size_t by, int nrc);
    typedef void (*lm_ggml_gemv_t)     (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT x,
                                       const void * LM_GGML_RESTRICT y, int nr, int nc);
    typedef void (*lm_ggml_gemm_t)     (int n, float * LM_GGML_RESTRICT s, size_t bs, const void * LM_GGML_RESTRICT x,
                                       const void * LM_GGML_RESTRICT y, int nr, int nc);

    struct lm_ggml_type_traits_cpu {
        lm_ggml_from_float_t        from_float;
        lm_ggml_from_float_to_mat_t from_float_to_mat;
        lm_ggml_vec_dot_t           vec_dot;
        enum lm_ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
        int64_t                  ncols; // number of columns to process simultaneously
        lm_ggml_gemv_t              gemv;
        lm_ggml_gemm_t              gemm;
    };

    LM_GGML_BACKEND_API const struct lm_ggml_type_traits_cpu * lm_ggml_get_type_traits_cpu(enum lm_ggml_type type);

    LM_GGML_BACKEND_API void lm_ggml_cpu_init(void);

    //
    // CPU backend
    //

    LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_cpu_init(void);

    LM_GGML_BACKEND_API bool lm_ggml_backend_is_cpu                (lm_ggml_backend_t backend);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_n_threads     (lm_ggml_backend_t backend_cpu, int n_threads);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_threadpool    (lm_ggml_backend_t backend_cpu, lm_ggml_threadpool_t threadpool);
    LM_GGML_BACKEND_API void lm_ggml_backend_cpu_set_abort_callback(lm_ggml_backend_t backend_cpu, lm_ggml_abort_callback abort_callback, void * abort_callback_data);

    LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_cpu_reg(void);

#ifdef LM_GGML_USE_CPU_HBM
    LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_hbm_buffer_type(void);
#endif

    LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_aarch64_buffer_type(void);
    LM_GGML_BACKEND_API bool lm_ggml_backend_cpu_buft_is_aarch64(lm_ggml_backend_buffer_type_t buft);

#ifdef __cplusplus
}
#endif
