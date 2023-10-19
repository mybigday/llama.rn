#pragma once

//
// GGML Tensor Library
//
// This documentation is still a work in progress.
// If you wish some specific topics to be covered, feel free to drop a comment:
//
//   https://github.com/ggerganov/whisper.cpp/issues/40
//
// ## Overview
//
// This library implements:
//
//  - a set of tensor operations
//  - automatic differentiation
//  - basic optimization algorithms
//
// The aim of this library is to provide a minimalistic approach for various machine learning tasks. This includes,
// but is not limited to, the following:
//
//  - linear regression
//  - support vector machines
//  - neural networks
//
// The library allows the user to define a certain function using the available tensor operations. This function
// definition is represented internally via a computation graph. Each tensor operation in the function definition
// corresponds to a node in the graph. Having the computation graph defined, the user can choose to compute the
// function's value and/or its gradient with respect to the input variables. Optionally, the function can be optimized
// using one of the available optimization algorithms.
//
// For example, here we define the function: f(x) = a*x^2 + b
//
//   {
//       struct lm_ggml_init_params params = {
//           .mem_size   = 16*1024*1024,
//           .mem_buffer = NULL,
//       };
//
//       // memory allocation happens here
//       struct lm_ggml_context * ctx = lm_ggml_init(params);
//
//       struct lm_ggml_tensor * x = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//
//       lm_ggml_set_param(ctx, x); // x is an input variable
//
//       struct lm_ggml_tensor * a  = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//       struct lm_ggml_tensor * b  = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);
//       struct lm_ggml_tensor * x2 = lm_ggml_mul(ctx, x, x);
//       struct lm_ggml_tensor * f  = lm_ggml_add(ctx, lm_ggml_mul(ctx, a, x2), b);
//
//       ...
//   }
//
// Notice that the function definition above does not involve any actual computation. The computation is performed only
// when the user explicitly requests it. For example, to compute the function's value at x = 2.0:
//
//   {
//       ...
//
//       struct lm_ggml_cgraph gf = lm_ggml_build_forward(f);
//
//       // set the input variable and parameter values
//       lm_ggml_set_f32(x, 2.0f);
//       lm_ggml_set_f32(a, 3.0f);
//       lm_ggml_set_f32(b, 4.0f);
//
//       lm_ggml_graph_compute_with_ctx(ctx, &gf, n_threads);
//
//       printf("f = %f\n", lm_ggml_get_f32_1d(f, 0));
//
//       ...
//   }
//
// The actual computation is performed in the lm_ggml_graph_compute() function.
//
// The lm_ggml_new_tensor_...() functions create new tensors. They are allocated in the memory buffer provided to the
// lm_ggml_init() function. You have to be careful not to exceed the memory buffer size. Therefore, you have to know
// in advance how much memory you need for your computation. Alternatively, you can allocate a large enough memory
// and after defining the computation graph, call the lm_ggml_used_mem() function to find out how much memory was
// actually needed.
//
// The lm_ggml_set_param() function marks a tensor as an input variable. This is used by the automatic
// differentiation and optimization algorithms.
//
// The described approach allows to define the function graph once and then compute its forward or backward graphs
// multiple times. All computations will use the same memory buffer allocated in the lm_ggml_init() function. This way
// the user can avoid the memory allocation overhead at runtime.
//
// The library supports multi-dimensional tensors - up to 4 dimensions. The FP16 and FP32 data types are first class
// citizens, but in theory the library can be extended to support FP8 and integer data types.
//
// Each tensor operation produces a new tensor. Initially the library was envisioned to support only the use of unary
// and binary operations. Most of the available operations fall into one of these two categories. With time, it became
// clear that the library needs to support more complex operations. The way to support these operations is not clear
// yet, but a few examples are demonstrated in the following operations:
//
//   - lm_ggml_permute()
//   - lm_ggml_conv_1d_1s()
//   - lm_ggml_conv_1d_2s()
//
// For each tensor operator, the library implements a forward and backward computation function. The forward function
// computes the output tensor value given the input tensor values. The backward function computes the adjoint of the
// input tensors given the adjoint of the output tensor. For a detailed explanation of what this means, take a
// calculus class, or watch the following video:
//
//   What is Automatic Differentiation?
//   https://www.youtube.com/watch?v=wG_nF1awSSY
//
//
// ## Tensor data (struct lm_ggml_tensor)
//
// The tensors are stored in memory via the lm_ggml_tensor struct. The structure provides information about the size of
// the tensor, the data type, and the memory buffer where the tensor data is stored. Additionally, it contains
// pointers to the "source" tensors - i.e. the tensors that were used to compute the current tensor. For example:
//
//   {
//       struct lm_ggml_tensor * c = lm_ggml_add(ctx, a, b);
//
//       assert(c->src[0] == a);
//       assert(c->src[1] == b);
//   }
//
// The multi-dimensional tensors are stored in row-major order. The lm_ggml_tensor struct contains fields for the
// number of elements in each dimension ("ne") as well as the number of bytes ("nb", a.k.a. stride). This allows
// to store tensors that are not contiguous in memory, which is useful for operations such as transposition and
// permutation. All tensor operations have to take the stride into account and not assume that the tensor is
// contiguous in memory.
//
// The data of the tensor is accessed via the "data" pointer. For example:
//
//   {
//       const int nx = 2;
//       const int ny = 3;
//
//       struct lm_ggml_tensor * a = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, nx, ny);
//
//       for (int y = 0; y < ny; y++) {
//           for (int x = 0; x < nx; x++) {
//               *(float *) ((char *) a->data + y*a->nb[1] + x*a->nb[0]) = x + y;
//           }
//       }
//
//       ...
//   }
//
// Alternatively, there are helper functions, such as lm_ggml_get_f32_1d() and lm_ggml_set_f32_1d() that can be used.
//
// ## The matrix multiplication operator (lm_ggml_mul_mat)
//
// TODO
//
//
// ## Multi-threading
//
// TODO
//
//
// ## Overview of ggml.c
//
// TODO
//
//
// ## SIMD optimizations
//
// TODO
//
//
// ## Debugging ggml
//
// TODO
//
//

#ifdef LM_GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LM_GGML_BUILD
#            define LM_GGML_API __declspec(dllexport)
#        else
#            define LM_GGML_API __declspec(dllimport)
#        endif
#    else
#        define LM_GGML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LM_GGML_API
#endif

// TODO: support for clang
#ifdef __GNUC__
#    define LM_GGML_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define LM_GGML_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define LM_GGML_DEPRECATED(func, hint) func
#endif

#ifndef __GNUC__
#    define LM_GGML_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__)
#    define LM_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LM_GGML_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define LM_GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define LM_GGML_FILE_VERSION 1

#define LM_GGML_QNT_VERSION        2    // bump this on quantization format changes
#define LM_GGML_QNT_VERSION_FACTOR 1000 // do not change this

#define LM_GGML_MAX_DIMS          4
#define LM_GGML_MAX_NODES         16384
#define LM_GGML_MAX_PARAMS        1024
#define LM_GGML_MAX_CONTEXTS      64
#define LM_GGML_MAX_SRC           6
#define LM_GGML_MAX_NAME          64
#define LM_GGML_MAX_OP_PARAMS     32
#define LM_GGML_DEFAULT_N_THREADS 4

#if UINTPTR_MAX == 0xFFFFFFFF
    #define LM_GGML_MEM_ALIGN 4
#else
    #define LM_GGML_MEM_ALIGN 16
#endif

#define LM_GGML_EXIT_SUCCESS 0
#define LM_GGML_EXIT_ABORTED 1

#define GGUF_MAGIC   0x46554747 // "GGUF"
#define GGUF_VERSION 2

#define GGUF_DEFAULT_ALIGNMENT 32

#define LM_GGML_UNUSED(x) (void)(x)

#define LM_GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define LM_GGML_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "LM_GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

#ifndef NDEBUG
#define LM_GGML_UNREACHABLE() LM_GGML_ASSERT(!"statement should not be reached")
#elif defined(__GNUC__)
#define LM_GGML_UNREACHABLE() __builtin_unreachable()
#else
#define LM_GGML_UNREACHABLE() ((void) 0)
#endif

// used to copy the number of elements and stride in bytes of tensors into local variables.
// main purpose is to reduce code duplication and improve readability.
//
// example:
//
//    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
//    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb);
//
#define LM_GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    LM_GGML_UNUSED(prefix##0);
#define LM_GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    LM_GGML_UNUSED(prefix##1);
#define LM_GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    LM_GGML_UNUSED(prefix##2);
#define LM_GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    LM_GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    LM_GGML_UNUSED(prefix##3);

#ifdef  __cplusplus
extern "C" {
#endif

#if defined(__ARM_NEON) && defined(__CUDACC__)
    typedef half lm_ggml_fp16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 lm_ggml_fp16_t;
#else
    typedef uint16_t lm_ggml_fp16_t;
#endif

    // convert FP16 <-> FP32
    LM_GGML_API float       lm_ggml_fp16_to_fp32(lm_ggml_fp16_t x);
    LM_GGML_API lm_ggml_fp16_t lm_ggml_fp32_to_fp16(float x);

    LM_GGML_API void lm_ggml_fp16_to_fp32_row(const lm_ggml_fp16_t * x, float * y, int n);
    LM_GGML_API void lm_ggml_fp32_to_fp16_row(const float * x, lm_ggml_fp16_t * y, int n);

    struct lm_ggml_object;
    struct lm_ggml_context;

    enum lm_ggml_type {
        LM_GGML_TYPE_F32  = 0,
        LM_GGML_TYPE_F16  = 1,
        LM_GGML_TYPE_Q4_0 = 2,
        LM_GGML_TYPE_Q4_1 = 3,
        // LM_GGML_TYPE_Q4_2 = 4, support has been removed
        // LM_GGML_TYPE_Q4_3 (5) support has been removed
        LM_GGML_TYPE_Q5_0 = 6,
        LM_GGML_TYPE_Q5_1 = 7,
        LM_GGML_TYPE_Q8_0 = 8,
        LM_GGML_TYPE_Q8_1 = 9,
        // k-quantizations
        LM_GGML_TYPE_Q2_K = 10,
        LM_GGML_TYPE_Q3_K = 11,
        LM_GGML_TYPE_Q4_K = 12,
        LM_GGML_TYPE_Q5_K = 13,
        LM_GGML_TYPE_Q6_K = 14,
        LM_GGML_TYPE_Q8_K = 15,
        LM_GGML_TYPE_I8,
        LM_GGML_TYPE_I16,
        LM_GGML_TYPE_I32,
        LM_GGML_TYPE_COUNT,
    };

    enum lm_ggml_backend_type {
        LM_GGML_BACKEND_CPU = 0,
        LM_GGML_BACKEND_GPU = 10,
        LM_GGML_BACKEND_GPU_SPLIT = 20,
    };

    // model file types
    enum lm_ggml_ftype {
        LM_GGML_FTYPE_UNKNOWN     = -1,
        LM_GGML_FTYPE_ALL_F32     = 0,
        LM_GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        LM_GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
        LM_GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
    };

    // available tensor operations:
    enum lm_ggml_op {
        LM_GGML_OP_NONE = 0,

        LM_GGML_OP_DUP,
        LM_GGML_OP_ADD,
        LM_GGML_OP_ADD1,
        LM_GGML_OP_ACC,
        LM_GGML_OP_SUB,
        LM_GGML_OP_MUL,
        LM_GGML_OP_DIV,
        LM_GGML_OP_SQR,
        LM_GGML_OP_SQRT,
        LM_GGML_OP_LOG,
        LM_GGML_OP_SUM,
        LM_GGML_OP_SUM_ROWS,
        LM_GGML_OP_MEAN,
        LM_GGML_OP_ARGMAX,
        LM_GGML_OP_REPEAT,
        LM_GGML_OP_REPEAT_BACK,
        LM_GGML_OP_CONCAT,
        LM_GGML_OP_SILU_BACK,
        LM_GGML_OP_NORM, // normalize
        LM_GGML_OP_RMS_NORM,
        LM_GGML_OP_RMS_NORM_BACK,
        LM_GGML_OP_GROUP_NORM,

        LM_GGML_OP_MUL_MAT,
        LM_GGML_OP_OUT_PROD,

        LM_GGML_OP_SCALE,
        LM_GGML_OP_SET,
        LM_GGML_OP_CPY,
        LM_GGML_OP_CONT,
        LM_GGML_OP_RESHAPE,
        LM_GGML_OP_VIEW,
        LM_GGML_OP_PERMUTE,
        LM_GGML_OP_TRANSPOSE,
        LM_GGML_OP_GET_ROWS,
        LM_GGML_OP_GET_ROWS_BACK,
        LM_GGML_OP_DIAG,
        LM_GGML_OP_DIAG_MASK_INF,
        LM_GGML_OP_DIAG_MASK_ZERO,
        LM_GGML_OP_SOFT_MAX,
        LM_GGML_OP_SOFT_MAX_BACK,
        LM_GGML_OP_ROPE,
        LM_GGML_OP_ROPE_BACK,
        LM_GGML_OP_ALIBI,
        LM_GGML_OP_CLAMP,
        LM_GGML_OP_CONV_1D,
        LM_GGML_OP_CONV_2D,
        LM_GGML_OP_CONV_TRANSPOSE_1D,
        LM_GGML_OP_CONV_TRANSPOSE_2D,
        LM_GGML_OP_POOL_1D,
        LM_GGML_OP_POOL_2D,

        LM_GGML_OP_CONV_1D_STAGE_0,  // internal
        LM_GGML_OP_CONV_1D_STAGE_1,  // internal

        LM_GGML_OP_UPSCALE, // nearest interpolate

        LM_GGML_OP_FLASH_ATTN,
        LM_GGML_OP_FLASH_FF,
        LM_GGML_OP_FLASH_ATTN_BACK,
        LM_GGML_OP_WIN_PART,
        LM_GGML_OP_WIN_UNPART,
        LM_GGML_OP_GET_REL_POS,
        LM_GGML_OP_ADD_REL_POS,

        LM_GGML_OP_UNARY,

        LM_GGML_OP_MAP_UNARY,
        LM_GGML_OP_MAP_BINARY,

        LM_GGML_OP_MAP_CUSTOM1_F32,
        LM_GGML_OP_MAP_CUSTOM2_F32,
        LM_GGML_OP_MAP_CUSTOM3_F32,

        LM_GGML_OP_MAP_CUSTOM1,
        LM_GGML_OP_MAP_CUSTOM2,
        LM_GGML_OP_MAP_CUSTOM3,

        LM_GGML_OP_CROSS_ENTROPY_LOSS,
        LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK,

        LM_GGML_OP_COUNT,
    };

    enum lm_ggml_unary_op {
        LM_GGML_UNARY_OP_ABS,
        LM_GGML_UNARY_OP_SGN,
        LM_GGML_UNARY_OP_NEG,
        LM_GGML_UNARY_OP_STEP,
        LM_GGML_UNARY_OP_TANH,
        LM_GGML_UNARY_OP_ELU,
        LM_GGML_UNARY_OP_RELU,
        LM_GGML_UNARY_OP_GELU,
        LM_GGML_UNARY_OP_GELU_QUICK,
        LM_GGML_UNARY_OP_SILU,
    };

    enum lm_ggml_object_type {
        LM_GGML_OBJECT_TENSOR,
        LM_GGML_OBJECT_GRAPH,
        LM_GGML_OBJECT_WORK_BUFFER
    };

    enum lm_ggml_log_level {
        LM_GGML_LOG_LEVEL_ERROR = 2,
        LM_GGML_LOG_LEVEL_WARN = 3,
        LM_GGML_LOG_LEVEL_INFO = 4
    };

    // ggml object
    struct lm_ggml_object {
        size_t offs;
        size_t size;

        struct lm_ggml_object * next;

        enum lm_ggml_object_type type;

        char padding[4];
    };

    static const size_t LM_GGML_OBJECT_SIZE = sizeof(struct lm_ggml_object);

    // n-dimensional tensor
    struct lm_ggml_tensor {
        enum lm_ggml_type         type;
        enum lm_ggml_backend_type backend;

        struct lm_ggml_backend_buffer * buffer;

        int     n_dims;
        int64_t ne[LM_GGML_MAX_DIMS]; // number of elements
        size_t  nb[LM_GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = lm_ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / lm_ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum lm_ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[LM_GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        bool is_param;

        struct lm_ggml_tensor * grad;
        struct lm_ggml_tensor * src[LM_GGML_MAX_SRC];

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;

        struct lm_ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[LM_GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[12];
    };

    static const size_t LM_GGML_TENSOR_SIZE = sizeof(struct lm_ggml_tensor);

    // the compute plan that needs to be prepared for lm_ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct lm_ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `lm_ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `lm_ggml_graph_compute()`

        int n_threads;

        // the `n_tasks` of nodes, 1:1 mapping to cgraph nodes
        int n_tasks[LM_GGML_MAX_NODES];

        // abort lm_ggml_graph_compute when true
        bool (*abort_callback)(void * data);
        void * abort_callback_data;
    };

    // next prime after LM_GGML_MAX_NODES
    // #define LM_GGML_GRAPH_HASHTABLE_SIZE 4099
    // next prime after LM_GGML_MAX_NODES * 2 (nodes + leafs)
    // #define LM_GGML_GRAPH_HASHTABLE_SIZE 8273
    // #define LM_GGML_GRAPH_HASHTABLE_SIZE 16411
    #define LM_GGML_GRAPH_HASHTABLE_SIZE 32771

    enum lm_ggml_cgraph_eval_order {
        LM_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        LM_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        LM_GGML_CGRAPH_EVAL_ORDER_COUNT
    };

    // computation graph
    struct lm_ggml_cgraph {
        int n_nodes;
        int n_leafs;

        struct lm_ggml_tensor * nodes[LM_GGML_MAX_NODES];
        struct lm_ggml_tensor * grads[LM_GGML_MAX_NODES];
        struct lm_ggml_tensor * leafs[LM_GGML_MAX_NODES];

        void * visited_hash_table[LM_GGML_GRAPH_HASHTABLE_SIZE];

        enum lm_ggml_cgraph_eval_order order;

        // performance
        int     perf_runs;
        int64_t perf_cycles;
        int64_t perf_time_us;
    };

    static const size_t LM_GGML_GRAPH_SIZE = sizeof(struct lm_ggml_cgraph);

    // scratch buffer
    struct lm_ggml_scratch {
        size_t offs;
        size_t size;
        void * data;
    };

    struct lm_ggml_init_params {
        // memory pool
        size_t mem_size;   // bytes
        void * mem_buffer; // if NULL, memory will be allocated internally
        bool   no_alloc;   // don't allocate memory for the tensor data
    };


    // compute types

    // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
    // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
    enum lm_ggml_task_type {
        LM_GGML_TASK_INIT = 0,
        LM_GGML_TASK_COMPUTE,
        LM_GGML_TASK_FINALIZE,
    };

    struct lm_ggml_compute_params {
        enum lm_ggml_task_type type;

        // ith = thread index, nth = number of threads
        int ith, nth;

        // work buffer for all threads
        size_t wsize;
        void * wdata;
    };

    // misc

    LM_GGML_API void    lm_ggml_time_init(void); // call this once at the beginning of the program
    LM_GGML_API int64_t lm_ggml_time_ms(void);
    LM_GGML_API int64_t lm_ggml_time_us(void);
    LM_GGML_API int64_t lm_ggml_cycles(void);
    LM_GGML_API int64_t lm_ggml_cycles_per_ms(void);

    LM_GGML_API void    lm_ggml_numa_init(void); // call once for better performance on NUMA systems
    LM_GGML_API bool    lm_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

    LM_GGML_API void    lm_ggml_print_object (const struct lm_ggml_object * obj);
    LM_GGML_API void    lm_ggml_print_objects(const struct lm_ggml_context * ctx);

    LM_GGML_API int64_t lm_ggml_nelements   (const struct lm_ggml_tensor * tensor);
    LM_GGML_API int64_t lm_ggml_nrows       (const struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t  lm_ggml_nbytes      (const struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t  lm_ggml_nbytes_pad  (const struct lm_ggml_tensor * tensor); // same as lm_ggml_nbytes() but padded to LM_GGML_MEM_ALIGN
    LM_GGML_API size_t  lm_ggml_nbytes_split(const struct lm_ggml_tensor * tensor, int nrows_split);

    LM_GGML_API int     lm_ggml_blck_size (enum lm_ggml_type type);
    LM_GGML_API size_t  lm_ggml_type_size (enum lm_ggml_type type); // size in bytes for all elements in a block
    LM_GGML_API float   lm_ggml_type_sizef(enum lm_ggml_type type); // lm_ggml_type_size()/lm_ggml_blck_size() as float

    LM_GGML_API const char * lm_ggml_type_name(enum lm_ggml_type type);
    LM_GGML_API const char * lm_ggml_op_name  (enum lm_ggml_op   op);
    LM_GGML_API const char * lm_ggml_op_symbol(enum lm_ggml_op   op);

    LM_GGML_API size_t  lm_ggml_element_size(const struct lm_ggml_tensor * tensor);

    LM_GGML_API bool    lm_ggml_is_quantized(enum lm_ggml_type type);

    // TODO: temporary until model loading of ggml examples is refactored
    LM_GGML_API enum lm_ggml_type lm_ggml_ftype_to_lm_ggml_type(enum lm_ggml_ftype ftype);

    LM_GGML_API bool lm_ggml_is_transposed(const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_contiguous(const struct lm_ggml_tensor * tensor);
    LM_GGML_API bool lm_ggml_is_permuted  (const struct lm_ggml_tensor * tensor);

    LM_GGML_API bool lm_ggml_are_same_shape(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1);

    // use this to compute the memory overhead of a tensor
    LM_GGML_API size_t lm_ggml_tensor_overhead(void);

    // main

    LM_GGML_API struct lm_ggml_context * lm_ggml_init(struct lm_ggml_init_params params);
    LM_GGML_API void                  lm_ggml_free(struct lm_ggml_context * ctx);

    LM_GGML_API size_t  lm_ggml_used_mem(const struct lm_ggml_context * ctx);

    LM_GGML_API size_t  lm_ggml_set_scratch (struct lm_ggml_context * ctx, struct lm_ggml_scratch scratch);
    LM_GGML_API bool    lm_ggml_get_no_alloc(struct lm_ggml_context * ctx);
    LM_GGML_API void    lm_ggml_set_no_alloc(struct lm_ggml_context * ctx, bool no_alloc);

    LM_GGML_API void *  lm_ggml_get_mem_buffer     (const struct lm_ggml_context * ctx);
    LM_GGML_API size_t  lm_ggml_get_mem_size       (const struct lm_ggml_context * ctx);
    LM_GGML_API size_t  lm_ggml_get_max_tensor_size(const struct lm_ggml_context * ctx);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int    n_dims,
            const int64_t *ne);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_1d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_2d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_3d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_tensor_4d(
            struct lm_ggml_context * ctx,
            enum   lm_ggml_type type,
            int64_t ne0,
            int64_t ne1,
            int64_t ne2,
            int64_t ne3);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_i32(struct lm_ggml_context * ctx, int32_t value);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_new_f32(struct lm_ggml_context * ctx, float value);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup_tensor (struct lm_ggml_context * ctx, const struct lm_ggml_tensor * src);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_tensor(struct lm_ggml_context * ctx, struct lm_ggml_tensor * src);

    // Context tensor enumeration and lookup
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_first_tensor(struct lm_ggml_context * ctx);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_next_tensor (struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_tensor(struct lm_ggml_context * ctx, const char * name);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_zero(struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_i32 (struct lm_ggml_tensor * tensor, int32_t value);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_f32 (struct lm_ggml_tensor * tensor, float value);

    // Converts a flat index into coordinates
    LM_GGML_API void    lm_ggml_unravel_index(const struct lm_ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);

    LM_GGML_API int32_t lm_ggml_get_i32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_API void    lm_ggml_set_i32_1d(const struct lm_ggml_tensor * tensor, int i, int32_t value);

    LM_GGML_API int32_t lm_ggml_get_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_API void    lm_ggml_set_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

    LM_GGML_API float   lm_ggml_get_f32_1d(const struct lm_ggml_tensor * tensor, int i);
    LM_GGML_API void    lm_ggml_set_f32_1d(const struct lm_ggml_tensor * tensor, int i, float value);

    LM_GGML_API float   lm_ggml_get_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3);
    LM_GGML_API void    lm_ggml_set_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

    LM_GGML_API void *  lm_ggml_get_data    (const struct lm_ggml_tensor * tensor);
    LM_GGML_API float * lm_ggml_get_data_f32(const struct lm_ggml_tensor * tensor);

    LM_GGML_API enum lm_ggml_unary_op lm_ggml_get_unary_op(const struct lm_ggml_tensor * tensor);

    LM_GGML_API const char *         lm_ggml_get_name   (const struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_name   (      struct lm_ggml_tensor * tensor, const char * name);
    LM_GGML_ATTRIBUTE_FORMAT(2, 3)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_format_name(      struct lm_ggml_tensor * tensor, const char * fmt, ...);

    //
    // operations on tensors with backpropagation
    //

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_dup_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_cast(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            enum   lm_ggml_type      type);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add1(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add1_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_acc(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_acc_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sub(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sub_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_div(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_div_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqr(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqr_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqrt(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sqrt_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_log(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_log_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // return scalar
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sum(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sum_rows(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // mean along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mean(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // argmax along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_argmax(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // if a is the same shape as b, and a is not parameter, return a
    // otherwise, return a new tensor: repeat(a) to fit in b
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_repeat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // sums repetitions in a into shape of b
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_repeat_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // concat a and b on dim 2
    // used in stable-diffusion
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_concat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_abs(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_abs_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sgn(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_sgn_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_neg(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_neg_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_step(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_step_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_tanh(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_tanh_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_elu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_elu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_relu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_relu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // TODO: double-check this computation is correct
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_quick(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_gelu_quick_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // a - x
    // b - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_silu_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // normalize along rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 eps);

    // group normalize along ne0*ne1*n_groups
    // used in stable-diffusion
    // TODO: eps is hardcoded to 1e-6 for now
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_group_norm(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_groups);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_group_norm_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_groups);

    // a - x
    // b - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rms_norm_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            float                 eps);

    // A: n columns, m rows
    // B: n columns, p rows  (i.e. we transpose it internally)
    // result is m columns, p rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_mul_mat(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // A: m columns, n rows,
    // B: p columns, n rows,
    // result is m columns, p rows
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_out_prod(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    //
    // operations on tensors without backpropagation
    //

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_scale(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_scale_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                nb2,
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_1d_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return modified a
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // b -> view(a,offset,nb1,nb2,3), return view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_set_2d_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            size_t                nb1,
            size_t                offset);

    // a -> b, return view(b)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cpy(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // a -> b, in-place, return view(b)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cpy_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // make contiguous
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // make contiguous, in-place
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // make contiguous, with new shape
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // return view(a), b specifies the new shape
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1);

    // return view(a)
    // TODO: when we start computing gradient, make a copy instead of view
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_reshape_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3);

    // offset in bytes
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            size_t                nb1, // row stride in bytes
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_3d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_view_4d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            int64_t               ne3,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                nb3,
            size_t                offset);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_permute(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   axis0,
            int                   axis1,
            int                   axis2,
            int                   axis3);

    // alias for lm_ggml_permute(ctx, a, 1, 0, 2, 3)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_transpose(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rows(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rows_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            struct lm_ggml_tensor  * c);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag(
        struct lm_ggml_context     * ctx,
        struct lm_ggml_tensor      * a);

    // set elements above the diagonal to -INF
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_inf(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_inf_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // set elements above the diagonal to 0
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_zero(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_diag_mask_zero_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_soft_max_back_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // rotary position embedding
    // if mode & 1 == 1, skip n_past elements (DEPRECATED)
    // if mode & 2 == 1, GPT-NeoX style
    // if mode & 4 == 1, ChatGLM style
    //
    // b is an int32 vector with size a->ne[2], it contains the positions
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx);

    // custom RoPE
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_custom(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_custom_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale);

    // xPos RoPE, in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_xpos_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            float                 base,
            bool                  down);

    // rotary position embedding backward, i.e compute dx from dy
    // a - dy
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_rope_back(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   n_dims,
            int                   mode,
            int                   n_ctx,
            float                 freq_base,
            float                 freq_scale,
            float                 xpos_base,
            bool                  xpos_down);

    // alibi position embedding
    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_alibi(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   n_past,
            int                   n_head,
            float                 bias_max);

    // clamp
    // in-place, returns view(a)
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_clamp(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            float                 min,
            float                 max);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   s0,  // stride
            int                   p0,  // padding
            int                   d0); // dilation

    // conv_1d with padding = half
    // alias for lm_ggml_conv_1d(a, b, s, a->ne[0]/2, d)
    LM_GGML_API struct lm_ggml_tensor* lm_ggml_conv_1d_ph(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   s,
            int                   d);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_transpose_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   s0,
            int                   p0,
            int                   d0);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1,
            int                   d0,
            int                   d1);


    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_sk_p0(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_2d_s1_ph(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_transpose_2d_p0(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b,
            int                   stride);

    enum lm_ggml_op_pool {
        LM_GGML_OP_POOL_MAX,
        LM_GGML_OP_POOL_AVG,
        LM_GGML_OP_POOL_COUNT,
    };

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pool_1d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum lm_ggml_op_pool     op,
            int                   k0, // kernel size
            int                   s0, // stride
            int                   p0); // padding

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_pool_2d(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            enum lm_ggml_op_pool     op,
            int                   k0,
            int                   k1,
            int                   s0,
            int                   s1,
            int                   p0,
            int                   p1);

    // nearest interpolate
    // used in stable-diffusion
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_upscale(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   scale_factor);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_flash_attn(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * q,
            struct lm_ggml_tensor  * k,
            struct lm_ggml_tensor  * v,
            bool                  masked);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_flash_attn_back(
           struct lm_ggml_context * ctx,
           struct lm_ggml_tensor  * q,
           struct lm_ggml_tensor  * k,
           struct lm_ggml_tensor  * v,
           struct lm_ggml_tensor  * d,
           bool                  masked);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_flash_ff(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * b0,
            struct lm_ggml_tensor  * b1,
            struct lm_ggml_tensor  * c0,
            struct lm_ggml_tensor  * c1);

    // partition into non-overlapping windows with padding if needed
    // example:
    // a:   768   64   64    1
    // w:    14
    // res: 768   14   14    25
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_win_part(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   w);

    // reverse of lm_ggml_win_part
    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_win_unpart(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   w0,
            int                   h0,
            int                   w);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_unary(
            struct lm_ggml_context * ctx,
             struct lm_ggml_tensor * a,
             enum lm_ggml_unary_op op);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_unary_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op op);

    // used in sam
    LM_GGML_API struct lm_ggml_tensor * lm_ggml_get_rel_pos(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            int                   qh,
            int                   kh);

    // used in sam

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_rel_pos(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * pw,
            struct lm_ggml_tensor  * ph);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_add_rel_pos_inplace(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * a,
            struct lm_ggml_tensor  * pw,
            struct lm_ggml_tensor  * ph);

    // custom operators

    typedef void (*lm_ggml_unary_op_f32_t) (const int, float *, const float *);
    typedef void (*lm_ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

    typedef void (*lm_ggml_custom1_op_f32_t)(struct lm_ggml_tensor *, const struct lm_ggml_tensor *);
    typedef void (*lm_ggml_custom2_op_f32_t)(struct lm_ggml_tensor *, const struct lm_ggml_tensor *, const struct lm_ggml_tensor *);
    typedef void (*lm_ggml_custom3_op_f32_t)(struct lm_ggml_tensor *, const struct lm_ggml_tensor *, const struct lm_ggml_tensor *, const struct lm_ggml_tensor *);

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_unary_f32(
            struct lm_ggml_context        * ctx,
            struct lm_ggml_tensor         * a,
                   lm_ggml_unary_op_f32_t   fun),
        "use lm_ggml_map_custom1 instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_unary_inplace_f32(
            struct lm_ggml_context        * ctx,
            struct lm_ggml_tensor         * a,
                   lm_ggml_unary_op_f32_t   fun),
        "use lm_ggml_map_custom1_inplace instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_binary_f32(
            struct lm_ggml_context         * ctx,
            struct lm_ggml_tensor          * a,
            struct lm_ggml_tensor          * b,
                   lm_ggml_binary_op_f32_t   fun),
        "use lm_ggml_map_custom2 instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_binary_inplace_f32(
            struct lm_ggml_context         * ctx,
            struct lm_ggml_tensor          * a,
            struct lm_ggml_tensor          * b,
                   lm_ggml_binary_op_f32_t   fun),
        "use lm_ggml_map_custom2_inplace instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
                   lm_ggml_custom1_op_f32_t   fun),
        "use lm_ggml_map_custom1 instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1_inplace_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
                   lm_ggml_custom1_op_f32_t   fun),
        "use lm_ggml_map_custom1_inplace instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
            struct lm_ggml_tensor           * b,
                   lm_ggml_custom2_op_f32_t   fun),
        "use lm_ggml_map_custom2 instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2_inplace_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
            struct lm_ggml_tensor           * b,
                   lm_ggml_custom2_op_f32_t   fun),
        "use lm_ggml_map_custom2_inplace instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
            struct lm_ggml_tensor           * b,
            struct lm_ggml_tensor           * c,
                   lm_ggml_custom3_op_f32_t   fun),
        "use lm_ggml_map_custom3 instead");

    LM_GGML_DEPRECATED(LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3_inplace_f32(
            struct lm_ggml_context          * ctx,
            struct lm_ggml_tensor           * a,
            struct lm_ggml_tensor           * b,
            struct lm_ggml_tensor           * c,
                   lm_ggml_custom3_op_f32_t   fun),
        "use lm_ggml_map_custom3_inplace instead");

    // custom operators v2

    typedef void (*lm_ggml_custom1_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, int ith, int nth, void * userdata);
    typedef void (*lm_ggml_custom2_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, const struct lm_ggml_tensor * b, int ith, int nth, void * userdata);
    typedef void (*lm_ggml_custom3_op_t)(struct lm_ggml_tensor * dst , const struct lm_ggml_tensor * a, const struct lm_ggml_tensor * b, const struct lm_ggml_tensor * c, int ith, int nth, void * userdata);

    #define LM_GGML_N_TASKS_MAX -1

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            lm_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom1_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            lm_ggml_custom1_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            lm_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom2_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            lm_ggml_custom2_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            struct lm_ggml_tensor    * c,
            lm_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_map_custom3_inplace(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_tensor    * a,
            struct lm_ggml_tensor    * b,
            struct lm_ggml_tensor    * c,
            lm_ggml_custom3_op_t       fun,
            int                     n_tasks,
            void                  * userdata);

    // loss function

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cross_entropy_loss(
            struct lm_ggml_context         * ctx,
            struct lm_ggml_tensor          * a,
            struct lm_ggml_tensor          * b);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_cross_entropy_loss_back(
            struct lm_ggml_context         * ctx,
            struct lm_ggml_tensor          * a,
            struct lm_ggml_tensor          * b,
            struct lm_ggml_tensor          * c);

    //
    // automatic differentiation
    //

    LM_GGML_API void lm_ggml_set_param(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * tensor);


    LM_GGML_API void lm_ggml_build_forward_expand (struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor);
    LM_GGML_API void lm_ggml_build_backward_expand(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * gf, struct lm_ggml_cgraph * gb, bool keep);

    LM_GGML_API struct lm_ggml_cgraph lm_ggml_build_forward (struct lm_ggml_tensor * tensor);
    LM_GGML_API struct lm_ggml_cgraph lm_ggml_build_backward(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * gf, bool keep);

    // graph allocation in a context
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_new_graph        (struct lm_ggml_context * ctx);
    LM_GGML_API struct lm_ggml_cgraph * lm_ggml_build_forward_ctx(struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor);
    LM_GGML_API size_t lm_ggml_graph_overhead(void);

    // lm_ggml_graph_plan() has to be called before lm_ggml_graph_compute()
    // when plan.work_size > 0, caller must allocate memory for plan.work_data
    LM_GGML_API struct lm_ggml_cplan lm_ggml_graph_plan   (struct lm_ggml_cgraph * cgraph, int n_threads /*= LM_GGML_DEFAULT_N_THREADS*/);
    LM_GGML_API               int lm_ggml_graph_compute(struct lm_ggml_cgraph * cgraph, struct lm_ggml_cplan * cplan);
    LM_GGML_API              void lm_ggml_graph_reset  (struct lm_ggml_cgraph * cgraph);

    // same as lm_ggml_graph_compute() but the work data is allocated as a part of the context
    // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    LM_GGML_API void lm_ggml_graph_compute_with_ctx(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, int n_threads);

    LM_GGML_API struct lm_ggml_tensor * lm_ggml_graph_get_tensor(struct lm_ggml_cgraph * cgraph, const char * name);

    LM_GGML_API void               lm_ggml_graph_export(const struct lm_ggml_cgraph * cgraph, const char * fname);
    LM_GGML_API struct lm_ggml_cgraph lm_ggml_graph_import(const char * fname, struct lm_ggml_context ** ctx_data, struct lm_ggml_context ** ctx_eval);

    // print info and performance information for the graph
    LM_GGML_API void lm_ggml_graph_print(const struct lm_ggml_cgraph * cgraph);

    // dump the graph into a file using the dot format
    LM_GGML_API void lm_ggml_graph_dump_dot(const struct lm_ggml_cgraph * gb, const struct lm_ggml_cgraph * gf, const char * filename);

    // build gradient checkpointing backward graph gb for gf using provided checkpoints
    // gb_tmp will contain original backward graph with rewritten backward process nodes,
    // but without the second forward pass nodes.
    LM_GGML_API void lm_ggml_build_backward_gradient_checkpointing(
            struct lm_ggml_context   * ctx,
            struct lm_ggml_cgraph    * gf,
            struct lm_ggml_cgraph    * gb,
            struct lm_ggml_cgraph    * gb_tmp,
            struct lm_ggml_tensor  * * checkpoints,
            int                     n_checkpoints);
    //
    // optimization
    //

    // optimization methods
    enum lm_ggml_opt_type {
        LM_GGML_OPT_ADAM,
        LM_GGML_OPT_LBFGS,
    };

    // linesearch methods
    enum lm_ggml_linesearch {
        LM_GGML_LINESEARCH_DEFAULT = 1,

        LM_GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
        LM_GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
        LM_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
    };

    // optimization return values
    enum lm_ggml_opt_result {
        LM_GGML_OPT_OK = 0,
        LM_GGML_OPT_DID_NOT_CONVERGE,
        LM_GGML_OPT_NO_CONTEXT,
        LM_GGML_OPT_INVALID_WOLFE,
        LM_GGML_OPT_FAIL,
        LM_GGML_OPT_CANCEL,

        LM_GGML_LINESEARCH_FAIL = -128,
        LM_GGML_LINESEARCH_MINIMUM_STEP,
        LM_GGML_LINESEARCH_MAXIMUM_STEP,
        LM_GGML_LINESEARCH_MAXIMUM_ITERATIONS,
        LM_GGML_LINESEARCH_INVALID_PARAMETERS,
    };

    typedef void (*lm_ggml_opt_callback)(void * data, int accum_step, float * sched, bool * cancel);
    typedef void (*lm_ggml_log_callback)(enum lm_ggml_log_level level, const char * text, void * user_data);

    // optimization parameters
    //
    //   see ggml.c (lm_ggml_opt_default_params) for default values
    //
    struct lm_ggml_opt_params {
        enum lm_ggml_opt_type type;

        int n_threads;

        // delta-based convergence test
        //
        //   if past == 0 - disabled
        //   if past > 0:
        //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
        //
        int past;
        float delta;

        // maximum number of iterations without improvement
        //
        //   if 0 - disabled
        //   if > 0:
        //     assume convergence if no cost improvement in this number of iterations
        //
        int max_no_improvement;

        bool print_forward_graph;
        bool print_backward_graph;

        int n_gradient_accumulation;

        // ADAM parameters
        struct {
            int n_iter;

            float sched; // schedule multiplier (fixed, decay or warmup)
            float decay; // weight decay for AdamW, use 0.0f to disable
            int   decay_min_ndim; // minimum number of tensor dimension to apply weight decay
            float alpha; // learning rate
            float beta1;
            float beta2;
            float eps;   // epsilon for numerical stability
            float eps_f; // epsilon for convergence test
            float eps_g; // epsilon for convergence test
            float gclip; // gradient clipping
        } adam;

        // LBFGS parameters
        struct {
            int m; // number of corrections to approximate the inv. Hessian
            int n_iter;
            int max_linesearch;

            float eps;      // convergence tolerance
            float ftol;     // line search tolerance
            float wolfe;
            float min_step;
            float max_step;

            enum lm_ggml_linesearch linesearch;
        } lbfgs;
    };

    struct lm_ggml_opt_context {
        struct lm_ggml_context * ctx;
        struct lm_ggml_opt_params params;

        int iter;
        int64_t nx; // number of parameter elements

        bool just_initialized;

        float loss_before;
        float loss_after;

        struct {
            struct lm_ggml_tensor * g;  // current gradient
            struct lm_ggml_tensor * m;  // first moment
            struct lm_ggml_tensor * v;  // second moment
            struct lm_ggml_tensor * pf; // past function values
            float fx_best;
            float fx_prev;
            int n_no_improvement;
        } adam;

        struct {
            struct lm_ggml_tensor * x;    // current parameters
            struct lm_ggml_tensor * xp;   // previous parameters
            struct lm_ggml_tensor * g;    // current gradient
            struct lm_ggml_tensor * gp;   // previous gradient
            struct lm_ggml_tensor * d;    // search direction
            struct lm_ggml_tensor * pf;   // past function values
            struct lm_ggml_tensor * lmal; // the L-BFGS memory alpha
            struct lm_ggml_tensor * lmys; // the L-BFGS memory ys
            struct lm_ggml_tensor * lms;  // the L-BFGS memory s
            struct lm_ggml_tensor * lmy;  // the L-BFGS memory y
            float fx_best;
            float step;
            int j;
            int k;
            int end;
            int n_no_improvement;
        } lbfgs;
    };

    LM_GGML_API struct lm_ggml_opt_params lm_ggml_opt_default_params(enum lm_ggml_opt_type type);

    // optimize the function defined by the tensor f
    LM_GGML_API enum lm_ggml_opt_result lm_ggml_opt(
            struct lm_ggml_context * ctx,
            struct lm_ggml_opt_params params,
            struct lm_ggml_tensor * f);

    // initialize optimizer context
    LM_GGML_API void lm_ggml_opt_init(
            struct lm_ggml_context     * ctx,
            struct lm_ggml_opt_context * opt,
            struct lm_ggml_opt_params    params,
            int64_t                   nx);

    // continue optimizing the function defined by the tensor f
    LM_GGML_API enum lm_ggml_opt_result lm_ggml_opt_resume(
            struct lm_ggml_context * ctx,
            struct lm_ggml_opt_context * opt,
            struct lm_ggml_tensor * f);

    // continue optimizing the function defined by the tensor f
    LM_GGML_API enum lm_ggml_opt_result lm_ggml_opt_resume_g(
            struct lm_ggml_context * ctx,
            struct lm_ggml_opt_context * opt,
            struct lm_ggml_tensor * f,
            struct lm_ggml_cgraph * gf,
            struct lm_ggml_cgraph * gb,
            lm_ggml_opt_callback callback,
            void * callback_data);

    //
    // quantization
    //

    LM_GGML_API size_t lm_ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
    LM_GGML_API size_t lm_ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
    LM_GGML_API size_t lm_ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
    LM_GGML_API size_t lm_ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
    LM_GGML_API size_t lm_ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);

    LM_GGML_API size_t lm_ggml_quantize_chunk(enum lm_ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);

    //
    // gguf
    //

    enum gguf_type {
        GGUF_TYPE_UINT8   = 0,
        GGUF_TYPE_INT8    = 1,
        GGUF_TYPE_UINT16  = 2,
        GGUF_TYPE_INT16   = 3,
        GGUF_TYPE_UINT32  = 4,
        GGUF_TYPE_INT32   = 5,
        GGUF_TYPE_FLOAT32 = 6,
        GGUF_TYPE_BOOL    = 7,
        GGUF_TYPE_STRING  = 8,
        GGUF_TYPE_ARRAY   = 9,
        GGUF_TYPE_UINT64  = 10,
        GGUF_TYPE_INT64   = 11,
        GGUF_TYPE_FLOAT64 = 12,
        GGUF_TYPE_COUNT,       // marks the end of the enum
    };

    struct gguf_context;

    struct gguf_init_params {
        bool no_alloc;

        // if not NULL, create a lm_ggml_context and allocate the tensor data in it
        struct lm_ggml_context ** ctx;
    };

    LM_GGML_API struct gguf_context * gguf_init_empty(void);
    LM_GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
    //LM_GGML_API struct gguf_context * gguf_init_from_buffer(..);

    LM_GGML_API void gguf_free(struct gguf_context * ctx);

    LM_GGML_API const char * gguf_type_name(enum gguf_type type);

    LM_GGML_API int    gguf_get_version    (const struct gguf_context * ctx);
    LM_GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
    LM_GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
    LM_GGML_API void * gguf_get_data       (const struct gguf_context * ctx);

    LM_GGML_API int          gguf_get_n_kv(const struct gguf_context * ctx);
    LM_GGML_API int          gguf_find_key(const struct gguf_context * ctx, const char * key);
    LM_GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int key_id);

    LM_GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int key_id);
    LM_GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int key_id);

    // will abort if the wrong type is used for the key
    LM_GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int key_id);
    LM_GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int key_id);
    LM_GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int key_id);
    LM_GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int key_id);
    LM_GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int key_id);
    LM_GGML_API int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);
    LM_GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int key_id);
    LM_GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int key_id, int i);

    LM_GGML_API int    gguf_get_n_tensors    (const struct gguf_context * ctx);
    LM_GGML_API int    gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
    LM_GGML_API size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int i);
    LM_GGML_API char * gguf_get_tensor_name  (const struct gguf_context * ctx, int i);

    // overrides existing values or adds a new one
    LM_GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
    LM_GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);
    LM_GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
    LM_GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);
    LM_GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
    LM_GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);
    LM_GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);
    LM_GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
    LM_GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);
    LM_GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);
    LM_GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);
    LM_GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
    LM_GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n);
    LM_GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, int n);

    // set or add KV pairs from another context
    LM_GGML_API void gguf_set_kv(struct gguf_context * ctx, struct gguf_context * src);

    // manage tensor info
    LM_GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct lm_ggml_tensor * tensor);
    LM_GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum lm_ggml_type type);
    LM_GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data, size_t size);

    // writing gguf files can be done in 2 ways:
    //
    // - write the entire gguf_context to a binary file in a single pass:
    //
    //   gguf_write_to_file(ctx, fname);
    //
    // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
    //
    //   FILE * f = fopen(fname, "wb");
    //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
    //   fwrite(f, ...);
    //   void * data = gguf_meta_get_meta_data(ctx);
    //   fseek(f, 0, SEEK_SET);
    //   fwrite(f, data, gguf_get_meta_size(ctx));
    //   free(data);
    //   fclose(f);
    //

    // write the entire context to a binary file
    LM_GGML_API void gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);

    // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
    LM_GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
    LM_GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);

    //
    // system info
    //

    LM_GGML_API int lm_ggml_cpu_has_avx        (void);
    LM_GGML_API int lm_ggml_cpu_has_avx2       (void);
    LM_GGML_API int lm_ggml_cpu_has_avx512     (void);
    LM_GGML_API int lm_ggml_cpu_has_avx512_vbmi(void);
    LM_GGML_API int lm_ggml_cpu_has_avx512_vnni(void);
    LM_GGML_API int lm_ggml_cpu_has_fma        (void);
    LM_GGML_API int lm_ggml_cpu_has_neon       (void);
    LM_GGML_API int lm_ggml_cpu_has_arm_fma    (void);
    LM_GGML_API int lm_ggml_cpu_has_metal      (void);
    LM_GGML_API int lm_ggml_cpu_has_f16c       (void);
    LM_GGML_API int lm_ggml_cpu_has_fp16_va    (void);
    LM_GGML_API int lm_ggml_cpu_has_wasm_simd  (void);
    LM_GGML_API int lm_ggml_cpu_has_blas       (void);
    LM_GGML_API int lm_ggml_cpu_has_cublas     (void);
    LM_GGML_API int lm_ggml_cpu_has_clblast    (void);
    LM_GGML_API int lm_ggml_cpu_has_gpublas    (void);
    LM_GGML_API int lm_ggml_cpu_has_sse3       (void);
    LM_GGML_API int lm_ggml_cpu_has_ssse3      (void);
    LM_GGML_API int lm_ggml_cpu_has_vsx        (void);

    //
    // Internal types and functions exposed for tests and benchmarks
    //

#ifdef  __cplusplus
// restrict not standard in C++
#define LM_GGML_RESTRICT
#else
#define LM_GGML_RESTRICT restrict
#endif
    typedef void (*lm_ggml_to_float_t)  (const void  * LM_GGML_RESTRICT x, float * LM_GGML_RESTRICT y, int k);
    typedef void (*lm_ggml_from_float_t)(const float * LM_GGML_RESTRICT x, void  * LM_GGML_RESTRICT y, int k);
    typedef void (*lm_ggml_vec_dot_t)   (const int n, float * LM_GGML_RESTRICT s, const void * LM_GGML_RESTRICT x, const void * LM_GGML_RESTRICT y);

    typedef struct {
        const char      * type_name;
        int               blck_size;
        size_t            type_size;
        bool              is_quantized;
        lm_ggml_to_float_t   to_float;
        lm_ggml_from_float_t from_float;
        lm_ggml_from_float_t from_float_reference;
        lm_ggml_vec_dot_t    vec_dot;
        enum lm_ggml_type    vec_dot_type;
    } lm_ggml_type_traits_t;

    LM_GGML_API lm_ggml_type_traits_t lm_ggml_internal_get_type_traits(enum lm_ggml_type type);

#ifdef  __cplusplus
}
#endif
