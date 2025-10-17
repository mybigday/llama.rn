#include "ggml-impl.h"
#include "ggml-blas.h"
#include "ggml-backend-impl.h"

#include <future>
#include <vector>
#include <cstring>

#if defined(LM_GGML_BLAS_USE_ACCELERATE)
#   include <Accelerate/Accelerate.h>
#elif defined(LM_GGML_BLAS_USE_MKL)
#   include <mkl.h>
#elif defined(LM_GGML_BLAS_USE_BLIS)
#   include <blis.h>
#elif defined(LM_GGML_BLAS_USE_NVPL)
#   include <nvpl_blas.h>
#else
#   include <cblas.h>
#endif

struct lm_ggml_backend_blas_context {
    int n_threads = LM_GGML_DEFAULT_N_THREADS;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
#ifndef LM_GGML_USE_OPENMP
    std::vector<std::future<void>> tasks;
#endif
};

static void lm_ggml_backend_blas_mul_mat(lm_ggml_backend_blas_context * ctx, struct lm_ggml_tensor * dst) {
    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const enum lm_ggml_type type = src0->type;

    LM_GGML_ASSERT(ne0 == ne01);
    LM_GGML_ASSERT(ne1 == ne11);
    LM_GGML_ASSERT(ne2 == ne12);
    LM_GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));
    LM_GGML_ASSERT(nb10 == lm_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    const int64_t ne_plane      = ne01*ne00;
    const size_t  desired_wsize = type == LM_GGML_TYPE_F32 ? 0 : ne03*ne02*ne_plane*sizeof(float);

    if (ctx->work_size < desired_wsize) {
        ctx->work_data.reset(new char[desired_wsize]);
        ctx->work_size = desired_wsize;
    }
    void * wdata = ctx->work_data.get();

    // convert src0 to float
    if (type != LM_GGML_TYPE_F32) {
        const auto * type_traits = lm_ggml_get_type_traits(type);
        lm_ggml_to_float_t const to_float = type_traits->to_float;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                const void  *       x      = (char *)  src0->data + i02*nb02          + i03*nb03;
                      float * const wplane = (float *) wdata      + i02*ne_plane      + i03*ne02*ne_plane;

                const int min_cols_per_thread = 4096;
                const int min_rows_per_thread = std::max((int)(min_cols_per_thread/ne00), 1);
                const int n_threads = std::max(std::min(ctx->n_threads, (int)(ne01/min_rows_per_thread)), 1);

#ifdef LM_GGML_USE_OPENMP
                #pragma omp parallel for num_threads(n_threads)
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                }
#else
                for (int i = 1; i < n_threads; i++) {
                    const int64_t start =       i*ne01/n_threads;
                    const int64_t end   = (i + 1)*ne01/n_threads;
                    if (start < end) {
                        ctx->tasks.push_back(std::async(std::launch::async, [=]() {
                            for (int64_t i01 = start; i01 < end; i01++) {
                                to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                            }
                        }));
                    }
                }
                {
                    // reuse the current thread for the first task
                    const int64_t start = 0;
                    const int64_t end   = ne01/n_threads;
                    for (int64_t i01 = start; i01 < end; i01++) {
                        to_float((const char *) x + i01*nb01, wplane + i01*ne00, ne00);
                    }
                }
#endif
            }
        }

#ifndef LM_GGML_USE_OPENMP
        // wait for all tasks to finish
        for (auto & task : ctx->tasks) {
            task.get();
        }
        ctx->tasks.clear();
#endif
    }

#if defined(OPENBLAS_VERSION)
    openblas_set_num_threads(ctx->n_threads);
#endif

#if defined(LM_GGML_BLAS_USE_BLIS)
    bli_thread_set_num_threads(ctx->n_threads);
#endif

#if defined(LM_GGML_BLAS_USE_NVPL)
    nvpl_blas_set_num_threads(ctx->n_threads);
#endif

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const float * x = (float *) ((char *) src0->data + i02*nb02 + i03*nb03);
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
                  float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != LM_GGML_TYPE_F32) {
                x = (float *) wdata + i02*ne_plane + i03*ne02*ne_plane;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,   y, ne10,
                                x, ne00,
                        0.0f,   d, ne01);
        }
    }
}

static void lm_ggml_backend_blas_out_prod(lm_ggml_backend_blas_context * ctx, struct lm_ggml_tensor * dst) {
    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT(ne0  == ne00);
    LM_GGML_ASSERT(ne1  == ne10);
    LM_GGML_ASSERT(ne2  == ne02);
    LM_GGML_ASSERT(ne02 == ne12);
    LM_GGML_ASSERT(ne3  == ne13);
    LM_GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    // LM_GGML_ASSERT(nb0 <= nb1);
    // LM_GGML_ASSERT(nb1 <= nb2);
    // LM_GGML_ASSERT(nb2 <= nb3);

    // Arguments to lm_ggml_compute_forward_out_prod (expressed as major,minor)
    // src0: (k,n)
    // src1: (k,m)
    // dst:  (m,n)
    //
    // Arguments to sgemm (see https://github.com/Reference-LAPACK/lapack/blob/master/BLAS/SRC/sgemm.f)
    // Also expressed as (major,minor)
    // a: (m,k): so src1 transposed
    // b: (k,n): so src0
    // c: (m,n)
    //
    // However, if lm_ggml_is_transposed(src1) is true, then
    // src1->data already contains a transposed version, so sgemm mustn't
    // transpose it further.

    int n = src0->ne[0];
    int k = src0->ne[1];
    int m = src1->ne[0];

    CBLAS_TRANSPOSE transposeA;
    int lda;

    if (!lm_ggml_is_transposed(src1)) {
        transposeA = CblasTrans;
        lda = m;
    } else {
        transposeA = CblasNoTrans;
        lda = k;
    }

    float * a = (float *) ((char *) src1->data);
    float * b = (float *) ((char *) src0->data);
    float * c = (float *) ((char *) dst->data);

    cblas_sgemm(CblasRowMajor, transposeA, CblasNoTrans, m, n, k, 1.0, a, lda, b, n, 0.0, c, n);

    LM_GGML_UNUSED(ctx);
}

// backend interface

static const char * lm_ggml_backend_blas_get_name(lm_ggml_backend_t backend) {
    return "BLAS";

    LM_GGML_UNUSED(backend);
}

static void lm_ggml_backend_blas_free(lm_ggml_backend_t backend) {
    lm_ggml_backend_blas_context * ctx = (lm_ggml_backend_blas_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum lm_ggml_status lm_ggml_backend_blas_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    lm_ggml_backend_blas_context * ctx = (lm_ggml_backend_blas_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        switch (node->op) {
            case LM_GGML_OP_MUL_MAT:
                lm_ggml_backend_blas_mul_mat(ctx, node);
                break;

            case LM_GGML_OP_OUT_PROD:
                lm_ggml_backend_blas_out_prod(ctx, node);
                break;

            case LM_GGML_OP_NONE:
            case LM_GGML_OP_RESHAPE:
            case LM_GGML_OP_VIEW:
            case LM_GGML_OP_PERMUTE:
            case LM_GGML_OP_TRANSPOSE:
                break;

            default:
                LM_GGML_ABORT("%s: unsupported op %s\n", __func__, lm_ggml_op_desc(node));
        }
    }

    return LM_GGML_STATUS_SUCCESS;

    LM_GGML_UNUSED(backend);
}

static struct lm_ggml_backend_i blas_backend_i = {
    /* .get_name                = */ lm_ggml_backend_blas_get_name,
    /* .free                    = */ lm_ggml_backend_blas_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ lm_ggml_backend_blas_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static lm_ggml_guid_t lm_ggml_backend_blas_guid(void) {
    static lm_ggml_guid guid = { 0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97, 0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d };
    return &guid;
}

lm_ggml_backend_t lm_ggml_backend_blas_init(void) {
    lm_ggml_backend_blas_context * ctx = new lm_ggml_backend_blas_context;

    lm_ggml_backend_t backend = new lm_ggml_backend {
        /* .guid    = */ lm_ggml_backend_blas_guid(),
        /* .iface   = */ blas_backend_i,
        /* .device  = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_blas_reg(), 0),
        /* .context = */ ctx,
    };

#if defined(OPENBLAS_VERSION) && defined(LM_GGML_USE_OPENMP)
    if (openblas_get_parallel() != OPENBLAS_OPENMP) {
        LM_GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but OpenBLAS was compiled without OpenMP support\n", __func__);
    }
#endif

#if defined(BLIS_ENABLE_CBLAS) && defined(LM_GGML_USE_OPENMP) && !defined(BLIS_ENABLE_OPENMP)
    LM_GGML_LOG_DEBUG("%s: warning: ggml is using OpenMP, but BLIS was compiled without OpenMP support\n", __func__);
#endif

    return backend;
}

bool lm_ggml_backend_is_blas(lm_ggml_backend_t backend) {
    return backend != NULL && lm_ggml_guid_matches(backend->guid, lm_ggml_backend_blas_guid());
}

void lm_ggml_backend_blas_set_n_threads(lm_ggml_backend_t backend_blas, int n_threads) {
    LM_GGML_ASSERT(lm_ggml_backend_is_blas(backend_blas));

    lm_ggml_backend_blas_context * ctx = (lm_ggml_backend_blas_context *)backend_blas->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * lm_ggml_backend_blas_device_get_name(lm_ggml_backend_dev_t dev) {
    return "BLAS";

    LM_GGML_UNUSED(dev);
}

static const char * lm_ggml_backend_blas_device_get_description(lm_ggml_backend_dev_t dev) {
    #if defined(LM_GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(LM_GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(LM_GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(LM_GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return "BLAS";
    #endif

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_blas_device_get_memory(lm_ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    *free = 0;
    *total = 0;

    LM_GGML_UNUSED(dev);
}

static enum lm_ggml_backend_dev_type lm_ggml_backend_blas_device_get_type(lm_ggml_backend_dev_t dev) {
    return LM_GGML_BACKEND_DEVICE_TYPE_ACCEL;

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_blas_device_get_props(lm_ggml_backend_dev_t dev, struct lm_ggml_backend_dev_props * props) {
    props->name        = lm_ggml_backend_blas_device_get_name(dev);
    props->description = lm_ggml_backend_blas_device_get_description(dev);
    props->type        = lm_ggml_backend_blas_device_get_type(dev);
    lm_ggml_backend_blas_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static lm_ggml_backend_t lm_ggml_backend_blas_device_init_backend(lm_ggml_backend_dev_t dev, const char * params) {
    return lm_ggml_backend_blas_init();

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(params);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_blas_device_get_buffer_type(lm_ggml_backend_dev_t dev) {
    return lm_ggml_backend_cpu_buffer_type();

    LM_GGML_UNUSED(dev);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_blas_device_buffer_from_host_ptr(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return lm_ggml_backend_cpu_buffer_from_ptr(ptr, size);

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(max_tensor_size);
}

static bool lm_ggml_backend_blas_device_supports_op(lm_ggml_backend_dev_t dev, const struct lm_ggml_tensor * op) {
    const struct lm_ggml_tensor * src0 = op->src[0];
    const struct lm_ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case LM_GGML_OP_NONE:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_PERMUTE:
        case LM_GGML_OP_TRANSPOSE:
            return true;

        case LM_GGML_OP_MUL_MAT:
        {
            // BLAS usually is only faster for large matrices
            const struct lm_ggml_tensor * src0 = op->src[0];
            const struct lm_ggml_tensor * src1 = op->src[1];

            const int64_t ne10 = src1->ne[0];

            const int64_t ne0 = op->ne[0];
            const int64_t ne1 = op->ne[1];

            // TODO: find the optimal value
            const int64_t min_batch = 32;

            return lm_ggml_is_contiguous(src0) &&
                   lm_ggml_is_contiguous(src1) &&
                   src1->type == LM_GGML_TYPE_F32 &&
                   (ne0 >= min_batch && ne1 >= min_batch && ne10 >= min_batch) &&
                   (src0->type == LM_GGML_TYPE_F32 || lm_ggml_get_type_traits(src0->type)->to_float != NULL);
        }

        case LM_GGML_OP_OUT_PROD:
            return op->src[0]->type == LM_GGML_TYPE_F32 &&
                   op->src[1]->type == LM_GGML_TYPE_F32 &&
                   lm_ggml_is_matrix(src0) &&
                   lm_ggml_is_matrix(src1) &&
                   lm_ggml_is_contiguous(src0) &&
                   (lm_ggml_is_contiguous(src1) || lm_ggml_is_transposed(src1)) &&
                   (src0->type == LM_GGML_TYPE_F32 || lm_ggml_get_type_traits(src0->type)->to_float != NULL);

        default:
            return false;

    }

    LM_GGML_UNUSED(dev);
}

static bool lm_ggml_backend_blas_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    return lm_ggml_backend_buft_is_host(buft);

    LM_GGML_UNUSED(dev);
}

static const struct lm_ggml_backend_device_i lm_ggml_backend_blas_device_i = {
    /* .get_name             = */ lm_ggml_backend_blas_device_get_name,
    /* .get_description      = */ lm_ggml_backend_blas_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_blas_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_blas_device_get_type,
    /* .get_props            = */ lm_ggml_backend_blas_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_blas_device_init_backend,
    /* .get_buffer_type      = */ lm_ggml_backend_blas_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ lm_ggml_backend_blas_device_buffer_from_host_ptr,
    /* .supports_op          = */ lm_ggml_backend_blas_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_blas_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * lm_ggml_backend_blas_reg_get_name(lm_ggml_backend_reg_t reg) {
    return "BLAS";

    LM_GGML_UNUSED(reg);
}

static size_t lm_ggml_backend_blas_reg_get_device_count(lm_ggml_backend_reg_t reg) {
    return 1;

    LM_GGML_UNUSED(reg);
}

static lm_ggml_backend_dev_t lm_ggml_backend_blas_reg_get_device(lm_ggml_backend_reg_t reg, size_t index) {
    LM_GGML_ASSERT(index == 0);

    static lm_ggml_backend_device lm_ggml_backend_blas_device = {
        /* .iface   = */ lm_ggml_backend_blas_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &lm_ggml_backend_blas_device;

    LM_GGML_UNUSED(reg);
    LM_GGML_UNUSED(index);
}

static void * lm_ggml_backend_blas_get_proc_address(lm_ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "lm_ggml_backend_set_n_threads") == 0) {
        return (void *)lm_ggml_backend_blas_set_n_threads;
    }
    return NULL;

    LM_GGML_UNUSED(reg);
    LM_GGML_UNUSED(name);
}

static const struct lm_ggml_backend_reg_i lm_ggml_backend_blas_reg_i = {
    /* .get_name         = */ lm_ggml_backend_blas_reg_get_name,
    /* .get_device_count = */ lm_ggml_backend_blas_reg_get_device_count,
    /* .get_device       = */ lm_ggml_backend_blas_reg_get_device,
    /* .get_proc_address = */ lm_ggml_backend_blas_get_proc_address,
};

lm_ggml_backend_reg_t lm_ggml_backend_blas_reg(void) {
    static struct lm_ggml_backend_reg lm_ggml_backend_blas_reg = {
        /* .api_version = */ LM_GGML_BACKEND_API_VERSION,
        /* .iface       = */ lm_ggml_backend_blas_reg_i,
        /* .context     = */ NULL,
    };

    return &lm_ggml_backend_blas_reg;
}

LM_GGML_BACKEND_DL_IMPL(lm_ggml_backend_blas_reg)
