#include "amx.h"
#include "common.h"
#include "mmq.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "traits.h"

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

// AMX type_trais
namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits {
    bool work_size(int /* n_threads */, const struct lm_ggml_tensor * op, size_t & size) override {
        size = lm_ggml_backend_amx_desired_wsize(op);
        return true;
    }

    bool compute_forward(struct lm_ggml_compute_params * params, struct lm_ggml_tensor * op) override {
        if (op->op == LM_GGML_OP_MUL_MAT) {
            lm_ggml_backend_amx_mul_mat(params, op);
            return true;
        }
        return false;
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(lm_ggml_backend_buffer_t, struct lm_ggml_tensor *) {
    static tensor_traits traits;
    return &traits;
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
static void lm_ggml_backend_amx_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * lm_ggml_backend_amx_buffer_get_base(lm_ggml_backend_buffer_t buffer) {
    return (void *) (buffer->context);
}

static enum lm_ggml_status lm_ggml_backend_amx_buffer_init_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    LM_GGML_UNUSED(buffer);
    return LM_GGML_STATUS_SUCCESS;
}

static void lm_ggml_backend_amx_buffer_memset_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    LM_GGML_UNUSED(buffer);
}

static void lm_ggml_backend_amx_buffer_set_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    if (qtype_has_amx_kernels(tensor->type)) {
        LM_GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, lm_ggml_type_name(tensor->type));
        lm_ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    LM_GGML_UNUSED(buffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void lm_ggml_backend_amx_buffer_get_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    LM_GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    LM_GGML_UNUSED(buffer);
}

static bool lm_ggml_backend_amx_buffer_cpy_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    if (lm_ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            lm_ggml_backend_amx_convert_weight(dst, src->data, 0, lm_ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, lm_ggml_nbytes(src));
        }
        return true;
    }
    return false;

    LM_GGML_UNUSED(buffer);
}
*/

static void lm_ggml_backend_amx_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static lm_ggml_backend_buffer_i lm_ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ lm_ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ lm_ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ lm_ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ lm_ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ lm_ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ lm_ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * lm_ggml_backend_amx_buffer_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "AMX";

    LM_GGML_UNUSED(buft);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_amx_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    void * data = lm_ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return lm_ggml_backend_buffer_init(buft, lm_ggml_backend_amx_buffer_interface, data, size);
}

static size_t lm_ggml_backend_amx_buffer_type_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    LM_GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(lm_ggml_backend_dev_t, const struct lm_ggml_tensor * op) override {
        // handle only 2d gemm for now
        auto is_contiguous_2d = [](const struct lm_ggml_tensor * t) {
            return lm_ggml_is_contiguous(t) && t->ne[3] == 1 && t->ne[2] == 1;
        };

        if (op->op == LM_GGML_OP_MUL_MAT && is_contiguous_2d(op->src[0]) &&  // src0 must be contiguous
            is_contiguous_2d(op->src[1]) &&                               // src1 must be contiguous
            op->src[0]->buffer && op->src[0]->buffer->buft == lm_ggml_backend_amx_buffer_type() &&
            op->ne[0] % (TILE_N * 2) == 0 &&                              // out_features is 32x
            (qtype_has_amx_kernels(op->src[0]->type) || (op->src[0]->type == LM_GGML_TYPE_F16))) {
            // src1 must be host buffer
            if (op->src[1]->buffer && !lm_ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            // src1 must be float32
            if (op->src[1]->type == LM_GGML_TYPE_F32) {
                return true;
            }
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct lm_ggml_tensor * op) override {
        if (op->op == LM_GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == lm_ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;
    }
};
}  // namespace ggml::cpu::amx

static size_t lm_ggml_backend_amx_buffer_type_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    return lm_ggml_backend_amx_get_alloc_size(tensor);

    LM_GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

static bool lm_ggml_amx_init() {
#if defined(__gnu_linux__)
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;
    }
    return true;
#elif defined(_WIN32)
    return true;
#endif
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_amx_buffer_type() {
    static struct lm_ggml_backend_buffer_type lm_ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name         = */ lm_ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer     = */ lm_ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ lm_ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ lm_ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .is_host          = */ nullptr,
                        },
        /* .device  = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
    };

    if (!lm_ggml_amx_init()) {
        return nullptr;
    }

    return &lm_ggml_backend_buffer_type_amx;
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)
