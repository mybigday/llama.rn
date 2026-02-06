#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

#include <mutex>
#include <string>

#define LM_GGML_METAL_NAME "MTL"
#define LM_GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overriden with LM_GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void lm_ggml_backend_metal_buffer_shared_free_buffer(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_free(ctx);
}

static void * lm_ggml_backend_metal_buffer_shared_get_base(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    return lm_ggml_metal_buffer_get_base(ctx);
}

static void lm_ggml_backend_metal_buffer_shared_memset_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void lm_ggml_backend_metal_buffer_shared_set_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void lm_ggml_backend_metal_buffer_shared_get_tensor(lm_ggml_backend_buffer_t buffer, const lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool lm_ggml_backend_metal_buffer_shared_cpy_tensor(lm_ggml_backend_buffer_t buffer, const lm_ggml_tensor * src, lm_ggml_tensor * dst) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    LM_GGML_UNUSED(buffer);
    LM_GGML_UNUSED(src);
    LM_GGML_UNUSED(dst);

    return false;
}

static void lm_ggml_backend_metal_buffer_shared_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_clear(ctx, value);
}

static lm_ggml_backend_buffer_i lm_ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer     = */ lm_ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base        = */ lm_ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ lm_ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor      = */ lm_ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_metal_buffer_shared_get_tensor,
    /* .cpy_tensor      = */ lm_ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear           = */ lm_ggml_backend_metal_buffer_shared_clear,
    /* .reset           = */ NULL,
};

// private buffer

static void lm_ggml_backend_metal_buffer_private_free_buffer(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_free(ctx);
}

static void * lm_ggml_backend_metal_buffer_private_get_base(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    return lm_ggml_metal_buffer_get_base(ctx);
}

static void lm_ggml_backend_metal_buffer_private_memset_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void lm_ggml_backend_metal_buffer_private_set_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void lm_ggml_backend_metal_buffer_private_get_tensor(lm_ggml_backend_buffer_t buffer, const lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool lm_ggml_backend_metal_buffer_private_cpy_tensor(lm_ggml_backend_buffer_t buffer, const lm_ggml_tensor * src, lm_ggml_tensor * dst) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    LM_GGML_UNUSED(buffer);
    LM_GGML_UNUSED(src);
    LM_GGML_UNUSED(dst);

    return false;
}

static void lm_ggml_backend_metal_buffer_private_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    lm_ggml_metal_buffer_t ctx = (lm_ggml_metal_buffer_t)buffer->context;

    LM_GGML_ASSERT(!lm_ggml_metal_buffer_is_shared(ctx));

    lm_ggml_metal_buffer_clear(ctx, value);
}

static lm_ggml_backend_buffer_i lm_ggml_backend_metal_buffer_private_i = {
    /* .free_buffer     = */ lm_ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base        = */ lm_ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ lm_ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor      = */ lm_ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_metal_buffer_private_get_tensor,
    /* .cpy_tensor      = */ lm_ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear           = */ lm_ggml_backend_metal_buffer_private_clear,
    /* .reset           = */ NULL,
};

static bool lm_ggml_backend_buffer_is_metal(lm_ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == lm_ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == lm_ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

struct lm_ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct lm_ggml_backend_metal_buffer_type_deleter {
    void operator()(lm_ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<lm_ggml_backend_metal_buffer_type, lm_ggml_backend_metal_buffer_type_deleter> lm_ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)buft->device->context;
    lm_ggml_metal_buffer_t res = lm_ggml_metal_buffer_init(ctx_dev, size, shared);

    lm_ggml_backend_buffer_i buf_i = lm_ggml_metal_buffer_is_shared(res)
        ? lm_ggml_backend_metal_buffer_shared_i
        : lm_ggml_backend_metal_buffer_private_i;

    return lm_ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t lm_ggml_backend_metal_buffer_type_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    size_t res = lm_ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case LM_GGML_OP_MUL_MAT_ID:
            {
                res += lm_ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += lm_ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case LM_GGML_OP_FLASH_ATTN_EXT:
            {
                res += lm_ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += lm_ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += lm_ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
            } break;
        case LM_GGML_OP_CUMSUM:
        case LM_GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case LM_GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*lm_ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    LM_GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * lm_ggml_backend_metal_buffer_type_shared_get_name(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_backend_metal_buffer_type * ctx = (lm_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_type_shared_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    return lm_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t lm_ggml_backend_metal_buffer_type_shared_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return 32;

    LM_GGML_UNUSED(buft);
}

static size_t lm_ggml_backend_metal_buffer_type_shared_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)buft->device->context;

    return lm_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t lm_ggml_backend_metal_buffer_type_shared_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    return lm_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool lm_ggml_backend_metal_buffer_type_shared_is_host(lm_ggml_backend_buffer_type_t buft) {
    return false;

    LM_GGML_UNUSED(buft);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<lm_ggml_backend_buffer_type> bufts;
    static std::vector<lm_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            lm_ggml_backend_metal_buffer_type * raw_ctx =
                new lm_ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ LM_GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            lm_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ lm_ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .is_host          = */ lm_ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * lm_ggml_backend_metal_buffer_type_private_get_name(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_backend_metal_buffer_type * ctx = (lm_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_type_private_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    return lm_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t lm_ggml_backend_metal_buffer_type_private_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return 32;

    LM_GGML_UNUSED(buft);
}

static size_t lm_ggml_backend_metal_buffer_type_private_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)buft->device->context;

    return lm_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t lm_ggml_backend_metal_buffer_type_private_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    return lm_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool lm_ggml_backend_metal_buffer_type_private_is_host(lm_ggml_backend_buffer_type_t buft) {
    return false;

    LM_GGML_UNUSED(buft);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<lm_ggml_backend_buffer_type> bufts;
    static std::vector<lm_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            lm_ggml_backend_metal_buffer_type * raw_ctx = new lm_ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ LM_GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            lm_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ lm_ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .is_host          = */ lm_ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * lm_ggml_backend_metal_buffer_type_mapped_get_name(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_backend_metal_buffer_type * ctx = (lm_ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_buffer_type_mapped_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return lm_ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t lm_ggml_backend_metal_buffer_type_mapped_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return 32;

    LM_GGML_UNUSED(buft);
}

static size_t lm_ggml_backend_metal_buffer_type_mapped_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)buft->device->context;

    return lm_ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t lm_ggml_backend_metal_buffer_type_mapped_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    return lm_ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool lm_ggml_backend_metal_buffer_type_mapped_is_host(lm_ggml_backend_buffer_type_t buft) {
    return false;

    LM_GGML_UNUSED(buft);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<lm_ggml_backend_buffer_type> bufts;
    static std::vector<lm_ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            lm_ggml_backend_metal_buffer_type * raw_ctx = new lm_ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ LM_GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            lm_ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ lm_ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .is_host          = */ lm_ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ lm_ggml_backend_reg_dev_get(lm_ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * lm_ggml_backend_metal_name(lm_ggml_backend_t backend) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    return lm_ggml_metal_get_name(ctx);
}

static void lm_ggml_backend_metal_free(lm_ggml_backend_t backend) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    lm_ggml_metal_synchronize(ctx);

    lm_ggml_metal_free(ctx);

    free(backend);
}

static void lm_ggml_backend_metal_synchronize(lm_ggml_backend_t backend) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_synchronize(ctx);
}

static void lm_ggml_backend_metal_set_tensor_async(lm_ggml_backend_t backend, lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void lm_ggml_backend_metal_get_tensor_async(lm_ggml_backend_t backend, const lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool lm_ggml_backend_metal_cpy_tensor_async(lm_ggml_backend_t backend_src, lm_ggml_backend_t backend_dst, const lm_ggml_tensor * src, lm_ggml_tensor * dst) {
    if (!lm_ggml_backend_is_metal(backend_src) || !lm_ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!lm_ggml_backend_buffer_is_metal(src->buffer) || !lm_ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    lm_ggml_metal_t ctx_src = (lm_ggml_metal_t)backend_src->context;
    lm_ggml_metal_t ctx_dst = (lm_ggml_metal_t)backend_dst->context;

    //lm_ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //lm_ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //lm_ggml_metal_buffer_t buf_ctx_src = (lm_ggml_metal_buffer_t)buf_src->context;
    //lm_ggml_metal_buffer_t buf_ctx_dst = (lm_ggml_metal_buffer_t)buf_dst->context;

    return lm_ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

static enum lm_ggml_status lm_ggml_backend_metal_graph_compute(lm_ggml_backend_t backend, lm_ggml_cgraph * cgraph) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    return lm_ggml_metal_graph_compute(ctx, cgraph);
}

static void lm_ggml_backend_metal_event_record(lm_ggml_backend_t backend, lm_ggml_backend_event_t event) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;
    lm_ggml_metal_event_t ev = (lm_ggml_metal_event_t)event->context;

    lm_ggml_metal_event_record(ctx, ev);
}

static void lm_ggml_backend_metal_event_wait(lm_ggml_backend_t backend, lm_ggml_backend_event_t event) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;
    lm_ggml_metal_event_t ev = (lm_ggml_metal_event_t)event->context;

    lm_ggml_metal_event_wait(ctx, ev);
}

static void lm_ggml_backend_metal_graph_optimize(lm_ggml_backend_t backend, lm_ggml_cgraph * cgraph) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_graph_optimize(ctx, cgraph);
}

static void lm_ggml_backend_metal_set_n_cb(lm_ggml_backend_t backend, int n_cb) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_set_n_cb(ctx, n_cb);
}

static lm_ggml_backend_i lm_ggml_backend_metal_i = {
    /* .get_name                = */ lm_ggml_backend_metal_name,
    /* .free                    = */ lm_ggml_backend_metal_free,
    /* .set_tensor_async        = */ lm_ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ lm_ggml_backend_metal_get_tensor_async,
    /* .cpy_tensor_async        = */ lm_ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ lm_ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ lm_ggml_backend_metal_graph_compute,
    /* .event_record            = */ lm_ggml_backend_metal_event_record,
    /* .event_wait              = */ lm_ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ lm_ggml_backend_metal_graph_optimize,
};

static lm_ggml_guid_t lm_ggml_backend_metal_guid(void) {
    static lm_ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

lm_ggml_backend_t lm_ggml_backend_metal_init(void) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_reg_dev_get(lm_ggml_backend_metal_reg(), 0);
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_t ctx = lm_ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        LM_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    lm_ggml_backend_t backend = (lm_ggml_backend_t) malloc(sizeof(lm_ggml_backend));

    *backend = {
        /* .guid      = */ lm_ggml_backend_metal_guid(),
        /* .interface = */ lm_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    lm_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool lm_ggml_backend_is_metal(lm_ggml_backend_t backend) {
    return backend != NULL && lm_ggml_guid_matches(backend->guid, lm_ggml_backend_metal_guid());
}

void lm_ggml_backend_metal_set_abort_callback(lm_ggml_backend_t backend, lm_ggml_abort_callback abort_callback, void * user_data) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool lm_ggml_backend_metal_supports_family(lm_ggml_backend_t backend, int family) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    return lm_ggml_metal_supports_family(ctx, family);
}

void lm_ggml_backend_metal_capture_next_compute(lm_ggml_backend_t backend) {
    LM_GGML_ASSERT(lm_ggml_backend_is_metal(backend));

    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    lm_ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * lm_ggml_backend_metal_device_get_name(lm_ggml_backend_dev_t dev) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    const lm_ggml_metal_device_props * props_dev = lm_ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * lm_ggml_backend_metal_device_get_description(lm_ggml_backend_dev_t dev) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    return lm_ggml_metal_device_get_props(ctx_dev)->desc;
}

static void lm_ggml_backend_metal_device_get_memory(lm_ggml_backend_dev_t dev, size_t * free, size_t * total) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum lm_ggml_backend_dev_type lm_ggml_backend_metal_device_get_type(lm_ggml_backend_dev_t dev) {
    return LM_GGML_BACKEND_DEVICE_TYPE_GPU;

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_metal_device_get_props(lm_ggml_backend_dev_t dev, lm_ggml_backend_dev_props * props) {
    props->name        = lm_ggml_backend_metal_device_get_name(dev);
    props->description = lm_ggml_backend_metal_device_get_description(dev);
    props->type        = lm_ggml_backend_metal_device_get_type(dev);

    lm_ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
    };
}

static lm_ggml_backend_t lm_ggml_backend_metal_device_init_backend(lm_ggml_backend_dev_t dev, const char * params) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_t ctx = lm_ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        LM_GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    lm_ggml_backend_t backend = (lm_ggml_backend_t) malloc(sizeof(lm_ggml_backend));

    *backend = {
        /* .guid      = */ lm_ggml_backend_metal_guid(),
        /* .interface = */ lm_ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    lm_ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    LM_GGML_UNUSED(params);
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_device_get_buffer_type(lm_ggml_backend_dev_t dev) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    const lm_ggml_metal_device_props * props_dev = lm_ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? lm_ggml_backend_metal_buffer_type_shared(props_dev->device) : lm_ggml_backend_metal_buffer_type_private(props_dev->device);
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_device_buffer_mapped(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_buffer_t res = lm_ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const lm_ggml_metal_device_props * props_dev = lm_ggml_metal_device_get_props(ctx_dev);

    return lm_ggml_backend_buffer_init(lm_ggml_backend_metal_buffer_type_mapped(props_dev->device), lm_ggml_backend_metal_buffer_shared_i, res, size);
}

static bool lm_ggml_backend_metal_device_supports_op(lm_ggml_backend_dev_t dev, const lm_ggml_tensor * op) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    return lm_ggml_metal_device_supports_op(ctx_dev, op);
}

static bool lm_ggml_backend_metal_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_mapped_get_name);

    LM_GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const lm_ggml_tensor * op) {
    switch (op->op) {
        case LM_GGML_OP_MUL_MAT:
            return op->ne[1];
        case LM_GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return lm_ggml_nrows(op);
    }
}

static bool lm_ggml_backend_metal_device_offload_op(lm_ggml_backend_dev_t dev, const lm_ggml_tensor * op) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    return (op->op == LM_GGML_OP_MUL_MAT ||
            op->op == LM_GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= lm_ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static lm_ggml_backend_event_t lm_ggml_backend_metal_device_event_new(lm_ggml_backend_dev_t dev) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_event_t event = lm_ggml_metal_device_event_init(ctx_dev);
    LM_GGML_ASSERT(event);

    lm_ggml_backend_event_t ev = new lm_ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void lm_ggml_backend_metal_device_event_free(lm_ggml_backend_dev_t dev, lm_ggml_backend_event_t event) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_event_t ev = (lm_ggml_metal_event_t)event->context;

    lm_ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void lm_ggml_backend_metal_device_event_synchronize(lm_ggml_backend_dev_t dev, lm_ggml_backend_event_t event) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_event_t evt = (lm_ggml_metal_event_t)event->context;

    lm_ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static lm_ggml_backend_device_i lm_ggml_backend_metal_device_i = {
    /* .get_name             = */ lm_ggml_backend_metal_device_get_name,
    /* .get_description      = */ lm_ggml_backend_metal_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_metal_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_metal_device_get_type,
    /* .get_props            = */ lm_ggml_backend_metal_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ lm_ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ lm_ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ lm_ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ lm_ggml_backend_metal_device_offload_op,
    /* .event_new            = */ lm_ggml_backend_metal_device_event_new,
    /* .event_free           = */ lm_ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ lm_ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct lm_ggml_backend_metal_reg {
    std::vector<lm_ggml_backend_dev_t> devices;
};

typedef struct lm_ggml_backend_metal_reg * lm_ggml_backend_metal_reg_t;

static lm_ggml_backend_metal_reg_t lm_ggml_backend_metal_reg_init(void) {
    lm_ggml_backend_metal_reg_t ctx = new struct lm_ggml_backend_metal_reg;

    return ctx;
}

static void lm_ggml_backend_metal_reg_free(lm_ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct lm_ggml_backend_metal_reg_deleter {
    void operator()(lm_ggml_backend_metal_reg_t ctx) {
        lm_ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct lm_ggml_backend_metal_reg, lm_ggml_backend_metal_reg_deleter> lm_ggml_backend_metal_reg_ptr;

static const char * lm_ggml_backend_metal_reg_get_name(lm_ggml_backend_reg_t reg) {
    return LM_GGML_METAL_NAME;

    LM_GGML_UNUSED(reg);
}

static size_t lm_ggml_backend_metal_reg_device_count(lm_ggml_backend_reg_t reg) {
    lm_ggml_backend_metal_reg_t ctx = (lm_ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static lm_ggml_backend_dev_t lm_ggml_backend_metal_reg_device_get(lm_ggml_backend_reg_t reg, size_t index) {
    lm_ggml_backend_metal_reg_t ctx = (lm_ggml_backend_metal_reg_t)reg->context;
    LM_GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static lm_ggml_backend_feature g_lm_ggml_backend_metal_features[] = {
#if defined(LM_GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static lm_ggml_backend_feature * lm_ggml_backend_metal_get_features(lm_ggml_backend_reg_t reg) {
    return g_lm_ggml_backend_metal_features;

    LM_GGML_UNUSED(reg);
}

static void * lm_ggml_backend_metal_get_proc_address(lm_ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "lm_ggml_backend_get_features") == 0) {
        return (void *)lm_ggml_backend_metal_get_features;
    }

    return NULL;

    LM_GGML_UNUSED(reg);
}

static lm_ggml_backend_reg_i lm_ggml_backend_metal_reg_i = {
    /* .get_name         = */ lm_ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ lm_ggml_backend_metal_reg_device_count,
    /* .get_device       = */ lm_ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ lm_ggml_backend_metal_get_proc_address,
};

static lm_ggml_backend_dev_t lm_ggml_backend_metal_device_init(lm_ggml_backend_reg_t reg, int device) {
    return new lm_ggml_backend_device {
        /* .iface   = */ lm_ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ lm_ggml_metal_device_get(device),
    };
}

static void lm_ggml_backend_metal_device_free(lm_ggml_backend_dev_t dev) {
    delete dev;
}

struct lm_ggml_backend_device_deleter {
    void operator()(lm_ggml_backend_dev_t ctx) {
        lm_ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<lm_ggml_backend_device, lm_ggml_backend_device_deleter> lm_ggml_backend_device_ptr;

lm_ggml_backend_reg_t lm_ggml_backend_metal_reg(void) {
    static lm_ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("LM_GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<lm_ggml_backend_device_ptr> devs;

        if (!initialized) {
            static lm_ggml_backend_metal_reg_ptr reg_ctx(lm_ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = lm_ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ LM_GGML_BACKEND_API_VERSION,
                /* .iface       = */ lm_ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

LM_GGML_BACKEND_DL_IMPL(lm_ggml_backend_metal_reg)
