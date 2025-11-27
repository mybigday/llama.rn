#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

// globals

// initialized in lm_ggml_backend_metal_reg
static lm_ggml_backend_reg    g_lm_ggml_metal_reg;
static lm_ggml_backend_device g_lm_ggml_metal_device;

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

//
// buffer types
//

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
    return "Metal";

    LM_GGML_UNUSED(buft);
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

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_shared(void) {
    static lm_ggml_backend_buffer_type lm_ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_metal_buffer_type_shared_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_shared_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_shared_get_alignment,
            /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_shared_get_max_size,
            /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_shared_get_alloc_size,
            /* .is_host          = */ lm_ggml_backend_metal_buffer_type_shared_is_host,
        },
        /* .device  = */ &g_lm_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_buffer_type_metal;
}

// default (private) buffer type

static const char * lm_ggml_backend_metal_buffer_type_private_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "Metal_Private";

    LM_GGML_UNUSED(buft);
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

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_private(void) {
    static lm_ggml_backend_buffer_type lm_ggml_backend_buffer_type_metal = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_metal_buffer_type_private_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_private_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_private_get_alignment,
            /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_private_get_max_size,
            /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_private_get_alloc_size,
            /* .is_host          = */ lm_ggml_backend_metal_buffer_type_private_is_host,
        },
        /* .device  = */ &g_lm_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_buffer_type_metal;
}

// mapped buffer type

static const char * lm_ggml_backend_metal_buffer_type_mapped_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "Metal_Mapped";

    LM_GGML_UNUSED(buft);
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

static lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type_mapped(void) {
    // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
    //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
    static lm_ggml_backend_buffer_type lm_ggml_backend_buffer_type_mapped_metal = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_metal_buffer_type_mapped_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_metal_buffer_type_mapped_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_metal_buffer_type_mapped_get_alignment,
            /* .get_max_size     = */ lm_ggml_backend_metal_buffer_type_mapped_get_max_size,
            /* .get_alloc_size   = */ lm_ggml_backend_metal_buffer_type_mapped_get_alloc_size,
            /* .is_host          = */ lm_ggml_backend_metal_buffer_type_mapped_is_host,
        },
        /* .device  = */ &g_lm_ggml_metal_device,
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_buffer_type_mapped_metal;
}

// backend

static const char * lm_ggml_backend_metal_name(lm_ggml_backend_t backend) {
    return "Metal";

    LM_GGML_UNUSED(backend);
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
    return false;

    LM_GGML_UNUSED(backend_src);
    LM_GGML_UNUSED(backend_dst);
    LM_GGML_UNUSED(src);
    LM_GGML_UNUSED(dst);
}

static enum lm_ggml_status lm_ggml_backend_metal_graph_compute(lm_ggml_backend_t backend, lm_ggml_cgraph * cgraph) {
    lm_ggml_metal_t ctx = (lm_ggml_metal_t)backend->context;

    return lm_ggml_metal_graph_compute(ctx, cgraph);
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

    // the events API is needed only for multi-GPU setups, so likely no need to implement it for Metal
    // in any case, these docs seem relevant if we ever decide to implement it:
    // https://developer.apple.com/documentation/metal/mtlcommandbuffer#Synchronizing-Passes-with-Events
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
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
    return "Metal";

    LM_GGML_UNUSED(dev);
}

static const char * lm_ggml_backend_metal_device_get_description(lm_ggml_backend_dev_t dev) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    return lm_ggml_metal_device_get_props(ctx_dev)->name;
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
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static lm_ggml_backend_t lm_ggml_backend_metal_device_init(lm_ggml_backend_dev_t dev, const char * params) {
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

    return props_dev->use_shared_buffers ? lm_ggml_backend_metal_buffer_type_shared() : lm_ggml_backend_metal_buffer_type_private();
}

static lm_ggml_backend_buffer_t lm_ggml_backend_metal_device_buffer_mapped(lm_ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    lm_ggml_metal_buffer_t res = lm_ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    return lm_ggml_backend_buffer_init(lm_ggml_backend_metal_buffer_type_mapped(), lm_ggml_backend_metal_buffer_shared_i, res, size);
}

static bool lm_ggml_backend_metal_device_supports_op(lm_ggml_backend_dev_t dev, const lm_ggml_tensor * op) {
    lm_ggml_metal_device_t ctx_dev = (lm_ggml_metal_device_t)dev->context;

    return lm_ggml_metal_device_supports_op(ctx_dev, op);
}

static bool lm_ggml_backend_metal_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    return
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == lm_ggml_backend_metal_buffer_type_mapped_get_name;

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
    const int min_batch_size = 32;

    return (op->op == LM_GGML_OP_MUL_MAT ||
            op->op == LM_GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= min_batch_size;

    LM_GGML_UNUSED(dev);
    LM_GGML_UNUSED(op);
}

static lm_ggml_backend_device_i lm_ggml_backend_metal_device_i = {
    /* .get_name             = */ lm_ggml_backend_metal_device_get_name,
    /* .get_description      = */ lm_ggml_backend_metal_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_metal_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_metal_device_get_type,
    /* .get_props            = */ lm_ggml_backend_metal_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_metal_device_init,
    /* .get_buffer_type      = */ lm_ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ lm_ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ lm_ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ lm_ggml_backend_metal_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend registry

static const char * lm_ggml_backend_metal_reg_get_name(lm_ggml_backend_reg_t reg) {
    return "Metal";

    LM_GGML_UNUSED(reg);
}

static size_t lm_ggml_backend_metal_reg_device_count(lm_ggml_backend_reg_t reg) {
    return 1;

    LM_GGML_UNUSED(reg);
}

static lm_ggml_backend_dev_t lm_ggml_backend_metal_reg_device_get(lm_ggml_backend_reg_t reg, size_t index) {
    LM_GGML_ASSERT(index == 0);

    return &g_lm_ggml_metal_device;

    LM_GGML_UNUSED(reg);
    LM_GGML_UNUSED(index);
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
    /* .device_count     = */ lm_ggml_backend_metal_reg_device_count,
    /* .device_get       = */ lm_ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ lm_ggml_backend_metal_get_proc_address,
};

lm_ggml_backend_reg_t lm_ggml_backend_metal_reg(void) {
    {
        g_lm_ggml_metal_reg = {
            /* .api_version = */ LM_GGML_BACKEND_API_VERSION,
            /* .iface       = */ lm_ggml_backend_metal_reg_i,
            /* .context     = */ NULL,
        };

        g_lm_ggml_metal_device = {
            /* .iface   = */ lm_ggml_backend_metal_device_i,
            /* .reg     = */ &g_lm_ggml_metal_reg,
            /* .context = */ lm_ggml_metal_device_get(),
        };
    }

    return &g_lm_ggml_metal_reg;
}

LM_GGML_BACKEND_DL_IMPL(lm_ggml_backend_metal_reg)
