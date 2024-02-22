#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX(a, b) ((a) > (b) ? (a) : (b))


// backend buffer type

const char * lm_ggml_backend_buft_name(lm_ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name(buft);
}

LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_buft_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    return buft->iface.alloc_buffer(buft, size);
}

size_t lm_ggml_backend_buft_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return buft->iface.get_alignment(buft);
}

size_t lm_ggml_backend_buft_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

LM_GGML_CALL size_t lm_ggml_backend_buft_get_alloc_size(lm_ggml_backend_buffer_type_t buft, struct lm_ggml_tensor * tensor) {
    // get_alloc_size is optional, defaults to lm_ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= lm_ggml_nbytes(tensor));
        return size;
    }
    return lm_ggml_nbytes(tensor);
}

bool lm_ggml_backend_buft_supports_backend(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend) {
    return buft->iface.supports_backend(buft, backend);
}

bool lm_ggml_backend_buft_is_host(lm_ggml_backend_buffer_type_t buft) {
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

// backend buffer

LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_buffer_init(
               lm_ggml_backend_buffer_type_t      buft,
        struct lm_ggml_backend_buffer_i           iface,
               lm_ggml_backend_buffer_context_t   context,
               size_t                          size) {
    lm_ggml_backend_buffer_t buffer = malloc(sizeof(struct lm_ggml_backend_buffer));

    (*buffer) = (struct lm_ggml_backend_buffer) {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ LM_GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * lm_ggml_backend_buffer_name(lm_ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name(buffer);
}

void lm_ggml_backend_buffer_free(lm_ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    free(buffer);
}

size_t lm_ggml_backend_buffer_get_size(lm_ggml_backend_buffer_t buffer) {
    return buffer->size;
}

void * lm_ggml_backend_buffer_get_base(lm_ggml_backend_buffer_t buffer) {
    void * base = buffer->iface.get_base(buffer);

    LM_GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

LM_GGML_CALL void lm_ggml_backend_buffer_init_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

size_t lm_ggml_backend_buffer_get_alignment (lm_ggml_backend_buffer_t buffer) {
    return lm_ggml_backend_buft_get_alignment(lm_ggml_backend_buffer_get_type(buffer));
}

size_t lm_ggml_backend_buffer_get_max_size(lm_ggml_backend_buffer_t buffer) {
    return lm_ggml_backend_buft_get_max_size(lm_ggml_backend_buffer_get_type(buffer));
}

size_t lm_ggml_backend_buffer_get_alloc_size(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor) {
    return lm_ggml_backend_buft_get_alloc_size(lm_ggml_backend_buffer_get_type(buffer), tensor);
}

void lm_ggml_backend_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    buffer->iface.clear(buffer, value);
}

bool lm_ggml_backend_buffer_is_host(lm_ggml_backend_buffer_t buffer) {
    return lm_ggml_backend_buft_is_host(lm_ggml_backend_buffer_get_type(buffer));
}

void lm_ggml_backend_buffer_set_usage(lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage) {
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (lm_ggml_backend_buffer_is_multi_buffer(buffer)) {
        lm_ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_buffer_get_type(lm_ggml_backend_buffer_t buffer) {
    return buffer->buft;
}

void lm_ggml_backend_buffer_reset(lm_ggml_backend_buffer_t buffer) {
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool lm_ggml_backend_buffer_copy_tensor(const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    lm_ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return src->buffer->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

const char * lm_ggml_backend_name(lm_ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void lm_ggml_backend_free(lm_ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_get_default_buffer_type(lm_ggml_backend_t backend) {
    return backend->iface.get_default_buffer_type(backend);
}

lm_ggml_backend_buffer_t lm_ggml_backend_alloc_buffer(lm_ggml_backend_t backend, size_t size) {
    return lm_ggml_backend_buft_alloc_buffer(lm_ggml_backend_get_default_buffer_type(backend), size);
}

size_t lm_ggml_backend_get_alignment(lm_ggml_backend_t backend) {
    return lm_ggml_backend_buft_get_alignment(lm_ggml_backend_get_default_buffer_type(backend));
}

size_t lm_ggml_backend_get_max_size(lm_ggml_backend_t backend) {
    return lm_ggml_backend_buft_get_max_size(lm_ggml_backend_get_default_buffer_type(backend));
}

void lm_ggml_backend_tensor_set_async(lm_ggml_backend_t backend, struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    LM_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    LM_GGML_ASSERT(offset + size <= lm_ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        lm_ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void lm_ggml_backend_tensor_get_async(lm_ggml_backend_t backend, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    LM_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    LM_GGML_ASSERT(offset + size <= lm_ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        lm_ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

LM_GGML_CALL void lm_ggml_backend_tensor_set(struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    lm_ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    LM_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    LM_GGML_ASSERT(buf != NULL && "tensor buffer not set");
    LM_GGML_ASSERT(offset + size <= lm_ggml_nbytes(tensor) && "tensor write out of bounds");

    if (!size) {
        return;
    }

    tensor->buffer->iface.set_tensor(buf, tensor, data, offset, size);
}

LM_GGML_CALL void lm_ggml_backend_tensor_get(const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    lm_ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    LM_GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    LM_GGML_ASSERT(tensor->buffer != NULL && "tensor buffer not set");
    LM_GGML_ASSERT(offset + size <= lm_ggml_nbytes(tensor) && "tensor read out of bounds");

    if (!size) {
        return;
    }

    tensor->buffer->iface.get_tensor(buf, tensor, data, offset, size);
}

void lm_ggml_backend_synchronize(lm_ggml_backend_t backend) {
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

lm_ggml_backend_graph_plan_t lm_ggml_backend_graph_plan_create(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    return backend->iface.graph_plan_create(backend, cgraph);
}

void lm_ggml_backend_graph_plan_free(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_free(backend, plan);
}

void lm_ggml_backend_graph_plan_compute(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    backend->iface.graph_plan_compute(backend, plan);
}

bool lm_ggml_backend_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}

bool lm_ggml_backend_supports_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

// backend copy

static bool lm_ggml_are_same_layout(const struct lm_ggml_tensor * a, const struct lm_ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void lm_ggml_backend_tensor_copy(struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    LM_GGML_ASSERT(lm_ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (lm_ggml_backend_buffer_is_host(src->buffer)) {
        lm_ggml_backend_tensor_set(dst, src->data, 0, lm_ggml_nbytes(src));
    } else if (lm_ggml_backend_buffer_is_host(dst->buffer)) {
        lm_ggml_backend_tensor_get(src, dst->data, 0, lm_ggml_nbytes(src));
    } else if (!lm_ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        fprintf(stderr, "%s: warning: slow copy from %s to %s\n", __func__, lm_ggml_backend_buffer_name(src->buffer), lm_ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = lm_ggml_nbytes(src);
        void * data = malloc(nbytes);
        lm_ggml_backend_tensor_get(src, data, 0, nbytes);
        lm_ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void lm_ggml_backend_tensor_copy_async(lm_ggml_backend_t backend, struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    LM_GGML_ASSERT(lm_ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (lm_ggml_backend_buft_supports_backend(src->buffer->buft, backend) && lm_ggml_backend_buft_supports_backend(dst->buffer->buft, backend)) {
        if (backend->iface.cpy_tensor_async != NULL) {
            if (backend->iface.cpy_tensor_async(backend, src, dst)) {
                return;
            }
        }
    }

    size_t nbytes = lm_ggml_nbytes(src);
    if (lm_ggml_backend_buffer_is_host(src->buffer)) {
        lm_ggml_backend_tensor_set_async(backend, dst, src->data, 0, nbytes);
    }
    else {
        lm_ggml_backend_tensor_copy(src, dst);
    }
}


// backend registry

#define LM_GGML_MAX_BACKENDS_REG 16

struct lm_ggml_backend_reg {
    char name[128];
    lm_ggml_backend_init_fn init_fn;
    lm_ggml_backend_buffer_type_t default_buffer_type;
    void * user_data;
};

static struct lm_ggml_backend_reg lm_ggml_backend_registry[LM_GGML_MAX_BACKENDS_REG];
static size_t lm_ggml_backend_registry_count = 0;

LM_GGML_CALL static lm_ggml_backend_t lm_ggml_backend_reg_cpu_init(const char * params, void * user_data);

LM_GGML_CALL static void lm_ggml_backend_registry_init(void) {
    static bool initialized = false;

    if (initialized) {
        return;
    }

    initialized = true;

    lm_ggml_backend_register("CPU", lm_ggml_backend_reg_cpu_init, lm_ggml_backend_cpu_buffer_type(), NULL);

    // add forward decls here to avoid including the backend headers
#ifdef LM_GGML_USE_CUBLAS
    extern LM_GGML_CALL void lm_ggml_backend_cuda_reg_devices(void);
    lm_ggml_backend_cuda_reg_devices();
#endif

#ifdef LM_GGML_USE_SYCL
    extern void lm_ggml_backend_sycl_reg_devices(void);
    lm_ggml_backend_sycl_reg_devices();
#endif

#ifdef LM_GGML_USE_METAL
    extern LM_GGML_CALL lm_ggml_backend_t lm_ggml_backend_reg_metal_init(const char * params, void * user_data);
    extern LM_GGML_CALL lm_ggml_backend_buffer_type_t lm_ggml_backend_metal_buffer_type(void);
    lm_ggml_backend_register("Metal", lm_ggml_backend_reg_metal_init, lm_ggml_backend_metal_buffer_type(), NULL);
#endif

#ifdef LM_GGML_USE_VULKAN
    extern LM_GGML_CALL int lm_ggml_backend_vk_reg_devices(void);
    lm_ggml_backend_vk_reg_devices();
#endif

#ifdef LM_GGML_USE_KOMPUTE
    extern LM_GGML_CALL void lm_ggml_backend_kompute_reg_devices(void);
    lm_ggml_backend_kompute_reg_devices();
#endif
}

LM_GGML_CALL void lm_ggml_backend_register(const char * name, lm_ggml_backend_init_fn init_fn, lm_ggml_backend_buffer_type_t default_buffer_type, void * user_data) {
    LM_GGML_ASSERT(lm_ggml_backend_registry_count < LM_GGML_MAX_BACKENDS_REG);

    size_t id = lm_ggml_backend_registry_count;

    lm_ggml_backend_registry[id] = (struct lm_ggml_backend_reg) {
        /* .name                = */ {0},
        /* .fn                  = */ init_fn,
        /* .default_buffer_type = */ default_buffer_type,
        /* .user_data           = */ user_data,
    };

    snprintf(lm_ggml_backend_registry[id].name, sizeof(lm_ggml_backend_registry[id].name), "%s", name);

#ifndef NDEBUG
    fprintf(stderr, "%s: registered backend %s\n", __func__, name);
#endif

    lm_ggml_backend_registry_count++;
}

size_t lm_ggml_backend_reg_get_count(void) {
    lm_ggml_backend_registry_init();

    return lm_ggml_backend_registry_count;
}

size_t lm_ggml_backend_reg_find_by_name(const char * name) {
    lm_ggml_backend_registry_init();

    for (size_t i = 0; i < lm_ggml_backend_registry_count; i++) {
        // TODO: case insensitive in a portable way
        if (strcmp(lm_ggml_backend_registry[i].name, name) == 0) {
            return i;
        }
    }

    // not found
    return SIZE_MAX;
}

// init from backend:params string
lm_ggml_backend_t lm_ggml_backend_reg_init_backend_from_str(const char * backend_str) {
    lm_ggml_backend_registry_init();

    const char * params = strchr(backend_str, ':');
    char backend_name[128];
    if (params == NULL) {
        snprintf(backend_name, sizeof(backend_name), "%s", backend_str);
        params = "";
    } else {
        snprintf(backend_name, sizeof(backend_name), "%.*s", (int)(params - backend_str), backend_str);
        params++;
    }

    size_t backend_i = lm_ggml_backend_reg_find_by_name(backend_name);

    if (backend_i == SIZE_MAX) {
        fprintf(stderr, "%s: backend %s not found\n", __func__, backend_name);
        return NULL;
    }

    return lm_ggml_backend_reg_init_backend(backend_i, params);
}

const char * lm_ggml_backend_reg_get_name(size_t i) {
    lm_ggml_backend_registry_init();

    LM_GGML_ASSERT(i < lm_ggml_backend_registry_count);
    return lm_ggml_backend_registry[i].name;
}

lm_ggml_backend_t lm_ggml_backend_reg_init_backend(size_t i, const char * params) {
    lm_ggml_backend_registry_init();

    LM_GGML_ASSERT(i < lm_ggml_backend_registry_count);
    return lm_ggml_backend_registry[i].init_fn(params, lm_ggml_backend_registry[i].user_data);
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_reg_get_default_buffer_type(size_t i) {
    lm_ggml_backend_registry_init();

    LM_GGML_ASSERT(i < lm_ggml_backend_registry_count);
    return lm_ggml_backend_registry[i].default_buffer_type;
}

lm_ggml_backend_buffer_t lm_ggml_backend_reg_alloc_buffer(size_t i, size_t size) {
    lm_ggml_backend_registry_init();

    LM_GGML_ASSERT(i < lm_ggml_backend_registry_count);
    return lm_ggml_backend_buft_alloc_buffer(lm_ggml_backend_registry[i].default_buffer_type, size);
}

// backend CPU

static const size_t TENSOR_ALIGNMENT = 32; // required for mmap as gguf only guarantees 32-byte alignment

LM_GGML_CALL static const char * lm_ggml_backend_cpu_buffer_name(lm_ggml_backend_buffer_t buffer) {
    return "CPU";

    LM_GGML_UNUSED(buffer);
}

LM_GGML_CALL static void * lm_ggml_backend_cpu_buffer_get_base(lm_ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = LM_GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

LM_GGML_CALL static void lm_ggml_backend_cpu_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_buffer_set_tensor(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    LM_GGML_UNUSED(buffer);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_buffer_get_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    LM_GGML_UNUSED(buffer);
}

LM_GGML_CALL static bool lm_ggml_backend_cpu_buffer_cpy_tensor(lm_ggml_backend_buffer_t buffer, const struct lm_ggml_tensor * src, struct lm_ggml_tensor * dst) {
    if (lm_ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, lm_ggml_nbytes(src));
        return true;
    }
    return false;

    LM_GGML_UNUSED(buffer);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static struct lm_ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .get_name        = */ lm_ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ lm_ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ lm_ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ lm_ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ lm_ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ lm_ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// for buffers from ptr, free is not called
static struct lm_ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .get_name        = */ lm_ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ lm_ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .set_tensor      = */ lm_ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ lm_ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ lm_ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

LM_GGML_CALL static const char * lm_ggml_backend_cpu_buffer_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "CPU";

    LM_GGML_UNUSED(buft);
}

LM_GGML_CALL static lm_ggml_backend_buffer_t lm_ggml_backend_cpu_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: use LM_GGML_ALIGNED_MALLOC (move to ggml-impl.h)
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return lm_ggml_backend_buffer_init(buft, cpu_backend_buffer_i, data, size);
}

LM_GGML_CALL static size_t lm_ggml_backend_cpu_buffer_type_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    LM_GGML_UNUSED(buft);
}

LM_GGML_CALL static bool lm_ggml_backend_cpu_buffer_type_supports_backend(lm_ggml_backend_buffer_type_t buft, lm_ggml_backend_t backend) {
    return lm_ggml_backend_is_cpu(backend);

    LM_GGML_UNUSED(buft);
}

LM_GGML_CALL static bool lm_ggml_backend_cpu_buffer_type_is_host(lm_ggml_backend_buffer_type_t buft) {
    return true;

    LM_GGML_UNUSED(buft);
}

LM_GGML_CALL lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_buffer_type(void) {
    static struct lm_ggml_backend_buffer_type lm_ggml_backend_cpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ lm_ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to lm_ggml_nbytes
            /* .supports_backend = */ lm_ggml_backend_cpu_buffer_type_supports_backend,
            /* .is_host          = */ lm_ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &lm_ggml_backend_cpu_buffer_type;
}

#ifdef LM_GGML_USE_CPU_HBM

// buffer type HBM

#include <hbwmalloc.h>

LM_GGML_CALL static const char * lm_ggml_backend_cpu_hbm_buffer_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    return "CPU_HBM";

    LM_GGML_UNUSED(buft);
}

LM_GGML_CALL static const char * lm_ggml_backend_cpu_hbm_buffer_get_name(lm_ggml_backend_buffer_t buf) {
    return "CPU_HBM";

    LM_GGML_UNUSED(buf);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_hbm_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

LM_GGML_CALL static lm_ggml_backend_buffer_t lm_ggml_backend_cpu_hbm_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    //void * ptr = hbw_malloc(size);
    void * ptr;
    int result = hbw_posix_memalign(&ptr, lm_ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    if (result != 0) {
        fprintf(stderr, "failed to allocate HBM buffer of size %zu\n", size);
        return NULL;
    }

    lm_ggml_backend_buffer_t buffer = lm_ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = lm_ggml_backend_cpu_hbm_buffer_get_name;
    buffer->iface.free_buffer = lm_ggml_backend_cpu_hbm_buffer_free_buffer;

    return buffer;
}

lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_hbm_buffer_type(void) {
    static struct lm_ggml_backend_buffer_type lm_ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
            /* .get_name         = */ lm_ggml_backend_cpu_hbm_buffer_type_get_name,
            /* .alloc_buffer     = */ lm_ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
            /* .get_alignment    = */ lm_ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to lm_ggml_nbytes
            /* .supports_backend = */ lm_ggml_backend_cpu_buffer_type_supports_backend,
            /* .is_host          = */ lm_ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context  = */ NULL,
    };

    return &lm_ggml_backend_cpu_buffer_type_hbm;
}
#endif

struct lm_ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;

    lm_ggml_abort_callback abort_callback;
    void *              abort_callback_data;
};

LM_GGML_CALL static const char * lm_ggml_backend_cpu_name(lm_ggml_backend_t backend) {
    return "CPU";

    LM_GGML_UNUSED(backend);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_free(lm_ggml_backend_t backend) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

LM_GGML_CALL static lm_ggml_backend_buffer_type_t lm_ggml_backend_cpu_get_default_buffer_type(lm_ggml_backend_t backend) {
    return lm_ggml_backend_cpu_buffer_type();

    LM_GGML_UNUSED(backend);
}

struct lm_ggml_backend_plan_cpu {
    struct lm_ggml_cplan cplan;
    struct lm_ggml_cgraph cgraph;
};

LM_GGML_CALL static lm_ggml_backend_graph_plan_t lm_ggml_backend_cpu_graph_plan_create(lm_ggml_backend_t backend, const struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;

    struct lm_ggml_backend_plan_cpu * cpu_plan = malloc(sizeof(struct lm_ggml_backend_plan_cpu));

    cpu_plan->cplan = lm_ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = malloc(cpu_plan->cplan.work_size);
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    return cpu_plan;
}

LM_GGML_CALL static void lm_ggml_backend_cpu_graph_plan_free(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    struct lm_ggml_backend_plan_cpu * cpu_plan = (struct lm_ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    LM_GGML_UNUSED(backend);
}

LM_GGML_CALL static void lm_ggml_backend_cpu_graph_plan_compute(lm_ggml_backend_t backend, lm_ggml_backend_graph_plan_t plan) {
    struct lm_ggml_backend_plan_cpu * cpu_plan = (struct lm_ggml_backend_plan_cpu *)plan;

    lm_ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    LM_GGML_UNUSED(backend);
}

LM_GGML_CALL static bool lm_ggml_backend_cpu_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_backend_cpu_context * cpu_ctx = (struct lm_ggml_backend_cpu_context *)backend->context;

    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        // TODO: may be faster to free and use malloc to avoid the copy
        cpu_ctx->work_data = realloc(cpu_ctx->work_data, cplan.work_size);
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;

    lm_ggml_graph_compute(cgraph, &cplan);
    return true;
}

LM_GGML_CALL static bool lm_ggml_backend_cpu_supports_op(lm_ggml_backend_t backend, const struct lm_ggml_tensor * op) {
    switch (op->op) {
        case LM_GGML_OP_CPY:
            return op->type != LM_GGML_TYPE_IQ2_XXS && op->type != LM_GGML_TYPE_IQ2_XS && op->type != LM_GGML_TYPE_IQ1_S; // missing type_traits.from_float
        case LM_GGML_OP_MUL_MAT:
            return op->src[1]->type == LM_GGML_TYPE_F32 || op->src[1]->type == lm_ggml_internal_get_type_traits(op->src[0]->type).vec_dot_type;
        default:
            return true;
    }

    LM_GGML_UNUSED(backend);
}

static struct lm_ggml_backend_i cpu_backend_i = {
    /* .get_name                = */ lm_ggml_backend_cpu_name,
    /* .free                    = */ lm_ggml_backend_cpu_free,
    /* .get_default_buffer_type = */ lm_ggml_backend_cpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ lm_ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ lm_ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_compute      = */ lm_ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ lm_ggml_backend_cpu_graph_compute,
    /* .supports_op             = */ lm_ggml_backend_cpu_supports_op,
};

lm_ggml_backend_t lm_ggml_backend_cpu_init(void) {
    struct lm_ggml_backend_cpu_context * ctx = malloc(sizeof(struct lm_ggml_backend_cpu_context));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = LM_GGML_DEFAULT_N_THREADS;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;

    lm_ggml_backend_t cpu_backend = malloc(sizeof(struct lm_ggml_backend));
    if (cpu_backend == NULL) {
        free(ctx);
        return NULL;
    }

    *cpu_backend = (struct lm_ggml_backend) {
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

LM_GGML_CALL bool lm_ggml_backend_is_cpu(lm_ggml_backend_t backend) {
    return backend && backend->iface.get_name == lm_ggml_backend_cpu_name;
}

void lm_ggml_backend_cpu_set_n_threads(lm_ggml_backend_t backend_cpu, int n_threads) {
    LM_GGML_ASSERT(lm_ggml_backend_is_cpu(backend_cpu));

    struct lm_ggml_backend_cpu_context * ctx = (struct lm_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void lm_ggml_backend_cpu_set_abort_callback(lm_ggml_backend_t backend_cpu, lm_ggml_abort_callback abort_callback, void * abort_callback_data) {
    LM_GGML_ASSERT(lm_ggml_backend_is_cpu(backend_cpu));

    struct lm_ggml_backend_cpu_context * ctx = (struct lm_ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    LM_GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return lm_ggml_backend_buffer_init(lm_ggml_backend_cpu_buffer_type(), cpu_backend_buffer_i_from_ptr, ptr, size);
}

LM_GGML_CALL static lm_ggml_backend_t lm_ggml_backend_reg_cpu_init(const char * params, void * user_data) {
    return lm_ggml_backend_cpu_init();

    LM_GGML_UNUSED(params);
    LM_GGML_UNUSED(user_data);
}

// multi-buffer buffer

struct lm_ggml_backend_multi_buffer_context {
    lm_ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

typedef struct lm_ggml_backend_multi_buffer_context * lm_ggml_backend_multi_buffer_context_t;

LM_GGML_CALL static const char * lm_ggml_backend_multi_buffer_get_name(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_backend_multi_buffer_context_t ctx = (lm_ggml_backend_multi_buffer_context_t) buffer->context;

    return ctx->buffers[0]->iface.get_name(ctx->buffers[0]);
}

LM_GGML_CALL static void lm_ggml_backend_multi_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    lm_ggml_backend_multi_buffer_context_t ctx = (lm_ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        lm_ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

LM_GGML_CALL static void lm_ggml_backend_multi_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    lm_ggml_backend_multi_buffer_context_t ctx = (lm_ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        lm_ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static struct lm_ggml_backend_buffer_i lm_ggml_backend_multi_buffer_context_interface(void) {
    static struct lm_ggml_backend_buffer_i multi_backend_buffer_i = {
        /* .get_name        = */ lm_ggml_backend_multi_buffer_get_name,
        /* .free_buffer     = */ lm_ggml_backend_multi_buffer_free_buffer,
        /* .get_base        = */ NULL,
        /* .init_tensor     = */ NULL,
        /* .set_tensor      = */ NULL,
        /* .get_tensor      = */ NULL,
        /* .cpy_tensor      = */ NULL,
        /* .clear           = */ lm_ggml_backend_multi_buffer_clear,
        /* .reset           = */ NULL,
    };

    return multi_backend_buffer_i;
}

LM_GGML_CALL lm_ggml_backend_buffer_t lm_ggml_backend_multi_buffer_alloc_buffer(lm_ggml_backend_buffer_t * buffers, size_t n_buffers) {
    lm_ggml_backend_multi_buffer_context_t ctx = (lm_ggml_backend_multi_buffer_context_t) malloc(sizeof(struct lm_ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (lm_ggml_backend_buffer_t *) malloc(n_buffers * sizeof(lm_ggml_backend_buffer_t));

    LM_GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += lm_ggml_backend_buffer_get_size(buffers[i]);
    }

    return lm_ggml_backend_buffer_init(buffers[0]->buft, lm_ggml_backend_multi_buffer_context_interface(), ctx, total_size);
}

LM_GGML_CALL bool lm_ggml_backend_buffer_is_multi_buffer(lm_ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == lm_ggml_backend_multi_buffer_get_name;
}

LM_GGML_CALL void lm_ggml_backend_multi_buffer_set_usage(lm_ggml_backend_buffer_t buffer, enum lm_ggml_backend_buffer_usage usage) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_multi_buffer(buffer));
    lm_ggml_backend_multi_buffer_context_t ctx = (lm_ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        lm_ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct lm_ggml_tensor * lm_ggml_dup_tensor_layout(struct lm_ggml_context * ctx, const struct lm_ggml_tensor * tensor) {
    struct lm_ggml_tensor * dup = lm_ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool lm_ggml_is_view_op(enum lm_ggml_op op) {
    return op == LM_GGML_OP_VIEW || op == LM_GGML_OP_RESHAPE || op == LM_GGML_OP_PERMUTE || op == LM_GGML_OP_TRANSPOSE;
}

// scheduler

#define LM_GGML_MAX_BACKENDS 16
#define LM_GGML_MAX_SPLITS 256
#define LM_GGML_MAX_SPLIT_INPUTS 16

struct lm_ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct lm_ggml_tensor * inputs[LM_GGML_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct lm_ggml_cgraph graph;
};

struct lm_ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split

    int n_backends;
    lm_ggml_backend_t backends[LM_GGML_MAX_BACKENDS];
    lm_ggml_backend_buffer_type_t bufts[LM_GGML_MAX_BACKENDS];

    lm_ggml_gallocr_t galloc;

    // hash keys of the nodes in the graph
    struct lm_ggml_hash_set    hash_set;
    // hash values
    int * tensor_backend_id;
    struct lm_ggml_tensor * (* tensor_copies)[LM_GGML_MAX_BACKENDS];

    int * node_backend_ids; // [n_nodes]
    int n_nodes;

    // copy of the graph with modified inputs
    struct lm_ggml_cgraph * graph;

    struct lm_ggml_backend_sched_split splits[LM_GGML_MAX_SPLITS];
    int n_splits;

    struct lm_ggml_context * ctx;

    lm_ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    // align context_buffer to LM_GGML_MEM_ALIGN
    #ifdef _MSC_VER
    __declspec(align(LM_GGML_MEM_ALIGN))
    #else
    __attribute__((aligned(LM_GGML_MEM_ALIGN)))
    #endif
    char context_buffer[LM_GGML_MAX_SPLITS*LM_GGML_MAX_SPLIT_INPUTS*2*sizeof(struct lm_ggml_tensor) + sizeof(struct lm_ggml_cgraph)];
};

#define hash_id(node) lm_ggml_hash_find_or_insert(sched->hash_set, node)
#define tensor_backend_id(node) sched->tensor_backend_id[hash_id(node)]
#define tensor_backend(node) (tensor_backend_id(node) == -1 ? NULL : sched->backends[tensor_backend_id(node)])

// returns the priority of the backend, lower id is higher priority
static int lm_ggml_backend_sched_backend_id(lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int lm_ggml_backend_sched_backend_from_buffer(lm_ggml_backend_sched_t sched, lm_ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type
    for (int i = 0; i < sched->n_backends; i++) {
        if (lm_ggml_backend_buft_supports_backend(buffer->buft, sched->backends[i])) {
            return i;
        }
    }
    LM_GGML_ASSERT(false && "tensor buffer type not supported by any backend");
    return -1; // silence warning
}

#if 0
static char causes[LM_GGML_DEFAULT_GRAPH_SIZE*16 + LM_GGML_MAX_SPLITS*LM_GGML_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int lm_ggml_backend_sched_backend_id_from_cur(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * tensor) {
    // TODO: use supports_op to check if the backend supports the op

    // assign pre-allocated nodes to their backend
    // dst
    int cur_backend = lm_ggml_backend_sched_backend_from_buffer(sched, tensor->buffer);
    if (cur_backend != -1) {
        SET_CAUSE(node, "1.dst");
        return cur_backend;
    }
    // view_src
    if (tensor->view_src != NULL) {
        cur_backend = lm_ggml_backend_sched_backend_from_buffer(sched, tensor->view_src->buffer);
        if (cur_backend != -1) {
            SET_CAUSE(node, "1.vsrc");
            return cur_backend;
        }
    }
    // assign nodes that use weights to the backend of the weights
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        const struct lm_ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        if (src->buffer != NULL && src->buffer->usage == LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend = lm_ggml_backend_sched_backend_from_buffer(sched, src->buffer);
            // operations with weights are always run on the same backend as the weights
            SET_CAUSE(node, "1.wgt%d", i);
            return src_backend;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        sprintf(buffer, "%zuM", size/1024/1024);
    } else {
        sprintf(buffer, "%zuK", size/1024);
    }
    return buffer;
}

static void lm_ggml_backend_sched_print_assignments(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            lm_ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            fprintf(stderr, "\n## SPLIT #%d: %s # %d inputs: ", cur_split, lm_ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                fprintf(stderr, "[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(lm_ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            fprintf(stderr, "\n");
            cur_split++;
        }
        struct lm_ggml_tensor * node = graph->nodes[i];
        if (lm_ggml_is_view_op(node->op)) {
            continue;
        }
        lm_ggml_backend_t tensor_backend = tensor_backend(node);
        fprintf(stderr, "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s]:", i, lm_ggml_op_name(node->op), node->name,
            fmt_size(lm_ggml_nbytes(node)), tensor_backend ? lm_ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node));
        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            struct lm_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            lm_ggml_backend_t src_backend = tensor_backend(src);
            fprintf(stderr, " %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                fmt_size(lm_ggml_nbytes(src)), src_backend ? lm_ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
        }
        fprintf(stderr, "\n");
    }
}

//#define DEBUG_PASS1
//#define DEBUG_PASS2
//#define DEBUG_PASS3
//#define DEBUG_PASS4

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static void lm_ggml_backend_sched_split_graph(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->is_reset = false;

    struct lm_ggml_init_params params = {
        /* .mem_size =   */ sizeof(sched->context_buffer),
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    lm_ggml_free(sched->ctx);

    sched->ctx = lm_ggml_init(params);
    if (sched->ctx == NULL) {
        fprintf(stderr, "%s: failed to initialize context\n", __func__);
        LM_GGML_ASSERT(false);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct lm_ggml_tensor * leaf = graph->leafs[i];
        if (tensor_backend_id(leaf) != -1) {
            // do not overwrite user assignments
            continue;
        }
        tensor_backend_id(leaf) = lm_ggml_backend_sched_backend_id_from_cur(sched, leaf);
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        if (tensor_backend_id(node) != -1) {
            // do not overwrite user assignments
            continue;
        }
        tensor_backend_id(node) = lm_ggml_backend_sched_backend_id_from_cur(sched, node);
        // src
        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            struct lm_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            if (tensor_backend_id(src) == -1) {
                tensor_backend_id(src) = lm_ggml_backend_sched_backend_id_from_cur(sched, src);
            }
        }
    }
#ifdef DEBUG_PASS1
    fprintf(stderr, "PASS 1 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops

    // pass 2.1 expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct lm_ggml_tensor * node = graph->nodes[i];
            if (lm_ggml_is_view_op(node->op)) {
                continue;
            }
            int tensor_backend_id = tensor_backend_id(node);
            if (tensor_backend_id != -1) {
                if (tensor_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = tensor_backend_id;
                }
            } else {
                tensor_backend_id(node) = cur_backend_id;
                SET_CAUSE(node, "2.1");
            }
        }
    }

    // pass 2.2 expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct lm_ggml_tensor * node = graph->nodes[i];
            if (lm_ggml_is_view_op(node->op)) {
                continue;
            }
            int tensor_backend_id = tensor_backend_id(node);
            if (tensor_backend_id != -1) {
                if (tensor_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = tensor_backend_id;
                }
            } else {
                tensor_backend_id(node) = cur_backend_id;
                SET_CAUSE(node, "2.2");
            }
        }
    }

    // pass 2.3 expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct lm_ggml_tensor * node = graph->nodes[i];
            if (lm_ggml_is_view_op(node->op)) {
                continue;
            }
            int tensor_backend_id = tensor_backend_id(node);
            if (tensor_backend_id != -1) {
                cur_backend_id = tensor_backend_id;
            } else {
                tensor_backend_id(node) = cur_backend_id;
                SET_CAUSE(node, "2.3");
            }
        }
    }

    // pass 2.4 expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct lm_ggml_tensor * node = graph->nodes[i];
            if (lm_ggml_is_view_op(node->op)) {
                continue;
            }
            int tensor_backend_id = tensor_backend_id(node);
            if (tensor_backend_id != -1) {
                cur_backend_id = tensor_backend_id;
            } else {
                tensor_backend_id(node) = cur_backend_id;
                SET_CAUSE(node, "2.4");
            }
        }
    }
#ifdef DEBUG_PASS2
    fprintf(stderr, "PASS 2 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 3: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        int cur_backend_id = tensor_backend_id(node);
        if (node->view_src != NULL && cur_backend_id == -1) {
            cur_backend_id = tensor_backend_id(node) = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "3.vsrc");
        }
        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            struct lm_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int src_backend_id = tensor_backend_id(src);
            if (src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    tensor_backend_id(src) = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "3.vsrc");
                } else {
                    tensor_backend_id(src) = cur_backend_id;
                    SET_CAUSE(src, "3.cur");
                }
            }
        }
    }
#ifdef DEBUG_PASS3
    fprintf(stderr, "PASS 3 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

    // pass 4: split graph, find tensors that need to be copied
    {
        int cur_split = 0;
        // find the backend of the first split, skipping view ops
        for (int i = 0; i < graph->n_nodes; i++) {
            struct lm_ggml_tensor * node = graph->nodes[i];
            if (!lm_ggml_is_view_op(node->op)) {
                sched->splits[0].backend_id = tensor_backend_id(node);
                break;
            }
        }
        sched->splits[0].i_start = 0;
        sched->splits[0].n_inputs = 0;
        memset(sched->splits[0].inputs, 0, sizeof(sched->splits[0].inputs)); //HACK
        int cur_backend_id = sched->splits[0].backend_id;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct lm_ggml_tensor * node = graph->nodes[i];

            if (lm_ggml_is_view_op(node->op)) {
                continue;
            }

            int tensor_backend_id = tensor_backend_id(node);

            LM_GGML_ASSERT(tensor_backend_id != -1); // all nodes should be assigned by now

            if (tensor_backend_id != cur_backend_id) {
                sched->splits[cur_split].i_end = i;
                cur_split++;
                LM_GGML_ASSERT(cur_split < LM_GGML_MAX_SPLITS);
                sched->splits[cur_split].backend_id = tensor_backend_id;
                sched->splits[cur_split].i_start = i;
                sched->splits[cur_split].n_inputs = 0;
                cur_backend_id = tensor_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
                struct lm_ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int src_backend_id = tensor_backend_id(src);
                assert(src_backend_id != -1); // all inputs should be assigned by now
                if (src_backend_id != tensor_backend_id) {
                    // create a copy of the input in the split's backend
                    size_t id = hash_id(src);
                    if (sched->tensor_copies[id][cur_backend_id] == NULL) {
                        lm_ggml_backend_t backend = sched->backends[cur_backend_id];
                        struct lm_ggml_tensor * tensor_copy = lm_ggml_dup_tensor_layout(sched->ctx, src);
                        lm_ggml_format_name(tensor_copy, "%s#%s", lm_ggml_backend_name(backend), src->name);

                        sched->tensor_copies[id][cur_backend_id] = tensor_copy;
                        tensor_backend_id(tensor_copy) = cur_backend_id;
                        SET_CAUSE(tensor_copy, "4.cpy");

                        int n_inputs = sched->splits[cur_split].n_inputs++;
                        LM_GGML_ASSERT(n_inputs < LM_GGML_MAX_SPLIT_INPUTS);
                        sched->splits[cur_split].inputs[n_inputs] = src;
                    }
                    node->src[j] = sched->tensor_copies[id][cur_backend_id];
                }
            }
        }
        sched->splits[cur_split].i_end = graph->n_nodes;
        sched->n_splits = cur_split + 1;
    }
#ifdef DEBUG_PASS4
    fprintf(stderr, "PASS 4 ASSIGNMENTS\n"); sched_print_assignments(sched, graph);
#endif

#ifndef NDEBUG
    // sanity check: all sources should have the same backend as the node
    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        lm_ggml_backend_t tensor_backend = tensor_backend(node);
        if (tensor_backend == NULL) {
            fprintf(stderr, "!!!!!!! %s has no backend\n", node->name);
        }
        if (node->view_src != NULL && tensor_backend != tensor_backend(node->view_src)) {
            fprintf(stderr, "!!!!!!! %s has backend %s, view_src %s has backend %s\n",
                node->name, tensor_backend ? lm_ggml_backend_name(tensor_backend) : "NULL",
                node->view_src->name, tensor_backend(node->view_src) ? lm_ggml_backend_name(tensor_backend(node->view_src)) : "NULL");
        }
        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            struct lm_ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            lm_ggml_backend_t src_backend = tensor_backend(src);
            if (src_backend != tensor_backend /* && src_backend != NULL */) {
                fprintf(stderr, "!!!! %s has backend %s, src %d (%s) has backend %s\n",
                    node->name, tensor_backend ? lm_ggml_backend_name(tensor_backend) : "NULL",
                    j, src->name, src_backend ? lm_ggml_backend_name(src_backend) : "NULL");
            }
            if (src->view_src != NULL && src_backend != tensor_backend(src->view_src)) {
                fprintf(stderr, "!!!!!!! [src] %s has backend %s, view_src %s has backend %s\n",
                    src->name, src_backend ? lm_ggml_backend_name(src_backend) : "NULL",
                    src->view_src->name, tensor_backend(src->view_src) ? lm_ggml_backend_name(tensor_backend(src->view_src)) : "NULL");
            }
        }
    }
    fflush(stderr);
#endif

    // create copies of the graph for each split
    // FIXME: avoid this copy, pass split inputs to lm_ggml_gallocr_alloc_graph_n in some other way
    struct lm_ggml_cgraph * graph_copy = lm_ggml_new_graph_custom(sched->ctx, graph->n_nodes + sched->n_splits*LM_GGML_MAX_SPLIT_INPUTS, false);
    for (int i = 0; i < sched->n_splits; i++) {
        struct lm_ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = lm_ggml_graph_view(graph, split->i_start, split->i_end);

        for (int j = 0; j < split->n_inputs; j++) {
            struct lm_ggml_tensor * input = split->inputs[j];
            struct lm_ggml_tensor * input_cpy = sched->tensor_copies[hash_id(input)][split->backend_id];

            // add a dependency to the input source so that it is not freed before the copy is done
            struct lm_ggml_tensor * input_dep = lm_ggml_view_tensor(sched->ctx, input);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(input);
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }
    sched->graph = graph_copy;
}

static bool lm_ggml_backend_sched_alloc_splits(lm_ggml_backend_sched_t sched) {
    // lm_ggml_gallocr_reserve_n(sched->galloc, sched->graph, sched->node_backend_ids);
    if (!lm_ggml_gallocr_alloc_graph(sched->galloc, sched->graph)) {
#ifndef NDEBUG
        fprintf(stderr, "lm_ggml_backend_sched: failed to allocate graph, reserving\n");
#endif
        lm_ggml_gallocr_reserve_n(sched->galloc, sched->graph, sched->node_backend_ids);
        if (!lm_ggml_gallocr_alloc_graph(sched->galloc, sched->graph)) {
            fprintf(stderr, "lm_ggml_backend_sched: failed to allocate graph\n");
            return false;
        }
    }

    return true;
}

static bool lm_ggml_backend_sched_compute_splits(lm_ggml_backend_sched_t sched) {
    uint64_t copy_us[LM_GGML_MAX_BACKENDS] = {0};
    uint64_t compute_us[LM_GGML_MAX_BACKENDS] = {0};

    struct lm_ggml_backend_sched_split * splits = sched->splits;

    for (int i = 0; i < sched->n_splits; i++) {
        struct lm_ggml_backend_sched_split * split = &splits[i];
        int split_backend_id = split->backend_id;
        lm_ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        uint64_t copy_start_us = lm_ggml_time_us();
        for (int j = 0; j < split->n_inputs; j++) {
            struct lm_ggml_tensor * input = split->inputs[j];
            struct lm_ggml_tensor * input_cpy = sched->tensor_copies[hash_id(input)][split_backend_id];

            LM_GGML_ASSERT(input->buffer != NULL);
            LM_GGML_ASSERT(input_cpy->buffer != NULL);

            lm_ggml_backend_tensor_copy_async(split_backend, input, input_cpy);
        }
        //lm_ggml_backend_synchronize(split_backend); // necessary to measure copy time
        int64_t copy_end_us = lm_ggml_time_us();
        copy_us[split_backend_id] += copy_end_us - copy_start_us;

#if 0
        char split_filename[LM_GGML_MAX_NAME];
        snprintf(split_filename, LM_GGML_MAX_NAME, "split_%i_%s.dot", i, lm_ggml_backend_name(split_backend));
        lm_ggml_graph_dump_dot(split->graph, NULL, split_filename);
#endif


        uint64_t compute_start_us = lm_ggml_time_us();
        if (!sched->callback_eval) {
            if (!lm_ggml_backend_graph_compute(split_backend, &split->graph)) {
                return false;
            }
            //lm_ggml_backend_synchronize(split_backend); // necessary to measure compute time
        } else {
            // similar to lm_ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct lm_ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct lm_ggml_cgraph gv = lm_ggml_graph_view(&split->graph, j0, j1 + 1);

                if (!lm_ggml_backend_graph_compute(split_backend, &gv)) {
                    return false;
                }

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }
        uint64_t compute_end_us = lm_ggml_time_us();
        compute_us[split_backend_id] += compute_end_us - compute_start_us;
    }

#if 0
    // per-backend timings
    fprintf(stderr, "sched_compute_splits times (%d splits):\n", sched->n_splits);
    for (int i = 0; i < sched->n_backends; i++) {
        if (copy_us[i] > 0 || compute_us[i] > 0) {
            fprintf(stderr, "\t%5.5s: %lu us copy, %lu us compute\n", lm_ggml_backend_name(sched->backends[i]), copy_us[i], compute_us[i]);
        }
    }
#endif

    return true;
}

lm_ggml_backend_sched_t lm_ggml_backend_sched_new(lm_ggml_backend_t * backends, lm_ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size) {
    LM_GGML_ASSERT(n_backends > 0);
    LM_GGML_ASSERT(n_backends <= LM_GGML_MAX_BACKENDS);

    struct lm_ggml_backend_sched * sched = calloc(sizeof(struct lm_ggml_backend_sched), 1);

    // initialize hash table
    sched->hash_set          = lm_ggml_hash_set_new(graph_size + LM_GGML_MAX_SPLITS*LM_GGML_MAX_SPLIT_INPUTS);
    sched->tensor_backend_id = calloc(sizeof(sched->tensor_backend_id[0]), sched->hash_set.size);
    sched->tensor_copies     = calloc(sizeof(sched->tensor_copies[0]), sched->hash_set.size);
    sched->node_backend_ids  = calloc(sizeof(sched->node_backend_ids[0]), graph_size);

    sched->n_backends = n_backends;
    for (int i = 0; i < n_backends; i++) {
        sched->backends[i] = backends[i];
        sched->bufts[i] = bufts ? bufts[i] : lm_ggml_backend_get_default_buffer_type(backends[i]);
    }

    sched->galloc = lm_ggml_gallocr_new_n(sched->bufts, n_backends);

    lm_ggml_backend_sched_reset(sched);

    return sched;
}

void lm_ggml_backend_sched_free(lm_ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    lm_ggml_gallocr_free(sched->galloc);
    lm_ggml_free(sched->ctx);
    free(sched->hash_set.keys);
    free(sched->tensor_backend_id);
    free(sched->tensor_copies);
    free(sched->node_backend_ids);
    free(sched);
}

void lm_ggml_backend_sched_reset(lm_ggml_backend_sched_t sched) {
    // reset state for the next run
    size_t hash_size = sched->hash_set.size;
    memset(sched->hash_set.keys,      0, sizeof(sched->hash_set.keys[0])     * hash_size); // NOLINT
    memset(sched->tensor_backend_id, -1, sizeof(sched->tensor_backend_id[0]) * hash_size);
    memset(sched->tensor_copies,      0, sizeof(sched->tensor_copies[0])     * hash_size);

    sched->is_reset = true;
}

bool lm_ggml_backend_sched_reserve(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * measure_graph) {
    lm_ggml_backend_sched_split_graph(sched, measure_graph);

    if (!lm_ggml_gallocr_reserve_n(sched->galloc, sched->graph, sched->node_backend_ids)) {
        return false;
    }

    lm_ggml_backend_sched_reset(sched);
    return true;
}

bool lm_ggml_backend_sched_graph_compute(lm_ggml_backend_sched_t sched, struct lm_ggml_cgraph * graph) {
    LM_GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + LM_GGML_MAX_SPLITS*LM_GGML_MAX_SPLIT_INPUTS);

    if (!sched->is_reset) {
        lm_ggml_backend_sched_reset(sched);
    }

    lm_ggml_backend_sched_split_graph(sched, graph);
    if (!lm_ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    if (!lm_ggml_backend_sched_compute_splits(sched)) {
        return false;
    }

    return true;
}

void lm_ggml_backend_sched_set_eval_callback(lm_ggml_backend_sched_t sched, lm_ggml_backend_sched_eval_callback callback, void * user_data) {
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int lm_ggml_backend_sched_get_n_splits(lm_ggml_backend_sched_t sched) {
    return sched->n_splits;
}

size_t lm_ggml_backend_sched_get_buffer_size(lm_ggml_backend_sched_t sched, lm_ggml_backend_t backend) {
    int backend_index = lm_ggml_backend_sched_backend_id(sched, backend);
    LM_GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    return lm_ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void lm_ggml_backend_sched_set_node_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node, lm_ggml_backend_t backend) {
    int backend_index = lm_ggml_backend_sched_backend_id(sched, backend);
    LM_GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
}

lm_ggml_backend_t lm_ggml_backend_sched_get_node_backend(lm_ggml_backend_sched_t sched, struct lm_ggml_tensor * node) {
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

void lm_ggml_backend_view_init(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor->buffer == NULL);
    LM_GGML_ASSERT(tensor->view_src != NULL);
    LM_GGML_ASSERT(tensor->view_src->buffer != NULL);
    LM_GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    tensor->backend = tensor->view_src->backend;
    lm_ggml_backend_buffer_init_tensor(buffer, tensor);
}

void lm_ggml_backend_tensor_alloc(lm_ggml_backend_buffer_t buffer, struct lm_ggml_tensor * tensor, void * addr) {
    LM_GGML_ASSERT(tensor->buffer == NULL);
    LM_GGML_ASSERT(tensor->data == NULL);
    LM_GGML_ASSERT(tensor->view_src == NULL);
    LM_GGML_ASSERT(addr >= lm_ggml_backend_buffer_get_base(buffer));
    LM_GGML_ASSERT((char *)addr + lm_ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)lm_ggml_backend_buffer_get_base(buffer) + lm_ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    lm_ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct lm_ggml_tensor * graph_copy_dup_tensor(struct lm_ggml_hash_set hash_set, struct lm_ggml_tensor ** node_copies,
    struct lm_ggml_context * ctx_allocated, struct lm_ggml_context * ctx_unallocated, struct lm_ggml_tensor * src) {

    LM_GGML_ASSERT(src != NULL);
    LM_GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = lm_ggml_hash_insert(hash_set, src);
    if (id == LM_GGML_HASHTABLE_ALREADY_EXISTS) {
        return node_copies[lm_ggml_hash_find(hash_set, src)];
    }

    struct lm_ggml_tensor * dst = lm_ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    lm_ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        struct lm_ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct lm_ggml_hash_set hash_set, struct lm_ggml_tensor ** node_copies, bool * node_init, struct lm_ggml_tensor * src) {
    size_t id = lm_ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct lm_ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        lm_ggml_backend_view_init(dst->view_src->buffer, dst);
    }
    else {
        lm_ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        struct lm_ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct lm_ggml_backend_graph_copy lm_ggml_backend_graph_copy(lm_ggml_backend_t backend, struct lm_ggml_cgraph * graph) {
    struct lm_ggml_hash_set hash_set = {
        /* .size = */ graph->visited_hash_table.size,
        /* .keys = */ calloc(sizeof(hash_set.keys[0]), graph->visited_hash_table.size) // NOLINT
    };
    struct lm_ggml_tensor ** node_copies = calloc(sizeof(node_copies[0]), hash_set.size); // NOLINT
    bool * node_init = calloc(sizeof(node_init[0]), hash_set.size);

    struct lm_ggml_init_params params = {
        /* .mem_size   = */ lm_ggml_tensor_overhead()*hash_set.size + lm_ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct lm_ggml_context * ctx_allocated = lm_ggml_init(params);
    struct lm_ggml_context * ctx_unallocated = lm_ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        fprintf(stderr, "failed to allocate context for graph copy\n");
        free(hash_set.keys);
        free(node_copies);
        free(node_init);
        lm_ggml_free(ctx_allocated);
        lm_ggml_free(ctx_unallocated);
        return (struct lm_ggml_backend_graph_copy) {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    lm_ggml_backend_buffer_t buffer = lm_ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        fprintf(stderr, "failed to allocate buffer for graph copy\n");
        free(hash_set.keys);
        free(node_copies);
        free(node_init);
        lm_ggml_free(ctx_allocated);
        lm_ggml_free(ctx_unallocated);
        return (struct lm_ggml_backend_graph_copy) {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", lm_ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct lm_ggml_cgraph * graph_copy = lm_ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct lm_ggml_tensor * node = graph->nodes[i];
        struct lm_ggml_tensor * node_copy = node_copies[lm_ggml_hash_find(hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    free(hash_set.keys);
    free(node_copies);
    free(node_init);

    return (struct lm_ggml_backend_graph_copy) {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void lm_ggml_backend_graph_copy_free(struct lm_ggml_backend_graph_copy copy) {
    lm_ggml_backend_buffer_free(copy.buffer);
    lm_ggml_free(copy.ctx_allocated);
    lm_ggml_free(copy.ctx_unallocated);
}

bool lm_ggml_backend_compare_graph_backend(lm_ggml_backend_t backend1, lm_ggml_backend_t backend2, struct lm_ggml_cgraph * graph, lm_ggml_backend_eval_callback callback, void * user_data) {
    struct lm_ggml_backend_graph_copy copy = lm_ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct lm_ggml_cgraph * g1 = graph;
    struct lm_ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    for (int i = 0; i < g1->n_nodes; i++) {
        //printf("eval %d/%d\n", i, g1->n_nodes);
        struct lm_ggml_tensor * t1 = g1->nodes[i];
        struct lm_ggml_tensor * t2 = g2->nodes[i];

        assert(t1->op == t2->op && lm_ggml_are_same_layout(t1, t2));

        struct lm_ggml_cgraph g1v = lm_ggml_graph_view(g1, i, i + 1);
        struct lm_ggml_cgraph g2v = lm_ggml_graph_view(g2, i, i + 1);

        lm_ggml_backend_graph_compute(backend1, &g1v);
        lm_ggml_backend_graph_compute(backend2, &g2v);

        if (lm_ggml_is_view_op(t1->op)) {
            continue;
        }

        // compare results, calculate rms etc
        if (!callback(i, t1, t2, user_data)) {
            break;
        }
    }

    lm_ggml_backend_graph_copy_free(copy);

    return true;
}
