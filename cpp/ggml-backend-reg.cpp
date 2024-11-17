#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cstring>
#include <vector>

// Backend registry

#ifdef LM_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef LM_GGML_USE_METAL
#include <TargetConditionals.h>

#if !TARGET_OS_SIMULATOR
#include "ggml-metal.h"
#endif

#endif

#ifdef LM_GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef LM_GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef LM_GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef LM_GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef LM_GGML_USE_AMX
#  include "ggml-amx.h"
#endif

#ifdef LM_GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef LM_GGML_USE_KOMPUTE
#include "ggml-kompute.h"
#endif

struct lm_ggml_backend_registry {
    std::vector<lm_ggml_backend_reg_t> backends;
    std::vector<lm_ggml_backend_dev_t> devices;

    lm_ggml_backend_registry() {
#ifdef LM_GGML_USE_CUDA
        register_backend(lm_ggml_backend_cuda_reg());
#endif
#ifdef LM_GGML_USE_METAL

#if !TARGET_OS_SIMULATOR
        register_backend(lm_ggml_backend_metal_reg());
#endif

#endif
#ifdef LM_GGML_USE_SYCL
        register_backend(lm_ggml_backend_sycl_reg());
#endif
#ifdef LM_GGML_USE_VULKAN
        register_backend(lm_ggml_backend_vk_reg());
#endif
#ifdef LM_GGML_USE_CANN
        register_backend(lm_ggml_backend_cann_reg());
#endif
#ifdef LM_GGML_USE_BLAS
        register_backend(lm_ggml_backend_blas_reg());
#endif
#ifdef LM_GGML_USE_RPC
        register_backend(lm_ggml_backend_rpc_reg());
#endif
#ifdef LM_GGML_USE_AMX
        register_backend(lm_ggml_backend_amx_reg());
#endif
#ifdef LM_GGML_USE_KOMPUTE
        register_backend(lm_ggml_backend_kompute_reg());
#endif

        register_backend(lm_ggml_backend_cpu_reg());
    }

    void register_backend(lm_ggml_backend_reg_t reg) {
        if (!reg) {
            return;
        }

#ifndef NDEBUG
        LM_GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, lm_ggml_backend_reg_name(reg), lm_ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back(reg);
        for (size_t i = 0; i < lm_ggml_backend_reg_dev_count(reg); i++) {
            register_device(lm_ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(lm_ggml_backend_dev_t device) {
#ifndef NDEBUG
        LM_GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, lm_ggml_backend_dev_name(device), lm_ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }
};

static lm_ggml_backend_registry & get_reg() {
    static lm_ggml_backend_registry reg;
    return reg;
}

// Internal API
void lm_ggml_backend_register(lm_ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void lm_ggml_backend_device_register(lm_ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
size_t lm_ggml_backend_reg_count() {
    return get_reg().backends.size();
}

lm_ggml_backend_reg_t lm_ggml_backend_reg_get(size_t index) {
    LM_GGML_ASSERT(index < lm_ggml_backend_reg_count());
    return get_reg().backends[index];
}

lm_ggml_backend_reg_t lm_ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < lm_ggml_backend_reg_count(); i++) {
        lm_ggml_backend_reg_t reg = lm_ggml_backend_reg_get(i);
        if (std::strcmp(lm_ggml_backend_reg_name(reg), name) == 0) {
            return reg;
        }
    }
    return NULL;
}

// Device enumeration
size_t lm_ggml_backend_dev_count() {
    return get_reg().devices.size();
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_get(size_t index) {
    LM_GGML_ASSERT(index < lm_ggml_backend_dev_count());
    return get_reg().devices[index];
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < lm_ggml_backend_dev_count(); i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (strcmp(lm_ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    return NULL;
}

lm_ggml_backend_dev_t lm_ggml_backend_dev_by_type(enum lm_ggml_backend_dev_type type) {
    for (size_t i = 0; i < lm_ggml_backend_dev_count(); i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (lm_ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return NULL;
}

// Convenience functions
lm_ggml_backend_t lm_ggml_backend_init_by_name(const char * name, const char * params) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_name(name);
    if (!dev) {
        return NULL;
    }
    return lm_ggml_backend_dev_init(dev, params);
}

lm_ggml_backend_t lm_ggml_backend_init_by_type(enum lm_ggml_backend_dev_type type, const char * params) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_type(type);
    if (!dev) {
        return NULL;
    }
    return lm_ggml_backend_dev_init(dev, params);
}

lm_ggml_backend_t lm_ggml_backend_init_best(void) {
    lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = lm_ggml_backend_dev_by_type(LM_GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return NULL;
    }
    return lm_ggml_backend_dev_init(dev, NULL);
}
