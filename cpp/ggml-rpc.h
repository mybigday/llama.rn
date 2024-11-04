#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define LM_GGML_RPC_MAX_SERVERS       16

// backend API
LM_GGML_API lm_ggml_backend_t lm_ggml_backend_rpc_init(const char * endpoint);
LM_GGML_API bool lm_ggml_backend_is_rpc(lm_ggml_backend_t backend);

LM_GGML_API lm_ggml_backend_buffer_type_t lm_ggml_backend_rpc_buffer_type(const char * endpoint);

LM_GGML_API void lm_ggml_backend_rpc_get_device_memory(const char * endpoint, size_t * free, size_t * total);

LM_GGML_API void lm_ggml_backend_rpc_start_server(lm_ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);

LM_GGML_API lm_ggml_backend_reg_t lm_ggml_backend_rpc_reg(void);

LM_GGML_API lm_ggml_backend_dev_t lm_ggml_backend_rpc_add_device(const char * endpoint);

#ifdef  __cplusplus
}
#endif
