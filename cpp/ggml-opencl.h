#ifndef LM_GGML_OPENCL_H
#define LM_GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_opencl_init(void);
LM_GGML_BACKEND_API bool lm_ggml_backend_is_opencl(lm_ggml_backend_t backend);

LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_opencl_buffer_type(void);
LM_GGML_BACKEND_API lm_ggml_backend_buffer_type_t lm_ggml_backend_opencl_host_buffer_type(void);

LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // LM_GGML_OPENCL_H
