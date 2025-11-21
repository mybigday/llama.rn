#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_hexagon_init(void);

LM_GGML_BACKEND_API bool lm_ggml_backend_is_hexagon(lm_ggml_backend_t backend);

LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_hexagon_reg(void);

#ifdef  __cplusplus
}
#endif
