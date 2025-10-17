#pragma once

#include "ggml.h"
#include "ggml-backend.h"


#ifdef  __cplusplus
extern "C" {
#endif

// backend API
LM_GGML_BACKEND_API lm_ggml_backend_t lm_ggml_backend_blas_init(void);

LM_GGML_BACKEND_API bool lm_ggml_backend_is_blas(lm_ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
LM_GGML_BACKEND_API void lm_ggml_backend_blas_set_n_threads(lm_ggml_backend_t backend_blas, int n_threads);

LM_GGML_BACKEND_API lm_ggml_backend_reg_t lm_ggml_backend_blas_reg(void);


#ifdef  __cplusplus
}
#endif
