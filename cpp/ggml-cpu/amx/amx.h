#include "ggml-backend.h"
#include "ggml-cpu-impl.h"

// GGML internal header

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
lm_ggml_backend_buffer_type_t lm_ggml_backend_amx_buffer_type(void);
#endif
