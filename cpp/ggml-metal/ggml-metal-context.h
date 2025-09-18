#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct lm_ggml_metal * lm_ggml_metal_t;

lm_ggml_metal_t lm_ggml_metal_init(lm_ggml_metal_device_t dev);
void lm_ggml_metal_free(lm_ggml_metal_t ctx);

void lm_ggml_metal_synchronize(lm_ggml_metal_t ctx);

void lm_ggml_metal_set_tensor_async(lm_ggml_metal_t ctx, struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void lm_ggml_metal_get_tensor_async(lm_ggml_metal_t ctx, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size);

enum lm_ggml_status lm_ggml_metal_graph_compute (lm_ggml_metal_t ctx, struct lm_ggml_cgraph * gf);
void             lm_ggml_metal_graph_optimize(lm_ggml_metal_t ctx, struct lm_ggml_cgraph * gf);

void lm_ggml_metal_set_n_cb            (lm_ggml_metal_t ctx, int n_cb);
void lm_ggml_metal_set_abort_callback  (lm_ggml_metal_t ctx, lm_ggml_abort_callback abort_callback, void * user_data);
bool lm_ggml_metal_supports_family     (lm_ggml_metal_t ctx, int family);
void lm_ggml_metal_capture_next_compute(lm_ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
