#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lm_ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct lm_ggml_metal_device * lm_ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct lm_ggml_metal_cv * lm_ggml_metal_cv_t;

lm_ggml_metal_cv_t lm_ggml_metal_cv_init(void);
void lm_ggml_metal_cv_free(lm_ggml_metal_cv_t cv);

void lm_ggml_metal_cv_set_int16(lm_ggml_metal_cv_t cv, int16_t value, int32_t idx);
void lm_ggml_metal_cv_set_int32(lm_ggml_metal_cv_t cv, int32_t value, int32_t idx);
void lm_ggml_metal_cv_set_bool (lm_ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct lm_ggml_metal_pipeline * lm_ggml_metal_pipeline_t;

lm_ggml_metal_pipeline_t lm_ggml_metal_pipeline_init(void);
void lm_ggml_metal_pipeline_free(lm_ggml_metal_pipeline_t pipeline);

void lm_ggml_metal_pipeline_set_nsg(lm_ggml_metal_pipeline_t pipeline, int nsg);
int  lm_ggml_metal_pipeline_get_nsg(lm_ggml_metal_pipeline_t pipeline);

void lm_ggml_metal_pipeline_set_nr0(lm_ggml_metal_pipeline_t pipeline, int nr0);
int  lm_ggml_metal_pipeline_get_nr0(lm_ggml_metal_pipeline_t pipeline);

void lm_ggml_metal_pipeline_set_nr1(lm_ggml_metal_pipeline_t pipeline, int nr1);
int  lm_ggml_metal_pipeline_get_nr1(lm_ggml_metal_pipeline_t pipeline);

void   lm_ggml_metal_pipeline_set_smem(lm_ggml_metal_pipeline_t pipeline, size_t smem);
size_t lm_ggml_metal_pipeline_get_smem(lm_ggml_metal_pipeline_t pipeline);

int lm_ggml_metal_pipeline_max_theads_per_threadgroup(lm_ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct lm_ggml_metal_pipelines * lm_ggml_metal_pipelines_t;

lm_ggml_metal_pipelines_t lm_ggml_metal_pipelines_init(void);
void lm_ggml_metal_pipelines_free(lm_ggml_metal_pipelines_t ppls);

void                  lm_ggml_metal_pipelines_add(lm_ggml_metal_pipelines_t ppls, const char * name, lm_ggml_metal_pipeline_t pipeline);
lm_ggml_metal_pipeline_t lm_ggml_metal_pipelines_get(lm_ggml_metal_pipelines_t ppls, const char * name);

//
// MTLCommandBuffer wrapper
//

typedef void * lm_ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct lm_ggml_metal_encoder * lm_ggml_metal_encoder_t;

lm_ggml_metal_encoder_t lm_ggml_metal_encoder_init(lm_ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void lm_ggml_metal_encoder_free(lm_ggml_metal_encoder_t encoder);

void lm_ggml_metal_encoder_debug_group_push(lm_ggml_metal_encoder_t encoder, const char * name);
void lm_ggml_metal_encoder_debug_group_pop (lm_ggml_metal_encoder_t encoder);

void lm_ggml_metal_encoder_set_pipeline(lm_ggml_metal_encoder_t encoder, lm_ggml_metal_pipeline_t pipeline);

void lm_ggml_metal_encoder_set_bytes (lm_ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void lm_ggml_metal_encoder_set_buffer(lm_ggml_metal_encoder_t encoder, struct lm_ggml_metal_buffer_id buffer, int idx);

void lm_ggml_metal_encoder_set_threadgroup_memory_size(lm_ggml_metal_encoder_t encoder, size_t size, int idx);

void lm_ggml_metal_encoder_dispatch_threadgroups(lm_ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void lm_ggml_metal_encoder_memory_barrier(lm_ggml_metal_encoder_t encoder);

void lm_ggml_metal_encoder_end_encoding(lm_ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

typedef struct lm_ggml_metal_library * lm_ggml_metal_library_t;

lm_ggml_metal_library_t lm_ggml_metal_library_init(lm_ggml_metal_device_t dev);
void lm_ggml_metal_library_free(lm_ggml_metal_library_t lib);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline    (lm_ggml_metal_library_t lib, const char * name);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_compile_pipeline(lm_ggml_metal_library_t lib, const char * base, const char * name, lm_ggml_metal_cv_t cv);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_base              (lm_ggml_metal_library_t lib, enum lm_ggml_op op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_cpy               (lm_ggml_metal_library_t lib, enum lm_ggml_type tsrc, enum lm_ggml_type tdst);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_pool_2d           (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op, enum lm_ggml_op_pool op_pool);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_get_rows          (lm_ggml_metal_library_t lib, enum lm_ggml_type tsrc);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_set_rows          (lm_ggml_metal_library_t lib, enum lm_ggml_type tidx, enum lm_ggml_type tdst);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_repeat            (lm_ggml_metal_library_t lib, enum lm_ggml_type tsrc);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_unary             (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_glu               (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_sum               (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_sum_rows          (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_soft_max          (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_ssm_conv          (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_ssm_scan          (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_rwkv              (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mv_ext        (lm_ggml_metal_library_t lib, enum lm_ggml_type tsrc0, enum lm_ggml_type tsrc1, int nsg, int nxpsg, int r1ptg);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mm            (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mv            (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mm_id_map0    (lm_ggml_metal_library_t lib, int ne02, int ne20);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mm_id         (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_mul_mv_id         (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_argmax            (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_argsort           (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_bin               (lm_ggml_metal_library_t lib, enum lm_ggml_op op, int32_t n_fuse, bool row);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_l2_norm           (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_group_norm        (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_norm              (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op, int32_t n_fuse);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_rope              (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_im2col            (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_conv_transpose_1d (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_conv_transpose_2d (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_upscale           (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_pad               (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_pad_reflect_1d    (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_arange            (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_timestep_embedding(lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_opt_step_adamw    (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);
lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_opt_step_sgd      (lm_ggml_metal_library_t lib, const struct lm_ggml_tensor * op);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        lm_ggml_metal_library_t lib,
        const struct lm_ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        lm_ggml_metal_library_t lib,
        const struct lm_ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_flash_attn_ext(
        lm_ggml_metal_library_t lib,
        const struct lm_ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        lm_ggml_metal_library_t lib,
        const struct lm_ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg);

lm_ggml_metal_pipeline_t lm_ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        lm_ggml_metal_library_t lib,
        const struct lm_ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

//
// device
//

struct lm_ggml_metal_device_props {
    char name[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;
};

lm_ggml_metal_device_t lm_ggml_metal_device_init(void);
void lm_ggml_metal_device_free(lm_ggml_metal_device_t dev);

// return a singleton that is automatically destroyed when the program exits
lm_ggml_metal_device_t lm_ggml_metal_device_get(void);

void * lm_ggml_metal_device_get_obj  (lm_ggml_metal_device_t dev); // id<MTLDevice>
void * lm_ggml_metal_device_get_queue(lm_ggml_metal_device_t dev); // id<MTLCommandQueue>

lm_ggml_metal_library_t lm_ggml_metal_device_get_library(lm_ggml_metal_device_t dev);

void lm_ggml_metal_device_get_memory(lm_ggml_metal_device_t dev, size_t * free, size_t * total);
bool lm_ggml_metal_device_supports_op(lm_ggml_metal_device_t dev, const struct lm_ggml_tensor * op);

const struct lm_ggml_metal_device_props * lm_ggml_metal_device_get_props(lm_ggml_metal_device_t dev);

//
// device buffers
//

typedef struct lm_ggml_metal_buffer * lm_ggml_metal_buffer_t;

lm_ggml_metal_buffer_t lm_ggml_metal_buffer_init(lm_ggml_metal_device_t dev, size_t size, bool shared);
lm_ggml_metal_buffer_t lm_ggml_metal_buffer_map (lm_ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   lm_ggml_metal_buffer_free     (lm_ggml_metal_buffer_t buf);
void * lm_ggml_metal_buffer_get_base (lm_ggml_metal_buffer_t buf);
bool   lm_ggml_metal_buffer_is_shared(lm_ggml_metal_buffer_t buf);

void   lm_ggml_metal_buffer_memset_tensor(lm_ggml_metal_buffer_t buf, struct lm_ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   lm_ggml_metal_buffer_set_tensor   (lm_ggml_metal_buffer_t buf, struct lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   lm_ggml_metal_buffer_get_tensor   (lm_ggml_metal_buffer_t buf, const struct lm_ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   lm_ggml_metal_buffer_clear        (lm_ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct lm_ggml_metal_buffer_id lm_ggml_metal_buffer_get_id(lm_ggml_metal_buffer_t buf, const struct lm_ggml_tensor * t);

#ifdef __cplusplus
}
#endif
