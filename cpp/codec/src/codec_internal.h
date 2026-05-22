#ifndef CODEC_INTERNAL_H
#define CODEC_INTERNAL_H

#include "codec.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <string>
#include <cstdint>
#include <vector>

struct codec_model {
    struct lm_gguf_context * gguf;
    struct lm_ggml_context * weights;
    lm_ggml_backend_t backend = nullptr;
    lm_ggml_backend_buffer_type_t buffer_type = nullptr;
    lm_ggml_backend_buffer_t weights_buffer = nullptr;

    struct codec_lm_gguf_metadata metadata;

    enum codec_arch arch;
    std::string name;
    int32_t n_tensors;
    bool use_gpu;
    int32_t n_threads;

    int32_t sample_rate;
    bool has_encoder;
    bool has_decoder;
    int32_t hop_size;
    int32_t n_q;
    int32_t codebook_size;
    int32_t n_fft;
    int32_t win_length;
    int32_t n_mels;
    int32_t latent_dim;
    const struct codec_model_vtable * vtable = nullptr;
    void * impl = nullptr;
};

// Graph cache key with named fields to avoid ambiguous p0..p3 usage.
// Only a subset of fields are used depending on kind; helper constructors must zero the unused fields.
struct codec_graph_cache_key {
    int32_t kind = 0;

    // Common decode shapes
    int32_t n_frames = 0;   // token frames / latent frames
    int32_t n_q = 0;        // number of quantizers used (codes input)
    int32_t hop = 0;        // hop size / stride used by the model

    // Encode shapes
    int32_t n_in = 0;       // input PCM samples

    // Latent decode shapes
    int32_t latent_dim = 0; // DAC latent dimension
};

typedef bool (*codec_graph_build_fn)(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out);

struct codec_graph_cache_entry {
    codec_graph_cache_key key;
    size_t required_mem_size = 0;
    codec_graph_build_fn build_fn = nullptr;
    std::vector<uint8_t> build_user_data;
    int32_t last_graph_size = 0;
    bool allocated = false; // whether this entry's graph has been allocated in the backend scheduler
};

struct codec_context {
    struct codec_model * model;
    lm_ggml_backend_t backend = nullptr;
    lm_ggml_backend_t cpu_backend = nullptr;
    lm_ggml_backend_sched_t sched = nullptr;
    struct codec_context_params params;
    std::string last_error;
    std::vector<codec_graph_cache_entry> graph_cache;
    void * eval_arena_buf = nullptr;
    size_t eval_arena_size = 0;
    lm_ggml_context * eval_ctx = nullptr;
    lm_ggml_cgraph * eval_graph = nullptr;
    lm_ggml_tensor * eval_output = nullptr;
    codec_graph_cache_entry * eval_entry = nullptr;
    codec_graph_cache_entry * eval_alloc_entry = nullptr;
    int32_t sched_reserved_graph_size = 0;
    bool sched_needs_reset = false;
};

struct codec_model_vtable {
    enum codec_arch arch;
    const char * name;
    void * (*create_impl)();
    void (*destroy_impl)(void *);
    enum codec_status (*init)(struct codec_model * model);
    enum codec_status (*encode)(
        struct codec_context * ctx,
        const std::vector<float> & pcm,
        struct codec_token_buffer * out_tokens,
        struct codec_latent_buffer * out_latent,
        struct codec_encode_params params);
    enum codec_status (*decode)(
        struct codec_context * ctx,
        const struct codec_token_buffer * tokens,
        struct codec_pcm_buffer * out_pcm,
        struct codec_decode_params params);
    enum codec_status (*decode_latent)(
        struct codec_context * ctx,
        const float * quantized_representation,
        int32_t latent_dim,
        int32_t n_frames,
        struct codec_pcm_buffer * out_pcm,
        struct codec_decode_params params);
};

enum codec_arch codec_arch_from_string(const std::string & arch);

enum codec_status codec_model_init_arch(codec_model * model);

void codec_context_set_error(codec_context * ctx, const std::string & error);
bool codec_prepare_mono_f32(const codec_audio * audio, std::vector<float> * mono, std::string * error);
void codec_token_buffer_reset(codec_token_buffer * tokens);
void codec_pcm_buffer_reset(codec_pcm_buffer * pcm);
void codec_latent_buffer_reset(codec_latent_buffer * latent);

const float * codec_tensor_data_f32(const lm_ggml_tensor * t);
int64_t codec_ne(const lm_ggml_tensor * t, int dim);
bool codec_tensor_as_vec_f32(const lm_ggml_tensor * t, std::vector<float> * out);

char * codec_strdup(const char * s);
void codec_metadata_add(codec_lm_gguf_metadata * meta, const char * key, const std::string & value);
std::string codec_lm_gguf_value_to_string(lm_gguf_context * gf, int key_id);
void codec_collect_lm_gguf_metadata(codec_model * model);
int32_t codec_read_i32_kv(lm_gguf_context * gf, const char * key, int32_t fallback);
int32_t codec_read_i32_kv_any(lm_gguf_context * gf, const char * const * keys, size_t n_keys, int32_t fallback);
float codec_read_f32_kv(lm_gguf_context * gf, const char * key, float fallback);
bool codec_read_bool_kv(lm_gguf_context * gf, const char * key, bool fallback);
int codec_count_tensors_with_prefix(const codec_model * model, const char * prefix);
int32_t codec_infer_n_q_from_tensor_names(const codec_model * model);

bool codec_safe_add_i32(int32_t a, int32_t b, int32_t * out);
bool codec_safe_mul_i32(int32_t a, int32_t b, int32_t * out);

#endif
