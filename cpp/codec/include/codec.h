#ifndef CODEC_H
#define CODEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CODEC_DEFAULT_SEED (int32_t)0xFFFFFFFFu

enum codec_arch {
    CODEC_ARCH_UNKNOWN = 0,
    CODEC_ARCH_WAVTOKENIZER_LARGE = 1,
    CODEC_ARCH_DAC = 2,
    CODEC_ARCH_MIMI = 3,
    CODEC_ARCH_QWEN3_TTS_TOKENIZER = 4,
};

enum codec_status {
    CODEC_STATUS_SUCCESS = 0,
    CODEC_STATUS_INVALID_ARG = 1,
    CODEC_STATUS_INVALID_STATE = 2,
    CODEC_STATUS_IO_ERROR = 3,
    CODEC_STATUS_NOT_SUPPORTED = 4,
    CODEC_STATUS_INTERNAL_ERROR = 5,
};

enum codec_pcm_type {
    CODEC_PCM_TYPE_F32 = 0,
    CODEC_PCM_TYPE_I16 = 1,
};

enum codec_batch_mode {
    CODEC_BATCH_MODE_CODES = 0,
    CODEC_BATCH_MODE_LATENT = 1,
};

struct codec_model;
struct codec_context;

struct codec_model_params {
    bool use_gpu;
    int32_t n_threads;
};

struct codec_context_params {
    int32_t seed;
};

struct codec_encode_params {
    int32_t n_threads;
    int32_t frame_size;
    int32_t hop_size;
    int32_t n_q;
};

struct codec_decode_params {
    int32_t n_threads;
    int32_t n_q;
};

struct codec_audio {
    const void * data;
    int32_t n_samples;
    int32_t sample_rate;
    int32_t n_channels;
    enum codec_pcm_type pcm_type;
};

struct codec_token_buffer {
    int32_t * data;
    int32_t n_tokens;
    int32_t n_frames;
    int32_t n_q;
    int32_t codebook_size;
    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_pcm_buffer {
    float * data;
    int32_t n_samples;
    int32_t sample_rate;
    int32_t n_channels;
};

struct codec_latent_buffer {
    float * data;
    int32_t latent_dim;
    int32_t n_frames;
    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_batch {
    int32_t n_seq;
    int32_t n_seq_alloc;
    int32_t n_seq_max;

    int32_t * seq_id;
    int32_t * n_frames;
    int32_t * n_q;

    enum codec_batch_mode mode;

    int32_t * codes;
    int32_t codes_size;
    int32_t codes_used;

    float * latent;
    int32_t latent_dim;
    int32_t latent_size;
    int32_t latent_used;

    int32_t * codes_offset;
    int32_t * latent_offset;

    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_lm_gguf_kv {
    const char * key;
    const char * value;
};

struct codec_lm_gguf_metadata {
    struct codec_lm_gguf_kv * items;
    size_t n_items;
};

struct codec_model_params codec_model_default_params(void);
struct codec_context_params codec_context_default_params(void);
struct codec_encode_params codec_encode_default_params(void);
struct codec_decode_params codec_decode_default_params(void);

struct codec_model * codec_model_load_from_file(const char * path_model, struct codec_model_params params);
void codec_model_free(struct codec_model * model);

struct codec_context * codec_init_from_model(struct codec_model * model, struct codec_context_params params);
void codec_free(struct codec_context * ctx);

enum codec_status codec_encode(struct codec_context * ctx, const struct codec_audio * audio, struct codec_token_buffer * out_tokens, struct codec_encode_params params);
enum codec_status codec_encode_latent(
    struct codec_context * ctx,
    const struct codec_audio * audio,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);
enum codec_status codec_decode(struct codec_context * ctx, const struct codec_token_buffer * tokens, struct codec_pcm_buffer * out_pcm, struct codec_decode_params params);
enum codec_status codec_decode_quantized_representation(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
struct codec_batch codec_batch_init_codes(int32_t n_seq_alloc, int32_t codes_alloc_total, int32_t n_seq_max);
struct codec_batch codec_batch_init_latent(int32_t n_seq_alloc, int32_t latent_dim, int32_t latent_alloc_total, int32_t n_seq_max);
void codec_batch_free(struct codec_batch batch);
int32_t codec_batch_add_seq_codes(struct codec_batch * batch, int32_t seq_id, int32_t n_frames, int32_t n_q, const int32_t * codes);
int32_t codec_batch_add_seq_latent(struct codec_batch * batch, int32_t seq_id, int32_t n_frames, const float * latent, int32_t latent_dim);
enum codec_status codec_decode_batch(
    struct codec_context * ctx,
    const struct codec_batch * batch,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
void codec_token_buffer_free(struct codec_token_buffer * tokens);
void codec_pcm_buffer_free(struct codec_pcm_buffer * pcm);
void codec_latent_buffer_free(struct codec_latent_buffer * latent);

const char * codec_get_last_error(const struct codec_context * ctx);

enum codec_arch codec_model_arch(const struct codec_model * model);
const char * codec_model_name(const struct codec_model * model);
int32_t codec_model_n_tensors(const struct codec_model * model);

int32_t codec_model_sample_rate(const struct codec_model * model);
bool codec_model_has_encoder(const struct codec_model * model);
bool codec_model_has_decoder(const struct codec_model * model);
int32_t codec_model_n_q(const struct codec_model * model);
int32_t codec_model_codebook_size(const struct codec_model * model);
int32_t codec_model_hop_size(const struct codec_model * model);
int32_t codec_model_n_fft(const struct codec_model * model);
int32_t codec_model_win_length(const struct codec_model * model);
int32_t codec_model_n_mels(const struct codec_model * model);
int32_t codec_model_latent_dim(const struct codec_model * model);

const struct codec_lm_gguf_metadata * codec_model_metadata(const struct codec_model * model);
void codec_metadata_free(struct codec_lm_gguf_metadata * meta);

const char * codec_arch_name(enum codec_arch arch);

#ifdef __cplusplus
}
#endif

#endif // CODEC_H
