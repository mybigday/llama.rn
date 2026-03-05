#ifndef CODEC_MODEL_QWEN3_TTS_TOKENIZER_H
#define CODEC_MODEL_QWEN3_TTS_TOKENIZER_H

#include "../codec_internal.h"
#include "mimi.h"

static constexpr int32_t CODEC_Q3T_MAX_UPSAMPLE = 8;

struct codec_qwen3_tts_tokenizer {
    int32_t sample_rate = 24000;
    int32_t hop_size = 1920;
    int32_t n_q = 16;
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 1024;
    int32_t latent_dim = 1024;
    bool has_encoder = false;
    bool has_decoder = false;

    int32_t hidden_size = 1024;
    int32_t num_hidden_layers = 8;
    int32_t num_attention_heads = 16;
    int32_t num_key_value_heads = 16;
    int32_t head_dim = 64;
    int32_t intermediate_size = 3072;
    float rope_theta = 10000.0f;
    int32_t sliding_window = 72;
    int32_t decoder_dim = 1536;

    int32_t n_upsample_rates = 0;
    int32_t n_upsampling_ratios = 0;
    int32_t upsample_rates[CODEC_Q3T_MAX_UPSAMPLE] = {};
    int32_t upsampling_ratios[CODEC_Q3T_MAX_UPSAMPLE] = {};
};

struct codec_qwen3_tts_tokenizer_impl {
    struct codec_qwen3_tts_tokenizer q3;
    struct codec_mimi mimi;
};

enum codec_status codec_qwen3_tts_tokenizer_init(struct codec_model * model);
enum codec_status codec_qwen3_tts_tokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);
enum codec_status codec_qwen3_tts_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_qwen3_tts_tokenizer_vtable();

#endif
