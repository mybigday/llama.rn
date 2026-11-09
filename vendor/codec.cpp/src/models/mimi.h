#ifndef CODEC_MODEL_MIMI_H
#define CODEC_MODEL_MIMI_H

#include "../codec_internal.h"

#include <array>
#include <string>

struct codec_mimi {
    int32_t sample_rate = 24000;
    int32_t hop_size = 1920;
    int32_t n_q = 32;
    int32_t num_semantic_quantizers = 1;
    int32_t codebook_size = 2048;
    int32_t codebook_dim = 256;
    int32_t hidden_size = 512;
    int32_t num_hidden_layers = 8;
    int32_t num_attention_heads = 8;
    int32_t head_dim = 64;
    int32_t intermediate_size = 2048;
    float rope_theta = 10000.0f;
    float rope_scaling_factor = 1.0f;
    bool has_encoder = false;
    bool has_decoder = false;
};


// Mimi's encoder graph, also built as a subgraph of other models' graphs
// (Qwen3-TTS speakers encode their reference with it).
struct mimi_encode_frontend_conv_desc {
    int32_t out_c;
    int32_t in_c;
    int32_t kernel;
    int32_t stride;
};

struct mimi_encode_frontend_build {
    int32_t n_in;
    std::array<mimi_encode_frontend_conv_desc, 14> conv;
};

struct mimi_encode_transformer_build {
    int32_t t;
    int32_t c;
    int32_t n_layers;
    int32_t n_heads;
    int32_t head_dim;
    int32_t intermediate_size;
    float rope_theta;
    float rope_scaling_factor;
};

struct mimi_encode_downsample_build {
    int32_t t;
    int32_t in_c;
    int32_t out_c;
    int32_t kernel;
    int32_t stride;
};

constexpr int32_t CODEC_MIMI_MAX_RVQ_LAYERS = 32;
constexpr int32_t CODEC_MIMI_RVQ_GROUP_SEMANTIC = 0;
constexpr int32_t CODEC_MIMI_RVQ_GROUP_ACOUSTIC = 1;

struct mimi_encode_rvq_layer_desc {
    int32_t group = CODEC_MIMI_RVQ_GROUP_SEMANTIC;
    int32_t group_layer = 0;
};

struct mimi_encode_build {
    mimi_encode_frontend_build frontend = {};
    mimi_encode_transformer_build transformer = {};
    mimi_encode_downsample_build downsample = {};
    int32_t n_q = 0;
    int32_t codebook_dim = 0;
    int32_t codebook_size = 0;
    std::array<mimi_encode_rvq_layer_desc, CODEC_MIMI_MAX_RVQ_LAYERS> rvq_layers = {};
    const codec_model * model = nullptr;
};

bool codec_mimi_init_encode_build(
    codec_context * ctx, const codec_mimi * mimi, int32_t n_in, mimi_encode_build * build, std::string * err);
// pcm: F32 [n_in, 1] at the model rate -> *out: I32 codes [t, n_q].
bool codec_mimi_build_encode_from(ggml_context * ctx_eval, const mimi_encode_build * build, ggml_tensor * pcm,
                                  ggml_tensor ** out);

enum codec_status codec_mimi_init(struct codec_model * model);

enum codec_status codec_mimi_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);

enum codec_status codec_mimi_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);

enum codec_status codec_mimi_encode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params);
enum codec_status codec_mimi_decode_with(
    struct codec_context * ctx,
    struct codec_mimi * mimi,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_mimi_vtable();

#endif
