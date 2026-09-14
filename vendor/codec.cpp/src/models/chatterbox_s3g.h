#ifndef CODEC_MODEL_CHATTERBOX_S3G_H
#define CODEC_MODEL_CHATTERBOX_S3G_H

#include "../codec_internal.h"

struct codec_chatterbox_s3g {
    int32_t sample_rate = 24000;
    int32_t hop_size = 960;
    int32_t n_q = 1;
    int32_t codebook_size = 6561;
    bool meanflow = false;
    bool has_builtin_conditioning = false;
    int32_t builtin_prompt_token_len = 0;
    int32_t builtin_prompt_feat_frames = 0;
    int32_t builtin_prompt_feat_dim = 0;
    int32_t builtin_embedding_dim = 0;
    std::vector<int32_t> builtin_prompt_token;
    bool has_encoder = false;
    bool has_decoder = true;
};

enum codec_status codec_chatterbox_s3g_init(struct codec_model * model);
enum codec_status codec_chatterbox_s3g_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
const struct codec_model_vtable * codec_chatterbox_s3g_vtable();

#endif
