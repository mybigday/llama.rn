#ifndef CODEC_MODEL_CHATTERBOX_S3G_H
#define CODEC_MODEL_CHATTERBOX_S3G_H

#include "../codec_internal.h"

struct codec_chatterbox_s3g {
    int32_t sample_rate = 24000;
    int32_t hop_size = 960;
    int32_t n_q = 1;
    int32_t codebook_size = 6561;
    bool meanflow = false;
    int32_t n_timesteps = 10;  // chatterbox_s3g.flow.n_timesteps (MeanFlow default 2)
    float cfg_rate = 0.7f;     // chatterbox_s3g.flow.cfg_rate (ignored by MeanFlow)
    float max_prompt_s = 0.0f; // chatterbox_s3g.flow.max_prompt_s: flow prompt kept (trailing seconds); 0 = all
    bool has_speaker_encoder = false;
    // LM-side speaker encoder bundled with the decoder (KittenTTS 2):
    // "xvector_sincnet" adds slots lm_cond [lm_cond_dim] and ref_tokens.
    std::string lm_encoder;
    int32_t lm_cond_dim = 0;  // chatterbox_s3g.spk.has_encoder: s3t.* + s3g.cp.* + s3g.spk.mel_fb
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
