#ifndef CODEC_LM_SPEAKER_CHATTERBOX_H
#define CODEC_LM_SPEAKER_CHATTERBOX_H

#include "lm_internal.h"

// Chatterbox VoiceEncoder + cond_enc + perceiver speaker encoder.
//
// Init / free are called from `codec_lm_create` / `codec_lm_free` when
// `codec.speaker.encoder_arch == "chatterbox_voice_encoder"`.
// `chatterbox_speaker_encode` is the runtime impl that backs the public
// `codec_lm_speaker_encode` API after arch-keyed dispatch.

bool chatterbox_speaker_init(codec_lm * lm);
void chatterbox_speaker_free(codec_lm * lm);

enum codec_status chatterbox_speaker_encode(
    codec_lm * lm,
    const struct codec_audio * ref_pcm,
    const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
    float emotion,
    float * out, int32_t out_n_elems);

// Skip the mel + LSTM front-end; feed a pre-computed `speaker_emb`
// (256-d, matching VoiceEncoder's projection output) straight into the
// cond_enc graph.  Used by the public
// `codec_lm_speaker_encode_from_embedding` entry.
enum codec_status chatterbox_speaker_encode_from_emb(
    codec_lm * lm,
    const float * speaker_emb,
    const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
    float emotion,
    float * out, int32_t out_n_elems);

#endif
