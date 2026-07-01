#ifndef CODEC_LM_SPEAKER_QWEN3_TTS_H
#define CODEC_LM_SPEAKER_QWEN3_TTS_H

#include "lm_internal.h"

// Qwen3-TTS ECAPA-TDNN speaker encoder.  Per the proposed framework:
//   * `_init`  — locate tensors + dequantise to F32 host buffers.
//   * `_free`  — release host buffers / contexts.
//   * `_encode` — PCM → (1, hidden_dim) F32.  No "from_emb" variant —
//     the x-vector is a single row that the LM consumes directly, so
//     skipping ECAPA-TDNN means there's no useful intermediate to
//     supply.

bool qwen3_tts_speaker_init(codec_lm * lm);
void qwen3_tts_speaker_free(codec_lm * lm);

enum codec_status qwen3_tts_speaker_encode(
    codec_lm * lm,
    const struct codec_audio * ref_pcm,
    const int32_t * ref_speech_tokens, int32_t n_ref_speech_tokens,
    float emotion,
    float * out, int32_t out_n_elems);

#endif
