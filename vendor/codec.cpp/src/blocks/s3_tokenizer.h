#ifndef CODEC_BLOCKS_S3_TOKENIZER_H
#define CODEC_BLOCKS_S3_TOKENIZER_H

#include "../codec_internal.h"

// S3TokenizerV2 (CosyVoice 2 / Chatterbox speech tokenizer, 25 Hz, 6561 FSQ
// codes): Whisper-style log-mel → 2 strided convs → FSMN transformer → FSQ.
// Weights live under `s3t.` (`s3t.mel_filters`, optional `s3t.window`,
// `s3t.enc.*`, `s3t.q.proj.*`).
struct codec_s3_tokenizer_params {
    int32_t n_fft = 400;
    int32_t n_mels = 128;
    int32_t hidden = 1280;
    int32_t n_heads = 20;
    int32_t n_layers = 6;
    int32_t fsmn_kernel = 31;
    float rope_theta = 10000.0f;
    // Zero-pad the PCM to a multiple of this many samples before the log-mel
    // (Chatterbox pads to whole tokens; CosyVoice's frontend does not pad: 1).
    int32_t pad_multiple = 640;
};

// Tokens produced for `n_samples` of 16 kHz audio.
int32_t codec_s3_tokenizer_n_tokens(int64_t n_samples, int32_t pad_multiple = 640);

// 16 kHz mono PCM [n] → token ids I32 [n_tokens]. Call from a graph build_fn.
ggml_tensor * codec_block_s3_tokenizer(
    ggml_context * ctx,
    const codec_model * model,
    const codec_s3_tokenizer_params & p,
    ggml_tensor * pcm16k);

// S3Tokenizer v1 (CosyVoice-300M, 50 Hz, 4096-entry VQ): Whisper log-mel →
// Conv(k3) → Conv(k3, s2) → sinusoid PE → Whisper encoder blocks → L2 norm →
// nearest codeword. Weights `s3t.mel_filters`, `s3t.enc.{conv1,conv2,pos,blk.<i>}`,
// `s3t.q.embed` [1280, 4096], `s3t.q.embed_sq` [4096]. No padding.
int32_t codec_s3_tokenizer_v1_n_tokens(int64_t n_samples);
ggml_tensor * codec_block_s3_tokenizer_v1(
    ggml_context * ctx,
    const codec_model * model,
    int32_t n_layers,
    ggml_tensor * pcm16k);

#endif
