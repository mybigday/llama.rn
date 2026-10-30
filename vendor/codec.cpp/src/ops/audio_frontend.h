#ifndef CODEC_OPS_AUDIO_FRONTEND_H
#define CODEC_OPS_AUDIO_FRONTEND_H

#include "../codec_internal.h"

#include <vector>

// In-graph audio front-ends (resampling, framed DFTs, mel features). Inputs
// are mono F32 PCM tensors of shape [n]; constant tables (resampler kernels,
// DFT bases, windows) are graph constants (codec_graph_const), so these must
// be called from a graph build_fn. Outputs keep time in ne[0] unless noted.

// Output length of codec_op_resample_sinc_hann for `n` input samples.
int64_t codec_resample_out_len(int64_t n, int32_t in_sr, int32_t out_sr);

// torchaudio.functional.resample, method sinc_interp_hann
// (lowpass_filter_width 6, rolloff 0.99). Returns x itself when in_sr == out_sr.
ggml_tensor * codec_op_resample_sinc_hann(ggml_context * ctx, ggml_tensor * x, int32_t in_sr, int32_t out_sr);

// Whisper-style log-mel (S3TokenizerV2): zero-pad to a multiple of
// `pad_multiple`, reflect-pad n_fft / 2, power STFT with `window`, drop the last
// frame, log10(clamp(mel, 1e-10)), floor at max - 8, (x + 4) / 4.
// `mel_fb` is [n_fft / 2 + 1, n_mels]. Returns [n_frames, n_mels].
int32_t codec_whisper_log_mel_frames(int64_t n, int32_t hop, int32_t pad_multiple);
ggml_tensor * codec_op_whisper_log_mel(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, const std::vector<float> & window,
    int32_t n_fft, int32_t hop, int32_t pad_multiple, const char * lut_name);

// Linear mel spectrogram: reflect-pad `pad` per side, frames
// x[f * hop : f * hop + n_fft] weighted by `window`, then the power spectrum
// |X|^2 or, with `magnitude`, sqrt(|X|^2 + mag_eps); projected by `mel_fb`
// ([n_fft / 2 + 1, n_mels]). Returns [n_mels, n_frames]; runs on CPU (see
// codec_graph_cpu_scope). `lut_name` names the windowed DFT basis constant.
int32_t codec_mel_spectrogram_frames(int64_t n, int32_t n_fft, int32_t hop, int32_t pad);
ggml_tensor * codec_op_mel_spectrogram(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, const std::vector<float> & window,
    int32_t n_fft, int32_t hop, int32_t pad, bool magnitude, float mag_eps, const char * lut_name);

// Matcha-TTS / CosyVoice `mel_spectrogram`: reflect-pad (n_fft - hop) / 2,
// periodic-Hann STFT (center=False), magnitude sqrt(re^2 + im^2 + 1e-9), mel,
// log(clamp(x, 1e-5)). `mel_fb` is [n_fft / 2 + 1, n_mels].
// Returns [n_mels, n_frames] (mel bins in ne[0]).
int32_t codec_matcha_log_mel_frames(int64_t n, int32_t n_fft, int32_t hop);
ggml_tensor * codec_op_matcha_log_mel(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, int32_t n_fft, int32_t hop);

// torchaudio.compliance.kaldi.fbank defaults at 16 kHz (25 ms / 10 ms frames,
// snip_edges, DC removal, preemphasis 0.97, Povey window, 512-point power
// spectrum, log(max(x, FLT_EPSILON))) followed by per-bin mean subtraction
// over frames (CAMPPlus `extract_feature`). `mel_fb` is [257, n_mels].
// Returns [n_frames, n_mels].
int32_t codec_kaldi_fbank_frames(int64_t n);
ggml_tensor * codec_op_kaldi_fbank_cmn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb);

#endif
