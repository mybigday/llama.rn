#ifndef CODEC_RUNTIME_AUDIO_DSP_H
#define CODEC_RUNTIME_AUDIO_DSP_H

#include <cstdint>
#include <string>
#include <vector>

// Inverse short-time Fourier transform from a Vocos/ISTFTHead-style head tensor.
// `head` is laid out as `[out_dim, n_frames]` (column-major in time): for each
// time frame, the first `n_bins = out_dim/2` values are log-magnitudes and the
// next `n_bins` values are phases. `n_fft = 2 * (n_bins - 1)`. If `window` is
// non-null and matches `n_fft`, it is used; otherwise a symmetric Hann window
// is computed. `skip_dc_nyquist` controls bin handling (Soprano zeros DC/Nyquist;
// Vocos-style includes them). `trim_pad_override` overrides the default trim:
// pass -1 to use the default (`n_fft/2` if `skip_dc_nyquist`, otherwise
// `(n_fft - hop)/2`); pass `n_fft/2` for HiFi-GAN-style `center=True` iSTFT.
bool codec_runtime_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    bool skip_dc_nyquist,
    int32_t trim_pad_override,
    std::vector<float> * out_pcm,
    std::string * err);

// Periodic Hann window of length `n_fft` (matches scipy.get_window("hann", n,
// fftbins=True)). Used by HiFi-GAN-style STFT pipelines.
void codec_runtime_periodic_hann_window(int32_t n_fft, std::vector<float> * out);

// STFT analysis basis kernels with the periodic Hann window pre-multiplied.
// Both outputs are laid out as ggml conv1d weights with ne = (n_fft, 1, n_bins):
//   re_kernel[k, 0, n] =  hann[n] * cos(2π k n / n_fft)
//   im_kernel[k, 0, n] = -hann[n] * sin(2π k n / n_fft)
// where `n_bins = n_fft/2 + 1`. Conv1d(signal, re_kernel) yields the real part,
// Conv1d(signal, im_kernel) the imaginary part.
void codec_runtime_stft_basis_kernels(
    int32_t n_fft,
    std::vector<float> * out_re_kernel,
    std::vector<float> * out_im_kernel,
    std::vector<float> * out_hann);

// iSTFT synthesis basis matrices, window pre-multiplied. Used by the
// matmul + ConvTranspose1d-OLA in-graph iSTFT path. Layout matches
// `codec_op_linear` weights — ggml ne[0]=n_bins, ne[1]=n_fft. Mid bins carry
// the ×2 conjugate-symmetry factor (HiFi-GAN style: includes DC and Nyquist).
//   re_basis[n_bins, n] = hann[n] * coef_re(k_bin, n)
//   im_basis[n_bins, n] = hann[n] * coef_im(k_bin, n)
// where coef_re/im are the standard inverse-DFT weights with k=0 → 1, k=N/2 →
// (-1)^n, mid → 2cos / 2sin.
void codec_runtime_istft_synthesis_basis(
    int32_t n_fft,
    const std::vector<float> & hann,
    std::vector<float> * out_re,
    std::vector<float> * out_im);

// Wav2Vec2-Bert / SeamlessM4T mel-fbank feature extractor (CPU-side).
// Reproduces `SeamlessM4TFeatureExtractor.__call__` exactly: per-frame
// remove-DC, preemphasis (0.97), Povey window, |FFT|^2 spectrogram, Kaldi
// triangle mel-filter, log, then per-mel-bin (time) zero-mean unit-variance
// normalization with `ddof=1`, finally stride-2 stacking.
//
// Inputs:
//   pcm: raw waveform (mono).
//   mel_filters: (n_freq=n_fft/2+1, n_mels) mel filterbank.
//   window: (win,) Povey window (already powered to 0.85).
//   n_fft, win, hop, n_mels, mel_floor, preemphasis, stride: feature config
//   from the GGUF metadata (codec.mel.*).
//
// Output `out_features` is laid out row-major as
//   `[n_frames_after_stride, n_mels * stride]` flattened, where
//   `n_frames_after_stride = (n_frames_raw // stride)` and a stride-2 frame
//   stacks two consecutive mel frames into a single 160-d feature vector.
//   `out_n_frames` is set to `n_frames_after_stride`.
bool codec_runtime_w2v_bert_features(
    const std::vector<float> & pcm,
    const std::vector<float> & mel_filters,
    int32_t n_freq,
    int32_t n_mels,
    const std::vector<float> & window,
    int32_t n_fft,
    int32_t win,
    int32_t hop,
    float preemphasis,
    float mel_floor,
    int32_t stride,
    std::vector<float> * out_features,
    int32_t * out_n_frames,
    std::string * err);

// OLA identity kernel for ConvTranspose1d-based overlap-add. Output layout:
// ggml ne = (k=n_fft, out=1, in=n_fft) with `weight[k=i, 0, in=i] = 1` and
// zeros elsewhere. ConvTranspose1d with this kernel and stride=hop scatters
// each input frame back to a length-`n_fft` slot offset by `hop`.
void codec_runtime_ola_identity_kernel(int32_t n_fft, std::vector<float> * out);

// Slaney triangular mel filterbank, matching
// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm='slaney')`.
// Output is row-major `[n_mels, n_freq]` where `n_freq = n_fft / 2 + 1`.
// Used by the Whisper-style mel-fbank below; also exposed standalone for
// codecs that want to fold the filterbank into a graph weight.
void codec_runtime_slaney_mel_filterbank(
    int32_t sr,
    int32_t n_fft,
    int32_t n_mels,
    float fmin,
    float fmax,
    std::vector<float> * out);

// Chatterbox VoiceEncoder mel front-end.  Reproduces the upstream
// `melspectrogram + stride_as_partials` pipeline:
//   1. Optional resample (caller's responsibility — pass PCM already at
//      `ve_sample_rate = 16000`).
//   2. STFT with center=True (reflect pad n_fft/2 each side), periodic
//      Hann window, hop=160, n_fft=400 → (n_freq=201, n_frames).
//   3. Magnitude → `^2.0` (mel_power=2.0).
//   4. Project with `mel_basis @ |X|^2` → (n_mels=40, n_frames).
//   5. Transpose to `(n_frames, n_mels)`.
//   6. `stride_as_partials` with frame_step computed from
//      `(sample_rate / rate) / partial_frames` (Chatterbox default
//      rate=1.3 → frame_step=77 at 16 kHz / 160-frame partials).
//      Pads or trims the mel to `target_n` so it fits a whole number of
//      strided partials, then strides into (n_partials, partial_frames,
//      n_mels).
//
// `mel_basis` is row-major `[n_mels, n_freq]` — the layout
// librosa.filters.mel returns natively (each row is a mel-bin
// triangle weighted across frequency).  `window` is the length-`n_fft`
// periodic Hann.
//
// `out_partials` is laid out row-major `[n_partials, partial_frames, n_mels]`
// flattened (F32).  `out_n_partials` is set to `n_partials`.  Trim-silence
// is intentionally not applied — the trained checkpoints' conds.pt were
// produced with `trim_top_db=20`, so callers wanting bit-parity with that
// path supply already-trimmed PCM (the host trims with librosa or an
// equivalent energy-RMS gate; not part of the model definition proper).
bool codec_runtime_chatterbox_ve_mel_partials(
    const std::vector<float> & pcm,
    int32_t                    sample_rate,
    const std::vector<float> & mel_basis,    // [n_freq * n_mels]
    int32_t                    n_freq,
    int32_t                    n_mels,
    const std::vector<float> & window,       // [n_fft]
    int32_t                    n_fft,
    int32_t                    hop,
    int32_t                    partial_frames,
    float                      overlap,
    float                      rate,
    float                      min_coverage,
    std::vector<float> *       out_partials,
    int32_t *                  out_n_partials,
    std::string *              err);

// Qwen3-TTS speaker-encoder mel front-end (matches the upstream
// `mel_spectrogram` helper in modeling_qwen3_tts.py exactly).
//
//   * reflect-pad PCM by (n_fft - hop)/2 each side  (HF/BigVGAN style;
//     differs from librosa center=True which pads n_fft/2)
//   * frame at `hop`, Hann window, n_fft FFT → magnitude
//   * mel_basis @ |X|  (Slaney-norm librosa mel triangles, fmin=0 fmax=sr/2)
//   * log clamp: log(max(x, 1e-5)) — dynamic-range compression
//
// `mel_basis` is row-major `(n_mels, n_freq)` exactly as librosa returns
// it.  Output `out_features` is row-major `(n_mels, n_frames)` (channel-
// major — ECAPA-TDNN convs are 1D along time so channel-major is what
// the runtime feeds the first TDNN block).
bool codec_runtime_qwen3_tts_speaker_mel(
    const std::vector<float> & pcm,
    const std::vector<float> & mel_basis,
    int32_t                    n_freq,
    int32_t                    n_mels,
    const std::vector<float> & window,
    int32_t                    n_fft,
    int32_t                    hop,
    std::vector<float> *       out_features,
    int32_t *                  out_n_frames,
    std::string *              err);

// Whisper-style mel-fbank feature extractor (HF `WhisperFeatureExtractor`).
//   1. Reflection-pad PCM by n_fft/2 each side ('center=True').
//   2. Frame with stride `hop`, window with periodic Hann (length n_fft).
//   3. |FFT|^2 power spectrogram, Slaney mel filterbank.
//   4. log10(max(1e-10, x)) compression.
//   5. Clamp to (max - 8.0, max), then `(x + 4.0) / 4.0`.
// Output `out_features` is row-major `[n_mels, n_frames]` (mel-major) with
// `n_frames = pcm.size() / hop` after the input has been padded UP to a
// multiple of `pad_to_samples` (set to 1 to disable; pass the codec's
// downsample step otherwise).  `out_n_frames` is set to `n_frames`.
bool codec_runtime_whisper_mel_features(
    const std::vector<float> & pcm,
    int32_t sr,
    int32_t n_fft,
    int32_t hop,
    int32_t n_mels,
    int32_t pad_to_samples,
    std::vector<float> * out_features,
    int32_t * out_n_frames,
    std::string * err);

#endif
