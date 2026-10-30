#include "audio_frontend.h"

#include "ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct resample_ratio {
    int64_t orig = 1;
    int64_t neww = 1;
};

resample_ratio reduce_ratio(int32_t in_sr, int32_t out_sr) {
    int64_t a = in_sr, b = out_sr;
    while (b != 0) { const int64_t r = a % b; a = b; b = r; }
    return {in_sr / a, out_sr / a};
}

// Strided correlation of a [n] signal with `kernel` ([k, 1, n_out] conv
// weight) → [n_frames, n_out]. Callers run it and everything downstream of it
// up to a non-view result inside a codec_graph_cpu_scope: these front-ends need
// full F32 accumulation, which offloaded matmuls do not guarantee.
ggml_tensor * frame_matmul(ggml_context * ctx, ggml_tensor * x, ggml_tensor * kernel, int32_t stride) {
    ggml_tensor * x_tc = ggml_reshape_2d(ctx, x, ggml_nelements(x), 1);
    ggml_tensor * col = ggml_im2col(ctx, kernel, x_tc, stride, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);  // [k, frames, 1]
    const int64_t k = kernel->ne[0];
    return ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, k, col->ne[1]), ggml_reshape_2d(ctx, kernel, k, kernel->ne[2]));
}

// |X|^2 of `kernel` frames (re bins then im bins in its outputs) → [n_frames, n_bins].
ggml_tensor * frame_power(ggml_context * ctx, ggml_tensor * x, ggml_tensor * kernel, int32_t n_bins, int32_t hop) {
    ggml_tensor * y = frame_matmul(ctx, x, kernel, hop);                         // [frames, 2 * bins]
    const int64_t t = y->ne[0];
    ggml_tensor * re = ggml_view_2d(ctx, y, t, n_bins, y->nb[1], 0);
    ggml_tensor * im = ggml_view_2d(ctx, y, t, n_bins, y->nb[1], (size_t) n_bins * y->nb[1]);
    return ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im));
}

// DFT kernel of `n_fft` points over `window`-weighted frames of window.size()
// samples: re = w cos, im = -w sin.
std::vector<float> dft_kernel(const std::vector<float> & window, int32_t n_fft) {
    const int32_t frame_len = (int32_t) window.size();
    const int32_t n_bins = n_fft / 2 + 1;
    std::vector<float> k((size_t) frame_len * 2 * n_bins);
    for (int32_t b = 0; b < n_bins; ++b) {
        for (int32_t n = 0; n < frame_len; ++n) {
            // Reduce k * n mod n_fft in integers so large angles stay exact.
            const double ang = 2.0 * kPi * (double) (((int64_t) b * n) % n_fft) / (double) n_fft;
            k[(size_t) b * frame_len + n] = (float) (window[(size_t) n] * std::cos(ang));
            k[(size_t) (n_bins + b) * frame_len + n] = (float) (-window[(size_t) n] * std::sin(ang));
        }
    }
    return k;
}

// mel_fb [n_bins, n_mels] · power [n_frames, n_bins]^T → [n_mels, n_frames].
ggml_tensor * mel_project(ggml_context * ctx, ggml_tensor * mel_fb, ggml_tensor * power_tb) {
    // The filterbank runs at its stored type; power spectra of raw audio need F32
    // (range and precision), so converters store filterbanks F32.
    return ggml_mul_mat(ctx, codec_graph_stored_weight(ctx, mel_fb), ggml_cont(ctx, ggml_transpose(ctx, power_tb)));
}

}  // namespace

int64_t codec_resample_out_len(int64_t n, int32_t in_sr, int32_t out_sr) {
    if (in_sr == out_sr) return n;
    const resample_ratio r = reduce_ratio(in_sr, out_sr);
    return (r.neww * n + r.orig - 1) / r.orig;
}

ggml_tensor * codec_op_resample_sinc_hann(ggml_context * ctx, ggml_tensor * x, int32_t in_sr, int32_t out_sr) {
    if (ctx == nullptr || x == nullptr || in_sr <= 0 || out_sr <= 0) return nullptr;
    if (in_sr == out_sr) return x;
    const resample_ratio r = reduce_ratio(in_sr, out_sr);

    // torchaudio _get_sinc_resample_kernel: float64 math, float32 storage.
    const double lowpass = 6.0;
    const double base_freq = (double) std::min(r.orig, r.neww) * 0.99;
    const int64_t width = (int64_t) std::ceil(lowpass * (double) r.orig / base_freq);
    const int64_t klen = 2 * width + r.orig;
    std::vector<float> kernel((size_t) (r.neww * klen));
    for (int64_t p = 0; p < r.neww; ++p) {
        for (int64_t j = 0; j < klen; ++j) {
            const double idx = (double) (j - width) / (double) r.orig;
            double t = (-(double) p / (double) r.neww + idx) * base_freq;
            t = std::max(-lowpass, std::min(lowpass, t));
            const double c = std::cos(t * kPi / lowpass / 2.0);
            t *= kPi;
            const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
            kernel[(size_t) (p * klen + j)] = (float) (sinc * c * c * base_freq / (double) r.orig);
        }
    }
    const std::string name = "lut.resample." + std::to_string(in_sr) + "." + std::to_string(out_sr);
    ggml_tensor * w = codec_graph_const_f32(ctx, name.c_str(), kernel, klen, 1, r.neww);
    if (w == nullptr) return nullptr;

    // Zero-pad (width, width + orig), strided conv (one output channel per
    // phase), interleave the phases, trim to ceil(n * neww / orig).
    const int64_t n = ggml_nelements(x);
    ggml_tensor * xp = codec_op_pad_1d(ctx, ggml_reshape_2d(ctx, x, n, 1), (int32_t) width, (int32_t) (width + r.orig));
    codec_graph_cpu_scope cpu(ctx);
    ggml_tensor * y = frame_matmul(ctx, xp, w, (int32_t) r.orig);                 // [frames, neww]
    y = ggml_cont(ctx, ggml_transpose(ctx, y));                                      // [neww, frames]
    y = ggml_reshape_1d(ctx, y, ggml_nelements(y));
    return ggml_cont(ctx, ggml_view_1d(ctx, y, codec_resample_out_len(n, in_sr, out_sr), 0));
}

int32_t codec_whisper_log_mel_frames(int64_t n, int32_t hop, int32_t pad_multiple) {
    const int64_t padded = (n + pad_multiple - 1) / pad_multiple * pad_multiple;
    return (int32_t) (padded / hop);
}

ggml_tensor * codec_op_whisper_log_mel(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, const std::vector<float> & window,
        int32_t n_fft, int32_t hop, int32_t pad_multiple, const char * lut_name) {
    if (ctx == nullptr || x == nullptr || mel_fb == nullptr || (int32_t) window.size() != n_fft || lut_name == nullptr) {
        return nullptr;
    }
    const int32_t n_bins = n_fft / 2 + 1;
    const int64_t n = ggml_nelements(x);
    const int32_t n_frames = codec_whisper_log_mel_frames(n, hop, pad_multiple);
    const int64_t padded = (int64_t) n_frames * hop;

    ggml_tensor * k = codec_graph_const_f32(ctx, lut_name, dft_kernel(window, n_fft), n_fft, 1, 2 * n_bins);
    if (k == nullptr) return nullptr;
    ggml_tensor * xp = ggml_pad(ctx, ggml_reshape_2d(ctx, x, n, 1), (int) (padded - n), 0, 0, 0);
    xp = ggml_pad_reflect_1d(ctx, xp, n_fft / 2, n_fft / 2);
    codec_graph_cpu_scope cpu(ctx);
    ggml_tensor * p = frame_power(ctx, xp, k, n_bins, hop);                          // [n_frames + 1, bins]
    p = ggml_view_2d(ctx, p, n_frames, n_bins, p->nb[1], 0);                         // stft[..., :-1]

    ggml_tensor * mel = mel_project(ctx, mel_fb, p);                                 // [mels, frames]
    mel = ggml_scale(ctx, ggml_log(ctx, codec_op_clamp_min(ctx, mel, 1e-10f)), (float) (1.0 / std::log(10.0)));

    // Floor at (global max - 8): argmax over the flattened spectrogram.
    const int64_t total = ggml_nelements(mel);
    ggml_tensor * idx = ggml_argmax(ctx, ggml_reshape_2d(ctx, mel, total, 1));
    ggml_tensor * mx = ggml_get_rows(ctx, ggml_reshape_2d(ctx, mel, 1, total), idx);  // [1, 1]
    ggml_tensor * floor_v = ggml_scale_bias(ctx, mx, 1.0f, -8.0f);
    mel = ggml_add(ctx, ggml_relu(ctx, ggml_sub(ctx, mel, floor_v)), floor_v);       // max(x, floor)
    mel = ggml_scale_bias(ctx, mel, 0.25f, 1.0f);
    return ggml_cont(ctx, ggml_transpose(ctx, mel));                                 // [frames, mels]
}

int32_t codec_mel_spectrogram_frames(int64_t n, int32_t n_fft, int32_t hop, int32_t pad) {
    return (int32_t) ((n + 2 * (int64_t) pad - n_fft) / hop + 1);
}

ggml_tensor * codec_op_mel_spectrogram(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, const std::vector<float> & window,
        int32_t n_fft, int32_t hop, int32_t pad, bool magnitude, float mag_eps, const char * lut_name) {
    const int64_t n = ctx != nullptr && x != nullptr ? ggml_nelements(x) : 0;
    if (mel_fb == nullptr || lut_name == nullptr || n_fft <= 0 || hop <= 0 || pad < 0 || pad >= n ||
        (int32_t) window.size() != n_fft || codec_mel_spectrogram_frames(n, n_fft, hop, pad) <= 0) {
        return nullptr;
    }
    const int32_t n_bins = n_fft / 2 + 1;
    ggml_tensor * k = codec_graph_const_f32(ctx, lut_name, dft_kernel(window, n_fft), n_fft, 1, 2 * n_bins);
    if (k == nullptr) return nullptr;
    ggml_tensor * xp = ggml_reshape_2d(ctx, x, n, 1);
    if (pad > 0) xp = ggml_pad_reflect_1d(ctx, xp, pad, pad);
    codec_graph_cpu_scope cpu(ctx);
    ggml_tensor * p = frame_power(ctx, xp, k, n_bins, hop);                          // [frames, bins]
    if (magnitude) p = ggml_sqrt(ctx, ggml_scale_bias(ctx, p, 1.0f, mag_eps));
    return mel_project(ctx, mel_fb, p);                                              // [mels, frames]
}

int32_t codec_matcha_log_mel_frames(int64_t n, int32_t n_fft, int32_t hop) {
    return codec_mel_spectrogram_frames(n, n_fft, hop, (n_fft - hop) / 2);
}

ggml_tensor * codec_op_matcha_log_mel(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb, int32_t n_fft, int32_t hop) {
    if (n_fft <= 0) return nullptr;
    std::vector<float> window((size_t) n_fft);
    for (int32_t i = 0; i < n_fft; ++i) {
        window[(size_t) i] = (float) (0.5 - 0.5 * std::cos(2.0 * kPi * (double) i / (double) n_fft));
    }
    const std::string name = "lut.matcha_stft." + std::to_string(n_fft);
    ggml_tensor * mel = codec_op_mel_spectrogram(ctx, x, mel_fb, window, n_fft, hop, (n_fft - hop) / 2,
                                                 /*magnitude=*/true, 1e-9f, name.c_str());
    return mel == nullptr ? nullptr : ggml_log(ctx, codec_op_clamp_min(ctx, mel, 1e-5f));
}

int32_t codec_kaldi_fbank_frames(int64_t n) {
    return n < 400 ? 0 : (int32_t) (1 + (n - 400) / 160);
}

ggml_tensor * codec_op_kaldi_fbank_cmn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * mel_fb) {
    constexpr int32_t kFrameLen = 400;
    constexpr int32_t kShift = 160;
    constexpr int32_t kFft = 512;
    constexpr int32_t kBins = kFft / 2 + 1;
    constexpr double kPreemph = 0.97;
    if (ctx == nullptr || x == nullptr || mel_fb == nullptr || codec_kaldi_fbank_frames(ggml_nelements(x)) <= 0) {
        return nullptr;
    }

    // Per frame, DC removal, preemphasis, the Povey window and the DFT are all
    // linear, so they fold into one [400 → 2 * 257] kernel: K = D · W · P · C.
    std::vector<double> win((size_t) kFrameLen);
    for (int32_t i = 0; i < kFrameLen; ++i) {
        win[(size_t) i] = std::pow(0.5 - 0.5 * std::cos(2.0 * kPi * (double) i / (double) (kFrameLen - 1)), 0.85);
    }
    std::vector<float> kernel((size_t) kFrameLen * 2 * kBins);
    std::vector<double> a((size_t) kFrameLen), b((size_t) kFrameLen);
    for (int32_t part = 0; part < 2; ++part) {
        for (int32_t bin = 0; bin < kBins; ++bin) {
            // a = row of D · W (window then DFT over the 400 frame samples).
            for (int32_t m = 0; m < kFrameLen; ++m) {
                const double ang = 2.0 * kPi * (double) (((int64_t) bin * m) % kFft) / (double) kFft;
                a[(size_t) m] = win[(size_t) m] * (part == 0 ? std::cos(ang) : -std::sin(ang));
            }
            // · P: z[m] = y[m] - 0.97 y[m - 1], with y[-1] = y[0].
            for (int32_t m = 0; m < kFrameLen; ++m) b[(size_t) m] = a[(size_t) m];
            for (int32_t m = 1; m < kFrameLen; ++m) b[(size_t) m - 1] -= kPreemph * a[(size_t) m];
            b[0] -= kPreemph * a[0];
            // · C: remove the frame mean.
            double mean = 0.0;
            for (double v : b) mean += v;
            mean /= (double) kFrameLen;
            float * dst = kernel.data() + (size_t) (part * kBins + bin) * kFrameLen;
            for (int32_t m = 0; m < kFrameLen; ++m) dst[m] = (float) (b[(size_t) m] - mean);
        }
    }
    ggml_tensor * k = codec_graph_const_f32(ctx, "lut.kaldi_fbank", kernel, kFrameLen, 1, 2 * kBins);
    if (k == nullptr) return nullptr;
    codec_graph_cpu_scope cpu(ctx);
    ggml_tensor * p = frame_power(ctx, x, k, kBins, kShift);                         // [frames, 257]
    ggml_tensor * mel = mel_project(ctx, mel_fb, p);                                 // [mels, frames]
    mel = ggml_log(ctx, codec_op_clamp_min(ctx, mel, FLT_EPSILON));
    mel = ggml_cont(ctx, ggml_transpose(ctx, mel));                                  // [frames, mels]
    return ggml_sub(ctx, mel, ggml_mean(ctx, mel));
}
