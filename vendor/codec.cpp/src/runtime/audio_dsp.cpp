#include "audio_dsp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

#define POCKETFFT_NO_MULTITHREADING
#include "../third_party/pocketfft/pocketfft_hdronly.h"

bool codec_runtime_istft_from_head(
    const std::vector<float> & head,
    int32_t out_dim,
    int32_t n_frames,
    int32_t hop,
    const std::vector<float> * window,
    bool skip_dc_nyquist,
    int32_t trim_pad_override,
    std::vector<float> * out_pcm,
    std::string * err) {

    if (out_pcm == nullptr || out_dim <= 0 || n_frames <= 0 || hop <= 0 || (out_dim % 2) != 0) {
        if (err != nullptr) {
            *err = "invalid ISTFT arguments";
        }
        return false;
    }
    const int32_t n_bins = out_dim / 2;
    const int32_t n_fft = 2 * (n_bins - 1);
    if (n_fft <= 0) {
        if (err != nullptr) {
            *err = "invalid ISTFT head output dimension";
        }
        return false;
    }
    const float pi = 3.14159265358979323846f;

    std::vector<float> win((size_t) n_fft, 0.0f);
    if (window != nullptr && (int32_t) window->size() == n_fft) {
        win = *window;
    } else {
        for (int32_t n = 0; n < n_fft; ++n) {
            win[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * pi * (float) n / (float) (n_fft - 1));
        }
    }

    const int32_t pad = trim_pad_override >= 0
        ? trim_pad_override
        : (skip_dc_nyquist ? (n_fft / 2) : ((n_fft - hop) / 2));
    const int32_t out_size = (n_frames - 1) * hop + n_fft;
    std::vector<float> y((size_t) out_size, 0.0f);
    std::vector<float> env((size_t) out_size, 0.0f);
    // Per frame: inverse real FFT of the (mag, phase) spectrum, then window and
    // overlap-add. The plan (twiddle factors) is built once per call.
    try {
        const pocketfft::detail::pocketfft_r<float> plan((size_t) n_fft);
        std::vector<float> frame((size_t) n_fft);
        for (int32_t ti = 0; ti < n_frames; ++ti) {
            const float * h = head.data() + (size_t) out_dim * (size_t) ti;
            // FFTPACK halfcomplex layout: [r0, r1, i1, ..., r_{N/2-1}, i_{N/2-1}, r_{N/2}].
            // DC / Nyquist imaginary parts are dropped (real output).
            for (int32_t k = 0; k < n_bins; ++k) {
                float mag = std::exp(h[k]);
                if (mag > 1e2f) mag = 1e2f;
                const float re = mag * std::cos(h[n_bins + k]);
                if (k == 0) {
                    frame[0] = skip_dc_nyquist ? 0.0f : re;
                } else if (k == n_bins - 1) {
                    frame[(size_t) (n_fft - 1)] = skip_dc_nyquist ? 0.0f : re;
                } else {
                    frame[(size_t) (2 * k - 1)] = re;
                    frame[(size_t) (2 * k)] = mag * std::sin(h[n_bins + k]);
                }
            }
            plan.exec(frame.data(), 1.0f / (float) n_fft, /*fwd=*/false);
            const int32_t off = ti * hop;
            for (int32_t n = 0; n < n_fft; ++n) {
                y[(size_t) (off + n)] += frame[(size_t) n] * win[(size_t) n];
                env[(size_t) (off + n)] += win[(size_t) n] * win[(size_t) n];
            }
        }
    } catch (const std::exception & e) {
        if (err != nullptr) {
            *err = std::string("ISTFT failed: ") + e.what();
        }
        return false;
    }

    const int32_t out_begin = std::max(0, pad);
    const int32_t out_end = std::max(out_begin, out_size - pad);
    out_pcm->assign((size_t) (out_end - out_begin), 0.0f);
    for (int32_t i = out_begin; i < out_end; ++i) {
        const float den = env[(size_t) i] > 1e-11f ? env[(size_t) i] : 1.0f;
        (*out_pcm)[(size_t) (i - out_begin)] = y[(size_t) i] / den;
    }
    return true;
}


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
    std::string * err) {

    if (out_features == nullptr || out_n_frames == nullptr) {
        if (err != nullptr) *err = "null output";
        return false;
    }
    if (n_fft <= 0 || win <= 0 || hop <= 0 || n_mels <= 0 || stride <= 0) {
        if (err != nullptr) *err = "invalid mel-fbank arguments";
        return false;
    }
    if ((int32_t) window.size() != win) {
        if (err != nullptr) *err = "window size mismatch";
        return false;
    }
    if (n_freq != n_fft / 2 + 1 ||
        (int32_t) mel_filters.size() != n_freq * n_mels) {
        if (err != nullptr) *err = "mel filter shape mismatch";
        return false;
    }
    const int64_t n = (int64_t) pcm.size();
    if (n < win) {
        if (err != nullptr) *err = "input shorter than win";
        return false;
    }

    const int32_t n_frames = (int32_t) ((n - win) / hop + 1);
    if (n_frames <= 0) {
        if (err != nullptr) *err = "no frames";
        return false;
    }

    // Compute log-mel features per frame.  Matches transformers' reference
    // exactly (kaldi-compliance scale 2^15, per-frame DC remove, preemphasis,
    // window, FFT, |X|^2 mel, log(max(., mel_floor))).
    const float pi = 3.14159265358979323846f;
    std::vector<float> log_mel((size_t) n_frames * (size_t) n_mels, 0.0f);
    std::vector<double> buffer((size_t) n_fft, 0.0);
    std::vector<double> re_v((size_t) n_freq, 0.0);
    std::vector<double> im_v((size_t) n_freq, 0.0);

    // Precompute DFT basis (real + imag) at double precision for parity.
    std::vector<double> dft_cos((size_t) n_freq * (size_t) n_fft, 0.0);
    std::vector<double> dft_sin((size_t) n_freq * (size_t) n_fft, 0.0);
    for (int32_t k = 0; k < n_freq; ++k) {
        for (int32_t m = 0; m < n_fft; ++m) {
            const double ang = -2.0 * pi * (double) k * (double) m / (double) n_fft;
            dft_cos[(size_t) k * (size_t) n_fft + (size_t) m] = std::cos(ang);
            dft_sin[(size_t) k * (size_t) n_fft + (size_t) m] = std::sin(ang);
        }
    }

    for (int32_t ti = 0; ti < n_frames; ++ti) {
        // 1. Extract frame (Kaldi-compliance: scale by 2^15) into a double buffer.
        const int64_t off = (int64_t) ti * hop;
        std::fill(buffer.begin(), buffer.end(), 0.0);
        for (int32_t k = 0; k < win; ++k) {
            buffer[(size_t) k] = (double) pcm[(size_t) (off + k)] * 32768.0;
        }
        // 2. Remove DC offset (subtract mean over the win samples).
        double mean = 0.0;
        for (int32_t k = 0; k < win; ++k) mean += buffer[(size_t) k];
        mean /= (double) win;
        for (int32_t k = 0; k < win; ++k) buffer[(size_t) k] -= mean;
        // 3. Pre-emphasis applied IN-FRAME (note: must go from k=win-1 down to k=1
        //    to avoid clobbering buffer[k-1] before it's used).
        for (int32_t k = win - 1; k >= 1; --k) {
            buffer[(size_t) k] -= (double) preemphasis * buffer[(size_t) (k - 1)];
        }
        buffer[0] *= (double) (1.0f - preemphasis);
        // 4. Window.
        for (int32_t k = 0; k < win; ++k) buffer[(size_t) k] *= (double) window[(size_t) k];

        // 5. DFT (zero-padded to n_fft, but win <= n_fft so trailing slots are 0).
        for (int32_t k = 0; k < n_freq; ++k) {
            double re = 0.0, im = 0.0;
            const double * cos_row = &dft_cos[(size_t) k * (size_t) n_fft];
            const double * sin_row = &dft_sin[(size_t) k * (size_t) n_fft];
            for (int32_t m = 0; m < n_fft; ++m) {
                re += buffer[(size_t) m] * cos_row[(size_t) m];
                im += buffer[(size_t) m] * sin_row[(size_t) m];
            }
            re_v[(size_t) k] = re;
            im_v[(size_t) k] = im;
        }
        // 6. Power spectrogram |X|^2 then mel matmul.
        for (int32_t mi = 0; mi < n_mels; ++mi) {
            double acc = 0.0;
            for (int32_t k = 0; k < n_freq; ++k) {
                const double power = re_v[(size_t) k] * re_v[(size_t) k] +
                                     im_v[(size_t) k] * im_v[(size_t) k];
                acc += power * (double) mel_filters[(size_t) k * (size_t) n_mels + (size_t) mi];
            }
            // 7. log(max(., mel_floor)).
            if (acc < (double) mel_floor) acc = (double) mel_floor;
            log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] = (float) std::log(acc);
        }
    }

    // 8. Per-mel-bin (time) zero-mean unit-variance normalize.  ddof=1 sample
    //    variance to match torch/numpy reference.
    if (n_frames > 1) {
        for (int32_t mi = 0; mi < n_mels; ++mi) {
            double sum = 0.0;
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                sum += (double) log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi];
            }
            const double m = sum / (double) n_frames;
            double var = 0.0;
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                const double d = (double) log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] - m;
                var += d * d;
            }
            var /= (double) (n_frames - 1);  // ddof=1
            const double s = 1.0 / std::sqrt(var + 1e-7);
            for (int32_t ti = 0; ti < n_frames; ++ti) {
                const float x = log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi];
                log_mel[(size_t) ti * (size_t) n_mels + (size_t) mi] = (float) (((double) x - m) * s);
            }
        }
    }

    // 9. Stride-2 stacking: drop trailing remainder, reshape (T, n_mels) →
    //    (T/stride, n_mels * stride).  Memory layout is identical (contiguous
    //    row-major), so we just truncate and reinterpret the buffer.
    const int32_t remainder = n_frames % stride;
    const int32_t n_frames_kept = n_frames - remainder;
    const int32_t n_frames_out = n_frames_kept / stride;
    const int32_t out_dim = n_mels * stride;

    out_features->assign((size_t) n_frames_out * (size_t) out_dim, 0.0f);
    for (int32_t ti = 0; ti < n_frames_out; ++ti) {
        for (int32_t s = 0; s < stride; ++s) {
            for (int32_t mi = 0; mi < n_mels; ++mi) {
                (*out_features)[(size_t) ti * (size_t) out_dim + (size_t) (s * n_mels + mi)] =
                    log_mel[(size_t) (ti * stride + s) * (size_t) n_mels + (size_t) mi];
            }
        }
    }
    *out_n_frames = n_frames_out;
    (void) err;
    return true;
}

void codec_runtime_periodic_hann_window(int32_t n_fft, std::vector<float> * out) {
    if (out == nullptr || n_fft <= 0) return;
    out->assign((size_t) n_fft, 0.0f);
    for (int32_t n = 0; n < n_fft; ++n) {
        (*out)[(size_t) n] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * (float) n / (float) n_fft);
    }
}

void codec_runtime_stft_basis_kernels(
    int32_t n_fft,
    std::vector<float> * out_re_kernel,
    std::vector<float> * out_im_kernel,
    std::vector<float> * out_hann) {
    if (out_re_kernel == nullptr || out_im_kernel == nullptr || n_fft <= 0) return;
    const int32_t n_bins = n_fft / 2 + 1;
    std::vector<float> hann;
    codec_runtime_periodic_hann_window(n_fft, &hann);
    if (out_hann != nullptr) *out_hann = hann;

    out_re_kernel->assign((size_t) n_fft * (size_t) n_bins, 0.0f);
    out_im_kernel->assign((size_t) n_fft * (size_t) n_bins, 0.0f);
    // ggml conv1d weight ne[0]=k, ne[1]=in=1, ne[2]=out=n_bins → flat index = k + out*K.
    for (int32_t k_bin = 0; k_bin < n_bins; ++k_bin) {
        for (int32_t n = 0; n < n_fft; ++n) {
            const float ang = 2.0f * (float) M_PI * (float) k_bin * (float) n / (float) n_fft;
            const float w = hann[(size_t) n];
            const size_t off = (size_t) n + (size_t) k_bin * (size_t) n_fft;
            (*out_re_kernel)[off] =  w * std::cos(ang);
            (*out_im_kernel)[off] = -w * std::sin(ang);
        }
    }
}

void codec_runtime_istft_synthesis_basis(
    int32_t n_fft,
    const std::vector<float> & hann,
    std::vector<float> * out_re,
    std::vector<float> * out_im) {
    if (out_re == nullptr || out_im == nullptr || n_fft <= 0 || (int32_t) hann.size() != n_fft) return;
    const int32_t n_bins = n_fft / 2 + 1;
    out_re->assign((size_t) n_bins * (size_t) n_fft, 0.0f);
    out_im->assign((size_t) n_bins * (size_t) n_fft, 0.0f);
    // ggml ne[0]=n_bins, ne[1]=n_fft → flat index = k_bin + n*n_bins.
    for (int32_t n = 0; n < n_fft; ++n) {
        for (int32_t k_bin = 0; k_bin < n_bins; ++k_bin) {
            const float ang = 2.0f * (float) M_PI * (float) k_bin * (float) n / (float) n_fft;
            float coef_re = 0.0f, coef_im = 0.0f;
            if (k_bin == 0) {
                coef_re = 1.0f;
            } else if (k_bin == n_bins - 1) {
                coef_re = (n & 1) ? -1.0f : 1.0f;
            } else {
                coef_re = 2.0f * std::cos(ang);
                coef_im = 2.0f * std::sin(ang);
            }
            const size_t off = (size_t) k_bin + (size_t) n * (size_t) n_bins;
            (*out_re)[off] = coef_re * hann[(size_t) n];
            (*out_im)[off] = coef_im * hann[(size_t) n];
        }
    }
}

void codec_runtime_ola_identity_kernel(int32_t n_fft, std::vector<float> * out) {
    if (out == nullptr || n_fft <= 0) return;
    // ggml convtr1d weight ne = (k, out=1, in) → flat index = k + out*K + in*K (out=1).
    out->assign((size_t) n_fft * (size_t) n_fft, 0.0f);
    for (int32_t i = 0; i < n_fft; ++i) {
        (*out)[(size_t) i + (size_t) i * (size_t) n_fft] = 1.0f;
    }
}

void codec_runtime_slaney_mel_filterbank(
    int32_t sr,
    int32_t n_fft,
    int32_t n_mels,
    float fmin,
    float fmax,
    std::vector<float> * out) {
    if (out == nullptr || sr <= 0 || n_fft <= 0 || n_mels <= 0) return;
    const int32_t n_freq = n_fft / 2 + 1;
    out->assign((size_t) n_mels * (size_t) n_freq, 0.0f);

    auto hz_to_mel = [](float hz) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;
        if (hz >= min_log_hz) {
            return min_log_mel + std::log(hz / min_log_hz) / logstep;
        }
        return hz / f_sp;
    };
    auto mel_to_hz = [](float mel) -> float {
        const float f_sp = 200.0f / 3.0f;
        const float min_log_hz = 1000.0f;
        const float min_log_mel = min_log_hz / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;
        if (mel >= min_log_mel) {
            return min_log_hz * std::exp(logstep * (mel - min_log_mel));
        }
        return f_sp * mel;
    };

    const float mmin = hz_to_mel(fmin);
    const float mmax = hz_to_mel(fmax);
    std::vector<float> bin_freqs((size_t) n_mels + 2);
    for (int32_t i = 0; i < n_mels + 2; ++i) {
        const float m = mmin + (mmax - mmin) * (float) i / (float) (n_mels + 1);
        bin_freqs[(size_t) i] = mel_to_hz(m);
    }
    std::vector<float> fft_freqs((size_t) n_freq);
    for (int32_t k = 0; k < n_freq; ++k) {
        fft_freqs[(size_t) k] = (float) sr * (float) k / (float) n_fft;
    }
    for (int32_t m = 0; m < n_mels; ++m) {
        const float left   = bin_freqs[(size_t) m];
        const float center = bin_freqs[(size_t) m + 1];
        const float right  = bin_freqs[(size_t) m + 2];
        const float enorm  = 2.0f / (right - left);
        for (int32_t k = 0; k < n_freq; ++k) {
            const float f = fft_freqs[(size_t) k];
            float w = 0.0f;
            if (f >= left && f < center) {
                w = (f - left) / (center - left);
            } else if (f >= center && f <= right) {
                w = (right - f) / (right - center);
            }
            (*out)[(size_t) m * (size_t) n_freq + (size_t) k] = w * enorm;
        }
    }
}

// O(N²) real-input DFT (small n_fft → tolerable for one-shot FE).
static void codec_runtime_rfft_naive(const float * in, int32_t n,
                                      std::vector<float> * out_re,
                                      std::vector<float> * out_im) {
    const int32_t n_freq = n / 2 + 1;
    out_re->assign((size_t) n_freq, 0.0f);
    out_im->assign((size_t) n_freq, 0.0f);
    for (int32_t k = 0; k < n_freq; ++k) {
        double sum_re = 0.0, sum_im = 0.0;
        const double w = -2.0 * M_PI * (double) k / (double) n;
        for (int32_t t = 0; t < n; ++t) {
            const double a = w * (double) t;
            sum_re += (double) in[t] * std::cos(a);
            sum_im += (double) in[t] * std::sin(a);
        }
        (*out_re)[(size_t) k] = (float) sum_re;
        (*out_im)[(size_t) k] = (float) sum_im;
    }
}

bool codec_runtime_whisper_mel_features(
    const std::vector<float> & pcm,
    int32_t sr,
    int32_t n_fft,
    int32_t hop,
    int32_t n_mels,
    int32_t pad_to_samples,
    std::vector<float> * out_features,
    int32_t * out_n_frames,
    std::string * err) {
    if (out_features == nullptr || out_n_frames == nullptr) {
        if (err) *err = "null output";
        return false;
    }
    if (sr <= 0 || n_fft <= 0 || hop <= 0 || n_mels <= 0) {
        if (err) *err = "invalid mel-fbank arguments";
        return false;
    }
    const int32_t pad_to = std::max(1, pad_to_samples);
    const int32_t in_n   = (int32_t) pcm.size();
    const int32_t target_len = ((in_n + pad_to - 1) / pad_to) * pad_to;
    std::vector<float> x = pcm;
    if ((int32_t) x.size() < target_len) {
        x.resize((size_t) target_len, 0.0f);
    }

    // Reflection pad (PyTorch/numpy 'reflect' mode: edge value not repeated).
    const int32_t pad = n_fft / 2;
    std::vector<float> xp((size_t) target_len + (size_t) 2 * pad, 0.0f);
    for (int32_t i = 0; i < target_len; ++i) {
        xp[(size_t) (i + pad)] = x[(size_t) i];
    }
    for (int32_t i = 0; i < pad; ++i) {
        const int32_t l = pad - i;
        if (l < target_len) xp[(size_t) i] = x[(size_t) l];
        const int32_t r = target_len - 2 - i;
        if (r >= 0) xp[(size_t) (target_len + pad + i)] = x[(size_t) r];
    }

    std::vector<float> hann;
    codec_runtime_periodic_hann_window(n_fft, &hann);

    std::vector<float> mel_fb;
    codec_runtime_slaney_mel_filterbank(sr, n_fft, n_mels, 0.0f, (float) sr / 2.0f, &mel_fb);
    const int32_t n_freq = n_fft / 2 + 1;

    const int32_t n_frames = target_len / hop;
    out_features->assign((size_t) n_mels * (size_t) n_frames, 0.0f);

    std::vector<float> frame((size_t) n_fft, 0.0f);
    std::vector<float> spec_re, spec_im;
    std::vector<float> mel_frame((size_t) n_mels, 0.0f);
    float global_max = -INFINITY;
    for (int32_t f = 0; f < n_frames; ++f) {
        const int32_t start = f * hop;
        for (int32_t i = 0; i < n_fft; ++i) {
            frame[(size_t) i] = xp[(size_t) (start + i)] * hann[(size_t) i];
        }
        codec_runtime_rfft_naive(frame.data(), n_fft, &spec_re, &spec_im);
        for (int32_t m = 0; m < n_mels; ++m) {
            double s = 0.0;
            const float * fb = &mel_fb[(size_t) m * (size_t) n_freq];
            for (int32_t k = 0; k < n_freq; ++k) {
                const float re = spec_re[(size_t) k];
                const float im = spec_im[(size_t) k];
                s += (double) fb[k] * (double) (re * re + im * im);
            }
            float v = std::log10(std::max(1e-10, s));
            mel_frame[(size_t) m] = v;
            if (v > global_max) global_max = v;
        }
        for (int32_t m = 0; m < n_mels; ++m) {
            (*out_features)[(size_t) m * (size_t) n_frames + (size_t) f] = mel_frame[(size_t) m];
        }
    }
    const float lo = global_max - 8.0f;
    for (size_t i = 0; i < out_features->size(); ++i) {
        float v = (*out_features)[i];
        if (v < lo) v = lo;
        (*out_features)[i] = (v + 4.0f) / 4.0f;
    }
    *out_n_frames = n_frames;
    (void) err;
    return true;
}

