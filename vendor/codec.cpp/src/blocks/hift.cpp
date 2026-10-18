#include "hift.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/ggml_ops.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace {

constexpr int32_t kF0NumLayers = 5;
constexpr float kLreluSlopeDefault = 0.01f;  // F.leaky_relu default before conv_post

std::string join(const char * prefix, const std::string & name) {
    return std::string(prefix) + "." + name;
}

// Conv1d over x [T, c] with explicit zero padding (left, right) on time.
ggml_tensor * conv_padded(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b,
                          int32_t left, int32_t right, int32_t stride = 1, int32_t dilation = 1) {
    if (x == nullptr || w == nullptr) return nullptr;
    if (left > 0 || right > 0) x = codec_op_pad_1d(ctx, x, left, right);
    return codec_conv1d(ctx, x, w, b, stride, dilation, 0);
}

// (Causal)ConvRNNF0Predictor: 5 × (Conv1d k3 → ELU) → Linear → |·|. mel [T, 80] → f0 [T].
// Causal: the first conv (k4) looks right, the rest left.
ggml_tensor * hift_f0(ggml_context * ctx, const codec_model * model, const char * prefix,
                      const codec_hift_params & p, ggml_tensor * mel_tc) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, join(prefix, n)); };
    ggml_tensor * x = mel_tc;
    for (int32_t i = 0; i < kF0NumLayers; ++i) {
        ggml_tensor * w = W("f0.cn." + std::to_string(i) + ".w");
        ggml_tensor * b = W("f0.cn." + std::to_string(i) + ".b");
        if (w == nullptr || b == nullptr) return nullptr;
        const int32_t k = (int32_t) codec_conv1d_kernel_size(w, x->ne[1]);
        if (p.causal) {
            x = conv_padded(ctx, x, w, b, i == 0 ? 0 : k - 1, i == 0 ? k - 1 : 0);
        } else {
            x = codec_conv1d(ctx, x, w, b, /*stride=*/1, /*dilation=*/1, /*padding=*/(k - 1) / 2);
        }
        if (x == nullptr) return nullptr;
        x = codec_op_unary(ctx, x, CODEC_UNARY_ELU);
    }
    ggml_tensor * cls_w = W("f0.cls.w");
    ggml_tensor * cls_b = W("f0.cls.b");
    if (cls_w == nullptr || cls_b == nullptr) return nullptr;
    ggml_tensor * x_ct = ggml_cont(ctx, ggml_transpose(ctx, x));
    ggml_tensor * f0 = ggml_add(ctx, codec_mul_mat(ctx, cls_w, x_ct), cls_b);
    return ggml_reshape_1d(ctx, ggml_abs(ctx, f0), x->ne[0]);
}

// HiFi-GAN ResBlock: one Snake/conv/Snake/conv branch per dilation
// (left-padded CausalConv1d for the causal generator).
ggml_tensor * hift_resblock(ggml_context * ctx, const codec_model * model, const std::string & prefix,
                            const codec_hift_params & p, ggml_tensor * x, int32_t kernel) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, prefix + "." + n); };
    for (size_t idx = 0; idx < p.resblock_dilations.size(); ++idx) {
        const std::string s = std::to_string(idx);
        const int32_t d = p.resblock_dilations[idx];
        if (!p.causal) {
            x = codec_op_hifigan_resblock_branch_ct(ctx, x, W("a1." + s), W("a2." + s),
                W("cv1." + s + ".w"), W("cv1." + s + ".b"), W("cv2." + s + ".w"), W("cv2." + s + ".b"),
                kernel, d);
        } else {
            ggml_tensor * h = codec_op_snake(ctx, x, W("a1." + s), 1e-9f);
            h = conv_padded(ctx, h, W("cv1." + s + ".w"), W("cv1." + s + ".b"), (kernel - 1) * d, 0, 1, d);
            if (h == nullptr) return nullptr;
            h = codec_op_snake(ctx, h, W("a2." + s), 1e-9f);
            h = conv_padded(ctx, h, W("cv2." + s + ".w"), W("cv2." + s + ".b"), kernel - 1, 0);
            x = h == nullptr ? nullptr : ggml_add(ctx, h, x);
        }
        if (x == nullptr) return nullptr;
    }
    return x;
}

// F.interpolate(x, scale_factor=up, mode='linear') along time of x [T, c]
// (half-pixel, edge-clamped).  Sample i*up + r reads x[i] plus a(r) of the
// forward difference (r >= up/2) or b(r) of the backward one (r < up/2),
// built from broadcast ops every backend runs (Metal UPSCALE is nearest-only).
ggml_tensor * upsample_linear_half_pixel(ggml_context * ctx, ggml_tensor * x, int32_t up, const char * lut) {
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    std::vector<float> w((size_t) 2 * up);
    for (int32_t r = 0; r < up; ++r) {
        const float f = ((float) r + 0.5f) / (float) up - 0.5f;
        w[(size_t) r] = f > 0.0f ? f : 0.0f;          // forward weight a(r)
        w[(size_t) (up + r)] = f < 0.0f ? -f : 0.0f;  // backward weight -b(r)
    }
    ggml_tensor * wt = codec_graph_const_f32(ctx, lut, w, up, 1, 2);
    ggml_tensor * wa = ggml_view_3d(ctx, wt, up, 1, 1, wt->nb[1], wt->nb[2], 0);
    ggml_tensor * wb = ggml_view_3d(ctx, wt, up, 1, 1, wt->nb[1], wt->nb[2], wt->nb[2]);
    ggml_tensor * d = T > 1 ? ggml_sub(ctx, ggml_view_2d(ctx, x, T - 1, C, x->nb[1], x->nb[0]),
                                       ggml_view_2d(ctx, x, T - 1, C, x->nb[1], 0))
                            : nullptr;
    ggml_tensor * fwd = d != nullptr ? codec_op_pad_1d(ctx, ggml_cont(ctx, d), 0, 1) : ggml_scale(ctx, x, 0.0f);
    ggml_tensor * bwd = d != nullptr ? codec_op_pad_1d(ctx, ggml_cont(ctx, d), 1, 0) : ggml_scale(ctx, x, 0.0f);
    ggml_tensor * tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, up, T, C);
    auto rep = [&](ggml_tensor * t) { return ggml_repeat(ctx, ggml_reshape_3d(ctx, t, 1, T, C), tmpl); };
    ggml_tensor * y = ggml_add(ctx, rep(x), ggml_mul(ctx, rep(fwd), wa));
    y = ggml_sub(ctx, y, ggml_mul(ctx, rep(bwd), wb));
    return ggml_reshape_2d(ctx, y, up * T, C);
}

// x - floor(x) for 0 <= x < 4 (STEP-based; Metal has no FLOOR).
ggml_tensor * frac_bounded(ggml_context * ctx, ggml_tensor * x) {
    ggml_tensor * fl = nullptr;
    for (int32_t n = 1; n < 4; ++n) {
        ggml_tensor * s = ggml_step(ctx, ggml_scale_bias(ctx, x, 1.0f, -(float) n + 0.0f));
        fl = fl == nullptr ? s : ggml_add(ctx, fl, s);
    }
    return ggml_sub(ctx, x, fl);
}

// NSF harmonic source (SourceModuleHnNSF): f0 [T] → sine_merge [T_pcm].
ggml_tensor * hift_source(ggml_context * ctx, const codec_model * model, const char * prefix,
                          const codec_hift_params & p, ggml_tensor * f0,
                          ggml_tensor * nsf_phase, ggml_tensor * nsf_noise) {
    const int32_t T = (int32_t) f0->ne[0];
    const int32_t up = codec_hift_upsample_factor(p);
    const int32_t T_pcm = T * up;
    const int32_t n_harm = p.nb_harmonics + 1;

    // f0_upsamp (nearest) → per-sample f0 [T_pcm, 1].
    ggml_tensor * f0_pcm = ggml_reshape_2d(ctx,
        ggml_interpolate(ctx, ggml_reshape_4d(ctx, f0, T, 1, 1, 1), T_pcm, 1, 1, 1, GGML_SCALE_MODE_NEAREST),
        T_pcm, 1);
    ggml_tensor * tmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_pcm, n_harm);
    ggml_tensor * scales = ggml_reshape_2d(ctx,
        ggml_scale(ctx, codec_op_arange(ctx, 1.0f, (float) (n_harm + 1), 1.0f), 1.0f / (float) p.sample_rate),
        1, n_harm);

    ggml_tensor * sine = nullptr;
    if (p.sinegen == 2) {
        // SineGen2: rad = (f0 * k / sr) % 1 at frame rate (the linear
        // 1/up downsample reads two equal samples of each frame, so the
        // per-sample random initial phase never survives), phase =
        // 2*pi*cumsum(rad)*up, then a half-pixel linear upsample by `up`.
        ggml_tensor * f0_2d = ggml_reshape_2d(ctx, f0, T, 1);
        ggml_tensor * ftmpl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, n_harm);
        ggml_tensor * rad = ggml_mul(ctx, ggml_repeat(ctx, f0_2d, ftmpl), ggml_repeat(ctx, scales, ftmpl));
        rad = frac_bounded(ctx, rad);
        ggml_tensor * phase = ggml_scale(ctx, ggml_cumsum(ctx, rad), 2.0f * (float) M_PI * (float) up);
        sine = codec_op_sin(ctx, p.causal ? ggml_interpolate(ctx, phase, T_pcm, n_harm, 1, 1, GGML_SCALE_MODE_NEAREST)
                                      : upsample_linear_half_pixel(ctx, phase, up, join(prefix, "lut.sg2_w").c_str()));
    } else {
        // SineGen: theta = 2*pi*cumsum_t(f0 * k / sr) + phi_k.
        ggml_tensor * f_harm = ggml_mul(ctx, ggml_repeat(ctx, f0_pcm, tmpl), ggml_repeat(ctx, scales, tmpl));
        ggml_tensor * theta = ggml_scale(ctx, ggml_cumsum(ctx, f_harm), 2.0f * (float) M_PI);
        ggml_tensor * phase_2d = ggml_reshape_2d(ctx, nsf_phase, 1, n_harm);
        sine = codec_op_sin(ctx, ggml_add(ctx, theta, ggml_repeat(ctx, phase_2d, tmpl)));
    }
    sine = ggml_scale(ctx, sine, p.nsf_alpha);

    // uv = f0 > threshold; noise_amp = uv*sigma + (1-uv)*alpha/3.
    ggml_tensor * uv = ggml_repeat(ctx, ggml_step(ctx, ggml_scale_bias(ctx, f0_pcm, 1.0f, -p.nsf_voiced_threshold)), tmpl);
    ggml_tensor * noise_amp = ggml_scale_bias(ctx, uv, p.nsf_sigma - p.nsf_alpha / 3.0f, p.nsf_alpha / 3.0f);
    ggml_tensor * waves = ggml_add(ctx, ggml_mul(ctx, sine, uv), ggml_mul(ctx, noise_amp, nsf_noise));

    // l_linear (n_harm → 1) + tanh.
    ggml_tensor * lin_w = codec_graph_weight(ctx, model, join(prefix, "src.lin.w"));
    ggml_tensor * lin_b = codec_graph_weight(ctx, model, join(prefix, "src.lin.b"));
    if (lin_w == nullptr || lin_b == nullptr) return nullptr;
    ggml_tensor * merged = codec_op_linear(ctx, ggml_cont(ctx, ggml_transpose(ctx, waves)), lin_w, lin_b);
    if (merged == nullptr) return nullptr;
    return ggml_tanh(ctx, ggml_reshape_1d(ctx, merged, T_pcm));
}

}  // namespace

int32_t codec_hift_upsample_factor(const codec_hift_params & p) {
    int32_t up = p.hop;
    for (int32_t r : p.upsample_rates) up *= r;
    return up;
}

ggml_tensor * codec_block_hift(
        ggml_context * ctx,
        const codec_model * model,
        const char * prefix,
        const codec_hift_params & p,
        ggml_tensor * mel_tc,
        ggml_tensor * nsf_phase,
        ggml_tensor * nsf_noise) {
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, model, join(prefix, n)); };
    const int32_t n_ups = (int32_t) p.upsample_rates.size();
    if (ctx == nullptr || mel_tc == nullptr || n_ups <= 0 || (int32_t) p.upsample_kernels.size() != n_ups ||
        (int32_t) p.source_resblock_kernels.size() != n_ups || p.resblock_kernels.empty()) {
        return nullptr;
    }
    const int32_t n_fft = p.n_fft;
    const int32_t n_bins = n_fft / 2 + 1;

    ggml_tensor * f0 = hift_f0(ctx, model, prefix, p, mel_tc);
    if (f0 == nullptr) return nullptr;
    ggml_tensor * source = hift_source(ctx, model, prefix, p, f0, nsf_phase, nsf_noise);
    if (source == nullptr) return nullptr;
    const int32_t T_pcm = (int32_t) source->ne[0];

    // DFT tables (window baked in) as graph constants.
    std::vector<float> hann, stft_re_k, stft_im_k, istft_re, istft_im, ola_w;
    codec_runtime_stft_basis_kernels(n_fft, &stft_re_k, &stft_im_k, &hann);
    codec_runtime_istft_synthesis_basis(n_fft, hann, &istft_re, &istft_im);
    codec_runtime_ola_identity_kernel(n_fft, &ola_w);
    const std::string cp = join(prefix, "lut.");
    ggml_tensor * t_stft_re = codec_graph_const_f32(ctx, (cp + "stft_re").c_str(), stft_re_k, n_fft, 1, n_bins);
    ggml_tensor * t_stft_im = codec_graph_const_f32(ctx, (cp + "stft_im").c_str(), stft_im_k, n_fft, 1, n_bins);
    ggml_tensor * t_istft_re = codec_graph_const_f32(ctx, (cp + "istft_re").c_str(), istft_re, n_bins, n_fft);
    ggml_tensor * t_istft_im = codec_graph_const_f32(ctx, (cp + "istft_im").c_str(), istft_im, n_bins, n_fft);
    ggml_tensor * t_ola = codec_graph_const_f32(ctx, (cp + "ola").c_str(), ola_w, n_fft, 1, n_fft);
    std::vector<float> hann_sq(hann.size());
    for (size_t i = 0; i < hann.size(); ++i) hann_sq[i] = hann[i] * hann[i];
    ggml_tensor * t_hann_sq = codec_graph_const_f32(ctx, (cp + "hann_sq").c_str(), hann_sq, 1, n_fft);

    // Source STFT (center, zero pad n_fft/2) → [T_stft, n_fft + 2] (re ‖ im).
    ggml_tensor * sm = codec_op_pad_1d(ctx, ggml_reshape_2d(ctx, source, T_pcm, 1), n_fft / 2, n_fft / 2);
    ggml_tensor * s_re = codec_conv1d(ctx, sm, t_stft_re, nullptr, p.hop, 1, 0);
    ggml_tensor * s_im = codec_conv1d(ctx, sm, t_stft_im, nullptr, p.hop, 1, 0);
    if (s_re == nullptr || s_im == nullptr) return nullptr;
    ggml_tensor * s_stft = ggml_concat(ctx, s_re, s_im, 1);

    // Main path.
    ggml_tensor * cpre = W("conv_pre.w");
    if (cpre == nullptr) return nullptr;
    const int32_t k_pre = (int32_t) cpre->ne[0];
    ggml_tensor * x = p.causal ? conv_padded(ctx, mel_tc, cpre, W("conv_pre.b"), 0, k_pre - 1)   // looks right
                               : codec_conv1d(ctx, mel_tc, cpre, W("conv_pre.b"), 1, 1, (k_pre - 1) / 2);
    if (x == nullptr) return nullptr;

    // Source-down stride of stage i = prod(rates[i+1:]) (the reversed cumprod
    // of [1] + rates[::-1][:-1] in HiFTGenerator.__init__).
    std::vector<int32_t> src_strides((size_t) n_ups, 1);
    for (int32_t i = 0; i < n_ups; ++i) {
        for (int32_t j = i + 1; j < n_ups; ++j) src_strides[(size_t) i] *= p.upsample_rates[(size_t) j];
    }

    const int32_t n_rb = (int32_t) p.resblock_kernels.size();
    for (int32_t i = 0; i < n_ups; ++i) {
        const std::string si_s = std::to_string(i);
        const int32_t u = p.upsample_rates[(size_t) i];
        const int32_t k = p.upsample_kernels[(size_t) i];
        x = codec_op_leaky_relu(ctx, x, p.lrelu_slope);
        if (p.causal) {
            // CausalConv1dUpsample: nearest x u, left-pad k - 1, Conv1d.
            x = ggml_interpolate(ctx, x, x->ne[0] * u, x->ne[1], 1, 1, GGML_SCALE_MODE_NEAREST);
            x = conv_padded(ctx, x, W("up." + si_s + ".w"), W("up." + si_s + ".b"), k - 1, 0);
        } else {
            x = codec_convtr1d(ctx, x, W("up." + si_s + ".w"), W("up." + si_s + ".b"), u, (k - u) / 2, 1);
        }
        if (x == nullptr) return nullptr;
        if (i == n_ups - 1) {
            // ReflectionPad1d((1, 0)).
            ggml_tensor * left = ggml_cont(ctx, ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], x->nb[0]));
            x = ggml_concat(ctx, left, x, 0);
        }

        const int32_t sd = src_strides[(size_t) i];
        ggml_tensor * si = p.causal
            ? conv_padded(ctx, s_stft, W("src_dn." + si_s + ".w"), W("src_dn." + si_s + ".b"), sd - 1, 0, sd)
            : codec_conv1d(ctx, s_stft, W("src_dn." + si_s + ".w"), W("src_dn." + si_s + ".b"), sd, 1,
                           sd == 1 ? 0 : sd / 2);
        if (si == nullptr) return nullptr;
        si = hift_resblock(ctx, model, join(prefix, "src_rb." + si_s), p, si, p.source_resblock_kernels[(size_t) i]);
        if (si == nullptr) return nullptr;
        if (si->ne[0] != x->ne[0]) {
            const int64_t common = std::min(si->ne[0], x->ne[0]);
            si = ggml_cont(ctx, ggml_view_2d(ctx, si, common, si->ne[1], si->nb[1], 0));
            x = ggml_cont(ctx, ggml_view_2d(ctx, x, common, x->ne[1], x->nb[1], 0));
        }
        x = ggml_add(ctx, x, si);

        ggml_tensor * xs = nullptr;
        for (int32_t j = 0; j < n_rb; ++j) {
            ggml_tensor * b = hift_resblock(ctx, model, join(prefix, "rb." + std::to_string(i * n_rb + j)), p, x,
                                            p.resblock_kernels[(size_t) j]);
            if (b == nullptr) return nullptr;
            xs = xs == nullptr ? b : ggml_add(ctx, xs, b);
        }
        x = ggml_scale(ctx, xs, 1.0f / (float) n_rb);
    }

    x = codec_op_leaky_relu(ctx, x, kLreluSlopeDefault);
    ggml_tensor * cpw = W("conv_post.w");
    if (cpw == nullptr) return nullptr;
    const int32_t k_post = (int32_t) cpw->ne[0];
    ggml_tensor * head = p.causal ? conv_padded(ctx, x, cpw, W("conv_post.b"), k_post - 1, 0)
                                  : codec_conv1d(ctx, x, cpw, W("conv_post.b"), 1, 1, (k_post - 1) / 2);
    if (head == nullptr) return nullptr;   // [T_head, n_fft + 2]

    // iSTFT: mag = min(exp(logmag), 100) = exp(min(logmag, ln 100)); phase =
    // sin(head) (HiFT's quirk).
    const int32_t T_head = (int32_t) head->ne[0];
    ggml_tensor * head_ct = ggml_cont(ctx, ggml_transpose(ctx, head));
    ggml_tensor * mag_log = ggml_cont(ctx, ggml_view_2d(ctx, head_ct, n_bins, T_head, head_ct->nb[1], 0));
    ggml_tensor * ph = ggml_cont(ctx, ggml_view_2d(ctx, head_ct, n_bins, T_head, head_ct->nb[1],
                                                   (size_t) n_bins * head_ct->nb[0]));
    ggml_tensor * mag = ggml_exp(ctx, ggml_clamp(ctx, mag_log, -1e30f, std::log(1e2f)));
    ggml_tensor * ph_s = codec_op_sin(ctx, ph);
    ggml_tensor * re = ggml_mul(ctx, mag, codec_op_cos(ctx, ph_s));
    ggml_tensor * im = ggml_mul(ctx, mag, codec_op_sin(ctx, ph_s));
    ggml_tensor * fr_re = codec_op_linear(ctx, re, t_istft_re, nullptr);
    ggml_tensor * fr_im = codec_op_linear(ctx, im, t_istft_im, nullptr);
    if (fr_re == nullptr || fr_im == nullptr) return nullptr;
    // Synthesis window is baked into the basis: frames = irfft * hann.
    ggml_tensor * frame = ggml_scale(ctx, ggml_sub(ctx, fr_re, fr_im), 1.0f / (float) n_fft);   // [n_fft, T_head]
    ggml_tensor * ola = codec_convtr1d(ctx, ggml_cont(ctx, ggml_transpose(ctx, frame)), t_ola, nullptr, p.hop, 0, 1);
    ggml_tensor * env_in = ggml_repeat(ctx, t_hann_sq, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_head, n_fft));
    ggml_tensor * env = codec_convtr1d(ctx, env_in, t_ola, nullptr, p.hop, 0, 1);
    ggml_tensor * sig = ggml_div(ctx, ola, ggml_clamp(ctx, env, 1e-11f, 1e30f));

    const int32_t pad = n_fft / 2;
    const int32_t pcm_len = (T_head - 1) * p.hop + n_fft - 2 * pad;
    ggml_tensor * s = ggml_cont(ctx, sig);
    ggml_tensor * pcm = ggml_cont(ctx, ggml_view_2d(ctx, s, pcm_len, 1, s->nb[1], (size_t) pad * s->nb[0]));
    pcm = ggml_reshape_1d(ctx, pcm, pcm_len);
    return ggml_clamp(ctx, pcm, -p.audio_limit, p.audio_limit);
}

void codec_hift_fill_random(
        const codec_hift_params & p,
        int32_t n_pcm,
        std::mt19937_64 & rng,
        std::vector<float> * phase,
        std::vector<float> * noise) {
    const int32_t n_harm = p.nb_harmonics + 1;
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni_phase(-(float) M_PI, (float) M_PI);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    phase->assign((size_t) n_harm, 0.0f);
    for (int32_t i = 1; i < n_harm; ++i) (*phase)[(size_t) i] = uni_phase(rng);
    noise->assign((size_t) n_pcm * (size_t) n_harm, 0.0f);
    for (float & v : *noise) v = p.causal ? uni01(rng) : gauss(rng);
}
