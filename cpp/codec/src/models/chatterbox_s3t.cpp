#include "chatterbox_s3t.h"

#include "../ops/conv1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/rope.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

struct codec_chatterbox_s3t_build {
    int32_t t_mel = 0;
    int32_t t_tok = 0;
    int32_t n_mels = 128;
    int32_t hidden = 1280;
    int32_t n_heads = 20;
    int32_t n_layers = 6;
    int32_t fsmn_kernel = 31;
    float rope_theta = 10000.0f;
    const codec_model * model = nullptr;
};


static std::string codec_chatterbox_s3t_block_prefix(int32_t li) {
    return "s3t.enc.blk." + std::to_string(li);
}

static lm_ggml_tensor * codec_chatterbox_s3t_block(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x_ct,
    lm_ggml_tensor * attn_ln_w,
    lm_ggml_tensor * attn_ln_b,
    lm_ggml_tensor * q_w,
    lm_ggml_tensor * q_b,
    lm_ggml_tensor * k_w,
    lm_ggml_tensor * v_w,
    lm_ggml_tensor * v_b,
    lm_ggml_tensor * o_w,
    lm_ggml_tensor * o_b,
    lm_ggml_tensor * fsmn_w,
    lm_ggml_tensor * mlp_ln_w,
    lm_ggml_tensor * mlp_ln_b,
    lm_ggml_tensor * fc1_w,
    lm_ggml_tensor * fc1_b,
    lm_ggml_tensor * fc2_w,
    lm_ggml_tensor * fc2_b,
    int32_t n_heads,
    float rope_theta,
    int32_t fsmn_kernel) {
    if (ctx_eval == nullptr || x_ct == nullptr || n_heads <= 0 || fsmn_kernel <= 0 || (fsmn_kernel % 2) == 0) {
        return nullptr;
    }

    const int32_t hidden = (int32_t) x_ct->ne[0];
    const int32_t t = (int32_t) x_ct->ne[1];
    if (hidden <= 0 || t <= 0 || hidden % n_heads != 0) {
        return nullptr;
    }
    const int32_t head_dim = hidden / n_heads;

    lm_ggml_tensor * h_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, attn_ln_w, attn_ln_b);
    if (h_ct == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * q_ct = codec_op_linear(ctx_eval, h_ct, q_w, q_b);
    lm_ggml_tensor * k_ct = codec_op_linear(ctx_eval, h_ct, k_w, nullptr);
    lm_ggml_tensor * v_ct = codec_op_linear(ctx_eval, h_ct, v_w, v_b);
    if (q_ct == nullptr || k_ct == nullptr || v_ct == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * q_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, q_ct, head_dim, n_heads, t), 0, 2, 1, 3);
    lm_ggml_tensor * k_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, k_ct, head_dim, n_heads, t), 0, 2, 1, 3);
    lm_ggml_tensor * v_dth = lm_ggml_permute(ctx_eval, lm_ggml_reshape_3d(ctx_eval, v_ct, head_dim, n_heads, t), 0, 2, 1, 3);
    lm_ggml_tensor * q_rope = codec_op_rope(ctx_eval, q_dth, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
    lm_ggml_tensor * k_rope = codec_op_rope(ctx_eval, k_dth, head_dim, rope_theta, 1.0f, CODEC_ROPE_MODE_NEOX);
    if (q_rope == nullptr || k_rope == nullptr || v_dth == nullptr) {
        return nullptr;
    }

    codec_lm_attn_params attn_p = {};
    attn_p.scale = 1.0f / std::sqrt((float) head_dim);
    attn_p.causal = false;
    lm_ggml_tensor * attn_ctx = codec_op_lm_attn_ctx_dth(ctx_eval, q_rope, k_rope, v_dth, &attn_p);
    if (attn_ctx == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * attn_ct = lm_ggml_reshape_2d(
        ctx_eval,
        lm_ggml_cont(ctx_eval, lm_ggml_permute(ctx_eval, attn_ctx, 0, 2, 1, 3)),
        hidden,
        t);
    lm_ggml_tensor * attn_proj = codec_op_linear(ctx_eval, attn_ct, o_w, o_b);
    if (attn_proj == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * v_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, v_ct));
    lm_ggml_tensor * fsmn_tc = codec_conv1d_depthwise(ctx_eval, v_tc, fsmn_w, nullptr, 1, 1, fsmn_kernel / 2);
    if (fsmn_tc == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * fsmn_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, fsmn_tc));
    fsmn_ct = lm_ggml_add(ctx_eval, fsmn_ct, v_ct);
    x_ct = lm_ggml_add(ctx_eval, x_ct, lm_ggml_add(ctx_eval, attn_proj, fsmn_ct));

    lm_ggml_tensor * m_ct = codec_op_layer_norm_ct(ctx_eval, x_ct, 1e-5f, mlp_ln_w, mlp_ln_b);
    if (m_ct == nullptr) {
        return nullptr;
    }
    lm_ggml_tensor * ff = codec_op_linear(ctx_eval, m_ct, fc1_w, fc1_b);
    if (ff == nullptr) {
        return nullptr;
    }
    ff = lm_ggml_gelu_erf(ctx_eval, ff);
    ff = codec_op_linear(ctx_eval, ff, fc2_w, fc2_b);
    if (ff == nullptr) {
        return nullptr;
    }

    return lm_ggml_add(ctx_eval, x_ct, ff);
}

static bool codec_chatterbox_s3t_build_encode(
    lm_ggml_context * ctx_eval,
    void * user_data,
    lm_ggml_tensor ** out) {
    if (ctx_eval == nullptr || user_data == nullptr || out == nullptr) {
        return false;
    }

    const codec_chatterbox_s3t_build * p = static_cast<const codec_chatterbox_s3t_build *>(user_data);
    if (p->t_mel <= 0 || p->t_tok <= 0 || p->n_mels <= 0 || p->hidden <= 0 || p->n_heads <= 0 ||
        p->n_layers <= 0 || p->hidden % p->n_heads != 0 || p->model == nullptr) {
        return false;
    }

    auto W = [&](const std::string & name) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx_eval, p->model, name);
    };

    lm_ggml_tensor * t_mel = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->t_mel, p->n_mels);
    lm_ggml_set_name(t_mel, "s3t.encode.mel");

    lm_ggml_tensor * t_conv1_w = W("s3t.enc.conv1.w");
    lm_ggml_tensor * t_conv1_b = W("s3t.enc.conv1.b");
    lm_ggml_tensor * t_conv2_w = W("s3t.enc.conv2.w");
    lm_ggml_tensor * t_conv2_b = W("s3t.enc.conv2.b");
    if (t_conv1_w == nullptr || t_conv1_b == nullptr || t_conv2_w == nullptr || t_conv2_b == nullptr) {
        return false;
    }

    lm_ggml_tensor * x_tc = codec_conv1d(ctx_eval, t_mel, t_conv1_w, t_conv1_b, 2, 1, 1);
    if (x_tc == nullptr) {
        return false;
    }
    x_tc = lm_ggml_gelu_erf(ctx_eval, x_tc);
    x_tc = codec_conv1d(ctx_eval, x_tc, t_conv2_w, t_conv2_b, 2, 1, 1);
    if (x_tc == nullptr) {
        return false;
    }
    x_tc = lm_ggml_gelu_erf(ctx_eval, x_tc);

    lm_ggml_tensor * x_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, x_tc));
    for (int32_t li = 0; li < p->n_layers; ++li) {
        const std::string base = codec_chatterbox_s3t_block_prefix(li);

        lm_ggml_tensor * t_attn_ln_w = W(base + ".attn_ln.w");
        lm_ggml_tensor * t_attn_ln_b = W(base + ".attn_ln.b");
        lm_ggml_tensor * t_q_w = W(base + ".attn.q.w");
        lm_ggml_tensor * t_q_b = W(base + ".attn.q.b");
        lm_ggml_tensor * t_k_w = W(base + ".attn.k.w");
        lm_ggml_tensor * t_v_w = W(base + ".attn.v.w");
        lm_ggml_tensor * t_v_b = W(base + ".attn.v.b");
        lm_ggml_tensor * t_o_w = W(base + ".attn.o.w");
        lm_ggml_tensor * t_o_b = W(base + ".attn.o.b");
        lm_ggml_tensor * t_fsmn_w = W(base + ".attn.fsmn.w");
        lm_ggml_tensor * t_mlp_ln_w = W(base + ".mlp_ln.w");
        lm_ggml_tensor * t_mlp_ln_b = W(base + ".mlp_ln.b");
        lm_ggml_tensor * t_fc1_w = W(base + ".mlp.fc1.w");
        lm_ggml_tensor * t_fc1_b = W(base + ".mlp.fc1.b");
        lm_ggml_tensor * t_fc2_w = W(base + ".mlp.fc2.w");
        lm_ggml_tensor * t_fc2_b = W(base + ".mlp.fc2.b");
        if (t_attn_ln_w == nullptr || t_attn_ln_b == nullptr || t_q_w == nullptr || t_q_b == nullptr ||
            t_k_w == nullptr || t_v_w == nullptr || t_v_b == nullptr || t_o_w == nullptr ||
            t_o_b == nullptr || t_fsmn_w == nullptr || t_mlp_ln_w == nullptr || t_mlp_ln_b == nullptr ||
            t_fc1_w == nullptr || t_fc1_b == nullptr || t_fc2_w == nullptr || t_fc2_b == nullptr) {
            return false;
        }

        x_ct = codec_chatterbox_s3t_block(
            ctx_eval,
            x_ct,
            t_attn_ln_w,
            t_attn_ln_b,
            t_q_w,
            t_q_b,
            t_k_w,
            t_v_w,
            t_v_b,
            t_o_w,
            t_o_b,
            t_fsmn_w,
            t_mlp_ln_w,
            t_mlp_ln_b,
            t_fc1_w,
            t_fc1_b,
            t_fc2_w,
            t_fc2_b,
            p->n_heads,
            p->rope_theta,
            p->fsmn_kernel);
        if (x_ct == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * t_q_proj_w = W("s3t.q.proj.w");
    lm_ggml_tensor * t_q_proj_b = W("s3t.q.proj.b");
    if (t_q_proj_w == nullptr || t_q_proj_b == nullptr) {
        return false;
    }
    // Token-id powers {1,3,9,...} are baked here as a graph leaf written at
    // runtime; they're a constant base-3 expansion, not in the GGUF.
    lm_ggml_tensor * t_q_powers = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, 8);
    lm_ggml_set_name(t_q_powers, "s3t.encode.q.powers");

    lm_ggml_tensor * q_ct = codec_op_linear(ctx_eval, x_ct, t_q_proj_w, t_q_proj_b);
    if (q_ct == nullptr) {
        return false;
    }
    q_ct = lm_ggml_tanh(ctx_eval, q_ct);
    q_ct = lm_ggml_scale(ctx_eval, q_ct, 0.9990000128746033f);
    q_ct = lm_ggml_round(ctx_eval, q_ct);
    q_ct = lm_ggml_scale_bias(ctx_eval, q_ct, 1.0f, 1.0f);

    lm_ggml_tensor * powers_2d = lm_ggml_reshape_2d(ctx_eval, t_q_powers, 8, 1);
    lm_ggml_tensor * idx_ct = lm_ggml_mul(ctx_eval, q_ct, lm_ggml_repeat(ctx_eval, powers_2d, q_ct));
    lm_ggml_tensor * idx_sum = lm_ggml_sum_rows(ctx_eval, idx_ct);
    lm_ggml_tensor * idx_i32 = lm_ggml_cast(ctx_eval, lm_ggml_reshape_1d(ctx_eval, idx_sum, p->t_tok), LM_GGML_TYPE_I32);
    lm_ggml_tensor * t_out = lm_ggml_reshape_2d(ctx_eval, idx_i32, p->t_tok, 1);
    lm_ggml_set_name(t_out, "s3t.encode.out");
    *out = t_out;
    return true;
}

static bool codec_chatterbox_s3t_write_encode_powers(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    std::string * err) {
    lm_ggml_tensor * t_q_powers = codec_graph_get_tensor(ctx, entry, "s3t.encode.q.powers");
    if (t_q_powers == nullptr) {
        if (err != nullptr) {
            *err = "missing Chatterbox-S3T powers tensor";
        }
        return false;
    }
    const float powers[8] = { 1.0f, 3.0f, 9.0f, 27.0f, 81.0f, 243.0f, 729.0f, 2187.0f };
    return codec_runtime_write_tensor(t_q_powers, powers, sizeof(powers), err);
}

static bool codec_chatterbox_s3t_prepare_log_mel(
    codec_context * ctx,
    const std::vector<float> & pcm,
    std::vector<float> * out_mel,
    int32_t * out_frames,
    std::string * err) {
    if (ctx == nullptr || ctx->model == nullptr || ctx->model->impl == nullptr || out_mel == nullptr || out_frames == nullptr) {
        if (err != nullptr) {
            *err = "invalid Chatterbox-S3T log-mel arguments";
        }
        return false;
    }
    if (pcm.empty()) {
        if (err != nullptr) {
            *err = "empty Chatterbox-S3T PCM input";
        }
        return false;
    }

    codec_chatterbox_s3t & s3t = *static_cast<codec_chatterbox_s3t *>(ctx->model->impl);
    if (s3t.n_fft <= 0 || s3t.win_length <= 0 || s3t.n_mels <= 0) {
        if (err != nullptr) {
            *err = "invalid Chatterbox-S3T frontend metadata";
        }
        return false;
    }

    lm_ggml_tensor * mel_tensor = codec_model_get_tensor(ctx->model, "s3t.mel_filters");
    if (mel_tensor == nullptr) {
        if (err != nullptr) {
            *err = "missing Chatterbox-S3T mel filter tensor";
        }
        return false;
    }
    std::vector<float> mel_filters;
    if (!codec_tensor_as_vec_f32(mel_tensor, &mel_filters)) {
        if (err != nullptr) {
            *err = "failed to read Chatterbox-S3T mel filter tensor";
        }
        return false;
    }

    const int32_t n_fft = s3t.n_fft;
    const int32_t hop = 160;
    const int32_t n_bins = n_fft / 2 + 1;
    if ((int32_t) mel_filters.size() != s3t.n_mels * n_bins) {
        if (err != nullptr) {
            *err = "unexpected Chatterbox-S3T mel filter shape";
        }
        return false;
    }

    lm_ggml_tensor * window_tensor = codec_model_get_tensor(ctx->model, "s3t.window");
    std::vector<float> window;
    if (window_tensor != nullptr) {
        if (!codec_tensor_as_vec_f32(window_tensor, &window)) {
            if (err != nullptr) {
                *err = "failed to read Chatterbox-S3T window tensor";
            }
            return false;
        }
    } else {
        window.assign((size_t) s3t.win_length, 0.0f);
        const float period = (float) std::max(1, s3t.win_length);
        for (int32_t i = 0; i < s3t.win_length; ++i) {
            window[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * (float) i / period);
        }
    }
    if ((int32_t) window.size() != s3t.win_length) {
        if (err != nullptr) {
            *err = "unexpected Chatterbox-S3T window shape";
        }
        return false;
    }

    const int32_t token_hop = 640;
    const int32_t padded_pcm = ((int32_t) pcm.size() + token_hop - 1) / token_hop * token_hop;
    const int32_t mel_frames = padded_pcm / hop;
    if (mel_frames <= 0) {
        if (err != nullptr) {
            *err = "invalid Chatterbox-S3T mel frame count";
        }
        return false;
    }

    std::vector<float> pcm_pad((size_t) padded_pcm, 0.0f);
    std::memcpy(pcm_pad.data(), pcm.data(), pcm.size() * sizeof(float));

    const int32_t center_pad = n_fft / 2;
    auto reflect_index = [](int32_t idx, int32_t len) -> int32_t {
        if (len <= 1) {
            return 0;
        }
        while (idx < 0 || idx >= len) {
            if (idx < 0) {
                idx = -idx;
            } else {
                idx = 2 * len - 2 - idx;
            }
        }
        return idx;
    };
    std::vector<float> centered((size_t) padded_pcm + (size_t) center_pad * 2, 0.0f);
    for (int32_t i = 0; i < (int32_t) centered.size(); ++i) {
        centered[(size_t) i] = pcm_pad[(size_t) reflect_index(i - center_pad, padded_pcm)];
    }

    std::vector<float> cos_table((size_t) n_bins * (size_t) n_fft, 0.0f);
    std::vector<float> sin_table((size_t) n_bins * (size_t) n_fft, 0.0f);
    for (int32_t k = 0; k < n_bins; ++k) {
        for (int32_t n = 0; n < n_fft; ++n) {
            const float ang = 2.0f * (float) M_PI * (float) k * (float) n / (float) n_fft;
            cos_table[(size_t) k * (size_t) n_fft + (size_t) n] = std::cos(ang);
            sin_table[(size_t) k * (size_t) n_fft + (size_t) n] = std::sin(ang);
        }
    }

    // ggml stores ne[0] as the fastest-moving dimension. For a logical [t, c]
    // input tensor, the host buffer must therefore be laid out as [c][t].
    out_mel->assign((size_t) mel_frames * (size_t) s3t.n_mels, 0.0f);
    std::vector<float> power((size_t) n_bins, 0.0f);
    float global_max = -std::numeric_limits<float>::infinity();
    for (int32_t ti = 0; ti < mel_frames; ++ti) {
        const int32_t start = ti * hop;
        for (int32_t k = 0; k < n_bins; ++k) {
            double re = 0.0;
            double im = 0.0;
            for (int32_t n = 0; n < n_fft; ++n) {
                float sample = centered[(size_t) start + (size_t) n];
                if (n < s3t.win_length) {
                    sample *= window[(size_t) n];
                } else {
                    sample = 0.0f;
                }
                const size_t off = (size_t) k * (size_t) n_fft + (size_t) n;
                re += (double) sample * (double) cos_table[off];
                im -= (double) sample * (double) sin_table[off];
            }
            power[(size_t) k] = (float) (re * re + im * im);
        }

        for (int32_t mi = 0; mi < s3t.n_mels; ++mi) {
            double mel = 0.0;
            const float * filt = mel_filters.data() + (size_t) mi * (size_t) n_bins;
            for (int32_t k = 0; k < n_bins; ++k) {
                mel += (double) filt[(size_t) k] * (double) power[(size_t) k];
            }
            const float log_spec = std::log10(std::max((float) mel, 1.0e-10f));
            (*out_mel)[(size_t) mi * (size_t) mel_frames + (size_t) ti] = log_spec;
            global_max = std::max(global_max, log_spec);
        }
    }

    const float floor_val = global_max - 8.0f;
    for (float & v : *out_mel) {
        v = std::max(v, floor_val);
        v = (v + 4.0f) * 0.25f;
    }

    *out_frames = mel_frames;
    return true;
}

enum codec_status codec_chatterbox_s3t_init(struct codec_model * model) {
    if (model == nullptr || model->impl == nullptr || model->gguf == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3t & s3t = *static_cast<codec_chatterbox_s3t *>(model->impl);
    s3t.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", s3t.sample_rate);
    s3t.encode_sample_rate = codec_read_i32_kv(model->gguf, "codec.encode_sample_rate", s3t.encode_sample_rate);
    s3t.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", s3t.hop_size);
    s3t.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", s3t.n_q);
    s3t.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", s3t.codebook_size);
    s3t.n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", s3t.n_fft);
    s3t.win_length = codec_read_i32_kv(model->gguf, "codec.win_length", s3t.win_length);
    s3t.n_mels = codec_read_i32_kv(model->gguf, "codec.n_mels", s3t.n_mels);
    s3t.audio_state = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_state", s3t.audio_state);
    s3t.audio_head = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_head", s3t.audio_head);
    s3t.audio_layer = codec_read_i32_kv(model->gguf, "chatterbox_s3t.audio_layer", s3t.audio_layer);
    s3t.fsmn_kernel_size = codec_read_i32_kv(model->gguf, "chatterbox_s3t.fsmn_kernel_size", s3t.fsmn_kernel_size);
    s3t.rope_theta = codec_read_f32_kv(model->gguf, "chatterbox_s3t.rope_theta", s3t.rope_theta);
    s3t.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", s3t.has_encoder);
    s3t.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", s3t.has_decoder);

    model->sample_rate = s3t.sample_rate;
    model->encode_sample_rate = s3t.encode_sample_rate;
    model->has_encoder = s3t.has_encoder;
    model->has_decoder = s3t.has_decoder;
    model->hop_size = s3t.hop_size;
    model->n_q = s3t.n_q;
    model->codebook_size = s3t.codebook_size;
    model->n_fft = s3t.n_fft;
    model->win_length = s3t.win_length;
    model->n_mels = s3t.n_mels;
    model->latent_dim = -1;

    if (s3t.n_q != 1 || s3t.codebook_size != 6561 || s3t.audio_state <= 0 || s3t.audio_head <= 0 ||
        s3t.audio_state % s3t.audio_head != 0 || s3t.audio_layer <= 0 || s3t.fsmn_kernel_size <= 0 ||
        (s3t.fsmn_kernel_size % 2) == 0 || s3t.rope_theta <= 0.0f) {
        return CODEC_STATUS_INVALID_ARG;
    }

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_chatterbox_s3t_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || out_tokens == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_chatterbox_s3t & s3t = *static_cast<codec_chatterbox_s3t *>(ctx->model->impl);
    if (!s3t.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (params.n_q != 0 && params.n_q != 1) {
        codec_context_set_error(ctx, "Chatterbox-S3T encode n_q must be 0 or 1");
        return CODEC_STATUS_INVALID_ARG;
    }
    (void) out_latent;

    std::vector<float> mel_tc;
    int32_t t_mel = 0;
    std::string err;
    if (!codec_chatterbox_s3t_prepare_log_mel(ctx, pcm, &mel_tc, &t_mel, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t t_tok = t_mel / 4;
    if (t_tok <= 0 || t_mel != t_tok * 4) {
        codec_context_set_error(ctx, "Chatterbox-S3T frontend produced invalid frame count");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_chatterbox_s3t_build build = {};
    build.t_mel = t_mel;
    build.t_tok = t_tok;
    build.n_mels = s3t.n_mels;
    build.hidden = s3t.audio_state;
    build.n_heads = s3t.audio_head;
    build.n_layers = s3t.audio_layer;
    build.fsmn_kernel = s3t.fsmn_kernel_size;
    build.rope_theta = s3t.rope_theta;
    build.model = ctx->model;

    codec_graph_eval_guard eval_guard(ctx);
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_CHATTERBOX_S3T_ENCODE, /*n_frames=*/t_tok, /*n_q=*/1, /*hop=*/s3t.hop_size, /*n_in=*/t_mel, /*latent_dim=*/build.hidden },
            codec_chatterbox_s3t_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_mel_tensor = codec_graph_get_tensor(ctx, entry, "s3t.encode.mel");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "s3t.encode.out");
    if (t_mel_tensor == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached Chatterbox-S3T encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_mel_tensor, mel_tc.data(), mel_tc.size() * sizeof(float), &err) ||
        !codec_chatterbox_s3t_write_encode_powers(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_compute(ctx, entry, ctx->model->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> tok;
    if (!codec_runtime_read_tensor_i32_2d_tq(t_out, &tok, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if ((int32_t) tok.size() != t_tok) {
        codec_context_set_error(ctx, "unexpected Chatterbox-S3T token output shape");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) t_tok * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), (size_t) t_tok * sizeof(int32_t));

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = t_tok;
    out_tokens->n_frames = t_tok;
    out_tokens->n_q = 1;
    out_tokens->codebook_size = s3t.codebook_size;
    out_tokens->sample_rate = s3t.sample_rate;
    out_tokens->hop_size = s3t.hop_size;
    return CODEC_STATUS_SUCCESS;
}

static void * codec_chatterbox_s3t_create_impl() {
    return new (std::nothrow) codec_chatterbox_s3t();
}

static void codec_chatterbox_s3t_destroy_impl(void * ptr) {
    delete static_cast<codec_chatterbox_s3t *>(ptr);
}

const struct codec_model_vtable * codec_chatterbox_s3t_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_CHATTERBOX_S3T,
        "Chatterbox-S3T",
        codec_chatterbox_s3t_create_impl,
        codec_chatterbox_s3t_destroy_impl,
        codec_chatterbox_s3t_init,
        codec_graph_size_exact,
        codec_chatterbox_s3t_encode,
        nullptr,
        nullptr,
    };
    return &vtable;
}
