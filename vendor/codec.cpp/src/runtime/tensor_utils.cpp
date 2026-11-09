#include "tensor_utils.h"
#include "graph.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

bool codec_runtime_write_tensor(ggml_tensor * t, const void * data, size_t n_bytes, std::string * error) {
    if (t == nullptr || data == nullptr) {
        if (error != nullptr) {
            *error = "invalid tensor set arguments";
        }
        return false;
    }
    if (n_bytes != ggml_nbytes(t)) {
        if (error != nullptr) {
            *error = "tensor set size mismatch";
        }
        return false;
    }
    if (t->buffer == nullptr) {
        if (error != nullptr) {
            const char * name = t->name[0] != '\0' ? t->name : "<unnamed>";
            *error = std::string("tensor buffer not set: ") + name;
        }
        return false;
    }
    ggml_backend_tensor_set(t, data, 0, n_bytes);
    return true;
}

bool codec_runtime_read_tensor(ggml_tensor * t, void * data, size_t n_bytes, std::string * error) {
    if (t == nullptr || data == nullptr) {
        if (error != nullptr) {
            *error = "invalid tensor get arguments";
        }
        return false;
    }
    if (n_bytes != ggml_nbytes(t)) {
        if (error != nullptr) {
            *error = "tensor get size mismatch";
        }
        return false;
    }
    if (t->buffer == nullptr) {
        if (error != nullptr) {
            const char * name = t->name[0] != '\0' ? t->name : "<unnamed>";
            *error = std::string("tensor buffer not set: ") + name;
        }
        return false;
    }
    ggml_backend_tensor_get(t, data, 0, n_bytes);
    return true;
}

bool codec_runtime_read_tensor_i32_2d_tq(ggml_tensor * t, std::vector<int32_t> * out, std::string * error) {
    if (t == nullptr || out == nullptr) {
        if (error != nullptr) {
            *error = "invalid token tensor get arguments";
        }
        return false;
    }
    if (t->type != GGML_TYPE_I32) {
        if (error != nullptr) {
            *error = "token tensor must be int32";
        }
        return false;
    }
    if (t->ne[0] <= 0 || t->ne[1] <= 0) {
        if (error != nullptr) {
            *error = "token tensor must be 2D with positive shape";
        }
        return false;
    }

    const int32_t n_frames = (int32_t) t->ne[0];
    const int32_t n_q = (int32_t) t->ne[1];
    std::vector<int32_t> raw((size_t) n_frames * (size_t) n_q, 0);
    if (!codec_runtime_read_tensor(t, raw.data(), raw.size() * sizeof(int32_t), error)) {
        return false;
    }

    out->assign(raw.size(), 0);
    for (int32_t qi = 0; qi < n_q; ++qi) {
        for (int32_t ti = 0; ti < n_frames; ++ti) {
            // ggml stores ne[0] as the fastest dimension; codec token buffers are time-major [t, q].
            (*out)[(size_t) ti * (size_t) n_q + (size_t) qi] =
                raw[(size_t) qi * (size_t) n_frames + (size_t) ti];
        }
    }
    return true;
}

ggml_tensor * codec_model_get_tensor(const codec_model * model, const char * name) {
    if (model == nullptr || model->weights == nullptr || name == nullptr) {
        return nullptr;
    }
    const auto it = model->weight_index.find(name);
    return it == model->weight_index.end() ? nullptr : it->second;
}

ggml_tensor * codec_model_get_tensor(const codec_model * model, const std::string & name) {
    return codec_model_get_tensor(model, name.c_str());
}

static ggml_tensor * codec_graph_cast_f32_new(ggml_context * ctx_eval, ggml_tensor * t);

ggml_tensor * codec_graph_cast_f32(ggml_context * ctx_eval, ggml_tensor * t) {
    if (ctx_eval == nullptr || t == nullptr) {
        return nullptr;
    }
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    // Small stored weights (biases, norm scales) are cast once per graph and
    // shared; large ones are not, so their F32 copies don't all stay alive.
    ggml_tensor ** slot = nullptr;
    if (t->op == GGML_OP_NONE && t->buffer != nullptr && ggml_nelements(t) <= 65536) {
        slot = codec_graph_weight_cast_slot(ctx_eval, t);
        if (slot != nullptr && *slot != nullptr) {
            return *slot;
        }
    }
    ggml_tensor * y = codec_graph_cast_f32_new(ctx_eval, t);
    if (slot != nullptr) {
        *slot = y;
    }
    return y;
}

static ggml_tensor * codec_graph_cast_f32_new(ggml_context * ctx_eval, ggml_tensor * t) {
    // Short rows (1x1 conv weights are [1, c_in, c_out]) make GPU copy kernels,
    // which run one threadgroup per row, crawl (Metal: 56 ms for a 1024x2048
    // weight).  Cast those as wide rows and view the result back.
    if (t->ne[0] < 32 && ggml_is_contiguous(t) && (t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16)) {
        const int64_t n = ggml_nelements(t);
        for (int64_t w = 4096; w >= 32; w /= 2) {
            if (n % w == 0) {
                ggml_tensor * flat = ggml_cast(ctx_eval, ggml_reshape_2d(ctx_eval, t, w, n / w), GGML_TYPE_F32);
                return ggml_reshape_4d(ctx_eval, flat, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            }
        }
    }
    return ggml_cast(ctx_eval, t, GGML_TYPE_F32);
}

ggml_tensor * codec_graph_mat_lhs(ggml_context * ctx_eval, ggml_tensor * t) {
    if (ctx_eval == nullptr || t == nullptr) {
        return nullptr;
    }
    return codec_graph_stored_weight(ctx_eval, t);
}

ggml_tensor * codec_graph_weight_or_null(ggml_context * ctx_eval, const codec_model * model, const char * name) {
    if (ctx_eval == nullptr || model == nullptr || model->weights == nullptr || name == nullptr) {
        return nullptr;
    }
    ggml_tensor * w = codec_model_get_tensor(model, name);
    if (w == nullptr) {
        return nullptr;
    }
    return codec_graph_cast_f32(ctx_eval, w);
}

ggml_tensor * codec_graph_weight_or_null(ggml_context * ctx_eval, const codec_model * model, const std::string & name) {
    return codec_graph_weight_or_null(ctx_eval, model, name.c_str());
}

ggml_tensor * codec_graph_weight(ggml_context * ctx_eval, const codec_model * model, const char * name) {
    return codec_graph_weight_or_null(ctx_eval, model, name);
}

ggml_tensor * codec_graph_weight_mat(ggml_context * ctx_eval, const codec_model * model, const char * name) {
    if (ctx_eval == nullptr || model == nullptr || model->weights == nullptr || name == nullptr) {
        return nullptr;
    }
    ggml_tensor * w = codec_model_get_tensor(model, name);
    if (w == nullptr) {
        return nullptr;
    }
    return w;
}

ggml_tensor * codec_graph_weight_mat(ggml_context * ctx_eval, const codec_model * model, const std::string & name) {
    return codec_graph_weight_mat(ctx_eval, model, name.c_str());
}

ggml_tensor * codec_graph_stored_weight(ggml_context * ctx_eval, ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    if (t->op == GGML_OP_RESHAPE && t->src[0] != nullptr) {
        ggml_tensor * src = codec_graph_stored_weight(ctx_eval, t->src[0]);
        if (src == t->src[0]) {
            return t;
        }
        // A block-quantized weight can only be viewed with its rows intact.
        if (ggml_is_quantized(src->type) && t->ne[0] % ggml_blck_size(src->type) != 0) {
            return t;
        }
        return ggml_reshape_4d(ctx_eval, src, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    }
    ggml_tensor * w = t->src[0];
    if (t->op != GGML_OP_CPY || t->type != GGML_TYPE_F32 || w == nullptr || w->op != GGML_OP_NONE ||
        w->buffer == nullptr || w->type == GGML_TYPE_F32 || !ggml_is_contiguous(w) || !ggml_are_same_shape(w, t)) {
        return t;
    }
    return w;
}

ggml_tensor * codec_mul_mat(ggml_context * ctx_eval, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * w = codec_graph_stored_weight(ctx_eval, a);
    ggml_tensor * y = ggml_mul_mat(ctx_eval, w, b);
    if (w->type != GGML_TYPE_F32) {
        // F32 accumulation where a backend would accumulate an F16 matmul in F16
        // (Vulkan: S3Gen decode GPU vs CPU 39 dB -> 20 dB).
        ggml_prec_set_acc(y, GGML_PREC_F32);
    }
    return y;
}

ggml_tensor * codec_graph_weight(ggml_context * ctx_eval, const codec_model * model, const std::string & name) {
    return codec_graph_weight_or_null(ctx_eval, model, name.c_str());
}

void codec_context_set_error(struct codec_context * ctx, const std::string & error) {
    if (ctx != nullptr) {
        ctx->last_error = error;
    }
}

bool codec_prepare_mono_f32(const struct codec_audio * audio, std::vector<float> * mono, std::string * error) {
    if (audio == nullptr || mono == nullptr) {
        if (error != nullptr) {
            *error = "audio/tensor output is null";
        }
        return false;
    }

    if (audio->data == nullptr) {
        if (error != nullptr) {
            *error = "audio.data is null";
        }
        return false;
    }

    if (audio->n_samples <= 0) {
        if (error != nullptr) {
            *error = "audio.n_samples must be > 0";
        }
        return false;
    }

    if (audio->n_channels < 1) {
        if (error != nullptr) {
            *error = "audio.n_channels must be >= 1";
        }
        return false;
    }

    // The output buffer is `n_samples * n_channels` floats laid out
    // interleaved (L0, R0, L1, R1, …) — i.e. the standard WAV memory order.
    // Mono codecs see a length-`n_samples` vector unchanged; multi-channel
    // models (e.g. MOSS-Audio-Tokenizer with `enable_channel_interleave=true`)
    // can fold this directly into their per-channel-interleaved input stream.
    const size_t total = (size_t) audio->n_samples * (size_t) audio->n_channels;
    mono->resize(total);

    if (audio->pcm_type == CODEC_PCM_TYPE_F32) {
        const float * src = static_cast<const float *>(audio->data);
        std::copy(src, src + total, mono->begin());
        return true;
    }

    if (audio->pcm_type == CODEC_PCM_TYPE_I16) {
        const int16_t * src = static_cast<const int16_t *>(audio->data);
        for (size_t i = 0; i < total; ++i) {
            (*mono)[i] = std::max(-1.0f, std::min(1.0f, src[i] / 32768.0f));
        }
        return true;
    }

    if (error != nullptr) {
        *error = "unsupported codec_pcm_type";
    }
    return false;
}

void codec_token_buffer_reset(struct codec_token_buffer * tokens) {
    if (tokens == nullptr) {
        return;
    }

    tokens->data = nullptr;
    tokens->n_tokens = 0;
    tokens->n_frames = 0;
    tokens->n_q = 0;
    tokens->codebook_size = 0;
    tokens->sample_rate = 0;
    tokens->hop_size = 0;
}

void codec_pcm_buffer_reset(struct codec_pcm_buffer * pcm) {
    if (pcm == nullptr) {
        return;
    }

    pcm->data = nullptr;
    pcm->n_samples = 0;
    pcm->sample_rate = 0;
    pcm->n_channels = 0;
}

void codec_latent_buffer_reset(struct codec_latent_buffer * latent) {
    if (latent == nullptr) {
        return;
    }

    latent->data = nullptr;
    latent->latent_dim = 0;
    latent->n_frames = 0;
    latent->sample_rate = 0;
    latent->hop_size = 0;
}

const float * codec_tensor_data_f32(const struct ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    if (t->type != GGML_TYPE_F32) {
        return nullptr;
    }

    if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
        return static_cast<const float *>(ggml_get_data(const_cast<struct ggml_tensor *>(t)));
    }

    thread_local std::unordered_map<const struct ggml_tensor *, std::vector<float>> tensor_cache;
    std::vector<float> & cached = tensor_cache[t];
    cached.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), cached.data(), 0, cached.size() * sizeof(float));
    return cached.data();
}

int64_t codec_ne(const struct ggml_tensor * t, int dim) {
    return t != nullptr && dim >= 0 && dim < GGML_MAX_DIMS ? t->ne[dim] : 0;
}

bool codec_tensor_as_vec_f32(const struct ggml_tensor * t, std::vector<float> * out) {
    if (t == nullptr || out == nullptr) {
        return false;
    }

    if (t->type == GGML_TYPE_F32) {
        const size_t n = (size_t)ggml_nelements(t);
        out->resize(n);
        if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
            const float * ptr = static_cast<const float *>(ggml_get_data(const_cast<struct ggml_tensor *>(t)));
            if (ptr == nullptr) {
                return false;
            }
            out->assign(ptr, ptr + n);
        } else {
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), out->data(), 0, n * sizeof(float));
        }
        return true;
    }

    if (t->type == GGML_TYPE_F16) {
        const size_t n = (size_t)ggml_nelements(t);
        out->resize(n);
        if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
            const ggml_fp16_t * ptr = static_cast<const ggml_fp16_t *>(ggml_get_data(const_cast<struct ggml_tensor *>(t)));
            if (ptr == nullptr) {
                return false;
            }
            for (size_t i = 0; i < n; ++i) {
                (*out)[i] = ggml_fp16_to_fp32(ptr[i]);
            }
        } else {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) {
                (*out)[i] = ggml_fp16_to_fp32(tmp[i]);
            }
        }
        return true;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (traits != nullptr && traits->to_float != nullptr && t->ne[0] > 0) {
        const int64_t ne0 = t->ne[0];
        const size_t n = (size_t) ggml_nelements(t);
        if ((n % (size_t) ne0) != 0) {
            return false;
        }

        std::vector<uint8_t> tmp;
        const uint8_t * data = nullptr;
        if (t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer)) {
            data = static_cast<const uint8_t *>(ggml_get_data(const_cast<struct ggml_tensor *>(t)));
            if (data == nullptr) {
                return false;
            }
        } else {
            tmp.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(t), tmp.data(), 0, tmp.size());
            data = tmp.data();
        }

        const size_t row_size = ggml_row_size(t->type, ne0);
        const size_t n_rows = n / (size_t) ne0;
        out->resize(n);
        for (size_t row = 0; row < n_rows; ++row) {
            traits->to_float(data + row * row_size, out->data() + row * (size_t) ne0, ne0);
        }
        return true;
    }

    return false;
}
