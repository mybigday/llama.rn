#include "tensor_utils.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>

bool codec_runtime_write_tensor(lm_ggml_tensor * t, const void * data, size_t n_bytes, std::string * error) {
    if (t == nullptr || data == nullptr) {
        if (error != nullptr) {
            *error = "invalid tensor set arguments";
        }
        return false;
    }
    if (n_bytes != lm_ggml_nbytes(t)) {
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
    lm_ggml_backend_tensor_set(t, data, 0, n_bytes);
    return true;
}

bool codec_runtime_read_tensor(lm_ggml_tensor * t, void * data, size_t n_bytes, std::string * error) {
    if (t == nullptr || data == nullptr) {
        if (error != nullptr) {
            *error = "invalid tensor get arguments";
        }
        return false;
    }
    if (n_bytes != lm_ggml_nbytes(t)) {
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
    lm_ggml_backend_tensor_get(t, data, 0, n_bytes);
    return true;
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

    if (audio->n_channels != 1) {
        if (error != nullptr) {
            *error = "codec core currently requires mono input (n_channels must be 1)";
        }
        return false;
    }

    mono->resize((size_t)audio->n_samples);

    if (audio->pcm_type == CODEC_PCM_TYPE_F32) {
        const float * src = static_cast<const float *>(audio->data);
        std::copy(src, src + audio->n_samples, mono->begin());
        return true;
    }

    if (audio->pcm_type == CODEC_PCM_TYPE_I16) {
        const int16_t * src = static_cast<const int16_t *>(audio->data);
        for (int32_t i = 0; i < audio->n_samples; ++i) {
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

const float * codec_tensor_data_f32(const struct lm_ggml_tensor * t) {
    if (t == nullptr) {
        return nullptr;
    }
    if (t->type != LM_GGML_TYPE_F32) {
        return nullptr;
    }

    if (t->buffer == nullptr || lm_ggml_backend_buffer_is_host(t->buffer)) {
        return static_cast<const float *>(lm_ggml_get_data(const_cast<struct lm_ggml_tensor *>(t)));
    }

    thread_local std::unordered_map<const struct lm_ggml_tensor *, std::vector<float>> tensor_cache;
    std::vector<float> & cached = tensor_cache[t];
    cached.resize((size_t) lm_ggml_nelements(t));
    lm_ggml_backend_tensor_get(const_cast<lm_ggml_tensor *>(t), cached.data(), 0, cached.size() * sizeof(float));
    return cached.data();
}

int64_t codec_ne(const struct lm_ggml_tensor * t, int dim) {
    return t != nullptr && dim >= 0 && dim < LM_GGML_MAX_DIMS ? t->ne[dim] : 0;
}

bool codec_tensor_as_vec_f32(const struct lm_ggml_tensor * t, std::vector<float> * out) {
    if (t == nullptr || out == nullptr) {
        return false;
    }

    if (t->type == LM_GGML_TYPE_F32) {
        const size_t n = (size_t)lm_ggml_nelements(t);
        out->resize(n);
        if (t->buffer == nullptr || lm_ggml_backend_buffer_is_host(t->buffer)) {
            const float * ptr = static_cast<const float *>(lm_ggml_get_data(const_cast<struct lm_ggml_tensor *>(t)));
            if (ptr == nullptr) {
                return false;
            }
            out->assign(ptr, ptr + n);
        } else {
            lm_ggml_backend_tensor_get(const_cast<lm_ggml_tensor *>(t), out->data(), 0, n * sizeof(float));
        }
        return true;
    }

    if (t->type == LM_GGML_TYPE_F16) {
        const size_t n = (size_t)lm_ggml_nelements(t);
        out->resize(n);
        if (t->buffer == nullptr || lm_ggml_backend_buffer_is_host(t->buffer)) {
            const lm_ggml_fp16_t * ptr = static_cast<const lm_ggml_fp16_t *>(lm_ggml_get_data(const_cast<struct lm_ggml_tensor *>(t)));
            if (ptr == nullptr) {
                return false;
            }
            for (size_t i = 0; i < n; ++i) {
                (*out)[i] = lm_ggml_fp16_to_fp32(ptr[i]);
            }
        } else {
            std::vector<lm_ggml_fp16_t> tmp(n);
            lm_ggml_backend_tensor_get(const_cast<lm_ggml_tensor *>(t), tmp.data(), 0, n * sizeof(lm_ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) {
                (*out)[i] = lm_ggml_fp16_to_fp32(tmp[i]);
            }
        }
        return true;
    }

    return false;
}
