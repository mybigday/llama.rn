#include "codec_internal.h"

#include "batch/batch.h"
#include "models/dac.h"
#include "models/mimi.h"
#include "models/qwen3_tts_tokenizer.h"
#include "models/wavtokenizer.h"
#include "ops/safe_math.h"
#include "runtime/graph.h"
#include "runtime/lm_gguf_kv.h"
#include "runtime/tensor_utils.h"

#include <ggml-backend.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <vector>

static void codec_backend_set_n_threads(lm_ggml_backend_t backend, int n_threads) {
    if (backend == nullptr || n_threads <= 0) {
        return;
    }

    lm_ggml_backend_dev_t dev = lm_ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return;
    }

    lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }

    lm_ggml_backend_set_n_threads_t set_n_threads = reinterpret_cast<lm_ggml_backend_set_n_threads_t>(
        lm_ggml_backend_reg_get_proc_address(reg, "lm_ggml_backend_set_n_threads"));
    if (set_n_threads != nullptr) {
        set_n_threads(backend, n_threads);
    }
}

static lm_ggml_backend_t codec_backend_init(bool use_gpu) {
    if (use_gpu) {
        // Load any dynamically available backends and pick the best available device.
        // This follows ggml's own backend selection logic and should work for any
        // ggml-native accelerator backend compiled/available (CUDA/Vulkan/Metal/SYCL/OpenCL/etc.).
        lm_ggml_backend_load_all();

        lm_ggml_backend_t backend = lm_ggml_backend_init_best();
        if (backend != nullptr) {
            return backend;
        }
    }

    // Explicit CPU fallback
    lm_ggml_backend_t backend = lm_ggml_backend_init_by_name("CPU", nullptr);
    if (backend == nullptr) {
        backend = lm_ggml_backend_init_by_type(LM_GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    return backend;
}

void codec_metadata_free(struct codec_lm_gguf_metadata * meta) {
    if (meta == nullptr || meta->items == nullptr) {
        return;
    }

    for (size_t i = 0; i < meta->n_items; ++i) {
        std::free(const_cast<char *>(meta->items[i].key));
        std::free(const_cast<char *>(meta->items[i].value));
    }

    std::free(meta->items);
    meta->items = nullptr;
    meta->n_items = 0;
}

enum codec_arch codec_arch_from_string(const std::string & arch) {
    if (arch == "wavtokenizer_large" || arch == "wavtokenizer-large") {
        return CODEC_ARCH_WAVTOKENIZER_LARGE;
    }

    if (arch == "dac") {
        return CODEC_ARCH_DAC;
    }

    if (arch == "mimi") {
        return CODEC_ARCH_MIMI;
    }

    if (arch == "qwen3_tts_tokenizer" || arch == "qwen3-tts-tokenizer" || arch == "qwen3") {
        return CODEC_ARCH_QWEN3_TTS_TOKENIZER;
    }

    return CODEC_ARCH_UNKNOWN;
}

static const codec_model_vtable * codec_model_vtable_for_arch(enum codec_arch arch) {
    switch (arch) {
        case CODEC_ARCH_WAVTOKENIZER_LARGE:
            return codec_wavtokenizer_vtable();
        case CODEC_ARCH_DAC:
            return codec_dac_vtable();
        case CODEC_ARCH_MIMI:
            return codec_mimi_vtable();
        case CODEC_ARCH_QWEN3_TTS_TOKENIZER:
            return codec_qwen3_tts_tokenizer_vtable();
        case CODEC_ARCH_UNKNOWN:
        default:
            return nullptr;
    }
}

const char * codec_arch_name(enum codec_arch arch) {
    switch (arch) {
        case CODEC_ARCH_WAVTOKENIZER_LARGE: return "WavTokenizer-Large";
        case CODEC_ARCH_DAC:                return "DAC";
        case CODEC_ARCH_MIMI:               return "Mimi";
        case CODEC_ARCH_QWEN3_TTS_TOKENIZER:return "Qwen3-TTS-Tokenizer";
        case CODEC_ARCH_UNKNOWN:
        default:                            return "unknown";
    }
}

enum codec_status codec_model_init_arch(struct codec_model * model) {
    if (model == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    if (model->vtable == nullptr) {
        model->vtable = codec_model_vtable_for_arch(model->arch);
    }
    if (model->vtable != nullptr && model->vtable->init != nullptr) {
        return model->vtable->init(model);
    }
    model->sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 0);
    model->has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", false);
    model->has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", false);
    model->hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 0);
    model->n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 0);
    model->codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 0);
    model->n_fft = codec_read_i32_kv(model->gguf, "codec.n_fft", -1);
    model->win_length = codec_read_i32_kv(model->gguf, "codec.win_length", -1);
    model->n_mels = codec_read_i32_kv(model->gguf, "codec.n_mels", -1);
    model->latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", -1);
    return CODEC_STATUS_SUCCESS;
}
static enum codec_status codec_dispatch_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || ctx->model->vtable == nullptr ||
        ctx->model->vtable->encode == nullptr) {
        codec_context_set_error(ctx, "codec_encode not implemented for this architecture");
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return ctx->model->vtable->encode(ctx, pcm, out_tokens, out_latent, params);
}

static enum codec_status codec_dispatch_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    if (ctx == nullptr || ctx->model == nullptr || ctx->model->vtable == nullptr ||
        ctx->model->vtable->decode == nullptr) {
        codec_context_set_error(ctx, "codec_decode not implemented for this architecture");
        return CODEC_STATUS_NOT_SUPPORTED;
    }
    return ctx->model->vtable->decode(ctx, tokens, out_pcm, params);
}

struct codec_model_params codec_model_default_params(void) {
    struct codec_model_params result = {
        /*.use_gpu   =*/ false,
        /*.n_threads =*/ 0,
    };

    return result;
}

struct codec_context_params codec_context_default_params(void) {
    struct codec_context_params result = {
        /*.seed =*/ CODEC_DEFAULT_SEED,
    };

    return result;
}

struct codec_encode_params codec_encode_default_params(void) {
    struct codec_encode_params result = {
        /*.n_threads =*/ 0,
        /*.frame_size =*/ 0,
        /*.hop_size =*/ 0,
        /*.n_q =*/ 0,
    };

    return result;
}

struct codec_decode_params codec_decode_default_params(void) {
    struct codec_decode_params result = {
        /*.n_threads =*/ 0,
        /*.n_q =*/ 0,
    };

    return result;
}

struct codec_model * codec_model_load_from_file(const char * path_model, struct codec_model_params params) {
    if (path_model == nullptr) {
        return nullptr;
    }

    struct lm_ggml_context * weights = nullptr;
    struct lm_gguf_init_params lm_gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &weights,
    };

    struct lm_gguf_context * gf = lm_gguf_init_from_file(path_model, lm_gguf_params);
    if (gf == nullptr) {
        return nullptr;
    }

    codec_model * model = new (std::nothrow) codec_model();
    if (model == nullptr) {
        lm_gguf_free(gf);
        lm_ggml_free(weights);
        return nullptr;
    }

    model->gguf = gf;
    model->weights = weights;
    model->metadata = { nullptr, 0 };
    model->arch = CODEC_ARCH_UNKNOWN;
    model->name = "unknown";
    model->n_tensors = lm_gguf_get_n_tensors(gf);
    model->use_gpu = params.use_gpu;
    model->n_threads = params.n_threads > 0 ? params.n_threads : 1;
    model->backend = codec_backend_init(params.use_gpu);
    if (model->backend == nullptr) {
        codec_model_free(model);
        return nullptr;
    }
    codec_backend_set_n_threads(model->backend, model->n_threads);
    model->buffer_type = lm_ggml_backend_get_default_buffer_type(model->backend);
    if (model->buffer_type == nullptr) {
        codec_model_free(model);
        return nullptr;
    }
    model->weights_buffer = lm_ggml_backend_alloc_ctx_tensors(model->weights, model->backend);
    if (model->weights_buffer == nullptr) {
        codec_model_free(model);
        return nullptr;
    }
    lm_ggml_backend_buffer_set_usage(model->weights_buffer, LM_GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const int64_t n_tensors = lm_gguf_get_n_tensors(gf);
    const size_t data_offset = lm_gguf_get_data_offset(gf);

    std::ifstream model_file(path_model, std::ios::binary);
    if (!model_file.is_open()) {
        codec_model_free(model);
        return nullptr;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = lm_gguf_get_tensor_name(gf, i);
        if (name == nullptr) {
            continue;
        }

        struct lm_ggml_tensor * t = lm_ggml_get_tensor(model->weights, name);
        if (t == nullptr) {
            continue;
        }

        const size_t t_offset = lm_gguf_get_tensor_offset(gf, i);
        const size_t t_size = lm_ggml_nbytes(t);
        if (t_offset > std::numeric_limits<size_t>::max() - data_offset) {
            codec_model_free(model);
            return nullptr;
        }
        const size_t file_pos = data_offset + t_offset;

        if (file_pos > (size_t) std::numeric_limits<std::streamoff>::max() ||
            t_size > (size_t) std::numeric_limits<std::streamsize>::max()) {
            codec_model_free(model);
            return nullptr;
        }

        model_file.seekg((std::streamoff) file_pos, std::ios::beg);
        if (!model_file.good()) {
            codec_model_free(model);
            return nullptr;
        }

        std::vector<uint8_t> temp_data(t_size);
        if (t_size > 0) {
            model_file.read(reinterpret_cast<char *>(temp_data.data()), (std::streamsize) t_size);
            if (!model_file.good()) {
                codec_model_free(model);
                return nullptr;
            }

            lm_ggml_backend_tensor_set(t, temp_data.data(), 0, t_size);
        }
    }

    model->sample_rate = 0;
    model->has_encoder = false;
    model->has_decoder = false;
    model->hop_size = 0;
    model->n_q = 0;
    model->codebook_size = 0;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = -1;

    const int arch_id = lm_gguf_find_key(gf, "general.architecture");
    if (arch_id >= 0 && lm_gguf_get_kv_type(gf, arch_id) == LM_GGUF_TYPE_STRING) {
        const char * arch = lm_gguf_get_val_str(gf, arch_id);
        if (arch != nullptr) {
            model->arch = codec_arch_from_string(arch);
        }
    }

    model->vtable = codec_model_vtable_for_arch(model->arch);
    if (model->vtable != nullptr && model->vtable->create_impl != nullptr) {
        model->impl = model->vtable->create_impl();
        if (model->impl == nullptr) {
            codec_model_free(model);
            return nullptr;
        }
    }

    const int name_id = lm_gguf_find_key(gf, "general.name");
    if (name_id >= 0 && lm_gguf_get_kv_type(gf, name_id) == LM_GGUF_TYPE_STRING) {
        const char * name = lm_gguf_get_val_str(gf, name_id);
        if (name != nullptr) {
            model->name = name;
        }
    }

    codec_collect_lm_gguf_metadata(model);
    const enum codec_status init_st = codec_model_init_arch(model);
    if (init_st != CODEC_STATUS_SUCCESS) {
        codec_model_free(model);
        return nullptr;
    }

    return model;
}

void codec_model_free(struct codec_model * model) {
    if (model == nullptr) {
        return;
    }

    codec_metadata_free(&model->metadata);

    if (model->vtable != nullptr && model->vtable->destroy_impl != nullptr && model->impl != nullptr) {
        model->vtable->destroy_impl(model->impl);
        model->impl = nullptr;
    }

    if (model->weights_buffer != nullptr) {
        lm_ggml_backend_buffer_free(model->weights_buffer);
        model->weights_buffer = nullptr;
    }

    if (model->backend != nullptr) {
        lm_ggml_backend_free(model->backend);
        model->backend = nullptr;
    }

    if (model->gguf != nullptr) {
        lm_gguf_free(model->gguf);
    }

    if (model->weights != nullptr) {
        lm_ggml_free(model->weights);
    }

    delete model;
}

struct codec_context * codec_init_from_model(struct codec_model * model, struct codec_context_params params) {
    if (model == nullptr) {
        return nullptr;
    }

    codec_context * ctx = new (std::nothrow) codec_context();
    if (ctx == nullptr) {
        return nullptr;
    }

    ctx->model = model;
    ctx->backend = model->backend;
    ctx->params = params;
    ctx->last_error.clear();

    std::string err;
    if (!codec_runtime_init(ctx, &err)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void codec_free(struct codec_context * ctx) {
    codec_runtime_free(ctx);
    delete ctx;
}

static enum codec_status codec_encode_impl(
    struct codec_context * ctx,
    const struct codec_audio * audio,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || audio == nullptr || out_tokens == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_token_buffer_free(out_tokens);
    if (out_latent != nullptr) {
        codec_latent_buffer_free(out_latent);
    }
    codec_context_set_error(ctx, "");

    if (ctx->model->sample_rate > 0 && audio->sample_rate != ctx->model->sample_rate) {
        std::ostringstream oss;
        oss << "sample_rate mismatch: got " << audio->sample_rate << ", expected " << ctx->model->sample_rate
            << " (resample before codec_encode)";
        codec_context_set_error(ctx, oss.str());
        return CODEC_STATUS_INVALID_ARG;
    }

    std::vector<float> mono;
    std::string prep_error;
    if (!codec_prepare_mono_f32(audio, &mono, &prep_error)) {
        codec_context_set_error(ctx, prep_error);
        return CODEC_STATUS_INVALID_ARG;
    }

    if (params.n_threads <= 0) {
        params.n_threads = ctx->model->n_threads;
    }

    return codec_dispatch_encode(ctx, mono, out_tokens, out_latent, params);
}

enum codec_status codec_encode(
    struct codec_context * ctx,
    const struct codec_audio * audio,
    struct codec_token_buffer * out_tokens,
    struct codec_encode_params params) {

    return codec_encode_impl(ctx, audio, out_tokens, nullptr, params);
}

enum codec_status codec_encode_latent(
    struct codec_context * ctx,
    const struct codec_audio * audio,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    if (out_latent == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    return codec_encode_impl(ctx, audio, out_tokens, out_latent, params);
}

enum codec_status codec_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || tokens == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_pcm_buffer_free(out_pcm);
    codec_context_set_error(ctx, "");

    if (tokens->sample_rate > 0 && ctx->model->sample_rate > 0 && tokens->sample_rate != ctx->model->sample_rate) {
        std::ostringstream oss;
        oss << "token sample_rate mismatch: got " << tokens->sample_rate << ", expected " << ctx->model->sample_rate;
        codec_context_set_error(ctx, oss.str());
        return CODEC_STATUS_INVALID_ARG;
    }

    if (params.n_threads <= 0) {
        params.n_threads = ctx->model->n_threads;
    }

    return codec_dispatch_decode(ctx, tokens, out_pcm, params);
}

enum codec_status codec_decode_quantized_representation(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_pcm_buffer_free(out_pcm);
    codec_context_set_error(ctx, "");

    if (params.n_threads <= 0) {
        params.n_threads = ctx->model->n_threads;
    }

    if (ctx->model->vtable == nullptr || ctx->model->vtable->decode_latent == nullptr) {
        codec_context_set_error(ctx, "codec_decode_quantized_representation is not implemented for this architecture");
        return CODEC_STATUS_NOT_SUPPORTED;
    }

    return ctx->model->vtable->decode_latent(ctx, quantized_representation, latent_dim, n_frames, out_pcm, params);
}

enum codec_status codec_decode_batch(
    struct codec_context * ctx,
    const struct codec_batch * batch,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    if (ctx == nullptr || ctx->model == nullptr || batch == nullptr || out_pcm == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_context_set_error(ctx, "");

    if (batch->n_seq < 0 || batch->n_seq_alloc < 0 || batch->n_seq > batch->n_seq_alloc || batch->n_seq_max <= 0) {
        codec_context_set_error(ctx, "invalid batch shape");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (batch->n_seq == 0) {
        return CODEC_STATUS_SUCCESS;
    }
    if (batch->seq_id == nullptr || batch->n_frames == nullptr || batch->n_q == nullptr ||
        batch->codes_offset == nullptr || batch->latent_offset == nullptr) {
        codec_context_set_error(ctx, "batch metadata buffers are null");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (batch->mode != CODEC_BATCH_MODE_CODES && batch->mode != CODEC_BATCH_MODE_LATENT) {
        codec_context_set_error(ctx, "invalid batch mode");
        return CODEC_STATUS_INVALID_ARG;
    }
    if (batch->mode == CODEC_BATCH_MODE_CODES) {
        if (batch->codes == nullptr || batch->codes_size <= 0 || batch->codes_used < 0 || batch->codes_used > batch->codes_size) {
            codec_context_set_error(ctx, "invalid codes batch storage");
            return CODEC_STATUS_INVALID_ARG;
        }
    } else {
        if (batch->latent == nullptr || batch->latent_dim <= 0 || batch->latent_size <= 0 ||
            batch->latent_used < 0 || batch->latent_used > batch->latent_size) {
            codec_context_set_error(ctx, "invalid latent batch storage");
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    enum codec_status status = CODEC_STATUS_SUCCESS;
    int32_t n_done = 0;

    int32_t i = 0;
    for (; i < batch->n_seq; ++i) {
        if (batch->seq_id[i] < 0 || batch->seq_id[i] >= batch->n_seq_max) {
            codec_context_set_error(ctx, "batch seq_id out of range");
            status = CODEC_STATUS_INVALID_ARG;
            break;
        }
        if (batch->n_frames[i] <= 0) {
            codec_context_set_error(ctx, "batch n_frames must be > 0");
            status = CODEC_STATUS_INVALID_ARG;
            break;
        }

        if (batch->mode == CODEC_BATCH_MODE_CODES) {
            if (batch->n_q[i] <= 0) {
                codec_context_set_error(ctx, "batch n_q must be > 0 in codes mode");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }

            int32_t seq_codes = 0;
            if (!codec_safe_mul_i32(batch->n_frames[i], batch->n_q[i], &seq_codes)) {
                codec_context_set_error(ctx, "batch codes size overflow");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }
            if (batch->codes_offset[i] < 0 || batch->codes_offset[i] % (int32_t)sizeof(int32_t) != 0) {
                codec_context_set_error(ctx, "invalid batch codes offset");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }
            const int32_t start = batch->codes_offset[i] / (int32_t)sizeof(int32_t);
            int32_t end = 0;
            if (!codec_safe_add_i32(start, seq_codes, &end) || start < 0 || end > batch->codes_used) {
                codec_context_set_error(ctx, "batch codes offset/size out of range");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }

            codec_pcm_buffer_free(&out_pcm[i]);

            struct codec_token_buffer tokens;
            codec_token_buffer_reset(&tokens);
            tokens.data = batch->codes + start;
            tokens.n_tokens = seq_codes;
            tokens.n_frames = batch->n_frames[i];
            tokens.n_q = batch->n_q[i];
            tokens.codebook_size = ctx->model->codebook_size;
            tokens.sample_rate = batch->sample_rate > 0 ? batch->sample_rate : ctx->model->sample_rate;
            tokens.hop_size = batch->hop_size > 0 ? batch->hop_size : ctx->model->hop_size;

            const enum codec_status st = codec_decode(ctx, &tokens, &out_pcm[i], params);
            if (st != CODEC_STATUS_SUCCESS) {
                status = st;
                break;
            }
            n_done = i + 1;
        } else {
            int32_t seq_latent = 0;
            if (!codec_safe_mul_i32(batch->n_frames[i], batch->latent_dim, &seq_latent)) {
                codec_context_set_error(ctx, "batch latent size overflow");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }
            if (batch->latent_offset[i] < 0 || batch->latent_offset[i] % (int32_t)sizeof(float) != 0) {
                codec_context_set_error(ctx, "invalid batch latent offset");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }
            const int32_t start = batch->latent_offset[i] / (int32_t)sizeof(float);
            int32_t end = 0;
            if (!codec_safe_add_i32(start, seq_latent, &end) || start < 0 || end > batch->latent_used) {
                codec_context_set_error(ctx, "batch latent offset/size out of range");
                status = CODEC_STATUS_INVALID_ARG;
                break;
            }

            codec_pcm_buffer_free(&out_pcm[i]);

            const enum codec_status st = codec_decode_quantized_representation(
                ctx,
                batch->latent + start,
                batch->latent_dim,
                batch->n_frames[i],
                &out_pcm[i],
                params);
            if (st != CODEC_STATUS_SUCCESS) {
                status = st;
                break;
            }
            n_done = i + 1;
        }
    }

    if (status != CODEC_STATUS_SUCCESS) {
        for (int32_t j = 0; j < n_done; ++j) {
            codec_pcm_buffer_free(&out_pcm[j]);
        }
        return status;
    }

    return CODEC_STATUS_SUCCESS;
}

void codec_token_buffer_free(struct codec_token_buffer * tokens) {
    if (tokens == nullptr) {
        return;
    }

    std::free(tokens->data);
    codec_token_buffer_reset(tokens);
}

void codec_pcm_buffer_free(struct codec_pcm_buffer * pcm) {
    if (pcm == nullptr) {
        return;
    }

    std::free(pcm->data);
    codec_pcm_buffer_reset(pcm);
}

void codec_latent_buffer_free(struct codec_latent_buffer * latent) {
    if (latent == nullptr) {
        return;
    }

    std::free(latent->data);
    codec_latent_buffer_reset(latent);
}

const char * codec_get_last_error(const struct codec_context * ctx) {
    if (ctx == nullptr || ctx->last_error.empty()) {
        return "";
    }

    return ctx->last_error.c_str();
}

enum codec_arch codec_model_arch(const struct codec_model * model) {
    return model ? model->arch : CODEC_ARCH_UNKNOWN;
}

const char * codec_model_name(const struct codec_model * model) {
    return model ? model->name.c_str() : "unknown";
}

int32_t codec_model_n_tensors(const struct codec_model * model) {
    return model ? model->n_tensors : 0;
}

int32_t codec_model_sample_rate(const struct codec_model * model) {
    return model ? model->sample_rate : 0;
}

bool codec_model_has_encoder(const struct codec_model * model) {
    return model ? model->has_encoder : false;
}

bool codec_model_has_decoder(const struct codec_model * model) {
    return model ? model->has_decoder : false;
}

int32_t codec_model_n_q(const struct codec_model * model) {
    return model ? model->n_q : 0;
}

int32_t codec_model_codebook_size(const struct codec_model * model) {
    return model ? model->codebook_size : 0;
}

int32_t codec_model_hop_size(const struct codec_model * model) {
    return model ? model->hop_size : 0;
}

int32_t codec_model_n_fft(const struct codec_model * model) {
    return model ? model->n_fft : -1;
}

int32_t codec_model_win_length(const struct codec_model * model) {
    return model ? model->win_length : -1;
}

int32_t codec_model_n_mels(const struct codec_model * model) {
    return model ? model->n_mels : -1;
}

int32_t codec_model_latent_dim(const struct codec_model * model) {
    return model ? model->latent_dim : -1;
}

const struct codec_lm_gguf_metadata * codec_model_metadata(const struct codec_model * model) {
    return model ? &model->metadata : nullptr;
}
