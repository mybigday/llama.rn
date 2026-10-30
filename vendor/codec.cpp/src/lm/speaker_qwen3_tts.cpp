#include "speaker_qwen3_tts.h"

#include "../blocks/ecapa_tdnn.h"
#include "../runtime/graph.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>

#include <new>
#include <string>
#include <vector>

// =====================================================================
// Qwen3-TTS ECAPA-TDNN speaker encoder: one cached graph per call
// (src/blocks/ecapa_tdnn), ref PCM (24 kHz mono) → (1, 1024) x-vector.
// =====================================================================

namespace {

struct qwen3_tts_speaker_impl {
    codec_ecapa_params ecapa;
    int32_t sample_rate = 24000;
    int32_t n_rows      = 1;
    int32_t hidden_dim  = 1024;
    codec_context * ctx = nullptr;   // graph runtime, created on first encode
};

struct qwen3_spk_build {
    codec_ecapa_params ecapa;
    int32_t n_samples = 0;
    const codec_model * model = nullptr;
};

bool build_qwen3_spk_graph(ggml_context * ctx_eval, void * ud, ggml_tensor ** out) {
    const auto * b = static_cast<const qwen3_spk_build *>(ud);
    if (ctx_eval == nullptr || b == nullptr || out == nullptr) return false;
    ggml_tensor * pcm = ggml_new_tensor_1d(ctx_eval, GGML_TYPE_F32, b->n_samples);
    ggml_set_name(pcm, "spk.in.pcm");
    ggml_tensor * e = codec_block_ecapa_tdnn(ctx_eval, b->model, "speaker.qwen3_tts", b->ecapa, pcm);
    if (e == nullptr) return false;
    ggml_set_name(e, "spk.out.xvector");
    *out = e;
    return true;
}

}  // namespace

bool qwen3_tts_speaker_init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr ||
        lm->codec->gguf == nullptr || lm->codec->weights == nullptr) {
        if (lm) lm->last_error = "speaker(qwen3_tts): codec has no gguf/weights";
        return false;
    }
    gguf_context * gf = lm->codec->gguf;

    auto * impl = new (std::nothrow) qwen3_tts_speaker_impl();
    if (impl == nullptr) {
        lm->last_error = "speaker(qwen3_tts): out of memory";
        return false;
    }
    codec_ecapa_params & e = impl->ecapa;
    e.n_mels        = codec_read_i32_kv(gf, "codec.speaker.ecapa.mel_dim", e.n_mels);
    e.res2net_scale = codec_read_i32_kv(gf, "codec.speaker.ecapa.enc_res2net_scale", e.res2net_scale);
    e.n_fft         = codec_read_i32_kv(gf, "codec.speaker.ecapa.n_fft", e.n_fft);
    e.hop           = codec_read_i32_kv(gf, "codec.speaker.ecapa.hop_size", e.hop);
    impl->sample_rate = lm->speaker_info.ref_sample_rate;
    impl->n_rows      = lm->speaker_info.n_rows;
    impl->hidden_dim  = lm->speaker_info.hidden_dim;

    std::vector<int32_t> channels, kernels, dilations;
    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_channels",     &channels);
    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_kernel_sizes", &kernels);
    codec_read_i32_array_kv_vec(gf, "codec.speaker.ecapa.enc_dilations",    &dilations);
    if (channels.size() < 3 || channels.size() > (size_t) kCodecEcapaMaxBlocks ||
        channels.size() != kernels.size() || channels.size() != dilations.size()) {
        lm->last_error = "speaker(qwen3_tts): enc_channels/kernels/dilations missing or mismatched";
        delete impl; return false;
    }
    e.n_blocks = (int32_t) channels.size();
    for (size_t i = 0; i < channels.size(); ++i) {
        e.channels[i] = channels[i];
        e.kernels[i] = kernels[i];
        e.dilations[i] = dilations[i];
    }
    for (const char * t : {"speaker.qwen3_tts.mel_basis", "speaker.qwen3_tts.window",
                           "speaker.qwen3_tts.blocks.0.conv.weight", "speaker.qwen3_tts.fc.weight"}) {
        if (ggml_get_tensor(lm->codec->weights, t) == nullptr) {
            lm->last_error = std::string("speaker(qwen3_tts): missing tensor: ") + t;
            delete impl; return false;
        }
    }
    lm->speaker_impl = impl;
    return true;
}

void qwen3_tts_speaker_free(codec_lm * lm) {
    if (lm == nullptr || lm->speaker_impl == nullptr) return;
    auto * impl = static_cast<qwen3_tts_speaker_impl *>(lm->speaker_impl);
    if (impl->ctx != nullptr) {
        codec_runtime_free(impl->ctx);
        delete impl->ctx;
    }
    delete impl;
    lm->speaker_impl = nullptr;
}

enum codec_status qwen3_tts_speaker_encode(
        codec_lm * lm,
        const struct codec_audio * ref_pcm,
        const int32_t * /*ref_speech_tokens*/, int32_t /*n_ref_speech_tokens*/,
        float /*emotion*/,
        float * out, int32_t out_n_elems) {
    if (lm == nullptr || lm->speaker_impl == nullptr) {
        return CODEC_STATUS_INVALID_STATE;
    }
    auto * impl = static_cast<qwen3_tts_speaker_impl *>(lm->speaker_impl);
    if (ref_pcm == nullptr || ref_pcm->data == nullptr || ref_pcm->n_samples <= 0) {
        lm->last_error = "qwen3_tts speaker_encode: ref_pcm missing";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->pcm_type != CODEC_PCM_TYPE_F32) {
        lm->last_error = "qwen3_tts speaker_encode: ref_pcm must be PCM_TYPE_F32";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (ref_pcm->sample_rate != impl->sample_rate) {
        lm->last_error =
            "qwen3_tts speaker_encode: sample_rate mismatch (caller must "
            "resample to ref_sample_rate)";
        return CODEC_STATUS_INVALID_ARG;
    }
    if (out == nullptr || out_n_elems < impl->n_rows * impl->hidden_dim) {
        lm->last_error = "qwen3_tts speaker_encode: out buffer too small";
        return CODEC_STATUS_INVALID_ARG;
    }

    const float * pcm_ptr = static_cast<const float *>(ref_pcm->data);
    std::vector<float> pcm(pcm_ptr, pcm_ptr + (size_t) ref_pcm->n_samples * (size_t) ref_pcm->n_channels);
    if (ref_pcm->n_channels > 1) {
        std::vector<float> mono((size_t) ref_pcm->n_samples, 0.0f);
        for (int32_t i = 0; i < ref_pcm->n_samples; ++i) {
            float a = 0.0f;
            for (int32_t c = 0; c < ref_pcm->n_channels; ++c) {
                a += pcm[(size_t) i * (size_t) ref_pcm->n_channels + (size_t) c];
            }
            mono[(size_t) i] = a / (float) ref_pcm->n_channels;
        }
        pcm.swap(mono);
    }
    if (codec_ecapa_n_frames(impl->ecapa, (int64_t) pcm.size()) < 2) {
        lm->last_error = "qwen3_tts speaker_encode: too few mel frames "
                          "(ref audio shorter than (n_fft + hop) at 24 kHz)";
        return CODEC_STATUS_INVALID_ARG;
    }

    std::string err;
    if (impl->ctx == nullptr) {
        codec_context * cctx = new (std::nothrow) codec_context();
        if (cctx == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
        cctx->model   = lm->codec;
        cctx->backend = lm->codec->backend;
        cctx->params  = codec_context_default_params();
        if (!codec_runtime_init(cctx, &err)) {
            delete cctx;
            lm->last_error = "qwen3_tts speaker_encode: " + err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        impl->ctx = cctx;
    }

    qwen3_spk_build build;
    build.ecapa = impl->ecapa;
    build.n_samples = (int32_t) pcm.size();
    build.model = lm->codec;
    codec_graph_eval_guard guard(impl->ctx);
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_LM_SPEAKER_QWEN3_TTS;
    key.n_in = build.n_samples;
    if (!codec_graph_cache_get_or_build(impl->ctx, key, build_qwen3_spk_graph, &build, sizeof(build), &entry, &err)) {
        lm->last_error = "qwen3_tts speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_pcm = codec_graph_get_tensor(impl->ctx, entry, "spk.in.pcm");
    ggml_tensor * t_out = codec_graph_get_tensor(impl->ctx, entry, "spk.out.xvector");
    if (t_pcm == nullptr || t_out == nullptr || ggml_nelements(t_out) != impl->hidden_dim) {
        lm->last_error = "qwen3_tts speaker_encode: invalid graph IO";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t nt = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
    if (!codec_graph_prepare_io(impl->ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err) ||
        !codec_graph_compute(impl->ctx, entry, nt, &err) ||
        !codec_runtime_read_tensor(t_out, out, (size_t) impl->hidden_dim * sizeof(float), &err)) {
        lm->last_error = "qwen3_tts speaker_encode: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}
