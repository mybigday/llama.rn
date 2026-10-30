// CosyVoice LM-adaptor helpers (host orchestration around the llama.cpp
// backbone). The speech head / speech embedding are the generic
// parallel_heads_delay pair (lm.heads_0 / lm.audio_embd_0); this file adds
// the CosyVoice-specific prompt rows:
//   * sos / task_id (lm.cosyvoice.llm_embd rows for v1/v2, speech-table rows for v3)
//   * v1 only: text embedding + 6-layer rel-pos Conformer text encoder +
//     affine, and the x-vector affine row (TransformerLM.inference), run as
//     one cached graph.
// Reference: FunAudioLLM/CosyVoice cosyvoice/llm/llm.py (TransformerLM,
// Qwen2LM, CosyVoice3LM) and scripts/converters/lm_adaptor/cosyvoice.py.

#include "lm_internal.h"

#include "../blocks/conformer.h"
#include "../ops/ggml_ops.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <ggml-backend.h>

#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

constexpr int32_t kSpkDim = 192;

struct cosyvoice_lm_state {
    codec_lm_cosyvoice_info info = {};
    std::vector<int32_t> silent;
    std::string prompt_prefix;
    std::vector<float> llm_embd;           // [2 * hidden] (v1 / v2)
    codec_conformer_params tenc;
    int32_t tenc_layers = 0;
    codec_context * ctx = nullptr;         // v1 text-graph runtime
};

cosyvoice_lm_state * get_state(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr) return nullptr;
    if (lm->cosyvoice != nullptr) return static_cast<cosyvoice_lm_state *>(lm->cosyvoice);
    gguf_context * g = lm->codec->gguf;
    if (g == nullptr || gguf_find_key(g, "codec.lm.cosyvoice.version") < 0) return nullptr;

    auto * s = new (std::nothrow) cosyvoice_lm_state();
    if (s == nullptr) return nullptr;
    auto & i = s->info;
    const char * p = "codec.lm.cosyvoice.";
    auto key = [&](const char * k) { return std::string(p) + k; };
    i.version = codec_read_i32_kv(g, key("version").c_str(), 2);
    i.hidden_dim = lm->info.hidden_dim;
    i.speech_token_size = codec_read_i32_kv(g, key("speech_token_size").c_str(), 6561);
    i.n_vocab = lm->info.codebook_sizes != nullptr ? lm->info.codebook_sizes[0] : 0;
    i.special_in_speech_table = codec_read_bool_kv(g, key("special_in_speech_table").c_str(), false) ? 1 : 0;
    i.sos = codec_read_i32_kv(g, key("sos").c_str(), 0);
    i.task_id = codec_read_i32_kv(g, key("task_id").c_str(), 1);
    i.stop_begin = codec_read_i32_kv(g, key("stop_begin").c_str(), i.speech_token_size);
    i.stop_end = codec_read_i32_kv(g, key("stop_end").c_str(), i.speech_token_size + 1);
    i.ignore_eos_id = codec_read_i32_kv(g, key("ignore_eos_id").c_str(), i.speech_token_size);
    i.text_in_backbone = codec_read_bool_kv(g, key("text_in_backbone").c_str(), true) ? 1 : 0;
    i.min_token_text_ratio = codec_read_f32_kv(g, key("min_token_text_ratio").c_str(), 2.0f);
    i.max_token_text_ratio = codec_read_f32_kv(g, key("max_token_text_ratio").c_str(), 20.0f);
    codec_read_i32_array_kv_vec(g, key("silent_tokens").c_str(), &s->silent);
    i.n_silent_tokens = (int32_t) s->silent.size();
    i.silent_tokens = s->silent.empty() ? nullptr : s->silent.data();
    i.max_silent_run = codec_read_i32_kv(g, key("max_silent_run").c_str(), 0);
    s->prompt_prefix = codec_read_str_kv(g, key("prompt_prefix").c_str(), "");
    i.prompt_prefix = s->prompt_prefix.c_str();
    i.bopomofo_frontend = codec_read_bool_kv(g, key("bopomofo_frontend").c_str(), false) ? 1 : 0;
    i.has_text_encoder = 0;

    if (ggml_tensor * le = ggml_get_tensor(lm->codec->weights, "lm.cosyvoice.llm_embd")) {
        // sos / task_id rows, handed to the backbone as input embeddings (a row lookup, any stored type)
        if (le->ne[0] == i.hidden_dim && le->ne[1] >= 2 && !codec_tensor_as_vec_f32(le, &s->llm_embd)) {
            s->llm_embd.clear();
        }
    }
    if (ggml_get_tensor(lm->codec->weights, "lm.cosyvoice.text_embd") != nullptr) {
        i.has_text_encoder = 1;
        s->tenc.hidden = codec_read_i32_kv(g, key("tenc.hidden").c_str(), 1024);
        s->tenc.n_heads = codec_read_i32_kv(g, key("tenc.heads").c_str(), 16);
        s->tenc.block_eps = codec_read_f32_kv(g, key("tenc.block_eps").c_str(), 1e-12f);
        s->tenc_layers = codec_read_i32_kv(g, key("tenc.layers").c_str(), 6);
        // text_encoder(..., decoding_chunk_size=1) with static_chunk_size 1: causal.
        s->tenc.causal = codec_read_bool_kv(g, key("tenc.causal").c_str(), true);
    }
    lm->cosyvoice = s;
    return s;
}

struct text_build {
    int32_t n_text = 0;
    int32_t with_spk = 0;
    const codec_model * model = nullptr;
    const cosyvoice_lm_state * st = nullptr;
};

// [spk_aff(F.normalize(xvec))] ++ tenc_aff(text_encoder(text_embd[ids])) → [hidden, rows].
bool build_text(ggml_context * ctx, void * user_data, ggml_tensor ** out) {
    const auto * b = static_cast<const text_build *>(user_data);
    auto W = [&](const std::string & n) { return codec_graph_weight(ctx, b->model, "lm.cosyvoice." + n); };
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b->n_text);
    ggml_set_name(ids, "cv.lm.text_ids");
    ggml_set_input(ids);
    ggml_tensor * te = W("text_embd");
    if (te == nullptr) return false;
    ggml_tensor * h = codec_block_conformer_embed(ctx, b->model, "lm.cosyvoice.tenc.embed", b->st->tenc,
                                                  ggml_get_rows(ctx, te, ids));
    h = codec_block_conformer_stack(ctx, b->model, "lm.cosyvoice.tenc.blk", b->st->tenc, h, b->st->tenc_layers);
    if (h == nullptr) return false;
    h = codec_op_layer_norm_tc(ctx, h, b->st->tenc.embed_eps, W("tenc.after_norm.w"), W("tenc.after_norm.b"));
    if (h == nullptr) return false;
    ggml_tensor * rows = codec_op_linear(ctx, ggml_cont(ctx, ggml_transpose(ctx, h)), W("tenc_aff.w"), W("tenc_aff.b"));
    if (rows == nullptr) return false;
    if (b->with_spk) {
        ggml_tensor * xv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kSpkDim, 1);
        ggml_set_name(xv, "cv.lm.xvec");
        ggml_set_input(xv);
        ggml_tensor * xn = ggml_scale(ctx, ggml_rms_norm(ctx, xv, 1e-12f), 1.0f / std::sqrt((float) kSpkDim));
        ggml_tensor * sr = codec_op_linear(ctx, xn, W("spk_aff.w"), W("spk_aff.b"));
        if (sr == nullptr) return false;
        rows = ggml_concat(ctx, sr, rows, 1);
    }
    rows = ggml_cont(ctx, rows);
    ggml_set_name(rows, "cv.lm.rows");
    *out = rows;
    return true;
}

}  // namespace

void codec_lm_cosyvoice_free_state(struct codec_lm * lm) {
    if (lm == nullptr || lm->cosyvoice == nullptr) return;
    auto * s = static_cast<cosyvoice_lm_state *>(lm->cosyvoice);
    if (s->ctx != nullptr) {
        codec_runtime_free(s->ctx);
        delete s->ctx;
    }
    delete s;
    lm->cosyvoice = nullptr;
}

const struct codec_lm_cosyvoice_info * codec_lm_cosyvoice_get_info(struct codec_lm * lm) {
    cosyvoice_lm_state * s = get_state(lm);
    return s == nullptr ? nullptr : &s->info;
}

enum codec_status codec_lm_cosyvoice_special_embd(struct codec_lm * lm, int32_t which, float * out, int32_t out_cap) {
    cosyvoice_lm_state * s = get_state(lm);
    if (s == nullptr) return CODEC_STATUS_NOT_SUPPORTED;
    const int32_t H = s->info.hidden_dim;
    if (out == nullptr || out_cap < H || (which != 0 && which != 1)) return CODEC_STATUS_INVALID_ARG;
    const int32_t row = which == 0 ? s->info.sos : s->info.task_id;
    if (s->info.special_in_speech_table) {
        const int32_t codes[1] = {row};
        return codec_lm_compose_audio_embd(lm, codes, out);
    }
    if ((int64_t) s->llm_embd.size() < (int64_t) (row + 1) * H) {
        lm->last_error = "cosyvoice: missing lm.cosyvoice.llm_embd";
        return CODEC_STATUS_INVALID_STATE;
    }
    std::memcpy(out, s->llm_embd.data() + (size_t) row * H, (size_t) H * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_cosyvoice_speech_embd(
        struct codec_lm * lm, const int32_t * codes, int32_t n, float * out, int32_t out_cap_rows) {
    cosyvoice_lm_state * s = get_state(lm);
    if (s == nullptr) return CODEC_STATUS_NOT_SUPPORTED;
    if ((n > 0 && (codes == nullptr || out == nullptr)) || out_cap_rows < n) return CODEC_STATUS_INVALID_ARG;
    const int32_t H = s->info.hidden_dim;
    for (int32_t i = 0; i < n; ++i) {
        if (codes[i] < 0 || codes[i] >= s->info.n_vocab) {
            lm->last_error = "cosyvoice: speech token out of range";
            return CODEC_STATUS_INVALID_ARG;
        }
        const enum codec_status st = codec_lm_compose_audio_embd(lm, codes + i, out + (size_t) i * H);
        if (st != CODEC_STATUS_SUCCESS) return st;
    }
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_cosyvoice_text_rows(
        struct codec_lm * lm, const int32_t * text_ids, int32_t n_text, const float * xvec, int32_t xvec_dim,
        float * out, int32_t out_cap_rows, int32_t * out_rows) {
    cosyvoice_lm_state * s = get_state(lm);
    if (s == nullptr || !s->info.has_text_encoder) return CODEC_STATUS_NOT_SUPPORTED;
    if (text_ids == nullptr || n_text <= 0 || out == nullptr || (xvec != nullptr && xvec_dim != kSpkDim)) {
        return CODEC_STATUS_INVALID_ARG;
    }
    const int32_t rows = n_text + (xvec != nullptr ? 1 : 0);
    if (out_cap_rows < rows) return CODEC_STATUS_INVALID_ARG;
    std::string err;
    if (s->ctx == nullptr) {
        auto * c = new (std::nothrow) codec_context();
        if (c == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
        c->model = lm->codec;
        c->backend = lm->codec->backend;
        c->params = codec_context_default_params();
        if (!codec_runtime_init(c, &err)) {
            delete c;
            lm->last_error = "cosyvoice text graph: " + err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        s->ctx = c;
    }
    text_build build;
    build.n_text = n_text;
    build.with_spk = xvec != nullptr ? 1 : 0;
    build.model = lm->codec;
    build.st = s;
    codec_graph_eval_guard guard(s->ctx);
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key k = {};
    k.kind = CODEC_GRAPH_LM_COSYVOICE_TEXT;
    k.n_frames = n_text;
    k.n_q = build.with_spk;
    if (!codec_graph_cache_get_or_build(s->ctx, k, build_text, &build, sizeof(build), &entry, &err)) {
        lm->last_error = "cosyvoice text graph: " + err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    ggml_tensor * t_ids = codec_graph_get_tensor(s->ctx, entry, "cv.lm.text_ids");
    ggml_tensor * t_xv = xvec != nullptr ? codec_graph_get_tensor(s->ctx, entry, "cv.lm.xvec") : nullptr;
    ggml_tensor * t_out = codec_graph_get_tensor(s->ctx, entry, "cv.lm.rows");
    if (t_ids == nullptr || t_out == nullptr || (xvec != nullptr && t_xv == nullptr) ||
        !codec_graph_prepare_io(s->ctx, entry, &err) ||
        !codec_runtime_write_tensor(t_ids, text_ids, (size_t) n_text * sizeof(int32_t), &err) ||
        (t_xv != nullptr && !codec_runtime_write_tensor(t_xv, xvec, (size_t) kSpkDim * sizeof(float), &err)) ||
        !codec_graph_compute(s->ctx, entry, lm->codec->n_threads > 0 ? lm->codec->n_threads : 1, &err) ||
        !codec_runtime_read_tensor(t_out, out, (size_t) rows * s->info.hidden_dim * sizeof(float), &err)) {
        lm->last_error = "cosyvoice text graph: " + (err.empty() ? std::string("missing graph tensors") : err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (out_rows != nullptr) *out_rows = rows;
    return CODEC_STATUS_SUCCESS;
}
