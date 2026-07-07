#include "lm_internal.h"

#include "../runtime/graph.h"
#include "../runtime/graph_exec.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

// =====================================================================
// codec_lm kind: parallel_heads_delay
//
// N parallel `Linear(hidden, vocab_i)` heads off a backbone hidden
// state.  No intra-step dependency.  Audio embeddings are unfused
// per-codebook tables `lm.audio_embd_{i}.weight` of shape
// [hidden, vocab_i].  Heads are unfused `lm.heads_{i}.weight` of shape
// [hidden, vocab_i] (PyTorch `Linear(hidden, vocab)` saves weight as
// (vocab, hidden); converter dumps row-major into GGUF, ggml reads
// `ne[0] = hidden` as the contiguous axis).
//
// Graph 1 — step_begin: matmul each head against the input hidden,
//   produces N logits tensors named `lm.step.logits_{i}`.  Hidden is
//   marked input (auto-flagged) so the runtime can write_tensor it.
//
// Graph 2 — compose_audio_embd: get_rows on each per-cb embed table
//   indexed by the corresponding code, sum across N rows, output a
//   single [hidden_dim] vector named `lm.compose.out`.
//
// Both graphs are fixed-shape per (hidden_dim, n_codebook), so the
// state's codec_context graph cache hits after the first call.
// =====================================================================

namespace {

struct phd_impl {
    int32_t n_codebook = 0;
    int32_t hidden_dim = 0;
    int32_t audio_embed_dim = 0;

    // Tensor pointers into codec->weights, resolved at init.
    std::vector<lm_ggml_tensor *> heads;       // [n_codebook]
    std::vector<lm_ggml_tensor *> audio_embds; // [n_codebook]

    // Optional learned per-step positional embedding.  Chatterbox T3
    // stores `lm.chatterbox.speech_pos_emb.weight` of shape
    // (hidden, max_speech_tokens) — added on top of the audio embed
    // for the next backbone input (Type B / embed-override).  Loaded
    // opportunistically at init; absent for plain TTS variants.
    lm_ggml_tensor *      pos_emb     = nullptr;
    int32_t            pos_emb_dim = 0;   // hidden_dim of pos table
    int32_t            pos_emb_max = 0;   // max position
    std::vector<float> pos_emb_f32;       // lazy-dequanted full table; cached on first use

    // Lazily-allocated codec_context for compose_audio_embd graphs;
    // owned, freed by free_lm.  Independent of any codec_lm_state's
    // ctx so concurrent step_begin / compose calls don't fight over
    // the same eval arena.
    codec_context * compose_ctx = nullptr;
};

struct phd_state {
    // Scratch buffers for logits returned to the caller from
    // step_logits.  One contiguous buffer per codebook (sized to that
    // codebook's vocab); pointer handed out is `logits_buf[i].data()`.
    std::vector<std::vector<float>> logits_buf;
};

// Builder user_data — captured by reference into build_fn, only used
// inside codec_graph_cache_get_or_build's call window.
struct phd_logits_build_data {
    phd_impl *  impl;
    int32_t     hidden_dim;
};

struct phd_compose_build_data {
    phd_impl *  impl;
    int32_t     hidden_dim;
};

// ---------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------

bool phd_build_logits(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out_terminal) {
    phd_logits_build_data * b = static_cast<phd_logits_build_data *>(user_data);
    if (ctx_eval == nullptr || b == nullptr || b->impl == nullptr || out_terminal == nullptr) {
        return false;
    }
    phd_impl * impl = b->impl;

    // Input: backbone hidden state, shape [hidden, 1].
    lm_ggml_tensor * t_h = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, b->hidden_dim, 1);
    lm_ggml_set_name(t_h, "lm.step.h_in");

    // For each codebook, compute logits = head @ h.  lm_ggml_mul_mat
    // contracts on ne[0] of both operands; head has ne=[hidden, vocab],
    // h has ne=[hidden, 1], so result is [vocab, 1].  Mark every per-cb
    // logits tensor as a graph output so galloc keeps each row pinned
    // (only the terminal is auto-flagged by the runtime; the other N-1
    // would otherwise be reused after their last graph use).
    lm_ggml_tensor * last = nullptr;
    char buf[64];
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        lm_ggml_tensor * head = impl->heads[(size_t) i];
        // Keep stored dtype (F16/F32/BF16) — mul_mat handles all of those
        // as src[0] without an extra dequant pass.  Casting unconditionally
        // bakes a full vocab-sized F16→F32 dequant into the cached graph
        // that runs every step (Chatterbox / WavTokenizer etc. hit this
        // path per AR step × n_codebook heads).
        lm_ggml_tensor * head_in_ctx = codec_graph_mat_lhs(ctx_eval, head);
        lm_ggml_tensor * logits = lm_ggml_mul_mat(ctx_eval, head_in_ctx, t_h);
        std::snprintf(buf, sizeof(buf), "lm.step.logits_%d", i);
        lm_ggml_set_name(logits, buf);
        lm_ggml_set_output(logits);
        last = logits;
    }
    if (last == nullptr) {
        return false;
    }
    *out_terminal = last;
    return true;
}

bool phd_build_compose(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out_terminal) {
    phd_compose_build_data * b = static_cast<phd_compose_build_data *>(user_data);
    if (ctx_eval == nullptr || b == nullptr || b->impl == nullptr || out_terminal == nullptr) {
        return false;
    }
    phd_impl * impl = b->impl;

    // Input: codes for each codebook, [n_codebook] i32.
    lm_ggml_tensor * t_codes = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_I32, impl->n_codebook);
    lm_ggml_set_name(t_codes, "lm.compose.codes");

    // Sum across n_codebook get_rows results.  lm_ggml_get_rows takes a
    // 1D index tensor; we view a single element of t_codes per call.
    lm_ggml_tensor * acc = nullptr;
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        lm_ggml_tensor * embd = impl->audio_embds[(size_t) i];
        lm_ggml_tensor * idx_view = lm_ggml_view_1d(
            ctx_eval, t_codes, /*ne0=*/1, /*offset=*/(size_t) i * sizeof(int32_t));
        lm_ggml_tensor * row = lm_ggml_get_rows(ctx_eval, embd, idx_view);
        // row has ne=[hidden, 1]; cast to F32 if the embed table is
        // F16/quantized.  lm_ggml_get_rows itself dequants for some types,
        // but be explicit so the accumulator type is always F32.
        row = codec_graph_cast_f32(ctx_eval, row);
        if (acc == nullptr) {
            acc = row;
        } else {
            acc = lm_ggml_add(ctx_eval, acc, row);
        }
    }
    if (acc == nullptr) {
        return false;
    }
    lm_ggml_set_name(acc, "lm.compose.out");
    *out_terminal = acc;
    return true;
}

// ---------------------------------------------------------------------
// init / free
// ---------------------------------------------------------------------

bool init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr || lm->codec->weights == nullptr) {
        return false;
    }

    // Heads can be either explicitly stored as `lm.heads_{i}.weight` or
    // tied to `lm.audio_embd_{i}.weight` (tie_word_embeddings-style),
    // which saves the cost of duplicating the c0 vocab table.  The
    // converter declares which via metadata.
    const bool tied_heads = codec_read_bool_kv(
        lm->codec->gguf, "codec.lm.parallel.tied_heads_to_embd", false);

    if (!codec_lm_check_unfused_audio_tables(
            lm,
            lm->info.audio_embed_dim,
            lm->codebook_sizes_buf,
            /*tied_heads=*/tied_heads)) {
        return false;
    }

    phd_impl * impl = new (std::nothrow) phd_impl();
    if (impl == nullptr) {
        lm->last_error = "out of memory";
        return false;
    }
    impl->n_codebook      = lm->info.n_codebook;
    impl->hidden_dim      = lm->info.hidden_dim;
    impl->audio_embed_dim = lm->info.audio_embed_dim;
    impl->heads.resize((size_t) impl->n_codebook, nullptr);
    impl->audio_embds.resize((size_t) impl->n_codebook, nullptr);

    char buf[64];
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        std::snprintf(buf, sizeof(buf), "lm.audio_embd_%d.weight", i);
        lm_ggml_tensor * embd = lm_ggml_get_tensor(lm->codec->weights, buf);
        impl->audio_embds[(size_t) i] = embd;

        if (tied_heads) {
            // Same tensor, two roles.
            impl->heads[(size_t) i] = embd;
        } else {
            std::snprintf(buf, sizeof(buf), "lm.heads_%d.weight", i);
            impl->heads[(size_t) i] = lm_ggml_get_tensor(lm->codec->weights, buf);
        }
        if (impl->heads[(size_t) i] == nullptr || impl->audio_embds[(size_t) i] == nullptr) {
            // Already validated via codec_lm_check_unfused_audio_tables, but
            // be defensive in case of name format drift.
            lm->last_error = std::string("missing tensor for codebook ") + std::to_string(i);
            delete impl;
            return false;
        }
    }

    // Optional Chatterbox-style learned position embedding.  Used by
    // codec_lm_compose_next_embd for the next backbone-input embed
    // (`speech_emb[code] + speech_pos_emb[step]`).  Absent for plain TTS
    // variants — compose_next_embd then degrades to compose_audio_embd.
    impl->pos_emb = lm_ggml_get_tensor(lm->codec->weights, "lm.chatterbox.speech_pos_emb.weight");
    if (impl->pos_emb != nullptr) {
        // ne[0] = hidden, ne[1] = max_pos (PyTorch nn.Embedding saved
        // row-major (V, H) lands in ggml as ne[0]=H, ne[1]=V).
        impl->pos_emb_dim = (int32_t) impl->pos_emb->ne[0];
        impl->pos_emb_max = (int32_t) impl->pos_emb->ne[1];
        if (impl->pos_emb_dim != impl->audio_embed_dim) {
            // Mismatched dim — bail out of the optional pos path.
            impl->pos_emb     = nullptr;
            impl->pos_emb_dim = 0;
            impl->pos_emb_max = 0;
        }
    }

    lm->impl = impl;
    return true;
}

void free_lm(codec_lm * lm) {
    if (lm == nullptr || lm->impl == nullptr) {
        return;
    }
    phd_impl * impl = static_cast<phd_impl *>(lm->impl);
    if (impl->compose_ctx != nullptr) {
        codec_runtime_free(impl->compose_ctx);
        delete impl->compose_ctx;
        impl->compose_ctx = nullptr;
    }
    delete impl;
    lm->impl = nullptr;
}

// ---------------------------------------------------------------------
// state init / free / reset
// ---------------------------------------------------------------------

bool state_init(codec_lm_state * st) {
    if (st == nullptr || st->lm == nullptr || st->lm->impl == nullptr) {
        return false;
    }
    phd_impl * impl = static_cast<phd_impl *>(st->lm->impl);
    phd_state * sst = new (std::nothrow) phd_state();
    if (sst == nullptr) {
        return false;
    }
    sst->logits_buf.resize((size_t) impl->n_codebook);
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        sst->logits_buf[(size_t) i].resize((size_t) st->lm->info.codebook_sizes[i]);
    }
    st->impl = sst;
    return true;
}

void state_free(codec_lm_state * st) {
    if (st == nullptr || st->impl == nullptr) {
        return;
    }
    delete static_cast<phd_state *>(st->impl);
    st->impl = nullptr;
}

void state_reset(codec_lm_state * st) {
    // Logits buffers don't need clearing — they're always overwritten
    // by step_begin before being handed out.  No KV cache for this kind.
    (void) st;
}

// ---------------------------------------------------------------------
// step machine
// ---------------------------------------------------------------------

enum codec_status step_begin(codec_lm_state * st, const float * h_in) {
    if (st == nullptr || h_in == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    phd_impl * impl = static_cast<phd_impl *>(st->lm->impl);
    phd_state * sst = static_cast<phd_state *>(st->impl);

    phd_logits_build_data build = { impl, impl->hidden_dim };

    codec_graph_eval_guard guard(st->ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind   = CODEC_GRAPH_LM_PARALLEL_HEADS_LOGITS;
    key.n_q    = impl->n_codebook;
    key.n_in   = impl->hidden_dim;

    if (!codec_graph_cache_get_or_build(
            st->ctx,
            key,
            phd_build_logits,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        st->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_h = codec_graph_get_tensor(st->ctx, entry, "lm.step.h_in");
    if (t_h == nullptr) {
        st->last_error = "logits graph missing input tensor";
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(st->ctx, entry, &err)) {
        st->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_h, h_in, (size_t) impl->hidden_dim * sizeof(float), &err)) {
        st->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_threads = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    if (!codec_graph_compute(st->ctx, entry, n_threads, &err)) {
        st->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Copy each logits tensor into the state's per-cb scratch.
    char buf[64];
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        std::snprintf(buf, sizeof(buf), "lm.step.logits_%d", i);
        lm_ggml_tensor * t_lg = codec_graph_get_tensor(st->ctx, entry, buf);
        if (t_lg == nullptr) {
            st->last_error = std::string("logits graph missing output: ") + buf;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const int32_t vocab = st->lm->info.codebook_sizes[i];
        const size_t n_bytes = (size_t) vocab * sizeof(float);
        if (!codec_runtime_read_tensor(
                t_lg,
                sst->logits_buf[(size_t) i].data(),
                n_bytes,
                &err)) {
            st->last_error = err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }
    return CODEC_STATUS_SUCCESS;
}

bool step_pending(const codec_lm_state * st) {
    return st != nullptr && st->next_cb < st->lm->info.n_codebook;
}

const float * step_logits(codec_lm_state * st, int32_t * out_cb_idx, int32_t * out_n) {
    if (st == nullptr || st->impl == nullptr) {
        return nullptr;
    }
    phd_state * sst = static_cast<phd_state *>(st->impl);
    const int32_t cb = st->next_cb;
    if (out_cb_idx != nullptr) *out_cb_idx = cb;
    if (out_n      != nullptr) *out_n      = st->lm->info.codebook_sizes[cb];
    return sst->logits_buf[(size_t) cb].data();
}

enum codec_status step_push_code(codec_lm_state * st, int32_t code) {
    // Generic state machine in lm.cpp records `code` into st->codes_buf
    // and advances next_cb; nothing kind-specific to do here for
    // parallel_heads_delay (delay shift register applies at sequence
    // assembly time, not per-step).
    (void) st;
    (void) code;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status step_finish(codec_lm_state * st, int32_t * out_codes) {
    // Default: copy st->codes_buf into out_codes.  Generic dispatch in
    // lm.cpp handles this when the vtable's step_finish is NULL, but
    // keeping it explicit makes flow easy to follow.
    if (st == nullptr || out_codes == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    std::memcpy(out_codes, st->codes_buf.data(),
                (size_t) st->lm->info.n_codebook * sizeof(int32_t));
    return CODEC_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------
// audio embed lookup
// ---------------------------------------------------------------------

const float * audio_embd(codec_lm * lm, int32_t cb_idx, int32_t code) {
    if (lm == nullptr || lm->impl == nullptr) {
        return nullptr;
    }
    phd_impl * impl = static_cast<phd_impl *>(lm->impl);
    if (cb_idx < 0 || cb_idx >= impl->n_codebook) {
        return nullptr;
    }
    lm_ggml_tensor * embd = impl->audio_embds[(size_t) cb_idx];
    if (embd == nullptr) {
        return nullptr;
    }
    // Decode the entire table once (lazy, thread_local cached by
    // codec_tensor_data_f32 for F32 / by codec_tensor_as_vec_f32 for
    // others).  Embedding tables are typically modest size (1025 × 2048
    // floats ≈ 8 MB for MOSS-TTSD c1..c7), but channel 0 of MOSS-TTSD
    // is 152697 × 2048 ≈ 1.2 GB FP32 — for big tables, callers should
    // prefer codec_lm_compose_audio_embd which goes through a graph
    // and gets a single row.
    const float * data = codec_tensor_data_f32(embd);
    if (data == nullptr) {
        return nullptr;
    }
    const size_t hidden = (size_t) impl->audio_embed_dim;
    return data + (size_t) code * hidden;
}

enum codec_status compose_audio_embd(codec_lm * lm, const int32_t * codes, float * out_embd) {
    if (lm == nullptr || lm->impl == nullptr || codes == nullptr || out_embd == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    phd_impl * impl = static_cast<phd_impl *>(lm->impl);
    for (int32_t i = 0; i < impl->n_codebook; ++i) {
        const int32_t v = lm->info.codebook_sizes[i];
        if (codes[i] < 0 || codes[i] >= v) {
            return CODEC_STATUS_INVALID_ARG;
        }
    }

    // Per-call graph: get_rows on each cb's table, sum, read out.
    // Cached after first call (key is fixed per (hidden_dim, n_cb)).
    // We borrow a scratch codec_context for this — but compose is
    // typically called outside step_begin by the caller (post-finish),
    // so we'd race with step_begin's eval ctx if we shared.  For
    // simplicity, build a fresh codec_context-less path: use a
    // standalone lm_ggml_init + alloc + compute, all on CPU backend.
    //
    // Implementation note: we delegate to a dedicated small codec_context
    // owned by the codec_lm itself for compose graphs.  See
    // codec_lm.compose_ctx, allocated lazily on first call.
    //
    // To avoid threading state through the public lm, we keep the
    // compose_ctx inside `phd_impl` (mutable).  Single-thread per lm —
    // multiple states should each call compose on their own lm
    // pointer.  (Caller-side concurrency: if you need per-state
    // compose, build it explicitly in the caller using
    // codec_lm_audio_embd row pointers + manual sum.)
    if (impl->compose_ctx == nullptr) {
        codec_context * cctx = new (std::nothrow) codec_context();
        if (cctx == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
        cctx->model   = lm->codec;
        cctx->backend = lm->codec->backend;
        cctx->params  = codec_context_default_params();
        std::string err;
        if (!codec_runtime_init(cctx, &err)) {
            delete cctx;
            lm->last_error = err;
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        impl->compose_ctx = cctx;
    }

    phd_compose_build_data build = { impl, impl->audio_embed_dim };

    codec_graph_eval_guard guard(impl->compose_ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {};
    key.kind = CODEC_GRAPH_LM_PARALLEL_HEADS_COMPOSE;
    key.n_q  = impl->n_codebook;
    key.n_in = impl->audio_embed_dim;
    if (!codec_graph_cache_get_or_build(
            impl->compose_ctx,
            key,
            phd_build_compose,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        lm->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_codes = codec_graph_get_tensor(impl->compose_ctx, entry, "lm.compose.codes");
    lm_ggml_tensor * t_out   = codec_graph_get_tensor(impl->compose_ctx, entry, "lm.compose.out");
    if (t_codes == nullptr || t_out == nullptr) {
        lm->last_error = "compose graph missing input/output tensor";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(impl->compose_ctx, entry, &err)) {
        lm->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_codes, codes, (size_t) impl->n_codebook * sizeof(int32_t), &err)) {
        lm->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    const int32_t n_threads = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
    if (!codec_graph_compute(impl->compose_ctx, entry, n_threads, &err)) {
        lm->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_read_tensor(
            t_out, out_embd,
            (size_t) impl->audio_embed_dim * sizeof(float),
            &err)) {
        lm->last_error = err;
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

enum codec_status compose_next_embd(codec_lm * lm, const int32_t * codes,
                                     int32_t step, float * out_embd) {
    if (lm == nullptr || lm->impl == nullptr || codes == nullptr || out_embd == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }

    // Base audio embed first — same compose graph used for the normal
    // depth-decoder / no-pos paths.
    const enum codec_status rc = compose_audio_embd(lm, codes, out_embd);
    if (rc != CODEC_STATUS_SUCCESS) {
        return rc;
    }

    phd_impl * impl = static_cast<phd_impl *>(lm->impl);
    if (impl->pos_emb == nullptr || step < 0 || step >= impl->pos_emb_max) {
        // No learned pos table (plain TTS), or step out of range — just
        // return the audio embed.  Out-of-range with a table loaded is a
        // soft fail: caller's sequence overshot max_speech_tokens.  We
        // log via last_error so the host can surface it without crashing.
        if (impl->pos_emb != nullptr && step >= impl->pos_emb_max) {
            lm->last_error = "compose_next_embd: step >= max_speech_tokens — pos emb skipped";
        }
        return CODEC_STATUS_SUCCESS;
    }

    // Lazy-dequant the entire pos table once.  ~4100 × 1024 floats =
    // 16 MB; one-time hit per lm.
    if (impl->pos_emb_f32.empty()) {
        if (!codec_tensor_as_vec_f32(impl->pos_emb, &impl->pos_emb_f32)) {
            lm->last_error = "compose_next_embd: failed to dequant speech_pos_emb";
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const size_t expect = (size_t) impl->pos_emb_dim * (size_t) impl->pos_emb_max;
        if (impl->pos_emb_f32.size() != expect) {
            lm->last_error = "compose_next_embd: pos table size mismatch";
            impl->pos_emb_f32.clear();
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    const float * row = impl->pos_emb_f32.data() + (size_t) step * (size_t) impl->pos_emb_dim;
    for (int32_t i = 0; i < impl->pos_emb_dim; ++i) {
        out_embd[i] += row[i];
    }
    return CODEC_STATUS_SUCCESS;
}

}  // namespace

const codec_lm_kind_vtable codec_lm_vtable_parallel_heads_delay = {
    /*.kind               =*/ CODEC_LM_KIND_PARALLEL_HEADS_DELAY,
    /*.name               =*/ "parallel_heads_delay",
    /*.init               =*/ init,
    /*.free               =*/ free_lm,
    /*.state_init         =*/ state_init,
    /*.state_free         =*/ state_free,
    /*.state_reset        =*/ state_reset,
    /*.step_begin         =*/ step_begin,
    /*.step_pending       =*/ step_pending,
    /*.step_logits        =*/ step_logits,
    /*.step_push_code     =*/ step_push_code,
    /*.step_finish        =*/ step_finish,
    /*.audio_embd         =*/ audio_embd,
    /*.compose_audio_embd =*/ compose_audio_embd,
    /*.compose_next_embd  =*/ compose_next_embd,
    /*.speaker_encode     =*/ nullptr,
};
