#include "rn-completion.h"
#include "rn-llama.h"
#include "rn-tts.h"
#include "rn-mtmd.hpp"
#include "rn-common.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// Include multimodal support
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "tools/mtmd/clip.h"

namespace rnllama {

// Constructor
llama_rn_context_completion::llama_rn_context_completion(llama_rn_context* parent)
    : parent_ctx(parent) {
}

// Destructor
llama_rn_context_completion::~llama_rn_context_completion() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
        ctx_sampling = nullptr;
    }
}

void llama_rn_context_completion::rewind() {
    is_interrupted = false;
    parent_ctx->params.antiprompt.clear();
    parent_ctx->params.sampling.grammar.clear();
    num_prompt_tokens = 0;
    num_tokens_predicted = 0;
    prefill_text = "";
    generated_text = "";
    generated_text.reserve(parent_ctx->params.n_ctx);
    embeddings.clear();
    embedding_dim = 0;
    truncated = false;
    context_full = false;
    stopped_eos = false;
    stopped_word = false;
    stopped_limit = false;
    stopping_word = "";
    incomplete = false;
    n_remain = 0;
    n_past = 0;
    parent_ctx->params.sampling.n_prev = parent_ctx->n_ctx;
    if (parent_ctx->isVocoderEnabled()) {
        parent_ctx->tts_wrapper->reset();
    }
}

bool llama_rn_context_completion::initSampling() {
    if (ctx_sampling != nullptr) {
        common_sampler_free(ctx_sampling);
    }
    ctx_sampling = common_sampler_init(parent_ctx->model, parent_ctx->params.sampling);
    return ctx_sampling != nullptr;
}

void llama_rn_context_completion::truncatePrompt(std::vector<llama_token> &prompt_tokens) {
    const int n_left = parent_ctx->n_ctx - parent_ctx->params.n_keep;
    const int n_block_size = n_left / 2;
    const int erased_blocks = (prompt_tokens.size() - parent_ctx->params.n_keep - n_block_size) / n_block_size;

    // Keep n_keep tokens at start of prompt (at most n_ctx - 4)
    std::vector<llama_token> new_tokens(prompt_tokens.begin(), prompt_tokens.begin() + parent_ctx->params.n_keep);

    new_tokens.insert(new_tokens.end(), prompt_tokens.begin() + parent_ctx->params.n_keep + erased_blocks * n_block_size, prompt_tokens.end());

    LOG_INFO("input truncated, n_ctx: %d, n_keep: %d, n_left: %d, old_size: %d, new_size: %d",
        parent_ctx->n_ctx,
        parent_ctx->params.n_keep,
        n_left,
        prompt_tokens.size(),
        new_tokens.size()
    );

    truncated = true;
    prompt_tokens = new_tokens;
}

void llama_rn_context_completion::loadPrompt(const std::vector<std::string> &media_paths) {
    bool has_media = !media_paths.empty();

    // Check if this is an encoder-decoder model (like T5)
    const bool is_enc_dec = llama_model_has_encoder(parent_ctx->model);
    const auto vocab = llama_model_get_vocab(parent_ctx->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    if (!has_media) {
        std::vector<llama_token> text_tokens;

        // Backbones with no text tokenizer (e.g. Chatterbox T3, tokenizer.ggml.model=none)
        // must not call llama_tokenize — it asserts.  Leave embd empty; the TTS prefill
        // block in nextToken() will take over from n_past = -1 (normalized to 0).
        if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
            embd.clear();
            n_past = -1;
            num_prompt_tokens = 0;
            has_next_token = true;   // let the completion loop start
            return;
        }

        // Text-only path - use modified tokenization for encoder-decoder models
        text_tokens = ::common_tokenize(parent_ctx->ctx, parent_ctx->params.prompt, add_bos || is_enc_dec, true);
        num_prompt_tokens = text_tokens.size();

        // LOG tokens
        std::stringstream ss;
        ss << "\n" << __func__ << ": prompt_tokens = ";
        for (auto& token : text_tokens) {
            ss << token << " ";
        }
        LOG_INFO("%s\n", ss.str().c_str());

        if (parent_ctx->params.n_keep < 0) {
            parent_ctx->params.n_keep = (int)num_prompt_tokens;
        }
        parent_ctx->params.n_keep = std::min(parent_ctx->n_ctx - 4, parent_ctx->params.n_keep);

        // Handle truncation if needed
        if (num_prompt_tokens >= (size_t)parent_ctx->n_ctx) {
            if (!parent_ctx->params.ctx_shift) {
                context_full = true;
                return;
            }
            truncatePrompt(text_tokens);
            num_prompt_tokens = text_tokens.size();
            LM_GGML_ASSERT(num_prompt_tokens < (size_t)parent_ctx->n_ctx);
        }

        // Update sampling context
        for (auto & token : text_tokens) {
            common_sampler_accept(ctx_sampling, token, false);
        }

        // compare the evaluated prompt with the new prompt
        n_past = is_enc_dec ? 0 : find_common_prefix_length(embd, text_tokens);

        embd = text_tokens;
        if (n_past == num_prompt_tokens) {
            // we have to evaluate at least 1 token to generate logits.
            n_past--;
        }

        // Manage KV cache
        auto * kv = llama_get_memory(parent_ctx->ctx);
        bool cache_remove_success = llama_memory_seq_rm(kv, 0, n_past, -1);

        // For hybrid models (LFM-2, Granite, Mamba, etc.), partial cache removal may fail
        // In that case, do a full cache clear to prevent contamination
        if (!cache_remove_success) {
            LOG_WARNING("Partial cache removal failed (likely hybrid/recurrent model), doing full cache clear");
            llama_memory_clear(kv, false);
            embd.clear();
            n_past = 0;
            // Re-assign all tokens to embd since we cleared everything
            embd = text_tokens;
        }

        LOG_VERBOSE("prompt ingested, n_past: %d, cached: %s, to_eval: %s",
            n_past,
            tokens_to_str(parent_ctx->ctx, embd.cbegin(), embd.cbegin() + n_past).c_str(),
            tokens_to_str(parent_ctx->ctx, embd.cbegin() + n_past, embd.cend()).c_str()
        );
    } else {
        // Multimodal path - process all media paths
        processMedia(parent_ctx->params.prompt, media_paths);
        num_prompt_tokens = embd.size();
    }

    // Handle encoder-decoder models (like T5) with special encoding phase
    if (is_enc_dec && !has_media) {
        // For encoder-decoder models, we need to encode the input tokens first
        if (embd.size() > n_past) {
            // Encode tokens in batches using n_batch as chunk size
            int n_past_batch = n_past;
            int n_remaining = embd.size() - n_past;

            while (n_remaining > 0) {
                int n_eval = n_remaining;
                if (n_eval > parent_ctx->params.n_batch) {
                    n_eval = parent_ctx->params.n_batch;
                }

                int ret = llama_encode(parent_ctx->ctx, llama_batch_get_one(embd.data() + n_past_batch, n_eval));
                if (ret < 0) {
                    LOG_ERROR("Failed to encode token batch, code: %d, n_eval: %d, n_past_batch: %d", ret, n_eval, n_past_batch);
                    has_next_token = false;
                    return;
                }

                n_past_batch += n_eval;
                n_remaining -= n_eval;
                n_past += n_eval;
            }
        }
        // Update token count for encoding
        num_prompt_tokens = embd.size();

        // Add decoder start token for encoder-decoder models
        llama_token decode_bos = llama_model_decoder_start_token(parent_ctx->model);
        if (decode_bos == LLAMA_TOKEN_NULL) {
            decode_bos = llama_vocab_bos(vocab);
        }

        // Add the decoder start token to begin generation
        embd.emplace_back(decode_bos);
        common_sampler_accept(ctx_sampling, decode_bos, false);

        LOG_INFO("[DEBUG] T5 encoding complete, added decoder BOS token: %d", decode_bos);
    }

    has_next_token = true;

    LOG_VERBOSE("loadPrompt: n_past=%d embd.size=%zu num_prompt_tokens=%zu has_media=%d",
               n_past, (size_t)embd.size(), num_prompt_tokens, has_media ? 1 : 0);
}

void llama_rn_context_completion::beginCompletion() {
    beginCompletion(COMMON_CHAT_FORMAT_CONTENT_ONLY, COMMON_REASONING_FORMAT_NONE, false);
}

void llama_rn_context_completion::beginCompletion(int chat_format, common_reasoning_format reasoning_format, bool thinking_forced_open, const std::string &chat_parser) {
    // number of tokens to keep when resetting context
    n_remain = parent_ctx->params.n_predict;
    llama_perf_context_reset(parent_ctx->ctx);
    is_predicting = true;

    current_chat_format = chat_format;
    current_reasoning_format = reasoning_format;
    current_thinking_forced_open = thinking_forced_open;
    current_chat_parser = chat_parser;
}

void llama_rn_context_completion::endCompletion() {
    is_predicting = false;
}

completion_token_output llama_rn_context_completion::nextToken()
{
    completion_token_output result;
    result.tok = -1;

    if (embd.size() >= (size_t)parent_ctx->params.n_ctx)
    {
        if (!parent_ctx->params.ctx_shift) {
            // If context shifting is disabled, stop generation
            LOG_WARNING("context full, n_ctx: %d, tokens: %d", parent_ctx->params.n_ctx, embd.size());
            has_next_token = false;
            context_full = true;
            return result;
        }

        // Shift context

        const int n_left    = n_past - parent_ctx->params.n_keep - 1;
        const int n_discard = n_left/2;

        auto * kv = llama_get_memory(parent_ctx->ctx);
        llama_memory_seq_rm (kv, 0, parent_ctx->params.n_keep + 1            , parent_ctx->params.n_keep + n_discard + 1);
        llama_memory_seq_add(kv, 0, parent_ctx->params.n_keep + 1 + n_discard, n_past, -n_discard);

        for (size_t i = parent_ctx->params.n_keep + 1 + n_discard; i < embd.size(); i++)
        {
            embd[i - n_discard] = embd[i];
        }
        embd.resize(embd.size() - n_discard);

        n_past -= n_discard;
        truncated = true;

        LOG_VERBOSE("context shifted, new n_past: %d, new size: %d", n_past, embd.size());
    }

    // Continuous-latent TTS flow (BlueMagpie-TTS / VoxCPM): after every
    // `llama_decode` we hand the just-produced hidden state to
    // `tryContinuousAudioStep` which runs the codec_lm's step machine
    // (step_generate + step_feedback_embd) and produces the LocEnc
    // feedback embedding for the NEXT decode.  Standard token sampling
    // is skipped — the continuous codec_lm doesn't emit codebook codes.
    // Terminates when the codec_lm's stop head fires.
    const bool is_continuous_tts =
        parent_ctx->isVocoderEnabled() &&
        parent_ctx->tts_wrapper != nullptr &&
        parent_ctx->tts_wrapper->isTTSContinuous(parent_ctx);

    // Codebook codec_lm-AR TTS flow (CSM / Qwen3-TTS / MOSS-TTSD /
    // MOSS-TTS-Realtime / Chatterbox): structurally identical to the
    // continuous flow above but the codec_lm produces N codebook codes
    // per step instead of a latent patch.  The hook appends codes to
    // `tts_wrapper->audio_tokens` (T, N interleaved) and composes the
    // next backbone embed via `codec_lm_compose_next_embd`.  Stop when
    // the codec_lm's model-specific EOS heuristic trips.  Standard token
    // sampling is bypassed: the backbone's own logits are consumed by
    // the codec_lm hook (for text-modality c0 models like MOSS-TTS-Realtime)
    // and discarded otherwise.
    const bool is_codec_lm_ar_tts =
        !is_continuous_tts &&
        parent_ctx->isVocoderEnabled() &&
        parent_ctx->tts_wrapper != nullptr &&
        parent_ctx->tts_wrapper->isTTSCodecLmAR(parent_ctx);
    LOG_VERBOSE("nextToken: is_continuous=%d is_codec_lm_ar=%d chatterbox_pending=%d",
               (int)is_continuous_tts, (int)is_codec_lm_ar_tts,
               parent_ctx->tts_wrapper ? (int)parent_ctx->tts_wrapper->chatterbox_prefill_pending : -1);

    if ((is_continuous_tts || is_codec_lm_ar_tts) && !parent_ctx->params.embedding) {
        LOG_ERROR("codec_lm TTS requires context created with embedding=true — please reinitialize the context");
        has_next_token = false;
        return result;
    }

    LOG_VERBOSE("nextToken: is_continuous=%d is_codec_lm_ar=%d talker_rows=%d embd.size=%zu n_past=%d",
                (int)is_continuous_tts, (int)is_codec_lm_ar_tts,
                parent_ctx->tts_wrapper ? parent_ctx->tts_wrapper->talker_prefix_rows : -1,
                embd.size(), (int)n_past);

    // vocab pointer needed in both the talker-prefix block and the
    // post-while termination logic below.
    const llama_vocab * vocab = llama_model_get_vocab(parent_ctx->model);

    // Speaker-conditioning prefix (voice-clone codec_lm-AR models: output
    // of `codec_lm_speaker_encode` stashed on the tts wrapper).  Fed once
    // via a manual embd-batch AHEAD of the token prompt so the codec_lm's
    // first hidden read sees the speaker context.  The KV cache position
    // shifts by `rows`; the token batch below starts from `n_past + rows`.
    // Only fires on codec_lm-AR flows to keep the standard path unchanged.
    if (is_codec_lm_ar_tts &&
        !parent_ctx->tts_wrapper->pending_speaker_emb_prefix.empty() &&
        parent_ctx->tts_wrapper->pending_speaker_emb_rows > 0 &&
        parent_ctx->tts_wrapper->pending_speaker_emb_hidden_dim ==
            llama_model_n_embd(parent_ctx->model)) {
        const int rows       = parent_ctx->tts_wrapper->pending_speaker_emb_rows;
        const int hidden_dim = parent_ctx->tts_wrapper->pending_speaker_emb_hidden_dim;
        llama_batch b = llama_batch_init(rows, hidden_dim, 1);
        b.n_tokens = rows;
        std::memcpy(b.embd,
                    parent_ctx->tts_wrapper->pending_speaker_emb_prefix.data(),
                    (size_t) rows * (size_t) hidden_dim * sizeof(float));
        for (int i = 0; i < rows; ++i) {
            b.pos[i]       = n_past + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0;
        }
        b.token = nullptr;
        const int rc = llama_decode(parent_ctx->ctx, b);
        llama_batch_free(b);
        if (rc) {
            LOG_ERROR("failed to eval speaker prefix, rows=%d", rows);
            has_next_token = false;
            return result;
        }
        n_past += rows;
        // Consume so subsequent nextToken calls don't re-inject.
        parent_ctx->tts_wrapper->pending_speaker_emb_prefix.clear();
        parent_ctx->tts_wrapper->pending_speaker_emb_rows = 0;
        parent_ctx->tts_wrapper->pending_speaker_emb_hidden_dim = 0;
    }

    // Qwen3-TTS talker prefix inject: `getFormattedAudioCompletion` built a
    // multi-row embedding matrix (role tokens + text tokens + optional
    // x-vector) via `audio_lm_build_talker_prefix`.  Decode it now as an
    // embd batch with logits enabled only on the last row (which gives us the
    // hidden state we pass to `tryCodecLmAudioStep`).  After the batch decode
    // we arm embed-override mode via `tryTalkerPrefill` and immediately fire
    // the first codec_lm step so that `pending_next_embd` is populated before
    // the while loop.  We then grow `embd` to `rows` dummy slots so that
    // n_past == embd.size() — the while loop is skipped, and the termination
    // block below queues the placeholder and returns to the outer caller.
    // Subsequent `nextToken` calls enter the normal inject_ar_embd → step path.
    if (is_codec_lm_ar_tts &&
        parent_ctx->tts_wrapper->talker_prefix_rows > 0 &&
        (int) parent_ctx->tts_wrapper->talker_prefix_embd.size() ==
            parent_ctx->tts_wrapper->talker_prefix_rows *
            parent_ctx->tts_wrapper->talker_prefix_hidden) {

        // loadPrompt sets n_past = -1 when embd is empty (its "must decode
        // at least one token" guard).  For the talker path the KV cache is
        // fresh and the prefix starts at position 0.
        if (n_past < 0) n_past = 0;

        const int rows       = parent_ctx->tts_wrapper->talker_prefix_rows;
        const int hidden_dim = parent_ctx->tts_wrapper->talker_prefix_hidden;
        llama_batch b = llama_batch_init(rows, hidden_dim, 1);
        b.n_tokens = rows;
        std::memcpy(b.embd,
                    parent_ctx->tts_wrapper->talker_prefix_embd.data(),
                    (size_t) rows * (size_t) hidden_dim * sizeof(float));
        for (int i = 0; i < rows; ++i) {
            b.pos[i]       = n_past + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = (i == rows - 1) ? 1 : 0;
        }
        b.token = nullptr;
        const int rc = llama_decode(parent_ctx->ctx, b);
        llama_batch_free(b);
        if (rc) {
            LOG_ERROR("failed to eval talker prefix, rows=%d", rows);
            has_next_token = false;
            return result;
        }
        n_past += rows;
        parent_ctx->tts_wrapper->talker_prefix_embd.clear();
        parent_ctx->tts_wrapper->talker_prefix_rows   = 0;
        parent_ctx->tts_wrapper->talker_prefix_hidden = 0;

        // Arm embed-override so per-step audio_lm_get_next_embed works.
        const float * last_h = llama_get_embeddings_ith(parent_ctx->ctx, -1);
        const int dim = llama_model_n_embd(parent_ctx->model);
        if (!last_h || dim <= 0) {
            LOG_ERROR("talker prefix: llama_get_embeddings_ith returned NULL after %d rows", rows);
            has_next_token = false;
            return result;
        }
        if (!parent_ctx->tts_wrapper->tryTalkerPrefill(parent_ctx, last_h, dim)) {
            LOG_ERROR("tryTalkerPrefill failed");
            has_next_token = false;
            return result;
        }

        // Fire the first codec_lm step using the prefix's last-row hidden.
        if (!parent_ctx->tts_wrapper->tryCodecLmAudioStep(
                parent_ctx, /*backbone_sampled_tok=*/-1, last_h, dim)) {
            LOG_ERROR("tryCodecLmAudioStep failed after talker prefill");
            has_next_token = false;
            return result;
        }

        // Grow embd to match n_past so the while loop below is skipped
        // (n_past == embd.size()) and subsequent calls drive the normal
        // inject_ar_embd → step_now_codec_ar cycle at the right KV positions.
        while ((llama_pos) embd.size() < n_past) {
            embd.push_back(llama_vocab_bos(vocab));
        }
    }

    // Chatterbox T3 prefill: when getFormattedAudioCompletion returned
    // flow="chatterbox_embd", tokenize the text (via the baked BPE in the
    // codec GGUF), build the CFG prompt embedding pair, and decode both
    // cond + uncond sequences into the backbone KV cache (seq_ids 0 and 1).
    // After prefill, fire the first codec_lm step and pad embd to match
    // chatterbox_n_past so the inject_ar_embd cycle continues from there.
    // Only runs once per generation: chatterbox_prefill_pending is cleared
    // here and chatterbox_n_past > 0 on subsequent nextToken calls.
    if (is_codec_lm_ar_tts &&
        parent_ctx->tts_wrapper->chatterbox_prefill_pending) {

        LOG_INFO("Chatterbox prefill: entering block, n_past=%d text='%s'",
                 n_past,
                 parent_ctx->tts_wrapper->chatterbox_text.substr(0,40).c_str());

        parent_ctx->tts_wrapper->chatterbox_prefill_pending = false;

        // Empty prompt → loadPrompt sets n_past = -1.  Normalize here.
        if (n_past < 0) n_past = 0;

        // Text is stored on tts_wrapper (backbone has no text tokenizer).
        const std::string & text = parent_ctx->tts_wrapper->chatterbox_text;
        const float cfg_weight   = parent_ctx->tts_wrapper->chatterbox_cfg_weight;

        if (!parent_ctx->tts_wrapper->tryChatterboxPrefill(
                parent_ctx, text,
                /*ref_pcm=*/nullptr, /*ref_n_samples=*/0,
                /*ref_sample_rate=*/0, cfg_weight)) {
            LOG_ERROR("tryChatterboxPrefill failed");
            has_next_token = false;
            return result;
        }

        n_past = parent_ctx->tts_wrapper->chatterbox_n_past;

        const float * last_h = llama_get_embeddings_ith(parent_ctx->ctx, -1);
        const int dim = llama_model_n_embd(parent_ctx->model);
        if (!last_h || dim <= 0) {
            LOG_ERROR("Chatterbox prefill: NULL hidden after decode");
            has_next_token = false;
            return result;
        }
        if (!parent_ctx->tts_wrapper->tryCodecLmAudioStep(
                parent_ctx, /*backbone_sampled_tok=*/-1, last_h, dim)) {
            LOG_ERROR("Chatterbox: first tryCodecLmAudioStep failed");
            has_next_token = false;
            return result;
        }
        while ((llama_pos) embd.size() < n_past) {
            embd.push_back(llama_vocab_bos(vocab));
        }
    }

    // MOSS-TTS-Realtime (streaming_interleave) prefill: when
    // getFormattedAudioCompletion returned flow="realtime_embd", compose the
    // streaming prefill block (context tokens + first prefill_text_len payload
    // tokens, each row = text_embd[tok] + compose_audio_codes_embd), decode it
    // as one embd batch, arm embed-override, then fire the first realtime step
    // so pending_next_embd is ready.  Pad embd to match n_past so the
    // inject_ar_embd cycle continues from there.  Runs once per generation.
    if (is_codec_lm_ar_tts &&
        parent_ctx->tts_wrapper->realtime_active &&
        parent_ctx->tts_wrapper->realtime_prefill_pending) {

        parent_ctx->tts_wrapper->realtime_prefill_pending = false;

        // Empty prompt → loadPrompt sets n_past = -1.  Normalize here.
        if (n_past < 0) n_past = 0;

        const int new_past = parent_ctx->tts_wrapper->tryRealtimePrefill(
            parent_ctx, (int) n_past);
        if (new_past < 0) {
            LOG_ERROR("tryRealtimePrefill failed");
            has_next_token = false;
            return result;
        }
        n_past = new_past;

        const float * last_h = llama_get_embeddings_ith(parent_ctx->ctx, -1);
        const int dim = llama_model_n_embd(parent_ctx->model);
        if (!last_h || dim <= 0) {
            LOG_ERROR("realtime prefill: NULL hidden after decode");
            has_next_token = false;
            return result;
        }
        if (!parent_ctx->tts_wrapper->tryCodecLmAudioStep(
                parent_ctx, /*backbone_sampled_tok=*/-1, last_h, dim)) {
            LOG_ERROR("realtime: first tryCodecLmAudioStep failed");
            has_next_token = false;
            return result;
        }
        while ((llama_pos) embd.size() < n_past) {
            embd.push_back(llama_vocab_bos(vocab));
        }
    }

    bool tg = true;
    while (n_past < embd.size())
    {
        int n_eval = (int)embd.size() - n_past;
        tg = n_eval == 1;
        if (n_eval > parent_ctx->params.n_batch)
        {
            n_eval = parent_ctx->params.n_batch;
        }
        // Standard path: token batch via llama_batch_get_one.  Continuous-
        // latent TTS injects the LocEnc feedback embedding via `b.embd`
        // (see below); we detect that by a pending flag on tts_wrapper.
        // Codec_lm-AR TTS does the same with the composed audio embedding
        // (`pending_next_embd`) so the codec_lm's compose_next_embd output
        // becomes the next `llama_decode`'s input.
        const bool inject_embd =
            is_continuous_tts &&
            n_eval == 1 &&
            parent_ctx->tts_wrapper->audio_embeddings_pending &&
            (int) parent_ctx->tts_wrapper->pending_feedback_embd.size() ==
                llama_model_n_embd(parent_ctx->model);
        const bool inject_ar_embd =
            is_codec_lm_ar_tts &&
            n_eval == 1 &&
            parent_ctx->tts_wrapper->codec_lm_ar_pending_embd &&
            (int) parent_ctx->tts_wrapper->pending_next_embd.size() ==
                llama_model_n_embd(parent_ctx->model);

        if (inject_embd || inject_ar_embd) {
            const int hidden_dim = llama_model_n_embd(parent_ctx->model);
            const float * src = inject_embd
                ? parent_ctx->tts_wrapper->pending_feedback_embd.data()
                : parent_ctx->tts_wrapper->pending_next_embd.data();
            llama_batch b = llama_batch_init(1, hidden_dim, 1);
            b.n_tokens = 1;
            std::memcpy(b.embd, src, (size_t) hidden_dim * sizeof(float));
            b.pos[0]       = n_past;
            b.n_seq_id[0]  = 1;
            b.seq_id[0][0] = 0;
            b.logits[0]    = 1;
            b.token        = nullptr;
            const int rc = llama_decode(parent_ctx->ctx, b);
            llama_batch_free(b);
            if (rc) {
                LOG_ERROR("failed to eval codec_lm TTS embd, n_past: %d", n_past);
                has_next_token = false;
                return result;
            }
            // Consume the pending embd so subsequent iterations don't
            // re-inject; the codec_lm step below re-populates it.
            if (inject_embd) {
                parent_ctx->tts_wrapper->audio_embeddings_pending = false;
                parent_ctx->tts_wrapper->pending_feedback_embd.clear();
            } else {
                parent_ctx->tts_wrapper->codec_lm_ar_pending_embd = false;
                parent_ctx->tts_wrapper->pending_next_embd.clear();
            }
        } else if (is_continuous_tts) {
            // Continuous-latent TTS prompt decode: the RALM must see the
            // WHOLE prompt's per-position backbone hiddens before generation
            // (codec_lm_text_prefill).  llama_batch_get_one only requests
            // logits/embeddings on the LAST position, so build a manual token
            // batch with logits[i]=1 for every token in the chunk and gather
            // each position's hidden into tts_wrapper->prompt_hiddens (in
            // order, across n_batch chunks).  tryContinuousPrefill fires once
            // the full prompt is decoded (below).
            llama_batch b = llama_batch_init(n_eval, 0, 1);
            b.n_tokens = n_eval;
            for (int i = 0; i < n_eval; ++i) {
                b.token[i]     = embd[n_past + i];
                b.pos[i]       = n_past + i;
                b.n_seq_id[i]  = 1;
                b.seq_id[i][0] = 0;
                b.logits[i]    = 1;
            }
            const int rc = llama_decode(parent_ctx->ctx, b);
            if (rc) {
                llama_batch_free(b);
                LOG_ERROR("failed to eval continuous TTS prompt, n_eval: %d, n_past: %d",
                          n_eval, (int) n_past);
                has_next_token = false;
                return result;
            }
            const int dim = llama_model_n_embd(parent_ctx->model);
            for (int i = 0; i < n_eval; ++i) {
                const float * h = llama_get_embeddings_ith(parent_ctx->ctx, i);
                if (h == nullptr || dim <= 0) {
                    llama_batch_free(b);
                    LOG_ERROR("continuous TTS: NULL hidden for prompt pos %d (n_past=%d)",
                              i, (int) n_past);
                    has_next_token = false;
                    return result;
                }
                parent_ctx->tts_wrapper->prompt_hiddens.insert(
                    parent_ctx->tts_wrapper->prompt_hiddens.end(), h, h + dim);
            }
            llama_batch_free(b);
        } else {
            if (llama_decode(parent_ctx->ctx, llama_batch_get_one(&embd[n_past], n_eval)))
            {
                LOG_ERROR("failed to eval, n_eval: %d, n_past: %d, n_threads: %d, embd: %s",
                    n_eval,
                    n_past,
                    parent_ctx->params.cpuparams.n_threads,
                    tokens_to_str(parent_ctx->ctx, embd.cbegin() + n_past, embd.cend()).c_str()
                );
                has_next_token = false;
                return result;
            }
        }
        n_past += n_eval;

        // For continuous / codec_lm-AR TTS: run one codec_lm step once
        // we've fully decoded the current pending sequence (prompt or
        // feedback embd).  The last decoded position's hidden state
        // seeds the next step; unlike standard capture we do NOT gate
        // on tg (multi-token prompt batches also produce a valid
        // last-position hidden via llama_get_embeddings_ith(-1)).
        const bool step_now_continuous =
            is_continuous_tts &&
            (llama_pos) n_past == (llama_pos) embd.size();
        const bool step_now_codec_ar =
            is_codec_lm_ar_tts &&
            (llama_pos) n_past == (llama_pos) embd.size();

        if (step_now_continuous) {
            const float *embedding = llama_get_embeddings_ith(parent_ctx->ctx, -1);
            const int dim = llama_model_n_embd(parent_ctx->model);
            if (embedding == nullptr || dim <= 0) {
                LOG_ERROR("continuous TTS: llama_get_embeddings_ith returned NULL at n_past=%d", (int) n_past);
                has_next_token = false;
                return result;
            }
            // Before the FIRST step, run the RALM text-prefill so it has
            // seen the whole prompt's per-position hiddens (call sequence:
            // prefill(all positions) → step (primed, patch 0, ignores h_in)
            // → feedback embd → decode → step → ...).  Guarded to run once
            // per generation; `reset()` clears the flag + the RALM state.
            if (!parent_ctx->tts_wrapper->continuous_prefill_done) {
                const int n_prompt =
                    (int) (parent_ctx->tts_wrapper->prompt_hiddens.size() / (size_t) dim);
                if (n_prompt <= 0) {
                    LOG_ERROR("continuous TTS: no prompt hiddens gathered for prefill (n_past=%d)",
                              (int) n_past);
                    has_next_token = false;
                    return result;
                }
                if (!parent_ctx->tts_wrapper->tryContinuousPrefill(
                        parent_ctx,
                        parent_ctx->tts_wrapper->prompt_hiddens.data(),
                        n_prompt, dim)) {
                    LOG_ERROR("tryContinuousPrefill failed at n_past=%d", (int) n_past);
                    has_next_token = false;
                    return result;
                }
                parent_ctx->tts_wrapper->continuous_prefill_done = true;
                // Free the scratch — the K/V is now in the RALM cache.
                parent_ctx->tts_wrapper->prompt_hiddens.clear();
                parent_ctx->tts_wrapper->prompt_hiddens.shrink_to_fit();
            }
            // Run one codec_lm step on the just-produced hidden state,
            // accumulating the latent patch into
            // tts_wrapper->audio_embeddings and preparing the next
            // b.embd via pending_feedback_embd.  Sets
            // audio_embeddings_done when the stop head fires.
            if (!parent_ctx->tts_wrapper->tryContinuousAudioStep(
                    parent_ctx, embedding, dim)) {
                LOG_ERROR("tryContinuousAudioStep failed at n_past=%d", (int) n_past);
                has_next_token = false;
                return result;
            }
            embedding_dim = parent_ctx->tts_wrapper->audio_embedding_dim;
            // Surface the accumulated latents through the standard
            // `embeddings` field so JS's `result.embeddings` +
            // `result.embedding_dim` are populated consistently.
            embeddings = parent_ctx->tts_wrapper->audio_embeddings;
        } else if (step_now_codec_ar) {
            const float *embedding = llama_get_embeddings_ith(parent_ctx->ctx, -1);
            const int dim = llama_model_n_embd(parent_ctx->model);
            if (embedding == nullptr || dim <= 0) {
                LOG_ERROR("codec_lm-AR TTS: llama_get_embeddings_ith returned NULL at n_past=%d", (int) n_past);
                has_next_token = false;
                return result;
            }
            // Run one codec_lm step: samples N codebook codes, appends
            // them to tts_wrapper->audio_tokens (T, N interleaved), and
            // composes the next backbone embed into pending_next_embd.
            // The backbone-sampled token is unused for text-modality-c0
            // models here (-1); tryCodecLmAudioStep pulls it from
            // llama_get_logits_ith internally when needed.
            if (!parent_ctx->tts_wrapper->tryCodecLmAudioStep(
                    parent_ctx, /*backbone_sampled_tok=*/-1,
                    embedding, dim)) {
                LOG_ERROR("tryCodecLmAudioStep failed at n_past=%d", (int) n_past);
                has_next_token = false;
                return result;
            }
        } else if (parent_ctx->params.embedding && tg && n_past > (llama_pos)num_prompt_tokens) {
            const float *embedding = llama_get_embeddings_ith(parent_ctx->ctx, -1);
            const int dim = llama_model_n_embd(parent_ctx->model);
            if (embedding != nullptr && dim > 0) {
                embedding_dim = dim;
                embeddings.insert(embeddings.end(), embedding, embedding + dim);
            }
        }

        if(is_interrupted) {
            LOG_INFO("Decoding Interrupted");
            embd.resize(n_past);
            has_next_token = false;
            return result;
        }
    }

    if (parent_ctx->params.n_predict == 0)
    {
        has_next_token = false;
        result.tok = llama_vocab_eos(vocab);
        return result;
    }

    // Continuous-latent TTS: skip token sampling entirely.  If the stop
    // head fired during the step hook, terminate.  Otherwise queue the
    // pending feedback embd — the NEXT `nextToken` will consume it via
    // the inject-embd path above.  We grow `embd` by one dummy slot so
    // the outer loop's `n_past < embd.size()` becomes true again next
    // time, driving another decode.
    if (is_continuous_tts) {
        if (parent_ctx->tts_wrapper->audio_embeddings_done) {
            has_next_token = false;
            stopped_eos = true;
            return result;
        }
        if (!parent_ctx->tts_wrapper->audio_embeddings_pending) {
            // No pending feedback and no stop — shouldn't happen unless
            // the codec step returned neither result.  Guard against a
            // spin loop by bailing.
            LOG_ERROR("continuous TTS: no feedback embd queued after step; stopping");
            has_next_token = false;
            return result;
        }
        // Placeholder token — never consumed since we replace the next
        // decode with our embd batch via `inject_embd`.  Using BOS keeps
        // any downstream vocab lookups sane if the caller ever peeks at
        // `embd` (currently: no one does for continuous flow).
        embd.push_back(llama_vocab_bos(vocab));
        result.tok = -1;
        --n_remain;
        num_tokens_predicted++;
        has_next_token = parent_ctx->params.n_predict == -1 || n_remain != 0;
        return result;
    }

    // Codebook codec_lm-AR TTS: same skip-sampling shape as continuous.
    // The codec_lm hook already appended this frame's N codes to
    // `tts_wrapper->audio_tokens` and set up `pending_next_embd` for
    // the next decode.  On CSM's audio-EOS heuristic (or any future
    // per-model stop condition wired into `tryCodecLmAudioStep`),
    // terminate here so the outer completion loop stops.
    if (is_codec_lm_ar_tts) {
        if (parent_ctx->tts_wrapper->codec_lm_ar_done) {
            has_next_token = false;
            stopped_eos = true;
            return result;
        }
        if (!parent_ctx->tts_wrapper->codec_lm_ar_pending_embd) {
            LOG_ERROR("codec_lm-AR TTS: no next embd queued after step; stopping");
            has_next_token = false;
            return result;
        }
        embd.push_back(llama_vocab_bos(vocab));
        result.tok = -1;
        --n_remain;
        num_tokens_predicted++;
        has_next_token = parent_ctx->params.n_predict == -1 || n_remain != 0;
        return result;
    }

    {
        // out of user input, sample next token
        std::vector<llama_token_data> candidates;
        candidates.reserve(llama_vocab_n_tokens(vocab));

        llama_token new_token_id = common_sampler_sample(ctx_sampling, parent_ctx->ctx, -1);

        const int32_t n_probs = parent_ctx->params.sampling.n_probs;
        if (n_probs > 0) {
          llama_token_data_array cur_p = *common_sampler_get_candidates(ctx_sampling, true);
          for (size_t i = 0; i < std::min(cur_p.size, (size_t)n_probs); ++i)
          {
              result.probs.push_back({cur_p.data[i].id, cur_p.data[i].p});
          }
        }

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            has_next_token = false;
            stopped_eos = true;
            LOG_VERBOSE("EOS: %s", common_token_to_piece(parent_ctx->ctx, new_token_id).c_str());
            return result;
        }

        result.tok = new_token_id;
        result.text = common_token_to_piece(parent_ctx->ctx, new_token_id);

        common_sampler_accept(ctx_sampling, result.tok, true);
        if (tg) {
            num_tokens_predicted++;
        }
    }

    // add it to the context
    embd.push_back(result.tok);
    // decrement remaining sampling budget
    --n_remain;

    has_next_token = parent_ctx->params.n_predict == -1 || n_remain != 0;
    return result;
}

size_t llama_rn_context_completion::findStoppingStrings(const std::string &text, const size_t last_token_size,
                            const stop_type type)
{
    size_t stop_pos = std::string::npos;
    for (const std::string &word : parent_ctx->params.antiprompt)
    {
        size_t pos;
        if (type == STOP_FULL)
        {
            const size_t tmp = word.size() + last_token_size;
            const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;
            pos = text.find(word, from_pos);
        }
        else
        {
            pos = find_partial_stop_string(word, text);
        }
        if (pos != std::string::npos &&
            (stop_pos == std::string::npos || pos < stop_pos))
        {
            if (type == STOP_FULL)
            {
                stopping_word = word;
                stopped_word = true;
                has_next_token = false;
            }
            stop_pos = pos;
        }
    }
    return stop_pos;
}

completion_token_output llama_rn_context_completion::doCompletion()
{
    completion_token_output token_with_probs = nextToken();

    const std::string token_text = token_with_probs.tok == -1 ? "" : common_token_to_piece(parent_ctx->ctx, token_with_probs.tok);
    generated_text += token_text;

    if (parent_ctx->isVocoderEnabled()) {
        tts_type type = parent_ctx->tts_wrapper->getTTSType(parent_ctx);
        if (parent_ctx->tts_wrapper->type == UNKNOWN) {
            parent_ctx->tts_wrapper->type = type;
        }
        parent_ctx->tts_wrapper->tryAddAudioToken(parent_ctx, token_with_probs.tok, token_text);
    }

    if (parent_ctx->params.sampling.n_probs > 0)
    {
        generated_token_probs.push_back(token_with_probs);
    }

    // check if there is incomplete UTF-8 character at the end
    for (unsigned i = 1; i < 5 && i <= generated_text.size(); ++i) {
        unsigned char c = generated_text[generated_text.size() - i];
        if ((c & 0xC0) == 0x80) {
            // continuation byte: 10xxxxxx
            continue;
        }
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character: 110xxxxx ...
            incomplete = i < 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte character: 1110xxxx ...
            incomplete = i < 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte character: 11110xxx ...
            incomplete = i < 4;
        }
        // else 1-byte character or invalid byte
        break;
    }

    if (incomplete && !has_next_token)
    {
        has_next_token = true;
        n_remain++;
    }

    if (!has_next_token && n_remain == 0)
    {
        stopped_limit = true;
    }

    LOG_VERBOSE("next token, token: %s, token_text: %s, has_next_token: %d, n_remain: %d, num_tokens_predicted: %d, stopped_eos: %d, stopped_word: %d, stopped_limit: %d, stopping_word: %s",
        common_token_to_piece(parent_ctx->ctx, token_with_probs.tok),
        tokens_to_output_formatted_string(parent_ctx->ctx, token_with_probs.tok).c_str(),
        has_next_token,
        n_remain,
        num_tokens_predicted,
        stopped_eos,
        stopped_word,
        stopped_limit,
        stopping_word.c_str()
    );
    return token_with_probs;
}

completion_chat_output llama_rn_context_completion::parseChatOutput(bool is_partial) {
    common_chat_parser_params syntax;
    syntax.format = static_cast<common_chat_format>(current_chat_format);
    syntax.reasoning_format = current_reasoning_format;
    syntax.thinking_forced_open = current_thinking_forced_open;
    syntax.parse_tool_calls = true;

    // Load the PEG parser if available (required for COMMON_CHAT_FORMAT_PEG_* formats)
    if (!current_chat_parser.empty()) {
        syntax.parser.load(current_chat_parser);
    }

    common_chat_msg parsed_msg = common_chat_parse(prefill_text + generated_text, is_partial, syntax);

    completion_chat_output result;

    result.content = parsed_msg.content;
    result.reasoning_content = parsed_msg.reasoning_content;
    result.accumulated_text = prefill_text + generated_text;
    result.tool_calls = parsed_msg.tool_calls;

    return result;
}

std::vector<float> llama_rn_context_completion::embedding(common_params &embd_params)
{
    llama_memory_clear(llama_get_memory(parent_ctx->ctx), true);

    rewind();
    llama_perf_context_reset(parent_ctx->ctx);
    if (!initSampling()) {
        throw std::runtime_error("Failed to initialize sampling");
    }
    beginCompletion();
    loadPrompt({});
    doCompletion();
    endCompletion();

    static const int n_embd = llama_model_n_embd(llama_get_model(parent_ctx->ctx));
    if (!embd_params.embedding)
    {
        LOG_WARNING("embedding disabled, embedding: %s", embd_params.embedding);
        return std::vector<float>(n_embd, 0.0f);
    }
    float *data;

    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        data = llama_get_embeddings(parent_ctx->ctx);
    } else {
        data = llama_get_embeddings_seq(parent_ctx->ctx, 0);
    }

    if (!data) {
        return std::vector<float>(n_embd, 0.0f);
    }
    std::vector<float> embedding(data, data + n_embd), out(data, data + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd, embd_params.embd_normalize);
    return out;
}

std::vector<float> llama_rn_context_completion::rerank(const std::string &query, const std::vector<std::string> &documents)
{
    std::vector<float> scores;

    // Check if this model supports reranking (requires rank pooling type)
    const enum llama_pooling_type pooling_type = llama_pooling_type(parent_ctx->ctx);
    if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
        throw std::runtime_error("reranking not supported, pooling_type: " + std::to_string(pooling_type));
    }

    if (!parent_ctx->params.embedding) {
        throw std::runtime_error("embedding disabled but required for reranking");
    }

    const llama_vocab * vocab = llama_model_get_vocab(parent_ctx->model);
    std::vector<llama_token> query_tokens = common_tokenize(vocab, query, false, true);

    scores.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); ++i) {
        rewind();
        embd = {};

        const std::string & document = documents[i];

        std::vector<llama_token> doc_tokens = common_tokenize(vocab, document, false, true);

        std::vector<llama_token> rerank_tokens = format_rerank_tokens(vocab, query_tokens, doc_tokens);

        llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);

        // Process the rerank input
        try {
            parent_ctx->params.prompt = tokens_to_str(parent_ctx->ctx, rerank_tokens.begin(), rerank_tokens.end());
            initSampling();
            loadPrompt({}); // No media paths for rerank
            beginCompletion();
            doCompletion();

            // Get the rerank score (single embedding value for rank pooling)
            float *data = llama_get_embeddings_seq(parent_ctx->ctx, 0);
            if (data) {
                scores.push_back(data[0]); // For rank pooling, the score is the first (and only) dimension
            } else {
                scores.push_back(-1e6f); // Default low score if computation failed
            }
        } catch (const std::exception &e) {
            LOG_WARNING("rerank computation failed for document %zu: %s", i, e.what());
            scores.push_back(-1e6f);
        }
        endCompletion();

        // Clear KV cache again to prepare for next document or restore original state
        llama_memory_clear(llama_get_memory(parent_ctx->ctx), false);
    }

    return scores;
}

std::string llama_rn_context_completion::bench(int pp, int tg, int pl, int nr) {
    if (is_predicting) {
        LOG_ERROR("cannot benchmark while predicting", "");
        return std::string("{}");
    }

    if (pp <= 0 || tg <= 0 || pl <= 0 || nr <= 0) {
        LOG_ERROR("invalid benchmark parameters pp=%d tg=%d pl=%d nr=%d", pp, tg, pl, nr);
        return std::string("{}");
    }

    is_predicting = true;

    auto * ctx = parent_ctx->ctx;
    auto * model = parent_ctx->model;
    auto * mem = llama_get_memory(ctx);

    const bool is_pp_shared = parent_ctx->params.is_pp_shared;
    const bool kv_unified   = parent_ctx->params.kv_unified;
    const int32_t n_batch   = parent_ctx->params.n_batch;
    const int32_t n_ubatch  = parent_ctx->params.n_ubatch;
    const int32_t flash_attn = static_cast<int32_t>(parent_ctx->params.flash_attn_type);
    const int32_t n_gpu_layers = parent_ctx->params.n_gpu_layers;
    const int32_t n_threads = llama_n_threads(ctx);
    const int32_t n_threads_batch = llama_n_threads_batch(ctx);
    const int32_t n_kv_max = llama_n_ctx(ctx);

    const int32_t n_ctx_req = is_pp_shared
        ? (kv_unified ? pp : pl * pp) + pl * tg
        : pl * (pp + tg);

    if (n_ctx_req > n_kv_max) {
        LOG_ERROR("benchmark requires n_ctx=%d but only %d available", n_ctx_req, n_kv_max);
        endCompletion();
        return std::string("{}");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;

    auto get_token_rand = [n_vocab]() -> llama_token {
        if (n_vocab <= 0) {
            return 0;
        }
        return std::rand() % n_vocab;
    };

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    auto decode_helper = [ctx](llama_batch & batch_ref, int32_t n_batch_ref, bool synchronize) -> bool {
        const int32_t total = batch_ref.n_tokens;
        for (int32_t i = 0; i < total; i += n_batch_ref) {
            const int32_t n_tokens_step = std::min(n_batch_ref, total - i);

            llama_batch batch_view = {
                n_tokens_step,
                batch_ref.token    + i,
                nullptr,
                batch_ref.pos      + i,
                batch_ref.n_seq_id + i,
                batch_ref.seq_id   + i,
                batch_ref.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERROR("llama_decode() failed during benchmark, n_batch=%d ret=%d", n_batch_ref, ret);
                return false;
            }

            if (synchronize) {
                llama_synchronize(ctx);
            }
        }

        return true;
    };

    // warm up like the CLI benchmark
    llama_batch_clear(&batch);
    const int warmup_tokens = std::min(16, n_kv_max);
    for (int i = 0; i < warmup_tokens; ++i) {
        llama_batch_add(&batch, get_token_rand(), i, {0}, i == warmup_tokens - 1);
    }
    if (!decode_helper(batch, n_batch, true)) {
        llama_batch_free(batch);
        endCompletion();
        return std::string("{}");
    }

    double acc_t_pp = 0.0;
    double acc_t_tg = 0.0;
    double acc_speed_pp = 0.0;
    double acc_speed_tg = 0.0;
    double acc_t_total = 0.0;
    double acc_speed_total = 0.0;

    int runs_completed = 0;

    for (int run = 0; run < nr && !is_interrupted; ++run) {
        bool run_failed = false;

        llama_batch_clear(&batch);

        const int prompt_sequences = is_pp_shared ? 1 : pl;
        for (int seq = 0; seq < prompt_sequences; ++seq) {
            for (int i = 0; i < pp; ++i) {
                llama_batch_add(&batch, get_token_rand(), i, {static_cast<llama_seq_id>(seq)}, i == pp - 1);
            }
        }

        llama_memory_clear(mem, false);

        const auto t_pp_start = lm_ggml_time_us();
        if (!decode_helper(batch, n_batch, false)) {
            run_failed = true;
            break;
        }

        llama_synchronize(ctx);
        const auto t_pp_end = lm_ggml_time_us();

        if (is_pp_shared && pl > 1) {
            for (int32_t seq = 1; seq < pl; ++seq) {
                llama_memory_seq_cp(mem, 0, seq, -1, -1);
            }

            if (!kv_unified) {
                llama_batch_clear(&batch);
                llama_batch_add(&batch, get_token_rand(), pp, {0}, true);
                if (!decode_helper(batch, n_batch, true)) {
                    run_failed = true;
                    break;
                }
                llama_memory_seq_rm(mem, 0, pp, -1);
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_start = lm_ggml_time_us();

        for (int i = 0; i < tg; ++i) {
            llama_batch_clear(&batch);

            for (int seq = 0; seq < pl; ++seq) {
                llama_batch_add(&batch, get_token_rand(), pp + i, {static_cast<llama_seq_id>(seq)}, true);
            }

            if (!decode_helper(batch, n_batch, true)) {
                run_failed = true;
                break;
            }
        }

        if (run_failed) {
            break;
        }

        const auto t_tg_end = lm_ggml_time_us();

        const double t_pp = (t_pp_end - t_pp_start) / 1e6;
        const double t_tg = (t_tg_end - t_tg_start) / 1e6;
        const double t_total = t_pp + t_tg;

        const double prompt_tokens = is_pp_shared ? static_cast<double>(pp) : static_cast<double>(pl * pp);
        const double generated_tokens = static_cast<double>(pl * tg);

        const double speed_pp = t_pp > 0.0 ? prompt_tokens / t_pp : 0.0;
        const double speed_tg = t_tg > 0.0 ? generated_tokens / t_tg : 0.0;
        const double speed_total = t_total > 0.0 ? (prompt_tokens + generated_tokens) / t_total : 0.0;

        acc_t_pp += t_pp;
        acc_t_tg += t_tg;
        acc_speed_pp += speed_pp;
        acc_speed_tg += speed_tg;
        acc_t_total += t_total;
        acc_speed_total += speed_total;

        ++runs_completed;
    }

    llama_memory_clear(mem, false);

    const double divisor = runs_completed > 0 ? static_cast<double>(runs_completed) : 1.0;

    json result_json = {
        {"n_kv_max", n_kv_max},
        {"n_batch", n_batch},
        {"n_ubatch", n_ubatch},
        {"flash_attn", flash_attn},
        {"is_pp_shared", is_pp_shared ? 1 : 0},
        {"n_gpu_layers", n_gpu_layers},
        {"n_threads", n_threads},
        {"n_threads_batch", n_threads_batch},
        {"pp", pp},
        {"tg", tg},
        {"pl", pl},
        {"n_kv", n_ctx_req},
        {"t_pp", acc_t_pp / divisor},
        {"speed_pp", acc_speed_pp / divisor},
        {"t_tg", acc_t_tg / divisor},
        {"speed_tg", acc_speed_tg / divisor},
        {"t", acc_t_total / divisor},
        {"speed", acc_speed_total / divisor}
    };

    llama_batch_free(batch);
    endCompletion();

    return result_json.dump();
}

void llama_rn_context_completion::processMedia(
    const std::string &prompt,
    const std::vector<std::string> &media_paths
) {
    if (!parent_ctx->isMultimodalEnabled()) {
        throw std::runtime_error("Multimodal is not enabled but image paths are provided");
    }

    // Delegate to the mtmd_wrapper method
    // For non-parallel mode, use the global bitmap_past_hashes from mtmd_wrapper
    parent_ctx->mtmd_wrapper->processMedia(
        parent_ctx->ctx,
        prompt,
        media_paths,
        parent_ctx->n_ctx,
        parent_ctx->params.n_batch,
        n_past,
        embd,
        context_full,
        ctx_sampling,
        parent_ctx->mtmd_wrapper->bitmap_past_hashes,
        0  // Use sequence ID 0 for non-parallel mode
    );
}

} // namespace rnllama
