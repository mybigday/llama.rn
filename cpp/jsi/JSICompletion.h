#pragma once

#include "JSIHelpers.h"
#include "JSINativeHeaders.h"
#include <string>
#include <vector>

// Completion results are assembled as json on the thread that produced them
// and converted once with fromJson() on the JS thread.
namespace rnllama_jsi {

    inline json toolCallsJson(const std::vector<common_chat_tool_call>& tool_calls) {
        json arr = json::array();
        for (const auto& tc : tool_calls) {
            json tool = json::object({
                {"type", "function"},
                {"function", json::object({{"name", tc.name}, {"arguments", tc.arguments}})},
                {"id", tc.id.empty() ? json(nullptr) : json(tc.id)},
            });
            arr.push_back(std::move(tool));
        }
        return arr;
    }

    inline void addChatOutputFields(json& target, const rnllama::completion_chat_output& output) {
        if (!output.content.empty()) {
            target["content"] = output.content;
        }
        if (!output.reasoning_content.empty()) {
            target["reasoning_content"] = output.reasoning_content;
        }
        if (!output.tool_calls.empty()) {
            target["tool_calls"] = toolCallsJson(output.tool_calls);
        }
        if (!output.accumulated_text.empty()) {
            target["accumulated_text"] = output.accumulated_text;
        }
    }

    inline std::string tokenPiece(rnllama::llama_rn_context* ctx, llama_token tok) {
        std::string piece = (ctx != nullptr && ctx->ctx != nullptr)
            ? rnllama::tokens_to_output_formatted_string(ctx->ctx, tok)
            : "";
        return piece.empty() ? "<UNKNOWN>" : piece;
    }

    inline json completionProbabilitiesJson(
        rnllama::llama_rn_context* ctx,
        const std::vector<rnllama::completion_token_output>& probs_vec
    ) {
        json out = json::array();
        if (ctx == nullptr || ctx->ctx == nullptr) {
            return out;
        }
        for (const auto& prob : probs_vec) {
            json probsForToken = json::array();
            for (const auto& p : prob.probs) {
                probsForToken.push_back(json::object({
                    {"tok_str", tokenPiece(ctx, p.tok)},
                    {"prob", (double) p.prob},
                }));
            }
            out.push_back(json::object({
                {"content", tokenPiece(ctx, prob.tok)},
                {"probs", std::move(probsForToken)},
            }));
        }
        return out;
    }

    inline json tokenResultJson(rnllama::llama_rn_context* ctx, const rnllama::completion_token_output& token) {
        json res = json::object({{"token", token.text}});

        if (!token.probs.empty()) {
            json probs = json::array();
            json probsWithText = json::array();
            for (const auto& p : token.probs) {
                probs.push_back(json::object({{"tok", (int) p.tok}, {"prob", (double) p.prob}}));
                probsWithText.push_back(json::object({{"tok_str", tokenPiece(ctx, p.tok)}, {"prob", (double) p.prob}}));
            }
            res["probs"] = std::move(probs);
            res["completion_probabilities"] = json::array({
                json::object({{"content", token.text}, {"probs", std::move(probsWithText)}}),
            });
        }

        // requestId for parallel
        if (token.request_id != -1) {
            res["requestId"] = (int) token.request_id;
        }
        return res;
    }

    inline json timingsJson(const rnllama::slot_timings& t) {
        return json::object({
            {"cache_n", t.cache_n},
            {"prompt_n", t.prompt_n},
            {"prompt_ms", t.prompt_ms},
            {"prompt_per_token_ms", t.prompt_per_token_ms},
            {"prompt_per_second", t.prompt_per_second},
            {"predicted_n", t.predicted_n},
            {"predicted_ms", t.predicted_ms},
            {"predicted_per_token_ms", t.predicted_per_token_ms},
            {"predicted_per_second", t.predicted_per_second},
        });
    }

    // Single-context completion result. The large numeric payloads stay out
    // of the json body: a json node per sample would cost more than the
    // direct jsi::Array build it replaces.
    struct CompletionResult {
        json body;
        std::vector<llama_token> audio_tokens;
        std::vector<float> embeddings;

        jsi::Value toJsi(jsi::Runtime& rt) const {
            jsi::Object res = fromJson(rt, body).getObject(rt);
            if (!audio_tokens.empty()) {
                res.setProperty(rt, "audio_tokens", toJsNumberArray(rt, audio_tokens));
            }
            if (!embeddings.empty()) {
                res.setProperty(rt, "embeddings", toJsNumberArray(rt, embeddings));
            }
            return res;
        }
    };

    inline CompletionResult completionResult(rnllama::llama_rn_context* ctx) {
        if (ctx == nullptr) {
            throw std::runtime_error("RNLLAMA_NULL_CONTEXT");
        }
        if (ctx->completion == nullptr) {
            throw std::runtime_error("RNLLAMA_NULL_COMPLETION");
        }
        if (ctx->ctx == nullptr) {
            throw std::runtime_error("RNLLAMA_NULL_LLAMA_CONTEXT");
        }
        auto& c = *ctx->completion;

        CompletionResult result;
        json& res = result.body;
        res = json::object({
            {"text", c.generated_text},
            {"chat_format", c.current_chat_format},
        });

        // Parse final chat output if available
        if (!c.is_interrupted) {
            try {
                addChatOutputFields(res, c.parseChatOutput(false));
            } catch (...) {
                // Ignore parsing errors
            }
        }

        res["completion_probabilities"] = completionProbabilitiesJson(ctx, c.generated_token_probs);
        res["tokens_predicted"] = c.num_tokens_predicted;
        res["tokens_evaluated"] = c.num_prompt_tokens;
        res["draft_tokens"] = c.num_draft_tokens;
        res["draft_tokens_accepted"] = c.num_draft_tokens_accepted;
        res["truncated"] = c.truncated;
        res["context_full"] = c.context_full;
        res["interrupted"] = c.is_interrupted;
        res["stopped_eos"] = c.stopped_eos;
        res["stopped_word"] = c.stopped_word;
        res["stopped_limit"] = c.stopped_limit;
        res["stopping_word"] = c.stopping_word;
        res["tokens_cached"] = c.n_past;

        if (ctx->isVocoderEnabled() && ctx->tts_wrapper != nullptr) {
            result.audio_tokens = ctx->tts_wrapper->audio_tokens;
        }
        if (!c.embeddings.empty()) {
            result.embeddings = c.embeddings;
            res["embedding_dim"] = c.embedding_dim;
        }

        const auto perf = llama_perf_context(ctx->ctx);
        rnllama::slot_timings t;
        t.cache_n = c.n_past;
        t.prompt_n = perf.n_p_eval;
        t.prompt_ms = perf.t_p_eval_ms;
        t.prompt_per_token_ms = perf.n_p_eval > 0 ? perf.t_p_eval_ms / perf.n_p_eval : 0.0;
        t.prompt_per_second = perf.t_p_eval_ms > 0 ? 1e3 / perf.t_p_eval_ms * perf.n_p_eval : 0.0;
        t.predicted_n = c.num_tokens_predicted;
        t.predicted_ms = c.t_token_generation * 1e3;
        t.predicted_per_token_ms = t.predicted_n > 0 ? t.predicted_ms / t.predicted_n : 0.0;
        t.predicted_per_second = c.t_token_generation > 0.0 ? t.predicted_n / c.t_token_generation : 0.0;
        res["timings"] = timingsJson(t);

        return result;
    }

    // Thread-safe copy of a slot's final state; the slot can be reused for
    // the next request as soon as the completion callback returns.
    struct ParallelCompletionResultSnapshot {
        int32_t request_id = -1;
        std::string text;
        int32_t chat_format = 0;
        bool stopped_eos = false;
        bool stopped_limit = false;
        bool stopped_word = false;
        bool context_full = false;
        bool incomplete = false;
        bool truncated = false;
        bool interrupted = false;
        std::string stopping_word;
        size_t tokens_predicted = 0;
        size_t tokens_evaluated = 0;
        size_t draft_tokens = 0;
        size_t draft_tokens_accepted = 0;
        llama_pos tokens_cached = 0;
        int32_t n_decoded = 0;
        std::string error_message;
        rnllama::slot_timings timings;
        std::vector<rnllama::completion_token_output> token_probs;
        rnllama::completion_chat_output final_output;
        bool has_final_output = false;
    };

    inline ParallelCompletionResultSnapshot captureParallelCompletionResult(
        rnllama::llama_rn_slot* slot
    ) {
        if (slot == nullptr) {
            throw std::runtime_error("RNLLAMA_NULL_SLOT");
        }

        ParallelCompletionResultSnapshot result;
        result.request_id = slot->request_id;
        result.text = slot->generated_text;
        result.chat_format = slot->current_chat_format;
        result.stopped_eos = slot->stopped_eos;
        result.stopped_limit = slot->stopped_limit;
        result.stopped_word = slot->stopped_word;
        result.context_full = slot->context_full;
        result.incomplete = slot->incomplete;
        result.truncated = slot->truncated;
        result.interrupted = slot->is_interrupted;
        result.stopping_word = slot->stopping_word;
        result.tokens_predicted = slot->num_tokens_predicted;
        result.tokens_evaluated = slot->num_prompt_tokens;
        result.draft_tokens = slot->num_draft_tokens;
        result.draft_tokens_accepted = slot->num_draft_tokens_accepted;
        result.tokens_cached = slot->n_past;
        result.n_decoded = slot->n_decoded;
        result.error_message = slot->error_message;
        result.timings = slot->get_timings();
        result.token_probs = slot->generated_token_probs;

        try {
            result.final_output = slot->parseChatOutput(false);
            result.has_final_output = true;
        } catch (...) {
            result.has_final_output = false;
        }

        return result;
    }

    inline ParallelCompletionResultSnapshot createQueuedCancellationSnapshot(int32_t request_id) {
        ParallelCompletionResultSnapshot result;
        result.request_id = request_id;
        result.interrupted = true;
        result.timings.cache_n = 0;
        result.timings.prompt_n = 0;
        result.timings.predicted_n = 0;
        return result;
    }

    inline json parallelCompletionResultJson(
        rnllama::llama_rn_context* ctx,
        const ParallelCompletionResultSnapshot& result
    ) {
        json res = json::object({
            {"requestId", result.request_id},
            {"text", result.text},
            {"chat_format", result.chat_format},
            {"stopped_eos", result.stopped_eos},
            {"stopped_limit", result.stopped_limit},
            {"stopped_word", result.stopped_word},
            {"context_full", result.context_full},
            {"incomplete", result.incomplete},
            {"truncated", result.truncated},
            {"interrupted", result.interrupted},
            {"stopping_word", result.stopping_word},
            {"tokens_predicted", result.tokens_predicted},
            {"tokens_evaluated", result.tokens_evaluated},
            {"draft_tokens", result.draft_tokens},
            {"draft_tokens_accepted", result.draft_tokens_accepted},
            {"tokens_cached", result.tokens_cached},
            {"n_decoded", result.n_decoded},
            {"completion_probabilities", completionProbabilitiesJson(ctx, result.token_probs)},
        });

        if (!result.error_message.empty()) {
            res["error"] = result.error_message;
        }
        if (result.has_final_output) {
            addChatOutputFields(res, result.final_output);
        }
        res["timings"] = timingsJson(result.timings);
        return res;
    }
}
