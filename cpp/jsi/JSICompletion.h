#pragma once

#include "JSIHelpers.h"
#include "JSINativeHeaders.h"
#include <string>
#include <vector>

namespace rnllama_jsi {

    inline jsi::Array createToolCalls(jsi::Runtime& runtime, const std::vector<common_chat_tool_call>& tool_calls) {
        jsi::Array arr(runtime, tool_calls.size());
        for (size_t i = 0; i < tool_calls.size(); ++i) {
            jsi::Object tool(runtime);
            tool.setProperty(runtime, "type", jsi::String::createFromUtf8(runtime, "function"));

            jsi::Object fn(runtime);
            fn.setProperty(runtime, "name", jsi::String::createFromUtf8(runtime, tool_calls[i].name));
            fn.setProperty(runtime, "arguments", jsi::String::createFromUtf8(runtime, tool_calls[i].arguments));
            tool.setProperty(runtime, "function", fn);

            if (!tool_calls[i].id.empty()) {
                tool.setProperty(runtime, "id", jsi::String::createFromUtf8(runtime, tool_calls[i].id));
            } else {
                tool.setProperty(runtime, "id", jsi::Value::null());
            }

            arr.setValueAtIndex(runtime, i, tool);
        }
        return arr;
    }

    inline void setChatOutputFields(jsi::Runtime& runtime, jsi::Object& target, const rnllama::completion_chat_output& output) {
        if (!output.content.empty()) {
            target.setProperty(runtime, "content", jsi::String::createFromUtf8(runtime, output.content));
        }
        if (!output.reasoning_content.empty()) {
            target.setProperty(runtime, "reasoning_content", jsi::String::createFromUtf8(runtime, output.reasoning_content));
        }
        if (!output.tool_calls.empty()) {
            target.setProperty(runtime, "tool_calls", createToolCalls(runtime, output.tool_calls));
        }
        if (!output.accumulated_text.empty()) {
            target.setProperty(runtime, "accumulated_text", jsi::String::createFromUtf8(runtime, output.accumulated_text));
        }
    }

    inline jsi::Array createCompletionProbabilities(
        jsi::Runtime& runtime,
        rnllama::llama_rn_context* ctx,
        const std::vector<rnllama::completion_token_output>& probs_vec
    ) {
        jsi::Array out(runtime, probs_vec.size());
        for (size_t i = 0; i < probs_vec.size(); ++i) {
            const auto &prob = probs_vec[i];

            jsi::Array probsForToken(runtime, prob.probs.size());
            for (size_t j = 0; j < prob.probs.size(); ++j) {
                jsi::Object p(runtime);
                std::string tokStr = rnllama::tokens_to_output_formatted_string(ctx->ctx, prob.probs[j].tok);
                if (tokStr.empty()) tokStr = "<UNKNOWN>";
                p.setProperty(runtime, "tok_str", jsi::String::createFromUtf8(runtime, tokStr));
                p.setProperty(runtime, "prob", (double)prob.probs[j].prob);
                probsForToken.setValueAtIndex(runtime, j, p);
            }

            std::string tokStr = rnllama::tokens_to_output_formatted_string(ctx->ctx, prob.tok);
            if (tokStr.empty()) tokStr = "<UNKNOWN>";

            jsi::Object completionProb(runtime);
            completionProb.setProperty(runtime, "content", jsi::String::createFromUtf8(runtime, tokStr));
            completionProb.setProperty(runtime, "probs", probsForToken);

            out.setValueAtIndex(runtime, i, completionProb);
        }
        return out;
    }

    inline jsi::Object createTokenProb(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx, const rnllama::completion_token_output::token_prob& p) {
        jsi::Object res(runtime);
        std::string tokStr = rnllama::tokens_to_output_formatted_string(ctx->ctx, p.tok);
        if (tokStr.empty()) tokStr = "<UNKNOWN>";
        res.setProperty(runtime, "tok_str", jsi::String::createFromUtf8(runtime, tokStr));
        res.setProperty(runtime, "prob", (double)p.prob);
        return res;
    }

    inline jsi::Object createTokenResult(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx, const rnllama::completion_token_output& token) {
        jsi::Object res(runtime);
        res.setProperty(runtime, "token", jsi::String::createFromUtf8(runtime, token.text));

        if (!token.probs.empty()) {
            jsi::Array probs(runtime, token.probs.size());
            for (size_t i = 0; i < token.probs.size(); i++) {
                jsi::Object prob(runtime);
                prob.setProperty(runtime, "tok", (int)token.probs[i].tok);
                prob.setProperty(runtime, "prob", (double)token.probs[i].prob);
                probs.setValueAtIndex(runtime, i, prob);
            }
            res.setProperty(runtime, "probs", probs);
        }

        if (token.probs.size() > 0) {
            jsi::Array probs = jsi::Array(runtime, token.probs.size());
            for (size_t i = 0; i < token.probs.size(); i++) {
                probs.setValueAtIndex(runtime, i, createTokenProb(runtime, ctx, token.probs[i]));
            }
            
            jsi::Object completionProb(runtime);
            completionProb.setProperty(runtime, "content", jsi::String::createFromUtf8(runtime, token.text));
            completionProb.setProperty(runtime, "probs", probs);
            
            jsi::Array completionProbs = jsi::Array(runtime, 1);
            completionProbs.setValueAtIndex(runtime, 0, completionProb);
            
            res.setProperty(runtime, "completion_probabilities", completionProbs);
        }
        
        // requestId for parallel
        if (token.request_id != -1) {
            res.setProperty(runtime, "requestId", (int)token.request_id);
        }
        
        return res;
    }

    inline jsi::Object createCompletionResult(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx) {
        jsi::Object res(runtime);
        res.setProperty(runtime, "text", jsi::String::createFromUtf8(runtime, ctx->completion->generated_text));

        res.setProperty(runtime, "chat_format", ctx->completion->current_chat_format);

        // Parse final chat output if available
        if (!ctx->completion->is_interrupted) {
            try {
                auto final_output = ctx->completion->parseChatOutput(false);
                setChatOutputFields(runtime, res, final_output);
            } catch (...) {
                // Ignore parsing errors
            }
        }

        res.setProperty(
            runtime,
            "completion_probabilities",
            createCompletionProbabilities(runtime, ctx, ctx->completion->generated_token_probs)
        );
        res.setProperty(runtime, "tokens_predicted", (double)ctx->completion->num_tokens_predicted);
        res.setProperty(runtime, "tokens_evaluated", (double)ctx->completion->num_prompt_tokens);
        res.setProperty(runtime, "truncated", ctx->completion->truncated);
        res.setProperty(runtime, "context_full", ctx->completion->context_full);
        res.setProperty(runtime, "interrupted", ctx->completion->is_interrupted);
        res.setProperty(runtime, "stopped_eos", ctx->completion->stopped_eos);
        res.setProperty(runtime, "stopped_word", ctx->completion->stopped_word);
        res.setProperty(runtime, "stopped_limit", ctx->completion->stopped_limit);
        res.setProperty(runtime, "stopping_word", jsi::String::createFromUtf8(runtime, ctx->completion->stopping_word));
        res.setProperty(runtime, "tokens_cached", (double)ctx->completion->n_past);

        if (ctx->isVocoderEnabled() && ctx->tts_wrapper != nullptr && !ctx->tts_wrapper->audio_tokens.empty()) {
            jsi::Array audioTokens(runtime, ctx->tts_wrapper->audio_tokens.size());
            for (size_t i = 0; i < ctx->tts_wrapper->audio_tokens.size(); i++) {
                audioTokens.setValueAtIndex(runtime, i, (double)ctx->tts_wrapper->audio_tokens[i]);
            }
            res.setProperty(runtime, "audio_tokens", audioTokens);
        }

        const auto timings = llama_perf_context(ctx->ctx);

        jsi::Object timingsObj(runtime);
        timingsObj.setProperty(runtime, "cache_n", (double)ctx->completion->n_past);
        timingsObj.setProperty(runtime, "prompt_n", (double)timings.n_p_eval);
        timingsObj.setProperty(runtime, "prompt_ms", (double)timings.t_p_eval_ms);
        const double prompt_per_token_ms = timings.n_p_eval > 0 ? timings.t_p_eval_ms / timings.n_p_eval : 0.0;
        const double prompt_per_second = timings.t_p_eval_ms > 0 ? 1e3 / timings.t_p_eval_ms * timings.n_p_eval : 0.0;
        timingsObj.setProperty(runtime, "prompt_per_token_ms", prompt_per_token_ms);
        timingsObj.setProperty(runtime, "prompt_per_second", prompt_per_second);

        timingsObj.setProperty(runtime, "predicted_n", (double)timings.n_eval);
        timingsObj.setProperty(runtime, "predicted_ms", (double)timings.t_eval_ms);
        const double predicted_per_token_ms = timings.n_eval > 0 ? timings.t_eval_ms / timings.n_eval : 0.0;
        const double predicted_per_second = timings.t_eval_ms > 0 ? 1e3 / timings.t_eval_ms * timings.n_eval : 0.0;
        timingsObj.setProperty(runtime, "predicted_per_token_ms", predicted_per_token_ms);
        timingsObj.setProperty(runtime, "predicted_per_second", predicted_per_second);

        res.setProperty(runtime, "timings", timingsObj);

        return res;
    }
    
    // Overload for parallel slot
    inline jsi::Object createCompletionResult(jsi::Runtime& runtime, rnllama::llama_rn_slot* slot) {
        jsi::Object res(runtime);
        res.setProperty(runtime, "text", jsi::String::createFromUtf8(runtime, slot->generated_text));

        res.setProperty(runtime, "chat_format", slot->current_chat_format);

        try {
            auto final_output = slot->parseChatOutput(false);
            setChatOutputFields(runtime, res, final_output);
        } catch (...) {
            // ignore
        }

        auto ctx = slot->parent_ctx;
        if (ctx != nullptr) {
            res.setProperty(
                runtime,
                "completion_probabilities",
                createCompletionProbabilities(runtime, ctx, slot->generated_token_probs)
            );
        }

        res.setProperty(runtime, "tokens_predicted", (double)slot->num_tokens_predicted);
        res.setProperty(runtime, "tokens_evaluated", (double)slot->num_prompt_tokens);
        res.setProperty(runtime, "truncated", slot->truncated);
        res.setProperty(runtime, "context_full", slot->context_full);
        res.setProperty(runtime, "interrupted", slot->is_interrupted);
        res.setProperty(runtime, "stopped_eos", slot->stopped_eos);
        res.setProperty(runtime, "stopped_word", slot->stopped_word);
        res.setProperty(runtime, "stopped_limit", slot->stopped_limit);
        res.setProperty(runtime, "stopping_word", jsi::String::createFromUtf8(runtime, slot->stopping_word));
        res.setProperty(runtime, "tokens_cached", (double)slot->n_past);

        auto timings = slot->get_timings();
        jsi::Object timingsObj(runtime);
        timingsObj.setProperty(runtime, "cache_n", (double)timings.cache_n);
        timingsObj.setProperty(runtime, "prompt_n", (double)timings.prompt_n);
        timingsObj.setProperty(runtime, "prompt_ms", (double)timings.prompt_ms);
        timingsObj.setProperty(runtime, "prompt_per_token_ms", (double)timings.prompt_per_token_ms);
        timingsObj.setProperty(runtime, "prompt_per_second", (double)timings.prompt_per_second);
        timingsObj.setProperty(runtime, "predicted_n", (double)timings.predicted_n);
        timingsObj.setProperty(runtime, "predicted_ms", (double)timings.predicted_ms);
        timingsObj.setProperty(runtime, "predicted_per_token_ms", (double)timings.predicted_per_token_ms);
        timingsObj.setProperty(runtime, "predicted_per_second", (double)timings.predicted_per_second);
        res.setProperty(runtime, "timings", timingsObj);

        return res;
    }
}
