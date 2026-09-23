#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>
#include <string>

// Include rnllama headers
#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-tts.h"
#include "common.h"

using namespace rnllama;

// Test result tracking
struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;

    void run_test(const std::string& name, bool result) {
        total_tests++;
        std::cout << "TEST: " << name << " ... ";
        if (result) {
            std::cout << "PASSED" << std::endl;
            passed_tests++;
        } else {
            std::cout << "FAILED" << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
        std::cout << "Success rate: " << (100.0 * passed_tests / total_tests) << "%" << std::endl;
    }
};

// Test basic context creation and model loading
bool test_context_creation_and_model_loading() {
    try {
        llama_rn_context ctx;

        // Setup basic parameters for the tiny model
        common_params params;

        // Get the path to our test model (relative to build directory)
        std::string model_path = "../tiny-random-llama.gguf";
        if (!std::filesystem::exists(model_path)) {
            std::cout << "Test model not found at: " << model_path << std::endl;
            return false;
        }

        params.model.path = model_path;
        params.n_ctx = 512; // Small context for testing
        params.n_batch = 128;
        params.cpuparams.n_threads = 1;
        params.n_gpu_layers = 0; // CPU only for tests
        params.no_kv_offload = true; // Force CPU-only mode // CPU only for tests
        params.no_kv_offload = true; // Force CPU-only mode

        // Try to load the model
        bool load_result = ctx.loadModel(params);

        if (!load_result) {
            std::cout << "Failed to load model" << std::endl;
            return false;
        }

        // Check that model was loaded
        if (ctx.model == nullptr) {
            std::cout << "Model is null after loading" << std::endl;
            return false;
        }

        // Check that context was created
        if (ctx.ctx == nullptr) {
            std::cout << "Context is null after loading" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Test tokenization functionality
bool test_tokenization() {
    try {
        llama_rn_context ctx;

        // Setup parameters
        common_params params;
        params.model.path = "../tiny-random-llama.gguf";
        params.n_ctx = 512;
        params.n_batch = 128;
        params.cpuparams.n_threads = 1;
        params.n_gpu_layers = 0; // CPU only for tests
        params.no_kv_offload = true; // Force CPU-only mode

        if (!ctx.loadModel(params)) {
            return false;
        }

        // Test tokenization with a simple string
        std::string test_text = "Hello, world!";
        std::vector<std::string> empty_media;

        llama_rn_tokenize_result result = ctx.tokenize(test_text, empty_media);

        // Should have some tokens
        if (result.tokens.empty()) {
            std::cout << "No tokens produced for test text" << std::endl;
            return false;
        }

        // Should not have media for text-only input
        if (result.has_media) {
            std::cout << "Unexpected media flag for text-only input" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Test completion functionality
bool test_completion() {
    try {
        llama_rn_context ctx;

        // Setup parameters
        common_params params;
        params.model.path = "../tiny-random-llama.gguf";
        params.n_ctx = 512;
        params.n_batch = 128;
        params.cpuparams.n_threads = 1;
        params.n_gpu_layers = 0; // CPU only for tests
        params.no_kv_offload = true; // Force CPU-only mode
        params.n_predict = -1;

        if (!ctx.loadModel(params)) {
            std::cout << "Failed to load model for completion test" << std::endl;
            return false;
        }

        // Create completion context
        if (ctx.completion == nullptr) {
            ctx.completion = new llama_rn_context_completion(&ctx);
        }

        // Initialize sampling
        if (!ctx.completion->initSampling()) {
            std::cout << "Failed to initialize sampling" << std::endl;
            return false;
        }

        // Test prompt loading
        std::string test_prompt = "Hello";
        std::vector<std::string> empty_media;

        // Set the prompt in params for loadPrompt to use
        ctx.params.prompt = test_prompt;

        // Load the prompt
        ctx.completion->loadPrompt(empty_media);

        // Test completion setup
        ctx.completion->beginCompletion();

        // Check that completion was initialized properly
        if (!ctx.completion->is_predicting) {
            std::cout << "Completion not in predicting state" << std::endl;
            return false;
        }

        // Test token generation
        int tokens_generated = 0;
        std::string generated_text = "";
        while (ctx.completion->has_next_token && tokens_generated < 10) { // Generate only a few tokens
            completion_token_output token_output = ctx.completion->nextToken();

            // Check that we got a valid token
            if (token_output.tok == -1) {
                std::cout << "Generated invalid token" << std::endl;
                break;
            }

            // Convert token to text and add to generated string
            std::string token_str = tokens_to_output_formatted_string(ctx.ctx, token_output.tok);
            generated_text += token_str;

            tokens_generated++;

            // Check if we should stop
            if (ctx.completion->stopped_eos || ctx.completion->stopped_limit ||
                ctx.completion->stopped_word || ctx.completion->is_interrupted) {
                break;
            }
        }

        // Print the generated text
        std::cout << "Generated text: \"" << generated_text << "\"" << std::endl;

        // Test interruption
        ctx.completion->is_interrupted = true;

        // End completion
        ctx.completion->endCompletion();

        // Verify state after completion
        if (ctx.completion->is_predicting) {
            std::cout << "Completion still in predicting state after end" << std::endl;
            return false;
        }

        // Check that we generated some tokens (even if just 1)
        if (tokens_generated == 0) {
            std::cout << "No tokens were generated" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Shared helper: run one classic (non-parallel) completion of up to `max_tokens`
// tokens through the same rewind -> loadPrompt -> beginCompletion -> doCompletion
// -> endCompletion sequence the JSI completion() binding uses.
static int run_classic_completion(llama_rn_context& ctx, const std::string& prompt, int max_tokens) {
    std::vector<std::string> empty_media;
    ctx.completion->rewind();
    if (!ctx.completion->initSampling()) {
        return -1;
    }
    ctx.params.prompt = prompt;
    ctx.completion->loadPrompt(empty_media);
    ctx.completion->beginCompletion();

    int generated = 0;
    while (ctx.completion->has_next_token && generated < max_tokens) {
        completion_token_output out = ctx.completion->doCompletion();
        if (out.tok == -1) break;
        generated++;
    }
    ctx.completion->is_interrupted = true;
    ctx.completion->endCompletion();
    return generated;
}

static bool setup_completion_context(llama_rn_context& ctx) {
    common_params params;
    params.model.path = "../tiny-random-llama.gguf";
    params.n_ctx = 512;
    params.n_batch = 128;
    params.cpuparams.n_threads = 1;
    params.n_gpu_layers = 0;
    params.no_kv_offload = true;
    params.n_predict = -1;
    if (!ctx.loadModel(params)) {
        std::cout << "Failed to load model" << std::endl;
        return false;
    }
    if (ctx.completion == nullptr) {
        ctx.completion = new llama_rn_context_completion(&ctx);
    }
    return true;
}

// Regression test for #401 (2): generated_token_probs must be reset by rewind(),
// otherwise completion_probabilities accumulates across completion() calls.
bool test_completion_probabilities_reset_between_completions() {
    try {
        llama_rn_context ctx;
        if (!setup_completion_context(ctx)) return false;

        ctx.params.sampling.n_probs = 3;

        const int first = run_classic_completion(ctx, "Hello", 4);
        if (first <= 0) {
            std::cout << "First completion generated no tokens" << std::endl;
            return false;
        }
        const size_t first_probs = ctx.completion->generated_token_probs.size();
        if (first_probs != (size_t) first) {
            std::cout << "First completion: expected " << first << " prob entries, got " << first_probs << std::endl;
            return false;
        }

        const int second = run_classic_completion(ctx, "World", 4);
        if (second <= 0) {
            std::cout << "Second completion generated no tokens" << std::endl;
            return false;
        }
        const size_t second_probs = ctx.completion->generated_token_probs.size();
        if (second_probs != (size_t) second) {
            std::cout << "Second completion: expected " << second << " prob entries, got " << second_probs
                      << " (stale entries from the first completion leaked)" << std::endl;
            return false;
        }

        for (const auto& entry : ctx.completion->generated_token_probs) {
            if (entry.probs.empty() || entry.probs.size() > 3) {
                std::cout << "Unexpected probs size " << entry.probs.size() << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// #401 (3): post_sampling_probs=false must report a raw softmax over the logits
// even when the sampler chain truncates candidates (top_k = 1 here), while the
// default post-sampling mode reflects that truncation.
bool test_n_probs_post_sampling_vs_raw() {
    try {
        llama_rn_context ctx;
        if (!setup_completion_context(ctx)) return false;

        ctx.params.sampling.n_probs = 4;
        ctx.params.sampling.top_k = 1;
        ctx.params.sampling.temp = 1.0f;

        // Post-sampling: top_k = 1 leaves a single candidate with p == 1
        ctx.post_sampling_probs = true;
        if (run_classic_completion(ctx, "Hello", 2) <= 0) return false;
        for (const auto& entry : ctx.completion->generated_token_probs) {
            if (entry.probs.size() != 1 || std::fabs(entry.probs[0].prob - 1.0f) > 1e-4f) {
                std::cout << "post-sampling: expected 1 candidate with p=1, got "
                          << entry.probs.size() << " (p0=" << (entry.probs.empty() ? -1.f : entry.probs[0].prob) << ")" << std::endl;
                return false;
            }
        }

        // Raw: full top-4 softmax, sorted descending, sum < 1 over a large vocab
        ctx.post_sampling_probs = false;
        if (run_classic_completion(ctx, "Hello", 2) <= 0) return false;
        for (const auto& entry : ctx.completion->generated_token_probs) {
            if (entry.probs.size() != 4) {
                std::cout << "raw: expected 4 entries, got " << entry.probs.size() << std::endl;
                return false;
            }
            float sum = 0.0f;
            for (size_t i = 0; i < entry.probs.size(); ++i) {
                const float p = entry.probs[i].prob;
                if (p <= 0.0f || p > 1.0f || (i > 0 && p > entry.probs[i - 1].prob)) {
                    std::cout << "raw: bad ordering / range at " << i << std::endl;
                    return false;
                }
                sum += p;
            }
            if (sum > 1.0f + 1e-4f) {
                std::cout << "raw: top-4 probabilities sum to " << sum << std::endl;
                return false;
            }
            // The greedy pick (top_k = 1) must be the raw argmax as well
            if (entry.probs[0].tok != entry.tok) {
                std::cout << "raw: argmax " << entry.probs[0].tok << " != sampled " << entry.tok << std::endl;
                return false;
            }
        }
        ctx.params.sampling.top_k = 40;
        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Regression test for #401 (1): logit_bias is a std::vector<llama_logit_bias> and
// must be populated with push_back (indexing an empty vector is an OOB write).
// A huge positive bias on one token must make the sampler pick it every time.
bool test_logit_bias_forces_token() {
    try {
        llama_rn_context ctx;
        if (!setup_completion_context(ctx)) return false;

        const llama_vocab * vocab = llama_model_get_vocab(ctx.model);
        const llama_token eos = llama_vocab_eos(vocab);
        // Pick a regular token that is not EOS so generation keeps going.
        llama_token forced = 5;
        if (forced == eos) forced = 6;

        ctx.params.sampling.logit_bias.clear();
        ctx.params.sampling.logit_bias.push_back({ forced, 1000.0f });
        ctx.params.sampling.logit_bias.push_back({ eos, -INFINITY });

        std::vector<std::string> empty_media;
        ctx.completion->rewind();
        if (!ctx.completion->initSampling()) return false;
        ctx.params.prompt = "Hello";
        ctx.completion->loadPrompt(empty_media);
        ctx.completion->beginCompletion();

        int generated = 0;
        while (ctx.completion->has_next_token && generated < 4) {
            completion_token_output out = ctx.completion->doCompletion();
            if (out.tok == -1) break;
            if (out.tok != forced) {
                std::cout << "Expected biased token " << forced << ", got " << out.tok << std::endl;
                ctx.completion->is_interrupted = true;
                ctx.completion->endCompletion();
                return false;
            }
            generated++;
        }
        ctx.completion->is_interrupted = true;
        ctx.completion->endCompletion();
        ctx.params.sampling.logit_bias.clear();
        return generated == 4;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Test that partial init failures return false instead of dereferencing a null context.
bool test_context_init_failure_is_graceful() {
    try {
        llama_rn_context ctx;

        common_params params;
        params.model.path = "../tiny-random-llama.gguf";
        params.n_ctx = 512;
        params.n_batch = 0;
        params.n_ubatch = 0;
        params.cpuparams.n_threads = 1;
        params.n_gpu_layers = 0;
        params.no_kv_offload = true;

        bool load_result = ctx.loadModel(params);

        if (load_result) {
            std::cout << "Expected loadModel to fail when n_batch and n_ubatch are zero" << std::endl;
            return false;
        }

        if (ctx.model == nullptr) {
            std::cout << "Expected model to load before context creation failed" << std::endl;
            return false;
        }

        if (ctx.ctx != nullptr) {
            std::cout << "Context should remain null after failed initialization" << std::endl;
            return false;
        }

        if (ctx.n_ctx != 0) {
            std::cout << "n_ctx should remain unset after failed initialization" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

// Test utility functions
bool test_utilities() {
    try {
        // Test kv_cache_type_from_str function
        ggml_type cache_type = kv_cache_type_from_str("f16");
        // Just ensure it doesn't crash and returns some valid value

        // Test flash attention type parsing
        enum llama_flash_attn_type flash_type = flash_attn_type_from_str("auto");
        // Just ensure it doesn't crash

        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

bool test_completion_generation_timing() {
    try {
        llama_rn_context_completion completion(nullptr);

        completion.startGenerationTiming();
        completion.t_start_generation = ggml_time_us() - 1000;
        completion.num_tokens_predicted = 3;
        completion.updateGenerationTiming();

        if (completion.t_token_generation <= 0.0) {
            std::cout << "Generation timing was not populated" << std::endl;
            return false;
        }

        completion.resetGenerationTimings();
        return completion.t_start_generation == 0 &&
               completion.t_token_generation == 0.0;
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "Starting rnllama API tests..." << std::endl;
    std::cout << "Using test model: ../tiny-random-llama.gguf" << std::endl;
    std::cout << "=========================" << std::endl;

    TestResults results;

    // Run all tests
    results.run_test("Context Creation and Model Loading", test_context_creation_and_model_loading());
    results.run_test("Tokenization", test_tokenization());
    results.run_test("Completion", test_completion());
    results.run_test("Completion Generation Timing", test_completion_generation_timing());
    results.run_test("Completion Probabilities Reset Between Completions", test_completion_probabilities_reset_between_completions());
    results.run_test("Logit Bias Forces Token", test_logit_bias_forces_token());
    results.run_test("n_probs Post-Sampling vs Raw", test_n_probs_post_sampling_vs_raw());
    results.run_test("Graceful Context Init Failure", test_context_init_failure_is_graceful());
    results.run_test("Utility Functions", test_utilities());

    // Print summary
    results.print_summary();

    // Return appropriate exit code
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
