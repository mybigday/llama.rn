#include <iostream>
#include <cassert>
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

// Test utility functions
bool test_utilities() {
    try {
        // Test kv_cache_type_from_str function
        lm_ggml_type cache_type = kv_cache_type_from_str("f16");
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

int main() {
    std::cout << "Starting rnllama API tests..." << std::endl;
    std::cout << "Using test model: ../tiny-random-llama.gguf" << std::endl;
    std::cout << "=========================" << std::endl;

    TestResults results;

    // Run all tests
    results.run_test("Context Creation and Model Loading", test_context_creation_and_model_loading());
    results.run_test("Tokenization", test_tokenization());
    results.run_test("Completion", test_completion());
    results.run_test("Utility Functions", test_utilities());

    // Print summary
    results.print_summary();

    // Return appropriate exit code
    return (results.passed_tests == results.total_tests) ? 0 : 1;
}
