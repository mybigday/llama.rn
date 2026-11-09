// read_gguf_file_info(): skipped keys report their array length.
//
// loadLlamaModelInfo() skips the four big tokenizer arrays, because stringifying
// them and handing a few megabytes of text to JS is pointless. The skip used to
// be applied after the read, so gguf_kv_to_str() had already flattened every
// array into a std::string that was then dropped -- and the caller was left
// unable to tell a 262144-token vocabulary from a 32000-token one.
//
// The skip now reaches the reader, which records the element count instead. The
// count is what a caller can act on: a compute buffer scales with n_vocab, and
// Gemma 3 and Qwen3.5 don't write <arch>.vocab_size at all.
//
// The fixture is convenient here: tiny-random-llama.gguf declares both
// llama.vocab_size (128256) and tokenizer.ggml.tokens (an array of 128256), so
// the reported length is checked against the file's own number rather than
// against a constant pasted into this test.
//
// Usage:
//   ./model_info_test [path/to/model.gguf]     # default: ../tiny-random-llama.gguf

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <optional>
#include <filesystem>

#include "rn-llama.h"

struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;

    void run_test(const std::string& name, bool result) {
        total_tests++;
        std::cout << "TEST: " << name << " ... ";
        std::cout << (result ? "PASSED" : "FAILED") << std::endl;
        if (result) passed_tests++;
    }

    void print_summary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
    }
};

// The keys src/index.ts passes as modelInfoSkip.
static const std::vector<std::string> kModelInfoSkip = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
};

static std::optional<std::string> kv_value(const rnllama::gguf_file_info& info,
                                           const std::string& key) {
    for (const auto& [k, v] : info.kv) {
        if (k == key) return v;
    }
    return std::nullopt;
}

static std::optional<size_t> arr_length(const rnllama::gguf_file_info& info,
                                        const std::string& key) {
    for (const auto& [k, n] : info.arr_lengths) {
        if (k == key) return n;
    }
    return std::nullopt;
}

// Total bytes held in the stringified values, i.e. what the read actually built.
static size_t kv_bytes(const rnllama::gguf_file_info& info) {
    size_t total = 0;
    for (const auto& [k, v] : info.kv) total += k.size() + v.size();
    return total;
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "../tiny-random-llama.gguf";

    std::cout << "=== model info array-length tests ===" << std::endl;
    std::cout << "model: " << path << std::endl << std::endl;

    if (!std::filesystem::exists(path)) {
        std::cout << "Model not found: " << path << std::endl;
        return 1;
    }

    rnllama::gguf_file_info plain;
    rnllama::gguf_file_info skipped;
    if (!rnllama::read_gguf_file_info(path, plain)) {
        std::cout << "Failed to read (no skip)" << std::endl;
        return 1;
    }
    if (!rnllama::read_gguf_file_info(path, skipped, kModelInfoSkip)) {
        std::cout << "Failed to read (with skip)" << std::endl;
        return 1;
    }

    TestResults results;

    // Without a skip list nothing changes: the arrays are stringified as before
    // and no lengths are collected. Existing callers see the old behaviour.
    results.run_test("no skip: tokenizer array is still stringified",
                     kv_value(plain, "tokenizer.ggml.tokens").has_value());
    results.run_test("no skip: no lengths collected",
                     plain.arr_lengths.empty());

    // With the skip list the arrays are gone from kv and present as lengths.
    results.run_test("skip: tokenizer array dropped from kv",
                     !kv_value(skipped, "tokenizer.ggml.tokens").has_value());

    const auto tokens_n = arr_length(skipped, "tokenizer.ggml.tokens");
    results.run_test("skip: tokenizer array reports a length",
                     tokens_n.has_value());

    // The point of the change: the length matches the vocabulary size the file
    // declares separately, so a caller can size a compute buffer from it.
    const auto declared = kv_value(skipped, "llama.vocab_size");
    results.run_test("length equals the file's own llama.vocab_size",
                     tokens_n.has_value() && declared.has_value() &&
                         std::to_string(*tokens_n) == *declared);

    results.run_test("skip: every skipped array present in the file has a length",
                     arr_length(skipped, "tokenizer.ggml.token_type").has_value() &&
                         arr_length(skipped, "tokenizer.ggml.merges").has_value());

    // tokenizer.ggml.scores is in the skip list but not in this file, so it must
    // not be invented as a zero-length entry.
    results.run_test("skip: a key absent from the file reports no length",
                     !arr_length(skipped, "tokenizer.ggml.scores").has_value());

    // Keys outside the skip list are untouched.
    results.run_test("skip: unskipped keys unaffected",
                     kv_value(skipped, "general.architecture") ==
                         kv_value(plain, "general.architecture"));
    results.run_test("skip: kv count drops by exactly the skipped keys",
                     plain.kv.size() - skipped.kv.size() ==
                         skipped.arr_lengths.size());

    // And the saving, which is the reason the skip exists at all.
    const size_t before = kv_bytes(plain);
    const size_t after = kv_bytes(skipped);
    results.run_test("skip: stringified bytes drop by over 1 MiB",
                     before > after && (before - after) > (1u << 20));

    std::cout << std::endl;
    std::cout << "stringified kv bytes: " << before << " -> " << after
              << "  (saved " << (before - after) / 1048576.0 << " MiB)" << std::endl;
    for (const auto& [k, n] : skipped.arr_lengths) {
        std::cout << "  " << k << " -> " << n << " elements" << std::endl;
    }

    results.print_summary();
    return results.passed_tests == results.total_tests ? 0 : 1;
}
