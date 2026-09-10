// On-device numerical regression for non-FA grouped-query attention. No model needed.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-opencl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

static float sample(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
}

static void check(int key_rows, int key_heads, int gqa) {
    const int width = 128;
    const int query_heads = key_heads * gqa;
    uint32_t state = 42;
    std::vector<lm_ggml_fp16_t> keys(width * key_rows * key_heads);
    std::vector<float> queries(width * query_heads);
    for (auto & value : keys) value = lm_ggml_fp32_to_fp16(sample(state));
    for (auto & value : queries) value = sample(state);

    std::vector<float> results[2];
    for (int device = 0; device < 2; ++device) {
        auto backend = device ? lm_ggml_backend_opencl_init() : lm_ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("required backend unavailable");
        if (!device) lm_ggml_backend_cpu_set_n_threads(backend, 4);
        auto ctx = lm_ggml_init({8 * lm_ggml_tensor_overhead() + lm_ggml_graph_overhead(), nullptr, true});
        if (!ctx) throw std::runtime_error("context allocation failed");
        auto key = lm_ggml_new_tensor_4d(ctx, LM_GGML_TYPE_F16, width, key_rows, key_heads, 1);
        auto query = lm_ggml_new_tensor_4d(ctx, LM_GGML_TYPE_F32, width, 1, query_heads, 1);
        auto output = lm_ggml_mul_mat(ctx, key, query);
        if (!lm_ggml_backend_supports_op(backend, output)) {
            throw std::runtime_error("operation unsupported; refusing CPU fallback");
        }
        auto graph = lm_ggml_new_graph(ctx);
        lm_ggml_build_forward_expand(graph, output);
        auto buffer = lm_ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("tensor allocation failed");
        lm_ggml_backend_tensor_set(key, keys.data(), 0, keys.size() * sizeof(keys[0]));
        lm_ggml_backend_tensor_set(query, queries.data(), 0, queries.size() * sizeof(queries[0]));
        if (lm_ggml_backend_graph_compute(backend, graph) != LM_GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
        results[device].resize(lm_ggml_nelements(output));
        lm_ggml_backend_tensor_get(output, results[device].data(), 0, results[device].size() * sizeof(float));
        lm_ggml_backend_buffer_free(buffer);
        lm_ggml_free(ctx);
        lm_ggml_backend_free(backend);
    }

    double error = 0;
    double energy = 0;
    for (size_t i = 0; i < results[0].size(); ++i) {
        const double reference = results[0][i];
        const double actual = results[1][i];
        if (!std::isfinite(reference) || !std::isfinite(actual)) {
            throw std::runtime_error("nonfinite attention output");
        }
        error += (actual - reference) * (actual - reference);
        energy += reference * reference;
    }
    const double nmse = error / energy;
    printf("KQ rows=%d key_heads=%d gqa=%d NMSE=%.12g\n", key_rows, key_heads, gqa, nmse);
    // Allows normal f16 arithmetic differences but rejects the ~0.25 regression.
    if (!std::isfinite(nmse) || nmse > 1e-5) throw std::runtime_error("attention numerical mismatch");
}

int main() try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    check(256, 8, 4); // Qwen3-VL 4B decode shape that originally exposed the issue.
    check(64, 2, 4);  // Minimum row count admitted by the R4 specialization.
    check(80, 2, 4);  // Multiple of 16, not a power of two.
    check(512, 4, 4);
    check(48, 2, 4);  // Below the R4 admission threshold.
    check(256, 2, 2); // Different GQA ratio: unaffected path.
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
