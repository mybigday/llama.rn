// A/B benchmark for the prompt state cache: multi-turn chat measuring per-turn
// time-to-first-token, generation speed, and process memory. Uses only the
// rn-completion API that exists both before and after the KV-reuse feature, so
// the SAME source builds at the merge-base (baseline) and at HEAD.
//
//   BENCH,<model>,<phase>,<prompt_tokens>,<reused>,<ttft_ms>,<gen_tps>,<rss_mb>,<hwm_mb>
//
// Env: MODELS_DIR, RNLLAMA_NGL, BENCH_TURNS (default 8), BENCH_GEN (default 32),
//      BENCH_BUDGET_MB (branch builds only; <0 = default, 0 = cache off).

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"
#include "nlohmann/json.hpp"

using namespace rnllama;
using json = nlohmann::ordered_json;

namespace {

struct MemStat { double rss_mb = 0, hwm_mb = 0; };

MemStat read_mem() {
    MemStat m;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        double kb = 0;
        if (sscanf(line.c_str(), "VmRSS: %lf kB", &kb) == 1) m.rss_mb = kb / 1024.0;
        if (sscanf(line.c_str(), "VmHWM: %lf kB", &kb) == 1) m.hwm_mb = kb / 1024.0;
    }
    return m;
}

int env_i(const char *k, int d) {
    const char *v = std::getenv(k);
    return v ? std::atoi(v) : d;
}

struct Bench {
    llama_rn_context ctx;
    std::string model_key;
    std::string system_prompt;
    std::vector<std::pair<std::string, std::string>> history;

    bool load(const std::string &path, int n_ctx) {
        common_params params;
        params.model.path = path;
        params.n_ctx = n_ctx;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.cpuparams.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2);
        const char *ngl = std::getenv("RNLLAMA_NGL");
        params.n_gpu_layers = ngl ? std::atoi(ngl) : 0;
        params.no_kv_offload = params.n_gpu_layers == 0;
        params.n_predict = -1;
        params.ctx_shift = false;
        params.sampling.temp = 0.0f;
        params.sampling.top_k = 1;
#ifdef KV_BENCH_HAS_STATE_CACHE
        const int budget = env_i("BENCH_BUDGET_MB", -1);
        if (budget >= 0) ctx.state_cache_budget_bytes = (size_t) budget * 1024 * 1024;
#endif
        if (!ctx.loadModel(params)) return false;
        if (ctx.completion == nullptr) ctx.completion = new llama_rn_context_completion(&ctx);
        return true;
    }

    std::string render() const {
        json msgs = json::array();
        if (!system_prompt.empty()) msgs.push_back({{"role", "system"}, {"content", system_prompt}});
        for (const auto &m : history) msgs.push_back({{"role", m.first}, {"content", m.second}});
        common_chat_params cp = ctx.getFormattedChatWithJinja(
            msgs.dump(-1, ' ', false, json::error_handler_t::replace),
            "", "", "", false, "", /*enable_thinking*/ false, "none",
            /*add_generation_prompt*/ true, "", {}, false);
        return cp.prompt;
    }

    // One assistant turn; prints a BENCH CSV line and appends the reply.
    void turn(const std::string &phase, int max_new) {
        auto *cmpl = ctx.completion;
        ctx.params.prompt = render();
        ctx.params.n_predict = max_new;
        cmpl->rewind();
        cmpl->initSampling();

        const auto t0 = std::chrono::steady_clock::now();
        cmpl->loadPrompt({});
        const size_t reused = (size_t) std::max<llama_pos>(0, cmpl->n_past);
        const size_t prompt_tokens = cmpl->num_prompt_tokens;
        cmpl->beginCompletion();
        std::string reply;
        int n = 0;
        double ttft_ms = 0;
        std::chrono::steady_clock::time_point t_first;
        while (cmpl->has_next_token && n < max_new) {
            auto o = cmpl->nextToken();
            if (o.tok == -1 || cmpl->stopped_eos) break;
            if (n == 0) {
                t_first = std::chrono::steady_clock::now();
                ttft_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
            }
            reply += o.text;
            n++;
            if (cmpl->stopped_limit) break;
        }
        const auto t1 = std::chrono::steady_clock::now();
        cmpl->endCompletion();
        const double gen_s = std::chrono::duration<double>(t1 - (n > 0 ? t_first : t0)).count();
        const double tps = (n > 1 && gen_s > 0) ? (n - 1) / gen_s : 0.0;
        const MemStat mem = read_mem();
        history.push_back({"assistant", reply});
        printf("BENCH,%s,%s,%zu,%zu,%.1f,%.2f,%.1f,%.1f\n",
               model_key.c_str(), phase.c_str(), prompt_tokens, reused,
               ttft_ms, tps, mem.rss_mb, mem.hwm_mb);
        fflush(stdout);
    }

    // Capture-cost probe (BENCH_PROBE=1): sizes the per-capture cost of the two
    // GPU checkpoint strategies on the current device.
    //   A = host readback   : llama_state_seq_get_data_ext(PARTIAL_ONLY)  (today's cost)
    //   C = one decode       : baseline, no checkpoint
    //   B = seq_cp + decode   : device-resident seq_cp(0->ckpt) then decode (COW rides it)
    // B-C isolates the on-device copy-on-write overhead of the seq_cp path.
    void probe(int warm_turns) {
        // Warm the state to a realistic mid-conversation point.
        for (int t = 0; t < warm_turns; t++) {
            history.push_back({"user",
                "Tell me two sentences about interesting topic number " +
                std::to_string(t + 1) + " and why people care about it."});
            turn("probe-warm-t" + std::to_string(t + 1), 24);
        }
        llama_context *c = ctx.ctx;
        llama_memory_t mem = llama_get_memory(c);
        const int n_layer = llama_model_n_layer(ctx.model);
        const int ckpt_seq = 1;              // n_seq_max=8, seq 1 is free
        llama_memory_seq_rm(mem, ckpt_seq, -1, -1);
        const auto vocab = llama_model_get_vocab(ctx.model);
        llama_token tok = llama_vocab_bos(vocab);
        if (tok < 0) tok = 0;

        const size_t size = llama_state_seq_get_size_ext(
            c, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> buf(size);

        auto ms = [](auto d){ return std::chrono::duration<double, std::milli>(d).count(); };
        auto now = []{ return std::chrono::steady_clock::now(); };
        const int K = 30;

        // A: host readback (read-only; does not mutate state)
        double A = 0;
        for (int i = 0; i < K; i++) {
            auto t0 = now();
            llama_state_seq_get_data_ext(c, buf.data(), size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            A += ms(now() - t0);
        }
        A /= K;

        // C: one decode on seq 0, no checkpoint (baseline)
        double C = 0;
        for (int i = 0; i < K; i++) {
            auto t0 = now();
            llama_decode(c, llama_batch_get_one(&tok, 1));
            C += ms(now() - t0);
        }
        C /= K;

        // B: seq_cp(0 -> ckpt) then decode (the COW copy rides this decode)
        double B = 0;
        for (int i = 0; i < K; i++) {
            auto t0 = now();
            llama_memory_seq_cp(mem, 0, ckpt_seq, -1, -1);
            llama_decode(c, llama_batch_get_one(&tok, 1));
            B += ms(now() - t0);
        }
        B /= K;

        printf("PROBE,%s,n_layer,%d,state_kib,%.1f,A_host_capture_ms,%.3f,"
               "C_decode_ms,%.3f,B_seqcp_decode_ms,%.3f,seqcp_overhead_ms,%.3f\n",
               model_key.c_str(), n_layer, size / 1024.0, A, C, B, B - C);
        fflush(stdout);
    }
};

} // namespace

int main(int argc, char **argv) {
    const char *env_dir = std::getenv("MODELS_DIR");
    std::filesystem::path models_dir =
        env_dir ? std::filesystem::path(env_dir)
                : std::filesystem::path(__FILE__).parent_path() / "models";
    const int turns   = env_i("BENCH_TURNS", 8);
    const int max_new = env_i("BENCH_GEN", 32);

    // key -> file (subset that exists is run)
    const std::vector<std::pair<std::string, std::string>> models = {
        {"lfm2", "lfm2.gguf"}, {"granite4", "granite4.gguf"},
        {"qwen35", "qwen35.gguf"}, {"gemma4", "gemma4.gguf"},
        {"smollm2", "smollm2.gguf"},
    };
    std::vector<std::string> want;
    for (int i = 1; i < argc; i++) want.push_back(argv[i]);

    printf("BENCH_HEADER,model,phase,prompt_tokens,reused,ttft_ms,gen_tps,rss_mb,hwm_mb\n");
    for (const auto &m : models) {
        if (!want.empty() && std::find(want.begin(), want.end(), m.first) == want.end()) continue;
        const auto p = models_dir / m.second;
        if (!std::filesystem::exists(p)) continue;

        Bench b;
        b.model_key = m.first;
        if (!b.load(p.string(), 4096)) {
            printf("BENCH,%s,load-failed,0,0,0,0,0,0\n", m.first.c_str());
            continue;
        }
        std::string sys = "You are a helpful, concise assistant. ";
        for (int i = 0; i < 24; i++) sys += "Always answer clearly and stay on topic. ";
        b.system_prompt = sys;

        if (env_i("BENCH_PROBE", 0)) {
            b.probe(/*warm_turns*/ env_i("BENCH_PROBE_WARM", 4));
            continue;
        }

        // Phase 1: append-only conversation (the common case).
        for (int t = 0; t < turns; t++) {
            b.history.push_back({"user",
                "Tell me two sentences about interesting topic number " + std::to_string(t + 1) +
                " and why people care about it."});
            b.turn("append-t" + std::to_string(t + 1), max_new);
        }
        // Phase 2: regenerate the last reply (divergence one turn back).
        b.history.pop_back();
        b.turn("regenerate", max_new);
        // Phase 3: fresh session sharing only the system prompt.
        b.history.clear();
        b.history.push_back({"user", "Hi! What can you help me with?"});
        b.turn("new-session", max_new);
    }
    return 0;
}
