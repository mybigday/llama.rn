// Decision model probe: answers a TypeSafe /v1/systemone request with
// llama_rn_context::decide() (or the slot manager, --parallel N) and prints the
// response JSON, so it can be diffed against llama-server's /v1/systemone.
//
//   decision_probe <model.gguf> <request.json> [--threads N] [--ctx N] [--parallel N]

#include "rn-llama.h"
#include "rn-slot-manager.h"

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>

using namespace rnllama;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <request.json> [--threads N] [--ctx N] [--parallel N]\n", argv[0]);
        return 1;
    }
    const std::string model_path   = argv[1];
    const std::string request_path = argv[2];
    int threads    = 8;
    int n_ctx      = 4096;
    int n_parallel = 0;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--threads") {
            threads = std::atoi(argv[i + 1]);
        } else if (arg == "--ctx") {
            n_ctx = std::atoi(argv[i + 1]);
        } else if (arg == "--parallel") {
            n_parallel = std::atoi(argv[i + 1]);
        }
    }

    std::ifstream f(request_path);
    std::stringstream ss;
    ss << f.rdbuf();
    const json request = json::parse(ss.str());

    llama_rn_context ctx;
    common_params params;
    params.model.path = model_path;
    params.n_ctx = n_ctx;
    params.n_parallel = std::max(1, n_parallel);
    params.cpuparams.n_threads = threads;
    params.cpuparams_batch.n_threads = threads;
    params.n_gpu_layers = 0;
    if (!ctx.loadModel(params)) {
        std::fprintf(stderr, "loadModel failed\n");
        return 2;
    }
    std::fprintf(stderr, "[probe] model.decision = %s\n", from_common_json(ctx.decision.info()).dump().c_str());

    json result;
    try {
        if (n_parallel > 0) {
            ctx.enableParallelMode(n_parallel, params.n_batch);
            ctx.slot_manager->start_processing_loop();
            std::mutex mtx;
            std::condition_variable cv;
            bool done = false;
            ctx.slot_manager->queue_decision_request(request, [&](int32_t, const json & r) {
                std::lock_guard<std::mutex> lock(mtx);
                result = r;
                done = true;
                cv.notify_one();
            });
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return done; });
        } else {
            result = ctx.decide(request);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "decide failed: %s\n", e.what());
        return 3;
    }

    std::printf("%s\n", result.dump(2).c_str());
    return 0;
}
