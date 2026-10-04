// Decision model probe: answers a TypeSafe /v1/systemone request with
// llama_rn_context::decide() (or the slot manager, --parallel N) and prints the
// response JSON, so it can be diffed against llama-server's /v1/systemone.
//
//   decision_probe <model.gguf> <request.json> [--threads N] [--ctx N] [--parallel N] [--mmproj F]

#include "rn-llama.h"
#include "rn-slot-manager.h"
#include "rn-completion.h"

#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>

using namespace rnllama;

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <request.json> [--threads N] [--ctx N] [--parallel N] [--mmproj F]\n", argv[0]);
        return 1;
    }
    const std::string model_path   = argv[1];
    const std::string request_path = argv[2];
    int threads    = 8;
    int n_ctx      = 4096;
    int n_parallel = 0;
    std::string mmproj_path;
    std::string complete_media; // --complete-media F: run <request.json> as a raw prompt through completion() instead
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if (arg == "--threads") {
            threads = std::atoi(argv[i + 1]);
        } else if (arg == "--ctx") {
            n_ctx = std::atoi(argv[i + 1]);
        } else if (arg == "--parallel") {
            n_parallel = std::atoi(argv[i + 1]);
        } else if (arg == "--mmproj") {
            mmproj_path = argv[i + 1];
        } else if (arg == "--complete-media") {
            complete_media = argv[i + 1];
        }
    }

    std::ifstream f(request_path);
    std::stringstream ss;
    ss << f.rdbuf();
    const json request = complete_media.empty() ? json::parse(ss.str()) : json();

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
    if (!mmproj_path.empty() && !ctx.initMultimodal(mmproj_path, /*use_gpu*/ false)) {
        std::fprintf(stderr, "initMultimodal failed\n");
        return 2;
    }
    std::fprintf(stderr, "[probe] model.decision = %s\n", from_common_json(ctx.decision.info()).dump().c_str());

    if (!complete_media.empty()) {
        // the existing multimodal completion path, no rn-decision involved: top probabilities of the next token
        ctx.completion->rewind();
        ctx.params.prompt = ss.str();
        ctx.params.n_predict = 1;
        ctx.params.sampling.n_probs = 20;
        ctx.post_sampling_probs = false;
        ctx.completion->initSampling();
        ctx.completion->loadPrompt({complete_media});
        ctx.completion->beginCompletion();
        const auto out = ctx.completion->doCompletion();
        json probs = json::array();
        for (const auto & p : out.probs) {
            probs.push_back({{"id", p.tok}, {"prob", p.prob}});
        }
        std::printf("%s\n", json{{"probs", probs}}.dump(2).c_str());
        return 0;
    }

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
