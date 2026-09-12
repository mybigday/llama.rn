#pragma once

#include "RNLlamaJSI.h"
#include "JSIContext.h"
#include "ThreadPool.h"
#include "JSIUtils.h"
#include "JSIParams.h"
#include "JSINativeHeaders.h"
#include <string>
#include <vector>

namespace rnllama_jsi {

    inline json modelInfoJson(const std::string& path, const std::vector<std::string>& skip) {
        struct lm_gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ NULL,
        };

        struct lm_gguf_context * ctx = lm_gguf_init_from_file(path.c_str(), params);

        if (!ctx) {
            throw std::runtime_error("Failed to load model info");
        }

        json info = json::object({
            {"version", (int) lm_gguf_get_version(ctx)},
            {"alignment", (int) lm_gguf_get_alignment(ctx)},
            {"data_offset", (int) lm_gguf_get_data_offset(ctx)},
        });

        const int n_kv = lm_gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = lm_gguf_get_key(ctx, i);
            std::string keyStr(key);

            bool shouldSkip = false;
            for (const auto& skipKey : skip) {
                if (keyStr == skipKey) {
                    shouldSkip = true;
                    break;
                }
            }
            if (shouldSkip) continue;

            info[keyStr] = lm_gguf_kv_to_str(ctx, i);
        }

        lm_gguf_free(ctx);
        return info;
    }

}
