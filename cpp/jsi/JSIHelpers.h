#pragma once

#include "RNLlamaJSI.h"
#include "JSIContext.h"
#include "ThreadPool.h"
#include "JSIUtils.h"
#include "JSIParams.h"
#include "JSINativeHeaders.h"
#include <algorithm>
#include <string>
#include <vector>

namespace rnllama_jsi {

    inline json modelInfoJson(const std::string& path, const std::vector<std::string>& skip) {
        rnllama::gguf_file_info gguf;
        if (!rnllama::read_gguf_file_info(path, gguf)) {
            throw std::runtime_error("Failed to load model info");
        }

        json info = json::object({
            {"version", (int) gguf.version},
            {"alignment", (int) gguf.alignment},
            {"data_offset", (int) gguf.data_offset},
        });

        for (const auto& [key, value] : gguf.kv) {
            if (std::find(skip.begin(), skip.end(), key) != skip.end()) {
                continue;
            }
            info[key] = value;
        }

        return info;
    }

}
