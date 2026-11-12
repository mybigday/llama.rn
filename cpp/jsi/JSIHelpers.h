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
        // Filtering here would have meant reading the arrays first; `skip` goes
        // to the reader so the text is never built.
        if (!rnllama::read_gguf_file_info(path, gguf, skip)) {
            throw std::runtime_error("Failed to load model info");
        }

        json info = json::object({
            {"version", (int) gguf.version},
            {"alignment", (int) gguf.alignment},
            {"data_offset", (int) gguf.data_offset},
        });

        for (const auto& [key, value] : gguf.kv) {
            info[key] = value;
        }

        // A skipped key used to vanish without trace, so a caller could not tell
        // a 262144-token vocabulary from a 32000-token one. Report the count.
        for (const auto& [key, length] : gguf.arr_lengths) {
            info[key + ".length"] = length;
        }

        return info;
    }

}
