#pragma once

#ifndef RN_COMMON_HPP
#define RN_COMMON_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cinttypes>
#include "llama.h"

// Include backend device support
#include "ggml-backend.h"
#ifdef LM_GGML_USE_METAL
#include "ggml-metal/ggml-metal-device.h"
#endif

#if defined(__ANDROID__)
#include <sys/sysinfo.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace rnllama {

inline static std::string backend_devices_info() {
    json devices_array = json::array();

    const size_t dev_count = lm_ggml_backend_dev_count();

    for (size_t i = 0; i < dev_count; i++) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (dev == nullptr) continue;

        // Get basic device properties
        lm_ggml_backend_dev_props props;
        lm_ggml_backend_dev_get_props(dev, &props);

        json device_info;

        // Get backend name from the device's backend registry
        lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
        const char* backend_name = reg ? lm_ggml_backend_reg_name(reg) : "unknown";
        device_info["backend"] = backend_name ? backend_name : "unknown";

        // Convert device type to string
        std::string type_str;
        switch (props.type) {
            case LM_GGML_BACKEND_DEVICE_TYPE_CPU:
                type_str = "cpu";
                break;
            case LM_GGML_BACKEND_DEVICE_TYPE_GPU:
                type_str = "gpu";
                break;
            case LM_GGML_BACKEND_DEVICE_TYPE_IGPU:
                type_str = "igpu";
                break;
            case LM_GGML_BACKEND_DEVICE_TYPE_ACCEL:
                type_str = "accel";
                break;
            default:
                type_str = "unknown";
        }
        device_info["type"] = type_str;

        // Device name and description
        device_info["deviceName"] = props.name ? props.name : "Unknown Device";

        // Memory information
        size_t memory_total = props.memory_total;

        // For Metal devices, use recommendedMaxWorkingSetSize instead
#ifdef LM_GGML_USE_METAL
        if (std::string(backend_name) == "Metal" || std::string(backend_name) == "metal") {
            // Try to get Metal-specific device properties
            lm_ggml_metal_device_t metal_dev = lm_ggml_metal_device_get();
            if (metal_dev) {
                const lm_ggml_metal_device_props* metal_props = lm_ggml_metal_device_get_props(metal_dev);
                if (metal_props) {
                    // Add Metal-specific metadata
                    json metadata;
                    metadata["supportsMTLGPUFamilyApple7"] = metal_props->supports_gpu_family_apple7;
                    metadata["hasBFloat16"] = metal_props->has_bfloat;
                    metadata["hasSimdgroupReduction"] = metal_props->has_simdgroup_reduction;
                    metadata["hasSimdgroupMM"] = metal_props->has_simdgroup_mm;
                    metadata["hasUnifiedMemory"] = metal_props->has_unified_memory;
                    metadata["maxBufferSize"] = metal_props->max_buffer_size;
                    metadata["maxThreadgroupMemorySize"] = metal_props->max_theadgroup_memory_size;
                    device_info["metadata"] = metadata;
                }
            }
        }
#endif

        // For some devices, the memory_total is just a placeholder (1 byte)
        // Fall back to system RAM since OpenCL devices typically share system memory
        bool is_fallback_memory = false;
        if (memory_total <= 1) {
#if defined(__ANDROID__)
            struct sysinfo si;
            if (sysinfo(&si) == 0) {
                memory_total = si.totalram;
                is_fallback_memory = true;
            }
#elif defined(__APPLE__)
            int mib[2] = {CTL_HW, HW_MEMSIZE};
            int64_t physical_memory = 0;
            size_t length = sizeof(physical_memory);
            if (sysctl(mib, 2, &physical_memory, &length, NULL, 0) == 0) {
                memory_total = (size_t)physical_memory;
                is_fallback_memory = true;
            }
#endif
        }

        device_info["maxMemorySize"] = memory_total;

        // Add metadata for fallback memory reporting
        if (is_fallback_memory) {
            if (!device_info.contains("metadata")) {
                device_info["metadata"] = json::object();
            }
            device_info["metadata"]["memoryFallback"] = true;
            device_info["metadata"]["memorySource"] = "systemRAM";
        }

        // Add to devices array
        devices_array.push_back(device_info);
    }

    return devices_array.dump();
}

// Helper function to check if string ends with suffix
static bool ends_with(const std::string& str, const std::string& suffix) {
  if (suffix.size() > str.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

// Helper function to find partial stop string at end of text
static size_t find_partial_stop_string(const std::string& stop, const std::string& text) {
  if (!text.empty() && !stop.empty()) {
      const char text_last_char = text.back();
      for (int64_t char_index = stop.size() - 1; char_index >= 0; char_index--) {
          if (stop[char_index] == text_last_char) {
              const std::string current_partial = stop.substr(0, char_index + 1);
              if (ends_with(text, current_partial)) {
                  return text.size() - char_index - 1;
              }
          }
      }
  }
  return std::string::npos;
}

// Helper function to find the length of common prefix between two token vectors
static size_t find_common_prefix_length(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
  size_t i;
  for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {
  }
  return i;
}

// Helper function to format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static std::vector<llama_token> format_rerank_tokens(
  const llama_vocab* vocab,
  const std::vector<llama_token>& query_tokens,
  const std::vector<llama_token>& doc_tokens
) {
  std::vector<llama_token> result;

  llama_token eos_token = llama_vocab_eos(vocab);
  if (eos_token == LLAMA_TOKEN_NULL) {
      eos_token = llama_vocab_sep(vocab);
  }

  result.reserve(doc_tokens.size() + query_tokens.size() + 4);

  if (llama_vocab_get_add_bos(vocab)) {
      result.push_back(llama_vocab_bos(vocab));
  }

  result.insert(result.end(), query_tokens.begin(), query_tokens.end());

  if (llama_vocab_get_add_eos(vocab)) {
      result.push_back(eos_token);
  }

  if (llama_vocab_get_add_sep(vocab)) {
      result.push_back(llama_vocab_sep(vocab));
  }

  result.insert(result.end(), doc_tokens.begin(), doc_tokens.end());

  if (llama_vocab_get_add_eos(vocab)) {
      result.push_back(eos_token);
  }

  return result;
}

}

#endif
