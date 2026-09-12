#pragma once
#include <jsi/jsi.h>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "JSINativeHeaders.h"
#include "JSIJson.h"

using namespace facebook;

namespace rnllama_jsi {
    // Lookups over a json value produced by toJson(). Safe on any thread.
    // Missing or wrong-typed values fall back to the default (a wrong type is
    // treated as absent rather than throwing like nlohmann's value()).
    std::string getPropertyAsString(const json& obj, const char* name, const std::string& defaultValue = "");
    int getPropertyAsInt(const json& obj, const char* name, int defaultValue = 0);
    double getPropertyAsDouble(const json& obj, const char* name, double defaultValue = 0.0);
    bool getPropertyAsBool(const json& obj, const char* name, bool defaultValue = false);
    float getPropertyAsFloat(const json& obj, const char* name, float defaultValue = 0.0f);

    bool hasSpeculativeType(const common_params_speculative& speculative, common_speculative_type type);

    // Both parsers take the JS params object already converted with toJson().
    // Note: toJson() drops `undefined` properties, so `{ key: undefined }`
    // now reads as "absent" (same as JSON.stringify) instead of "present but
    // wrong type".
    void parseCommonParams(const json& params, common_params& cparams);
    void parseCompletionParams(const json& params, rnllama::llama_rn_context* ctx);

    // `[{ path, scaled? }]` as sent by applyLoraAdapters / lora_list.
    // Entries that are not objects or have an empty path are skipped.
    std::vector<common_adapter_lora_info> parseLoraAdapters(const json& list);

    // Fixed-shape option objects. Member names double as the JS keys through
    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT: a missing key keeps the
    // member default, a wrong-typed value throws and optionsFromJson() turns
    // that into a JSError. Unlike the big params bags above these are strict
    // on purpose; the TS declarations already pin their shape.

    struct MultimodalInitOptions {
        std::string path;
        bool use_gpu = true;
        int image_min_tokens = -1;
        int image_max_tokens = -1;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(MultimodalInitOptions, path, use_gpu, image_min_tokens, image_max_tokens)

    struct VocoderInitOptions {
        std::string path;
        int n_batch = 512;
        // When the key is absent the binding follows the main context's GPU
        // offload instead of this value.
        bool use_gpu = false;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(VocoderInitOptions, path, n_batch, use_gpu)

    struct ParallelModeOptions {
        bool enabled = true;
        int n_parallel = 2;
        int n_batch = 512;
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ParallelModeOptions, enabled, n_parallel, n_batch)

    // createSpeaker(ctxId, pcm, opts); keys are camelCase like the public API.
    struct SpeakerOptions {
        int inputSampleRate = 0;
        std::string refText;
        bool bake = false;
        // NaN = not provided, so the model's own default applies.
        float emotion = std::numeric_limits<float>::quiet_NaN();

        bool hasEmotion() const { return !std::isnan(emotion); }
    };
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(SpeakerOptions, inputSampleRate, refText, bake, emotion)

    // Convert a JS options object (already run through toJson) into one of the
    // structs above. Must run on the JS thread because it throws JSError.
    template <typename T>
    T optionsFromJson(jsi::Runtime& rt, const json& j, const char* fnName) {
        if (!j.is_object()) {
            throw jsi::JSError(rt, std::string(fnName) + ": options must be an object");
        }
        try {
            return j.get<T>();
        } catch (const json::exception& e) {
            throw jsi::JSError(rt, std::string(fnName) + ": invalid options: " + e.what());
        }
    }
}

namespace rnllama {
    // Key-mapping shim for generateAudioCodes: the public API uses camelCase
    // while the core struct is snake_case. Found by ADL from json::get<>().
    template <typename BasicJsonType>
    void from_json(const BasicJsonType& j, llama_rn_audio_codes_options& o) {
        const llama_rn_audio_codes_options d{};
        o.prompt      = j.value("prompt",      d.prompt);
        o.max_frames  = j.value("maxFrames",   d.max_frames);
        o.temperature = j.value("temperature", d.temperature);
        o.top_p       = j.value("topP",        d.top_p);
        o.top_k       = j.value("topK",        d.top_k);
        o.seed        = j.value("seed",        d.seed);
    }
}
