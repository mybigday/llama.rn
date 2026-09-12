#pragma once
#include <jsi/jsi.h>
#include <string>
#include "JSINativeHeaders.h"
#include "JSIJson.h"

using namespace facebook;

namespace rnllama_jsi {
    // jsi::Object lookups (JS thread only). Missing or wrong-typed values
    // fall back to the default.
    std::string getPropertyAsString(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, const std::string& defaultValue = "");
    int getPropertyAsInt(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, int defaultValue = 0);
    double getPropertyAsDouble(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, double defaultValue = 0.0);
    bool getPropertyAsBool(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, bool defaultValue = false);
    float getPropertyAsFloat(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, float defaultValue = 0.0f);

    // Same lookups over a json value produced by toJson(). Safe on any
    // thread; identical fallback semantics (a wrong type is treated as absent
    // rather than throwing like nlohmann's value()).
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
}
