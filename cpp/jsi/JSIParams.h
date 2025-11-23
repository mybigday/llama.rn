#pragma once
#include <jsi/jsi.h>
#include <string>
#include "JSINativeHeaders.h"

using namespace facebook;

namespace rnllama_jsi {
    std::string getPropertyAsString(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, const std::string& defaultValue = "");
    int getPropertyAsInt(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, int defaultValue = 0);
    double getPropertyAsDouble(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, double defaultValue = 0.0);
    bool getPropertyAsBool(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, bool defaultValue = false);
    float getPropertyAsFloat(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, float defaultValue = 0.0f);

    void parseCommonParams(jsi::Runtime& runtime, const jsi::Object& params, common_params& cparams);
    void parseCompletionParams(jsi::Runtime& runtime, const jsi::Object& params, rnllama::llama_rn_context* ctx);
}
