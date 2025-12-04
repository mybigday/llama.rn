#pragma once
#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <functional>
#include <memory>
#include <string>

using namespace facebook;

namespace rnllama_jsi {

    using PromiseResultGenerator = std::function<jsi::Value(jsi::Runtime&)>;
    using PromiseTask = std::function<PromiseResultGenerator()>;

    jsi::Value createPromiseTask(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker,
        PromiseTask task,
        int contextId = -1,
        bool trackTask = true
    );

    // Safe console.log wrapper for JSI context
    inline void consoleLog(jsi::Runtime& runtime, const std::string& message) {
        auto console = runtime.global().getPropertyAsObject(runtime, "console");
        auto log = console.getPropertyAsFunction(runtime, "log");
        log.call(runtime, jsi::String::createFromUtf8(runtime, message));
    }

    inline void consoleWarn(jsi::Runtime& runtime, const std::string& message) {
        auto console = runtime.global().getPropertyAsObject(runtime, "console");
        auto warn = console.getPropertyAsFunction(runtime, "warn");
        warn.call(runtime, jsi::String::createFromUtf8(runtime, message));
    }

    inline void consoleError(jsi::Runtime& runtime, const std::string& message) {
        auto console = runtime.global().getPropertyAsObject(runtime, "console");
        auto error = console.getPropertyAsFunction(runtime, "error");
        error.call(runtime, jsi::String::createFromUtf8(runtime, message));
    }
}
