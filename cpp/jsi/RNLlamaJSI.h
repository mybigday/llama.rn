#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <cstdint>
#include <string>

using namespace facebook;

namespace rnllama_jsi {
    // Context management functions
    void addContext(int contextId, long contextPtr);
    void removeContext(int contextId);
    void setContextLimit(int64_t limit);
#if defined(__ANDROID__)
    void setAndroidLoadedLibrary(const std::string& name);
#elif defined(__APPLE__)
    struct MetalAvailability {
        bool available;
        std::string reason;
    };
    std::string resolveIosModelPath(const std::string& path, bool isAsset);
    MetalAvailability getMetalAvailability(bool skipGpuDevices);
#endif

    // Main JSI installation function
    void installJSIBindings(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker
    );

    // Cleanup function
    void cleanupJSIBindings();
}
