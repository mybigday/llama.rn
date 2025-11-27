#import "RNLlama.h"
#import <jsi/jsi.h>

#if RNLLAMA_BUILD_FROM_SOURCE
#import "../cpp/llama-impl.h"
#import "../cpp/llama.h"
#else
#import <rnllama/llama-impl.h>
#import <rnllama/llama.h>
#endif

#import "../cpp/jsi/RNLlamaJSI.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNLlamaSpec.h"
#endif

namespace rnllama_jsi {
std::string resolveIosModelPath(const std::string& path, bool isAsset) {
    if (!isAsset) {
        return path;
    }

    NSString *nsPath = [NSString stringWithUTF8String:path.c_str()];
    if (!nsPath) {
        return path;
    }

    NSString *resolved = [[NSBundle mainBundle] pathForResource:nsPath ofType:nil];
    if (!resolved) {
        return path;
    }

    return std::string([resolved UTF8String]);
}

MetalAvailability getMetalAvailability(bool skipGpuDevices) {
    MetalAvailability availability{false, ""};

    if (skipGpuDevices) {
        availability.reason = "GPU devices disabled by user";
        return availability;
    }

#if defined(LM_GGML_USE_METAL)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    bool supportsMetal = false;

    if (device) {
        supportsMetal = [device supportsFamily:MTLGPUFamilyApple7];
        if (@available(iOS 16.0, tvOS 16.0, *)) {
            supportsMetal = supportsMetal && [device supportsFamily:MTLGPUFamilyMetal3];
        }
    }

#if TARGET_OS_SIMULATOR
    supportsMetal = false;
#endif

    availability.available = supportsMetal;
    if (!supportsMetal) {
        availability.reason = "Metal is not supported in this device";
    }
#else
    availability.available = false;
    availability.reason = "Metal is not enabled in this build";
#endif

    return availability;
}
} // namespace rnllama_jsi

@implementation RNLlama

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(install:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RCTBridge *bridge = [RCTBridge currentBridge];
    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
    auto callInvoker = bridge.jsCallInvoker;
    if (!cxxBridge.runtime) {
        resolve(@false);
        return;
    }

    facebook::jsi::Runtime *runtime = static_cast<facebook::jsi::Runtime *>(cxxBridge.runtime);

    if (callInvoker) {
        callInvoker->invokeAsync([runtime, callInvoker]() {
            rnllama_jsi::installJSIBindings(*runtime, callInvoker);
        });
    } else {
        resolve(@false);
        return;
    }

    resolve(@true);
}

- (void)invalidate {
    rnllama_jsi::cleanupJSIBindings();
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNLlamaSpecJSI>(params);
}
#endif

@end
