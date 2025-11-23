#import "RNLlamaJSI.h"
#import "../cpp/jsi/RNLlamaJSI.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <TargetConditionals.h>

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

@implementation RNLlama (JSI)

- (void)installJSIBindingsWithRuntime:(facebook::jsi::Runtime&)runtime
                          callInvoker:(std::shared_ptr<facebook::react::CallInvoker>)callInvoker {
    rnllama_jsi::installJSIBindings(runtime, callInvoker);
}

@end
