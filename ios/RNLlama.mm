#import "RNLlama.h"
#import "RNLlamaJSI.h"

#if RNLLAMA_BUILD_FROM_SOURCE
#import "../cpp/llama-impl.h"
#import "../cpp/llama.h"
#else
#import <rnllama/llama-impl.h>
#import <rnllama/llama.h>
#endif

#import "../cpp/jsi/RNLlamaJSI.h"

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNLlamaSpec.h"
#endif

@implementation RNLlama

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(install:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RCTBridge *bridge = [RCTBridge currentBridge];
    RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
    if (!cxxBridge.runtime) {
         reject(@"RNLLAMA_ERROR", @"RNLLAMA_ERROR", nil);
         return;
    }

    [self installJSIBindingsWithRuntime:*(facebook::jsi::Runtime *)cxxBridge.runtime
                           callInvoker:cxxBridge.jsCallInvoker];
    resolve(@true);
}

- (void)invalidate {
    rnllama_jsi::cleanupJSIBindings();
    llama_log_set(llama_log_callback_default, nullptr);
    Class superClass = [RNLlama superclass];
    if ([superClass instancesRespondToSelector:@selector(invalidate)]) {
        [super invalidate];
    }
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
