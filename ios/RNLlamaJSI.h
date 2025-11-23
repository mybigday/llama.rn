#import "RNLlama.h"
#import <React/RCTBridge+Private.h>
#import <jsi/jsi.h>
#import <ReactCommon/CallInvoker.h>

@interface RNLlama (JSI)

- (void)installJSIBindingsWithRuntime:(facebook::jsi::Runtime&)runtime
                          callInvoker:(std::shared_ptr<facebook::react::CallInvoker>)callInvoker;

@end
