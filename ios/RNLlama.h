#import <React/RCTBridgeModule.h>
#import <React/RCTBridge+Private.h>
#import <ReactCommon/RCTTurboModule.h>

#if RNLLAMA_BUILD_FROM_SOURCE
#import "json.hpp"
#else
#import <rnllama/nlohmann/json.hpp>
#endif

@interface RNLlama : NSObject <RCTBridgeModule>

@end
