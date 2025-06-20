#import <React/RCTEventEmitter.h>
#import <React/RCTBridgeModule.h>

#if RNLLAMA_BUILD_FROM_SOURCE
#import "json.hpp"
#else
#import <rnllama/nlohmann/json.hpp>
#endif

// TODO: Use RNLlamaSpec (Need to refactor NSDictionary usage)
@interface RNLlama : RCTEventEmitter <RCTBridgeModule>

@end
