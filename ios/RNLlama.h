#ifdef __cplusplus
#if RNLLAMA_BUILD_FROM_SOURCE
#import "rn-llama.h"
#else
#import <rnllama/rn-llama.h>
#endif
#endif

#import <React/RCTEventEmitter.h>
#import <React/RCTBridgeModule.h>

// TODO: Use RNLlamaSpec (Need to refactor NSDictionary usage)
@interface RNLlama : RCTEventEmitter <RCTBridgeModule>

@end
