#ifdef __cplusplus
#import "rn-llama.hpp"
#endif

#import <React/RCTEventEmitter.h>
#import <React/RCTBridgeModule.h>

// TODO: Use RNLlamaSpec (Need to refactor NSDictionary usage)
@interface Llama : RCTEventEmitter <RCTBridgeModule>

@end
