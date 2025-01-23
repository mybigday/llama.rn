#ifdef __cplusplus
#import "rn-llama.h"
#endif

#import <React/RCTEventEmitter.h>
#import <React/RCTBridgeModule.h>

// TODO: Use RNLlamaSpec (Need to refactor NSDictionary usage)
@interface RNLlama : RCTEventEmitter <RCTBridgeModule>

@end
