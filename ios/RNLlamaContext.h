#ifdef __cplusplus
#import "llama.h"
#import "rn-llama.hpp"
#endif


@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    NSString * reason_no_metal;
    bool is_model_loaded;
    bool is_predicting;
    bool is_interrupted;

    rnllama::llama_rn_context * llama;
}

+ (instancetype)initWithParams:(NSDictionary *)params;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSDictionary *tokenResult))onToken;
- (void)stopCompletion;

- (void)invalidate;

@end
