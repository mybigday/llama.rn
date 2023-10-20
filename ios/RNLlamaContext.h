#ifdef __cplusplus
#import "llama.h"
#import "rn-llama.hpp"
#endif


@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    NSString * reason_no_metal;
    bool is_model_loaded;

    rnllama::llama_rn_context * llama;
}

+ (instancetype)initWithParams:(NSDictionary *)params;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;
- (void)stopCompletion;
- (NSArray *)tokenize:(NSString *)text;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSArray *)embedding:(NSString *)text;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;

- (void)invalidate;

@end
