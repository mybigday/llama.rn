#ifdef __cplusplus
#import "llama.h"
#import "rn-llama.hpp"
#endif


@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    NSString * reason_no_metal;
    bool is_model_loaded;
    NSString * model_desc;
    uint64_t model_size;
    uint64_t model_n_params;
    NSDictionary * metadata;

    rnllama::llama_rn_context * llama;
}

+ (instancetype)initWithParams:(NSDictionary *)params;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (NSDictionary *)metadata;
- (NSString *)modelDesc;
- (uint64_t)modelSize;
- (uint64_t)modelNParams;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;
- (void)stopCompletion;
- (NSArray *)tokenize:(NSString *)text;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSArray *)embedding:(NSString *)text;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;
- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr;

- (void)invalidate;

@end
