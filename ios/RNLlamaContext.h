#ifdef __cplusplus
#if RNLLAMA_BUILD_FROM_SOURCE
#import "llama.h"
#import "llama-impl.h"
#import "ggml.h"
#import "rn-llama.h"
#import "json-schema-to-grammar.h"
#else
#import <rnllama/llama.h>
#import <rnllama/llama-impl.h>
#import <rnllama/ggml.h>
#import <rnllama/rn-llama.h>
#import <rnllama/json-schema-to-grammar.h>
#endif
#endif


@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    bool is_model_loaded;
    NSString * reason_no_metal;

    void (^onProgress)(unsigned int progress);

    rnllama::llama_rn_context * llama;
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog;
+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip;
+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress;
- (void)interruptLoad;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (NSDictionary *)modelInfo;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;
- (void)stopCompletion;
- (NSArray *)tokenize:(NSString *)text;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params;
- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
    withChatTemplate:(NSString *)chatTemplate
    withJsonSchema:(NSString *)jsonSchema
    withTools:(NSString *)tools
    withParallelToolCalls:(BOOL)parallelToolCalls
    withToolChoice:(NSString *)toolChoice;
- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;
- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr;
- (void)applyLoraAdapters:(NSArray *)loraAdapters;
- (void)removeLoraAdapters;
- (NSArray *)getLoadedLoraAdapters;
- (void)invalidate;

@end
