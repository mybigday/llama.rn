#ifdef __cplusplus
#include <list>
#if RNLLAMA_BUILD_FROM_SOURCE
#import "llama.h"
#import "llama-impl.h"
#import "ggml.h"
#import "rn-llama.h"
#import "rn-completion.h"
#import "rn-slot.h"
#import "rn-slot-manager.h"
#import "json-schema-to-grammar.h"
#else
#import <rnllama/llama.h>
#import <rnllama/llama-impl.h>
#import <rnllama/ggml.h>
#import <rnllama/rn-llama.h>
#import <rnllama/rn-completion.h>
#import <rnllama/rn-slot.h>
#import <rnllama/rn-slot-manager.h>
#import <rnllama/json-schema-to-grammar.h>
#endif
#endif


@interface RNLlamaContext : NSObject {
    bool is_metal_enabled;
    bool is_model_loaded;
    NSString * reason_no_metal;
    NSString * gpu_device_name;

    void (^onProgress)(unsigned int progress);

    rnllama::llama_rn_context * llama;
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog;
+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip;
+ (NSString *)getBackendDevicesInfo;
+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress;
- (void)interruptLoad;
- (bool)isMetalEnabled;
- (NSString *)reasonNoMetal;
- (NSString *)gpuDeviceName;
- (NSDictionary *)modelInfo;
- (bool)isModelLoaded;
- (bool)isPredicting;
- (bool)initMultimodal:(NSDictionary *)params;
- (NSDictionary *)getMultimodalSupport;
- (bool)isMultimodalEnabled;
- (void)releaseMultimodal;
- (NSDictionary *)completion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken;
- (void)stopCompletion;
- (NSNumber *)queueCompletion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *tokenResult))onToken onComplete:(void (^)(NSDictionary *result))onComplete;
- (NSNumber *)queueEmbedding:(NSString *)text params:(NSDictionary *)params onResult:(void (^)(int32_t requestId, NSArray *embedding))onResult;
- (NSNumber *)queueRerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params onResults:(void (^)(int32_t requestId, NSArray *results))onResults;
- (void)cancelRequest:(NSNumber *)requestId;
- (BOOL)enableParallelMode:(int)nParallel nBatch:(int)nBatch;
- (void)disableParallelMode;
- (NSDictionary *)tokenize:(NSString *)text mediaPaths:(NSArray *)mediaPaths;
- (NSString *)detokenize:(NSArray *)tokens;
- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params;
- (NSArray *)rerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params;
- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
                           withChatTemplate:(NSString *)chatTemplate
                             withJsonSchema:(NSString *)jsonSchema
                                  withTools:(NSString *)tools
                      withParallelToolCalls:(BOOL)parallelToolCalls
                             withToolChoice:(NSString *)toolChoice
                         withEnableThinking:(BOOL)enableThinking
                    withAddGenerationPrompt:(BOOL)addGenerationPrompt
                                    withNow:(NSString *)nowStr
                     withChatTemplateKwargs:(NSString *)chatTemplateKwargs;
- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate;
- (NSDictionary *)loadSession:(NSString *)path;
- (int)saveSession:(NSString *)path size:(int)size;
- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr;
- (void)applyLoraAdapters:(NSArray *)loraAdapters;
- (void)removeLoraAdapters;
- (NSArray *)getLoadedLoraAdapters;
- (bool)initVocoder:(NSDictionary *)params;
- (bool)isVocoderEnabled;
- (NSDictionary *)getFormattedAudioCompletion:(NSString *)speakerJsonStr textToSpeak:(NSString *)textToSpeak;
- (NSArray *)getAudioCompletionGuideTokens:(NSString *)textToSpeak;
- (NSArray *)decodeAudioTokens:(NSArray *)tokens;
- (void)releaseVocoder;
- (void)invalidate;

@end
