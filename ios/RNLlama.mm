#import "RNLlama.h"
#import "RNLlamaContext.h"

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNLlamaSpec.h"
#endif

@implementation RNLlama

NSMutableDictionary *llamaContexts;
double llamaContextLimit = -1;
dispatch_queue_t llamaDQueue;

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(toggleNativeLog:(BOOL)enabled) {
    void (^onEmitLog)(NSString *level, NSString *text) = nil;
    if (enabled) {
        onEmitLog = ^(NSString *level, NSString *text) {
            [self sendEventWithName:@"@RNLlama_onNativeLog" body:@{ @"level": level, @"text": text }];
        };
    }
    [RNLlamaContext toggleNativeLog:enabled onEmitLog:onEmitLog];
}

RCT_EXPORT_METHOD(setContextLimit:(double)limit
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    llamaContextLimit = limit;
    resolve(nil);
}

RCT_EXPORT_METHOD(modelInfo:(NSString *)path
                 withSkip:(NSArray *)skip
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    resolve([RNLlamaContext modelInfo:path skip:skip]);
}

RCT_EXPORT_METHOD(getBackendDevicesInfo:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    @try {
        NSString *devicesInfoJson = [RNLlamaContext getBackendDevicesInfo];
        resolve(devicesInfoJson);
    } @catch (NSException *exception) {
        reject(@"llama_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(initContext:(double)contextId
                 withContextParams:(NSDictionary *)contextParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    NSNumber *contextIdNumber = [NSNumber numberWithDouble:contextId];
    if (llamaContexts[contextIdNumber] != nil) {
        reject(@"llama_error", @"Context already exists", nil);
        return;
    }

    if (llamaDQueue == nil) {
        llamaDQueue = dispatch_queue_create("com.rnllama", DISPATCH_QUEUE_SERIAL);
    }

    if (llamaContexts == nil) {
        llamaContexts = [[NSMutableDictionary alloc] init];
    }

    if (llamaContextLimit > -1 && [llamaContexts count] >= llamaContextLimit) {
        reject(@"llama_error", @"Context limit reached", nil);
        return;
    }

    @try {
      RNLlamaContext *context = [RNLlamaContext initWithParams:contextParams onProgress:^(unsigned int progress) {
          dispatch_async(dispatch_get_main_queue(), ^{
              [self sendEventWithName:@"@RNLlama_onInitContextProgress" body:@{ @"contextId": @(contextId), @"progress": @(progress) }];
          });
      }];
      if (![context isModelLoaded]) {
          reject(@"llama_cpp_error", @"Failed to load the model", nil);
          return;
      }

      [llamaContexts setObject:context forKey:contextIdNumber];

      NSMutableDictionary *result = [@{
          @"gpu": @([context isMetalEnabled]),
          @"reasonNoGPU": [context reasonNoMetal],
          @"model": [context modelInfo],
      } mutableCopy];
      NSString *gpuDevice = [context gpuDeviceName];
      if (gpuDevice != nil && [gpuDevice length] > 0) {
          result[@"gpuDevice"] = gpuDevice;
      }
      resolve(result);
    } @catch (NSException *exception) {
      reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(getFormattedChat:(double)contextId
                 withMessages:(NSString *)messages
                 withTemplate:(NSString *)chatTemplate
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    try {
        if ([params[@"jinja"] boolValue]) {
            NSString *jsonSchema = params[@"json_schema"];
            NSString *tools = params[@"tools"];
            BOOL parallelToolCalls = [params[@"parallel_tool_calls"] boolValue];
            NSString *toolChoice = params[@"tool_choice"];
            BOOL enableThinking = [params[@"enable_thinking"] boolValue];
            BOOL addGenerationPrompt = params[@"add_generation_prompt"] ? [params[@"add_generation_prompt"] boolValue] : YES;
            NSString *nowStr = params[@"now"] ?: @"";
            NSString *chatTemplateKwargs = params[@"chat_template_kwargs"] ?: @"";
            resolve([context getFormattedChatWithJinja:messages
                withChatTemplate:chatTemplate
                withJsonSchema:jsonSchema
                withTools:tools
                withParallelToolCalls:parallelToolCalls
                withToolChoice:toolChoice
                withEnableThinking:enableThinking
                withAddGenerationPrompt:addGenerationPrompt
                withNow:nowStr
                withChatTemplateKwargs:chatTemplateKwargs
            ]);
        } else {
            resolve([context getFormattedChat:messages withChatTemplate:chatTemplate]);
        }
    } catch (const nlohmann::json_abi_v3_12_0::detail::parse_error& e) {
        NSString *errorMessage = [NSString stringWithUTF8String:e.what()];
        reject(@"llama_json_parse_error", [NSString stringWithFormat:@"JSON parse error in getFormattedChat: %@", errorMessage], nil);
    } catch (const std::invalid_argument& e) {
        NSString *errorMessage = [NSString stringWithUTF8String:e.what()];
        reject(@"llama_invalid_argument", [NSString stringWithFormat:@"Invalid argument in getFormattedChat: %@", errorMessage], nil);
    } catch (const std::runtime_error& e) {
        NSString *errorMessage = [NSString stringWithUTF8String:e.what()];
        reject(@"llama_runtime_error", [NSString stringWithFormat:@"Runtime error in getFormattedChat: %@", errorMessage], nil);
    } catch (const std::exception& e) {
        NSString *errorMessage = [NSString stringWithUTF8String:e.what()];
        reject(@"llama_cpp_error", [NSString stringWithFormat:@"C++ exception in getFormattedChat: %@", errorMessage], nil);
    } catch (NSException *nsException) {
        reject(@"llama_objc_error", [NSString stringWithFormat:@"Objective-C exception in getFormattedChat: %@", nsException.reason], nil);
    }
}

RCT_EXPORT_METHOD(loadSession:(double)contextId
                 withFilePath:(NSString *)filePath
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                resolve([context loadSession:filePath]);
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
        }
    });
}

RCT_EXPORT_METHOD(saveSession:(double)contextId
                 withFilePath:(NSString *)filePath
                 withSize:(double)size
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                int count = [context saveSession:filePath size:(int)size];
                resolve(@(count));
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
        }
    });
}

- (NSArray *)supportedEvents {
  return@[
    @"@RNLlama_onInitContextProgress",
    @"@RNLlama_onToken",
    @"@RNLlama_onComplete",
    @"@RNLlama_onEmbeddingResult",
    @"@RNLlama_onRerankResults",
    @"@RNLlama_onNativeLog",
  ];
}

RCT_EXPORT_METHOD(completion:(double)contextId
                 withCompletionParams:(NSDictionary *)completionParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    dispatch_async(llamaDQueue, ^{
        @try {
            @autoreleasepool {
                NSDictionary* completionResult = [context completion:completionParams
                    onToken:^(NSMutableDictionary *tokenResult) {
                        if (![completionParams[@"emit_partial_completion"] boolValue]) return;
                        dispatch_async(dispatch_get_main_queue(), ^{
                            [self sendEventWithName:@"@RNLlama_onToken"
                                body:@{
                                    @"contextId": [NSNumber numberWithDouble:contextId],
                                    @"tokenResult": tokenResult
                                }
                            ];
                            [tokenResult release];
                        });
                    }
                ];
                resolve(completionResult);
            }
        } @catch (NSException *exception) {
            reject(@"llama_cpp_error", exception.reason, nil);
            [context stopCompletion];
        }
    });

}

RCT_EXPORT_METHOD(stopCompletion:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    [context stopCompletion];
    resolve(nil);
}

RCT_EXPORT_METHOD(tokenize:(double)contextId
                  text:(NSString *)text
                  mediaPaths:(NSArray *)mediaPaths
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSMutableDictionary *result = [context tokenize:text mediaPaths:mediaPaths];
        resolve(result);
        [result release];
    } @catch (NSException *exception) {
        reject(@"llama_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(detokenize:(double)contextId
                  tokens:(NSArray *)tokens
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve([context detokenize:tokens]);
}

RCT_EXPORT_METHOD(embedding:(double)contextId
                  text:(NSString *)text
                  params:(NSDictionary *)params
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSDictionary *embedding = [context embedding:text params:params];
        resolve(embedding);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(rerank:(double)contextId
                  query:(NSString *)query
                  documents:(NSArray<NSString *> *)documents
                  params:(NSDictionary *)params
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject) {
  RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
  if (context == nil) {
    reject(@"context_not_found", @"Context not found", nil);
    return;
  }
  @try {
    NSArray *result = [context rerank:query documents:documents params:params];
    resolve(result);
  } @catch (NSException *exception) {
    reject(@"rerank_error", exception.reason, nil);
  }
}

RCT_EXPORT_METHOD(bench:(double)contextId
                  pp:(int)pp
                  tg:(int)tg
                  pl:(int)pl
                  nr:(int)nr
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSString *benchResults = [context bench:pp tg:tg pl:pl nr:nr];
        resolve(benchResults);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(applyLoraAdapters:(double)contextId
                 withLoraAdapters:(NSArray *)loraAdapters
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    [context applyLoraAdapters:loraAdapters];
    resolve(nil);
}

RCT_EXPORT_METHOD(removeLoraAdapters:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }
    [context removeLoraAdapters];
    resolve(nil);
}

RCT_EXPORT_METHOD(getLoadedLoraAdapters:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve([context getLoadedLoraAdapters]);
}

RCT_EXPORT_METHOD(initMultimodal:(double)contextId
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }

    @try {
        bool success = [context initMultimodal:params];
        resolve(@(success));
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(isMultimodalEnabled:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    resolve(@([context isMultimodalEnabled]));
}

RCT_EXPORT_METHOD(getMultimodalSupport:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    if (![context isMultimodalEnabled]) {
        reject(@"llama_error", @"Multimodal is not enabled", nil);
        return;
    }

    NSDictionary *multimodalSupport = [context getMultimodalSupport];
    resolve(multimodalSupport);
}

RCT_EXPORT_METHOD(releaseMultimodal:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    [context releaseMultimodal];
    resolve(nil);
}

RCT_EXPORT_METHOD(initVocoder:(double)contextId
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if ([context isPredicting]) {
        reject(@"llama_error", @"Context is busy", nil);
        return;
    }

    @try {
        bool success = [context initVocoder:params];
        resolve(@(success));
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(isVocoderEnabled:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    resolve(@([context isVocoderEnabled]));
}

RCT_EXPORT_METHOD(getFormattedAudioCompletion:(double)contextId
                 withSpeakerJsonStr:(NSString *)speakerJsonStr
                 withTextToSpeak:(NSString *)textToSpeak
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    if (![context isVocoderEnabled]) {
        reject(@"llama_error", @"Vocoder is not enabled", nil);
        return;
    }

    @try {
        NSDictionary *result = [context getFormattedAudioCompletion:speakerJsonStr textToSpeak:textToSpeak];
        resolve(result);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(getAudioCompletionGuideTokens:(double)contextId
                 withTextToSpeak:(NSString *)textToSpeak
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    if (![context isVocoderEnabled]) {
        reject(@"llama_error", @"Vocoder is not enabled", nil);
        return;
    }

    @try {
        NSArray *guideTokens = [context getAudioCompletionGuideTokens:textToSpeak];
        resolve(guideTokens);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(decodeAudioTokens:(double)contextId
                 withTokens:(NSArray *)tokens
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    if (![context isVocoderEnabled]) {
        reject(@"llama_error", @"Vocoder is not enabled", nil);
        return;
    }

    @try {
        NSArray *audioData = [context decodeAudioTokens:tokens];
        resolve(audioData);
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

RCT_EXPORT_METHOD(releaseVocoder:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    [context releaseVocoder];
    resolve(nil);
}

RCT_EXPORT_METHOD(releaseContext:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    if (![context isModelLoaded]) {
      [context interruptLoad];
    }
    [context stopCompletion];
    dispatch_barrier_sync(llamaDQueue, ^{});
    [context invalidate];
    [llamaContexts removeObjectForKey:[NSNumber numberWithDouble:contextId]];
    resolve(nil);
}

RCT_EXPORT_METHOD(releaseAllContexts:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    [self invalidate];
    resolve(nil);
}


- (void)invalidate {
    if (llamaContexts == nil) {
        return;
    }

    for (NSNumber *contextId in llamaContexts) {
        RNLlamaContext *context = llamaContexts[contextId];
        [context stopCompletion];
        dispatch_barrier_sync(llamaDQueue, ^{});
        [context invalidate];
    }

    [llamaContexts removeAllObjects];
    [llamaContexts release];
    llamaContexts = nil;

    if (llamaDQueue != nil) {
        dispatch_release(llamaDQueue);
        llamaDQueue = nil;
    }

    [super invalidate];
}

// Enable or disable parallel decoding mode
RCT_EXPORT_METHOD(enableParallelMode:(double)contextId
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    BOOL enabled = params[@"enabled"] ? [params[@"enabled"] boolValue] : YES;
    int nParallel = params[@"n_parallel"] ? [params[@"n_parallel"] intValue] : 2;
    int nBatch = params[@"n_batch"] ? [params[@"n_batch"] intValue] : 512;

    @try {
        if (enabled) {
            BOOL success = [context enableParallelMode:nParallel nBatch:nBatch];
            resolve(@(success));
        } else {
            [context disableParallelMode];
            resolve(@(YES));
        }
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

// Parallel decoding: Queue a completion request
RCT_EXPORT_METHOD(queueCompletion:(double)contextId
                 withCompletionParams:(NSDictionary *)completionParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    @try {
        NSNumber *requestId = [context queueCompletion:completionParams
            onToken:^(NSMutableDictionary *tokenResult) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    NSNumber *reqId = tokenResult[@"requestId"];
                    [self sendEventWithName:@"@RNLlama_onToken"
                        body:@{
                            @"contextId": [NSNumber numberWithDouble:contextId],
                            @"requestId": reqId ?: @(-1),
                            @"tokenResult": tokenResult
                        }
                    ];
                });
            }
            onComplete:^(NSDictionary *result) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    // Extract requestId from result (set by RNLlamaContext)
                    NSNumber *reqId = result[@"requestId"];
                    [self sendEventWithName:@"@RNLlama_onComplete"
                        body:@{
                            @"contextId": [NSNumber numberWithDouble:contextId],
                            @"requestId": reqId ?: @(-1),
                            @"result": result
                        }
                    ];
                });
            }
        ];

        if ([requestId intValue] == -1) {
            reject(@"llama_error", @"Failed to queue completion request", nil);
            return;
        }

        resolve(@{@"requestId": requestId});
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

// Cancel a queued request
RCT_EXPORT_METHOD(cancelRequest:(double)contextId
                 withRequestId:(double)requestId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    [context cancelRequest:[NSNumber numberWithInt:(int)requestId]];
    resolve(nil);
}

// Queue an embedding request (async, non-blocking)
RCT_EXPORT_METHOD(queueEmbedding:(double)contextId
                 withText:(NSString *)text
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    @try {
        NSNumber *requestId = [context queueEmbedding:text params:params onResult:^(int32_t reqId, NSArray *embedding) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self sendEventWithName:@"@RNLlama_onEmbeddingResult"
                    body:@{
                        @"contextId": [NSNumber numberWithDouble:contextId],
                        @"requestId": @(reqId),
                        @"embedding": embedding
                    }
                ];
            });
        }];

        if ([requestId intValue] == -1) {
            reject(@"llama_error", @"Failed to queue embedding request", nil);
            return;
        }

        resolve(@{@"requestId": requestId});
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

// Queue a rerank request (async, non-blocking)
RCT_EXPORT_METHOD(queueRerank:(double)contextId
                 withQuery:(NSString *)query
                 withDocuments:(NSArray<NSString *> *)documents
                 withParams:(NSDictionary *)params
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }

    @try {
        NSNumber *requestId = [context queueRerank:query documents:documents params:params onResults:^(int32_t reqId, NSArray *results) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [self sendEventWithName:@"@RNLlama_onRerankResults"
                    body:@{
                        @"contextId": [NSNumber numberWithDouble:contextId],
                        @"requestId": @(reqId),
                        @"results": results
                    }
                ];
            });
        }];

        if ([requestId intValue] == -1) {
            reject(@"llama_error", @"Failed to queue rerank request", nil);
            return;
        }

        resolve(@{@"requestId": requestId});
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
    }
}

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNLlamaSpecJSI>(params);
}
#endif

@end
