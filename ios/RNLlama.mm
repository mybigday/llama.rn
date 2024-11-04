#import "RNLlama.h"
#import "RNLlamaContext.h"

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNLlamaSpec.h"
#endif

@implementation RNLlama

NSMutableDictionary *llamaContexts;
double llamaContextLimit = 1;
dispatch_queue_t llamaDQueue;

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(setContextLimit:(double)limit
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    llamaContextLimit = limit;
    resolve(nil);
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

    if (llamaContextLimit > 0 && [llamaContexts count] >= llamaContextLimit) {
        reject(@"llama_error", @"Context limit reached", nil);
        return;
    }

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

    resolve(@{
        @"gpu": @([context isMetalEnabled]),
        @"reasonNoGPU": [context reasonNoMetal],
        @"model": [context modelInfo],
    });
}

RCT_EXPORT_METHOD(getFormattedChat:(double)contextId
                 withMessages:(NSArray *)messages
                 withTemplate:(NSString *)chatTemplate
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    resolve([context getFormattedChat:messages withTemplate:chatTemplate]);
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
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    NSMutableArray *tokens = [context tokenize:text];
    resolve(@{ @"tokens": tokens });
    [tokens release];
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
                  withResolver:(RCTPromiseResolveBlock)resolve
                  withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    @try {
        NSMutableArray *embedding = [context embedding:text];
        resolve(@{ @"embedding": embedding });
        [embedding release];
    } @catch (NSException *exception) {
        reject(@"llama_cpp_error", exception.reason, nil);
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

// Don't compile this code when we build for the old architecture.
#ifdef RCT_NEW_ARCH_ENABLED
- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams &)params
{
    return std::make_shared<facebook::react::NativeRNLlamaSpecJSI>(params);
}
#endif

@end
