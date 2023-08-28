#import "RNLlama.h"
#import "RNLlamaContext.h"

#ifdef RCT_NEW_ARCH_ENABLED
#import "RNLlamaSpec.h"
#endif

@implementation RNLlama

NSMutableDictionary *llamaContexts;
double llamaContextLimit = 1;

RCT_EXPORT_MODULE()

RCT_EXPORT_METHOD(setContextLimit:(double)limit
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    llamaContextLimit = limit;
    resolve(nil);
}

RCT_EXPORT_METHOD(initContext:(NSDictionary *)contextParams
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    if (llamaContexts == nil) {
        llamaContexts = [[NSMutableDictionary alloc] init];
    }

    if (llamaContextLimit > 0 && [llamaContexts count] >= llamaContextLimit) {
        reject(@"llama_error", @"Context limit reached", nil);
        return;
    }

    RNLlamaContext *context = [RNLlamaContext initWithParams:contextParams];
    if (![context isModelLoaded]) {
        reject(@"llama_cpp_error", @"Failed to load the model", nil);
        return;
    }

    double contextId = (double) arc4random_uniform(1000000);

    NSNumber *contextIdNumber = [NSNumber numberWithDouble:contextId];
    [llamaContexts setObject:context forKey:contextIdNumber];

    resolve(@{
        @"contextId": contextIdNumber,
        @"gpu": @([context isMetalEnabled]),
        @"reasonNoGPU": [context reasonNoMetal],
    });
}

- (NSArray *)supportedEvents {
  return@[
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
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        @try {
            @autoreleasepool {
                NSMutableDictionary* completionResult = [context completion:completionParams
                    onToken:^(NSMutableDictionary *tokenResult) {
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

RCT_EXPORT_METHOD(releaseContext:(double)contextId
                 withResolver:(RCTPromiseResolveBlock)resolve
                 withRejecter:(RCTPromiseRejectBlock)reject)
{
    RNLlamaContext *context = llamaContexts[[NSNumber numberWithDouble:contextId]];
    if (context == nil) {
        reject(@"llama_error", @"Context not found", nil);
        return;
    }
    [context stopCompletion];
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
        [context invalidate];
    }

    [llamaContexts removeAllObjects];
    [llamaContexts release];
    llamaContexts = nil;

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
