#import "RNLlamaContext.h"
#import <Metal/Metal.h>

@implementation RNLlamaContext

+ (instancetype)initWithParams:(NSDictionary *)params {
    // llama_backend_init(false);
    gpt_params defaultParams;

    NSString *modelPath = params[@"model"];
    BOOL isAsset = [params[@"is_model_asset"] boolValue];
    NSString *path = modelPath;
    if (isAsset) path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    defaultParams.model = [path UTF8String];

    if (params[@"embedding"] && [params[@"embedding"] boolValue]) {
        defaultParams.embedding = true;
    }

    if (params[@"n_ctx"]) defaultParams.n_ctx = [params[@"n_ctx"] intValue];
    if (params[@"use_mlock"]) defaultParams.use_mlock = [params[@"use_mlock"]boolValue];

    BOOL isMetalEnabled = false;
    NSString *reasonNoMetal = @"";
    defaultParams.n_gpu_layers = 0;
    if (params[@"n_gpu_layers"] && [params[@"n_gpu_layers"] intValue] > 0) {
#ifdef LM_GGML_USE_METAL
        // Check ggml-metal availability
        NSError * error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device
            newLibraryWithSource:@"#include <metal_stdlib>\n"
                                    "using namespace metal;"
                                    "kernel void test() { simd_sum(0); }"
            options:nil
            error:&error
        ];
        if (error) {
            reasonNoMetal = [error localizedDescription];
        } else {
            id<MTLFunction> kernel = [library newFunctionWithName:@"test"];
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
            if (pipeline == nil) {
                reasonNoMetal = [error localizedDescription];
            } else {
                defaultParams.n_gpu_layers = [params[@"n_gpu_layers"] intValue];
                isMetalEnabled = true;
            }
        }
        device = nil;
#else
        reasonNoMetal = @"Metal is not enabled in this build";
        isMetalEnabled = false;
#endif
    }
    if (params[@"n_batch"]) defaultParams.n_batch = [params[@"n_batch"] intValue];
    if (params[@"use_mmap"]) defaultParams.use_mmap = [params[@"use_mmap"] boolValue];
    if (params[@"memory_f16"]) defaultParams.memory_f16 = [params[@"memory_f16"] boolValue];

    if (params[@"lora"]) {
        defaultParams.lora_adapter = [params[@"lora"] UTF8String];
        defaultParams.use_mmap = false;
    }
    if (params[@"lora_base"]) defaultParams.lora_base = [params[@"lora_base"] UTF8String];

    if (params[@"rope_freq_base"]) defaultParams.rope_freq_base = [params[@"rope_freq_base"] floatValue];
    if (params[@"rope_freq_scale"]) defaultParams.rope_freq_scale = [params[@"rope_freq_scale"] floatValue];

    int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : 0;
    const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
    defaultParams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;

    RNLlamaContext *context = [[RNLlamaContext alloc] init];
    if (context->llama == nullptr) {
        context->llama = new rnllama::llama_rn_context();
    }
    context->is_model_loaded = context->llama->loadModel(defaultParams);
    context->is_metal_enabled = isMetalEnabled;
    context->reason_no_metal = reasonNoMetal;
    return context;
}

- (bool)isMetalEnabled {
    return self->is_metal_enabled;
}

- (NSString *)reasonNoMetal {
    return self->reason_no_metal;
}

- (bool)isModelLoaded {
    return self->is_model_loaded;
}

- (bool)isPredicting {
    return self->is_predicting;
}

- (NSArray *)tokenProbsToDict:(std::vector<rnllama::completion_token_output>)probs {
    NSMutableArray *out = [[NSMutableArray alloc] init];
    for (const auto &prob : probs)
    {
        NSMutableArray *probsForToken = [[NSMutableArray alloc] init];
        for (const auto &p : prob.probs)
        {
            std::string tokStr = rnllama::tokens_to_output_formatted_string(self->llama->ctx, p.tok);
            [probsForToken addObject:@{
                @"tok_str": [NSString stringWithUTF8String:tokStr.c_str()],
                @"prob": [NSNumber numberWithDouble:p.prob]
            }];
        }
        std::string tokStr = rnllama::tokens_to_output_formatted_string(self->llama->ctx, prob.tok);
        [out addObject:@{
            @"content": [NSString stringWithUTF8String:tokStr.c_str()],
            @"probs": probsForToken
        }];
    }
    return out;
}

- (NSDictionary *)completion:(NSDictionary *)params
    onToken:(void (^)(NSMutableDictionary * tokenResult))onToken
{
    self->is_predicting = true;
    self->is_interrupted = false;

    self->llama->rewind();

    llama_reset_timings(self->llama->ctx);

    NSString *prompt = [params objectForKey:@"prompt"];

    self->llama->params.prompt = [prompt UTF8String];

    if (params[@"grammar"]) {
        self->llama->params.grammar = [params[@"grammar"] UTF8String];
    }

    if (params[@"temperature"]) self->llama->params.temp = [params[@"temperature"] doubleValue];

    if (params[@"n_threads"]) {
        int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : self->llama->params.n_threads;
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        // Use 2 threads by default on 4-core devices, 4 threads on more cores
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        self->llama->params.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) self->llama->params.n_predict = [params[@"n_predict"] intValue];
    if (params[@"n_probs"]) self->llama->params.n_probs = [params[@"n_probs"] intValue];

    if (params[@"repeat_last_n"]) self->llama->params.repeat_last_n = [params[@"repeat_last_n"] intValue];
    if (params[@"repeat_penalty"]) self->llama->params.repeat_penalty = [params[@"repeat_penalty"] doubleValue];
    if (params[@"presence_penalty"]) self->llama->params.presence_penalty = [params[@"presence_penalty"] doubleValue];
    if (params[@"frequency_penalty"]) self->llama->params.frequency_penalty = [params[@"frequency_penalty"] doubleValue];

    if (params[@"mirostat"]) self->llama->params.mirostat = [params[@"mirostat"] intValue];
    if (params[@"mirostat_tau"]) self->llama->params.mirostat_tau = [params[@"mirostat_tau"] doubleValue];
    if (params[@"mirostat_eta"]) self->llama->params.mirostat_eta = [params[@"mirostat_eta"] doubleValue];

    if (params[@"top_k"]) self->llama->params.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) self->llama->params.top_p = [params[@"top_p"] doubleValue];
    if (params[@"tfs_z"]) self->llama->params.tfs_z = [params[@"tfs_z"] doubleValue];

    if (params[@"typical_p"]) self->llama->params.typical_p = [params[@"typical_p"] doubleValue];

    self->llama->params.antiprompt.clear();
    if (params[@"stop"]) {
        NSArray *stop = params[@"stop"];
        for (NSString *s in stop) {
            self->llama->params.antiprompt.push_back([s UTF8String]);
        }
    }

    self->llama->params.logit_bias.clear();
    if (params[@"ignore_eos"] && [params[@"ignore_eos"] boolValue]) {
        self->llama->params.logit_bias[llama_token_eos(self->llama->ctx)] = -INFINITY;
    }

    if (params[@"logit_bias"] && [params[@"logit_bias"] isKindOfClass:[NSArray class]]) {
        const int n_vocab = llama_n_vocab(self->llama->ctx);
        NSArray *logit_bias = params[@"logit_bias"];
        for (NSArray *el in logit_bias) {
            if ([el isKindOfClass:[NSArray class]] && [el count] == 2) {
                llama_token tok = [el[0] intValue];
                if (tok >= 0 && tok < n_vocab) {
                    if ([el[1] isKindOfClass:[NSNumber class]]) {
                        self->llama->params.logit_bias[tok] = [el[1] doubleValue];
                    } else if ([el[1] isKindOfClass:[NSNumber class]] && ![el[1] boolValue]) {
                        self->llama->params.logit_bias[tok] = -INFINITY;
                    }
                }
            }
        }
    }

    if (!self->llama->loadGrammar()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to load grammar" userInfo:nil];
    }

    self->llama->loadPrompt();
    self->llama->beginCompletion();

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->has_next_token) {
        const rnllama::completion_token_output token_with_probs = self->llama->doCompletion();
        const std::string token_text = token_with_probs.tok == -1 ? "" : llama_token_to_str(self->llama->ctx, token_with_probs.tok);
        if (self->llama->multibyte_pending > 0) {
            continue;
        }

        size_t pos = std::min(sent_count, self->llama->generated_text.size());

        const std::string str_test = self->llama->generated_text.substr(pos);
        size_t stop_pos =
            self->llama->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
        if (stop_pos != std::string::npos) {
            self->llama->generated_text.erase(
                self->llama->generated_text.begin() + pos + stop_pos,
                self->llama->generated_text.end());
            pos = std::min(sent_count, self->llama->generated_text.size());
        } else {
            stop_pos = self->llama->findStoppingStrings(str_test, token_text.size(),
                rnllama::STOP_PARTIAL);
        }

        const std::string to_send = stop_pos == std::string::npos ?
          self->llama->generated_text.substr(pos, std::string::npos) :
          ""; // just don't send anything if we're not done
        sent_count += to_send.size();

        std::vector<rnllama::completion_token_output> probs_output = {};

        NSMutableDictionary *tokenResult = [[NSMutableDictionary alloc] init];
        tokenResult[@"token"] = [NSString stringWithUTF8String:to_send.c_str()];

        if (self->llama->params.n_probs > 0) {
            const std::vector<llama_token> to_send_toks = llama_tokenize(self->llama->ctx, to_send, false);
            size_t probs_pos = std::min(sent_token_probs_index, self->llama->generated_token_probs.size());
            size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), self->llama->generated_token_probs.size());
            if (probs_pos < probs_stop_pos) {
                probs_output = std::vector<rnllama::completion_token_output>(self->llama->generated_token_probs.begin() + probs_pos, self->llama->generated_token_probs.begin() + probs_stop_pos);
            }
            sent_token_probs_index = probs_stop_pos;

            tokenResult[@"completion_probabilities"] = [self tokenProbsToDict:probs_output];
        }

        onToken(tokenResult);

        if (self->is_interrupted) {
            break;
        }
    }

    llama_print_timings(llama->ctx);
    self->is_predicting = false;

    const auto timings = llama_get_timings(self->llama->ctx);
    return @{
        @"text": [NSString stringWithUTF8String:self->llama->generated_text.c_str()],
        @"completion_probabilities": [self tokenProbsToDict:self->llama->generated_token_probs],
        @"tokens_predicted": @(self->llama->num_tokens_predicted),
        @"tokens_evaluated": @(self->llama->num_prompt_tokens),
        @"truncated": @(self->llama->truncated),
        @"stopped_eos": @(self->llama->stopped_eos),
        @"stopped_word": @(self->llama->stopped_word),
        @"stopped_limit": @(self->llama->stopped_limit),
        @"stopping_word": [NSString stringWithUTF8String:self->llama->stopping_word.c_str()],
        @"tokens_cached": @(self->llama->n_past),
        @"timings": @{
            @"prompt_n": @(timings.n_p_eval),
            @"prompt_ms": @(timings.t_p_eval_ms),
            @"prompt_per_token_ms": @(timings.t_p_eval_ms / timings.n_p_eval),
            @"prompt_per_second": @(1e3 / timings.t_p_eval_ms * timings.n_p_eval),

            @"predicted_n": @(timings.n_eval),
            @"predicted_ms": @(timings.t_eval_ms),
            @"predicted_per_token_ms": @(timings.t_eval_ms / timings.n_eval),
            @"predicted_per_second": @(1e3 / timings.t_eval_ms * timings.n_eval),
        }
    };
}

- (void)stopCompletion {
    self->is_interrupted = true;
}

- (NSArray *)tokenize:(NSString *)text {
    const std::vector<llama_token> toks = llama_tokenize(self->llama->ctx, [text UTF8String], false);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (llama_token tok : toks) {
      printf("%d\n", tok);
        [result addObject:@(tok)];
    }
    return result;
}

- (NSArray *)embedding:(NSString *)text {
    if (self->llama->params.embedding != true) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding not enabled" userInfo:nil];
    }

    self->is_predicting = true;
    self->is_interrupted = false;

    self->llama->rewind();

    llama_reset_timings(self->llama->ctx);

    self->llama->params.prompt = [text UTF8String];

    self->llama->params.n_predict = 0;
    self->llama->loadPrompt();
    self->llama->beginCompletion();
    self->llama->doCompletion();

    std::vector<float> result = self->llama->getEmbedding();

    NSMutableArray *embeddingResult = [[NSMutableArray alloc] init];
    for (float f : result) {
        [embeddingResult addObject:@(f)];
    }

    self->is_predicting = false;
    return embeddingResult;
}

- (void)invalidate {
    if (self->llama->grammar != nullptr) {
        llama_grammar_free(self->llama->grammar);
    }
    delete self->llama;

    // llama_backend_free();
}

@end
