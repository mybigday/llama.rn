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

    if (params[@"lora"]) {
        float lora_scaled = 1.0f;
        if (params[@"lora_scaled"]) lora_scaled = [params[@"lora_scaled"] floatValue];
        defaultParams.lora_adapter.push_back({[params[@"lora"] UTF8String], lora_scaled});
        defaultParams.use_mmap = false;
    }
    if (params[@"lora_base"]) defaultParams.lora_base = [params[@"lora_base"] UTF8String];

    if (params[@"rope_freq_base"]) defaultParams.rope_freq_base = [params[@"rope_freq_base"] floatValue];
    if (params[@"rope_freq_scale"]) defaultParams.rope_freq_scale = [params[@"rope_freq_scale"] floatValue];

    if (params[@"seed"]) defaultParams.seed = [params[@"seed"] intValue];

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
    return is_metal_enabled;
}

- (NSString *)reasonNoMetal {
    return reason_no_metal;
}

- (bool)isModelLoaded {
    return is_model_loaded;
}

- (bool)isPredicting {
    return llama->is_predicting;
}

- (NSArray *)tokenProbsToDict:(std::vector<rnllama::completion_token_output>)probs {
    NSMutableArray *out = [[NSMutableArray alloc] init];
    for (const auto &prob : probs)
    {
        NSMutableArray *probsForToken = [[NSMutableArray alloc] init];
        for (const auto &p : prob.probs)
        {
            std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, p.tok);
            [probsForToken addObject:@{
                @"tok_str": [NSString stringWithUTF8String:tokStr.c_str()],
                @"prob": [NSNumber numberWithDouble:p.prob]
            }];
        }
        std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, prob.tok);
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
    llama->rewind();

    llama_reset_timings(llama->ctx);

    NSString *prompt = [params objectForKey:@"prompt"];

    llama->params.prompt = [prompt UTF8String];
    llama->params.seed = params[@"seed"] ? [params[@"seed"] intValue] : -1;

    if (params[@"n_threads"]) {
        int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : llama->params.n_threads;
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        // Use 2 threads by default on 4-core devices, 4 threads on more cores
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        llama->params.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) llama->params.n_predict = [params[@"n_predict"] intValue];

    auto & sparams = llama->params.sparams;

    if (params[@"temperature"]) sparams.temp = [params[@"temperature"] doubleValue];

    if (params[@"n_probs"]) sparams.n_probs = [params[@"n_probs"] intValue];

    if (params[@"penalty_last_n"]) sparams.penalty_last_n = [params[@"penalty_last_n"] intValue];
    if (params[@"penalty_repeat"]) sparams.penalty_repeat = [params[@"penalty_repeat"] doubleValue];
    if (params[@"penalty_freq"]) sparams.penalty_freq = [params[@"penalty_freq"] doubleValue];
    if (params[@"penalty_present"]) sparams.penalty_present = [params[@"penalty_present"] doubleValue];

    if (params[@"mirostat"]) sparams.mirostat = [params[@"mirostat"] intValue];
    if (params[@"mirostat_tau"]) sparams.mirostat_tau = [params[@"mirostat_tau"] doubleValue];
    if (params[@"mirostat_eta"]) sparams.mirostat_eta = [params[@"mirostat_eta"] doubleValue];
    if (params[@"penalize_nl"]) sparams.penalize_nl = [params[@"penalize_nl"] boolValue];

    if (params[@"top_k"]) sparams.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) sparams.top_p = [params[@"top_p"] doubleValue];
    if (params[@"min_p"]) sparams.min_p = [params[@"min_p"] doubleValue];
    if (params[@"tfs_z"]) sparams.tfs_z = [params[@"tfs_z"] doubleValue];

    if (params[@"typical_p"]) sparams.typical_p = [params[@"typical_p"] doubleValue];

    if (params[@"grammar"]) {
        sparams.grammar = [params[@"grammar"] UTF8String];
    }

    llama->params.antiprompt.clear();
    if (params[@"stop"]) {
        NSArray *stop = params[@"stop"];
        for (NSString *s in stop) {
            llama->params.antiprompt.push_back([s UTF8String]);
        }
    }

    sparams.logit_bias.clear();
    if (params[@"ignore_eos"] && [params[@"ignore_eos"] boolValue]) {
        sparams.logit_bias[llama_token_eos(llama->model)] = -INFINITY;
    }

    if (params[@"logit_bias"] && [params[@"logit_bias"] isKindOfClass:[NSArray class]]) {
        const int n_vocab = llama_n_vocab(llama_get_model(llama->ctx));
        NSArray *logit_bias = params[@"logit_bias"];
        for (NSArray *el in logit_bias) {
            if ([el isKindOfClass:[NSArray class]] && [el count] == 2) {
                llama_token tok = [el[0] intValue];
                if (tok >= 0 && tok < n_vocab) {
                    if ([el[1] isKindOfClass:[NSNumber class]]) {
                        sparams.logit_bias[tok] = [el[1] doubleValue];
                    } else if ([el[1] isKindOfClass:[NSNumber class]] && ![el[1] boolValue]) {
                        sparams.logit_bias[tok] = -INFINITY;
                    }
                }
            }
        }
    }

    if (!llama->initSampling()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to initialize sampling" userInfo:nil];
    }
    llama->beginCompletion();
    llama->loadPrompt();

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->has_next_token && !llama->is_interrupted) {
        const rnllama::completion_token_output token_with_probs = llama->doCompletion();
        if (token_with_probs.tok == -1 || llama->multibyte_pending > 0) {
            continue;
        }
        const std::string token_text = llama_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->generated_text.size());

        const std::string str_test = llama->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->generated_text.erase(
                llama->generated_text.begin() + pos + stop_pos,
                llama->generated_text.end());
            pos = std::min(sent_count, llama->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->findStoppingStrings(str_test, token_text.size(),
                rnllama::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<rnllama::completion_token_output> probs_output = {};

            NSMutableDictionary *tokenResult = [[NSMutableDictionary alloc] init];
            tokenResult[@"token"] = [NSString stringWithUTF8String:to_send.c_str()];

            if (llama->params.sparams.n_probs > 0) {
                const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
                size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
                size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
                if (probs_pos < probs_stop_pos) {
                    probs_output = std::vector<rnllama::completion_token_output>(llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
                }
                sent_token_probs_index = probs_stop_pos;

                tokenResult[@"completion_probabilities"] = [self tokenProbsToDict:probs_output];
            }

            onToken(tokenResult);
        }
    }

    llama_print_timings(llama->ctx);
    llama->is_predicting = false;

    const auto timings = llama_get_timings(llama->ctx);
    return @{
        @"text": [NSString stringWithUTF8String:llama->generated_text.c_str()],
        @"completion_probabilities": [self tokenProbsToDict:llama->generated_token_probs],
        @"tokens_predicted": @(llama->num_tokens_predicted),
        @"tokens_evaluated": @(llama->num_prompt_tokens),
        @"truncated": @(llama->truncated),
        @"stopped_eos": @(llama->stopped_eos),
        @"stopped_word": @(llama->stopped_word),
        @"stopped_limit": @(llama->stopped_limit),
        @"stopping_word": [NSString stringWithUTF8String:llama->stopping_word.c_str()],
        @"tokens_cached": @(llama->n_past),
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
    llama->is_interrupted = true;
}

- (NSArray *)tokenize:(NSString *)text {
    const std::vector<llama_token> toks = llama_tokenize(llama->ctx, [text UTF8String], false);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (llama_token tok : toks) {
        [result addObject:@(tok)];
    }
    return result;
}

- (NSString *)detokenize:(NSArray *)tokens {
    std::vector<llama_token> toks;
    for (NSNumber *tok in tokens) {
        toks.push_back([tok intValue]);
    }
    const std::string text = rnllama::tokens_to_str(llama->ctx, toks.cbegin(), toks.cend());
    return [NSString stringWithUTF8String:text.c_str()];
}

- (NSArray *)embedding:(NSString *)text {
    if (llama->params.embedding != true) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not enabled" userInfo:nil];
    }

    llama->rewind();

    llama_reset_timings(llama->ctx);

    llama->params.prompt = [text UTF8String];

    llama->params.n_predict = 0;
    llama->loadPrompt();
    llama->beginCompletion();
    llama->doCompletion();

    std::vector<float> result = llama->getEmbedding();

    NSMutableArray *embeddingResult = [[NSMutableArray alloc] init];
    for (float f : result) {
        [embeddingResult addObject:@(f)];
    }

    llama->is_predicting = false;
    return embeddingResult;
}

- (NSDictionary *)loadSession:(NSString *)path {
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session file does not exist" userInfo:nil];
    }

    size_t n_token_count_out = 0;
    llama->embd.resize(llama->params.n_ctx);
    if (!llama_load_session_file(llama->ctx, [path UTF8String], llama->embd.data(), llama->embd.capacity(), &n_token_count_out)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to load session" userInfo:nil];
    }
    llama->embd.resize(n_token_count_out);
    const std::string text = rnllama::tokens_to_str(llama->ctx, llama->embd.cbegin(), llama->embd.cend());
    return @{
        @"tokens_loaded": @(n_token_count_out),
        @"prompt": [NSString stringWithUTF8String:text.c_str()]
    };
}

- (int)saveSession:(NSString *)path size:(int)size {
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    std::vector<llama_token> session_tokens = llama->embd;
    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_save_session_file(llama->ctx, [path UTF8String], session_tokens.data(), save_size)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to save session" userInfo:nil];
    }
    return session_tokens.size();
}

- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr {
    return [NSString stringWithUTF8String:llama->bench(pp, tg, pl, nr).c_str()];
}

- (void)invalidate {
    delete llama;
    // llama_backend_free();
}

@end
