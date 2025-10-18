#import "RNLlamaContext.h"
#import <Metal/Metal.h>

@implementation RNLlamaContext

+ (BOOL)isGpuAvailable:(NSDictionary *)params {
    BOOL skipGpuDevices = params[@"no_gpu_devices"] && [params[@"no_gpu_devices"] boolValue];

    if (skipGpuDevices) {
        return false;
    }

#ifdef LM_GGML_USE_METAL
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    // Check ggml-metal availability
    BOOL supportsGgmlMetal = [device supportsFamily:MTLGPUFamilyApple7];
    if (@available(iOS 16.0, tvOS 16.0, *)) {
        supportsGgmlMetal = supportsGgmlMetal && [device supportsFamily:MTLGPUFamilyMetal3];
    }

#if TARGET_OS_SIMULATOR
    // GPU acceleration not fully supported on simulator
    device = nil;
    return false;
#else
    device = nil;
    return supportsGgmlMetal;
#endif

#else
    // Metal is not enabled in this build
    return false;
#endif
}

+ (void)toggleNativeLog:(BOOL)enabled onEmitLog:(void (^)(NSString *level, NSString *text))onEmitLog {
  if (enabled) {
      void (^copiedBlock)(NSString *, NSString *) = [onEmitLog copy];
      llama_log_set([](lm_ggml_log_level level, const char * text, void * data) {
          llama_log_callback_default(level, text, data);
          NSString *levelStr = @"";
          if (level == LM_GGML_LOG_LEVEL_ERROR) {
              levelStr = @"error";
          } else if (level == LM_GGML_LOG_LEVEL_INFO) {
              levelStr = @"info";
          } else if (level == LM_GGML_LOG_LEVEL_WARN) {
              levelStr = @"warn";
          }

          NSString *textStr = [NSString stringWithUTF8String:text];
          // NOTE: Convert to UTF-8 string may fail
          if (!textStr) {
              return;
          }
          void (^block)(NSString *, NSString *) = (__bridge void (^)(NSString *, NSString *))(data);
          block(levelStr, textStr);
      }, copiedBlock);
  } else {
      llama_log_set(llama_log_callback_default, nullptr);
  }
}

+ (NSDictionary *)modelInfo:(NSString *)path skip:(NSArray *)skip {
    struct lm_gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct lm_gguf_context * ctx = lm_gguf_init_from_file([path UTF8String], params);

    if (!ctx) {
        NSLog(@"%s: failed to load '%s'\n", __func__, [path UTF8String]);
        return @{};
    }

    NSMutableDictionary *info = [[NSMutableDictionary alloc] init];

    info[@"version"] = @(lm_gguf_get_version(ctx));
    info[@"alignment"] = @(lm_gguf_get_alignment(ctx));
    info[@"data_offset"] = @(lm_gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = lm_gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = lm_gguf_get_key(ctx, i);

            if (skip && [skip containsObject:[NSString stringWithUTF8String:key]]) {
                continue;
            }
            const std::string value = lm_gguf_kv_to_str(ctx, i);
            info[[NSString stringWithUTF8String:key]] = [NSString stringWithUTF8String:value.c_str()];
        }
    }

    lm_gguf_free(ctx);

    return info;
}

+ (NSString *)getBackendDevicesInfo {
    std::string devices_info_json = rnllama::get_backend_devices_info();
    return [NSString stringWithUTF8String:devices_info_json.c_str()];
}

+ (instancetype)initWithParams:(NSDictionary *)params onProgress:(void (^)(unsigned int progress))onProgress {
    // llama_backend_init(false);
    common_params defaultParams;

    if (params[@"vocab_only"]) {
        defaultParams.vocab_only = [params[@"vocab_only"] boolValue];
        defaultParams.warmup = false;
    }

    NSString *modelPath = params[@"model"];
    BOOL isAsset = [params[@"is_model_asset"] boolValue];
    NSString *path = modelPath;
    if (isAsset) path = [[NSBundle mainBundle] pathForResource:modelPath ofType:nil];
    defaultParams.model.path = [path UTF8String];

    NSString *chatTemplate = params[@"chat_template"];
    if (chatTemplate) {
        defaultParams.chat_template = [chatTemplate UTF8String];
        NSLog(@"chatTemplate: %@", chatTemplate);
    }

    if (params[@"n_ctx"]) defaultParams.n_ctx = [params[@"n_ctx"] intValue];
    if (params[@"use_mlock"]) defaultParams.use_mlock = [params[@"use_mlock"]boolValue];

    BOOL isGpuAvailable = [self isGpuAvailable:params];
    BOOL skipGpuDevices = !isGpuAvailable || (params[@"no_gpu_devices"] && [params[@"no_gpu_devices"] boolValue]);

    BOOL isMetalEnabled = false;
    NSString *reasonNoMetal = @"";
    NSString *gpuDeviceName = @"";
    defaultParams.n_gpu_layers = 0;

    if (isGpuAvailable) {
#if TARGET_OS_SIMULATOR
        // Use the backend, but no layers because not supported fully on simulator
        defaultParams.n_gpu_layers = 0;
        isMetalEnabled = true;
#else
        defaultParams.n_gpu_layers = [params[@"n_gpu_layers"] intValue];
        isMetalEnabled = true;
        if (!skipGpuDevices && defaultParams.n_gpu_layers > 0) {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device) {
                gpuDeviceName = device.name ?: @"";
            }
        }
#endif
    } else {
        if (params[@"no_gpu_devices"] && [params[@"no_gpu_devices"] boolValue]) {
            reasonNoMetal = @"GPU devices disabled by user";
        } else {
#ifdef LM_GGML_USE_METAL
            reasonNoMetal = @"Metal is not supported in this device";
#else
            reasonNoMetal = @"Metal is not enabled in this build";
#endif
        }
        isMetalEnabled = false;
    }

    if (skipGpuDevices) {
        std::vector<lm_ggml_backend_dev_t> cpu_devs;
        for (size_t i = 0; i < lm_ggml_backend_dev_count(); ++i) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
            switch (lm_ggml_backend_dev_type(dev)) {
                case LM_GGML_BACKEND_DEVICE_TYPE_CPU:
#ifndef TARGET_OS_SIMULATOR
                case LM_GGML_BACKEND_DEVICE_TYPE_ACCEL:
#endif
                    cpu_devs.push_back(dev);
                    break;
                case LM_GGML_BACKEND_DEVICE_TYPE_GPU:
                    break;
            }
        }
        if (cpu_devs.size() > 0) {
            defaultParams.devices = cpu_devs;
            defaultParams.n_gpu_layers = 0;
            isMetalEnabled = false;
            gpuDeviceName = @"";
        }
    }

    if (params[@"n_batch"]) defaultParams.n_batch = [params[@"n_batch"] intValue];
    if (params[@"n_ubatch"]) defaultParams.n_ubatch = [params[@"n_ubatch"] intValue];
    if (params[@"n_parallel"]) defaultParams.n_parallel = [params[@"n_parallel"] intValue];
    if (params[@"use_mmap"]) defaultParams.use_mmap = [params[@"use_mmap"] boolValue];

    if (params[@"pooling_type"] && [params[@"pooling_type"] isKindOfClass:[NSNumber class]]) {
      defaultParams.pooling_type = static_cast<enum llama_pooling_type>([params[@"pooling_type"] intValue]);
    }

    if (params[@"embedding"] && [params[@"embedding"] boolValue]) {
        defaultParams.embedding = true;
        // For non-causal models, batch size must be equal to ubatch size
        defaultParams.n_ubatch = defaultParams.n_batch;

        if (params[@"embd_normalize"] && [params[@"embd_normalize"] isKindOfClass:[NSNumber class]]) {
            defaultParams.embd_normalize = [params[@"embd_normalize"] intValue];
        }
    }

    if (params[@"rope_freq_base"]) defaultParams.rope_freq_base = [params[@"rope_freq_base"] floatValue];
    if (params[@"rope_freq_scale"]) defaultParams.rope_freq_scale = [params[@"rope_freq_scale"] floatValue];

    if (params[@"flash_attn_type"] && [params[@"flash_attn_type"] isKindOfClass:[NSString class]]) {
      const char* flash_attn_type_str = [params[@"flash_attn_type"] UTF8String];
      if (flash_attn_type_str) {
        defaultParams.flash_attn_type = static_cast<enum llama_flash_attn_type>(rnllama::flash_attn_type_from_str(flash_attn_type_str));
      }
    } else {
      // DEPRECATED: use flash_attn_type instead
      if (params[@"flash_attn"]) {
        defaultParams.flash_attn_type = [params[@"flash_attn"] boolValue] ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
      }
    }

    if (params[@"ctx_shift"]) defaultParams.ctx_shift = [params[@"ctx_shift"] boolValue];

    if (params[@"kv_unified"]) defaultParams.kv_unified = [params[@"kv_unified"] boolValue];

    if (params[@"swa_full"]) defaultParams.swa_full = [params[@"swa_full"] boolValue];

    // Handle n_cpu_moe parameter
    if (params[@"n_cpu_moe"] && [params[@"n_cpu_moe"] isKindOfClass:[NSNumber class]]) {
        int nCpuMoe = [params[@"n_cpu_moe"] intValue];
        if (nCpuMoe > 0) {
            for (int i = 0; i < nCpuMoe; ++i) {
                static std::list<std::string> buft_overrides;
                std::string pattern = "blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate)_exps";
                buft_overrides.push_back(pattern);
                defaultParams.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), lm_ggml_backend_cpu_buffer_type()});
            }
            defaultParams.tensor_buft_overrides.push_back({nullptr, nullptr});
        }
    }

    if (params[@"cache_type_k"] && [params[@"cache_type_k"] isKindOfClass:[NSString class]]) {
        const char* cache_type_k_str = [params[@"cache_type_k"] UTF8String];
        if (cache_type_k_str) {
            defaultParams.cache_type_k = rnllama::kv_cache_type_from_str(cache_type_k_str);
        }
    }
    if (params[@"cache_type_v"] && [params[@"cache_type_v"] isKindOfClass:[NSString class]]) {
        const char* cache_type_v_str = [params[@"cache_type_v"] UTF8String];
        if (cache_type_v_str) {
            defaultParams.cache_type_v = rnllama::kv_cache_type_from_str(cache_type_v_str);
        }
    }

    int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : 0;
    const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
    defaultParams.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;

    RNLlamaContext *context = [[RNLlamaContext alloc] init];
    context->llama = new rnllama::llama_rn_context();
    context->llama->is_load_interrupted = false;
    context->llama->loading_progress = 0;
    context->onProgress = onProgress;

    if (params[@"use_progress_callback"] && [params[@"use_progress_callback"] boolValue]) {
        defaultParams.progress_callback = [](float progress, void * user_data) {
            RNLlamaContext *context = (__bridge RNLlamaContext *)(user_data);
            unsigned percentage = (unsigned) (100 * progress);
            if (percentage > context->llama->loading_progress) {
                context->llama->loading_progress = percentage;
                context->onProgress(percentage);
            }
            return !context->llama->is_load_interrupted;
        };
        defaultParams.progress_callback_user_data = context;
    }

    context->is_model_loaded = context->llama->loadModel(defaultParams);

    if (
        params[@"embedding"] && [params[@"embedding"] boolValue] &&
        llama_model_has_encoder(context->llama->model) && llama_model_has_decoder(context->llama->model)
    ) {
        delete context->llama;
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not supported in encoder-decoder models" userInfo:nil];
    }

    std::vector<common_adapter_lora_info> lora;
    if (params[@"lora"]) {
        common_adapter_lora_info la;
        la.path = [params[@"lora"] UTF8String];
        la.scale = 1.0f;
        if (params[@"lora_scaled"]) la.scale = [params[@"lora_scaled"] floatValue];
        lora.push_back(la);
    }
    if (params[@"lora_list"] && [params[@"lora_list"] isKindOfClass:[NSArray class]]) {
        NSArray *lora_list = params[@"lora_list"];
        for (NSDictionary *lora_adapter in lora_list) {
          NSString *path = lora_adapter[@"path"];
          if (!path) continue;
          float scale = [lora_adapter[@"scaled"] floatValue];
          common_adapter_lora_info la;
          la.path = [path UTF8String];
          la.scale = scale;
          lora.push_back(la);
        }
    }
    if (lora.size() > 0) {
        int result = context->llama->applyLoraAdapters(lora);
        if (result != 0) {
            delete context->llama;
            @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapters" userInfo:nil];
        }
    }

    context->is_metal_enabled = isMetalEnabled;
    context->reason_no_metal = reasonNoMetal;
    context->gpu_device_name = gpuDeviceName ?: @"";

    return context;
}

- (void)interruptLoad {
    llama->is_load_interrupted = true;
}

- (bool)isMetalEnabled {
    return is_metal_enabled;
}

- (NSString *)reasonNoMetal {
    return reason_no_metal;
}

- (NSString *)gpuDeviceName {
    return gpu_device_name;
}

- (NSDictionary *)modelInfo {
    char desc[1024];
    llama_model_desc(llama->model, desc, sizeof(desc));

    int count = llama_model_meta_count(llama->model);
    NSDictionary *meta = [[NSMutableDictionary alloc] init];
    for (int i = 0; i < count; i++) {
        char key[256];
        llama_model_meta_key_by_index(llama->model, i, key, sizeof(key));
        char val[16384];  // gpt-oss's chat template is 12kb
        llama_model_meta_val_str_by_index(llama->model, i, val, sizeof(val));

        NSString *keyStr = [NSString stringWithUTF8String:key];
        NSString *valStr = [NSString stringWithUTF8String:val];
        [meta setValue:valStr forKey:keyStr];
    }

    auto template_tool_use = llama->templates.get()->template_tool_use.get();
    NSDictionary *tool_use_caps_dir = nil;
    if (template_tool_use) {
        auto tool_use_caps = template_tool_use->original_caps();
        tool_use_caps_dir = @{
            @"tools": @(tool_use_caps.supports_tools),
            @"toolCalls": @(tool_use_caps.supports_tool_calls),
            @"toolResponses": @(tool_use_caps.supports_tool_responses),
            @"systemRole": @(tool_use_caps.supports_system_role),
            @"parallelToolCalls": @(tool_use_caps.supports_parallel_tool_calls),
            @"toolCallId": @(tool_use_caps.supports_tool_call_id)
        };
    }

    auto default_tmpl = llama->templates.get()->template_default.get();
    auto default_tmpl_caps = default_tmpl->original_caps();

    return @{
        @"desc": [NSString stringWithUTF8String:desc],
        @"size": @(llama_model_size(llama->model)),
        @"nEmbd": @(llama_model_n_embd(llama->model)),
        @"nParams": @(llama_model_n_params(llama->model)),
        @"chatTemplates": @{
            @"llamaChat": @(llama->validateModelChatTemplate(false, nullptr)),
            @"minja": @{
                @"default": @(llama->validateModelChatTemplate(true, nullptr)),
                @"defaultCaps": @{
                    @"tools": @(default_tmpl_caps.supports_tools),
                    @"toolCalls": @(default_tmpl_caps.supports_tool_calls),
                    @"toolResponses": @(default_tmpl_caps.supports_tool_responses),
                    @"systemRole": @(default_tmpl_caps.supports_system_role),
                    @"parallelToolCalls": @(default_tmpl_caps.supports_parallel_tool_calls),
                    @"toolCallId": @(default_tmpl_caps.supports_tool_call_id)
                },
                @"toolUse": @(llama->validateModelChatTemplate(true, "tool_use")),
                @"toolUseCaps": tool_use_caps_dir ?: @{}
            }
        },
        @"metadata": meta,

        // deprecated
        @"isChatTemplateSupported": @(llama->validateModelChatTemplate(false, nullptr))
    };
}

- (bool)isModelLoaded {
    return is_model_loaded;
}

- (bool)isPredicting {
    if (llama->completion == nullptr) {
        return false;
    }
    return llama->completion->is_predicting;
}

- (bool)initMultimodal:(NSDictionary *)params {
    NSString *mmproj_path = params[@"path"];
    BOOL use_gpu = params[@"use_gpu"] ? [params[@"use_gpu"] boolValue] : true;

    // Check GPU availability using utility function
    if (use_gpu) {
        use_gpu = [RNLlamaContext isGpuAvailable:params];
    }

    return llama->initMultimodal([mmproj_path UTF8String], use_gpu);
}

- (NSDictionary *)getMultimodalSupport {
    if (!is_model_loaded) return nil;
    return @{
        @"vision": @(llama->isMultimodalSupportVision()),
        @"audio": @(llama->isMultimodalSupportAudio())
    };
}

- (bool)isMultimodalEnabled {
    if (!is_model_loaded) return false;
    return llama->isMultimodalEnabled();
}

- (void)releaseMultimodal {
    if (!is_model_loaded) return;
    llama->releaseMultimodal();
}

- (NSDictionary *)getFormattedChatWithJinja:(NSString *)messages
    withChatTemplate:(NSString *)chatTemplate
    withJsonSchema:(NSString *)jsonSchema
    withTools:(NSString *)tools
    withParallelToolCalls:(BOOL)parallelToolCalls
    withToolChoice:(NSString *)toolChoice
    withEnableThinking:(BOOL)enableThinking
    withAddGenerationPrompt:(BOOL)addGenerationPrompt
    withNow:(NSString *)nowStr
    withChatTemplateKwargs:(NSString *)chatTemplateKwargs
{
    auto tmpl_str = chatTemplate == nil ? "" : [chatTemplate UTF8String];

    // Parse chat template kwargs
    std::map<std::string, std::string> kwargs_map;
    if (chatTemplateKwargs != nil && [chatTemplateKwargs length] > 0) {
        try {
            auto kwargs_json = json::parse([chatTemplateKwargs UTF8String]);
            for (auto& [key, value] : kwargs_json.items()) {
                if (value.is_string()) {
                    kwargs_map[key] = value.get<std::string>();
                }
            }
        } catch (...) {
            // Ignore JSON parsing errors for kwargs
        }
    }

    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    auto chatParams = llama->getFormattedChatWithJinja(
        [messages UTF8String],
        tmpl_str,
        jsonSchema == nil ? "" : [jsonSchema UTF8String],
        tools == nil ? "" : [tools UTF8String],
        parallelToolCalls,
        toolChoice == nil ? "" : [toolChoice UTF8String],
        enableThinking,
        addGenerationPrompt,
        nowStr == nil ? "" : [nowStr UTF8String],
        kwargs_map
    );
    result[@"prompt"] = [NSString stringWithUTF8String:chatParams.prompt.c_str()];
    result[@"chat_format"] = @(static_cast<int>(chatParams.format));
    result[@"grammar"] = [NSString stringWithUTF8String:chatParams.grammar.c_str()];
    result[@"grammar_lazy"] = @(chatParams.grammar_lazy);
    NSMutableArray *grammar_triggers = [[NSMutableArray alloc] init];
    for (const auto & trigger : chatParams.grammar_triggers) {
        [grammar_triggers addObject:@{
            @"type": @(trigger.type),
            @"value": [NSString stringWithUTF8String:trigger.value.c_str()],
            @"token": @(trigger.token),
        }];
    }
    result[@"thinking_forced_open"] = @(chatParams.thinking_forced_open);
    result[@"grammar_triggers"] = grammar_triggers;
    NSMutableArray *preserved_tokens = [[NSMutableArray alloc] init];
    for (const auto & token : chatParams.preserved_tokens) {
        [preserved_tokens addObject:[NSString stringWithUTF8String:token.c_str()]];
    }
    result[@"preserved_tokens"] = preserved_tokens;
    NSMutableArray *additional_stops = [[NSMutableArray alloc] init];
    for (const auto & stop : chatParams.additional_stops) {
        [additional_stops addObject:[NSString stringWithUTF8String:stop.c_str()]];
    }
    result[@"additional_stops"] = additional_stops;

    return result;
}

- (NSString *)getFormattedChat:(NSString *)messages withChatTemplate:(NSString *)chatTemplate {
    auto tmpl_str = chatTemplate == nil ? "" : [chatTemplate UTF8String];
    return [NSString stringWithUTF8String:llama->getFormattedChat(
        [messages UTF8String],
        tmpl_str
    ).c_str()];;
}

- (NSArray *)tokenProbsToDict:(std::vector<rnllama::completion_token_output>)probs {
    NSMutableArray *out = [[NSMutableArray alloc] init];
    for (const auto &prob : probs)
    {
        NSMutableArray *probsForToken = [[NSMutableArray alloc] init];
        for (const auto &p : prob.probs)
        {
            std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, p.tok);
            NSString *tokStrString = [NSString stringWithUTF8String:tokStr.c_str()];
            if (tokStrString == nil) tokStrString = @"<UNKNOWN>";
            [probsForToken addObject:@{
                @"tok_str": tokStrString,
                @"prob": [NSNumber numberWithDouble:p.prob]
            }];
        }
        std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, prob.tok);
        NSString *tokStrString = [NSString stringWithUTF8String:tokStr.c_str()];
        if (tokStrString == nil) tokStrString = @"<UNKNOWN>";
        [out addObject:@{
            @"content": tokStrString,
            @"probs": probsForToken
        }];
    }
    return out;
}

- (NSDictionary *)completion:(NSDictionary *)params
    onToken:(void (^)(NSMutableDictionary * tokenResult))onToken
{
    if (llama->completion == nullptr) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Completion not initialized" userInfo:nil];
    }
    llama->completion->rewind();

    //llama_reset_timings(llama->ctx);

    NSString *prompt = [params objectForKey:@"prompt"];

    llama->params.prompt = [prompt UTF8String];
    llama->params.sampling.seed = params[@"seed"] ? [params[@"seed"] intValue] : -1;

    if (params[@"n_threads"]) {
        int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : llama->params.cpuparams.n_threads;
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        // Use 2 threads by default on 4-core devices, 4 threads on more cores
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        llama->params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) llama->params.n_predict = [params[@"n_predict"] intValue];
    if (params[@"ignore_eos"]) llama->params.sampling.ignore_eos = [params[@"ignore_eos"] boolValue];

    auto & sparams = llama->params.sampling;

    if (params[@"temperature"]) sparams.temp = [params[@"temperature"] doubleValue];

    if (params[@"n_probs"]) sparams.n_probs = [params[@"n_probs"] intValue];

    if (params[@"penalty_last_n"]) sparams.penalty_last_n = [params[@"penalty_last_n"] intValue];
    if (params[@"penalty_repeat"]) sparams.penalty_repeat = [params[@"penalty_repeat"] doubleValue];
    if (params[@"penalty_freq"]) sparams.penalty_freq = [params[@"penalty_freq"] doubleValue];
    if (params[@"penalty_present"]) sparams.penalty_present = [params[@"penalty_present"] doubleValue];

    if (params[@"mirostat"]) sparams.mirostat = [params[@"mirostat"] intValue];
    if (params[@"mirostat_tau"]) sparams.mirostat_tau = [params[@"mirostat_tau"] doubleValue];
    if (params[@"mirostat_eta"]) sparams.mirostat_eta = [params[@"mirostat_eta"] doubleValue];

    if (params[@"top_k"]) sparams.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) sparams.top_p = [params[@"top_p"] doubleValue];
    if (params[@"min_p"]) sparams.min_p = [params[@"min_p"] doubleValue];
    if (params[@"xtc_threshold"]) sparams.xtc_threshold = [params[@"xtc_threshold"] doubleValue];
    if (params[@"xtc_probability"]) sparams.xtc_probability = [params[@"xtc_probability"] doubleValue];
    if (params[@"typical_p"]) sparams.typ_p = [params[@"typical_p"] doubleValue];

    if (params[@"dry_multiplier"]) sparams.dry_multiplier = [params[@"dry_multiplier"] doubleValue];
    if (params[@"dry_base"]) sparams.dry_base = [params[@"dry_base"] doubleValue];
    if (params[@"dry_allowed_length"]) sparams.dry_allowed_length = [params[@"dry_allowed_length"] intValue];
    if (params[@"dry_penalty_last_n"]) sparams.dry_penalty_last_n = [params[@"dry_penalty_last_n"] intValue];

    if (params[@"top_n_sigma"]) sparams.top_n_sigma = [params[@"top_n_sigma"] doubleValue];

    // dry break seq
    if (params[@"dry_sequence_breakers"] && [params[@"dry_sequence_breakers"] isKindOfClass:[NSArray class]]) {
        NSArray *dry_sequence_breakers = params[@"dry_sequence_breakers"];
        for (NSString *s in dry_sequence_breakers) {
            sparams.dry_sequence_breakers.push_back([s UTF8String]);
        }
    }

    if (params[@"grammar"]) {
        sparams.grammar = [params[@"grammar"] UTF8String];
    }

    if (params[@"json_schema"] && !params[@"grammar"]) {
        sparams.grammar = json_schema_to_grammar(json::parse([params[@"json_schema"] UTF8String]));
    }

    if (params[@"grammar_lazy"]) {
        sparams.grammar_lazy = [params[@"grammar_lazy"] boolValue];
    }

    if (params[@"preserved_tokens"] && [params[@"preserved_tokens"] isKindOfClass:[NSArray class]]) {
        NSArray *preserved_tokens = params[@"preserved_tokens"];
        for (NSString *token in preserved_tokens) {
            auto ids = common_tokenize(llama->ctx, [token UTF8String], /* add_special= */ false, /* parse_special= */ true);
            if (ids.size() == 1) {
                sparams.preserved_tokens.insert(ids[0]);
            } else {
//                LOG_WRN("Not preserved because more than 1 token (wrong chat template override?): %s\n", [token UTF8String]);
            }
        }
    }

    if (params[@"grammar_triggers"] && [params[@"grammar_triggers"] isKindOfClass:[NSArray class]]) {
        NSArray *grammar_triggers = params[@"grammar_triggers"];
        for (NSDictionary *grammar_trigger in grammar_triggers) {
            const auto type = static_cast<common_grammar_trigger_type>([grammar_trigger[@"type"] intValue]);
            const auto & word = [grammar_trigger[@"value"] UTF8String];

            if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
              auto ids = common_tokenize(llama->ctx, word, /* add_special= */ false, /* parse_special= */ true);
              if (ids.size() == 1) {
                  auto token = ids[0];
                  if (std::find(sparams.preserved_tokens.begin(), sparams.preserved_tokens.end(), (llama_token) token) == sparams.preserved_tokens.end()) {
                      throw std::runtime_error("Grammar trigger word should be marked as preserved token");
                  }
                  common_grammar_trigger trigger;
                  trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                  trigger.value = word;
                  trigger.token = token;
                  sparams.grammar_triggers.push_back(std::move(trigger));
              } else {
                  sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
              }
            } else {
                common_grammar_trigger trigger;
                trigger.type = type;
                trigger.value = word;
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    const auto token = (llama_token) [grammar_trigger[@"token"] intValue];
                    trigger.token = token;
                }
                sparams.grammar_triggers.push_back(std::move(trigger));
            }
        }
    }

    llama->params.antiprompt.clear();
    if (params[@"stop"]) {
        NSArray *stop = params[@"stop"];
        for (NSString *s in stop) {
            llama->params.antiprompt.push_back([s UTF8String]);
        }
    }

    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    sparams.logit_bias.clear();
    if (params[@"ignore_eos"] && [params[@"ignore_eos"] boolValue]) {
        sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
    }

    if (params[@"logit_bias"] && [params[@"logit_bias"] isKindOfClass:[NSArray class]]) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        NSArray *logit_bias = params[@"logit_bias"];
        for (NSArray *el in logit_bias) {
            if ([el isKindOfClass:[NSArray class]] && [el count] == 2) {
                llama_token tok = [el[0] intValue];
                if (tok >= 0 && tok < n_vocab) {
                    if ([el[1] isKindOfClass:[NSNumber class]]) {
                        sparams.logit_bias[tok].bias = [el[1] doubleValue];
                    } else if ([el[1] isKindOfClass:[NSNumber class]] && ![el[1] boolValue]) {
                        sparams.logit_bias[tok].bias = -INFINITY;
                    }
                }
            }
        }
    }

    if (params[@"guide_tokens"] && [params[@"guide_tokens"] isKindOfClass:[NSArray class]]) {
        NSArray *guide_tokens_array = params[@"guide_tokens"];
        std::vector<llama_token> guide_tokens;
        guide_tokens.reserve([guide_tokens_array count]);
        for (NSNumber *token_num in guide_tokens_array) {
            guide_tokens.push_back([token_num intValue]);
        }
        if (llama->tts_wrapper != nullptr) {
            llama->tts_wrapper->setGuideTokens(guide_tokens);
        }
    }

    if (!llama->completion->initSampling()) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to initialize sampling" userInfo:nil];
    }

    NSString *prefillText = params[@"prefill_text"];
    if (prefillText) {
        llama->completion->prefill_text = [prefillText UTF8String];
    }

    auto chat_format = params[@"chat_format"] ? [params[@"chat_format"] intValue] : COMMON_CHAT_FORMAT_CONTENT_ONLY;
    bool thinking_forced_open = [params[@"thinking_forced_open"] boolValue];

    NSString *reasoningFormat = params[@"reasoning_format"];
    if (!reasoningFormat) reasoningFormat = @"none";
    std::string reasoningFormatStr = [reasoningFormat UTF8String];
    common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);

    llama->completion->beginCompletion(chat_format, reasoning_format, thinking_forced_open);
    try {
        // Use the unified loadPrompt function with media paths if available
        NSArray *mediaPaths = params[@"media_paths"];
        if (mediaPaths && [mediaPaths count] > 0) {
            // Multiple media paths
            std::vector<std::string> media_paths_vector;
            for (NSString *path in mediaPaths) {
                if ([path isKindOfClass:[NSString class]]) {
                    media_paths_vector.push_back([path UTF8String]);
                }
            }
            llama->completion->loadPrompt(media_paths_vector);
        } else {
            llama->completion->loadPrompt({});
        }
    } catch (const std::exception &e) {
        llama->completion->endCompletion();
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    } catch (const std::runtime_error& e) {
        llama->completion->endCompletion();
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }

    if (llama->completion->context_full) {
        llama->completion->endCompletion();
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context is full" userInfo:nil];
    }

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->completion->has_next_token && !llama->completion->is_interrupted) {
        const rnllama::completion_token_output token_with_probs = llama->completion->doCompletion();
        if (token_with_probs.tok == -1 || llama->completion->incomplete) {
            continue;
        }
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->completion->generated_text.size());

        const std::string str_test = llama->completion->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->completion->generated_text.erase(
                llama->completion->generated_text.begin() + pos + stop_pos,
                llama->completion->generated_text.end());
            pos = std::min(sent_count, llama->completion->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->completion->findStoppingStrings(str_test, token_text.size(),
                rnllama::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->completion->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->completion->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<rnllama::completion_token_output> probs_output = {};

            NSMutableDictionary *tokenResult = [[NSMutableDictionary alloc] init];
            tokenResult[@"token"] = [NSString stringWithUTF8String:to_send.c_str()];

            if (llama->params.sampling.n_probs > 0) {
                const std::vector<llama_token> to_send_toks = common_tokenize(llama->ctx, to_send, false);
                size_t probs_pos = std::min(sent_token_probs_index, llama->completion->generated_token_probs.size());
                size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->completion->generated_token_probs.size());
                if (probs_pos < probs_stop_pos) {
                    probs_output = std::vector<rnllama::completion_token_output>(llama->completion->generated_token_probs.begin() + probs_pos, llama->completion->generated_token_probs.begin() + probs_stop_pos);
                }
                sent_token_probs_index = probs_stop_pos;

                tokenResult[@"completion_probabilities"] = [self tokenProbsToDict:probs_output];
            }

            auto partial_output = llama->completion->parseChatOutput(true);
            if (!partial_output.content.empty()) {
                tokenResult[@"content"] = [NSString stringWithUTF8String:partial_output.content.c_str()];
            }
            if (!partial_output.reasoning_content.empty()) {
                tokenResult[@"reasoning_content"] = [NSString stringWithUTF8String:partial_output.reasoning_content.c_str()];
            }
            if (!partial_output.tool_calls.empty()) {
                NSMutableArray *toolCalls = [[NSMutableArray alloc] init];
                for (const auto &tc : partial_output.tool_calls) {
                    [toolCalls addObject:@{
                        @"type": @"function",
                        @"function": @{
                            @"name": [NSString stringWithUTF8String:tc.name.c_str()],
                            @"arguments": [NSString stringWithUTF8String:tc.arguments.c_str()],
                        },
                        @"id": tc.id.empty() ? [NSNull null] : [NSString stringWithUTF8String:tc.id.c_str()],
                    }];
                }
                tokenResult[@"tool_calls"] = toolCalls;
            }
            if (!partial_output.accumulated_text.empty()) {
                tokenResult[@"accumulated_text"] = [NSString stringWithUTF8String:partial_output.accumulated_text.c_str()];
            }

            onToken(tokenResult);
        }
    }

    llama_perf_context_print(llama->ctx);
    llama->completion->endCompletion();

    const auto timings = llama_perf_context(llama->ctx);

    NSMutableArray *toolCalls = nil;
    NSString *reasoningContent = nil;
    NSString *content = nil;
    NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
    result[@"chat_format"] = @(chat_format);

    if (!llama->completion->is_interrupted) {
        try {
            auto final_output = llama->completion->parseChatOutput(false);
            if (!final_output.reasoning_content.empty()) {
                reasoningContent = [NSString stringWithUTF8String:final_output.reasoning_content.c_str()];
            }
            content = [NSString stringWithUTF8String:final_output.content.c_str()];
            toolCalls = [[NSMutableArray alloc] init];
            for (const auto &tc : final_output.tool_calls) {
                [toolCalls addObject:@{
                    @"type": @"function",
                    @"function": @{
                        @"name": [NSString stringWithUTF8String:tc.name.c_str()],
                        @"arguments": [NSString stringWithUTF8String:tc.arguments.c_str()],
                    },
                    @"id": tc.id.empty() ? [NSNull null] : [NSString stringWithUTF8String:tc.id.c_str()],
                }];
            }
        } catch (const std::exception &e) {
        } catch (...) {
        }
    }

    result[@"text"] = [NSString stringWithUTF8String:llama->completion->generated_text.c_str()]; // Original text
    if (content) result[@"content"] = content;
    if (reasoningContent) result[@"reasoning_content"] = reasoningContent;
    if (toolCalls && toolCalls.count > 0) result[@"tool_calls"] = toolCalls;
    result[@"completion_probabilities"] = [self tokenProbsToDict:llama->completion->generated_token_probs];
    result[@"tokens_predicted"] = @(llama->completion->num_tokens_predicted);
    result[@"tokens_evaluated"] = @(llama->completion->num_prompt_tokens);
    result[@"truncated"] = @(llama->completion->truncated);
    result[@"context_full"] = @(llama->completion->context_full);
    result[@"interrupted"] = @(llama->completion->is_interrupted);
    result[@"stopped_eos"] = @(llama->completion->stopped_eos);
    result[@"stopped_word"] = @(llama->completion->stopped_word);
    result[@"stopped_limit"] = @(llama->completion->stopped_limit);
    result[@"stopping_word"] = [NSString stringWithUTF8String:llama->completion->stopping_word.c_str()];
    result[@"tokens_cached"] = @(llama->completion->n_past);

    if (llama->isVocoderEnabled() && llama->tts_wrapper != nullptr && !llama->tts_wrapper->audio_tokens.empty()) {
        NSMutableArray *audioTokens = [[NSMutableArray alloc] init];
        for (llama_token token : llama->tts_wrapper->audio_tokens) {
            [audioTokens addObject:@(token)];
        }
        result[@"audio_tokens"] = audioTokens;
    }

    result[@"timings"] = @{
        @"prompt_n": @(timings.n_p_eval),
        @"prompt_ms": @(timings.t_p_eval_ms),
        @"prompt_per_token_ms": @(timings.t_p_eval_ms / timings.n_p_eval),
        @"prompt_per_second": @(1e3 / timings.t_p_eval_ms * timings.n_p_eval),
        @"predicted_n": @(timings.n_eval),
        @"predicted_n": @(timings.n_eval),
        @"predicted_ms": @(timings.t_eval_ms),
        @"predicted_per_token_ms": @(timings.t_eval_ms / timings.n_eval),
        @"predicted_per_second": @(1e3 / timings.t_eval_ms * timings.n_eval),
    };
    return result;
}

- (void)stopCompletion {
    if (llama->completion == nullptr) {
        return;
    }
    llama->completion->is_interrupted = true;
}

- (NSDictionary *)tokenize:(NSString *)text mediaPaths:(NSArray *)mediaPaths {
    std::vector<std::string> media_paths_vector;
    if (mediaPaths && [mediaPaths count] > 0) {
        for (NSString *path in mediaPaths) {
            if ([path isKindOfClass:[NSString class]]) {
                media_paths_vector.push_back([path UTF8String]);
            }
        }
    }
    try {
        rnllama::llama_rn_tokenize_result tokenize_result = llama->tokenize([text UTF8String], media_paths_vector);

        NSMutableDictionary *result = [[NSMutableDictionary alloc] init];

        result[@"tokens"] = [NSMutableArray arrayWithCapacity:tokenize_result.tokens.size()];
        for (llama_token tok : tokenize_result.tokens) {
            [result[@"tokens"] addObject:@(tok)];
        }
        result[@"has_media"] = @(tokenize_result.has_media);

        NSMutableArray *bitmap_hashes = [[NSMutableArray alloc] init];
        for (std::string hash : tokenize_result.bitmap_hashes) {
            [bitmap_hashes addObject:[NSString stringWithUTF8String:hash.c_str()]];
        }
        result[@"bitmap_hashes"] = bitmap_hashes;

        NSMutableArray *chunk_pos = [[NSMutableArray alloc] init];
        for (int pos : tokenize_result.chunk_pos) {
            [chunk_pos addObject:@(pos)];
        }
        result[@"chunk_pos"] = chunk_pos;

        NSMutableArray *chunk_pos_media = [[NSMutableArray alloc] init];
        for (int pos : tokenize_result.chunk_pos_media) {
            [chunk_pos_media addObject:@(pos)];
        }
        result[@"chunk_pos_media"] = chunk_pos_media;

        return result;
    } catch (const std::exception &e) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    } catch (const std::runtime_error& e) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }
}

- (NSString *)detokenize:(NSArray *)tokens {
    std::vector<llama_token> toks;
    for (NSNumber *tok in tokens) {
        toks.push_back([tok intValue]);
    }
    const std::string text = rnllama::tokens_to_str(llama->ctx, toks.cbegin(), toks.cend());
    return [NSString stringWithUTF8String:text.c_str()];
}

- (NSDictionary *)embedding:(NSString *)text params:(NSDictionary *)params {
    if (llama->completion == nullptr) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context has been released" userInfo:nil];
    }
    if (llama->params.embedding != true) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not enabled" userInfo:nil];
    }
    if ([self isPredicting]) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context is predicting" userInfo:nil];
    }

    common_params embdParams;
    embdParams.embedding = true;
    embdParams.embd_normalize = llama->params.embd_normalize;

    if (params[@"embd_normalize"] && [params[@"embd_normalize"] isKindOfClass:[NSNumber class]]) {
        embdParams.embd_normalize = [params[@"embd_normalize"] intValue];
    }

    llama->params.prompt = [text UTF8String];
    llama->params.n_predict = 0;
    try {
        std::vector<float> result = llama->completion->embedding(embdParams);

        NSMutableDictionary *resultDict = [[NSMutableDictionary alloc] init];
        NSMutableArray *embeddingResult = [[NSMutableArray alloc] init];
        for (float f : result) {
            [embeddingResult addObject:@(f)];
        }
        resultDict[@"embedding"] = embeddingResult;
        NSMutableArray *promptTokens = [[NSMutableArray alloc] init];
        for (llama_token tok : llama->completion->embd) {
            [promptTokens addObject:[NSString stringWithUTF8String:common_token_to_piece(llama->ctx, tok).c_str()]];
        }
        resultDict[@"prompt_tokens"] = promptTokens;
        return resultDict;
    } catch (const std::exception &e) {
        llama->completion->endCompletion();
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    } catch (const std::runtime_error& e) {
        llama->completion->endCompletion();
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }
}

- (NSArray *)rerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params {
    if (llama->completion == nullptr) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context has been released" userInfo:nil];
    }
    if (llama->params.embedding != true) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Embedding is not enabled" userInfo:nil];
    }
    if ([self isPredicting]) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context is predicting" userInfo:nil];
    }

    // Convert NSArray to std::vector
    std::vector<std::string> documentsVector;
    for (NSString *doc in documents) {
        documentsVector.push_back(std::string([doc UTF8String]));
    }

    NSMutableArray *resultArray = [[NSMutableArray alloc] init];

    try {
        std::vector<float> scores = llama->completion->rerank(std::string([query UTF8String]), documentsVector);

        // Create result array with score and index
        for (size_t i = 0; i < scores.size(); i++) {
            NSMutableDictionary *item = [[NSMutableDictionary alloc] init];
            item[@"score"] = @(scores[i]);
            item[@"index"] = @((int)i);
            [resultArray addObject:item];
        }
    } catch (const std::exception &e) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    } catch (const std::runtime_error& e) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }

    return resultArray;
}

- (NSDictionary *)loadSession:(NSString *)path {
    if (llama->completion == nullptr) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context has been released" userInfo:nil];
    }
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session file does not exist" userInfo:nil];
    }

    size_t n_token_count_out = 0;
    llama->completion->embd.resize(llama->params.n_ctx);
    if (!llama_state_load_file(llama->ctx, [path UTF8String], llama->completion->embd.data(), llama->completion->embd.capacity(), &n_token_count_out)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to load session" userInfo:nil];
    }
    llama->completion->embd.resize(n_token_count_out);
    // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
    auto null_token_iter = std::find(llama->completion->embd.begin(), llama->completion->embd.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != llama->completion->embd.end()) {
        llama->completion->embd.resize(std::distance(llama->completion->embd.begin(), null_token_iter));
    }
    const std::string text = rnllama::tokens_to_str(llama->ctx, llama->completion->embd.cbegin(), llama->completion->embd.cend());
    return @{
        @"tokens_loaded": @(n_token_count_out),
        @"prompt": [NSString stringWithUTF8String:text.c_str()]
    };
}

- (int)saveSession:(NSString *)path size:(int)size {
    if (llama->completion == nullptr) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Context has been released" userInfo:nil];
    }
    if (!path || [path length] == 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Session path is empty" userInfo:nil];
    }
    std::vector<llama_token> session_tokens = llama->completion->embd;
    // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
    auto null_token_iter = std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != session_tokens.end()) {
        session_tokens.resize(std::distance(session_tokens.begin(), null_token_iter));
    }
    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_state_save_file(llama->ctx, [path UTF8String], session_tokens.data(), save_size)) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to save session" userInfo:nil];
    }
    return session_tokens.size();
}

- (NSString *)bench:(int)pp tg:(int)tg pl:(int)pl nr:(int)nr {
    if (llama->completion == nullptr) {
        return @"";
    }
    return [NSString stringWithUTF8String:llama->completion->bench(pp, tg, pl, nr).c_str()];
}

- (void)applyLoraAdapters:(NSArray *)loraAdapters {
    std::vector<common_adapter_lora_info> lora_adapters;
    for (NSDictionary *loraAdapter in loraAdapters) {
        common_adapter_lora_info la;
        la.path = [loraAdapter[@"path"] UTF8String];
        la.scale = [loraAdapter[@"scaled"] doubleValue];
        la.ptr = llama_adapter_lora_init(llama->model, la.path.c_str());
        if (la.ptr == nullptr) {
            @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapter" userInfo:nil];
        }
        lora_adapters.push_back(la);
    }
    int result = llama->applyLoraAdapters(lora_adapters);
    if (result != 0) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:@"Failed to apply lora adapters" userInfo:nil];
    }
}

- (void)removeLoraAdapters {
    llama->removeLoraAdapters();
}

- (NSArray *)getLoadedLoraAdapters {
    std::vector<common_adapter_lora_info> loaded_lora_adapters = llama->getLoadedLoraAdapters();
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (common_adapter_lora_info &la : loaded_lora_adapters) {
        [result addObject:@{
            @"path": [NSString stringWithUTF8String:la.path.c_str()],
            @"scale": @(la.scale)
        }];
    }
    return result;
}

- (bool)initVocoder:(NSDictionary *)params {
    int n_batch = params[@"n_batch"] ? [params[@"n_batch"] intValue] : 512;
    return llama->initVocoder([params[@"path"] UTF8String], n_batch);
}

- (bool)isVocoderEnabled {
    return llama->isVocoderEnabled();
}

- (NSDictionary *)getFormattedAudioCompletion:(NSString *)speakerJsonStr textToSpeak:(NSString *)textToSpeak {
    std::string speakerStr = speakerJsonStr ? [speakerJsonStr UTF8String] : "";
    try {
        auto audio_result = llama->tts_wrapper->getFormattedAudioCompletion(llama, speakerStr, [textToSpeak UTF8String]);
        return @{
            @"prompt": [NSString stringWithUTF8String:audio_result.prompt.c_str()],
            @"grammar": [NSString stringWithUTF8String:audio_result.grammar]
        };
    } catch (const std::exception &e) {
        @throw [NSException exceptionWithName:@"LlamaException" reason:[NSString stringWithUTF8String:e.what()] userInfo:nil];
    }
}

- (NSArray *)getAudioCompletionGuideTokens:(NSString *)textToSpeak {
    std::vector<llama_token> guide_tokens = llama->tts_wrapper->getAudioCompletionGuideTokens(llama, [textToSpeak UTF8String]);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (llama_token token : guide_tokens) {
        [result addObject:@(token)];
    }
    return result;
}

- (NSArray *)decodeAudioTokens:(NSArray *)tokens {
    std::vector<llama_token> token_vector;
    for (NSNumber *token in tokens) {
        token_vector.push_back([token intValue]);
    }
    std::vector<float> audio_data = llama->tts_wrapper->decodeAudioTokens(llama, token_vector);
    NSMutableArray *result = [[NSMutableArray alloc] init];
    for (float sample : audio_data) {
        [result addObject:@(sample)];
    }
    return result;
}

- (void)releaseVocoder {
    llama->releaseVocoder();
}

// Parallel decoding: Queue a completion request
- (NSNumber *)queueCompletion:(NSDictionary *)params onToken:(void (^)(NSMutableDictionary *))onToken onComplete:(void (^)(NSDictionary *))onComplete {
    if (!is_model_loaded) {
        return @(-1);
    }

    // Check if parallel mode is enabled
    if (!llama->parallel_mode_enabled) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:@"Parallel mode is not enabled. Call enableParallelMode() first."
                                     userInfo:nil];
    }

    __block int requestId = -1;

    // Tokenize prompt
    NSString *prompt = params[@"prompt"];
    NSArray *mediaPaths = params[@"media_paths"];
    rnllama::llama_rn_tokenize_result tokenize_result = llama->tokenize(
        prompt ? [prompt UTF8String] : "",
        mediaPaths ? [self convertNSArrayToStdVector:mediaPaths] : std::vector<std::string>()
    );

    // Convert params to common_params (match completion method parameter handling)
    common_params cpp_params = llama->params;

    // Sampling parameters
    cpp_params.sampling.seed = params[@"seed"] ? [params[@"seed"] intValue] : -1;

    if (params[@"n_threads"]) {
        int nThreads = params[@"n_threads"] ? [params[@"n_threads"] intValue] : cpp_params.cpuparams.n_threads;
        const int maxThreads = (int) [[NSProcessInfo processInfo] processorCount];
        const int defaultNThreads = nThreads == 4 ? 2 : MIN(4, maxThreads);
        cpp_params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
    }
    if (params[@"n_predict"]) cpp_params.n_predict = [params[@"n_predict"] intValue];
    if (params[@"ignore_eos"]) cpp_params.sampling.ignore_eos = [params[@"ignore_eos"] boolValue];

    auto & sparams = cpp_params.sampling;

    if (params[@"temperature"]) sparams.temp = [params[@"temperature"] doubleValue];
    if (params[@"n_probs"]) sparams.n_probs = [params[@"n_probs"] intValue];

    if (params[@"penalty_last_n"]) sparams.penalty_last_n = [params[@"penalty_last_n"] intValue];
    if (params[@"penalty_repeat"]) sparams.penalty_repeat = [params[@"penalty_repeat"] doubleValue];
    if (params[@"penalty_freq"]) sparams.penalty_freq = [params[@"penalty_freq"] doubleValue];
    if (params[@"penalty_present"]) sparams.penalty_present = [params[@"penalty_present"] doubleValue];

    if (params[@"mirostat"]) sparams.mirostat = [params[@"mirostat"] intValue];
    if (params[@"mirostat_tau"]) sparams.mirostat_tau = [params[@"mirostat_tau"] doubleValue];
    if (params[@"mirostat_eta"]) sparams.mirostat_eta = [params[@"mirostat_eta"] doubleValue];

    if (params[@"top_k"]) sparams.top_k = [params[@"top_k"] intValue];
    if (params[@"top_p"]) sparams.top_p = [params[@"top_p"] doubleValue];
    if (params[@"min_p"]) sparams.min_p = [params[@"min_p"] doubleValue];
    if (params[@"xtc_threshold"]) sparams.xtc_threshold = [params[@"xtc_threshold"] doubleValue];
    if (params[@"xtc_probability"]) sparams.xtc_probability = [params[@"xtc_probability"] doubleValue];
    if (params[@"typical_p"]) sparams.typ_p = [params[@"typical_p"] doubleValue];

    if (params[@"dry_multiplier"]) sparams.dry_multiplier = [params[@"dry_multiplier"] doubleValue];
    if (params[@"dry_base"]) sparams.dry_base = [params[@"dry_base"] doubleValue];
    if (params[@"dry_allowed_length"]) sparams.dry_allowed_length = [params[@"dry_allowed_length"] intValue];
    if (params[@"dry_penalty_last_n"]) sparams.dry_penalty_last_n = [params[@"dry_penalty_last_n"] intValue];

    if (params[@"top_n_sigma"]) sparams.top_n_sigma = [params[@"top_n_sigma"] doubleValue];

    // Dry sequence breakers
    if (params[@"dry_sequence_breakers"] && [params[@"dry_sequence_breakers"] isKindOfClass:[NSArray class]]) {
        NSArray *dry_sequence_breakers = params[@"dry_sequence_breakers"];
        for (NSString *s in dry_sequence_breakers) {
            sparams.dry_sequence_breakers.push_back([s UTF8String]);
        }
    }

    // Grammar
    if (params[@"grammar"]) {
        sparams.grammar = [params[@"grammar"] UTF8String];
    }

    if (params[@"json_schema"] && !params[@"grammar"]) {
        sparams.grammar = json_schema_to_grammar(json::parse([params[@"json_schema"] UTF8String]));
    }

    if (params[@"grammar_lazy"]) {
        sparams.grammar_lazy = [params[@"grammar_lazy"] boolValue];
    }

    // Preserved tokens
    if (params[@"preserved_tokens"] && [params[@"preserved_tokens"] isKindOfClass:[NSArray class]]) {
        NSArray *preserved_tokens = params[@"preserved_tokens"];
        for (NSString *token in preserved_tokens) {
            auto ids = common_tokenize(llama->ctx, [token UTF8String], false, true);
            if (ids.size() == 1) {
                sparams.preserved_tokens.insert(ids[0]);
            }
        }
    }

    // Grammar triggers
    if (params[@"grammar_triggers"] && [params[@"grammar_triggers"] isKindOfClass:[NSArray class]]) {
        NSArray *grammar_triggers = params[@"grammar_triggers"];
        for (NSDictionary *grammar_trigger in grammar_triggers) {
            const auto type = static_cast<common_grammar_trigger_type>([grammar_trigger[@"type"] intValue]);
            const auto & word = [grammar_trigger[@"value"] UTF8String];

            if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                auto ids = common_tokenize(llama->ctx, word, false, true);
                if (ids.size() == 1) {
                    auto token = ids[0];
                    if (std::find(sparams.preserved_tokens.begin(), sparams.preserved_tokens.end(), (llama_token) token) == sparams.preserved_tokens.end()) {
                        throw std::runtime_error("Grammar trigger word should be marked as preserved token");
                    }
                    common_grammar_trigger trigger;
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    trigger.value = word;
                    trigger.token = token;
                    sparams.grammar_triggers.push_back(std::move(trigger));
                } else {
                    sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                }
            } else {
                common_grammar_trigger trigger;
                trigger.type = type;
                trigger.value = word;
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    const auto token = (llama_token) [grammar_trigger[@"token"] intValue];
                    trigger.token = token;
                }
                sparams.grammar_triggers.push_back(std::move(trigger));
            }
        }
    }

    // Stop words (antiprompt)
    cpp_params.antiprompt.clear();
    if (params[@"stop"]) {
        NSArray *stop = params[@"stop"];
        for (NSString *s in stop) {
            cpp_params.antiprompt.push_back([s UTF8String]);
        }
    }

    // Logit bias
    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    sparams.logit_bias.clear();
    if (params[@"ignore_eos"] && [params[@"ignore_eos"] boolValue]) {
        sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
    }

    if (params[@"logit_bias"] && [params[@"logit_bias"] isKindOfClass:[NSArray class]]) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        NSArray *logit_bias = params[@"logit_bias"];
        for (NSArray *el in logit_bias) {
            if ([el isKindOfClass:[NSArray class]] && [el count] == 2) {
                llama_token tok = [el[0] intValue];
                if (tok >= 0 && tok < n_vocab) {
                    if ([el[1] isKindOfClass:[NSNumber class]]) {
                        sparams.logit_bias[tok].bias = [el[1] doubleValue];
                    } else if ([el[1] isKindOfClass:[NSNumber class]] && ![el[1] boolValue]) {
                        sparams.logit_bias[tok].bias = -INFINITY;
                    }
                }
            }
        }
    }

    // Get chat format params
    int chat_format = params[@"chat_format"] ? [params[@"chat_format"] intValue] : 0;

    // Convert reasoning_format from string to enum (same as completion method)
    NSString *reasoningFormat = params[@"reasoning_format"];
    if (!reasoningFormat) reasoningFormat = @"none";
    std::string reasoningFormatStr = [reasoningFormat UTF8String];
    common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);

    bool thinking_forced_open = params[@"thinking_forced_open"] ? [params[@"thinking_forced_open"] boolValue] : false;

    // Get prefill text
    NSString *prefillText = params[@"prefill_text"];
    std::string prefill_text_str = prefillText ? [prefillText UTF8String] : "";

    // Get state parameters
    std::string load_state_path;
    std::string save_state_path;
    int32_t save_state_size = -1;

    if (params[@"load_state_path"] && [params[@"load_state_path"] isKindOfClass:[NSString class]]) {
        NSString *path = params[@"load_state_path"];
        // Remove file:// prefix if present
        if ([path hasPrefix:@"file://"]) {
            path = [path substringFromIndex:7];
        }
        load_state_path = [path UTF8String];
    }

    if (params[@"save_state_path"] && [params[@"save_state_path"] isKindOfClass:[NSString class]]) {
        NSString *path = params[@"save_state_path"];
        // Remove file:// prefix if present
        if ([path hasPrefix:@"file://"]) {
            path = [path substringFromIndex:7];
        }
        save_state_path = [path UTF8String];
    }

    if (params[@"save_state_size"] && [params[@"save_state_size"] isKindOfClass:[NSNumber class]]) {
        save_state_size = [params[@"save_state_size"] intValue];
    }

    // Copy blocks to ensure they're retained on the heap for async use
    void (^onTokenCopy)(NSMutableDictionary *) = [onToken copy];
    void (^onCompleteCopy)(NSDictionary *) = [onComplete copy];

    // Capture slot manager for parseChatOutput
    auto* slot_mgr = llama->slot_manager;

    // Create callbacks
    auto token_callback = [onTokenCopy, slot_mgr](const rnllama::completion_token_output& token) {
        // Capture all needed data from token before dispatching
        int32_t reqId = token.request_id;
        int32_t tok = token.tok;
        std::string token_text = token.text;

        // Copy probabilities vector to avoid dangling reference
        std::vector<rnllama::completion_token_output::token_prob> probs_copy = token.probs;

        // Find slot and parse chat output
        rnllama::completion_chat_output parsed_output;
        bool has_parsed_output = false;
        if (slot_mgr) {
            auto* slot = slot_mgr->get_slot_by_request_id(reqId);
            if (slot) {
                parsed_output = slot->parseChatOutput(true);  // is_partial = true
                has_parsed_output = true;
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
            result[@"requestId"] = @(reqId);
            result[@"token"] = [NSString stringWithUTF8String:token_text.c_str()];

            // Add probabilities if available
            if (!probs_copy.empty()) {
                NSMutableArray *probs = [[NSMutableArray alloc] init];
                for (const auto& prob : probs_copy) {
                    [probs addObject:@{
                        @"tok": @(prob.tok),
                        @"prob": @(prob.prob)
                    }];
                }
                result[@"probs"] = probs;
            }

            // Add parsed chat output (content, reasoning_content, tool_calls, accumulated_text)
            if (has_parsed_output) {
                if (!parsed_output.content.empty()) {
                    result[@"content"] = [NSString stringWithUTF8String:parsed_output.content.c_str()];
                }
                if (!parsed_output.reasoning_content.empty()) {
                    result[@"reasoning_content"] = [NSString stringWithUTF8String:parsed_output.reasoning_content.c_str()];
                }
                if (!parsed_output.tool_calls.empty()) {
                    NSMutableArray *toolCalls = [[NSMutableArray alloc] init];
                    for (const auto &tc : parsed_output.tool_calls) {
                        [toolCalls addObject:@{
                            @"type": @"function",
                            @"function": @{
                                @"name": [NSString stringWithUTF8String:tc.name.c_str()],
                                @"arguments": [NSString stringWithUTF8String:tc.arguments.c_str()],
                            },
                            @"id": tc.id.empty() ? [NSNull null] : [NSString stringWithUTF8String:tc.id.c_str()],
                        }];
                    }
                    result[@"tool_calls"] = toolCalls;
                }
                if (!parsed_output.accumulated_text.empty()) {
                    result[@"accumulated_text"] = [NSString stringWithUTF8String:parsed_output.accumulated_text.c_str()];
                }
            }

            if (onTokenCopy) {
                onTokenCopy(result);
            }
        });
    };

    auto complete_callback = [onCompleteCopy](rnllama::llama_rn_slot* slot) {
        // Capture all needed data from slot before dispatching
        int32_t reqId = slot->request_id;
        std::string generated_text = slot->generated_text;
        bool stopped_eos = slot->stopped_eos;
        bool stopped_limit = slot->stopped_limit;
        bool stopped_word = slot->stopped_word;
        bool context_full = slot->context_full;
        bool incomplete = slot->incomplete;
        int n_decoded = slot->n_decoded;
        std::string error_message = slot->error_message;

        // Get timings
        rnllama::slot_timings timings = slot->get_timings();

        // Parse final chat output
        rnllama::completion_chat_output final_output;
        bool has_final_output = false;
        try {
            final_output = slot->parseChatOutput(false);  // is_partial = false
            has_final_output = true;
        } catch (...) {
            // Ignore parsing errors
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            NSMutableDictionary *result = [[NSMutableDictionary alloc] init];
            result[@"requestId"] = @(reqId);
            result[@"text"] = [NSString stringWithUTF8String:generated_text.c_str()];
            result[@"stopped_eos"] = @(stopped_eos);
            result[@"stopped_limit"] = @(stopped_limit);
            result[@"stopped_word"] = @(stopped_word);
            result[@"context_full"] = @(context_full);
            result[@"incomplete"] = @(incomplete);
            result[@"n_decoded"] = @(n_decoded);

            // Add timings
            result[@"timings"] = @{
                @"cache_n": @(timings.cache_n),
                @"prompt_n": @(timings.prompt_n),
                @"prompt_ms": @(timings.prompt_ms),
                @"prompt_per_token_ms": @(timings.prompt_per_token_ms),
                @"prompt_per_second": @(timings.prompt_per_second),
                @"predicted_n": @(timings.predicted_n),
                @"predicted_ms": @(timings.predicted_ms),
                @"predicted_per_token_ms": @(timings.predicted_per_token_ms),
                @"predicted_per_second": @(timings.predicted_per_second)
            };

            // Add error message if present
            if (!error_message.empty()) {
                result[@"error"] = [NSString stringWithUTF8String:error_message.c_str()];
            }

            // Add parsed chat output (final)
            if (has_final_output) {
                if (!final_output.content.empty()) {
                    result[@"content"] = [NSString stringWithUTF8String:final_output.content.c_str()];
                }
                if (!final_output.reasoning_content.empty()) {
                    result[@"reasoning_content"] = [NSString stringWithUTF8String:final_output.reasoning_content.c_str()];
                }
                if (!final_output.tool_calls.empty()) {
                    NSMutableArray *toolCalls = [[NSMutableArray alloc] init];
                    for (const auto &tc : final_output.tool_calls) {
                        [toolCalls addObject:@{
                            @"type": @"function",
                            @"function": @{
                                @"name": [NSString stringWithUTF8String:tc.name.c_str()],
                                @"arguments": [NSString stringWithUTF8String:tc.arguments.c_str()],
                            },
                            @"id": tc.id.empty() ? [NSNull null] : [NSString stringWithUTF8String:tc.id.c_str()],
                        }];
                    }
                    result[@"tool_calls"] = toolCalls;
                }
            }

            if (onCompleteCopy) {
                onCompleteCopy(result);
            }
        });
    };

    // Queue the request
    requestId = llama->slot_manager->queue_request(
        cpp_params,
        tokenize_result.tokens,
        tokenize_result.has_media ? [self convertNSArrayToStdVector:mediaPaths] : std::vector<std::string>(),
        prompt ? [prompt UTF8String] : "",  // Original prompt text (needed for media processing)
        chat_format,
        reasoning_format,
        thinking_forced_open,
        prefill_text_str,
        load_state_path,
        save_state_path,
        save_state_size,
        token_callback,
        complete_callback
    );

    return @(requestId);
}

// Cancel a queued request
- (void)cancelRequest:(NSNumber *)requestId {
    if (llama && llama->parallel_mode_enabled && llama->slot_manager) {
        llama->slot_manager->cancel_request([requestId intValue]);
    }
}

// Queue an embedding request (async, non-blocking)
- (NSNumber *)queueEmbedding:(NSString *)text params:(NSDictionary *)params onResult:(void (^)(int32_t, NSArray *))onResult {
    if (!is_model_loaded) {
        return @(-1);
    }

    // Check if parallel mode is enabled
    if (!llama->parallel_mode_enabled) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:@"Parallel mode is not enabled. Call enableParallelMode() first."
                                     userInfo:nil];
    }

    // Get normalization parameter
    int embd_normalize = llama->params.embd_normalize;
    if (params[@"embd_normalize"] && [params[@"embd_normalize"] isKindOfClass:[NSNumber class]]) {
        embd_normalize = [params[@"embd_normalize"] intValue];
    }

    // Tokenize text
    const llama_vocab* vocab = llama_model_get_vocab(llama->model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const bool is_enc_dec = llama_model_has_encoder(llama->model);
    std::vector<llama_token> tokens = common_tokenize(
        llama->ctx,
        [text UTF8String],
        add_bos || is_enc_dec,
        true
    );

    // Copy callback to ensure it's retained on the heap for async use
    void (^onResultCopy)(int32_t, NSArray *) = [onResult copy];

    // Queue embedding request
    int32_t request_id = llama->slot_manager->queue_embedding_request(
        tokens,
        embd_normalize,
        [onResultCopy](int32_t request_id, const std::vector<float>& embedding) {
            // Copy embedding vector to avoid dangling reference
            NSLog(@"embedding: 0: %f, 1: %f, 2: %f", embedding[0], embedding[1], embedding[2]);
            std::vector<float> embedding_copy = embedding;

            // Convert result to NSArray and dispatch to main queue
            dispatch_async(dispatch_get_main_queue(), ^{
                NSMutableArray *embeddingArray = [[NSMutableArray alloc] init];
                for (float val : embedding_copy) {
                    [embeddingArray addObject:@(val)];
                }

                if (onResultCopy) {
                    onResultCopy(request_id, embeddingArray);
                }
            });
        }
    );

    return @(request_id);
}

// Queue a rerank request (async, non-blocking)
- (NSNumber *)queueRerank:(NSString *)query documents:(NSArray<NSString *> *)documents params:(NSDictionary *)params onResults:(void (^)(int32_t, NSArray *))onResults {
    if (!is_model_loaded) {
        return @(-1);
    }

    // Check if parallel mode is enabled
    if (!llama->parallel_mode_enabled) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:@"Parallel mode is not enabled. Call enableParallelMode() first."
                                     userInfo:nil];
    }

    // Get normalization parameter
    int normalize = 0;  // Default for rerank
    if (params[@"normalize"] && [params[@"normalize"] isKindOfClass:[NSNumber class]]) {
        normalize = [params[@"normalize"] intValue];
    }

    // Convert NSArray to std::vector<std::string>
    std::vector<std::string> docs_vector;
    for (NSString *doc in documents) {
        docs_vector.push_back([doc UTF8String]);
    }

    // Copy callback to ensure it's retained on the heap for async use
    void (^onResultsCopy)(int32_t, NSArray *) = [onResults copy];

    // Queue rerank request
    int32_t request_id = llama->slot_manager->queue_rerank_request(
        std::string([query UTF8String]),
        docs_vector,
        normalize,
        [onResultsCopy](int32_t request_id, const std::vector<float>& scores) {
            // Copy scores vector to avoid dangling reference
            std::vector<float> scores_copy = scores;

            // Convert results to NSArray and dispatch to main queue
            dispatch_async(dispatch_get_main_queue(), ^{
                NSMutableArray *resultsArray = [[NSMutableArray alloc] init];
                for (size_t i = 0; i < scores_copy.size(); i++) {
                    [resultsArray addObject:@{
                        @"score": @(scores_copy[i]),
                        @"index": @((int)i)
                    }];
                }

                if (onResultsCopy) {
                    onResultsCopy(request_id, resultsArray);
                }
            });
        }
    );

    return @(request_id);
}

- (BOOL)enableParallelMode:(int)nParallel nBatch:(int)nBatch {
    if (!llama) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:@"Cannot enable parallel mode: context not initialized"
                                     userInfo:nil];
    }

    // If parallel mode is already enabled, stop the processing loop first
    if (llama->parallel_mode_enabled) {
        NSLog(@"Reconfiguring parallel mode with %d slots", nParallel);
        [self stopProcessingLoop];
    }

    try {
        llama->enableParallelMode(nParallel, nBatch);
    } catch (const std::runtime_error& e) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:[NSString stringWithUTF8String:e.what()]
                                     userInfo:nil];
    } catch (const std::exception& e) {
        @throw [NSException exceptionWithName:@"LlamaException"
                                       reason:[NSString stringWithUTF8String:e.what()]
                                     userInfo:nil];
    }

    [self startProcessingLoop];
    return YES;
}

- (void)disableParallelMode {
    if (!llama) {
        return;
    }

    [self stopProcessingLoop];
    llama->disableParallelMode();
}

// Start background processing loop
- (void)startProcessingLoop {
    if (llama && llama->parallel_mode_enabled && llama->slot_manager) {
        llama->slot_manager->start_processing_loop();
    }
}

// Stop background processing loop
- (void)stopProcessingLoop {
    if (llama && llama->slot_manager) {
        llama->slot_manager->stop_processing_loop();
    }
}

// Helper method to convert NSArray to std::vector<std::string>
- (std::vector<std::string>)convertNSArrayToStdVector:(NSArray *)array {
    std::vector<std::string> result;
    for (NSString *str in array) {
        result.push_back([str UTF8String]);
    }
    return result;
}

- (void)invalidate {
    [self stopProcessingLoop];

    delete llama;
    // llama_backend_free();
}

@end
