llama.rn

# llama.rn

## Table of contents

### Classes

- [LlamaContext](classes/LlamaContext.md)

### Type Aliases

- [BenchResult](README.md#benchresult)
- [CompletionBaseParams](README.md#completionbaseparams)
- [CompletionParams](README.md#completionparams)
- [CompletionResponseFormat](README.md#completionresponseformat)
- [ContextParams](README.md#contextparams)
- [EmbeddingParams](README.md#embeddingparams)
- [FormattedChatResult](README.md#formattedchatresult)
- [JinjaFormattedChatResult](README.md#jinjaformattedchatresult)
- [NativeBackendDeviceInfo](README.md#nativebackenddeviceinfo)
- [NativeCompletionParams](README.md#nativecompletionparams)
- [NativeCompletionResult](README.md#nativecompletionresult)
- [NativeCompletionResultTimings](README.md#nativecompletionresulttimings)
- [NativeCompletionTokenProb](README.md#nativecompletiontokenprob)
- [NativeCompletionTokenProbItem](README.md#nativecompletiontokenprobitem)
- [NativeContextParams](README.md#nativecontextparams)
- [NativeEmbeddingParams](README.md#nativeembeddingparams)
- [NativeEmbeddingResult](README.md#nativeembeddingresult)
- [NativeImageProcessingResult](README.md#nativeimageprocessingresult)
- [NativeLlamaContext](README.md#nativellamacontext)
- [NativeParallelCompletionParams](README.md#nativeparallelcompletionparams)
- [NativeRerankParams](README.md#nativererankparams)
- [NativeRerankResult](README.md#nativererankresult)
- [NativeSessionLoadResult](README.md#nativesessionloadresult)
- [NativeTokenizeResult](README.md#nativetokenizeresult)
- [ParallelCompletionParams](README.md#parallelcompletionparams)
- [ParallelRequestStatus](README.md#parallelrequeststatus)
- [ParallelStatus](README.md#parallelstatus)
- [RNLlamaMessagePart](README.md#rnllamamessagepart)
- [RNLlamaOAICompatibleMessage](README.md#rnllamaoaicompatiblemessage)
- [RerankParams](README.md#rerankparams)
- [RerankResult](README.md#rerankresult)
- [TokenData](README.md#tokendata)
- [ToolCall](README.md#toolcall)

### Variables

- [BuildInfo](README.md#buildinfo)
- [RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER](README.md#rnllama_mtmd_default_media_marker)

### Functions

- [addNativeLogListener](README.md#addnativeloglistener)
- [getBackendDevicesInfo](README.md#getbackenddevicesinfo)
- [initLlama](README.md#initllama)
- [installJsi](README.md#installjsi)
- [loadLlamaModelInfo](README.md#loadllamamodelinfo)
- [releaseAllLlama](README.md#releaseallllama)
- [setContextLimit](README.md#setcontextlimit)
- [toggleNativeLog](README.md#togglenativelog)

## Type Aliases

### BenchResult

Ƭ **BenchResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `flashAttn` | `number` |
| `isPpShared` | `number` |
| `nBatch` | `number` |
| `nGpuLayers` | `number` |
| `nKv` | `number` |
| `nKvMax` | `number` |
| `nThreads` | `number` |
| `nThreadsBatch` | `number` |
| `nUBatch` | `number` |
| `pl` | `number` |
| `pp` | `number` |
| `speed` | `number` |
| `speedPp` | `number` |
| `speedTg` | `number` |
| `t` | `number` |
| `tPp` | `number` |
| `tTg` | `number` |
| `tg` | `number` |

#### Defined in

[index.ts:286](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L286)

___

### CompletionBaseParams

Ƭ **CompletionBaseParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `add_generation_prompt?` | `boolean` | - |
| `chatTemplate?` | `string` | - |
| `chat_template?` | `string` | - |
| `chat_template_kwargs?` | `Record`<`string`, `string`\> | - |
| `jinja?` | `boolean` | - |
| `media_paths?` | `string` \| `string`[] | - |
| `messages?` | [`RNLlamaOAICompatibleMessage`](README.md#rnllamaoaicompatiblemessage)[] | - |
| `now?` | `string` \| `number` | - |
| `parallel_tool_calls?` | `object` | - |
| `prefill_text?` | `string` | Prefill text to be used for chat parsing (Generation Prompt + Content) Used for if last assistant message is for prefill purpose |
| `prompt?` | `string` | - |
| `response_format?` | [`CompletionResponseFormat`](README.md#completionresponseformat) | - |
| `tool_choice?` | `string` | - |
| `tools?` | `object` | - |

#### Defined in

[index.ts:247](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L247)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<[`NativeCompletionParams`](README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](README.md#completionbaseparams)

#### Defined in

[index.ts:270](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L270)

___

### CompletionResponseFormat

Ƭ **CompletionResponseFormat**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `json_schema?` | { `schema`: `object` ; `strict?`: `boolean`  } |
| `json_schema.schema` | `object` |
| `json_schema.strict?` | `boolean` |
| `schema?` | `object` |
| `type` | ``"text"`` \| ``"json_object"`` \| ``"json_schema"`` |

#### Defined in

[index.ts:238](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L238)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<[`NativeContextParams`](README.md#nativecontextparams), ``"flash_attn_type"`` \| ``"cache_type_k"`` \| ``"cache_type_v"`` \| ``"pooling_type"``\> & { `cache_type_k?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `cache_type_v?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `flash_attn_type?`: ``"auto"`` \| ``"on"`` \| ``"off"`` ; `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:188](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L188)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: [`NativeEmbeddingParams`](README.md#nativeembeddingparams)

#### Defined in

[index.ts:226](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L226)

___

### FormattedChatResult

Ƭ **FormattedChatResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `has_media` | `boolean` |
| `media_paths?` | `string`[] |
| `prompt` | `string` |
| `type` | ``"jinja"`` \| ``"llama-chat"`` |

#### Defined in

[types.ts:493](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L493)

___

### JinjaFormattedChatResult

Ƭ **JinjaFormattedChatResult**: [`FormattedChatResult`](README.md#formattedchatresult) & { `additional_stops?`: `string`[] ; `chat_format?`: `number` ; `chat_parser?`: `string` ; `grammar?`: `string` ; `grammar_lazy?`: `boolean` ; `grammar_triggers?`: { `token`: `number` ; `type`: `number` ; `value`: `string`  }[] ; `preserved_tokens?`: `string`[] ; `thinking_forced_open?`: `boolean`  }

#### Defined in

[types.ts:500](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L500)

___

### NativeBackendDeviceInfo

Ƭ **NativeBackendDeviceInfo**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `backend` | `string` |
| `deviceName` | `string` |
| `maxMemorySize` | `number` |
| `metadata?` | `Record`<`string`, `any`\> |
| `type` | `string` |

#### Defined in

[types.ts:534](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L534)

___

### NativeCompletionParams

Ƭ **NativeCompletionParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `chat_format?` | `number` | - |
| `chat_parser?` | `string` | Serialized PEG parser for chat output parsing. Required for COMMON_CHAT_FORMAT_PEG_* formats. This is typically obtained from getFormattedChat with jinja enabled. |
| `dry_allowed_length?` | `number` | Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2` |
| `dry_base?` | `number` | Set the DRY repetition penalty base value. Default: `1.75` |
| `dry_multiplier?` | `number` | Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled. |
| `dry_penalty_last_n?` | `number` | How many tokens to scan for repetitions. Default: `-1`, where `0` is disabled and `-1` is context size. |
| `dry_sequence_breakers?` | `string`[] | Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']` |
| `emit_partial_completion` | `boolean` | - |
| `enable_thinking?` | `boolean` | Enable thinking if jinja is enabled. Default: true |
| `grammar?` | `string` | Set grammar for grammar-based sampling. Default: no grammar |
| `grammar_lazy?` | `boolean` | Lazy grammar sampling, trigger by grammar_triggers. Default: false |
| `grammar_triggers?` | { `token`: `number` ; `type`: `number` ; `value`: `string`  }[] | Lazy grammar triggers. Default: [] |
| `guide_tokens?` | `number`[] | Guide tokens for the completion. Help prevent hallucinations by forcing the TTS to use the correct words. Default: `[]` |
| `ignore_eos?` | `boolean` | Ignore end of stream token and continue generating. Default: `false` |
| `jinja?` | `boolean` | Enable Jinja. Default: true if supported by the model |
| `json_schema?` | `string` | JSON schema for convert to grammar for structured JSON output. It will be override by grammar if both are set. |
| `logit_bias?` | `number`[][] | Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g.`[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does. Default: `[]` |
| `media_paths?` | `string`[] | Path to an image file to process before generating text. When provided, the image will be processed and added to the context. Requires multimodal support to be enabled via initMultimodal. |
| `min_p?` | `number` | The minimum probability for a token to be considered, relative to the probability of the most likely token. Default: `0.05` |
| `mirostat?` | `number` | Enable Mirostat sampling, controlling perplexity during text generation. Default: `0`, where `0` is disabled, `1` is Mirostat, and `2` is Mirostat 2.0. |
| `mirostat_eta?` | `number` | Set the Mirostat learning rate, parameter eta. Default: `0.1` |
| `mirostat_tau?` | `number` | Set the Mirostat target entropy, parameter tau. Default: `5.0` |
| `n_predict?` | `number` | Set the maximum number of tokens to predict when generating text. **Note:** May exceed the set limit slightly if the last token is a partial multibyte character. When 0,no tokens will be generated but the prompt is evaluated into the cache. Default: `-1`, where `-1` is infinity. |
| `n_probs?` | `number` | If greater than 0, the response also contains the probabilities of top N tokens for each generated token given the sampling settings. Note that for temperature < 0 the tokens are sampled greedily but token probabilities are still being calculated via a simple softmax of the logits without considering any other sampler settings. Default: `0` |
| `n_threads?` | `number` | - |
| `penalty_freq?` | `number` | Repeat alpha frequency penalty. Default: `0.0`, which is disabled. |
| `penalty_last_n?` | `number` | Last n tokens to consider for penalizing repetition. Default: `64`, where `0` is disabled and `-1` is ctx-size. |
| `penalty_present?` | `number` | Repeat alpha presence penalty. Default: `0.0`, which is disabled. |
| `penalty_repeat?` | `number` | Control the repetition of token sequences in the generated text. Default: `1.0` |
| `preserved_tokens?` | `string`[] | - |
| `prompt` | `string` | - |
| `reasoning_format?` | ``"none"`` \| ``"auto"`` \| ``"deepseek"`` | - |
| `seed?` | `number` | Set the random number generator (RNG) seed. Default: `-1`, which is a random seed. |
| `stop?` | `string`[] | Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]` |
| `temperature?` | `number` | Adjust the randomness of the generated text. Default: `0.8` |
| `thinking_forced_open?` | `boolean` | Force thinking to be open. Default: false |
| `top_k?` | `number` | Limit the next token selection to the K most probable tokens. Default: `40` |
| `top_n_sigma?` | `number` | Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641. Default: `-1.0` (Disabled) |
| `top_p?` | `number` | Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95` |
| `typical_p?` | `number` | Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled. |
| `xtc_probability?` | `number` | Set the chance for token removal via XTC sampler. Default: `0.0`, which is disabled. |
| `xtc_threshold?` | `number` | Set a minimum probability threshold for tokens to be removed via XTC sampler. Default: `0.1` (> `0.5` disables XTC) |

#### Defined in

[types.ts:123](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L123)

___

### NativeCompletionResult

Ƭ **NativeCompletionResult**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `audio_tokens?` | `number`[] | - |
| `chat_format` | `number` | - |
| `completion_probabilities?` | [`NativeCompletionTokenProb`](README.md#nativecompletiontokenprob)[] | - |
| `content` | `string` | Content text (Filtered text by reasoning_content / tool_calls) |
| `context_full` | `boolean` | - |
| `interrupted` | `boolean` | - |
| `reasoning_content` | `string` | Reasoning content (parsed for reasoning model) |
| `stopped_eos` | `boolean` | - |
| `stopped_limit` | `number` | - |
| `stopped_word` | `string` | - |
| `stopping_word` | `string` | - |
| `text` | `string` | Original text (Ignored reasoning_content / tool_calls) |
| `timings` | [`NativeCompletionResultTimings`](README.md#nativecompletionresulttimings) | - |
| `tokens_cached` | `number` | - |
| `tokens_evaluated` | `number` | - |
| `tokens_predicted` | `number` | - |
| `tool_calls` | { `function`: { `arguments`: `string` ; `name`: `string`  } ; `id?`: `string` ; `type`: ``"function"``  }[] | Tool calls |
| `truncated` | `boolean` | - |

#### Defined in

[types.ts:366](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L366)

___

### NativeCompletionResultTimings

Ƭ **NativeCompletionResultTimings**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `cache_n` | `number` |
| `predicted_ms` | `number` |
| `predicted_n` | `number` |
| `predicted_per_second` | `number` |
| `predicted_per_token_ms` | `number` |
| `prompt_ms` | `number` |
| `prompt_n` | `number` |
| `prompt_per_second` | `number` |
| `prompt_per_token_ms` | `number` |

#### Defined in

[types.ts:354](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L354)

___

### NativeCompletionTokenProb

Ƭ **NativeCompletionTokenProb**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content` | `string` |
| `probs` | [`NativeCompletionTokenProbItem`](README.md#nativecompletiontokenprobitem)[] |

#### Defined in

[types.ts:349](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L349)

___

### NativeCompletionTokenProbItem

Ƭ **NativeCompletionTokenProbItem**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prob` | `number` |
| `tok_str` | `string` |

#### Defined in

[types.ts:344](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L344)

___

### NativeContextParams

Ƭ **NativeContextParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `cache_type_k?` | `string` | KV cache data type for the K (Experimental in llama.cpp) |
| `cache_type_v?` | `string` | KV cache data type for the V (Experimental in llama.cpp) |
| `chat_template?` | `string` | Chat template to override the default one from the model. |
| `cpu_mask?` | `string` | CPU affinity mask string (e.g., "0-3" or "0,2,4,6"). Specifies which CPU cores to use for inference. |
| `cpu_strict?` | `boolean` | Use strict CPU placement. When true, enforces strict CPU core affinity. Default: false |
| `ctx_shift?` | `boolean` | Enable context shifting to handle prompts larger than context size |
| `devices?` | `string`[] | Backend devices choice to use. Default equals to result of `getBackendDevicesInfo. |
| `embd_normalize?` | `number` | - |
| `embedding?` | `boolean` | - |
| `flash_attn?` | `boolean` | Enable flash attention, only recommended in GPU device Deprecated: use flash_attn_type instead |
| `flash_attn_type?` | `string` | Enable flash attention, only recommended in GPU device. |
| `is_model_asset?` | `boolean` | - |
| `kv_unified?` | `boolean` | Use a unified buffer across the input sequences when computing the attention. Try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix. |
| `lora?` | `string` | Single LoRA adapter path |
| `lora_list?` | { `path`: `string` ; `scaled?`: `number`  }[] | LoRA adapter list |
| `lora_scaled?` | `number` | Single LoRA adapter scale |
| `model` | `string` | - |
| `n_batch?` | `number` | - |
| `n_cpu_moe?` | `number` | Number of layers to keep MoE weights on CPU |
| `n_ctx?` | `number` | - |
| `n_gpu_layers?` | `number` | Number of layers to store in VRAM (Currently only for iOS) |
| `n_parallel?` | `number` | Number of parallel sequences to support (sets n_seq_max). This determines the maximum number of parallel slots that can be used. Default: 8 |
| `n_threads?` | `number` | - |
| `n_ubatch?` | `number` | - |
| `no_gpu_devices?` | `boolean` | Skip GPU devices (iOS only) (Deprecated: Please set devices params instead) |
| `pooling_type?` | `number` | - |
| `rope_freq_base?` | `number` | - |
| `rope_freq_scale?` | `number` | - |
| `swa_full?` | `boolean` | Use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055) |
| `use_mlock?` | `boolean` | - |
| `use_mmap?` | `boolean` | - |
| `use_progress_callback?` | `boolean` | - |
| `vocab_only?` | `boolean` | - |

#### Defined in

[types.ts:5](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L5)

___

### NativeEmbeddingParams

Ƭ **NativeEmbeddingParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embd_normalize?` | `number` |

#### Defined in

[types.ts:1](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L1)

___

### NativeEmbeddingResult

Ƭ **NativeEmbeddingResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embedding` | `number`[] |

#### Defined in

[types.ts:429](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L429)

___

### NativeImageProcessingResult

Ƭ **NativeImageProcessingResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `error?` | `string` |
| `prompt` | `string` |
| `success` | `boolean` |

#### Defined in

[types.ts:519](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L519)

___

### NativeLlamaContext

Ƭ **NativeLlamaContext**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `androidLib?` | `string` | Loaded library name for Android |
| `contextId` | `number` | - |
| `devices?` | `string`[] | Name of the GPU device used on Android/iOS (if available) |
| `gpu` | `boolean` | - |
| `model` | { `chatTemplates`: { `jinja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } ; `llamaChat`: `boolean`  } ; `desc`: `string` ; `isChatTemplateSupported`: `boolean` ; `is_hybrid`: `boolean` ; `is_recurrent`: `boolean` ; `metadata`: `Object` ; `nEmbd`: `number` ; `nParams`: `number` ; `size`: `number`  } | - |
| `model.chatTemplates` | { `jinja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } ; `llamaChat`: `boolean`  } | - |
| `model.chatTemplates.jinja` | { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } | - |
| `model.chatTemplates.jinja.default` | `boolean` | - |
| `model.chatTemplates.jinja.defaultCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } | - |
| `model.chatTemplates.jinja.defaultCaps.parallelToolCalls` | `boolean` | - |
| `model.chatTemplates.jinja.defaultCaps.systemRole` | `boolean` | - |
| `model.chatTemplates.jinja.defaultCaps.toolCalls` | `boolean` | - |
| `model.chatTemplates.jinja.defaultCaps.tools` | `boolean` | - |
| `model.chatTemplates.jinja.toolUse` | `boolean` | - |
| `model.chatTemplates.jinja.toolUseCaps?` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } | - |
| `model.chatTemplates.jinja.toolUseCaps.parallelToolCalls` | `boolean` | - |
| `model.chatTemplates.jinja.toolUseCaps.systemRole` | `boolean` | - |
| `model.chatTemplates.jinja.toolUseCaps.toolCalls` | `boolean` | - |
| `model.chatTemplates.jinja.toolUseCaps.tools` | `boolean` | - |
| `model.chatTemplates.llamaChat` | `boolean` | - |
| `model.desc` | `string` | - |
| `model.isChatTemplateSupported` | `boolean` | - |
| `model.is_hybrid` | `boolean` | - |
| `model.is_recurrent` | `boolean` | - |
| `model.metadata` | `Object` | - |
| `model.nEmbd` | `number` | - |
| `model.nParams` | `number` | - |
| `model.size` | `number` | - |
| `reasonNoGPU` | `string` | - |
| `systemInfo` | `string` | - |

#### Defined in

[types.ts:433](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L433)

___

### NativeParallelCompletionParams

Ƭ **NativeParallelCompletionParams**: [`NativeCompletionParams`](README.md#nativecompletionparams) & { `load_state_path?`: `string` ; `load_state_size?`: `number` ; `save_prompt_state_path?`: `string` ; `save_state_path?`: `string` ; `save_state_size?`: `number`  }

Parameters for parallel completion requests (queueCompletion).
Extends NativeCompletionParams with parallel-mode specific options.

#### Defined in

[types.ts:303](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L303)

___

### NativeRerankParams

Ƭ **NativeRerankParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `normalize?` | `number` |

#### Defined in

[types.ts:525](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L525)

___

### NativeRerankResult

Ƭ **NativeRerankResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `index` | `number` |
| `score` | `number` |

#### Defined in

[types.ts:529](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L529)

___

### NativeSessionLoadResult

Ƭ **NativeSessionLoadResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prompt` | `string` |
| `tokens_loaded` | `number` |

#### Defined in

[types.ts:478](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L478)

___

### NativeTokenizeResult

Ƭ **NativeTokenizeResult**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `bitmap_hashes` | `number`[] | Bitmap hashes of the media |
| `chunk_pos` | `number`[] | Chunk positions of the text and media |
| `chunk_pos_media` | `number`[] | Chunk positions of the media |
| `has_media` | `boolean` | Whether the tokenization contains media |
| `tokens` | `number`[] | - |

#### Defined in

[types.ts:409](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L409)

___

### ParallelCompletionParams

Ƭ **ParallelCompletionParams**: `Omit`<[`NativeParallelCompletionParams`](README.md#nativeparallelcompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](README.md#completionbaseparams)

Parameters for parallel completion requests.
Extends CompletionParams with parallel-mode specific options like state management.

#### Defined in

[index.ts:280](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L280)

___

### ParallelRequestStatus

Ƭ **ParallelRequestStatus**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `generation_ms` | `number` |
| `prompt_length` | `number` |
| `prompt_ms` | `number` |
| `request_id` | `number` |
| `state` | ``"queued"`` \| ``"processing_prompt"`` \| ``"generating"`` \| ``"done"`` |
| `tokens_generated` | `number` |
| `tokens_per_second` | `number` |
| `type` | ``"completion"`` \| ``"embedding"`` \| ``"rerank"`` |

#### Defined in

[types.ts:542](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L542)

___

### ParallelStatus

Ƭ **ParallelStatus**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `active_slots` | `number` |
| `n_parallel` | `number` |
| `queued_requests` | `number` |
| `requests` | [`ParallelRequestStatus`](README.md#parallelrequeststatus)[] |

#### Defined in

[types.ts:553](https://github.com/mybigday/llama.rn/blob/426b246/src/types.ts#L553)

___

### RNLlamaMessagePart

Ƭ **RNLlamaMessagePart**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `image_url?` | { `url?`: `string`  } |
| `image_url.url?` | `string` |
| `input_audio?` | { `data?`: `string` ; `format`: `string` ; `url?`: `string`  } |
| `input_audio.data?` | `string` |
| `input_audio.format` | `string` |
| `input_audio.url?` | `string` |
| `text?` | `string` |
| `type` | `string` |

#### Defined in

[index.ts:29](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L29)

___

### RNLlamaOAICompatibleMessage

Ƭ **RNLlamaOAICompatibleMessage**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content?` | `string` \| [`RNLlamaMessagePart`](README.md#rnllamamessagepart)[] |
| `reasoning_content?` | `string` |
| `role` | `string` |

#### Defined in

[index.ts:42](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L42)

___

### RerankParams

Ƭ **RerankParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `normalize?` | `number` |

#### Defined in

[index.ts:228](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L228)

___

### RerankResult

Ƭ **RerankResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `document?` | `string` |
| `index` | `number` |
| `score` | `number` |

#### Defined in

[index.ts:232](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L232)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `accumulated_text?` | `string` |
| `completion_probabilities?` | [`NativeCompletionTokenProb`](README.md#nativecompletiontokenprob)[] |
| `content?` | `string` |
| `reasoning_content?` | `string` |
| `requestId?` | `number` |
| `token` | `string` |
| `tool_calls?` | [`ToolCall`](README.md#toolcall)[] |

#### Defined in

[index.ts:177](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L177)

___

### ToolCall

Ƭ **ToolCall**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `function` | { `arguments`: `string` ; `name`: `string`  } |
| `function.arguments` | `string` |
| `function.name` | `string` |
| `id?` | `string` |
| `type` | ``"function"`` |

#### Defined in

[index.ts:168](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L168)

## Variables

### BuildInfo

• `Const` **BuildInfo**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `commit` | `string` |
| `number` | `string` |

#### Defined in

[index.ts:1226](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1226)

___

### RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER

• `Const` **RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER**: ``"<__media__>"``

#### Defined in

[index.ts:71](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L71)

## Functions

### addNativeLogListener

▸ **addNativeLogListener**(`listener`): `Object`

#### Parameters

| Name | Type |
| :------ | :------ |
| `listener` | (`level`: `string`, `text`: `string`) => `void` |

#### Returns

`Object`

| Name | Type |
| :------ | :------ |
| `remove` | () => `void` |

#### Defined in

[index.ts:1045](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1045)

___

### getBackendDevicesInfo

▸ **getBackendDevicesInfo**(): `Promise`<[`NativeBackendDeviceInfo`](README.md#nativebackenddeviceinfo)[]\>

#### Returns

`Promise`<[`NativeBackendDeviceInfo`](README.md#nativebackenddeviceinfo)[]\>

#### Defined in

[index.ts:1089](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1089)

___

### initLlama

▸ **initLlama**(`«destructured»`, `onProgress?`): `Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | [`ContextParams`](README.md#contextparams) |
| `onProgress?` | (`progress`: `number`) => `void` |

#### Returns

`Promise`<[`LlamaContext`](classes/LlamaContext.md)\>

#### Defined in

[index.ts:1106](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1106)

___

### installJsi

▸ **installJsi**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:156](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L156)

___

### loadLlamaModelInfo

▸ **loadLlamaModelInfo**(`model`): `Promise`<`Object`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `model` | `string` |

#### Returns

`Promise`<`Object`\>

#### Defined in

[index.ts:1073](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1073)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1220](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1220)

___

### setContextLimit

▸ **setContextLimit**(`limit`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `limit` | `number` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1056](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1056)

___

### toggleNativeLog

▸ **toggleNativeLog**(`enabled`): `Promise`<`void`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `enabled` | `boolean` |

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1039](https://github.com/mybigday/llama.rn/blob/426b246/src/index.ts#L1039)
