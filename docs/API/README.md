llama.rn

# llama.rn

## Table of contents

### Classes

- [LlamaContext](classes/LlamaContext.md)
- [SchemaGrammarConverter](classes/SchemaGrammarConverter.md)
- [SchemaGrammarConverterBuiltinRule](classes/SchemaGrammarConverterBuiltinRule.md)

### Interfaces

- [SchemaGrammarConverterPropOrder](interfaces/SchemaGrammarConverterPropOrder.md)

### Type Aliases

- [BenchResult](README.md#benchresult)
- [CompletionBaseParams](README.md#completionbaseparams)
- [CompletionParams](README.md#completionparams)
- [CompletionResponseFormat](README.md#completionresponseformat)
- [ContextParams](README.md#contextparams)
- [EmbeddingParams](README.md#embeddingparams)
- [JinjaFormattedChatResult](README.md#jinjaformattedchatresult)
- [NativeCompletionParams](README.md#nativecompletionparams)
- [NativeCompletionResult](README.md#nativecompletionresult)
- [NativeCompletionResultTimings](README.md#nativecompletionresulttimings)
- [NativeCompletionTokenProb](README.md#nativecompletiontokenprob)
- [NativeCompletionTokenProbItem](README.md#nativecompletiontokenprobitem)
- [NativeContextParams](README.md#nativecontextparams)
- [NativeEmbeddingParams](README.md#nativeembeddingparams)
- [NativeEmbeddingResult](README.md#nativeembeddingresult)
- [NativeLlamaContext](README.md#nativellamacontext)
- [NativeSessionLoadResult](README.md#nativesessionloadresult)
- [NativeTokenizeResult](README.md#nativetokenizeresult)
- [RNLlamaMessagePart](README.md#rnllamamessagepart)
- [RNLlamaOAICompatibleMessage](README.md#rnllamaoaicompatiblemessage)
- [TokenData](README.md#tokendata)

### Functions

- [addNativeLogListener](README.md#addnativeloglistener)
- [convertJsonSchemaToGrammar](README.md#convertjsonschematogrammar)
- [initLlama](README.md#initllama)
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
| `modelDesc` | `string` |
| `modelNParams` | `number` |
| `modelSize` | `number` |
| `ppAvg` | `number` |
| `ppStd` | `number` |
| `tgAvg` | `number` |
| `tgStd` | `number` |

#### Defined in

[index.ts:138](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L138)

___

### CompletionBaseParams

Ƭ **CompletionBaseParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `chatTemplate?` | `string` |
| `chat_template?` | `string` |
| `jinja?` | `boolean` |
| `messages?` | [`RNLlamaOAICompatibleMessage`](README.md#rnllamaoaicompatiblemessage)[] |
| `parallel_tool_calls?` | `object` |
| `prompt?` | `string` |
| `response_format?` | [`CompletionResponseFormat`](README.md#completionresponseformat) |
| `tool_choice?` | `string` |
| `tools?` | `object` |

#### Defined in

[index.ts:121](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L121)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<[`NativeCompletionParams`](README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](README.md#completionbaseparams)

#### Defined in

[index.ts:132](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L132)

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

[index.ts:112](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L112)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<[`NativeContextParams`](README.md#nativecontextparams), ``"cache_type_k"`` \| ``"cache_type_v"`` \| ``"pooling_type"``\> & { `cache_type_k?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `cache_type_v?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:85](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L85)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: [`NativeEmbeddingParams`](README.md#nativeembeddingparams)

#### Defined in

[index.ts:110](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L110)

___

### JinjaFormattedChatResult

Ƭ **JinjaFormattedChatResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `additional_stops?` | `string`[] |
| `chat_format?` | `number` |
| `grammar?` | `string` |
| `grammar_lazy?` | `boolean` |
| `grammar_triggers?` | { `at_start`: `boolean` ; `word`: `string`  }[] |
| `preserved_tokens?` | `string`[] |
| `prompt` | `string` |

#### Defined in

[NativeRNLlama.ts:309](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L309)

___

### NativeCompletionParams

Ƭ **NativeCompletionParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `chat_format?` | `number` | - |
| `dry_allowed_length?` | `number` | Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2` |
| `dry_base?` | `number` | Set the DRY repetition penalty base value. Default: `1.75` |
| `dry_multiplier?` | `number` | Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled. |
| `dry_penalty_last_n?` | `number` | How many tokens to scan for repetitions. Default: `-1`, where `0` is disabled and `-1` is context size. |
| `dry_sequence_breakers?` | `string`[] | Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']` |
| `emit_partial_completion` | `boolean` | - |
| `grammar?` | `string` | Set grammar for grammar-based sampling. Default: no grammar |
| `grammar_lazy?` | `boolean` | Lazy grammar sampling, trigger by grammar_triggers. Default: false |
| `grammar_triggers?` | { `at_start`: `boolean` ; `word`: `string`  }[] | Lazy grammar triggers. Default: [] |
| `ignore_eos?` | `boolean` | Ignore end of stream token and continue generating. Default: `false` |
| `json_schema?` | `string` | JSON schema for convert to grammar for structured JSON output. It will be override by grammar if both are set. |
| `logit_bias?` | `number`[][] | Modify the likelihood of a token appearing in the generated text completion. For example, use `"logit_bias": [[15043,1.0]]` to increase the likelihood of the token 'Hello', or `"logit_bias": [[15043,-1.0]]` to decrease its likelihood. Setting the value to false, `"logit_bias": [[15043,false]]` ensures that the token `Hello` is never produced. The tokens can also be represented as strings, e.g.`[["Hello, World!",-0.5]]` will reduce the likelihood of all the individual tokens that represent the string `Hello, World!`, just like the `presence_penalty` does. Default: `[]` |
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
| `seed?` | `number` | Set the random number generator (RNG) seed. Default: `-1`, which is a random seed. |
| `stop?` | `string`[] | Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]` |
| `temperature?` | `number` | Adjust the randomness of the generated text. Default: `0.8` |
| `top_k?` | `number` | Limit the next token selection to the K most probable tokens. Default: `40` |
| `top_p?` | `number` | Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95` |
| `typical_p?` | `number` | Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled. |
| `xtc_probability?` | `number` | Set the chance for token removal via XTC sampler. Default: `0.0`, which is disabled. |
| `xtc_threshold?` | `number` | Set a minimum probability threshold for tokens to be removed via XTC sampler. Default: `0.1` (> `0.5` disables XTC) |

#### Defined in

[NativeRNLlama.ts:74](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L74)

___

### NativeCompletionResult

Ƭ **NativeCompletionResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | [`NativeCompletionTokenProb`](README.md#nativecompletiontokenprob)[] |
| `stopped_eos` | `boolean` |
| `stopped_limit` | `number` |
| `stopped_word` | `string` |
| `stopping_word` | `string` |
| `text` | `string` |
| `timings` | [`NativeCompletionResultTimings`](README.md#nativecompletionresulttimings) |
| `tokens_cached` | `number` |
| `tokens_evaluated` | `number` |
| `tokens_predicted` | `number` |
| `truncated` | `boolean` |

#### Defined in

[NativeRNLlama.ts:233](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L233)

___

### NativeCompletionResultTimings

Ƭ **NativeCompletionResultTimings**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `predicted_ms` | `number` |
| `predicted_n` | `number` |
| `predicted_per_second` | `number` |
| `predicted_per_token_ms` | `number` |
| `prompt_ms` | `number` |
| `prompt_n` | `number` |
| `prompt_per_second` | `number` |
| `prompt_per_token_ms` | `number` |

#### Defined in

[NativeRNLlama.ts:222](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L222)

___

### NativeCompletionTokenProb

Ƭ **NativeCompletionTokenProb**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content` | `string` |
| `probs` | [`NativeCompletionTokenProbItem`](README.md#nativecompletiontokenprobitem)[] |

#### Defined in

[NativeRNLlama.ts:217](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L217)

___

### NativeCompletionTokenProbItem

Ƭ **NativeCompletionTokenProbItem**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prob` | `number` |
| `tok_str` | `string` |

#### Defined in

[NativeRNLlama.ts:212](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L212)

___

### NativeContextParams

Ƭ **NativeContextParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `cache_type_k?` | `string` | KV cache data type for the K (Experimental in llama.cpp) |
| `cache_type_v?` | `string` | KV cache data type for the V (Experimental in llama.cpp) |
| `chat_template?` | `string` | Chat template to override the default one from the model. |
| `embd_normalize?` | `number` | - |
| `embedding?` | `boolean` | - |
| `flash_attn?` | `boolean` | Enable flash attention, only recommended in GPU device (Experimental in llama.cpp) |
| `is_model_asset?` | `boolean` | - |
| `lora?` | `string` | Single LoRA adapter path |
| `lora_list?` | { `path`: `string` ; `scaled?`: `number`  }[] | LoRA adapter list |
| `lora_scaled?` | `number` | Single LoRA adapter scale |
| `model` | `string` | - |
| `n_batch?` | `number` | - |
| `n_ctx?` | `number` | - |
| `n_gpu_layers?` | `number` | Number of layers to store in VRAM (Currently only for iOS) |
| `n_threads?` | `number` | - |
| `n_ubatch?` | `number` | - |
| `no_gpu_devices?` | `boolean` | Skip GPU devices (iOS only) |
| `pooling_type?` | `number` | - |
| `rope_freq_base?` | `number` | - |
| `rope_freq_scale?` | `number` | - |
| `use_mlock?` | `boolean` | - |
| `use_mmap?` | `boolean` | - |
| `use_progress_callback?` | `boolean` | - |
| `vocab_only?` | `boolean` | - |

#### Defined in

[NativeRNLlama.ts:8](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L8)

___

### NativeEmbeddingParams

Ƭ **NativeEmbeddingParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embd_normalize?` | `number` |

#### Defined in

[NativeRNLlama.ts:4](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L4)

___

### NativeEmbeddingResult

Ƭ **NativeEmbeddingResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embedding` | `number`[] |

#### Defined in

[NativeRNLlama.ts:253](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L253)

___

### NativeLlamaContext

Ƭ **NativeLlamaContext**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `androidLib?` | `string` | Loaded library name for Android |
| `contextId` | `number` | - |
| `gpu` | `boolean` | - |
| `model` | { `chatTemplates`: { `llamaChat`: `boolean` ; `minja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  }  }  } ; `desc`: `string` ; `isChatTemplateSupported`: `boolean` ; `metadata`: `Object` ; `nEmbd`: `number` ; `nParams`: `number` ; `size`: `number`  } | - |
| `model.chatTemplates` | { `llamaChat`: `boolean` ; `minja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  }  }  } | - |
| `model.chatTemplates.llamaChat` | `boolean` | - |
| `model.chatTemplates.minja` | { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  }  } | - |
| `model.chatTemplates.minja.default` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } | - |
| `model.chatTemplates.minja.defaultCaps.parallelToolCalls` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps.systemRole` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps.toolCallId` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps.toolCalls` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps.toolResponses` | `boolean` | - |
| `model.chatTemplates.minja.defaultCaps.tools` | `boolean` | - |
| `model.chatTemplates.minja.toolUse` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps` | { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCallId`: `boolean` ; `toolCalls`: `boolean` ; `toolResponses`: `boolean` ; `tools`: `boolean`  } | - |
| `model.chatTemplates.minja.toolUseCaps.parallelToolCalls` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps.systemRole` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps.toolCallId` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps.toolCalls` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps.toolResponses` | `boolean` | - |
| `model.chatTemplates.minja.toolUseCaps.tools` | `boolean` | - |
| `model.desc` | `string` | - |
| `model.isChatTemplateSupported` | `boolean` | - |
| `model.metadata` | `Object` | - |
| `model.nEmbd` | `number` | - |
| `model.nParams` | `number` | - |
| `model.size` | `number` | - |
| `reasonNoGPU` | `string` | - |

#### Defined in

[NativeRNLlama.ts:257](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L257)

___

### NativeSessionLoadResult

Ƭ **NativeSessionLoadResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prompt` | `string` |
| `tokens_loaded` | `number` |

#### Defined in

[NativeRNLlama.ts:299](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L299)

___

### NativeTokenizeResult

Ƭ **NativeTokenizeResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] |

#### Defined in

[NativeRNLlama.ts:249](https://github.com/mybigday/llama.rn/blob/64b02c4/src/NativeRNLlama.ts#L249)

___

### RNLlamaMessagePart

Ƭ **RNLlamaMessagePart**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `text?` | `string` |

#### Defined in

[chat.ts:3](https://github.com/mybigday/llama.rn/blob/64b02c4/src/chat.ts#L3)

___

### RNLlamaOAICompatibleMessage

Ƭ **RNLlamaOAICompatibleMessage**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content?` | `string` \| [`RNLlamaMessagePart`](README.md#rnllamamessagepart)[] \| `any` |
| `role` | `string` |

#### Defined in

[chat.ts:7](https://github.com/mybigday/llama.rn/blob/64b02c4/src/chat.ts#L7)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | [`NativeCompletionTokenProb`](README.md#nativecompletiontokenprob)[] |
| `token` | `string` |

#### Defined in

[index.ts:75](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L75)

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

[index.ts:372](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L372)

___

### convertJsonSchemaToGrammar

▸ **convertJsonSchemaToGrammar**(`«destructured»`): `string` \| `Promise`<`string`\>

#### Parameters

| Name | Type |
| :------ | :------ |
| `«destructured»` | `Object` |
| › `allowFetch?` | `boolean` |
| › `dotall?` | `boolean` |
| › `propOrder?` | [`SchemaGrammarConverterPropOrder`](interfaces/SchemaGrammarConverterPropOrder.md) |
| › `schema` | `any` |

#### Returns

`string` \| `Promise`<`string`\>

#### Defined in

[grammar.ts:829](https://github.com/mybigday/llama.rn/blob/64b02c4/src/grammar.ts#L829)

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

[index.ts:412](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L412)

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

[index.ts:397](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L397)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:478](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L478)

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

[index.ts:383](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L383)

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

[index.ts:368](https://github.com/mybigday/llama.rn/blob/64b02c4/src/index.ts#L368)
