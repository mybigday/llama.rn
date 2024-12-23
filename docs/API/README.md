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
- [CompletionParams](README.md#completionparams)
- [ContextParams](README.md#contextparams)
- [EmbeddingParams](README.md#embeddingparams)
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

- [convertJsonSchemaToGrammar](README.md#convertjsonschematogrammar)
- [initLlama](README.md#initllama)
- [loadLlamaModelInfo](README.md#loadllamamodelinfo)
- [releaseAllLlama](README.md#releaseallllama)
- [setContextLimit](README.md#setcontextlimit)

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

[index.ts:103](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L103)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<[`NativeCompletionParams`](README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & { `chatTemplate?`: `string` ; `messages?`: [`RNLlamaOAICompatibleMessage`](README.md#rnllamaoaicompatiblemessage)[] ; `prompt?`: `string`  }

#### Defined in

[index.ts:94](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L94)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<[`NativeContextParams`](README.md#nativecontextparams), ``"cache_type_k"`` \| ``"cache_type_v"`` \| ``"pooling_type"``\> & { `cache_type_k?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `cache_type_v?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:67](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L67)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: [`NativeEmbeddingParams`](README.md#nativeembeddingparams)

#### Defined in

[index.ts:92](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L92)

___

### NativeCompletionParams

Ƭ **NativeCompletionParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `dry_allowed_length?` | `number` | Tokens that extend repetition beyond this receive exponentially increasing penalty: multiplier * base ^ (length of repeating sequence before token - allowed length). Default: `2` |
| `dry_base?` | `number` | Set the DRY repetition penalty base value. Default: `1.75` |
| `dry_multiplier?` | `number` | Set the DRY (Don't Repeat Yourself) repetition penalty multiplier. Default: `0.0`, which is disabled. |
| `dry_penalty_last_n?` | `number` | How many tokens to scan for repetitions. Default: `-1`, where `0` is disabled and `-1` is context size. |
| `dry_sequence_breakers?` | `string`[] | Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted. Default: `['\n', ':', '"', '*']` |
| `emit_partial_completion` | `boolean` | - |
| `grammar?` | `string` | Set grammar for grammar-based sampling. Default: no grammar |
| `ignore_eos?` | `boolean` | Ignore end of stream token and continue generating. Default: `false` |
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

[NativeRNLlama.ts:60](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L60)

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

[NativeRNLlama.ts:201](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L201)

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

[NativeRNLlama.ts:190](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L190)

___

### NativeCompletionTokenProb

Ƭ **NativeCompletionTokenProb**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content` | `string` |
| `probs` | [`NativeCompletionTokenProbItem`](README.md#nativecompletiontokenprobitem)[] |

#### Defined in

[NativeRNLlama.ts:185](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L185)

___

### NativeCompletionTokenProbItem

Ƭ **NativeCompletionTokenProbItem**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prob` | `number` |
| `tok_str` | `string` |

#### Defined in

[NativeRNLlama.ts:180](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L180)

___

### NativeContextParams

Ƭ **NativeContextParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `cache_type_k?` | `string` | KV cache data type for the K (Experimental in llama.cpp) |
| `cache_type_v?` | `string` | KV cache data type for the V (Experimental in llama.cpp) |
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
| `n_gpu_layers?` | `number` | - |
| `n_threads?` | `number` | - |
| `pooling_type?` | `number` | - |
| `rope_freq_base?` | `number` | - |
| `rope_freq_scale?` | `number` | - |
| `use_mlock?` | `boolean` | - |
| `use_mmap?` | `boolean` | - |
| `use_progress_callback?` | `boolean` | - |
| `vocab_only?` | `boolean` | - |

#### Defined in

[NativeRNLlama.ts:8](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L8)

___

### NativeEmbeddingParams

Ƭ **NativeEmbeddingParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embd_normalize?` | `number` |

#### Defined in

[NativeRNLlama.ts:4](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L4)

___

### NativeEmbeddingResult

Ƭ **NativeEmbeddingResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embedding` | `number`[] |

#### Defined in

[NativeRNLlama.ts:221](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L221)

___

### NativeLlamaContext

Ƭ **NativeLlamaContext**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `contextId` | `number` |
| `gpu` | `boolean` |
| `model` | `Object` |
| `reasonNoGPU` | `string` |

#### Defined in

[NativeRNLlama.ts:225](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L225)

___

### NativeSessionLoadResult

Ƭ **NativeSessionLoadResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prompt` | `string` |
| `tokens_loaded` | `number` |

#### Defined in

[NativeRNLlama.ts:232](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L232)

___

### NativeTokenizeResult

Ƭ **NativeTokenizeResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `tokens` | `number`[] |

#### Defined in

[NativeRNLlama.ts:217](https://github.com/mybigday/llama.rn/blob/402a590/src/NativeRNLlama.ts#L217)

___

### RNLlamaMessagePart

Ƭ **RNLlamaMessagePart**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `text?` | `string` |

#### Defined in

[chat.ts:3](https://github.com/mybigday/llama.rn/blob/402a590/src/chat.ts#L3)

___

### RNLlamaOAICompatibleMessage

Ƭ **RNLlamaOAICompatibleMessage**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content?` | `string` \| [`RNLlamaMessagePart`](README.md#rnllamamessagepart)[] \| `any` |
| `role` | `string` |

#### Defined in

[chat.ts:7](https://github.com/mybigday/llama.rn/blob/402a590/src/chat.ts#L7)

___

### TokenData

Ƭ **TokenData**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `completion_probabilities?` | [`NativeCompletionTokenProb`](README.md#nativecompletiontokenprob)[] |
| `token` | `string` |

#### Defined in

[index.ts:57](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L57)

## Functions

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

[grammar.ts:826](https://github.com/mybigday/llama.rn/blob/402a590/src/grammar.ts#L826)

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

[index.ts:295](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L295)

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

[index.ts:280](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L280)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:354](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L354)

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

[index.ts:266](https://github.com/mybigday/llama.rn/blob/402a590/src/index.ts#L266)
