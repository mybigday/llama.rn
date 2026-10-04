llama.rn

# llama.rn

## Table of contents

### Classes

- [LlamaContext](classes/LlamaContext.md)
- [LlamaSpeaker](classes/LlamaSpeaker.md)

### Interfaces

- [NeuTTSSpeaker](interfaces/NeuTTSSpeaker.md)
- [OuteTTSSpeaker](interfaces/OuteTTSSpeaker.md)
- [OuteTTSWord](interfaces/OuteTTSWord.md)
- [TTSCapabilities](interfaces/TTSCapabilities.md)

### Type Aliases

- [BenchResult](README.md#benchresult)
- [ChatTemplateKwargs](README.md#chattemplatekwargs)
- [CompletionBaseParams](README.md#completionbaseparams)
- [CompletionParams](README.md#completionparams)
- [CompletionResponseFormat](README.md#completionresponseformat)
- [ContextParams](README.md#contextparams)
- [DecisionAnswer](README.md#decisionanswer)
- [DecisionAnswerOf](README.md#decisionanswerof)
- [DecisionChoiceAnswer](README.md#decisionchoiceanswer)
- [DecisionChoiceQuestion](README.md#decisionchoicequestion)
- [DecisionContent](README.md#decisioncontent)
- [DecisionModelType](README.md#decisionmodeltype)
- [DecisionNoulAnswer](README.md#decisionnoulanswer)
- [DecisionNoulQuestion](README.md#decisionnoulquestion)
- [DecisionQuestion](README.md#decisionquestion)
- [DecisionQuestions](README.md#decisionquestions)
- [DecisionReadout](README.md#decisionreadout)
- [DecisionRequest](README.md#decisionrequest)
- [DecisionResult](README.md#decisionresult)
- [DecisionScoreAnswer](README.md#decisionscoreanswer)
- [DecisionScoreQuestion](README.md#decisionscorequestion)
- [DecisionValue](README.md#decisionvalue)
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
- [NativeDecisionResult](README.md#nativedecisionresult)
- [NativeEmbeddingParams](README.md#nativeembeddingparams)
- [NativeEmbeddingResult](README.md#nativeembeddingresult)
- [NativeImageProcessingResult](README.md#nativeimageprocessingresult)
- [NativeLlamaContext](README.md#nativellamacontext)
- [NativeParallelCompletionParams](README.md#nativeparallelcompletionparams)
- [NativeRerankParams](README.md#nativererankparams)
- [NativeRerankResult](README.md#nativererankresult)
- [NativeSessionLoadResult](README.md#nativesessionloadresult)
- [NativeSpeculativeConfig](README.md#nativespeculativeconfig)
- [NativeSpeculativeParams](README.md#nativespeculativeparams)
- [NativeSpeculativeType](README.md#nativespeculativetype)
- [NativeTokenizeResult](README.md#nativetokenizeresult)
- [ParallelCompletionParams](README.md#parallelcompletionparams)
- [ParallelRequestStatus](README.md#parallelrequeststatus)
- [ParallelStatus](README.md#parallelstatus)
- [RNLlamaMessagePart](README.md#rnllamamessagepart)
- [RNLlamaOAICompatibleMessage](README.md#rnllamaoaicompatiblemessage)
- [RerankParams](README.md#rerankparams)
- [RerankResult](README.md#rerankresult)
- [SpeakerPayload](README.md#speakerpayload)
- [TokenData](README.md#tokendata)
- [ToolCall](README.md#toolcall)

### Variables

- [BuildInfo](README.md#buildinfo)
- [RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER](README.md#rnllama_mtmd_default_media_marker)

### Functions

- [addNativeLogListener](README.md#addnativeloglistener)
- [getBackendDevicesInfo](README.md#getbackenddevicesinfo)
- [getTTSVoice](README.md#getttsvoice)
- [initLlama](README.md#initllama)
- [installJsi](README.md#installjsi)
- [listTTSLanguages](README.md#listttslanguages)
- [listTTSVoices](README.md#listttsvoices)
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

[index.ts:369](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L369)

___

### ChatTemplateKwargs

Ƭ **ChatTemplateKwargs**: `Record`<`string`, `string` \| `number` \| `boolean`\>

#### Defined in

[index.ts:309](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L309)

___

### CompletionBaseParams

Ƭ **CompletionBaseParams**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `add_generation_prompt?` | `boolean` | - |
| `chatTemplate?` | `string` | - |
| `chat_template?` | `string` | - |
| `chat_template_kwargs?` | [`ChatTemplateKwargs`](README.md#chattemplatekwargs) | - |
| `force_pure_content?` | `boolean` | When enabled, forces the chat parser to treat the entire model output as plain content, skipping separate parsing of reasoning tokens and tool calls. Also bypasses jinja template validation so templates that only accept typed content (e.g. TranslateGemma) are not rejected during capability detection. |
| `jinja?` | `boolean` | - |
| `media_paths?` | `string` \| `string`[] | - |
| `messages?` | [`RNLlamaOAICompatibleMessage`](README.md#rnllamaoaicompatiblemessage)[] | - |
| `now?` | `string` \| `number` | - |
| `parallel_tool_calls?` | `boolean` | - |
| `prefill_text?` | `string` | Prefill text to be used for chat parsing (Generation Prompt + Content) Used for if last assistant message is for prefill purpose |
| `prompt?` | `string` | - |
| `response_format?` | [`CompletionResponseFormat`](README.md#completionresponseformat) | - |
| `tool_choice?` | `string` | - |
| `tools?` | `object` | - |

#### Defined in

[index.ts:311](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L311)

___

### CompletionParams

Ƭ **CompletionParams**: `Omit`<[`NativeCompletionParams`](README.md#nativecompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](README.md#completionbaseparams)

#### Defined in

[index.ts:342](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L342)

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

[index.ts:300](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L300)

___

### ContextParams

Ƭ **ContextParams**: `Omit`<[`NativeContextParams`](README.md#nativecontextparams), ``"flash_attn_type"`` \| ``"cache_type_k"`` \| ``"cache_type_v"`` \| ``"pooling_type"``\> & { `cache_type_k?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `cache_type_v?`: ``"f16"`` \| ``"f32"`` \| ``"q8_0"`` \| ``"q4_0"`` \| ``"q4_1"`` \| ``"iq4_nl"`` \| ``"q5_0"`` \| ``"q5_1"`` ; `flash_attn_type?`: ``"auto"`` \| ``"on"`` \| ``"off"`` ; `pooling_type?`: ``"none"`` \| ``"mean"`` \| ``"cls"`` \| ``"last"`` \| ``"rank"``  }

#### Defined in

[index.ts:250](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L250)

___

### DecisionAnswer

Ƭ **DecisionAnswer**: [`DecisionChoiceAnswer`](README.md#decisionchoiceanswer) \| [`DecisionScoreAnswer`](README.md#decisionscoreanswer) \| [`DecisionNoulAnswer`](README.md#decisionnoulanswer)

#### Defined in

[types.ts:685](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L685)

___

### DecisionAnswerOf

Ƭ **DecisionAnswerOf**<`Q`\>: `Q` extends [`DecisionChoiceQuestion`](README.md#decisionchoicequestion)<infer K\> ? [`DecisionChoiceAnswer`](README.md#decisionchoiceanswer)<`K`\> : `Q` extends [`DecisionScoreQuestion`](README.md#decisionscorequestion) ? [`DecisionScoreAnswer`](README.md#decisionscoreanswer) : [`DecisionNoulAnswer`](README.md#decisionnoulanswer)

The answer type of a question type, choice keys included.

#### Type parameters

| Name | Type |
| :------ | :------ |
| `Q` | extends [`DecisionQuestion`](README.md#decisionquestion) |

#### Defined in

[types.ts:691](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L691)

___

### DecisionChoiceAnswer

Ƭ **DecisionChoiceAnswer**<`K`\>: `Object`

#### Type parameters

| Name | Type |
| :------ | :------ |
| `K` | extends `string` = `string` |

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `choice` | `K` | The option with the highest probability. |
| `confidence` | `number` | 0 when all options are equally likely, 1 when one option has all the mass. |
| `probabilities` | { [key in K]: number } | Option to its probability, they sum to 1. Look an option up by its key: the key order is not guaranteed. |
| `type` | ``"choice"`` | - |

#### Defined in

[types.ts:655](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L655)

___

### DecisionChoiceQuestion

Ƭ **DecisionChoiceQuestion**<`K`\>: `Object`

#### Type parameters

| Name | Type |
| :------ | :------ |
| `K` | extends `string` = `string` |

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `criteria` | { [key in K]: string \| null } | Maps each option to its description, in the order the options are shown. That is the order JavaScript enumerates the keys in: keys that are whole numbers (`'1'`, `'10'`) come first, in ascending order, wherever they were written. Give such options a prefix (`'n1'`) to keep them where they are. |
| `instructions` | [`DecisionContent`](README.md#decisioncontent) | - |
| `type` | ``"choice"`` | - |

#### Defined in

[types.ts:605](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L605)

___

### DecisionContent

Ƭ **DecisionContent**: `string` \| readonly [`DecisionValue`](README.md#decisionvalue)[] \| { `[key: string]`: [`DecisionValue`](README.md#decisionvalue);  }

A string, an object or an array. A value that is not a string is given to the model as JSON text.

#### Defined in

[types.ts:600](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L600)

___

### DecisionModelType

Ƭ **DecisionModelType**: ``"openjev"`` \| ``"lev"`` \| ``"kev"`` \| ``"nimble"`` \| ``"laya"`` \| ``"clef"`` \| ``"system_one"`` \| ``"unknown"``

The readout a decision model declares in `<arch>.decision.type`.
`system_one`: an older model that declares no type but carries a `system_one`
template; its readout is derived from the model, see `readout`.
`unknown`: the model is a decision model this build cannot serve, see `error`.

#### Defined in

[types.ts:572](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L572)

___

### DecisionNoulAnswer

Ƭ **DecisionNoulAnswer**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `noul` | `number` | The probability that the answer is true. |
| `type` | ``"noul"`` | - |

#### Defined in

[types.ts:679](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L679)

___

### DecisionNoulQuestion

Ƭ **DecisionNoulQuestion**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `criteria?` | { `false?`: `string` \| ``null`` ; `true?`: `string` \| ``null``  } |
| `criteria.false?` | `string` \| ``null`` |
| `criteria.true?` | `string` \| ``null`` |
| `instructions` | [`DecisionContent`](README.md#decisioncontent) |
| `type` | ``"noul"`` |

#### Defined in

[types.ts:624](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L624)

___

### DecisionQuestion

Ƭ **DecisionQuestion**: [`DecisionChoiceQuestion`](README.md#decisionchoicequestion) \| [`DecisionScoreQuestion`](README.md#decisionscorequestion) \| [`DecisionNoulQuestion`](README.md#decisionnoulquestion)

#### Defined in

[types.ts:630](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L630)

___

### DecisionQuestions

Ƭ **DecisionQuestions**: `Record`<`string`, [`DecisionQuestion`](README.md#decisionquestion)\>

#### Defined in

[types.ts:635](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L635)

___

### DecisionReadout

Ƭ **DecisionReadout**: ``"letter_slot"`` \| ``"rank_head"``

How a `system_one` model is read:
`letter_slot`: a causal model, every question answered from one prompt.
`rank_head`: a classification head scoring one prompt per option. The context
must be initialized with `pooling_type: 'rank'` and `embedding: true`.

#### Defined in

[types.ts:588](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L588)

___

### DecisionRequest

Ƭ **DecisionRequest**<`Q`\>: `Object`

#### Type parameters

| Name | Type |
| :------ | :------ |
| `Q` | extends [`DecisionQuestions`](README.md#decisionquestions) = [`DecisionQuestions`](README.md#decisionquestions) |

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `images?` | `string`[] | Images, as file paths or data URLs. Needs a model that supports image input and `initMultimodal()`. All images go before the state, these first. |
| `questions` | `Q` | - |
| `state` | [`DecisionContent`](README.md#decisioncontent) | The content to evaluate. A state made of chat messages (an array of messages, or an object with a `messages` array) may carry `image_url` parts, they are taken as images and removed from the state. For a `system_one` model the state may also be a list of content parts (`{ type: 'text', text }`, `{ type: 'image_url', image_url: { url } }`): the text is joined and each image stays where it is. |

#### Defined in

[types.ts:637](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L637)

___

### DecisionResult

Ƭ **DecisionResult**<`Q`\>: `Object`

#### Type parameters

| Name | Type |
| :------ | :------ |
| `Q` | extends [`DecisionQuestions`](README.md#decisionquestions) = [`DecisionQuestions`](README.md#decisionquestions) |

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `answers` | { [id in keyof Q]: DecisionAnswerOf<Q[id]\> } | - |
| `model` | `string` | `general.name` of the model, or its file name if it has none |
| `usage` | { `input_tokens`: `number` ; `output_tokens`: `number`  } | - |
| `usage.input_tokens` | `number` | Prompt tokens of all the questions |
| `usage.output_tokens` | `number` | Always 0 |

#### Defined in

[types.ts:698](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L698)

___

### DecisionScoreAnswer

Ƭ **DecisionScoreAnswer**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `confidence` | `number` | - |
| `legend` | `Record`<`string`, `string`\> | Level index to its description. |
| `probabilities` | `Record`<`string`, `number`\> | Level index to its probability, they sum to 1. |
| `score` | `number` | The expected level index, weighted by probability. Can be between two levels. |
| `type` | ``"score"`` | - |

#### Defined in

[types.ts:668](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L668)

___

### DecisionScoreQuestion

Ƭ **DecisionScoreQuestion**: `Object`

#### Type declaration

| Name | Type | Description |
| :------ | :------ | :------ |
| `criteria` | readonly `string`[] | 2 to 10 level descriptions, lowest level first. |
| `instructions` | [`DecisionContent`](README.md#decisioncontent) | - |
| `type` | ``"score"`` | - |

#### Defined in

[types.ts:617](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L617)

___

### DecisionValue

Ƭ **DecisionValue**: `string` \| `number` \| `boolean` \| ``null`` \| readonly [`DecisionValue`](README.md#decisionvalue)[] \| { `[key: string]`: [`DecisionValue`](README.md#decisionvalue);  }

Any JSON value

#### Defined in

[types.ts:591](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L591)

___

### EmbeddingParams

Ƭ **EmbeddingParams**: [`NativeEmbeddingParams`](README.md#nativeembeddingparams)

#### Defined in

[index.ts:288](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L288)

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

[types.ts:791](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L791)

___

### JinjaFormattedChatResult

Ƭ **JinjaFormattedChatResult**: [`FormattedChatResult`](README.md#formattedchatresult) & { `additional_stops?`: `string`[] ; `chat_format?`: `number` ; `chat_parser?`: `string` ; `generation_prompt?`: `string` ; `grammar?`: `string` ; `grammar_lazy?`: `boolean` ; `grammar_triggers?`: { `token`: `number` ; `type`: `number` ; `value`: `string`  }[] ; `preserved_tokens?`: `string`[] ; `thinking_end_tag?`: `string` ; `thinking_forced_open?`: `boolean` ; `thinking_start_tag?`: `string`  }

#### Defined in

[types.ts:798](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L798)

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

[types.ts:858](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L858)

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
| `embedding?` | `boolean` | Output token embeddings during generation. When enabled, completion results include generated token embeddings and their dimension. Default: `false` |
| `emit_partial_completion` | `boolean` | - |
| `enable_thinking?` | `boolean` | Enable thinking if jinja is enabled. Default: true |
| `generation_prompt?` | `string` | Assistant generation prompt returned by jinja chat formatting. Used for PEG chat parsing and grammar prefill. |
| `grammar?` | `string` | Set grammar for grammar-based sampling. Default: no grammar |
| `grammar_lazy?` | `boolean` | Lazy grammar sampling, trigger by grammar_triggers. Default: false |
| `grammar_triggers?` | { `token`: `number` ; `type`: `number` ; `value`: `string`  }[] | Lazy grammar triggers. Default: [] |
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
| `n_probs?` | `number` | If greater than 0, the response also contains the probabilities of top N tokens for each generated token. By default these are the sampler chain's candidate probabilities, i.e. after `top_k` / `top_p` / `min_p` / `temperature` / `grammar` / ... have been applied and renormalised. Set `post_sampling_probs: false` to get a plain softmax of the raw logits instead. Default: `0` |
| `n_threads?` | `number` | - |
| `penalty_freq?` | `number` | Repeat alpha frequency penalty. Default: `0.0`, which is disabled. |
| `penalty_last_n?` | `number` | Last n tokens to consider for penalizing repetition. Default: `64`, where `0` is disabled and `-1` is ctx-size. |
| `penalty_present?` | `number` | Repeat alpha presence penalty. Default: `0.0`, which is disabled. |
| `penalty_repeat?` | `number` | Control the repetition of token sequences in the generated text. Default: `1.0` |
| `post_sampling_probs?` | `boolean` | Controls what `n_probs` reports. `true`: probabilities from the sampler chain's candidates (post-sampling). `false`: softmax of the raw logits, ignoring every sampler setting. Use this for calibrated readouts / thresholds. Default: `true` |
| `preserved_tokens?` | `string`[] | - |
| `prompt` | `string` | - |
| `reasoning_format?` | ``"none"`` \| ``"auto"`` \| ``"deepseek"`` | - |
| `seed?` | `number` | Set the random number generator (RNG) seed. Default: `-1`, which is a random seed. |
| `spec_draft_n_max?` | `number` | - |
| `spec_draft_n_min?` | `number` | - |
| `spec_draft_p_min?` | `number` | - |
| `spec_draft_p_split?` | `number` | - |
| `spec_type?` | [`NativeSpeculativeType`](README.md#nativespeculativetype) \| [`NativeSpeculativeType`](README.md#nativespeculativetype)[] | - |
| `speculative?` | [`NativeSpeculativeConfig`](README.md#nativespeculativeconfig) | Per-completion speculative decoding override. For MTP on recurrent/hybrid models, load the model with matching MTP options first. |
| `stop?` | `string`[] | Specify a JSON array of stopping strings. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration. Default: `[]` |
| `temperature?` | `number` | Adjust the randomness of the generated text. Default: `0.8` |
| `thinking_budget_message?` | `string` | Message injected before the thinking end tag when the thinking budget is exhausted. |
| `thinking_budget_tokens?` | `number` | Maximum number of tokens allowed inside a thinking block before forcing it to close. Only applies when chat formatting exposes thinking tags. |
| `thinking_forced_open?` | `boolean` | Force thinking to be open. Default: false |
| `top_k?` | `number` | Limit the next token selection to the K most probable tokens. Default: `40` |
| `top_n_sigma?` | `number` | Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641. Default: `-1.0` (Disabled) |
| `top_p?` | `number` | Limit the next token selection to a subset of tokens with a cumulative probability above a threshold P. Default: `0.95` |
| `typical_p?` | `number` | Enable locally typical sampling with parameter p. Default: `1.0`, which is disabled. |
| `xtc_probability?` | `number` | Set the chance for token removal via XTC sampler. Default: `0.0`, which is disabled. |
| `xtc_threshold?` | `number` | Set a minimum probability threshold for tokens to be removed via XTC sampler. Default: `0.1` (> `0.5` disables XTC) |

#### Defined in

[types.ts:209](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L209)

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
| `draft_tokens` | `number` | - |
| `draft_tokens_accepted` | `number` | - |
| `embedding_dim?` | `number` | - |
| `embeddings?` | `number`[] | - |
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

[types.ts:487](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L487)

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

[types.ts:475](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L475)

___

### NativeCompletionTokenProb

Ƭ **NativeCompletionTokenProb**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `content` | `string` |
| `probs` | [`NativeCompletionTokenProbItem`](README.md#nativecompletiontokenprobitem)[] |

#### Defined in

[types.ts:470](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L470)

___

### NativeCompletionTokenProbItem

Ƭ **NativeCompletionTokenProbItem**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prob` | `number` |
| `tok_id` | `number` |
| `tok_str` | `string` |

#### Defined in

[types.ts:464](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L464)

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
| `devices?` | `string`[] | Backend devices to use. If omitted, llama.rn uses the platform default selection. On Android, Hexagon is opt-in: use `['HTP0']` for one session, select multiple HTP devices explicitly, or use `['HTP*']` for every available HTP session. |
| `draft_model?` | `string` | Alias for model_draft. |
| `embd_normalize?` | `number` | - |
| `embedding?` | `boolean` | - |
| `flash_attn?` | `boolean` | Enable flash attention, only recommended in GPU device Deprecated: use flash_attn_type instead |
| `flash_attn_type?` | `string` | Enable flash attention, only recommended in GPU device. |
| `is_model_asset?` | `boolean` | - |
| `is_model_draft_asset?` | `boolean` | - |
| `kv_unified?` | `boolean` | Use a unified buffer across the input sequences when computing the attention. Try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix. |
| `lora?` | `string` | Single LoRA adapter path |
| `lora_list?` | { `path`: `string` ; `scaled?`: `number`  }[] | LoRA adapter list |
| `lora_scaled?` | `number` | Single LoRA adapter scale |
| `model` | `string` | - |
| `model_draft?` | `string` | Optional separate draft model path for MTP/speculative decoding. Leave unset for hybrid/embedded MTP models such as Qwen MTP. |
| `n_batch?` | `number` | - |
| `n_cpu_moe?` | `number` | Number of layers to keep MoE weights on CPU |
| `n_ctx?` | `number` | - |
| `n_gpu_layers?` | `number` | Number of layers to store in VRAM (Currently only for iOS) |
| `n_parallel?` | `number` | Number of parallel sequences to support (sets n_seq_max). This determines the maximum number of parallel slots that can be used. Default: 8 |
| `n_threads?` | `number` | - |
| `n_ubatch?` | `number` | - |
| `no_extra_bufts?` | `boolean` | Disable extra buffer types for weight repacking. Reduces memory usage at the cost of slower prompt processing. Default: false |
| `no_gpu_devices?` | `boolean` | Skip GPU devices (iOS only) (Deprecated: Please set devices params instead) |
| `pooling_type?` | `number` | - |
| `rope_freq_base?` | `number` | - |
| `rope_freq_scale?` | `number` | - |
| `spec_draft_cache_type_k?` | `string` | - |
| `spec_draft_cache_type_v?` | `string` | - |
| `spec_draft_n_gpu_layers?` | `number` | - |
| `spec_draft_n_max?` | `number` | - |
| `spec_draft_n_min?` | `number` | - |
| `spec_draft_p_min?` | `number` | - |
| `spec_draft_p_split?` | `number` | - |
| `spec_type?` | [`NativeSpeculativeType`](README.md#nativespeculativetype) \| [`NativeSpeculativeType`](README.md#nativespeculativetype)[] | - |
| `speculative?` | [`NativeSpeculativeConfig`](README.md#nativespeculativeconfig) | Enable speculative decoding support at context creation time. MTP on recurrent/hybrid models must be enabled here so llama.cpp can allocate recurrent-state rollback slots. |
| `state_cache_budget_mb?` | `number` | Memory budget (MiB) for the cross-turn KV prefix cache on recurrent/hybrid models. 0 disables it; no-op on pure-attention models. Default 160. |
| `state_cache_max_checkpoints?` | `number` | Max snapshots to keep (secondary cap; the byte budget is primary). 0 = no count cap. Default 8. |
| `swa_full?` | `boolean` | Use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055) |
| `use_mlock?` | `boolean` | - |
| `use_mmap?` | `boolean` | - |
| `use_progress_callback?` | `boolean` | - |
| `vocab_only?` | `boolean` | - |

#### Defined in

[types.ts:45](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L45)

___

### NativeDecisionResult

Ƭ **NativeDecisionResult**: [`DecisionResult`](README.md#decisionresult)

#### Defined in

[types.ts:710](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L710)

___

### NativeEmbeddingParams

Ƭ **NativeEmbeddingParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embd_normalize?` | `number` |

#### Defined in

[types.ts:1](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L1)

___

### NativeEmbeddingResult

Ƭ **NativeEmbeddingResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `embedding` | `number`[] |

#### Defined in

[types.ts:554](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L554)

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

[types.ts:820](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L820)

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
| `model` | { `chatTemplates`: { `jinja`: { `default`: `boolean` ; `defaultCaps`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  } ; `toolUse`: `boolean` ; `toolUseCaps?`: { `parallelToolCalls`: `boolean` ; `systemRole`: `boolean` ; `toolCalls`: `boolean` ; `tools`: `boolean`  }  } ; `llamaChat`: `boolean`  } ; `decision?`: { `error?`: `string` ; `imageInput`: `boolean` ; `nOptionsMax`: `number` ; `readout?`: [`DecisionReadout`](README.md#decisionreadout) ; `textGeneration`: `boolean` ; `type`: [`DecisionModelType`](README.md#decisionmodeltype)  } ; `desc`: `string` ; `isChatTemplateSupported`: `boolean` ; `is_hybrid`: `boolean` ; `is_recurrent`: `boolean` ; `metadata`: `Object` ; `nEmbd`: `number` ; `nParams`: `number` ; `size`: `number`  } | - |
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
| `model.decision?` | { `error?`: `string` ; `imageInput`: `boolean` ; `nOptionsMax`: `number` ; `readout?`: [`DecisionReadout`](README.md#decisionreadout) ; `textGeneration`: `boolean` ; `type`: [`DecisionModelType`](README.md#decisionmodeltype)  } | Set if the model is a typed decision model, see `LlamaContext.decide()` |
| `model.decision.error?` | `string` | Why the model cannot be used, when `type` is `unknown` |
| `model.decision.imageInput` | `boolean` | The prompt has a place for images (multimodal still has to be initialized) |
| `model.decision.nOptionsMax` | `number` | Most options a `choice` question can have |
| `model.decision.readout?` | [`DecisionReadout`](README.md#decisionreadout) | Only for `system_one` models |
| `model.decision.textGeneration` | `boolean` | If false, `completion()` rejects: the model only answers decisions (clef, for one) |
| `model.decision.type` | [`DecisionModelType`](README.md#decisionmodeltype) | - |
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

[types.ts:715](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L715)

___

### NativeParallelCompletionParams

Ƭ **NativeParallelCompletionParams**: [`NativeCompletionParams`](README.md#nativecompletionparams) & { `load_state_path?`: `string` ; `load_state_size?`: `number` ; `save_prompt_state_path?`: `string` ; `save_state_path?`: `string` ; `save_state_size?`: `number`  }

Parameters for parallel completion requests (queueCompletion).
Extends NativeCompletionParams with parallel-mode specific options.

#### Defined in

[types.ts:421](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L421)

___

### NativeRerankParams

Ƭ **NativeRerankParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `normalize?` | `number` |

#### Defined in

[types.ts:826](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L826)

___

### NativeRerankResult

Ƭ **NativeRerankResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `index` | `number` |
| `score` | `number` |

#### Defined in

[types.ts:830](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L830)

___

### NativeSessionLoadResult

Ƭ **NativeSessionLoadResult**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `prompt` | `string` |
| `tokens_loaded` | `number` |

#### Defined in

[types.ts:776](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L776)

___

### NativeSpeculativeConfig

Ƭ **NativeSpeculativeConfig**: [`NativeSpeculativeParams`](README.md#nativespeculativeparams) \| [`NativeSpeculativeType`](README.md#nativespeculativetype) \| `boolean`

#### Defined in

[types.ts:40](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L40)

___

### NativeSpeculativeParams

Ƭ **NativeSpeculativeParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `draft?` | { `cache_type_k?`: `string` ; `cache_type_v?`: `string` ; `draft_model?`: `string` ; `model?`: `string` ; `model_draft?`: `string` ; `n_gpu_layers?`: `number` ; `n_max?`: `number` ; `n_min?`: `number` ; `p_min?`: `number` ; `p_split?`: `number` ; `path?`: `string`  } |
| `draft.cache_type_k?` | `string` |
| `draft.cache_type_v?` | `string` |
| `draft.draft_model?` | `string` |
| `draft.model?` | `string` |
| `draft.model_draft?` | `string` |
| `draft.n_gpu_layers?` | `number` |
| `draft.n_max?` | `number` |
| `draft.n_min?` | `number` |
| `draft.p_min?` | `number` |
| `draft.p_split?` | `number` |
| `draft.path?` | `string` |
| `enabled?` | `boolean` |
| `n_max?` | `number` |
| `n_min?` | `number` |
| `p_min?` | `number` |
| `p_split?` | `number` |
| `type?` | [`NativeSpeculativeType`](README.md#nativespeculativetype) |
| `types?` | [`NativeSpeculativeType`](README.md#nativespeculativetype)[] |

#### Defined in

[types.ts:13](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L13)

___

### NativeSpeculativeType

Ƭ **NativeSpeculativeType**: ``"none"`` \| ``"draft-mtp"`` \| ``"mtp"``

#### Defined in

[types.ts:5](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L5)

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

[types.ts:534](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L534)

___

### ParallelCompletionParams

Ƭ **ParallelCompletionParams**: `Omit`<[`NativeParallelCompletionParams`](README.md#nativeparallelcompletionparams), ``"emit_partial_completion"`` \| ``"prompt"``\> & [`CompletionBaseParams`](README.md#completionbaseparams)

Parameters for parallel completion requests.
Extends CompletionParams with parallel-mode specific options like state management.

#### Defined in

[index.ts:352](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L352)

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
| `type` | ``"completion"`` \| ``"embedding"`` \| ``"rerank"`` \| ``"decision"`` |

#### Defined in

[types.ts:866](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L866)

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

[types.ts:877](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/types.ts#L877)

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

[index.ts:50](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L50)

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

[index.ts:63](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L63)

___

### RerankParams

Ƭ **RerankParams**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `normalize?` | `number` |

#### Defined in

[index.ts:290](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L290)

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

[index.ts:294](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L294)

___

### SpeakerPayload

Ƭ **SpeakerPayload**: [`OuteTTSSpeaker`](interfaces/OuteTTSSpeaker.md) \| [`NeuTTSSpeaker`](interfaces/NeuTTSSpeaker.md) \| { `[k: string]`: `any`; `text?`: `string`  }

#### Defined in

[tts-voices.ts:25](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/tts-voices.ts#L25)

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

[index.ts:239](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L239)

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

[index.ts:230](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L230)

## Variables

### BuildInfo

• `Const` **BuildInfo**: `Object`

#### Type declaration

| Name | Type |
| :------ | :------ |
| `commit` | `string` |
| `number` | `string` |

#### Defined in

[index.ts:1700](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1700)

___

### RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER

• `Const` **RNLLAMA\_MTMD\_DEFAULT\_MEDIA\_MARKER**: ``"<__media__>"``

#### Defined in

[index.ts:112](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L112)

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

[index.ts:1438](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1438)

___

### getBackendDevicesInfo

▸ **getBackendDevicesInfo**(): `Promise`<[`NativeBackendDeviceInfo`](README.md#nativebackenddeviceinfo)[]\>

#### Returns

`Promise`<[`NativeBackendDeviceInfo`](README.md#nativebackenddeviceinfo)[]\>

#### Defined in

[index.ts:1555](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1555)

___

### getTTSVoice

▸ **getTTSVoice**(`family`, `name`, `language?`): [`SpeakerPayload`](README.md#speakerpayload) \| ``null``

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `family` | `string` | `undefined` |
| `name` | `string` | `undefined` |
| `language` | `string` | `DEFAULT_LANGUAGE` |

#### Returns

[`SpeakerPayload`](README.md#speakerpayload) \| ``null``

#### Defined in

[tts-voices.ts:122](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/tts-voices.ts#L122)

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

[index.ts:1571](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1571)

___

### installJsi

▸ **installJsi**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:218](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L218)

___

### listTTSLanguages

▸ **listTTSLanguages**(`family`): `string`[]

#### Parameters

| Name | Type |
| :------ | :------ |
| `family` | `string` |

#### Returns

`string`[]

#### Defined in

[tts-voices.ts:137](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/tts-voices.ts#L137)

___

### listTTSVoices

▸ **listTTSVoices**(`family`, `language?`): `string`[]

#### Parameters

| Name | Type | Default value |
| :------ | :------ | :------ |
| `family` | `string` | `undefined` |
| `language` | `string` | `DEFAULT_LANGUAGE` |

#### Returns

`string`[]

#### Defined in

[tts-voices.ts:130](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/tts-voices.ts#L130)

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

[index.ts:1466](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1466)

___

### releaseAllLlama

▸ **releaseAllLlama**(): `Promise`<`void`\>

#### Returns

`Promise`<`void`\>

#### Defined in

[index.ts:1694](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1694)

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

[index.ts:1449](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1449)

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

[index.ts:1432](https://github.com/mybigday/llama.rn/blob/3ee457d5/src/index.ts#L1432)
