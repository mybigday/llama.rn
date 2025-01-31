import { NativeEventEmitter, DeviceEventEmitter, Platform } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import RNLlama from './NativeRNLlama'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
} from './NativeRNLlama'
import type {
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
} from './grammar'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'
import type { RNLlamaMessagePart, RNLlamaOAICompatibleMessage } from './chat'
import { formatChat } from './chat'

export type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  RNLlamaMessagePart,
  RNLlamaOAICompatibleMessage,
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
}

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_INIT_CONTEXT_PROGRESS = '@RNLlama_onInitContextProgress'
const EVENT_ON_TOKEN = '@RNLlama_onToken'

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(RNLlama)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

type TokenNativeEvent = {
  contextId: number
  tokenResult: TokenData
}

export type ContextParams = Omit<
  NativeContextParams,
  'cache_type_k' | 'cache_type_v' | 'pooling_type'
> & {
  cache_type_k?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  cache_type_v?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
}

export type EmbeddingParams = NativeEmbeddingParams

export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> & {
  prompt?: string
  messages?: RNLlamaOAICompatibleMessage[]
  chatTemplate?: string
  jinja?: boolean
  tools?: object
  parallel_tool_calls?: object
  tool_choice?: string
}

export type BenchResult = {
  modelDesc: string
  modelSize: number
  modelNParams: number
  ppAvg: number
  ppStd: number
  tgAvg: number
  tgStd: number
}

type JinjaFormattedChatResult = {
  prompt: string
  chat_format?: number
  grammar?: string
  grammar_lazy?: boolean
  grammar_triggers?: Array<{
    at_start: boolean
    word: string
  }>
  additional_stops?: Array<string>
}

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  model: {
    isChatTemplateSupported?: boolean
    isJinjaChatTemplateSupported?: boolean
  } = {}

  constructor({ contextId, gpu, reasonNoGPU, model }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
    this.model = model
  }

  /**
   * Load cached prompt & completion state from a file.
   */
  async loadSession(filepath: string): Promise<NativeSessionLoadResult> {
    let path = filepath
    if (path.startsWith('file://')) path = path.slice(7)
    return RNLlama.loadSession(this.id, path)
  }

  /**
   * Save current cached prompt & completion state to a file.
   */
  async saveSession(
    filepath: string,
    options?: { tokenSize: number },
  ): Promise<number> {
    return RNLlama.saveSession(this.id, filepath, options?.tokenSize || -1)
  }

  async getFormattedChat(
    messages: RNLlamaOAICompatibleMessage[],
    template?: string,
    params?: {
      jinja?: boolean
      tools?: object
      parallel_tool_calls?: object
      tool_choice?: string
    },
  ): Promise<JinjaFormattedChatResult | string> {
    const chat = formatChat(messages)
    let tmpl = this.model?.isChatTemplateSupported ? undefined : 'chatml'
    if (template) tmpl = template // Force replace if provided
    return RNLlama.getFormattedChat(this.id, JSON.stringify(chat), tmpl, {
      jinja: params?.jinja || false,
      tools: params?.tools ? JSON.stringify(params.tools) : undefined,
      parallel_tool_calls: params?.parallel_tool_calls
        ? JSON.stringify(params.parallel_tool_calls)
        : undefined,
      tool_choice: params?.tool_choice,
    })
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    const nativeParams = {
      ...params,
      prompt: params.prompt || '',
      emit_partial_completion: !!callback,
    }
    if (params.messages) {
      // messages always win
      const formattedResult = await this.getFormattedChat(
        params.messages,
        params.chatTemplate,
        {
          jinja: this.model?.isJinjaChatTemplateSupported && params.jinja,
          tools: params.tools,
          parallel_tool_calls: params.parallel_tool_calls,
          tool_choice: params.tool_choice,
        },
      )
      if (typeof formattedResult === 'string') {
        nativeParams.prompt = formattedResult || ''
      } else {
        nativeParams.prompt = formattedResult.prompt || ''
        if (typeof formattedResult.chat_format === 'number')
          nativeParams.chat_format = formattedResult.chat_format
        if (formattedResult.grammar)
          nativeParams.grammar = formattedResult.grammar
        if (formattedResult.grammar_lazy)
          nativeParams.grammar_lazy = formattedResult.grammar_lazy
        if (formattedResult.grammar_triggers)
          nativeParams.grammar_triggers = formattedResult.grammar_triggers
        if (formattedResult.additional_stops) {
          if (!nativeParams.stop) nativeParams.stop = []
          nativeParams.stop.push(...formattedResult.additional_stops)
        }
      }
    } else {
      nativeParams.prompt = params.prompt || ''
    }

    let tokenListener: any =
      callback &&
      EventEmitter.addListener(EVENT_ON_TOKEN, (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      })

    if (!nativeParams.prompt) throw new Error('Prompt is required')

    const promise = RNLlama.completion(this.id, nativeParams)
    return promise
      .then((completionResult) => {
        tokenListener?.remove()
        tokenListener = null
        return completionResult
      })
      .catch((err: any) => {
        tokenListener?.remove()
        tokenListener = null
        throw err
      })
  }

  stopCompletion(): Promise<void> {
    return RNLlama.stopCompletion(this.id)
  }

  tokenize(text: string): Promise<NativeTokenizeResult> {
    return RNLlama.tokenize(this.id, text)
  }

  detokenize(tokens: number[]): Promise<string> {
    return RNLlama.detokenize(this.id, tokens)
  }

  embedding(
    text: string,
    params?: EmbeddingParams,
  ): Promise<NativeEmbeddingResult> {
    return RNLlama.embedding(this.id, text, params || {})
  }

  async bench(
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ): Promise<BenchResult> {
    const result = await RNLlama.bench(this.id, pp, tg, pl, nr)
    const [modelDesc, modelSize, modelNParams, ppAvg, ppStd, tgAvg, tgStd] =
      JSON.parse(result)
    return {
      modelDesc,
      modelSize,
      modelNParams,
      ppAvg,
      ppStd,
      tgAvg,
      tgStd,
    }
  }

  async applyLoraAdapters(
    loraList: Array<{ path: string; scaled?: number }>,
  ): Promise<void> {
    let loraAdapters: Array<{ path: string; scaled?: number }> = []
    if (loraList)
      loraAdapters = loraList.map((l) => ({
        path: l.path.replace(/file:\/\//, ''),
        scaled: l.scaled,
      }))
    return RNLlama.applyLoraAdapters(this.id, loraAdapters)
  }

  async removeLoraAdapters(): Promise<void> {
    return RNLlama.removeLoraAdapters(this.id)
  }

  async getLoadedLoraAdapters(): Promise<
    Array<{ path: string; scaled?: number }>
  > {
    return RNLlama.getLoadedLoraAdapters(this.id)
  }

  async release(): Promise<void> {
    return RNLlama.releaseContext(this.id)
  }
}

export async function setContextLimit(limit: number): Promise<void> {
  return RNLlama.setContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  // Large fields
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
]
export async function loadLlamaModelInfo(model: string): Promise<Object> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  return RNLlama.modelInfo(path, modelInfoSkip)
}

const poolTypeMap = {
  // -1 is unspecified as undefined
  none: 0,
  mean: 1,
  cls: 2,
  last: 3,
  rank: 4,
}

export async function initLlama(
  {
    model,
    is_model_asset: isModelAsset,
    pooling_type: poolingType,
    lora,
    lora_list: loraList,
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)

  let loraPath = lora
  if (loraPath?.startsWith('file://')) loraPath = loraPath.slice(7)

  let loraAdapters: Array<{ path: string; scaled?: number }> = []
  if (loraList)
    loraAdapters = loraList.map((l) => ({
      path: l.path.replace(/file:\/\//, ''),
      scaled: l.scaled,
    }))

  const contextId = contextIdCounter + contextIdRandom()
  contextIdCounter += 1

  let removeProgressListener: any = null
  if (onProgress) {
    removeProgressListener = EventEmitter.addListener(
      EVENT_ON_INIT_CONTEXT_PROGRESS,
      (evt: { contextId: number; progress: number }) => {
        if (evt.contextId !== contextId) return
        onProgress(evt.progress)
      },
    )
  }

  const poolType = poolTypeMap[poolingType as keyof typeof poolTypeMap]
  const {
    gpu,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
  } = await RNLlama.initContext(contextId, {
    model: path,
    is_model_asset: !!isModelAsset,
    use_progress_callback: !!onProgress,
    pooling_type: poolType,
    lora: loraPath,
    lora_list: loraAdapters,
    ...rest,
  }).catch((err: any) => {
    removeProgressListener?.remove()
    throw err
  })
  removeProgressListener?.remove()
  return new LlamaContext({
    contextId,
    gpu,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
  })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}
