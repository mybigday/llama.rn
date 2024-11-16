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
} from './NativeRNLlama'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'
import type { RNLlamaOAICompatibleMessage } from './chat'
import { formatChat } from './chat'

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

export type ContextParams = Omit<NativeContextParams, 'pooling_type'> & {
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
}

export type EmbeddingParams = NativeEmbeddingParams

export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> & {
  prompt?: string
  messages?: RNLlamaOAICompatibleMessage[]
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

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  model: {
    isChatTemplateSupported?: boolean
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
  ): Promise<string> {
    const chat = formatChat(messages)
    return RNLlama.getFormattedChat(
      this.id,
      chat,
      this.model?.isChatTemplateSupported ? undefined : 'chatml',
    )
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    let finalPrompt = params.prompt
    if (params.messages) {
      // messages always win
      finalPrompt = await this.getFormattedChat(params.messages)
    }

    let tokenListener: any =
      callback &&
      EventEmitter.addListener(EVENT_ON_TOKEN, (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      })

    if (!finalPrompt) throw new Error('Prompt is required')
    const promise = RNLlama.completion(this.id, {
      ...params,
      prompt: finalPrompt,
      emit_partial_completion: !!callback,
    })
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
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
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
  } = await RNLlama.initContext(contextId, {
    model: path,
    is_model_asset: !!isModelAsset,
    use_progress_callback: !!onProgress,
    pooling_type: poolType,
    ...rest,
  }).catch((err: any) => {
    removeProgressListener?.remove()
    throw err
  })
  removeProgressListener?.remove()
  return new LlamaContext({ contextId, gpu, reasonNoGPU, model: modelDetails })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}
