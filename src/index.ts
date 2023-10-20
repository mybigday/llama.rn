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
} from './NativeRNLlama'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

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

export type ContextParams = NativeContextParams

export type CompletionParams = Omit<NativeCompletionParams, 'emit_partial_completion'>

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  constructor({
    contextId,
    gpu,
    reasonNoGPU,
  }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.reasonNoGPU = reasonNoGPU
  }

  /**
   * Load cached prompt & completion state from a file.
   */
  async loadSession(filepath: string): Promise<NativeSessionLoadResult> {
    return RNLlama.loadSession(this.id, filepath)
  }

  /**
   * Save current cached prompt & completion state to a file.
   */
  async saveSession(filepath: string, options?: { tokenSize: number }): Promise<number> {
    return RNLlama.saveSession(this.id, filepath, options?.tokenSize || -1)
  }

  async completion(
    params: CompletionParams,
    callback?: (data: TokenData) => void,
  ): Promise<NativeCompletionResult> {
    let tokenListener: any = callback && EventEmitter.addListener(
      EVENT_ON_TOKEN,
      (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      },
    )
    const promise = RNLlama.completion(this.id, {
      ...params,
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

  embedding(text: string): Promise<NativeEmbeddingResult> {
    return RNLlama.embedding(this.id, text)
  }

  async release(): Promise<void> {
    return RNLlama.releaseContext(this.id)
  }
}

export async function setContextLimit(limit: number): Promise<void> {
  return RNLlama.setContextLimit(limit)
}

export async function initLlama({
  model,
  is_model_asset: isModelAsset,
  ...rest
}: ContextParams): Promise<LlamaContext> {
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  const { contextId, gpu, reasonNoGPU } =
    await RNLlama.initContext({
      model: path,
      is_model_asset: !!isModelAsset,
      ...rest,
    })
  return new LlamaContext({ contextId, gpu, reasonNoGPU })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}
