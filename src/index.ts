import { NativeEventEmitter } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import RNLlama from './NativeRNLlama'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
  NativeTokenizeResult,
  NativeEmbeddingResult,
} from './NativeRNLlama'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_TOKEN = '@RNLlama_onToken'

const EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic =
  // @ts-ignore
  new NativeEventEmitter(RNLlama)

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

type TokenNativeEvent = {
  contextId: number
  tokenResult: TokenData
}

export type ContextParams = NativeContextParams

export type CompletionParams = NativeCompletionParams

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

  async completion(
    params: CompletionParams,
    callback: (data: TokenData) => void,
  ) {
    let tokenListener: any = EventEmitter.addListener(
      EVENT_ON_TOKEN,
      (evt: TokenNativeEvent) => {
        const { contextId, tokenResult } = evt
        if (contextId !== this.id) return
        callback(tokenResult)
      },
    )
    const promise = RNLlama.completion(this.id, params)
    return promise
      .then((completionResult) => {
        tokenListener.remove()
        tokenListener = null
        return completionResult
      })
      .catch((err: any) => {
        tokenListener.remove()
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
