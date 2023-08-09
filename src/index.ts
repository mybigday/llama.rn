import { NativeEventEmitter } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import RNLlama from './NativeRNLlama'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeCompletionTokenProb,
} from './NativeRNLlama'

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

  isMetalEnabled: boolean = false

  reasonNoMetal: string = ''

  constructor({
    contextId,
    isMetalEnabled,
    reasonNoMetal,
  }: NativeLlamaContext) {
    this.id = contextId
    this.isMetalEnabled = isMetalEnabled
    this.reasonNoMetal = reasonNoMetal
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
  const { contextId, isMetalEnabled, reasonNoMetal } =
    await RNLlama.initContext({
      model: path,
      is_model_asset: !!isModelAsset,
      ...rest,
    })
  return new LlamaContext({ contextId, isMetalEnabled, reasonNoMetal })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}
