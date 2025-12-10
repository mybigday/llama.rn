/* eslint-disable no-var */
import type {
  NativeContextParams,
  NativeCompletionParams,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeRerankResult,
  JinjaFormattedChatResult,
} from './types'

declare global {
  var llamaInitContext: (
    contextId: number,
    params: NativeContextParams,
    onProgress?: (progress: number) => void,
  ) => Promise<any>
  var llamaReleaseContext: (contextId: number) => Promise<void>
  var llamaReleaseAllContexts: () => Promise<void>
  var llamaModelInfo: (path: string, skip: string[]) => Promise<object>
  var llamaGetBackendDevicesInfo: () => Promise<string>
  var llamaLoadSession: (
    contextId: number,
    path: string,
  ) => Promise<NativeSessionLoadResult>
  var llamaSaveSession: (
    contextId: number,
    path: string,
    size: number,
  ) => Promise<number>
  var llamaTokenize: (
    contextId: number,
    text: string,
    mediaPaths?: string[],
  ) => Promise<NativeTokenizeResult>
  var llamaDetokenize: (contextId: number, tokens: number[]) => Promise<string>
  var llamaGetFormattedChat: (
    contextId: number,
    messages: string,
    chatTemplate?: string,
    params?: object,
  ) => Promise<string | JinjaFormattedChatResult>
  var llamaEmbedding: (
    contextId: number,
    text: string,
    params: object,
  ) => Promise<NativeEmbeddingResult>
  var llamaRerank: (
    contextId: number,
    query: string,
    documents: string[],
    params: object,
  ) => Promise<NativeRerankResult[]>
  var llamaBench: (
    contextId: number,
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ) => Promise<string>
  var llamaToggleNativeLog: (
    enabled: boolean,
    onLog?: (level: string, text: string) => void,
  ) => Promise<void>
  var llamaSetContextLimit: (limit: number) => Promise<void>
  var llamaCompletion: (
    contextId: number,
    params: NativeCompletionParams,
    onToken?: (token: any) => void,
  ) => Promise<NativeCompletionResult>
  var llamaStopCompletion: (contextId: number) => Promise<void>
  var llamaApplyLoraAdapters: (
    contextId: number,
    adapters: Array<{ path: string; scaled?: number }>,
  ) => Promise<void>
  var llamaRemoveLoraAdapters: (contextId: number) => Promise<void>
  var llamaGetLoadedLoraAdapters: (
    contextId: number,
  ) => Promise<Array<{ path: string; scaled?: number }>>
  var llamaInitMultimodal: (
    contextId: number,
    params: { path: string; use_gpu?: boolean },
  ) => Promise<boolean>
  var llamaIsMultimodalEnabled: (contextId: number) => Promise<boolean>
  var llamaGetMultimodalSupport: (
    contextId: number,
  ) => Promise<{ vision: boolean; audio: boolean }>
  var llamaReleaseMultimodal: (contextId: number) => Promise<void>
  var llamaInitVocoder: (
    contextId: number,
    params: { path: string; n_batch?: number },
  ) => Promise<boolean>
  var llamaIsVocoderEnabled: (contextId: number) => Promise<boolean>
  var llamaGetFormattedAudioCompletion: (
    contextId: number,
    speaker: string,
    text: string,
  ) => Promise<{ prompt: string; grammar?: string }>
  var llamaGetAudioCompletionGuideTokens: (
    contextId: number,
    text: string,
  ) => Promise<number[]>
  var llamaDecodeAudioTokens: (
    contextId: number,
    tokens: number[],
  ) => Promise<number[]>
  var llamaReleaseVocoder: (contextId: number) => Promise<void>
  var llamaClearCache: (contextId: number, clearData: boolean) => Promise<void>

  // Parallel decoding
  var llamaEnableParallelMode: (
    contextId: number,
    params: { enabled: boolean; n_parallel?: number; n_batch?: number },
  ) => Promise<boolean>
  var llamaQueueCompletion: (
    contextId: number,
    params: NativeCompletionParams,
    onToken: (token: any, requestId: number) => void,
    onComplete: (result: any) => void,
  ) => Promise<{ requestId: number }>
  var llamaCancelRequest: (
    contextId: number,
    requestId: number,
  ) => Promise<void>
  var llamaQueueEmbedding: (
    contextId: number,
    text: string,
    params: object,
    onResult: (result: number[]) => void,
  ) => Promise<{ requestId: number }>
  var llamaQueueRerank: (
    contextId: number,
    query: string,
    documents: string[],
    params: object,
    onResult: (result: NativeRerankResult[]) => void,
  ) => Promise<{ requestId: number }>
}
