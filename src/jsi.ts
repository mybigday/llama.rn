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
  ParallelStatus,
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
    params: {
      path: string
      use_gpu?: boolean
      image_min_tokens?: number
      image_max_tokens?: number
    },
  ) => Promise<boolean>
  var llamaIsMultimodalEnabled: (contextId: number) => Promise<boolean>
  var llamaGetMultimodalSupport: (
    contextId: number,
  ) => Promise<{ vision: boolean; audio: boolean }>
  var llamaReleaseMultimodal: (contextId: number) => Promise<void>
  var llamaInitVocoder: (
    contextId: number,
    params: { path: string; n_batch?: number; use_gpu?: boolean },
  ) => Promise<boolean>
  var llamaIsVocoderEnabled: (contextId: number) => Promise<boolean>
  var llamaGetFormattedAudioCompletion: (
    contextId: number,
    speaker: string,
    text: string,
  ) => Promise<{
    prompt: string
    grammar?: string
    embedding: boolean
    // 'tokens'           — feed `prompt` through `completion()` and collect audio tokens.
    //                      Now covers the codec_lm-AR family too (CSM /
    //                      Qwen3-TTS / MOSS-TTSD / MOSS-TTS-Realtime /
    //                      Chatterbox): the native completion loop drives
    //                      the codec_lm step machine per `llama_decode` and
    //                      appends codes to the standard audio-token buffer.
    // 'codec_lm_ar'      — DEPRECATED (kept as a source-compat literal for
    //                      older native builds).  New native always emits
    //                      'tokens' for codec_lm-AR models.  If you still
    //                      see this, `generateAudioCodes()` remains a
    //                      wrapper that internally runs `completion`.
    // 'continuous_embd'  — feed `prompt` through `completion()`; the loop drives the
    //                      codec_lm's continuous-latent step machine per
    //                      `llama_decode` (BlueMagpie-TTS / VoxCPM).  Collect
    //                      `embeddings` + `embedding_dim` from the completion
    //                      result and pass them to `decodeAudioEmbeddings`.
    flow: 'tokens' | 'codec_lm_ar' | 'continuous_embd' | ''
  }>
  var llamaGetTTSCapabilities: (contextId: number) => Promise<{
    type: number
    promptKind:
      | 'outetts_legacy'
      | 'outetts_v0_3'
      | 'outetts_v1_0'
      | 'soprano'
      | 'neutts'
      | 'csm'
      | 'qwen3_tts'
      | 'moss_tts_realtime'
      | 'moss_ttsd'
      | 'chatterbox'
      | 'chatterbox_multilingual'
      | ''
    family:
      | 'outetts'
      | 'soprano'
      | 'neutts'
      | 'csm'
      | 'qwen3_tts'
      | 'moss_tts'
      | 'moss_ttsd'
      | 'chatterbox'
      | ''
    requiresPhonemes: boolean
    defaultLanguage: string
  }>
  var llamaDecodeAudioTokens: (
    contextId: number,
    tokens: number[],
  ) => Promise<number[]>
  var llamaGenerateAudioCodes: (
    contextId: number,
    optsJson: string,
    onFrame?: (step: number, codes: number[]) => void,
  ) => Promise<{
    codes: number[]
    nCodebook: number
    nFrames: number
    stoppedOnEos: boolean
    aborted: boolean
  }>
  var llamaEncodeSpeaker: (
    contextId: number,
    optsJson: string,
  ) => Promise<{
    refCodes: number[]
    nQ: number
    nFrames: number
    sampleRate: number
    codebookSize: number
    refText: string
    speakerEmb?: number[]
    speakerNRows: number
    speakerHiddenDim: number
  }>
  var llamaDecodeAudioEmbeddings: (
    contextId: number,
    embeddings: number[],
    embeddingDim: number,
  ) => Promise<number[]>
  var llamaGetAudioSampleRate: (contextId: number) => Promise<number>
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
  var llamaGetParallelStatus: (contextId: number) => Promise<ParallelStatus>
  var llamaSubscribeParallelStatus: (
    contextId: number,
    onStatus: (status: ParallelStatus) => void,
  ) => Promise<{ subscriberId: number }>
  var llamaUnsubscribeParallelStatus: (
    contextId: number,
    subscriberId: number,
  ) => void
}
