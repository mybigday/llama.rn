import { Platform } from 'react-native'
import RNLlama from './NativeRNLlama'
import './jsi'
import type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeParallelCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeRerankParams,
  NativeRerankResult,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  JinjaFormattedChatResult,
  FormattedChatResult,
  NativeImageProcessingResult,
  NativeLlamaChatMessage,
  NativeBackendDeviceInfo,
} from './types'
import { BUILD_NUMBER, BUILD_COMMIT } from './version'

export type RNLlamaMessagePart = {
  type: string
  text?: string
  image_url?: {
    url?: string
  }
  input_audio?: {
    format: string
    data?: string
    url?: string
  }
}

export type RNLlamaOAICompatibleMessage = {
  role: string
  content?: string | RNLlamaMessagePart[]
  reasoning_content?: string
}

export type {
  NativeContextParams,
  NativeLlamaContext,
  NativeCompletionParams,
  NativeParallelCompletionParams,
  NativeCompletionTokenProb,
  NativeCompletionResult,
  NativeTokenizeResult,
  NativeEmbeddingResult,
  NativeSessionLoadResult,
  NativeEmbeddingParams,
  NativeRerankParams,
  NativeRerankResult,
  NativeCompletionTokenProbItem,
  NativeCompletionResultTimings,
  FormattedChatResult,
  JinjaFormattedChatResult,
  NativeImageProcessingResult,
  NativeBackendDeviceInfo,
}

export const RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER = '<__media__>'

const logListeners: Array<(level: string, text: string) => void> = []
const emitNativeLog = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

const jsiBindingKeys = [
  'llamaInitContext',
  'llamaReleaseContext',
  'llamaReleaseAllContexts',
  'llamaModelInfo',
  'llamaGetBackendDevicesInfo',
  'llamaLoadSession',
  'llamaSaveSession',
  'llamaTokenize',
  'llamaDetokenize',
  'llamaGetFormattedChat',
  'llamaEmbedding',
  'llamaRerank',
  'llamaBench',
  'llamaToggleNativeLog',
  'llamaSetContextLimit',
  'llamaCompletion',
  'llamaStopCompletion',
  'llamaApplyLoraAdapters',
  'llamaRemoveLoraAdapters',
  'llamaGetLoadedLoraAdapters',
  'llamaInitMultimodal',
  'llamaIsMultimodalEnabled',
  'llamaGetMultimodalSupport',
  'llamaReleaseMultimodal',
  'llamaInitVocoder',
  'llamaIsVocoderEnabled',
  'llamaGetFormattedAudioCompletion',
  'llamaGetAudioCompletionGuideTokens',
  'llamaDecodeAudioTokens',
  'llamaReleaseVocoder',
  'llamaClearCache',
  'llamaEnableParallelMode',
  'llamaQueueCompletion',
  'llamaCancelRequest',
  'llamaQueueEmbedding',
  'llamaQueueRerank',
] as const

type JsiBindingKey = (typeof jsiBindingKeys)[number]
type JsiBindings = { [K in JsiBindingKey]: NonNullable<(typeof globalThis)[K]> }

let jsiBindings: JsiBindings | null = null

const bindJsiFromGlobal = () => {
  const bindings: Partial<JsiBindings> = {}
  const missing: string[] = []

  jsiBindingKeys.forEach((key) => {
    const value = global[key]
    if (typeof value === 'function') {
      ;(bindings as Record<string, unknown>)[key] =
        value as JsiBindings[typeof key]
      delete global[key]
    } else {
      missing.push(key)
    }
  })

  if (missing.length > 0) {
    throw new Error(`[RNLlama] Missing JSI bindings: ${missing.join(', ')}`)
  }

  jsiBindings = bindings as JsiBindings
}

const getJsi = (): JsiBindings => {
  if (!jsiBindings) {
    throw new Error('JSI bindings not installed')
  }
  return jsiBindings
}

// JSI Installation
let isJsiInstalled = false
export const installJsi = async () => {
  if (isJsiInstalled) return
  if (typeof global.llamaInitContext !== 'function') {
    const installed = await RNLlama.install()
    if (!installed && typeof global.llamaInitContext !== 'function') {
      throw new Error('JSI bindings not installed')
    }
  }
  bindJsiFromGlobal()
  isJsiInstalled = true
}

export type ToolCall = {
  type: 'function'
  id?: string
  function: {
    name: string
    arguments: string // JSON string
  }
}

export type TokenData = {
  token: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
  // Parsed content from accumulated text
  content?: string
  reasoning_content?: string
  tool_calls?: Array<ToolCall>
  accumulated_text?: string
  requestId?: number
}

export type ContextParams = Omit<
  NativeContextParams,
  'flash_attn_type' | 'cache_type_k' | 'cache_type_v' | 'pooling_type'
> & {
  flash_attn_type?: 'auto' | 'on' | 'off'
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

const validCacheTypes = [
  'f16',
  'f32',
  'bf16',
  'q8_0',
  'q4_0',
  'q4_1',
  'iq4_nl',
  'q5_0',
  'q5_1',
]

export type EmbeddingParams = NativeEmbeddingParams

export type RerankParams = {
  normalize?: number
}

export type RerankResult = {
  score: number
  index: number
  document?: string
}

export type CompletionResponseFormat = {
  type: 'text' | 'json_object' | 'json_schema'
  json_schema?: {
    strict?: boolean
    schema: object
  }
  schema?: object // for json_object type
}

export type CompletionBaseParams = {
  prompt?: string
  messages?: RNLlamaOAICompatibleMessage[]
  chatTemplate?: string // deprecated
  chat_template?: string
  jinja?: boolean
  tools?: object
  parallel_tool_calls?: object
  tool_choice?: string
  response_format?: CompletionResponseFormat
  media_paths?: string | string[]
  add_generation_prompt?: boolean
  /*
   * Timestamp in seconds since epoch to apply to chat template's strftime_now
   */
  now?: string | number
  chat_template_kwargs?: Record<string, string>
  /**
   * Prefill text to be used for chat parsing (Generation Prompt + Content)
   * Used for if last assistant message is for prefill purpose
   */
  prefill_text?: string
}
export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> &
  CompletionBaseParams

/**
 * Parameters for parallel completion requests.
 * Extends CompletionParams with parallel-mode specific options like state management.
 */
export type ParallelCompletionParams = Omit<
  NativeParallelCompletionParams,
  'emit_partial_completion' | 'prompt'
> &
  CompletionBaseParams

export type BenchResult = {
  nKvMax: number
  nBatch: number
  nUBatch: number
  flashAttn: number
  isPpShared: number
  nGpuLayers: number
  nThreads: number
  nThreadsBatch: number
  pp: number
  tg: number
  pl: number
  nKv: number
  tPp: number
  speedPp: number
  tTg: number
  speedTg: number
  t: number
  speed: number
}

const getJsonSchema = (responseFormat?: CompletionResponseFormat) => {
  if (responseFormat?.type === 'json_schema') {
    return responseFormat.json_schema?.schema
  }
  if (responseFormat?.type === 'json_object') {
    return responseFormat.schema || {}
  }
  return null
}

export class LlamaContext {
  id: number

  gpu: boolean = false

  reasonNoGPU: string = ''

  devices: NativeLlamaContext['devices']

  model: NativeLlamaContext['model']

  androidLib: NativeLlamaContext['androidLib']

  systemInfo: NativeLlamaContext['systemInfo']

  /**
   * Parallel processing namespace for non-blocking queue operations
   */
  parallel = {
    /**
     * Queue a completion request for parallel processing (non-blocking)
     * @param params Parallel completion parameters (includes state management)
     * @param onToken Callback fired for each generated token
     * @returns Promise resolving to object with requestId, promise (resolves to completion result), and stop function
     */
    completion: async (
      params: ParallelCompletionParams,
      onToken?: (requestId: number, data: TokenData) => void,
    ): Promise<{
      requestId: number
      promise: Promise<NativeCompletionResult>
      stop: () => Promise<void>
    }> => {
      const { llamaQueueCompletion, llamaCancelRequest } = getJsi()
      const nativeParams = {
        ...params,
        prompt: params.prompt || '',
        emit_partial_completion: true, // Always emit for queued requests
      }

      // Process messages same as completion()
      if (params.messages) {
        const formattedResult = await this.getFormattedChat(
          params.messages,
          params.chat_template || params.chatTemplate,
          {
            jinja: params.jinja,
            tools: params.tools,
            parallel_tool_calls: params.parallel_tool_calls,
            tool_choice: params.tool_choice,
            enable_thinking: params.enable_thinking,
            add_generation_prompt: params.add_generation_prompt,
            now: params.now,
            chat_template_kwargs: params.chat_template_kwargs,
          },
        )
        if (formattedResult.type === 'jinja') {
          const jinjaResult = formattedResult as JinjaFormattedChatResult
          nativeParams.prompt = jinjaResult.prompt || ''
          if (typeof jinjaResult.chat_format === 'number')
            nativeParams.chat_format = jinjaResult.chat_format
          if (jinjaResult.grammar) nativeParams.grammar = jinjaResult.grammar
          if (typeof jinjaResult.grammar_lazy === 'boolean')
            nativeParams.grammar_lazy = jinjaResult.grammar_lazy
          if (jinjaResult.grammar_triggers)
            nativeParams.grammar_triggers = jinjaResult.grammar_triggers
          if (jinjaResult.preserved_tokens)
            nativeParams.preserved_tokens = jinjaResult.preserved_tokens
          if (jinjaResult.additional_stops) {
            if (!nativeParams.stop) nativeParams.stop = []
            nativeParams.stop.push(...jinjaResult.additional_stops)
          }
          if (jinjaResult.has_media) {
            nativeParams.media_paths = jinjaResult.media_paths
          }
        } else if (formattedResult.type === 'llama-chat') {
          const llamaChatResult = formattedResult as FormattedChatResult
          nativeParams.prompt = llamaChatResult.prompt || ''
          if (llamaChatResult.has_media) {
            nativeParams.media_paths = llamaChatResult.media_paths
          }
        }
      } else {
        nativeParams.prompt = params.prompt || ''
      }

      if (!nativeParams.media_paths && params.media_paths) {
        nativeParams.media_paths = params.media_paths
      }

      if (nativeParams.response_format && !nativeParams.grammar) {
        const jsonSchema = getJsonSchema(params.response_format)
        if (jsonSchema) nativeParams.json_schema = JSON.stringify(jsonSchema)
      }

      if (!nativeParams.prompt) throw new Error('Prompt is required')

      return new Promise(async (resolveOuter, rejectOuter) => {
        try {
          let resolveResult: (
            value: NativeCompletionResult | PromiseLike<NativeCompletionResult>,
          ) => void
          let rejectResult: (reason?: any) => void

          const resultPromise = new Promise<NativeCompletionResult>(
            (res, rej) => {
              resolveResult = res
              rejectResult = rej
            },
          )

          const { requestId } = await llamaQueueCompletion(
            this.id,
            nativeParams,
            (tokenResult, reqId) => {
              if (onToken) onToken(reqId, tokenResult)
            },
            (result) => {
              if (result.error) {
                rejectResult(new Error(result.error))
              } else {
                resolveResult(result)
              }
            },
          )

          resolveOuter({
            requestId,
            promise: resultPromise,
            stop: async () => {
              await llamaCancelRequest(this.id, requestId)
            },
          })
        } catch (e) {
          rejectOuter(e)
        }
      })
    },

    /**
     * Queue an embedding request for parallel processing (non-blocking)
     * @param text Text to embed
     * @param params Optional embedding parameters
     * @returns Promise resolving to object with requestId and promise (resolves to embedding result)
     */
    embedding: async (
      text: string,
      params?: EmbeddingParams,
    ): Promise<{
      requestId: number
      promise: Promise<NativeEmbeddingResult>
    }> =>
      new Promise(async (resolveOuter, rejectOuter) => {
        const { llamaQueueEmbedding } = getJsi()
        try {
          let resolveResult: (value: NativeEmbeddingResult) => void
          const resultPromise = new Promise<NativeEmbeddingResult>((res) => {
            resolveResult = res
          })

          const { requestId } = await llamaQueueEmbedding(
            this.id,
            text,
            params || {},
            (embedding) => {
              resolveResult({ embedding })
            },
          )

          resolveOuter({
            requestId,
            promise: resultPromise,
          })
        } catch (e) {
          rejectOuter(e)
        }
      }),

    /**
     * Queue rerank requests for parallel processing (non-blocking)
     * @param query The query text to rank documents against
     * @param documents Array of document texts to rank
     * @param params Optional reranking parameters
     * @returns Promise resolving to object with requestId and promise (resolves to rerank results)
     */
    rerank: async (
      query: string,
      documents: string[],
      params?: RerankParams,
    ): Promise<{
      requestId: number
      promise: Promise<RerankResult[]>
    }> =>
      new Promise(async (resolveOuter, rejectOuter) => {
        const { llamaQueueRerank } = getJsi()
        try {
          let resolveResult: (value: RerankResult[]) => void
          const resultPromise = new Promise<RerankResult[]>((res) => {
            resolveResult = res
          })

          const { requestId } = await llamaQueueRerank(
            this.id,
            query,
            documents,
            params || {},
            (results) => {
              const sortedResults = results
                .map((result: NativeRerankResult) => ({
                  ...result,
                  document: documents[result.index],
                }))
                .sort((a: RerankResult, b: RerankResult) => b.score - a.score)
              resolveResult(sortedResults)
            },
          )

          resolveOuter({
            requestId,
            promise: resultPromise,
          })
        } catch (e) {
          rejectOuter(e)
        }
      }),

    enable: (config?: { n_parallel?: number; n_batch?: number }) =>
      getJsi().llamaEnableParallelMode(this.id, { enabled: true, ...config }),

    disable: () =>
      getJsi().llamaEnableParallelMode(this.id, { enabled: false }),

    configure: (config: { n_parallel?: number; n_batch?: number }) =>
      getJsi().llamaEnableParallelMode(this.id, { enabled: true, ...config }),
  }

  constructor({
    contextId,
    gpu,
    devices,
    reasonNoGPU,
    model,
    androidLib,
    systemInfo,
  }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.devices = devices
    this.reasonNoGPU = reasonNoGPU
    this.model = model
    this.androidLib = androidLib
    this.systemInfo = systemInfo
  }

  async loadSession(filepath: string): Promise<NativeSessionLoadResult> {
    const { llamaLoadSession } = getJsi()
    let path = filepath
    if (path.startsWith('file://')) path = path.slice(7)
    return llamaLoadSession(this.id, path)
  }

  async saveSession(
    filepath: string,
    options?: { tokenSize: number },
  ): Promise<number> {
    const { llamaSaveSession } = getJsi()
    return llamaSaveSession(this.id, filepath, options?.tokenSize || -1)
  }

  isLlamaChatSupported(): boolean {
    return !!this.model.chatTemplates.llamaChat
  }

  isJinjaSupported(): boolean {
    const { minja } = this.model.chatTemplates
    return !!minja?.toolUse || !!minja?.default
  }

  async getFormattedChat(
    messages: RNLlamaOAICompatibleMessage[],
    template?: string | null,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: object
      parallel_tool_calls?: object
      tool_choice?: string
      enable_thinking?: boolean
      add_generation_prompt?: boolean
      now?: string | number
      chat_template_kwargs?: Record<string, string>
    },
  ): Promise<FormattedChatResult | JinjaFormattedChatResult> {
    const mediaPaths: string[] = []
    const chat = messages.map((msg) => {
      if (Array.isArray(msg.content)) {
        const content = msg.content.map((part) => {
          if (part.type === 'image_url') {
            let path = part.image_url?.url || ''
            if (path?.startsWith('file://')) path = path.slice(7)
            mediaPaths.push(path)
            return {
              type: 'text',
              text: RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER,
            }
          } else if (part.type === 'input_audio') {
            const { input_audio: audio } = part
            if (!audio) throw new Error('input_audio is required')

            const { format } = audio
            if (format != 'wav' && format != 'mp3') {
              throw new Error(`Unsupported audio format: ${format}`)
            }
            if (audio.url) {
              const path = audio.url.replace(/file:\/\//, '')
              mediaPaths.push(path)
            } else if (audio.data) {
              mediaPaths.push(audio.data)
            }
            return {
              type: 'text',
              text: RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER,
            }
          }
          return part
        })

        return {
          ...msg,
          content,
        }
      }
      return msg
    }) as NativeLlamaChatMessage[]

    const useJinja = this.isJinjaSupported() && (params?.jinja ?? true)
    let tmpl
    if (template) tmpl = template
    const jsonSchema = getJsonSchema(params?.response_format)

    const { llamaGetFormattedChat } = getJsi()
    const result = await llamaGetFormattedChat(
      this.id,
      JSON.stringify(chat),
      tmpl,
      {
        jinja: useJinja,
        json_schema: jsonSchema ? JSON.stringify(jsonSchema) : undefined,
        tools: params?.tools ? JSON.stringify(params.tools) : undefined,
        parallel_tool_calls: params?.parallel_tool_calls
          ? JSON.stringify(params.parallel_tool_calls)
          : undefined,
        tool_choice: params?.tool_choice,
        enable_thinking: params?.enable_thinking ?? true,
        add_generation_prompt: params?.add_generation_prompt,
        now:
          typeof params?.now === 'number' ? params.now.toString() : params?.now,
        chat_template_kwargs: params?.chat_template_kwargs
          ? JSON.stringify(
              Object.entries(params.chat_template_kwargs).reduce(
                (acc, [key, value]) => {
                  acc[key] = JSON.stringify(value)
                  return acc
                },
                {} as Record<string, any>,
              ),
            )
          : undefined,
      },
    )
    if (!useJinja) {
      return {
        type: 'llama-chat',
        prompt: result as string,
        has_media: mediaPaths.length > 0,
        media_paths: mediaPaths,
      }
    }
    const jinjaResult = result as JinjaFormattedChatResult
    jinjaResult.type = 'jinja'
    jinjaResult.has_media = mediaPaths.length > 0
    jinjaResult.media_paths = mediaPaths
    return jinjaResult
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
      const formattedResult = await this.getFormattedChat(
        params.messages,
        params.chat_template || params.chatTemplate,
        {
          jinja: params.jinja,
          tools: params.tools,
          parallel_tool_calls: params.parallel_tool_calls,
          tool_choice: params.tool_choice,
          enable_thinking: params.enable_thinking,
          add_generation_prompt: params.add_generation_prompt,
          now: params.now,
          chat_template_kwargs: params.chat_template_kwargs,
        },
      )
      if (formattedResult.type === 'jinja') {
        const jinjaResult = formattedResult as JinjaFormattedChatResult

        nativeParams.prompt = jinjaResult.prompt || ''
        if (typeof jinjaResult.chat_format === 'number')
          nativeParams.chat_format = jinjaResult.chat_format
        if (jinjaResult.grammar) nativeParams.grammar = jinjaResult.grammar
        if (typeof jinjaResult.grammar_lazy === 'boolean')
          nativeParams.grammar_lazy = jinjaResult.grammar_lazy
        if (jinjaResult.grammar_triggers)
          nativeParams.grammar_triggers = jinjaResult.grammar_triggers
        if (jinjaResult.preserved_tokens)
          nativeParams.preserved_tokens = jinjaResult.preserved_tokens
        if (jinjaResult.additional_stops) {
          if (!nativeParams.stop) nativeParams.stop = []
          nativeParams.stop.push(...jinjaResult.additional_stops)
        }
        if (jinjaResult.has_media) {
          nativeParams.media_paths = jinjaResult.media_paths
        }
      } else if (formattedResult.type === 'llama-chat') {
        const llamaChatResult = formattedResult as FormattedChatResult
        nativeParams.prompt = llamaChatResult.prompt || ''
        if (llamaChatResult.has_media) {
          nativeParams.media_paths = llamaChatResult.media_paths
        }
      }
    } else {
      nativeParams.prompt = params.prompt || ''
    }

    if (!nativeParams.media_paths && params.media_paths) {
      nativeParams.media_paths = params.media_paths
    }

    if (nativeParams.response_format && !nativeParams.grammar) {
      const jsonSchema = getJsonSchema(params.response_format)
      if (jsonSchema) nativeParams.json_schema = JSON.stringify(jsonSchema)
    }

    if (!nativeParams.prompt) throw new Error('Prompt is required')

    const { llamaCompletion } = getJsi()
    return llamaCompletion(this.id, nativeParams, callback)
  }

  stopCompletion(): Promise<void> {
    const { llamaStopCompletion } = getJsi()
    return llamaStopCompletion(this.id)
  }

  tokenize(
    text: string,
    {
      media_paths: mediaPaths,
    }: {
      media_paths?: string[]
    } = {},
  ): Promise<NativeTokenizeResult> {
    const { llamaTokenize } = getJsi()
    return llamaTokenize(this.id, text, mediaPaths)
  }

  detokenize(tokens: number[]): Promise<string> {
    const { llamaDetokenize } = getJsi()
    return llamaDetokenize(this.id, tokens)
  }

  embedding(
    text: string,
    params?: EmbeddingParams,
  ): Promise<NativeEmbeddingResult> {
    const { llamaEmbedding } = getJsi()
    return llamaEmbedding(this.id, text, params || {})
  }

  async rerank(
    query: string,
    documents: string[],
    params?: RerankParams,
  ): Promise<RerankResult[]> {
    const { llamaRerank } = getJsi()
    const results = await llamaRerank(this.id, query, documents, params || {})

    return results
      .map((result) => ({
        ...result,
        document: documents[result.index],
      }))
      .sort((a, b) => b.score - a.score)
  }

  async bench(
    pp: number,
    tg: number,
    pl: number,
    nr: number,
  ): Promise<BenchResult> {
    const { llamaBench } = getJsi()
    const result = await llamaBench(this.id, pp, tg, pl, nr)
    const parsed = JSON.parse(result)
    return {
      nKvMax: parsed.n_kv_max,
      nBatch: parsed.n_batch,
      nUBatch: parsed.n_ubatch,
      flashAttn: parsed.flash_attn,
      isPpShared: parsed.is_pp_shared,
      nGpuLayers: parsed.n_gpu_layers,
      nThreads: parsed.n_threads,
      nThreadsBatch: parsed.n_threads_batch,
      pp: parsed.pp,
      tg: parsed.tg,
      pl: parsed.pl,
      nKv: parsed.n_kv,
      tPp: parsed.t_pp,
      speedPp: parsed.speed_pp,
      tTg: parsed.t_tg,
      speedTg: parsed.speed_tg,
      t: parsed.t,
      speed: parsed.speed,
    }
  }

  async applyLoraAdapters(
    loraList: Array<{ path: string; scaled?: number }>,
  ): Promise<void> {
    const { llamaApplyLoraAdapters } = getJsi()
    let loraAdapters: Array<{ path: string; scaled?: number }> = []
    if (loraList)
      loraAdapters = loraList.map((l) => ({
        path: l.path.replace(/file:\/\//, ''),
        scaled: l.scaled,
      }))
    return llamaApplyLoraAdapters(this.id, loraAdapters)
  }

  async removeLoraAdapters(): Promise<void> {
    const { llamaRemoveLoraAdapters } = getJsi()
    return llamaRemoveLoraAdapters(this.id)
  }

  async getLoadedLoraAdapters(): Promise<
    Array<{ path: string; scaled?: number }>
  > {
    const { llamaGetLoadedLoraAdapters } = getJsi()
    return llamaGetLoadedLoraAdapters(this.id)
  }

  async initMultimodal({
    path,
    use_gpu: useGpu,
  }: {
    path: string
    use_gpu?: boolean
  }): Promise<boolean> {
    const { llamaInitMultimodal } = getJsi()
    if (path.startsWith('file://')) path = path.slice(7)
    return llamaInitMultimodal(this.id, {
      path,
      use_gpu: useGpu ?? true,
    })
  }

  async isMultimodalEnabled(): Promise<boolean> {
    const { llamaIsMultimodalEnabled } = getJsi()
    return await llamaIsMultimodalEnabled(this.id)
  }

  async getMultimodalSupport(): Promise<{
    vision: boolean
    audio: boolean
  }> {
    const { llamaGetMultimodalSupport } = getJsi()
    return await llamaGetMultimodalSupport(this.id)
  }

  async releaseMultimodal(): Promise<void> {
    const { llamaReleaseMultimodal } = getJsi()
    return await llamaReleaseMultimodal(this.id)
  }

  async initVocoder({
    path,
    n_batch: nBatch,
  }: {
    path: string
    n_batch?: number
  }): Promise<boolean> {
    const { llamaInitVocoder } = getJsi()
    if (path.startsWith('file://')) path = path.slice(7)
    return await llamaInitVocoder(this.id, { path, n_batch: nBatch })
  }

  async isVocoderEnabled(): Promise<boolean> {
    const { llamaIsVocoderEnabled } = getJsi()
    return await llamaIsVocoderEnabled(this.id)
  }

  async getFormattedAudioCompletion(
    speaker: object | null,
    textToSpeak: string,
  ): Promise<{
    prompt: string
    grammar?: string
  }> {
    const { llamaGetFormattedAudioCompletion } = getJsi()
    return await llamaGetFormattedAudioCompletion(
      this.id,
      speaker ? JSON.stringify(speaker) : '',
      textToSpeak,
    )
  }

  async getAudioCompletionGuideTokens(
    textToSpeak: string,
  ): Promise<Array<number>> {
    const { llamaGetAudioCompletionGuideTokens } = getJsi()
    return await llamaGetAudioCompletionGuideTokens(this.id, textToSpeak)
  }

  async decodeAudioTokens(tokens: number[]): Promise<Array<number>> {
    const { llamaDecodeAudioTokens } = getJsi()
    return await llamaDecodeAudioTokens(this.id, tokens)
  }

  async releaseVocoder(): Promise<void> {
    const { llamaReleaseVocoder } = getJsi()
    return await llamaReleaseVocoder(this.id)
  }

  /**
   * Clear the KV cache and reset conversation state
   * @param clearData If true, clears both metadata and tensor data buffers (slower). If false, only clears metadata (faster).
   * @returns Promise that resolves when cache is cleared
   *
   * Call this method between different conversations to prevent cache contamination.
   * Without clearing, the model may use cached context from previous conversations,
   * leading to incorrect or unexpected responses.
   *
   * For hybrid architecture models (e.g., LFM2), this is essential as they
   * use recurrent state that cannot be partially removed - only fully cleared.
   */
  async clearCache(clearData: boolean = false): Promise<void> {
    const { llamaClearCache } = getJsi()
    return llamaClearCache(this.id, clearData)
  }

  async release(): Promise<void> {
    const { llamaReleaseContext } = getJsi()
    return llamaReleaseContext(this.id)
  }
}

export async function toggleNativeLog(enabled: boolean): Promise<void> {
  await installJsi()
  const { llamaToggleNativeLog } = getJsi()
  return llamaToggleNativeLog(enabled, emitNativeLog)
}

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      logListeners.splice(logListeners.indexOf(listener), 1)
    },
  }
}

export async function setContextLimit(limit: number): Promise<void> {
  await installJsi()
  const { llamaSetContextLimit } = getJsi()
  return llamaSetContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  /* @ts-ignore */
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
  'tokenizer.ggml.scores',
]
export async function loadLlamaModelInfo(model: string): Promise<Object> {
  await installJsi()
  const { llamaModelInfo } = getJsi()
  let path = model
  if (path.startsWith('file://')) path = path.slice(7)
  return llamaModelInfo(path, modelInfoSkip)
}

const poolTypeMap = {
  none: 0,
  mean: 1,
  cls: 2,
  last: 3,
  rank: 4,
}

export async function getBackendDevicesInfo(): Promise<
  Array<NativeBackendDeviceInfo>
> {
  await installJsi()
  const { llamaGetBackendDevicesInfo } = getJsi()
  try {
    const jsonString = await llamaGetBackendDevicesInfo()
    return JSON.parse(jsonString as string)
  } catch (e) {
    console.warn(
      '[RNLlama] Failed to parse backend devices info, falling back to empty list',
      e,
    )
    return []
  }
}

export async function initLlama(
  {
    model,
    is_model_asset: isModelAsset,
    pooling_type: poolingType,
    lora,
    lora_list: loraList,
    devices,
    ...rest
  }: ContextParams,
  onProgress?: (progress: number) => void,
): Promise<LlamaContext> {
  await installJsi()
  const { llamaInitContext } = getJsi()
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

  let lastProgress = 0
  const progressCallback = onProgress
    ? (progress: number) => {
        lastProgress = progress
        try {
          onProgress(progress)
        } catch (err) {
          console.warn('[RNLlama] onProgress callback failed', err)
        }
      }
    : undefined

  if (progressCallback) progressCallback(0)

  const poolType = poolTypeMap[poolingType as keyof typeof poolTypeMap]

  if (rest.cache_type_k && !validCacheTypes.includes(rest.cache_type_k)) {
    console.warn(
      `[RNLlama] initLlama: Invalid cache K type: ${rest.cache_type_k}, falling back to f16`,
    )
    delete rest.cache_type_k
  }
  if (rest.cache_type_v && !validCacheTypes.includes(rest.cache_type_v)) {
    console.warn(
      `[RNLlama] initLlama: Invalid cache V type: ${rest.cache_type_v}, falling back to f16`,
    )
    delete rest.cache_type_v
  }

  let filteredDevs: Array<string> = []
  if (Array.isArray(devices)) {
    filteredDevs = [...devices]
    const backendDevices = await getBackendDevicesInfo()

    if (Platform.OS === 'android' && devices.includes('HTP*')) {
      const htpDevices = backendDevices
        .filter((d) => d.deviceName.startsWith('HTP'))
        .map((d) => d.deviceName)
      filteredDevs = filteredDevs.reduce((acc, dev) => {
        if (dev.startsWith('HTP*')) {
          acc.push(...htpDevices)
        } else if (!dev.startsWith('HTP')) {
          acc.push(dev)
        }
        return acc
      }, [] as Array<string>)
    }
  }

  const {
    gpu,
    devices: usedDevices,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
    systemInfo,
  } = await llamaInitContext(
    contextId,
    {
      model: path,
      is_model_asset: !!isModelAsset,
      use_progress_callback: !!progressCallback,
      pooling_type: poolType,
      lora: loraPath,
      lora_list: loraAdapters,
      devices: filteredDevs.length > 0 ? filteredDevs : undefined,
      ...rest,
    },
    progressCallback,
  )

  if (progressCallback && lastProgress < 100) progressCallback(100)

  return new LlamaContext({
    contextId,
    gpu,
    devices: usedDevices,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
    systemInfo,
  })
}

export async function releaseAllLlama(): Promise<void> {
  if (!isJsiInstalled) return
  const { llamaReleaseAllContexts } = getJsi()
  return llamaReleaseAllContexts()
}

export const BuildInfo = {
  number: BUILD_NUMBER,
  commit: BUILD_COMMIT,
}
