import { NativeEventEmitter, DeviceEventEmitter, Platform } from 'react-native'
import type { DeviceEventEmitterStatic } from 'react-native'
import RNLlama from './NativeRNLlama'
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
} from './NativeRNLlama'
import type {
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
} from './grammar'
import { SchemaGrammarConverter, convertJsonSchemaToGrammar } from './grammar'
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

  // Deprecated
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
}

export const RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER = '<__media__>'

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_INIT_CONTEXT_PROGRESS = '@RNLlama_onInitContextProgress'
const EVENT_ON_TOKEN = '@RNLlama_onToken'
const EVENT_ON_COMPLETE = '@RNLlama_onComplete'
const EVENT_ON_EMBEDDING_RESULT = '@RNLlama_onEmbeddingResult'
const EVENT_ON_RERANK_RESULTS = '@RNLlama_onRerankResults'
const EVENT_ON_NATIVE_LOG = '@RNLlama_onNativeLog'

let EventEmitter: NativeEventEmitter | DeviceEventEmitterStatic
if (Platform.OS === 'ios') {
  // @ts-ignore
  EventEmitter = new NativeEventEmitter(RNLlama)
}
if (Platform.OS === 'android') {
  EventEmitter = DeviceEventEmitter
}

const logListeners: Array<(level: string, text: string) => void> = []

// @ts-ignore
if (EventEmitter) {
  EventEmitter.addListener(
    EVENT_ON_NATIVE_LOG,
    (evt: { level: string; text: string }) => {
      logListeners.forEach((listener) => listener(evt.level, evt.text))
    },
  )
  // Trigger unset to use default log callback
  RNLlama?.toggleNativeLog?.(false)?.catch?.(() => {})
}

export type ToolCall = {
  type: 'function'
  id?: string
  function: {
    name: string
    arguments: string  // JSON string
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
}

type TokenNativeEvent = {
  contextId: number
  requestId?: number
  tokenResult: TokenData
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

  gpuDevice: NativeLlamaContext['gpuDevice']

  reasonNoGPU: string = ''

  model: NativeLlamaContext['model']

  androidLib: NativeLlamaContext['androidLib']

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

      // Set up listeners for this specific request
      let tokenListener: any
      let completeListener: any

      const { requestId } = await RNLlama.queueCompletion(this.id, nativeParams)

      // Create promise that resolves when completion finishes
      const promise = new Promise<NativeCompletionResult>((resolve, reject) => {
        if (onToken) {
          tokenListener = EventEmitter.addListener(EVENT_ON_TOKEN, (evt: TokenNativeEvent) => {
            const { contextId, requestId: evtRequestId, tokenResult } = evt
            if (contextId !== this.id) return
            if (evtRequestId !== requestId) return
            onToken(requestId, tokenResult)
          })
        }

        completeListener = EventEmitter.addListener(EVENT_ON_COMPLETE, (evt: any) => {
          const { contextId, requestId: evtRequestId, result } = evt
          if (contextId !== this.id || evtRequestId !== requestId) return

          // Clean up listeners
          tokenListener?.remove()
          completeListener?.remove()

          // Check if there's an error in the result (e.g., state load failure)
          if (result.error) {
            reject(new Error(result.error))
          } else {
            // Resolve the promise
            resolve(result as NativeCompletionResult)
          }
        })
      })

      // Create stop function
      const stop = async () => {
        await RNLlama.cancelRequest(this.id, requestId)
        // Clean up listeners
        tokenListener?.remove()
        completeListener?.remove()
      }

      return {
        requestId,
        promise,
        stop,
      }
    },

    /**
     * Queue an embedding request for parallel processing (non-blocking)
     * @param text Text to embed
     * @param params Optional embedding parameters
     * @returns Promise resolving to object with requestId and promise (resolves to embedding result)
     */
    embedding: async (text: string, params?: EmbeddingParams): Promise<{
      requestId: number
      promise: Promise<NativeEmbeddingResult>
    }> => {
      let embeddingListener: any

      const { requestId } = await RNLlama.queueEmbedding(this.id, text, params || {})

      // Create promise that resolves when embedding completes
      const promise = new Promise<NativeEmbeddingResult>((resolve, _reject) => {
        embeddingListener = EventEmitter.addListener(EVENT_ON_EMBEDDING_RESULT, (evt: any) => {
          const { contextId, requestId: evtRequestId, embedding } = evt
          // Filter by both contextId AND requestId to ensure correct matching
          if (contextId !== this.id || evtRequestId !== requestId) return

          // Clean up listener
          embeddingListener?.remove()

          // Resolve the promise
          resolve({ embedding })
        })
      })

      return {
        requestId,
        promise,
      }
    },

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
    }> => {
      let rerankListener: any

      const { requestId } = await RNLlama.queueRerank(this.id, query, documents, params || {})

      // Create promise that resolves when reranking completes
      const promise = new Promise<RerankResult[]>((resolve, _reject) => {
        rerankListener = EventEmitter.addListener(EVENT_ON_RERANK_RESULTS, (evt: any) => {
          const { contextId, requestId: evtRequestId, results } = evt
          // Filter by both contextId AND requestId to ensure correct matching
          if (contextId !== this.id || evtRequestId !== requestId) return

          // Clean up listener
          rerankListener?.remove()

          // Sort by score descending (highest score first = most relevant) and add document text
          const sortedResults = results
            .map((result: NativeRerankResult) => ({
              ...result,
              document: documents[result.index],
            }))
            .sort((a: RerankResult, b: RerankResult) => b.score - a.score)

          // Resolve the promise
          resolve(sortedResults)
        })
      })

      return {
        requestId,
        promise,
      }
    },

    /**
     * Enable parallel decoding mode
     *
     * Note: The context must be initialized with a sufficient n_parallel value to support
     * the requested number of slots. By default, contexts are initialized with n_parallel=8,
     * which supports up to 8 parallel slots. To use more slots, specify a higher n_parallel
     * value when calling initLlama().
     *
     * @param params Configuration for parallel mode
     * @param params.n_parallel Number of parallel slots (default: 2). Must be <= context's n_seq_max
     * @param params.n_batch Batch size for processing (default: 512)
     * @returns Promise resolving to true if successful
     *
     * @example
     * // Initialize context with support for up to 16 parallel slots
     * const context = await initLlama({ model: 'model.gguf', n_parallel: 16 })
     *
     * // Enable parallel mode with 4 slots
     * await context.parallel.enable({ n_parallel: 4 })
     *
     * // Later, reconfigure to use 8 slots
     * await context.parallel.configure({ n_parallel: 8 })
     */
    enable: (config?: { n_parallel?: number; n_batch?: number }) =>
      RNLlama.enableParallelMode(this.id, { enabled: true, ...config }),

    /**
     * Disable parallel decoding mode
     * @returns Promise resolving to true if successful
     */
    disable: () => RNLlama.enableParallelMode(this.id, { enabled: false }),

    /**
     * Configure parallel decoding mode (enables if not already enabled)
     * @param config Configuration for parallel mode
     * @param config.n_parallel Number of parallel slots (default: 2)
     * @param config.n_batch Batch size for processing (default: 512)
     * @returns Promise resolving to true if successful
     */
    configure: (config: { n_parallel?: number; n_batch?: number }) =>
      RNLlama.enableParallelMode(this.id, { enabled: true, ...config }),
  }

  constructor({ contextId, gpu, gpuDevice, reasonNoGPU, model, androidLib }: NativeLlamaContext) {
    this.id = contextId
    this.gpu = gpu
    this.gpuDevice = gpuDevice
    this.reasonNoGPU = reasonNoGPU
    this.model = model
    this.androidLib = androidLib
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
      tool_choice?: string,
      enable_thinking?: boolean,
      add_generation_prompt?: boolean,
      now?: string | number,
      chat_template_kwargs?: Record<string, string>,
    },
  ): Promise<FormattedChatResult | JinjaFormattedChatResult> {
    const mediaPaths: string[] = []
    const chat = messages.map((msg) => {
      if (Array.isArray(msg.content)) {
        const content = msg.content.map((part) => {
          // Handle multimodal content
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

    const useJinja = this.isJinjaSupported() && params?.jinja
    let tmpl
    if (template) tmpl = template // Force replace if provided
    const jsonSchema = getJsonSchema(params?.response_format)

    const result = await RNLlama.getFormattedChat(
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
        now: typeof params?.now === 'number' ? params.now.toString() : params?.now,
        chat_template_kwargs: params?.chat_template_kwargs ? JSON.stringify(
          Object.entries(params.chat_template_kwargs).reduce((acc, [key, value]) => {
            acc[key] = JSON.stringify(value) // Each value is a stringified JSON object
            return acc
          }, {} as Record<string, any>)
        ) : undefined,
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

  /**
   * Generate a completion based on the provided parameters
   * @param params Completion parameters including prompt or messages
   * @param callback Optional callback for token-by-token streaming
   * @returns Promise resolving to the completion result
   *
   * Note: For multimodal support, you can include an media_paths parameter.
   * This will process the images and add them to the context before generating text.
   * Multimodal support must be enabled via initMultimodal() first.
   */
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

    // If media_paths were explicitly provided or extracted from messages, use them
    if (!nativeParams.media_paths && params.media_paths) {
      nativeParams.media_paths = params.media_paths
    }

    if (nativeParams.response_format && !nativeParams.grammar) {
      const jsonSchema = getJsonSchema(params.response_format)
      if (jsonSchema) nativeParams.json_schema = JSON.stringify(jsonSchema)
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

  /**
   * Tokenize text or text with images
   * @param text Text to tokenize
   * @param params.media_paths Array of image paths to tokenize (if multimodal is enabled)
   * @returns Promise resolving to the tokenize result
   */
  tokenize(
    text: string,
    {
      media_paths: mediaPaths,
    }: {
      media_paths?: string[]
    } = {},
  ): Promise<NativeTokenizeResult> {
    return RNLlama.tokenize(this.id, text, mediaPaths)
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

  /**
   * Rerank documents based on relevance to a query
   * @param query The query text to rank documents against
   * @param documents Array of document texts to rank
   * @param params Optional reranking parameters
   * @returns Promise resolving to an array of ranking results with scores and indices
   */
  async rerank(
    query: string,
    documents: string[],
    params?: RerankParams,
  ): Promise<RerankResult[]> {
    const results = await RNLlama.rerank(this.id, query, documents, params || {})

    // Sort by score descending and add document text if requested
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
    const result = await RNLlama.bench(this.id, pp, tg, pl, nr)
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

  /**
   * Initialize multimodal support with a mmproj file
   * @param params Parameters for multimodal support
   * @param params.path Path to the multimodal projector file
   * @param params.use_gpu Whether to use GPU
   * @returns Promise resolving to true if initialization was successful
   */
  async initMultimodal({
    path,
    use_gpu: useGpu,
  }: {
    path: string
    use_gpu?: boolean
  }): Promise<boolean> {
    if (path.startsWith('file://')) path = path.slice(7)
    return RNLlama.initMultimodal(this.id, {
      path,
      use_gpu: useGpu ?? true,
    })
  }

  /**
   * Check if multimodal support is enabled
   * @returns Promise resolving to true if multimodal is enabled
   */
  async isMultimodalEnabled(): Promise<boolean> {
    return await RNLlama.isMultimodalEnabled(this.id)
  }

  /**
   * Check multimodal support
   * @returns Promise resolving to an object with vision and audio support
   */
  async getMultimodalSupport(): Promise<{
    vision: boolean
    audio: boolean
  }> {
    return await RNLlama.getMultimodalSupport(this.id)
  }

  /**
   * Release multimodal support
   * @returns Promise resolving to void
   */
  async releaseMultimodal(): Promise<void> {
    return await RNLlama.releaseMultimodal(this.id)
  }

  /**
   * Initialize TTS support with a vocoder model
   * @param params Parameters for TTS support
   * @param params.path Path to the vocoder model
   * @param params.n_batch Batch size for the vocoder model
   * @returns Promise resolving to true if initialization was successful
   */
  async initVocoder({ path, n_batch: nBatch }: { path: string; n_batch?: number }): Promise<boolean> {
    if (path.startsWith('file://')) path = path.slice(7)
    return await RNLlama.initVocoder(this.id, { path, n_batch: nBatch })
  }

  /**
   * Check if TTS support is enabled
   * @returns Promise resolving to true if TTS is enabled
   */
  async isVocoderEnabled(): Promise<boolean> {
    return await RNLlama.isVocoderEnabled(this.id)
  }

  /**
   * Get a formatted audio completion prompt
   * @param speakerJsonStr JSON string representing the speaker
   * @param textToSpeak Text to speak
   * @returns Promise resolving to the formatted audio completion result with prompt and grammar
   */
  async getFormattedAudioCompletion(
    speaker: object | null,
    textToSpeak: string,
  ): Promise<{
    prompt: string
    grammar?: string
  }> {
    return await RNLlama.getFormattedAudioCompletion(
      this.id,
      speaker ? JSON.stringify(speaker) : '',
      textToSpeak,
    )
  }

  /**
   * Get guide tokens for audio completion
   * @param textToSpeak Text to speak
   * @returns Promise resolving to the guide tokens
   */
  async getAudioCompletionGuideTokens(
    textToSpeak: string,
  ): Promise<Array<number>> {
    return await RNLlama.getAudioCompletionGuideTokens(this.id, textToSpeak)
  }

  /**
   * Decode audio tokens
   * @param tokens Array of audio tokens
   * @returns Promise resolving to the decoded audio tokens
   */
  async decodeAudioTokens(tokens: number[]): Promise<Array<number>> {
    return await RNLlama.decodeAudioTokens(this.id, tokens)
  }

  /**
   * Release TTS support
   * @returns Promise resolving to void
   */
  async releaseVocoder(): Promise<void> {
    return await RNLlama.releaseVocoder(this.id)
  }

  async release(): Promise<void> {
    return RNLlama.releaseContext(this.id)
  }
}

export async function toggleNativeLog(enabled: boolean): Promise<void> {
  return RNLlama.toggleNativeLog(enabled)
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
  return RNLlama.setContextLimit(limit)
}

let contextIdCounter = 0
const contextIdRandom = () =>
  /* @ts-ignore */
  process.env.NODE_ENV === 'test' ? 0 : Math.floor(Math.random() * 100000)

const modelInfoSkip = [
  // Large fields
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
  'tokenizer.ggml.scores',
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

  if (rest.cache_type_k && !validCacheTypes.includes(rest.cache_type_k)) {
    console.warn(`[RNLlama] initLlama: Invalid cache K type: ${rest.cache_type_k}, falling back to f16`)
    delete rest.cache_type_k
  }
  if (rest.cache_type_v && !validCacheTypes.includes(rest.cache_type_v)) {
    console.warn(`[RNLlama] initLlama: Invalid cache V type: ${rest.cache_type_v}, falling back to f16`)
    delete rest.cache_type_v
  }

  const {
    gpu,
    gpuDevice,
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
    gpuDevice,
    reasonNoGPU,
    model: modelDetails,
    androidLib,
  })
}

export async function releaseAllLlama(): Promise<void> {
  return RNLlama.releaseAllContexts()
}

export async function getBackendDevicesInfo(): Promise<Array<NativeBackendDeviceInfo>> {
  const jsonString = await RNLlama.getBackendDevicesInfo()
  return JSON.parse(jsonString as string)
}

export const BuildInfo = {
  number: BUILD_NUMBER,
  commit: BUILD_COMMIT,
}
