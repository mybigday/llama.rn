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
  JinjaFormattedChatResult,
  FormattedChatResult,
  NativeImageProcessingResult,
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
  JinjaFormattedChatResult,
  NativeImageProcessingResult,

  // Deprecated
  SchemaGrammarConverterPropOrder,
  SchemaGrammarConverterBuiltinRule,
}

export { SchemaGrammarConverter, convertJsonSchemaToGrammar }

const EVENT_ON_INIT_CONTEXT_PROGRESS = '@RNLlama_onInitContextProgress'
const EVENT_ON_TOKEN = '@RNLlama_onToken'
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
  image_paths?: string | string[]
}
export type CompletionParams = Omit<
  NativeCompletionParams,
  'emit_partial_completion' | 'prompt'
> &
  CompletionBaseParams

export type BenchResult = {
  modelDesc: string
  modelSize: number
  modelNParams: number
  ppAvg: number
  ppStd: number
  tgAvg: number
  tgStd: number
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

  model: NativeLlamaContext['model']

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
    },
  ): Promise<FormattedChatResult> {
    const imagePaths: string[] = []
    const processedMessages = messages.map((msg) => {
      if (Array.isArray(msg.content)) {
        const content = msg.content
          .map((part) => {
            // Handle multimodal content
            if (part.type === 'image_url') {
              let path = part.image_url?.url
              if (path?.startsWith('file://')) path = path.slice(7)
              imagePaths.push(path)
              return {
                type: 'text',
                text: '<__image__>',
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
    })

    const chat = formatChat(processedMessages)
    const useJinja = this.isJinjaSupported() && params?.jinja
    let tmpl = this.isLlamaChatSupported() || useJinja ? undefined : 'chatml'
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
      },
    )
    if (!useJinja) {
      return {
        type: 'llama-chat',
        prompt: result as string,
        has_image: imagePaths.length > 0,
        image_paths: imagePaths,
      }
    }
    const jinjaResult = result as JinjaFormattedChatResult
    jinjaResult.type = 'jinja'
    jinjaResult.has_image = imagePaths.length > 0
    jinjaResult.image_paths = imagePaths
    return jinjaResult
  }

  /**
   * Generate a completion based on the provided parameters
   * @param params Completion parameters including prompt or messages
   * @param callback Optional callback for token-by-token streaming
   * @returns Promise resolving to the completion result
   *
   * Note: For multimodal support, you can include an image_paths parameter.
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
        },
      )
      if (formattedResult.type === 'jinja') {
        const jinjaResult = formattedResult as JinjaFormattedChatResult

        nativeParams.prompt = jinjaResult.prompt || ''
        if (typeof jinjaResult.chat_format === 'number')
          nativeParams.chat_format = jinjaResult.chat_format
        if (jinjaResult.grammar)
          nativeParams.grammar = jinjaResult.grammar
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
        if (jinjaResult.has_image) {
          nativeParams.image_paths = jinjaResult.image_paths
        }
      } else if (formattedResult.type === 'llama-chat') {
        const llamaChatResult = formattedResult as FormattedChatResult
        nativeParams.prompt = llamaChatResult.prompt || ''
        if (llamaChatResult.has_image) {
          nativeParams.image_paths = llamaChatResult.image_paths
        }
      }
    } else {
      nativeParams.prompt = params.prompt || ''
    }

    // If image_paths were explicitly provided or extracted from messages, use them
    if (!nativeParams.image_paths && params.image_paths) {
      nativeParams.image_paths = params.image_paths
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

  /**
   * Initialize multimodal support with a mmproj file
   * @param params Parameters for multimodal support
   * @param params.path Path to the multimodal projector file
   * @param params.use_gpu Whether to use GPU
   * @returns Promise resolving to true if initialization was successful
   */
  async initMultimodal(
    { path, use_gpu: useGpu }: { path: string; use_gpu?: boolean },
  ): Promise<boolean> {
    if (path.startsWith('file://')) path = path.slice(7)
    return RNLlama.initMultimodal(this.id, {
      path,
      use_gpu: useGpu || true,
    })
  }

  /**
   * Check if multimodal support is enabled
   * @returns Promise resolving to true if multimodal is enabled
   */
  async isMultimodalEnabled(): Promise<boolean> {
    return RNLlama.isMultimodalEnabled(this.id)
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
