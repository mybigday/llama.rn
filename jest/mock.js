const { NativeModules } = require('react-native')

if (!NativeModules.RNLlama) {
  const demoEmbedding = new Array(768).fill(0.01)
  const benchJson =
    '{"n_kv_max":2048,"n_batch":2048,"n_ubatch":512,"flash_attn":0,"is_pp_shared":0,"n_gpu_layers":99,"n_threads":8,"n_threads_batch":8,"pp":128,"tg":128,"pl":1,"n_kv":256,"t_pp":0.23381,"speed_pp":547.453064,"t_tg":3.503684,"speed_tg":36.532974,"t":3.737494,"speed":68.495094}'

  const contextMap = {}
  const vocoderMap = {}
  const multimodalMap = {}
  let requestIdCounter = 0
  const getNextRequestId = () => {
    requestIdCounter += 1
    return requestIdCounter
  }

  const ensureJSIFunctions = () => {
    if (global.llamaInitContext) return

    const completionProbabilities = [
      {
        content: ' *',
        probs: [
          { prob: 0.9658700227737427, tok_str: ' *' },
          { prob: 0.021654844284057617, tok_str: ' Hi' },
          { prob: 0.012475099414587021, tok_str: ' Hello' },
        ],
      },
      {
        content: 'g',
        probs: [
          { prob: 0.5133139491081238, tok_str: 'g' },
          { prob: 0.3046242296695709, tok_str: 'ch' },
          { prob: 0.18206188082695007, tok_str: 'bl' },
        ],
      },
      {
        content: 'igg',
        probs: [
          { prob: 0.9886618852615356, tok_str: 'igg' },
          { prob: 0.008458126336336136, tok_str: 'ig' },
          { prob: 0.002879939740523696, tok_str: 'reet' },
        ],
      },
      {
        content: 'les',
        probs: [
          { prob: 1, tok_str: 'les' },
          { prob: 1.8753286923356427e-8, tok_str: 'ling' },
          { prob: 3.312444318837038e-9, tok_str: 'LES' },
        ],
      },
      {
        content: '*',
        probs: [
          { prob: 1, tok_str: '*' },
          { prob: 4.459857905203535e-8, tok_str: '*.' },
          { prob: 3.274198334679568e-8, tok_str: '**' },
        ],
      },
    ]

    const tokenEvents = []
    let cumulative = ''
    completionProbabilities.forEach((entry) => {
      cumulative += entry.content
      tokenEvents.push({
        token: entry.content,
        content: cumulative,
        completion_probabilities: entry.probs,
      })
    })

    const completionResult = {
      audio_tokens: [
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
      ],
      completion_probabilities: completionProbabilities,
      content: '*giggles*',
      text: '*giggles*',
      stopped_eos: true,
      stopped_limit: false,
      stopped_word: false,
      stopping_word: '',
      tokens_cached: 54,
      tokens_evaluated: 15,
      tokens_predicted: 6,
      truncated: false,
      timings: {
        predicted_ms: 1330.6290000000001,
        predicted_n: 5,
        predicted_per_second: 16.533534140620713,
        predicted_per_token_ms: 60.48313636363637,
        prompt_ms: 3805.6730000000002,
        prompt_n: 5,
        prompt_per_second: 8.408499626741445,
        prompt_per_token_ms: 118.92728125000001,
      },
    }

    const setGlobal = (name, fn) => {
      Object.defineProperty(global, name, {
        value: fn,
        writable: true,
        configurable: true,
        enumerable: true,
      })
    }

    const mockInitContextResult = {
      gpu: false,
      reasonNoGPU: 'mock',
      model: {
        nEmbd: 768,
        metadata: {
          'general.architecture': 'llama',
          'llama.embedding_length': 768,
        },
        chatTemplates: {
          llamaChat: true,
          minja: {
            default: true,
            defaultCaps: {
              parallelToolCalls: false,
              systemRole: true,
              toolCallId: false,
              toolCalls: false,
              toolResponses: false,
              tools: false,
            },
            toolUse: false,
          },
        },
      },
      devices: [],
      systemInfo: 'mock',
    }

    setGlobal(
      'llamaInitContext',
      jest.fn(async (contextId, _params, onProgress) => {
        if (typeof onProgress === 'function') {
          onProgress(0)
          onProgress(50)
          onProgress(100)
        }
        contextMap[contextId] = true
        return mockInitContextResult
      }),
    )
    setGlobal(
      'llamaReleaseContext',
      jest.fn(async (contextId) => {
        contextMap[contextId] = false
      }),
    )
    setGlobal(
      'llamaReleaseAllContexts',
      jest.fn(async () => {
        Object.keys(contextMap).forEach((contextId) => {
          contextMap[contextId] = false
        })
      }),
    )
    setGlobal(
      'llamaModelInfo',
      jest.fn(async () => ({})),
    )
    setGlobal(
      'llamaGetBackendDevicesInfo',
      jest.fn(async () => '[]'),
    )
    setGlobal(
      'llamaLoadSession',
      jest.fn(async () => ({ tokens_loaded: 0, prompt: '' })),
    )
    setGlobal(
      'llamaSaveSession',
      jest.fn(async () => 0),
    )
    setGlobal(
      'llamaTokenize',
      jest.fn(async () => ({
        tokens: [],
        has_media: false,
        bitmap_hashes: [],
        chunk_pos: [],
        chunk_pos_media: [],
      })),
    )
    setGlobal(
      'llamaDetokenize',
      jest.fn(async () => ''),
    )
    setGlobal(
      'llamaGetFormattedChat',
      jest.fn(async () => ({
        type: 'llama-chat',
        prompt: '',
        has_media: false,
        media_paths: [],
      })),
    )
    setGlobal(
      'llamaEmbedding',
      jest.fn(async () => ({ embedding: demoEmbedding })),
    )
    setGlobal(
      'llamaRerank',
      jest.fn(async () => [
        { score: 0.9, index: 0 },
        { score: 0.5, index: 1 },
        { score: 0.2, index: 2 },
      ]),
    )
    setGlobal(
      'llamaBench',
      jest.fn(async () => benchJson),
    )
    setGlobal(
      'llamaCompletion',
      jest.fn(async (_ctx, _params, onToken) => {
        if (typeof onToken === 'function') {
          tokenEvents.forEach((event) => onToken({ ...event }))
        }
        return { ...completionResult }
      }),
    )
    setGlobal(
      'llamaStopCompletion',
      jest.fn(async () => {}),
    )
    setGlobal(
      'llamaApplyLoraAdapters',
      jest.fn(async () => {}),
    )
    setGlobal(
      'llamaRemoveLoraAdapters',
      jest.fn(async () => {}),
    )
    setGlobal(
      'llamaGetLoadedLoraAdapters',
      jest.fn(async () => []),
    )
    setGlobal(
      'llamaInitMultimodal',
      jest.fn(async (contextId) => {
        multimodalMap[contextId] = true
        return true
      }),
    )
    setGlobal(
      'llamaIsMultimodalEnabled',
      jest.fn(async (contextId) => !!multimodalMap[contextId]),
    )
    setGlobal(
      'llamaGetMultimodalSupport',
      jest.fn(async () => ({ vision: true, audio: true })),
    )
    setGlobal(
      'llamaReleaseMultimodal',
      jest.fn(async (contextId) => {
        delete multimodalMap[contextId]
      }),
    )
    setGlobal(
      'llamaInitVocoder',
      jest.fn(async (contextId) => {
        vocoderMap[contextId] = true
        return true
      }),
    )
    setGlobal(
      'llamaIsVocoderEnabled',
      jest.fn(async (contextId) => !!vocoderMap[contextId]),
    )
    setGlobal(
      'llamaGetFormattedAudioCompletion',
      jest.fn(async () => ({ prompt: '', grammar: '' })),
    )
    setGlobal(
      'llamaGetAudioCompletionGuideTokens',
      jest.fn(async () => []),
    )
    setGlobal(
      'llamaDecodeAudioTokens',
      jest.fn(async () => []),
    )
    setGlobal(
      'llamaReleaseVocoder',
      jest.fn(async (contextId) => {
        delete vocoderMap[contextId]
      }),
    )
    setGlobal(
      'llamaEnableParallelMode',
      jest.fn(async () => true),
    )
    setGlobal(
      'llamaQueueCompletion',
      jest.fn(async (_ctx, _params, onToken, onComplete) => {
        const reqId = getNextRequestId()
        if (typeof onToken === 'function') {
          tokenEvents.forEach((event) => onToken({ ...event }, reqId))
        }
        if (typeof onComplete === 'function')
          onComplete({ ...completionResult })
        return { requestId: reqId }
      }),
    )
    setGlobal(
      'llamaCancelRequest',
      jest.fn(async () => {}),
    )
    setGlobal(
      'llamaQueueEmbedding',
      jest.fn(async (_ctx, _text, _params, onResult) => {
        const reqId = getNextRequestId()
        if (typeof onResult === 'function') onResult([...demoEmbedding])
        return { requestId: reqId }
      }),
    )
    setGlobal(
      'llamaQueueRerank',
      jest.fn(async (_ctx, _query, documents, _params, onResult) => {
        const reqId = getNextRequestId()
        const results = (documents || []).map((_, index) => ({
          index,
          score: 1 - index * 0.1,
        }))
        if (typeof onResult === 'function') onResult(results)
        return { requestId: reqId }
      }),
    )
    setGlobal(
      'llamaToggleNativeLog',
      jest.fn(async (enabled, onLog) => {
        if (enabled && typeof onLog === 'function') {
          onLog('info', 'mock log')
        }
      }),
    )
    setGlobal(
      'llamaSetContextLimit',
      jest.fn(async () => {}),
    )
    setGlobal(
      'llamaClearCache',
      jest.fn(async () => {}),
    )
  }

  NativeModules.RNLlama = {
    install: jest.fn(async () => {
      ensureJSIFunctions()
      return true
    }),
  }
}

module.exports = jest.requireActual('llama.rn')
