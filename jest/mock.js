const { NativeModules, DeviceEventEmitter } = require('react-native')

if (!NativeModules.RNLlama) {
  const demoEmbedding = new Array(768).fill(0.01)

  const contextMap = {}
  const vocoderMap = {}
  let requestIdCounter = 0
  const getNextRequestId = () => {
    requestIdCounter += 1
    return requestIdCounter
  }
  NativeModules.RNLlama = {
    getBackendDevicesInfo: jest.fn(async () => JSON.stringify([{
      backend: 'cpu',
      type: 'cpu',
      deviceName: 'Test',
      maxMemorySize: 12345678,
      metadata: {},
    }])),

    setContextLimit: jest.fn(),

    modelInfo: jest.fn(async () => ({})),

    initContext: jest.fn(() =>
      Promise.resolve({
        gpu: false,
        reasonNoGPU: 'Test',
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
      }),
    ),

    getFormattedChat: jest.fn(
      async (contextId, messagesStr, chatTemplate, options) => {
        const messages = JSON.parse(messagesStr)
        const fullPrompt = (messages || [])
          .map((m) => `${m.role}: ${JSON.stringify(m.content)}`)
          .join('\n')
        if (options.jinja) {
          return { type: 'jinja', prompt: fullPrompt, chat_format: 0 }
        }
        return { type: 'llama-chat', prompt: fullPrompt }
      },
    ),

    completion: jest.fn(async (contextId, jobId) => {
      const testResult = {
        audio_tokens: [
          1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
        ],
        content: '*giggles*',
        text: '*giggles*',
        completion_probabilities: [
          {
            content: ' *',
            probs: [
              {
                prob: 0.9658700227737427,
                tok_str: ' *',
              },
              {
                prob: 0.021654844284057617,
                tok_str: ' Hi',
              },
              {
                prob: 0.012475099414587021,
                tok_str: ' Hello',
              },
            ],
          },
          {
            content: 'g',
            probs: [
              {
                prob: 0.5133139491081238,
                tok_str: 'g',
              },
              {
                prob: 0.3046242296695709,
                tok_str: 'ch',
              },
              {
                prob: 0.18206188082695007,
                tok_str: 'bl',
              },
            ],
          },
          {
            content: 'igg',
            probs: [
              {
                prob: 0.9886618852615356,
                tok_str: 'igg',
              },
              {
                prob: 0.008458126336336136,
                tok_str: 'ig',
              },
              {
                prob: 0.002879939740523696,
                tok_str: 'reet',
              },
            ],
          },
          {
            content: 'les',
            probs: [
              {
                prob: 1,
                tok_str: 'les',
              },
              {
                prob: 1.8753286923356427e-8,
                tok_str: 'ling',
              },
              {
                prob: 3.312444318837038e-9,
                tok_str: 'LES',
              },
            ],
          },
          {
            content: '*',
            probs: [
              {
                prob: 1,
                tok_str: '*',
              },
              {
                prob: 4.459857905203535e-8,
                tok_str: '*.',
              },
              {
                prob: 3.274198334679568e-8,
                tok_str: '**',
              },
            ],
          },
        ],
        stopped_eos: true,
        stopped_limit: false,
        stopped_word: false,
        stopping_word: '',
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
        tokens_cached: 54,
        tokens_evaluated: 15,
        tokens_predicted: 6,
        truncated: false,
      }
      const emitEvent = async (data) => {
        await new Promise((resolve) => setTimeout(resolve))
        DeviceEventEmitter.emit('@RNLlama_onToken', data)
      }
      await testResult.completion_probabilities.reduce(
        (promise, item, index) =>
          promise.then(() => {
            const content = testResult.completion_probabilities.slice(0, index + 1).map((p) => p.content).join('')
            return emitEvent({
              contextId,
              jobId,
              tokenResult: {
                token: item.content,
                content,
                completion_probabilities: item.probs,
              },
            })
          }),
        Promise.resolve(),
      )
      return Promise.resolve(testResult)
    }),

    stopCompletion: jest.fn(),

    tokenize: jest.fn(async (_, content, mediaPaths) => ({
      tokens: content.split(''),
      has_media: mediaPaths?.length > 0,
      chunk_pos: mediaPaths?.length > 0 ? [0] : [],
      chunk_pos_media: mediaPaths?.length > 0 ? [0] : [],
      bitmap_hashes: mediaPaths?.length > 0 ? [0] : [],
    })),
    detokenize: jest.fn(async () => ''),
    embedding: jest.fn(async () => ({ embedding: demoEmbedding })),
    rerank: jest.fn(async () => []),

    loadSession: jest.fn(async () => ({
      tokens_loaded: 1,
      prompt: 'Hello',
    })),
    saveSession: jest.fn(async () => 1),

    bench: jest.fn(
      async () =>
        '{"n_kv_max":2048,"n_batch":2048,"n_ubatch":512,"flash_attn":0,"is_pp_shared":0,"n_gpu_layers":99,"n_threads":8,"n_threads_batch":8,"pp":128,"tg":128,"pl":1,"n_kv":256,"t_pp":0.23381,"speed_pp":547.453064,"t_tg":3.503684,"speed_tg":36.532974,"t":3.737494,"speed":68.495094}',
    ),

    enableParallelMode: jest.fn(async () => true),

    queueCompletion: jest.fn(async (contextId) => {
      const requestId = getNextRequestId()
      const testResult = {
        audio_tokens: [
          1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
        ],
        content: '*giggles*',
        text: '*giggles*',
        completion_probabilities: [
          {
            content: ' *',
            probs: [
              {
                prob: 0.9658700227737427,
                tok_str: ' *',
              },
              {
                prob: 0.021654844284057617,
                tok_str: ' Hi',
              },
              {
                prob: 0.012475099414587021,
                tok_str: ' Hello',
              },
            ],
          },
          {
            content: 'g',
            probs: [
              {
                prob: 0.5133139491081238,
                tok_str: 'g',
              },
              {
                prob: 0.3046242296695709,
                tok_str: 'ch',
              },
              {
                prob: 0.18206188082695007,
                tok_str: 'bl',
              },
            ],
          },
          {
            content: 'igg',
            probs: [
              {
                prob: 0.9886618852615356,
                tok_str: 'igg',
              },
              {
                prob: 0.008458126336336136,
                tok_str: 'ig',
              },
              {
                prob: 0.002879939740523696,
                tok_str: 'reet',
              },
            ],
          },
          {
            content: 'les',
            probs: [
              {
                prob: 1,
                tok_str: 'les',
              },
              {
                prob: 1.8753286923356427e-8,
                tok_str: 'ling',
              },
              {
                prob: 3.312444318837038e-9,
                tok_str: 'LES',
              },
            ],
          },
          {
            content: '*',
            probs: [
              {
                prob: 1,
                tok_str: '*',
              },
              {
                prob: 4.459857905203535e-8,
                tok_str: '*.',
              },
              {
                prob: 3.274198334679568e-8,
                tok_str: '**',
              },
            ],
          },
        ],
        stopped_eos: true,
        stopped_limit: false,
        stopped_word: false,
        stopping_word: '',
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
        tokens_cached: 54,
        tokens_evaluated: 15,
        tokens_predicted: 6,
        truncated: false,
      }

      // Emit tokens asynchronously
      const emitEvent = async (data) => {
        await new Promise((resolve) => setTimeout(resolve, 10))
        DeviceEventEmitter.emit('@RNLlama_onToken', data)
      }

      // Emit all tokens
      testResult.completion_probabilities.reduce(
        (promise, item, index) =>
          promise.then(() => {
            const content = testResult.completion_probabilities
              .slice(0, index + 1)
              .map((p) => p.content)
              .join('')
            return emitEvent({
              contextId,
              requestId,
              tokenResult: {
                token: item.content,
                content,
                completion_probabilities: item.probs,
              },
            })
          }),
        Promise.resolve(),
      )
        .then(() => {
          // Emit completion event after all tokens
          setTimeout(() => {
            DeviceEventEmitter.emit('@RNLlama_onComplete', {
              contextId,
              requestId,
              result: testResult,
            })
          }, 100)
        })

      return { requestId }
    }),

    cancelRequest: jest.fn(async () => {}),

    queueEmbedding: jest.fn(async (contextId) => {
      const requestId = getNextRequestId()

      // Emit embedding result asynchronously
      setTimeout(() => {
        DeviceEventEmitter.emit('@RNLlama_onEmbeddingResult', {
          contextId,
          requestId,
          embedding: demoEmbedding,
        })
      }, 50)

      return { requestId }
    }),

    queueRerank: jest.fn(async (contextId, _query, documents) => {
      const requestId = getNextRequestId()

      // Create mock results with random scores
      const results = documents.map((_, index) => ({
        index,
        score: Math.random(),
      }))

      // Emit rerank results asynchronously
      setTimeout(() => {
        DeviceEventEmitter.emit('@RNLlama_onRerankResults', {
          contextId,
          requestId,
          results,
        })
      }, 50)

      return { requestId }
    }),

    releaseContext: jest.fn(() => Promise.resolve()),
    releaseAllContexts: jest.fn(() => Promise.resolve()),

    // For NativeEventEmitter
    addListener: jest.fn(),
    removeListeners: jest.fn(),

    applyLoraAdapters: jest.fn(async () => {}),
    removeLoraAdapters: jest.fn(async () => {}),
    getLoadedLoraAdapters: jest.fn(async () => []),

    initMultimodal: jest.fn(async (_id) => {
      contextMap[_id] = true
      return true
    }),
    isMultimodalEnabled: jest.fn(async (_id) => contextMap[_id] || false),
    releaseMultimodal: jest.fn(async (_id) => {
      delete contextMap[_id]
    }),

    initVocoder: jest.fn(async (_id) => {
      vocoderMap[_id] = true
      return true
    }),
    releaseVocoder: jest.fn(async (_id) => {
      delete vocoderMap[_id]
    }),
    isVocoderEnabled: jest.fn(async (_id) => vocoderMap[_id] || false),
    getFormattedAudioCompletion: jest.fn(
      async (_id, speakerJsonStr, textToSpeak) => ({
        prompt: `${speakerJsonStr || '<default speaker>'}<sep>${textToSpeak}`,
      }),
    ),
    getAudioCompletionGuideTokens: jest.fn(async (_id, textToSpeak) =>
      textToSpeak.split('').map((char) => char.charCodeAt(0) + 1000),
    ),
    decodeAudioTokens: jest.fn(async (_id, tokens) =>
      tokens.map((token) => token - 1000).map((token) => token / 1024),
    ),
  }
}

module.exports = jest.requireActual('llama.rn')
