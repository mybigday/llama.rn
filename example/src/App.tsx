import React, { useState, useRef } from 'react'
import type { ReactNode } from 'react'
import { Platform } from 'react-native'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import { pick } from '@react-native-documents/picker'
import type { DocumentPickerResponse } from '@react-native-documents/picker'
import { Chat, darkTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import json5 from 'json5'
import ReactNativeBlobUtil from 'react-native-blob-util'
import type { CompletionParams, LlamaContext } from 'llama.rn'
import {
  initLlama,
  loadLlamaModelInfo,
  toggleNativeLog,
  addNativeLogListener,
  // eslint-disable-next-line import/no-unresolved
} from 'llama.rn'
import { Bubble } from './Bubble'

// Example: Catch logs from llama.cpp
toggleNativeLog(true)
addNativeLogListener((level, text) => {
  // eslint-disable-next-line prefer-const
  let log = (t: string) => t // noop
  // Uncomment to test:
  // ({log} = console)
  log(['[rnllama]', level ? `[${level}]` : '', text].filter(Boolean).join(' '))
})

const { dirs } = ReactNativeBlobUtil.fs

// Example grammar for output JSON
const testGbnf = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\\"" (
    [^"\\\\\\x7F\\x00-\\x1F] |
    "\\\\" (["\\\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\\n" [ \\t]{0,20}`

const randId = () => Math.random().toString(36).substr(2, 9)

const user = { id: 'y9d7f8pgn' }

const systemId = 'h3o3lc5xj'
const system = { id: systemId }

const systemMessage = {
  role: 'system',
  content:
    'This is a conversation between user and assistant, a friendly chatbot.\n\n',
}

const defaultConversationId = 'default'

const renderBubble = ({
  child,
  message,
}: {
  child: ReactNode
  message: MessageType.Any
}) => <Bubble child={child} message={message} />

export default function App() {
  const [context, setContext] = useState<LlamaContext | undefined>(undefined)
  const [inferencing, setInferencing] = useState<boolean>(false)
  const [messages, setMessages] = useState<MessageType.Any[]>([])

  const conversationIdRef = useRef<string>(defaultConversationId)

  const addMessage = (message: MessageType.Any, batching = false) => {
    if (batching) {
      // This can avoid the message duplication in a same batch
      setMessages([message, ...messages])
    } else {
      setMessages((msgs) => [message, ...msgs])
    }
  }

  const addSystemMessage = (text: string, metadata = {}) => {
    const textMessage: MessageType.Text = {
      author: system,
      createdAt: Date.now(),
      id: randId(),
      text,
      type: 'text',
      metadata: { system: true, ...metadata },
    }
    addMessage(textMessage)
    return textMessage.id
  }

  const handleReleaseContext = async () => {
    if (!context) return
    addSystemMessage('Releasing context...')
    context
      .release()
      .then(() => {
        setContext(undefined)
        addSystemMessage('Context released!')
      })
      .catch((err) => {
        addSystemMessage(`Context release failed: ${err}`)
      })
  }

  const handleReleaseMultimodal = async () => {
    if (!context) return
    addSystemMessage('Releasing multimodal context...')
    context
      .releaseMultimodal()
      .then(() => {
        addSystemMessage('Multimodal context released!')
      })
      .catch((err) => {
        addSystemMessage(`Multimodal context release failed: ${err}`)
      })
  }

  const handleReleaseVocoder = async () => {
    if (!context) return
    addSystemMessage('Releasing vocoder context...')
    context
      .releaseVocoder()
      .then(() => {
        addSystemMessage('Vocoder context released!')
      })
      .catch((err) => {
        addSystemMessage(`Vocoder context release failed: ${err}`)
      })
  }

  // Example: Get model info without initializing context
  const getModelInfo = async (model: string): Promise<any> => {
    const t0 = Date.now()
    const info = await loadLlamaModelInfo(model)
    console.log(`Model info (took ${Date.now() - t0}ms): `, info)
    return info
  }

  const handleInitMultimodal = async (
    ctx: LlamaContext,
    file: DocumentPickerResponse,
  ) => {
    if (!ctx) return
    addSystemMessage('Initializing multimodal support...')
    const success = await ctx.initMultimodal({
      path: file.uri,
      use_gpu: true,
    })
    if (success) {
      addSystemMessage('Multimodal support initialized successfully!')
    } else {
      addSystemMessage('Failed to initialize multimodal support.')
    }
  }
  const handleInitVocoder = async (
    ctx: LlamaContext,
    file: DocumentPickerResponse,
  ) => {
    if (!ctx) return
    addSystemMessage('Initializing Vocoder support...')
    const success = await ctx.initVocoder({ path: file.uri })
    if (success) {
      addSystemMessage('Vocoder support initialized successfully!')
    } else {
      addSystemMessage('Failed to initialize Vocoder support.')
    }
  }
  const handleInitContext = async (
    file: DocumentPickerResponse,
    loraFile: DocumentPickerResponse | null,
  ) => {
    await handleReleaseContext()
    const info = await getModelInfo(file.uri)
    const msgId = addSystemMessage('Initializing context...')
    const t0 = Date.now()
    initLlama(
      {
        model: file.uri,
        use_mlock: true,
        lora_list: loraFile ? [{ path: loraFile.uri, scaled: 1.0 }] : undefined, // Or lora: loraFile?.uri,

        // Currently only for iOS
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0,
        // no_gpu_devices: true, // (iOS only)

        ctx_shift: false,
        // OuteTTS recommends 8192
        n_ctx: info['general.basename']?.startsWith('OuteTTS') ? 8192 : 512,
      },
      (progress) => {
        setMessages((msgs) => {
          const index = msgs.findIndex((msg) => msg.id === msgId)
          if (index >= 0) {
            return msgs.map((msg, i) => {
              if (msg.type == 'text' && i === index) {
                return {
                  ...msg,
                  text: `Initializing context... ${progress}%`,
                }
              }
              return msg
            })
          }
          return msgs
        })
      },
    )
      .then(async (ctx) => {
        const t1 = Date.now()
        setContext(ctx)

        // Create commands list
        const commands = [
          '- /info: to get the model info',
          '- /bench: to benchmark the model',
          '- /release: release the context',
          '- /stop: stop the current completion',
          '- /reset: reset the conversation',
          '- /save-session: save the session tokens',
          '- /load-session: load the session tokens',
          '- /lora: apply a lora adapter',
          '- /remove-lora: remove all lora adapters',
          '- /lora-list: list all lora adapters',
          '- /mtmd: pick mmproj file and initialize multimodal support',
          '- /vocoder: pick vocoder model file and initialize TTS support',
        ]

        addSystemMessage(
          `Context initialized!\n\n` +
            `Load time: ${t1 - t0}ms\n` +
            `GPU: ${ctx.gpu ? 'YES' : 'NO'} (${ctx.reasonNoGPU})\n` +
            `Chat Template: ${
              ctx.model.chatTemplates.llamaChat ? 'YES' : 'NO'
            }\n` +
            `Multimodal: NO (use /mtmd to enable)\n` +
            `Vocoder: NO (use /vocoder to enable)\n\n` +
            `You can use the following commands:\n\n${commands.join('\n')}`,
        )
      })
      .catch((err) => {
        addSystemMessage(`Context initialization failed: ${err.message}`)
      })
  }

  const copyFileIfNeeded = async (
    type = 'model',
    file: DocumentPickerResponse,
  ) => {
    if (Platform.OS === 'android' && file.uri.startsWith('content://')) {
      const dir = `${ReactNativeBlobUtil.fs.dirs.CacheDir}/${type}s`
      const filepath = `${dir}/${file.uri.split('/').pop() || type}.gguf`

      if (!(await ReactNativeBlobUtil.fs.isDir(dir)))
        await ReactNativeBlobUtil.fs.mkdir(dir)

      if (await ReactNativeBlobUtil.fs.exists(filepath))
        return { uri: filepath } as DocumentPickerResponse

      await ReactNativeBlobUtil.fs.unlink(dir) // Clean up old files in models

      addSystemMessage(`Copying ${type} to internal storage...`)
      await ReactNativeBlobUtil.MediaCollection.copyToInternal(
        file.uri,
        filepath,
      )
      addSystemMessage(`${type} copied!`)
      return { uri: filepath } as DocumentPickerResponse
    }
    return file
  }

  const pickLora = async () => {
    let loraFile
    const loraRes = await pick({
      type: Platform.OS === 'ios' ? 'public.data' : 'application/octet-stream',
    }).catch((e) => console.log('No lora file picked, error: ', e.message))
    if (loraRes?.[0]) loraFile = await copyFileIfNeeded('lora', loraRes[0])
    return loraFile
  }

  const pickMmproj = async () => {
    let mmProjFile
    const mmProjRes = await pick({
      type: Platform.OS === 'ios' ? 'public.data' : 'application/octet-stream',
    }).catch((e) => console.log('No mmproj file picked, error: ', e.message))
    if (mmProjRes?.[0])
      mmProjFile = await copyFileIfNeeded('mmproj', mmProjRes[0])
    return mmProjFile
  }

  const pickVocoderModel = async () => {
    let vocoderModelFile
    const vocoderModelRes = await pick({
      type: Platform.OS === 'ios' ? 'public.data' : 'application/octet-stream',
    }).catch((e) =>
      console.log('No vocoder model file picked, error: ', e.message),
    )
    if (vocoderModelRes?.[0])
      vocoderModelFile = await copyFileIfNeeded('vocoder', vocoderModelRes[0])
    return vocoderModelFile
  }

  // We'll use this function directly in the /image command handler

  const handlePickModel = async () => {
    const modelRes = await pick().catch((e) =>
      console.log('No model file picked, error: ', e.message),
    )
    if (!modelRes?.[0]) return
    const modelFile = await copyFileIfNeeded('model', modelRes?.[0])

    let loraFile: any = null
    // Example: Apply lora adapter (Currently only select one lora file) (Uncomment to use)
    // loraFile = await pickLora()
    loraFile = null

    handleInitContext(modelFile, loraFile)
  }

  const handleSendPress = async (message: MessageType.PartialText) => {
    if (!context) return
    switch (message.text) {
      case '/info':
        addSystemMessage(
          `// Model Info\n${json5.stringify(context.model, null, 2)}`,
          { copyable: true },
        )
        return
      case '/bench':
        addSystemMessage('Heating up the model...')
        const t0 = Date.now()
        await context.bench(8, 4, 1, 1)
        const tHeat = Date.now() - t0
        if (tHeat > 1e4) {
          addSystemMessage('Heat up time is too long, please try again.')
          return
        }
        addSystemMessage(`Heat up time: ${tHeat}ms`)

        addSystemMessage('Benchmarking the model...')
        const {
          modelDesc,
          modelSize,
          modelNParams,
          ppAvg,
          ppStd,
          tgAvg,
          tgStd,
        } = await context.bench(512, 128, 1, 3)

        const size = `${(modelSize / 1024.0 / 1024.0 / 1024.0).toFixed(2)} GiB`
        const nParams = `${(modelNParams / 1e9).toFixed(2)}B`
        const md =
          '| model | size | params | test | t/s |\n' +
          '| --- | --- | --- | --- | --- |\n' +
          `| ${modelDesc} | ${size} | ${nParams} | pp 512 | ${ppAvg.toFixed(
            2,
          )} ± ${ppStd.toFixed(2)} |\n` +
          `| ${modelDesc} | ${size} | ${nParams} | tg 128 | ${tgAvg.toFixed(
            2,
          )} ± ${tgStd.toFixed(2)}`
        addSystemMessage(md, { copyable: true })
        return
      case '/release':
        await handleReleaseContext()
        return
      case '/mtmd':
        const mmProjFile = await pickMmproj()
        if (mmProjFile) {
          await handleInitMultimodal(context, mmProjFile)
          // Update commands after successful initialization
          const { vision, audio } = await context.getMultimodalSupport()
          const additionalCommands = [
            '- /release-mtmd: release the multimodal context',
            ...(vision
              ? ['- /image <content>: pick an image and start a conversation']
              : []),
            ...(audio
              ? ['- /audio <content>: pick an audio and start a conversation']
              : []),
          ]
          addSystemMessage(
            `Additional commands available:\n${additionalCommands.join('\n')}`,
          )
        } else {
          addSystemMessage('No mmproj file selected.')
        }
        return
      case '/release-mtmd':
        await handleReleaseMultimodal()
        return
      case '/vocoder':
        const vocoderModelFile = await pickVocoderModel()
        if (vocoderModelFile) {
          await handleInitVocoder(context, vocoderModelFile)
          const isTTSEnabled = await context.isVocoderEnabled()
          if (isTTSEnabled) {
            const additionalCommands = [
              '- /tts <text>: to speak something',
              '- /release-vocoder: to release TTS support',
            ]
            addSystemMessage(
              `TTS enabled! Additional commands available:\n${additionalCommands.join(
                '\n',
              )}`,
            )
          }
        } else {
          addSystemMessage('No vocoder model file selected.')
        }
        return
      case '/release-vocoder':
        await handleReleaseVocoder()
        return
      case '/stop':
        if (inferencing) context.stopCompletion()
        return
      case '/reset':
        conversationIdRef.current = randId()
        addSystemMessage('Conversation reset!')
        return
      case '/save-session':
        context
          .saveSession(`${dirs.DocumentDir}/llama-session.bin`)
          .then((tokensSaved) => {
            console.log('Session tokens saved:', tokensSaved)
            addSystemMessage(`Session saved! ${tokensSaved} tokens saved.`)
          })
          .catch((e) => {
            console.log('Session save failed:', e)
            addSystemMessage(`Session save failed: ${e.message}`)
          })
        return
      case '/load-session':
        context
          .loadSession(`${dirs.DocumentDir}/llama-session.bin`)
          .then((details) => {
            console.log('Session loaded:', details)
            addSystemMessage(
              `Session loaded! ${details.tokens_loaded} tokens loaded.`,
            )
          })
          .catch((e) => {
            console.log('Session load failed:', e)
            addSystemMessage(`Session load failed: ${e.message}`)
          })
        return
      case '/lora':
        pickLora()
          .then((loraFile) => {
            if (loraFile) context.applyLoraAdapters([{ path: loraFile.uri }])
          })
          .then(() => context.getLoadedLoraAdapters())
          .then((loraList) =>
            addSystemMessage(
              `Loaded lora adapters: ${JSON.stringify(loraList)}`,
            ),
          )
        return
      case '/remove-lora':
        context.removeLoraAdapters().then(() => {
          addSystemMessage('Lora adapters removed!')
        })
        return
      case '/lora-list':
        context.getLoadedLoraAdapters().then((loraList) => {
          addSystemMessage(`Loaded lora adapters: ${JSON.stringify(loraList)}`)
        })
        return
    }

    let textContent = message.text
    let hasMedia = false
    let mediaType: string | null = null
    let mediaFormat: string | null = null
    let mediaPath = null
    // Handle /image <content> or /audio <content>
    // TODO: File picker button without slash command
    if (
      message.text.startsWith('/image ') ||
      message.text.startsWith('/audio ')
    ) {
      mediaType = message.text.startsWith('/image ') ? 'image' : 'audio'

      textContent = message.text.slice(7)

      // Check if multimodal can be enabled
      if (!(await context.isMultimodalEnabled())) {
        addSystemMessage(
          'Multimodal support is not enabled. Please initialize a model with a mmproj file.',
        )
        return
      }

      try {
        const imageTypes =
          Platform.OS === 'ios'
            ? ['public.jpeg', 'public.png', 'public.gif', 'com.microsoft.bmp']
            : ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
        const audioTypes =
          Platform.OS === 'ios'
            ? ['public.wav', 'public.mp3']
            : ['audio/wav', 'audio/mpeg']

        // Use DocumentPicker to pick an image or audio
        // The underlying stb_image library in llama.cpp supports these formats:
        // Image: JPEG, PNG, BMP, PSD (Photoshop), TGA, GIF, HDR, PIC (Softimage), PNM
        // Audio: WAV, MP3
        // We limit to the most common formats (we perhaps need to add webp by converting it to supported formats):
        const mediaRes = await pick({
          type: mediaType === 'image' ? imageTypes : audioTypes,
        }).catch((e) => {
          console.log(`No ${mediaType} picked, error: `, e.message)
          return null
        })

        if (!mediaRes?.[0]) {
          addSystemMessage(`No ${mediaType} selected.`)
          return
        }

        // Get the file extension from the URI
        const getFileExtension = (uri: string) => {
          const match = uri.match(/\.([\dA-Za-z]+)$/)
          return match && match[1] ? match[1].toLowerCase() : ''
        }

        // Map file extension to MIME type
        const getMimeType = (extension: string) => {
          const mimeTypes: { [key: string]: string } = {
            // Image
            jpg: 'image/jpeg',
            jpeg: 'image/jpeg',
            png: 'image/png',
            gif: 'image/gif',
            bmp: 'image/bmp',
            psd: 'image/vnd.adobe.photoshop',
            tga: 'image/x-tga',
            hdr: 'image/vnd.radiance',
            pic: 'image/x-softimage-pic',
            ppm: 'image/x-portable-pixmap',
            pgm: 'image/x-portable-graymap',

            // Audio
            wav: 'audio/wav',
            mp3: 'audio/mpeg',
          }
          if (mediaType === 'image') {
            return mimeTypes[extension] || 'image/jpeg' // Default to jpeg if unknown
          } else if (mediaType === 'audio') {
            return mimeTypes[extension] || 'audio/wav' // Default to wav if unknown
          }
          return ''
        }

        // Read image path as base64
        const mediaBase64 =
          mediaRes[0].uri &&
          (await ReactNativeBlobUtil.fs
            .readFile(mediaRes[0].uri.replace('file://', ''), 'base64')
            .catch((e) => {
              console.log('Error reading image:', e.message)
              return null
            }))

        console.log(`${mediaType} path:`, mediaRes[0].uri)

        // Get the extension and MIME type
        // Although, the stb_image library doesn't rely on MIME types but instead detects formats by examining
        // the file's binary header/signature (magic numbers)
        // From stb_image.h:
        /*
           // test the formats with a very explicit header first (at least a FOURCC
           // or distinctive magic number first)
        */
        // Still, let's get this right.
        mediaFormat = getFileExtension(mediaRes[0].uri)
        const mimeType = getMimeType(mediaFormat)

        mediaPath = mediaBase64
          ? `data:${mimeType};base64,${mediaBase64}`
          : mediaRes[0].uri
        hasMedia = true
      } catch (error) {
        addSystemMessage(
          `Error selecting ${mediaType}: ${
            error instanceof Error ? error.message : String(error)
          }`,
        )
      }
    }

    let audioParams: Partial<CompletionParams> | null = null
    if (message.text.startsWith('/tts ')) {
      const text = message.text.slice(5)
      const prompt = await context.getFormattedAudioCompletion(null, text)
      const guideTokens = await context.getAudioCompletionGuideTokens(text)
      audioParams = {
        prompt,
        guide_tokens: guideTokens,
        temperature: 0.1,
        penalty_repeat: 1.1,
        n_predict: 4096,
        messages: undefined,
      }
    }

    const textMessage: MessageType.Text = {
      author: user,
      createdAt: Date.now(),
      id: randId(),
      text: textContent,
      type: 'text',
      metadata: {
        contextId: context?.id,
        conversationId: conversationIdRef.current,
        hasMedia,
        mediaType,
        mediaFormat,
        mediaPath,
      },
    }

    const id = randId()
    const createdAt = Date.now()
    const msgs = [
      systemMessage,
      ...[...messages]
        .reverse()
        .map((msg) => {
          if (
            !msg.metadata?.system &&
            msg.metadata?.conversationId === conversationIdRef.current &&
            msg.metadata?.contextId === context?.id &&
            msg.type === 'text'
          ) {
            return {
              role: msg.author.id === systemId ? 'assistant' : 'user',
              content: msg.metadata?.hasMedia
                ? [
                    {
                      type: 'text',
                      text: msg.text,
                    },
                    msg.metadata?.mediaType === 'image'
                      ? {
                          type: 'image_url',
                          image_url: { url: msg.metadata?.mediaPath },
                        }
                      : {
                          type: 'input_audio',
                          input_audio: {
                            data: msg.metadata?.mediaPath,
                            format: msg.metadata?.mediaFormat,
                          },
                        },
                  ]
                : msg.text,
            }
          }
          return { role: '', content: '' }
        })
        .filter((msg) => msg.role),
      {
        role: 'user',
        content: hasMedia
          ? [
              {
                type: 'text',
                text: textContent,
              },
              mediaType === 'image'
                ? {
                    type: 'image_url',
                    image_url: { url: mediaPath },
                  }
                : {
                    type: 'input_audio',
                    input_audio: { data: mediaPath, format: mediaFormat },
                  },
            ]
          : textContent,
      },
    ]
    addMessage(textMessage)
    setInferencing(true)

    let responseFormat
    {
      // Test JSON Schema
      responseFormat = {
        type: 'json_schema',
        json_schema: {
          schema: {
            oneOf: [
              {
                type: 'object',
                properties: {
                  function: { const: 'create_event' },
                  arguments: {
                    type: 'object',
                    properties: {
                      title: { type: 'string' },
                      date: { type: 'string' },
                      time: { type: 'string' },
                    },
                    required: ['title', 'date'],
                  },
                },
                required: ['function', 'arguments'],
              },
              {
                type: 'object',
                properties: {
                  function: { const: 'image_search' },
                  arguments: {
                    type: 'object',
                    properties: {
                      query: { type: 'string' },
                    },
                    required: ['query'],
                  },
                },
                required: ['function', 'arguments'],
              },
            ],
          },
        },
      }
      // Comment to test:
      responseFormat = undefined
    }

    let grammar
    {
      // Test grammar (It will override responseFormat)
      grammar = testGbnf
      // Comment to test:
      grammar = undefined
    }

    let jinjaParams: any = {}
    // Test jinja & tools
    {
      jinjaParams = {
        jinja: true,
        response_format: responseFormat,
        tool_choice: 'auto',
        tools: [
          {
            type: 'function',
            function: {
              name: 'ipython',
              description:
                'Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.',
              parameters: {
                type: 'object',
                properties: {
                  code: {
                    type: 'string',
                    description: 'The code to run in the ipython interpreter.',
                  },
                },
                required: ['code'],
              },
            },
          },
        ],
        enable_thinking: true,
      }
      // Comment to test:
      jinjaParams = { jinja: true, enable_thinking: true }
    }

    // Test area
    {
      // Test tokenize
      const formatted = await context.getFormattedChat(msgs, null, jinjaParams)
      const prompt =
        typeof formatted === 'string' ? formatted : formatted.prompt
      const t0 = Date.now()
      const tokenizeResult = await context.tokenize(prompt, {
        media_paths: formatted.media_paths,
      })
      const t1 = Date.now()
      console.log(
        'Formatted:',
        formatted,
        '\nTokenize:',
        tokenizeResult,
        `(${tokenizeResult.tokens?.length} tokens, ${t1 - t0}ms})`,
      )

      // Test embedding
      // await context.embedding(prompt).then((result) => {
      //   console.log('Embedding:', result)
      // })

      // Test detokenize
      // await context.detokenize(tokens).then((result) => {
      //   console.log('Detokenize:', result)
      // })
    }

    console.log('msgs: ', msgs)
    context
      .completion(
        {
          messages: msgs,
          n_predict: 2048,

          response_format: responseFormat,
          grammar,
          // If use deepseek r1 distill
          reasoning_format: 'deepseek',
          ...jinjaParams,

          seed: -1,
          n_probs: 0,

          // Sampling params
          top_k: 40,
          top_p: 0.5,
          min_p: 0.05,
          xtc_probability: 0.5,
          xtc_threshold: 0.1,
          typical_p: 1.0,
          temperature: 0.7,
          penalty_last_n: 64,
          penalty_repeat: 1.0,
          penalty_freq: 0.0,
          penalty_present: 0.0,
          dry_multiplier: 0,
          dry_base: 1.75,
          dry_allowed_length: 2,
          dry_penalty_last_n: -1,
          dry_sequence_breakers: ['\n', ':', '"', '*'],
          mirostat: 0,
          mirostat_tau: 5,
          mirostat_eta: 0.1,
          ignore_eos: false,
          stop: [
            '</s>',
            '<|end|>',
            '<|eot_id|>',
            '<|end_of_text|>',
            '<|im_end|>',
            '<|EOT|>',
            '<|END_OF_TURN_TOKEN|>',
            '<|end_of_turn|>',
            '<|endoftext|>',
            '<end_of_turn>',
            '<eos>',
            '<｜end▁of▁sentence｜>',
          ],
          // n_threads: 4,
          // logit_bias: [[15043,1.0]],

          ...audioParams,
        },
        (data) => {
          const { token } = data
          setMessages((currentMsgs) => {
            const index = currentMsgs.findIndex((msg) => msg.id === id)
            if (index >= 0) {
              return currentMsgs.map((msg, i) => {
                if (msg.type == 'text' && i === index) {
                  return {
                    ...msg,
                    text: (msg.text + token).replace(/^\s+/, ''),
                  }
                }
                return msg
              })
            }
            return [
              {
                author: system,
                createdAt,
                id,
                text: token,
                type: audioParams ? 'custom' : 'text',
                metadata: {
                  contextId: context?.id,
                  conversationId: conversationIdRef.current,
                },
              },
              ...currentMsgs,
            ]
          })
        },
      )
      .then((completionResult) => {
        console.log('completionResult: ', completionResult)
        const timings = `${completionResult.timings.predicted_per_token_ms.toFixed()}ms per token, ${completionResult.timings.predicted_per_second.toFixed(
          2,
        )} tokens per second`
        if (!audioParams) {
          setMessages((currentMsgs) => {
            const index = currentMsgs.findIndex((msg) => msg.id === id)
            if (index >= 0) {
              return currentMsgs.map((msg, i) => {
                if (msg.type == 'text' && i === index) {
                  return {
                    ...msg,
                    metadata: {
                      ...msg.metadata,
                      timings,
                    },
                  }
                }
                return msg
              })
            }
            return currentMsgs
          })
          setInferencing(false)

          return
        } else {
          return context
            .decodeAudioTokens(completionResult.audio_tokens!)
            .then((audio) => {
              console.log('audio length', audio.length / 24000)

              setMessages((currentMsgs) => {
                const index = currentMsgs.findIndex((msg) => msg.id === id)
                if (index >= 0) {
                  return currentMsgs.map((msg, i) => {
                    if (msg.type == 'custom' && i === index) {
                      return {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          audio,
                          sr: 24000,
                          timings,
                        },
                      }
                    }
                    return msg
                  })
                }
                return currentMsgs
              })
              setInferencing(false)
            })
        }
      })
      .catch((e) => {
        console.log('completion error: ', e)
        setInferencing(false)
        addSystemMessage(`Completion failed: ${e.message}`)
      })
  }

  return (
    <SafeAreaProvider>
      <Chat
        renderBubble={renderBubble}
        theme={darkTheme}
        messages={messages}
        onSendPress={handleSendPress}
        user={user}
        onAttachmentPress={!context ? handlePickModel : undefined}
        textInputProps={{
          editable: !!context,
          placeholder: !context
            ? 'Press the file icon to pick a model'
            : 'Type your message here',
        }}
      />
    </SafeAreaProvider>
  )
}
