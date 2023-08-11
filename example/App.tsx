import React, { useState, useRef } from 'react'
import { SafeAreaProvider } from 'react-native-safe-area-context'
import DocumentPicker from 'react-native-document-picker'
import type { DocumentPickerResponse } from 'react-native-document-picker'
import { Chat, darkTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
// eslint-disable-next-line import/no-unresolved
import { initLlama, LlamaContext } from 'llama.rn'

const randId = () => Math.random().toString(36).substr(2, 9)

const user = { id: 'y9d7f8pgn' }

const systemId = 'h3o3lc5xj'
const system = { id: systemId }

const initialChatPrompt =
  'This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\n'

const generateChatPrompt = (
  context: LlamaContext | undefined,
  conversationId: string,
  messages: MessageType.Any[],
) => {
  const prompt = [...messages]
    .reverse()
    .map((msg) => {
      if (
        !msg.metadata?.system &&
        msg.metadata?.conversationId === conversationId &&
        msg.metadata?.contextId === context?.id &&
        msg.type === 'text'
      ) {
        return `${msg.author.id === systemId ? 'llama' : 'User'}: ${msg.text}`
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
  return initialChatPrompt + prompt
}

const defaultConversationId = 'default'

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

  const addSystemMessage = (text: string) => {
    const textMessage: MessageType.Text = {
      author: system,
      createdAt: Date.now(),
      id: randId(),
      text,
      type: 'text',
      metadata: { system: true },
    }
    addMessage(textMessage)
  }

  const handleReleaseCont = async () => {
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

  const handleInitContext = async (file: DocumentPickerResponse) => {
    await handleReleaseCont()
    addSystemMessage('Initializing context...')
    initLlama({
      model: file.uri,
      use_mlock: true,
      n_gpu_layers: 0, // > 0: enable metal
    })
      .then((ctx) => {
        setContext(ctx)
        addSystemMessage(
          `Context initialized! \n\nMetal: ${ctx.isMetalEnabled ? 'YES' : 'NO'} (${ctx.reasonNoMetal})\n\n` +
            'You can use the following commands:\n\n' +
            '- /release: release the context\n' +
            '- /stop: stop the current completion\n' +
            '- /reset: reset the conversation',
        )
      })
      .catch((err) => {
        addSystemMessage(`Context initialization failed: ${err.message}`)
      })
  }

  const handlePickModel = async () => {
    DocumentPicker.pick({ type: ['public.archive'] })
      .then(async (res) => {
        const [file] = res
        if (file) handleInitContext(file)
      })
      .catch(() => {
        console.log('No file picked')
      })
  }

  const handleSendPress = async (message: MessageType.PartialText) => {
    if (context) {
      switch (message.text) {
        case '/release':
          await handleReleaseCont()
          return
        case '/stop':
          if (inferencing) context.stopCompletion()
          return
        case '/reset':
          conversationIdRef.current = randId()
          addMessage({
            author: system,
            createdAt: Date.now(),
            id: randId(),
            text: 'Conversation reset!',
            type: 'text',
            metadata: { system: true },
          })
          return
      }
    }
    const textMessage: MessageType.Text = {
      author: user,
      createdAt: Date.now(),
      id: randId(),
      text: message.text,
      type: 'text',
      metadata: {
        contextId: context?.id,
        conversationId: conversationIdRef.current,
      },
    }
    addMessage(textMessage)
    setInferencing(true)

    const id = randId()
    const createdAt = Date.now()
    const prompt = generateChatPrompt(context, conversationIdRef.current, [
      textMessage,
      ...messages,
    ])

    context
      ?.completion(
        {
          prompt: `${prompt}\nllama:`,
          n_predict: 400,
          temperature: 0.7,
          repeat_last_n: 256, // 0 = disable penalty, -1 = context size
          repeat_penalty: 1.18, // 1.0 = disabled
          top_k: 40, // <= 0 to use vocab size
          top_p: 0.5, // 1.0 = disabled
          tfs_z: 1.0, // 1.0 = disabled
          typical_p: 1.0, // 1.0 = disabled
          presence_penalty: 0.0, // 0.0 = disabled
          frequency_penalty: 0.0, // 0.0 = disabled
          mirostat: 0, // 0/1/2
          mirostat_tau: 5, // target entropy
          mirostat_eta: 0.1, // learning rate
          n_probs: 0, // Show probabilities
          stop: ['</s>', 'llama:', 'User:'],
          // n_threads: 4,
          // logit_bias: [[15043,1.0]],
        },
        (data) => {
          const { token } = data
          setMessages((msgs) => {
            const index = msgs.findIndex((msg) => msg.id === id)
            if (index >= 0) {
              return msgs.map((msg, i) => {
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
                type: 'text',
                metadata: { contextId: context?.id },
              },
              ...msgs,
            ]
          })
        },
      )
      .then((completionResult) => {
        console.log('completionResult: ', completionResult)
        setInferencing(false)
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
