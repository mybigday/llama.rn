import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Chat, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { initLlama, LlamaContext } from '../../../src'
import ModelDownloadCard from '../components/ModelDownloadCard'
import { Bubble } from '../components/Bubble'
import { MODELS } from '../utils/constants'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 9)

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    textAlign: 'center',
  },
  setupContainer: {
    flex: 1,
    padding: 16,
  },
  setupDescription: {
    fontSize: 16,
    color: '#666',
    lineHeight: 24,
    marginBottom: 24,
    textAlign: 'center',
  },
  loadingContainer: {
    alignItems: 'center',
    marginTop: 24,
  },
  loadingText: {
    marginTop: 8,
    fontSize: 16,
    color: '#666',
  },
  progressContainer: {
    marginTop: 16,
    width: '100%',
    alignItems: 'center',
  },
  progressBar: {
    width: '80%',
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
})

export default function SimpleChatScreen() {
  const [messages, setMessages] = useState<MessageType.Any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)

  useEffect(
    () => () => {
      if (context) {
        context.release()
      }
    },
    [context],
  )

  const addMessage = (message: MessageType.Any) => {
    setMessages((msgs) => [message, ...msgs])
  }

  const addSystemMessage = (text: string, metadata = {}) => {
    const textMessage: MessageType.Text = {
      author: assistant,
      createdAt: Date.now(),
      id: randId(),
      text,
      type: 'text',
      metadata: { system: true, ...metadata },
    }
    addMessage(textMessage)
    return textMessage.id
  }

  const initializeModel = async (modelPath: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const llamaContext = await initLlama({
        model: modelPath,
        n_ctx: 4096,
        n_gpu_layers: 99, // Use GPU acceleration on iOS
        use_mlock: true,
        use_mmap: true,
      }, (progress) => {
        // Progress is reported as 1 to 100
        setInitProgress(progress)
      })

      setContext(llamaContext)
      setIsModelReady(true)
      setInitProgress(100)

      // Add welcome message
      addSystemMessage(
        "Hello! I'm ready to chat with you. How can I help you today?",
      )
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleSendPress = async (message: MessageType.PartialText) => {
    if (!context || isLoading) return

    const userMessage: MessageType.Text = {
      author: user,
      createdAt: Date.now(),
      id: randId(),
      text: message.text,
      type: 'text',
    }

    addMessage(userMessage)
    setIsLoading(true)

    try {
      // Build messages array for conversation context
      const conversationMessages = [
        {
          role: 'system' as const,
          content:
            'You are a helpful, harmless, and honest AI assistant. Be concise and helpful in your responses.',
        },
        // Add previous messages from chat history
        ...messages
          .filter(
            (msg): msg is MessageType.Text =>
              msg.type === 'text' && !msg.metadata?.system,
          )
          .reverse() // Reverse to get chronological order
          .slice(-10) // Keep last 10 messages for context
          .map((msg) => ({
            role:
              msg.author.id === user.id
                ? ('user' as const)
                : ('assistant' as const),
            content: msg.text,
          })),
        // Add current user message
        {
          role: 'user' as const,
          content: message.text,
        },
      ]

      let response = ''
      const responseId = randId()
      const responseMessage: MessageType.Text = {
        author: assistant,
        createdAt: Date.now(),
        id: responseId,
        text: '',
        type: 'text',
      }

      addMessage(responseMessage)

      await context.completion(
        {
          messages: conversationMessages,
          n_predict: 512,
          temperature: 0.7,
          top_p: 0.9,
          stop: ['<|im_end|>', '<end_of_turn>']
        },
        (data) => {
          const { token } = data
          response += token

          setMessages((currentMsgs) => {
            const index = currentMsgs.findIndex((msg) => msg.id === responseId)
            if (index >= 0) {
              return currentMsgs.map((msg, i) => {
                if (msg.type === 'text' && i === index) {
                  return {
                    ...msg,
                    text: response.replace(/^\s+/, ''),
                  }
                }
                return msg
              })
            }
            return currentMsgs
          })
        },
      )

      // Update final message with timing info
      setMessages((currentMsgs) => {
        const index = currentMsgs.findIndex((msg) => msg.id === responseId)
        if (index >= 0) {
          return currentMsgs.map((msg, i) => {
            if (msg.type === 'text' && i === index) {
              return {
                ...msg,
                metadata: {
                  ...msg.metadata,
                  timings: 'Response completed',
                },
              }
            }
            return msg
          })
        }
        return currentMsgs
      })
    } catch (error: any) {
      Alert.alert('Error', `Failed to generate response: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const renderBubble = ({
    child,
    message,
  }: {
    child: React.ReactNode
    message: MessageType.Any
  }) => <Bubble child={child} message={message} />

  if (!isModelReady) {
    return (
      <SafeAreaView style={styles.container}>
        <ScrollView style={styles.setupContainer}>
          <Text style={styles.setupDescription}>
            Download the model to start chatting. This model provides
            fast, efficient text generation for conversational AI.
          </Text>

          {['SMOL_LM', 'GEMMA_3N'].map((model) => {
            const modelInfo = MODELS[model as keyof typeof MODELS]
            return (
              <ModelDownloadCard
                key={model}
                title={modelInfo.name}
                repo={modelInfo.repo}
                filename={modelInfo.filename}
                size={modelInfo.size}
                onInitialize={initializeModel}
              />
            )
          })}

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>
                {`Initializing model... ${initProgress}%`}
              </Text>
              {initProgress > 0 && (
                <View style={styles.progressContainer}>
                  <View style={styles.progressBar}>
                    <View
                      style={[
                        styles.progressFill,
                        { width: `${initProgress}%` }
                      ]}
                    />
                  </View>
                </View>
              )}
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={styles.container}>
      <Chat
        renderBubble={renderBubble}
        theme={defaultTheme}
        messages={messages}
        onSendPress={handleSendPress}
        user={user}
        textInputProps={{
          editable: !isLoading,
          placeholder: isLoading
            ? 'AI is thinking...'
            : 'Type your message here',
        }}
      />
    </SafeAreaView>
  )
}
