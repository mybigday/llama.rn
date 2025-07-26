import React, { useState, useEffect, useLayoutEffect } from 'react'
import { Text, ScrollView, Alert } from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Chat, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { initLlama, LlamaContext } from '../../../src'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { Bubble } from '../components/Bubble'
import { LoadingIndicator } from '../components/LoadingIndicator'
import { ProgressBar } from '../components/ProgressBar'
import { HeaderButton } from '../components/HeaderButton'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 9)

// Using shared styles, keeping only component-specific styles if needed
const styles = {
  container: CommonStyles.container,
  setupContainer: CommonStyles.setupContainer,
  scrollContent: CommonStyles.scrollContent,
  setupDescription: CommonStyles.setupDescription,
}

export default function SimpleChatScreen({ navigation }: { navigation: any }) {
  const [messages, setMessages] = useState<MessageType.Any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)

  useEffect(
    () => () => {
      if (context) {
        context.release()
      }
    },
    [context],
  )

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  // Set up navigation header button
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            title="Params"
            onPress={() => setShowCompletionParamsModal(true)}
          />
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            title="Context Params"
            onPress={() => setShowContextParamsModal(true)}
          />
        ),
      })
    }
  }, [navigation, isModelReady])

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

      const params = contextParams || (await loadContextParams())
      const llamaContext = await initLlama(
        {
          model: modelPath,
          ...params,
        },
        (progress) => {
          // Progress is reported as 1 to 100
          setInitProgress(progress)
        },
      )

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

      const params = completionParams || (await loadCompletionParams())
      await context.completion(
        {
          messages: conversationMessages,
          ...params,
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
        <ScrollView style={styles.setupContainer} contentContainerStyle={styles.scrollContent}>
          <Text style={styles.setupDescription}>
            Download the model to start chatting. This model provides fast,
            efficient text generation for conversational AI.
          </Text>

          {['SMOL_LM', 'GEMMA_3N_E2B', 'GEMMA_3N_E4B', 'GEMMA_3'].map((model) => {
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
            <>
              <LoadingIndicator
                text={`Initializing model... ${initProgress}%`}
              />
              {initProgress > 0 && <ProgressBar progress={initProgress} />}
            </>
          )}
        </ScrollView>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />
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

      <CompletionParamsModal
        visible={showCompletionParamsModal}
        onClose={() => setShowCompletionParamsModal(false)}
        onSave={handleSaveCompletionParams}
      />
    </SafeAreaView>
  )
}
