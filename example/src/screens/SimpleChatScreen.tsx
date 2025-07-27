import React, { useState, useEffect, useLayoutEffect, useRef } from 'react'
import { View, Text, ScrollView, Alert } from 'react-native'
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
import { MessagesModal } from '../components/MessagesModal'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import type { LLMMessage } from '../utils/llmMessages'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 9)

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful, harmless, and honest AI assistant. Be concise and helpful in your responses.'

// Using shared styles, keeping only component-specific styles if needed
const styles = {
  container: CommonStyles.container,
  setupContainer: CommonStyles.setupContainer,
  scrollContent: CommonStyles.scrollContent,
  setupDescription: CommonStyles.setupDescription,
}

export default function SimpleChatScreen({ navigation }: { navigation: any }) {
  const messagesRef = useRef<MessageType.Any[]>([])
  const [, setMessagesVersion] = useState(0) // For UI updates
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showMessagesModal, setShowMessagesModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)

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

  const buildLLMMessages = (): LLMMessage[] => {
    const conversationMessages: LLMMessage[] = [
      {
        role: 'system',
        content: systemPrompt,
      },
    ]

    // Add previous messages from chat history
    const recentMessages = messagesRef.current
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
      }))

    return [...conversationMessages, ...recentMessages]
  }

  // Set up navigation header buttons
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton
              title="Messages"
              onPress={() => setShowMessagesModal(true)}
            />
            <HeaderButton
              title="Params"
              onPress={() => setShowCompletionParamsModal(true)}
            />
          </View>
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton
              title="Context Params"
              onPress={() => setShowContextParamsModal(true)}
            />
          </View>
        ),
      })
    }
  }, [navigation, isModelReady])

  const addMessage = (message: MessageType.Any) => {
    messagesRef.current = [message, ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }

  const updateMessage = (
    messageId: string,
    updateFn: (msg: MessageType.Any) => MessageType.Any,
  ) => {
    const index = messagesRef.current.findIndex((msg) => msg.id === messageId)
    if (index >= 0) {
      messagesRef.current = messagesRef.current.map((msg, i) => {
        if (i === index) {
          return updateFn(msg)
        }
        return msg
      })
      setMessagesVersion((prev) => prev + 1)
    }
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

  const handleImportMessages = (newMessages: MessageType.Any[]) => {
    // Reset messages and add system message back
    messagesRef.current = []
    setMessagesVersion((prev) => prev + 1)

    // Add the initial system message
    addSystemMessage(
      "Hello! I'm ready to chat with you. How can I help you today?",
    )

    // Add imported messages
    messagesRef.current = [...newMessages.reverse(), ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }

  const handleUpdateSystemPrompt = (newSystemPrompt: string) => {
    setSystemPrompt(newSystemPrompt)
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
      // Build conversation messages using the reusable function
      const conversationMessages = buildLLMMessages()

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
      const completionResult = await context.completion(
        {
          messages: conversationMessages,
          ...params,
        },
        (data) => {
          const { token } = data
          response += token

          updateMessage(responseId, (msg) => {
            if (msg.type === 'text') {
              return {
                ...msg,
                text: response.replace(/^\s+/, ''),
              }
            }
            return msg
          })
        },
      )

      // Update final message with timing info
      updateMessage(responseId, (msg) => {
        if (msg.type === 'text') {
          return {
            ...msg,
            text: completionResult.content,
            metadata: {
              ...msg.metadata,
              timings: 'Response completed',
            },
          }
        }
        return msg
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
        messages={messagesRef.current}
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

      <MessagesModal
        visible={showMessagesModal}
        onClose={() => setShowMessagesModal(false)}
        messages={buildLLMMessages()}
        context={context}
        onImportMessages={handleImportMessages}
        onUpdateSystemPrompt={handleUpdateSystemPrompt}
        defaultSystemPrompt={DEFAULT_SYSTEM_PROMPT}
      />
    </SafeAreaView>
  )
}
