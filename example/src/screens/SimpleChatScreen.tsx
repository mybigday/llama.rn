import React, { useState, useRef, useCallback } from 'react'
import { View, Alert } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { Chat } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { Bubble } from '../components/Bubble'
import { Menu } from '../components/Menu'
import { MessagesModal } from '../components/MessagesModal'
import SessionModal from '../components/SessionModal'
import { StopButton } from '../components/StopButton'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import {
  createThemedStyles,
  chatDarkTheme,
  chatLightTheme,
} from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import { initLlama } from '../../../src' // import 'llama.rn'
import {
  useStoredCompletionParams,
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { useExampleContext } from '../hooks/useExampleContext'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import { createExampleModelDefinitions } from '../utils/exampleModels'
import {
  CHAT_ASSISTANT,
  CHAT_USER,
  buildConversationMessages,
  createMessageId,
  createSystemTextMessage,
  createUserTextMessage,
} from '../features/chatHelpers'

const DEFAULT_SYSTEM_PROMPT =
  'You are a helpful, harmless, and honest AI assistant. Be concise and helpful in your responses.'

const SIMPLE_CHAT_MODELS = createExampleModelDefinitions([
  'SMOL_LM_3',
  'GEMMA_3_4B_QAT',
  'QWEN_3_4B',
  'GEMMA_3N_E2B',
  'GEMMA_3N_E4B',
])

export default function SimpleChatScreen({ navigation }: { navigation: any }) {
  const { isDark, theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const messagesRef = useRef<MessageType.Any[]>([])
  const [, setMessagesVersion] = useState(0) // For UI updates
  const [isLoading, setIsLoading] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showMessagesModal, setShowMessagesModal] = useState(false)
  const [showSessionModal, setShowSessionModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const insets = useSafeAreaInsets()
  const {
    context,
    initProgress,
    isModelReady,
    replaceContext,
    setInitProgress,
  } = useExampleContext()
  const { value: contextParams, setValue: setContextParams } =
    useStoredContextParams()
  const { value: completionParams, setValue: setCompletionParams } =
    useStoredCompletionParams()
  const { value: customModels, reload: reloadCustomModels } =
    useStoredCustomModels()

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const buildLLMMessages = () =>
    buildConversationMessages(messagesRef.current, systemPrompt, CHAT_USER.id)

  const addMessage = useCallback((message: MessageType.Any) => {
    messagesRef.current = [message, ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }, [])

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

  const addSystemMessage = useCallback(
    (text: string, metadata = {}) => {
      const textMessage = createSystemTextMessage(text, metadata)
      addMessage(textMessage)
      return textMessage.id
    },
    [addMessage],
  )

  const handleReset = useCallback(() => {
    Alert.alert(
      'Reset Chat',
      'Are you sure you want to clear all messages? This action cannot be undone.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            messagesRef.current = []
            if (context) {
              await context.clearCache(false)
            }
            setMessagesVersion((prev) => prev + 1)
            addSystemMessage(
              "Hello! I'm ready to chat with you. How can I help you today?",
            )
          },
        },
      ],
    )
  }, [addSystemMessage, context])

  useExampleScreenHeader({
    navigation,
    isModelReady,
    readyActions: [
      {
        key: 'reset',
        iconName: 'refresh',
        onPress: handleReset,
      },
      {
        key: 'completion-settings',
        iconName: 'cog-outline',
        onPress: () => setShowCompletionParamsModal(true),
      },
    ],
    setupActions: [
      {
        key: 'context-settings',
        iconName: 'cog-outline',
        onPress: () => setShowContextParamsModal(true),
      },
    ],
    renderReadyExtras: () => (
      <Menu
        actions={[
          {
            id: 'messages',
            title: 'Messages',
            onPress: () => setShowMessagesModal(true),
          },
          {
            id: 'sessions',
            title: 'Sessions',
            onPress: () => setShowSessionModal(true),
          },
        ]}
      />
    ),
  })

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
          // devices: ['HTP0'],
          // devices: ['HTP0', 'HTP1', 'CPU'],
          // devices: ['HTP*'],
        },
        (progress) => {
          // Progress is reported as 1 to 100
          setInitProgress(progress)
        },
      )

      await replaceContext(llamaContext)
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
      ...createUserTextMessage(message.text),
    }

    addMessage(userMessage)
    setIsLoading(true)

    try {
      // Build conversation messages using the reusable function
      const conversationMessages = buildLLMMessages()

      const responseId = createMessageId()
      const responseMessage: MessageType.Text = {
        author: CHAT_ASSISTANT,
        createdAt: Date.now(),
        id: responseId,
        text: '',
        type: 'text',
      }

      addMessage(responseMessage)

      const params = completionParams || (await loadCompletionParams())
      const completionResult = await context.completion(
        {
          ...params,
          messages: conversationMessages,
          reasoning_format: 'auto',
        },
        (data) => {
          const { content = '', reasoning_content: reasoningContent } = data

          updateMessage(responseId, (msg) => {
            if (msg.type === 'text') {
              return {
                ...msg,
                text: content.replace(/^\s+/, ''),
                metadata: {
                  ...msg.metadata,
                  partialCompletionResult: {
                    reasoning_content: reasoningContent,
                    content: content.replace(/^\s+/, ''),
                  },
                },
              }
            }
            return msg
          })
        },
      )

      const content = completionResult.interrupted
        ? completionResult.text
        : completionResult.content

      // Update final message with timing info
      updateMessage(responseId, (msg) => {
        if (msg.type === 'text') {
          return {
            ...msg,
            text: content,
            metadata: {
              ...msg.metadata,
              timings: completionResult.timings,
              completionResult,
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
      <>
        <ExampleModelSetup
          description="Download the model to start chatting. This model provides fast, efficient text generation for conversational AI."
          defaultModels={SIMPLE_CHAT_MODELS}
          customModels={customModels || []}
          onInitializeCustomModel={(_model, modelPath) =>
            initializeModel(modelPath)
          }
          onInitializeModel={(_model, modelPath) => initializeModel(modelPath)}
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          isLoading={isLoading}
          initProgress={initProgress}
          progressText={`Initializing model... ${initProgress}%`}
        />

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />
      </>
    )
  }

  return (
    <View style={themedStyles.container}>
      {/* @ts-ignore */}
      <Chat
        renderBubble={renderBubble}
        theme={isDark ? chatDarkTheme : chatLightTheme}
        messages={messagesRef.current}
        onSendPress={handleSendPress}
        user={CHAT_USER}
        textInputProps={{
          editable: !isLoading,
          placeholder: isLoading
            ? 'AI is thinking...'
            : 'Type your message here',
          keyboardType: 'ascii-capable',
        }}
      />

      <StopButton context={context} insets={insets} isLoading={isLoading} />

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
        completionParams={completionParams}
        onImportMessages={handleImportMessages}
        onUpdateSystemPrompt={handleUpdateSystemPrompt}
        defaultSystemPrompt={DEFAULT_SYSTEM_PROMPT}
      />

      <SessionModal
        visible={showSessionModal}
        onClose={() => setShowSessionModal(false)}
        context={context}
      />
    </View>
  )
}
