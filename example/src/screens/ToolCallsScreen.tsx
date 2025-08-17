import React, {
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  useCallback,
} from 'react'
import { z } from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  TouchableOpacity,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { Chat, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
import { Bubble } from '../components/Bubble'
import { HeaderButton } from '../components/HeaderButton'
import { Menu } from '../components/Menu'
import { MessagesModal } from '../components/MessagesModal'
import { MaskedProgress } from '../components/MaskedProgress'
import SessionModal from '../components/SessionModal'
import { StopButton } from '../components/StopButton'
import ToolsModal from '../components/ToolsModal'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type {
  ContextParams,
  CompletionParams,
  CustomModel,
} from '../utils/storage'
import {
  loadContextParams,
  loadCompletionParams,
  loadCustomModels,
} from '../utils/storage'
import type { LLMMessage } from '../utils/llmMessages'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 7)

const DEFAULT_SYSTEM_PROMPT =
  'You are a helpful AI assistant with access to tools. You can call tools to help answer user questions.'

const styles = StyleSheet.create({
  // Using shared styles for common patterns
  container: CommonStyles.container,
  header: {
    ...CommonStyles.header,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: CommonStyles.headerTitle,
  headerSubtitle: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
  },
  setupContainer: CommonStyles.setupContainer,
  scrollContent: CommonStyles.scrollContent,
  setupDescription: CommonStyles.setupDescription,
  loadingContainer: CommonStyles.loadingContainer,
  loadingText: CommonStyles.loadingText,
  progressContainer: CommonStyles.progressContainer,
  progressBar: CommonStyles.progressBar,
  progressFill: CommonStyles.progressFill,
  settingsContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
  settingsButtonStyle: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  settingsButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  settingsButton: {
    color: '#007AFF',
    fontSize: 14,
    textDecorationLine: 'underline',
  },
})

interface ToolCall {
  id: string
  name: string
  arguments: Record<string, any>
}

interface ToolResult {
  id: string
  result: any
  error?: string
}

interface ToolMessage {
  tool_call_id: string
  role: string
  content: string
}

// Generate tool definitions using zodToJsonSchema
const AVAILABLE_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get current weather information for a location',
      parameters: zodToJsonSchema(
        z.object({
          location: z
            .string()
            .describe('The city and country, e.g. "San Francisco, CA"'),
        }),
        { target: 'openApi3' },
      ),
    },
  },
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Perform mathematical calculations',
      parameters: zodToJsonSchema(
        z.object({
          expression: z
            .string()
            .describe('Mathematical expression to evaluate, e.g. "2 + 2 * 3"'),
        }),
        { target: 'openApi3' },
      ),
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_time',
      description: 'Get current time in a specific timezone',
      parameters: zodToJsonSchema(
        z.object({
          timezone: z
            .string()
            .describe('Timezone identifier, e.g. "America/New_York"'),
        }),
        { target: 'openApi3' },
      ),
    },
  },
]

export default function ToolCallsScreen({ navigation }: { navigation: any }) {
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
  const [showSessionModal, setShowSessionModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [showToolsModal, setShowToolsModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])
  const [currentTools, setCurrentTools] = useState(AVAILABLE_TOOLS)
  const [mockResponses, setMockResponses] = useState<Record<string, string>>({
    get_weather: "It's sunny and 72°F in your location with light clouds.",
    calculate: 'The calculation result is 42.',
    get_time: 'The current time is 2:30 PM on Tuesday, January 15, 2025.',
  })
  const insets = useSafeAreaInsets()

  useEffect(
    () => () => {
      if (context) {
        context.release()
      }
    },
    [context],
  )

  // Load custom models on mount
  useEffect(() => {
    const loadCustomModelsData = async () => {
      try {
        const models = await loadCustomModels()
        setCustomModels(models)
      } catch (error) {
        console.error('Error loading custom models:', error)
      }
    }
    loadCustomModelsData()
  }, [])

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const handleCustomModelAdded = async (_model: CustomModel) => {
    // Reload custom models to reflect the new addition
    const models = await loadCustomModels()
    setCustomModels(models)
  }

  const handleCustomModelRemoved = async () => {
    // Reload custom models to reflect the removal
    const models = await loadCustomModels()
    setCustomModels(models)
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
      .reduce((acc: LLMMessage[], msg) => {
        if (msg.metadata?.tool_calls) {
          // This is an assistant message that made tool calls
          acc.push({
            role: 'assistant',
            content: msg.text,
            tool_calls: msg.metadata.tool_calls || [],
          })
        } else if (msg.metadata?.toolMessage) {
          // This contains tool results, add them as individual tool messages
          const { toolMessage } = msg.metadata
          acc.push(toolMessage)
        } else if (msg.author.id === user.id) {
          // Regular user message
          acc.push({
            role: 'user',
            content: msg.text,
          })
        } else if (!msg.metadata?.tool_calls) {
          // Regular assistant message (only if not tool-related)
          acc.push({
            role: 'assistant',
            content: msg.text,
            reasoning_content:
              msg.metadata?.completionResult?.reasoning_content,
          })
        }
        return acc
      }, [])

    return [...conversationMessages, ...recentMessages]
  }

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
          onPress: () => {
            messagesRef.current = []
            setMessagesVersion((prev) => prev + 1)
            addSystemMessage(
              `Hello! I'm a tool-calling AI assistant. You can customize my tools using the tools button in the header. Try asking me something!`,
            )
          },
        },
      ],
    )
  }, [addSystemMessage])

  // Set up navigation header buttons
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton iconName="refresh" onPress={handleReset} />
            <HeaderButton
              iconName="cog-outline"
              onPress={() => setShowCompletionParamsModal(true)}
            />
            <Menu
              actions={[
                {
                  id: 'tools',
                  title: 'Tools',
                  onPress: () => setShowToolsModal(true),
                },
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
          </View>
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            iconName="cog-outline"
            onPress={() => setShowContextParamsModal(true)}
          />
        ),
      })
    }
  }, [navigation, isModelReady, handleReset])

  const handleImportMessages = (newMessages: MessageType.Any[]) => {
    // Reset messages and add system message back
    messagesRef.current = []
    setMessagesVersion((prev) => prev + 1)

    // Add the initial system message
    addSystemMessage(
      `Hello! I'm a tool-calling AI assistant. You can customize my tools using the tools button in the header. Try asking me something!`,
    )

    // Add imported messages
    messagesRef.current = [...newMessages.reverse(), ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }

  const handleUpdateSystemPrompt = (newSystemPrompt: string) => {
    setSystemPrompt(newSystemPrompt)
  }

  const handleSaveTools = (
    tools: any[],
    newMockResponses: Record<string, string>,
  ) => {
    setCurrentTools(tools)
    setMockResponses(newMockResponses)
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

      addSystemMessage(
        `Hello! I'm a tool-calling AI assistant. You can customize my tools using the tools button in the header. Try asking me something!`,
      )
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const executeTool = async (toolCall: ToolCall): Promise<ToolResult> => {
    const { name, arguments: args } = toolCall

    if (!currentTools.find((tool) => tool.function.name === name)) {
      return {
        id: toolCall.id,
        result: `Error: Tool not found: ${name}`,
      }
    }

    // Check if we have a mock response for this tool
    if (mockResponses[name]) {
      return {
        id: toolCall.id,
        result: mockResponses[name],
      }
    }
    return {
      id: toolCall.id,
      result: `Error: Response not implemented for: ${name}(${args})`,
    }
  }

  const performCompletion = async (userMessageText?: string) => {
    if (!context || isLoading) return

    if (userMessageText) {
      const userMessage: MessageType.Text = {
        author: user,
        createdAt: Date.now(),
        id: randId(),
        text: userMessageText,
        type: 'text',
      }
      addMessage(userMessage)
    }

    setIsLoading(true)

    try {
      // Build conversation messages using the reusable function
      const conversationMessages = buildLLMMessages()

      const responseId = randId()
      const responseMessage: MessageType.Text = {
        author: assistant,
        createdAt: Date.now(),
        id: responseId,
        text: '',
        type: 'text',
      }

      addMessage(responseMessage)

      const completionParameters =
        completionParams || (await loadCompletionParams())
      const completionResult = await context.completion(
        {
          ...completionParameters,
          reasoning_format: 'auto',
          messages: conversationMessages,
          tools: currentTools,
          tool_choice: 'auto',
          jinja: true,
        },
        (data) => {
          const {
            content = '',
            reasoning_content: reasoningContent,
            tool_calls: toolCalls,
          } = data

          // Update message with streaming data
          updateMessage(responseId, (msg) => {
            if (msg.type === 'text') {
              return {
                ...msg,
                text: content.replace(/^\s+/, ''),
                metadata: {
                  ...msg.metadata,
                  partialCompletionResult: {
                    reasoning_content: reasoningContent,
                    tool_calls: toolCalls,
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

      // update last message
      updateMessage(responseId, (msg) => {
        // if not tool_calls, update the text
        if (msg.type === 'text' && !msg.metadata?.tool_calls) {
          return {
            ...msg,
            text: content,
            metadata: {
              ...msg.metadata,
              completionResult,
            },
          }
        }
        return msg
      })

      let toolCalls = completionResult.tool_calls || []

      // Handle tool calls if any were made
      if (toolCalls && toolCalls.length > 0) {
        // Ensure all tool calls have IDs
        toolCalls.forEach((toolCall) => {
          if (!toolCall.id) toolCall.id = randId()
        })
        // Unique by id (last one wins)
        toolCalls = toolCalls.filter(
          (toolCall, index, self) =>
            index === self.findLastIndex((t) => t.id === toolCall.id),
        )

        // Update the response message to store tool calls in metadata
        updateMessage(responseId, (msg) => {
          if (msg.type === 'text') {
            return {
              ...msg,
              // NOTE: Special case for Gemma3 - keep content for good response
              text: !content
                ? `Call: ${toolCalls
                    .map(
                      (t: any) => `${t.function.name}(${t.function.arguments})`,
                    )
                    .join(', ')}`
                : content,
              metadata: {
                ...msg.metadata,
                tool_calls: toolCalls,
                completionResult,
              },
            }
          }
          return msg
        })

        // Execute tool calls
        await toolCalls.reduce(async (promise, toolCall) => {
          await promise
          // Ask user for confirmation before executing tools
          const shouldExecute = await new Promise<boolean>((resolve) => {
            Alert.alert(
              'Tool Execution Request',
              `The AI wants to execute tool:\n\n${toolCall.function.name}(${toolCall.function.arguments})`,
              [
                {
                  text: 'Cancel',
                  style: 'cancel',
                  onPress: () => resolve(false),
                },
                {
                  text: 'Allow',
                  onPress: () => resolve(true),
                },
              ],
            )
          })
          let toolMessage: ToolMessage
          if (!shouldExecute) {
            toolMessage = {
              tool_call_id: toolCall.id!,
              role: 'tool',
              content: 'Error: Tool execution was declined by the user',
            }
          } else {
            const result = await executeTool({
              id: toolCall.id!,
              name: toolCall.function.name,
              arguments: JSON.parse(toolCall.function.arguments),
            })
            toolMessage = {
              tool_call_id: toolCall.id!,
              role: 'tool',
              content: result.error
                ? result.error
                : JSON.stringify(result.result),
            }
          }
          addMessage({
            author: assistant,
            createdAt: Date.now() + 2,
            id: randId(),
            text: `📊 Tool Result:\n${toolMessage.content}`,
            type: 'text',
            metadata: { toolMessage },
          })
        }, Promise.resolve())

        // Continue the conversation with tool results by calling performCompletion recursively
        // Wait a bit to let the UI update, then continue recursively
        setTimeout(() => {
          performCompletion()
        }, 1000)
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to generate response: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendPress = async (message: MessageType.PartialText) => {
    await performCompletion(message.text)
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
      <View style={styles.container}>
        <ScrollView
          style={styles.setupContainer}
          contentContainerStyle={styles.scrollContent}
        >
          <Text style={styles.setupDescription}>
            Download a compatible model to demonstrate tool calling
            capabilities. The AI can execute functions like weather queries,
            calculations, and time lookups.
          </Text>

          {/* Custom Models Section */}
          {customModels.filter((model) => !model.mmprojFilename).length > 0 && (
            <>
              <Text style={CommonStyles.modelSectionTitle}>Custom Models</Text>
              {customModels
                .filter((model) => !model.mmprojFilename) // Only show non-multimodal models
                .map((model) => (
                  <CustomModelCard
                    key={model.id}
                    model={model}
                    onInitialize={(modelPath: string) =>
                      initializeModel(modelPath)
                    }
                    onModelRemoved={handleCustomModelRemoved}
                    initializeButtonText="Initialize"
                  />
                ))}
            </>
          )}

          {/* Add Custom Model Button */}
          <TouchableOpacity
            style={CommonStyles.addCustomModelButton}
            onPress={() => setShowCustomModelModal(true)}
          >
            <Text style={CommonStyles.addCustomModelButtonText}>
              + Add Custom Model
            </Text>
          </TouchableOpacity>

          {/* Predefined Models Section */}
          <Text style={CommonStyles.modelSectionTitle}>Default Models</Text>
          {[
            'SMOL_LM_3',
            'GEMMA_3_4B_QAT',
            'QWEN_3_4B',
            'GEMMA_3N_E2B',
            'GEMMA_3N_E4B',
          ].map((model) => {
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
        </ScrollView>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />

        <CustomModelModal
          visible={showCustomModelModal}
          onClose={() => setShowCustomModelModal(false)}
          onModelAdded={handleCustomModelAdded}
          title="Add Custom Model"
          enableFileSelection
        />

        <MaskedProgress
          visible={isLoading}
          text={`Initializing model... ${initProgress}%`}
          progress={initProgress}
          showProgressBar={initProgress > 0}
        />
      </View>
    )
  }

  return (
    <View style={styles.container}>
      <Chat
        renderBubble={renderBubble}
        theme={defaultTheme}
        messages={messagesRef.current}
        onSendPress={handleSendPress}
        user={user}
        textInputProps={{
          editable: !isLoading,
          placeholder: isLoading
            ? 'Responding...'
            : 'Ask me to use tools...',
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
        tools={currentTools}
        context={context}
        onImportMessages={handleImportMessages}
        onUpdateSystemPrompt={handleUpdateSystemPrompt}
        defaultSystemPrompt={DEFAULT_SYSTEM_PROMPT}
      />

      <SessionModal
        visible={showSessionModal}
        onClose={() => setShowSessionModal(false)}
        context={context}
      />

      <ToolsModal
        visible={showToolsModal}
        onClose={() => setShowToolsModal(false)}
        tools={currentTools}
        onSave={handleSaveTools}
        mockResponses={mockResponses}
      />
    </View>
  )
}
