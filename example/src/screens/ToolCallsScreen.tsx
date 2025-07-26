import React, { useState, useEffect } from 'react'
import { z } from 'zod'
import { zodToJsonSchema } from 'zod-to-json-schema'
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
import { MODELS } from '../utils/constants'
import { Bubble } from '../Bubble'

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
  headerSubtitle: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 4,
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

export default function ToolCallsScreen() {
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

      const llamaContext = await initLlama(
        {
          model: modelPath,
          n_ctx: 4096,
          n_gpu_layers: 99,
          use_mlock: true,
          use_mmap: true,
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
        `Hello! I'm a tool-calling AI assistant. I can help you with:\n\n• Weather information ("What's the weather in New York?")\n• Calculations ("Calculate 15 * 24 + 37")\n• Time queries ("What time is it in Tokyo?")\n\nTry asking me something!`,
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

    try {
      switch (name) {
        case 'get_weather':
          // Simulate weather API call
          const weather = {
            location: args.location,
            temperature: Math.floor(Math.random() * 30) + 10,
            condition: ['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'][
              Math.floor(Math.random() * 4)
            ],
            humidity: Math.floor(Math.random() * 50) + 30,
          }
          return {
            id: toolCall.id,
            result: weather,
          }

        case 'calculate':
          // Simple expression evaluation (in a real app, use a safer evaluator)
          try {
            // Convert ^ to ** for exponentiation and sanitize input
            const sanitizedExpression = args.expression
              .replace(/\^/g, '**') // Convert ^ to ** for exponentiation
              .replace(/[^\d\s()*+./\-]/g, '') // Remove unsafe characters

            const result = eval(sanitizedExpression)
            return {
              id: toolCall.id,
              result: { expression: args.expression, result },
            }
          } catch {
            return {
              id: toolCall.id,
              result: null,
              error: 'Invalid mathematical expression',
            }
          }

        case 'get_time':
          const now = new Date()
          const timeString = now.toLocaleString('en-US', {
            timeZone: args.timezone || 'UTC',
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          })
          return {
            id: toolCall.id,
            result: { timezone: args.timezone, time: timeString },
          }

        default:
          return {
            id: toolCall.id,
            result: null,
            error: `Unknown tool: ${name}`,
          }
      }
    } catch (error: any) {
      return {
        id: toolCall.id,
        result: null,
        error: error.message,
      }
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
            'You are a helpful AI assistant with access to tools. You can call tools to help answer user questions.',
        },
        // Add previous messages from chat history
        ...messages
          .filter(
            (msg): msg is MessageType.Text =>
              msg.type === 'text' &&
              !msg.metadata?.system &&
              !msg.metadata?.toolCalls &&
              !msg.metadata?.toolResults,
          )
          .reverse() // Reverse to get chronological order
          .slice(-8) // Keep fewer messages when tools are involved
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

      const completionResult = await context.completion(
        {
          messages: conversationMessages,
          tools: AVAILABLE_TOOLS,
          tool_choice: 'auto',
          jinja: true,
          n_predict: 512,
          temperature: 0.7,
          top_p: 0.9,
          stop: ['<|im_end|>'],
        },
        (data) => {
          const { token } = data

          if (token) {
            response += token

            setMessages((currentMsgs) => {
              const index = currentMsgs.findIndex(
                (msg) => msg.id === responseId,
              )
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
          }
        },
      )

      const toolCalls = completionResult.tool_calls || []

      // Handle tool calls if any were made
      if (toolCalls && toolCalls.length > 0) {
        // Ensure all tool calls have IDs
        toolCalls.forEach((toolCall) => {
          if (!toolCall.id) toolCall.id = randId()
        })

        // Show tool execution message
        const toolMessage: MessageType.Text = {
          author: assistant,
          createdAt: Date.now() + 1,
          id: randId(),
          text: `🛠️ Executing ${toolCalls.length} tool(s):\n${toolCalls
            .map((t) => `• ${t.function.name}(${t.function.arguments})`)
            .join('\n')}`,
          type: 'text',
          metadata: { system: true, toolCalls: true },
        }
        addMessage(toolMessage)

        // Execute tool calls
        const toolResults = await Promise.all(
          toolCalls.map(async (toolCall) => {
            const result = await executeTool({
              id: toolCall.id!,
              name: toolCall.function.name,
              arguments: JSON.parse(toolCall.function.arguments),
            })
            return {
              tool_call_id: toolCall.id!,
              role: 'tool' as const,
              content: result.error
                ? result.error
                : JSON.stringify(result.result),
            }
          }),
        )

        // Show tool results
        const resultsMessage: MessageType.Text = {
          author: assistant,
          createdAt: Date.now() + 2,
          id: randId(),
          text: `📊 Tool Results:\n${toolResults
            .map((r) => `• ${r.tool_call_id}: ✅ ${r.content}`)
            .join('\n')}`,
          type: 'text',
          metadata: { system: true, toolResults: true },
        }
        addMessage(resultsMessage)

        // Generate final response with tool results
        const followUpMessages = [
          ...conversationMessages,
          {
            role: 'assistant' as const,
            content: completionResult.content || response,
            tool_calls: toolCalls,
          },
          ...toolResults,
        ]

        response = ''
        const finalResponseId = randId()
        const finalResponseMessage: MessageType.Text = {
          author: assistant,
          createdAt: Date.now() + 3,
          id: finalResponseId,
          text: '',
          type: 'text',
        }

        addMessage(finalResponseMessage)

        await context.completion(
          {
            messages: followUpMessages,
            tools: AVAILABLE_TOOLS,
            tool_choice: 'auto',
            jinja: true,
            n_predict: 512,
            temperature: 0.7,
            top_p: 0.9,
            stop: ['<|im_end|>'],
          },
          (data) => {
            const { token } = data
            if (token) {
              response += token

              setMessages((currentMsgs) => {
                const index = currentMsgs.findIndex(
                  (msg) => msg.id === finalResponseId,
                )
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
            }
          },
        )
      }
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
            Download a compatible model to demonstrate tool calling
            capabilities. The AI can execute functions like weather queries,
            calculations, and time lookups.
          </Text>

          <ModelDownloadCard
            title={MODELS.SMOL_LM.name}
            description="Compatible model for tool calling demonstrations"
            repo={MODELS.SMOL_LM.repo}
            filename={MODELS.SMOL_LM.filename}
            size={MODELS.SMOL_LM.size}
            onInitialize={initializeModel}
          />

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>
                {`Initializing tool-calling model... ${initProgress}%`}
              </Text>
              {initProgress > 0 && (
                <View style={styles.progressContainer}>
                  <View style={styles.progressBar}>
                    <View
                      style={[
                        styles.progressFill,
                        { width: `${initProgress}%` },
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
            ? 'Executing tools...'
            : 'Ask me to use tools...',
        }}
      />
    </SafeAreaView>
  )
}
