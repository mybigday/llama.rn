import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  Image,
  TouchableOpacity,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Chat, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { pick } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import { initLlama, LlamaContext } from '../../../src'
import { VLMModelDownloadCard } from '../components/ModelDownloadCard'
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
  pendingImageContainer: {
    position: 'absolute',
    bottom: 60,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 12,
    padding: 12,
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 1000,
  },
  pendingImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginRight: 12,
  },
  pendingImageText: {
    flex: 1,
    color: 'white',
    fontSize: 14,
  },
  removePendingButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  removePendingText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
})

export default function MultimodalScreen() {
  const [messages, setMessages] = useState<MessageType.Any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [pendingImage, setPendingImage] = useState<string | null>(null) // base64 data URL

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
  }

  const convertImageToBase64 = async (imagePath: string): Promise<string> => {
    try {
      // Remove file:// prefix if present
      const cleanPath = imagePath.replace(/^file:\/\//, '')

      // Read file as base64
      const base64 = await ReactNativeBlobUtil.fs.readFile(cleanPath, 'base64')

      // Return with data URL prefix (assuming JPEG, but could be enhanced to detect type)
      return `data:image/jpeg;base64,${base64}`
    } catch (error: any) {
      console.error('Error converting image to base64:', error)
      throw new Error(`Failed to convert image to base64: ${error.message}`)
    }
  }

  const initializeModel = async (modelPath: string, mmprojPath: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const llamaContext = await initLlama({
        model: modelPath,
        n_ctx: 4096,
        n_gpu_layers: 99,
        use_mlock: true,
        use_mmap: true,
      }, (progress) => {
        // Progress is reported as 1 to 100
        setInitProgress(Math.round(progress * 0.8)) // 80% for model loading
      })

      // Initialize multimodal support
      setInitProgress(85)
      const multimodalInitialized = await llamaContext.initMultimodal({
        path: mmprojPath,
        use_gpu: true,
      })

      if (!multimodalInitialized) {
        throw new Error('Failed to initialize multimodal support')
      }

      setContext(llamaContext)
      setIsModelReady(true)
      setInitProgress(100)

      addSystemMessage(
        'SmolVLM model loaded! You can now chat about images. Send an image to start the conversation.',
      )
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleMultimodalMessage = async (imageBase64: string, question?: string) => {
    if (!context || isLoading) return

    try {
      setIsLoading(true)

      // Add user image message (using base64 as URI for display)
      const imageMessage: MessageType.Image = {
        author: user,
        createdAt: Date.now(),
        id: randId(),
        name: 'image.jpg',
        size: 0,
        type: 'image',
        uri: imageBase64,
      }
      addMessage(imageMessage)

      // Add user text if provided
      if (question) {
        const userMessage: MessageType.Text = {
          author: user,
          createdAt: Date.now(),
          id: randId(),
          text: question,
          type: 'text',
        }
        addMessage(userMessage)
      }

      const text = question || 'Describe this image in detail.'

      let response = ''

      await context.completion(
        {
          messages: [
            {
              role: 'system',
              content: [
                {
                  type: 'text',
                  text: 'You are a helpful AI assistant with vision capabilities. You can see and analyze images that users share. Be descriptive when analyzing images and helpful in answering questions about visual content.',
                },
              ],
            },
            {
              role: 'user',
              content: [
                {
                  type: 'text',
                  text,
                },
                {
                  type: 'image_url',
                  image_url: {
                    url: imageBase64,
                  },
                },
              ],
            },
          ],
          n_predict: 512,
          temperature: 0.7,
          top_p: 0.9,
          stop: ['<|im_end|>'],
        },
        (data) => {
          const { token } = data
          response += token
        },
      )

      if (response.trim()) {
        const botMessage: MessageType.Text = {
          author: assistant,
          createdAt: Date.now(),
          id: randId(),
          text: response.trim(),
          type: 'text',
        }
        addMessage(botMessage)
      }
    } catch (error: any) {
      console.error('Vision error:', error)
      addSystemMessage(`Error: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendPress = async (message: MessageType.PartialText) => {
    if (!context || isLoading) return

    // If there's a pending image, handle it as a multimodal message
    if (pendingImage) {
      await handleMultimodalMessage(pendingImage, message.text.trim() || undefined)
      setPendingImage(null) // Clear pending image after processing
      return
    }

    const userMessage: MessageType.Text = {
      author: user,
      createdAt: Date.now(),
      id: randId(),
      text: message.text,
      type: 'text',
    }
    addMessage(userMessage)

    try {
      setIsLoading(true)

      // Build messages array for conversation context with system prompt
      const conversationMessages: Array<{
        role: 'system' | 'user' | 'assistant'
        content: Array<{
          type: 'text' | 'image_url'
          text?: string
          image_url?: { url: string }
        }>
      }> = [
        {
          role: 'system' as const,
          content: [
            {
              type: 'text' as const,
              text: 'You are a helpful AI assistant with vision capabilities. You can see and analyze images that users share. Be descriptive when analyzing images and helpful in answering questions about visual content. Be concise and helpful in your responses.',
            },
          ],
        },
      ]

      // Add previous messages from chat history (excluding system messages)
      const recentMessages = await Promise.all(
        messages
          .filter(
            (msg): msg is MessageType.Text | MessageType.Image =>
              (msg.type === 'text' && !msg.metadata?.system) || msg.type === 'image'
          )
          .reverse() // Reverse to get chronological order
          .slice(-10) // Keep last 10 messages for context
          .map(async (msg) => {
            if (msg.type === 'text') {
              const role = msg.author.id === user.id ? 'user' as const : 'assistant' as const
              return {
                role,
                content: [
                  {
                    type: 'text' as const,
                    text: (msg as MessageType.Text).text,
                  },
                ],
              }
            } else if (msg.type === 'image' && msg.author.id === 'user') {
              // Convert image to base64 for conversation context
              const imageBase64 = await convertImageToBase64((msg as MessageType.Image).uri)
              return {
                role: 'user' as const,
                content: [
                  {
                    type: 'image_url' as const,
                    image_url: {
                      url: imageBase64,
                    },
                  },
                ],
              }
            }
            return null
          })
      )

      // Filter out null values and add to conversation
      const validRecentMessages = recentMessages.filter((msg): msg is NonNullable<typeof msg> => msg !== null)
      conversationMessages.push(...validRecentMessages)

      // Add current user message
      conversationMessages.push({
        role: 'user' as const,
        content: [
          {
            type: 'text' as const,
            text: message.text,
          },
        ],
      })

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
          stop: ['<|im_end|>', 'User:', '\n\n'],
        },
        (data) => {
          const { token } = data
          response += token

          // Update message in real-time
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

      // Update final message with completion status
      setMessages((currentMsgs) => {
        const index = currentMsgs.findIndex((msg) => msg.id === responseId)
        if (index >= 0) {
          return currentMsgs.map((msg, i) => {
            if (msg.type === 'text' && i === index) {
              return {
                ...msg,
                text: response.trim(),
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
      console.error('Chat error:', error)
      addSystemMessage(`Error: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

    const handleAttachmentPress = async () => {
    try {
      const result = await pick({
        type: 'image/*',
        copyTo: 'documentDirectory',
      })

      if (result && result.length > 0) {
        const file = result[0]
        if (file.uri) {
          // Convert to base64 immediately to prevent file access issues later
          try {
            const imageBase64 = await convertImageToBase64(file.uri)
            setPendingImage(imageBase64)
          } catch (conversionError: any) {
            Alert.alert('Error', `Failed to process image: ${conversionError.message}`)
          }
        }
      }
    } catch (error: any) {
      if (error.message !== 'User canceled the picker') {
        Alert.alert('Error', `Failed to pick image: ${error.message}`)
      }
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
            Download the SmolVLM model to start analyzing images. This model can
            understand and describe images, answer questions about visual
            content, and engage in vision-language conversations.
          </Text>

          <VLMModelDownloadCard
            title={MODELS.SMOL_VLM.name}
            description={MODELS.SMOL_VLM.description}
            repo={MODELS.SMOL_VLM.repo}
            filename={MODELS.SMOL_VLM.filename}
            mmproj={MODELS.SMOL_VLM.mmproj}
            size={MODELS.SMOL_VLM.size}
            onInitialize={initializeModel}
          />

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
        messages={messages}
        onSendPress={handleSendPress}
        onAttachmentPress={handleAttachmentPress}
        user={user}
        renderBubble={renderBubble}
        theme={defaultTheme}
        showUserAvatars
        showUserNames
        disableImageGallery={false}
        textInputProps={{
          placeholder: pendingImage
            ? 'Ask about the image above or send without text...'
            : 'Type your message...',
        }}
      />

      {/* Pending Image Preview */}
      {pendingImage && (
        <View style={styles.pendingImageContainer}>
          <Image source={{ uri: pendingImage }} style={styles.pendingImage} />
          <Text style={styles.pendingImageText}>
            Image ready to send. Type a question or tap send.
          </Text>
          <TouchableOpacity
            style={styles.removePendingButton}
            onPress={() => setPendingImage(null)}
          >
            <Text style={styles.removePendingText}>Remove</Text>
          </TouchableOpacity>
        </View>
      )}

      {isLoading && (
        <View
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.3)',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={{ color: 'white', marginTop: 10 }}>Processing...</Text>
        </View>
      )}
    </SafeAreaView>
  )
}
