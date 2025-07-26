import React, { useState, useEffect, useLayoutEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Alert,
  ActivityIndicator,
  Image,
  TouchableOpacity,
  Platform,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { Chat, defaultTheme } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { pick } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import { initLlama, LlamaContext } from '../../../src'
import { VLMModelDownloadCard } from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { Bubble } from '../components/Bubble'
import { HeaderButton } from '../components/HeaderButton'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 9)

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
  setupContainer: CommonStyles.setupContainer,
  setupDescription: CommonStyles.setupDescription,
  loadingContainer: CommonStyles.loadingContainer,
  loadingText: CommonStyles.loadingText,
  progressContainer: CommonStyles.progressContainer,
  progressBar: CommonStyles.progressBar,
  progressFill: CommonStyles.progressFill,
  pendingImageContainer: {
    position: 'absolute',
    bottom: 80,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.8)',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  pendingImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginBottom: 4,
  },
  pendingImageText: {
    color: 'white',
    fontSize: 14,
    textAlign: 'center',
    marginLeft: 16,
  },
  removePendingButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  removePendingText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  settingsContainer: {
    alignItems: 'center',
    marginVertical: 16,
  },
  settingsButtonStyle: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  settingsButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  settingsButton: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '500',
  },
})

export default function MultimodalScreen({ navigation }: { navigation: any }) {
  const [messages, setMessages] = useState<MessageType.Any[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [pendingImage, setPendingImage] = useState<string | null>(null) // base64 data URL
  const [pendingImageMimeType, setPendingImageMimeType] = useState<
    string | null
  >(null)
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
  }

  const convertImageToBase64 = async (
    imagePath: string,
    mimeType: string,
  ): Promise<string> => {
    try {
      // Remove file:// prefix if present
      const cleanPath = imagePath.replace(/^file:\/\//, '')

      // Read file as base64
      const base64 = await ReactNativeBlobUtil.fs.readFile(cleanPath, 'base64')

      // Return with data URL prefix (assuming JPEG, but could be enhanced to detect type)
      return `data:${mimeType};base64,${base64}`
    } catch (error: any) {
      console.error('Error converting image to base64:', error)
      throw new Error(`Failed to convert image to base64: ${error.message}`)
    }
  }

  const initializeModel = async (modelPath: string, mmprojPath: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const params = contextParams || (await loadContextParams())
      const llamaContext = await initLlama(
        {
          model: modelPath,
          n_ctx: params.n_ctx,
          n_gpu_layers: params.n_gpu_layers,
          use_mlock: params.use_mlock,
          use_mmap: params.use_mmap,
          n_batch: params.n_batch,
          n_ubatch: params.n_ubatch,
          n_threads: params.n_threads,
          ctx_shift: params.ctx_shift,
        },
        (progress) => {
          // Progress is reported as 1 to 100
          setInitProgress(Math.round(progress * 0.8)) // 80% for model loading
        },
      )

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
        "Hello! I'm a vision-capable AI assistant. You can share images with me and I'll analyze them, answer questions about what I see, or engage in conversations about visual content. How can I help you today?",
      )
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleMultimodalMessage = async (
    imageBase64: string,
    mimeType: string,
    question?: string,
  ) => {
    if (!context || isLoading) return

    try {
      setIsLoading(true)

      // Add user image message (using base64 as URI for display)
      const imageMessage: MessageType.Image = {
        author: user,
        createdAt: Date.now(),
        id: randId(),
        name: 'image',
        size: 0,
        type: 'image',
        uri: imageBase64,
        metadata: {
          mimeType,
        },
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

      const completionParameters =
        completionParams || (await loadCompletionParams())
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
          ...completionParameters,
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
      await handleMultimodalMessage(
        pendingImage,
        pendingImageMimeType || 'image/jpeg',
        message.text.trim() || undefined,
      )
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
              (msg.type === 'text' && !msg.metadata?.system) ||
              msg.type === 'image',
          )
          .reverse() // Reverse to get chronological order
          .slice(-10) // Keep last 10 messages for context
          .map(async (msg) => {
            if (msg.type === 'text') {
              const role =
                msg.author.id === user.id
                  ? ('user' as const)
                  : ('assistant' as const)
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
              const imageBase64 = await convertImageToBase64(
                (msg as MessageType.Image).uri,
                (msg as MessageType.Image).metadata?.mimeType || 'image/jpeg',
              )
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
          }),
      )

      // Filter out null values and add to conversation
      const validRecentMessages = recentMessages.filter(
        (msg): msg is NonNullable<typeof msg> => msg !== null,
      )
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

      const completionParameters =
        completionParams || (await loadCompletionParams())
      await context.completion(
        {
          messages: conversationMessages,
          ...completionParameters,
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
      const imageTypes =
        Platform.OS === 'ios'
          ? ['public.jpeg', 'public.png', 'public.gif', 'com.microsoft.bmp']
          : ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
      const result = await pick({
        type: imageTypes,
        copyTo: 'documentDirectory',
      })

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
        return mimeTypes[extension] || 'image/jpeg' // Default to jpeg if unknown
      }

      if (result && result.length > 0) {
        const file = result[0]
        if (file.uri) {
          // Get the extension and MIME type
          // Although, the stb_image library doesn't rely on MIME types but instead detects formats by examining
          // the file's binary header/signature (magic numbers)
          // From stb_image.h:
          /*
              // test the formats with a very explicit header first (at least a FOURCC
              // or distinctive magic number first)
          */
          // Still, let's get this right.
          const mediaFormat = getFileExtension(file.uri)
          const mimeType = getMimeType(mediaFormat)
          // Convert to base64 immediately to prevent file access issues later
          try {
            const imageBase64 = await convertImageToBase64(file.uri, mimeType)
            setPendingImage(imageBase64)
            setPendingImageMimeType(mimeType)
          } catch (conversionError: any) {
            Alert.alert(
              'Error',
              `Failed to process image: ${conversionError.message}`,
            )
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

          {['SMOL_VLM', 'GEMMA_3'].map((model) => {
            const modelInfo = MODELS[model as keyof typeof MODELS]
            return (
              <VLMModelDownloadCard
                key={model}
                title={modelInfo.name}
                repo={modelInfo.repo}
                filename={modelInfo.filename}
                mmproj={modelInfo.mmproj || ''}
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
                        { width: `${initProgress}%` },
                      ]}
                    />
                  </View>
                </View>
              )}
            </View>
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

      <CompletionParamsModal
        visible={showCompletionParamsModal}
        onClose={() => setShowCompletionParamsModal(false)}
        onSave={handleSaveCompletionParams}
      />
    </SafeAreaView>
  )
}
