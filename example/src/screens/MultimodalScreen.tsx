import React, { useState, useEffect, useLayoutEffect, useRef } from 'react'
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
import { MessagesModal } from '../components/MessagesModal'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import type { LLMMessage } from '../utils/llmMessages'

const user = { id: 'user' }
const assistant = { id: 'assistant' }

const randId = () => Math.random().toString(36).substr(2, 9)

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful AI assistant with vision capabilities. You can see and analyze images that users share. Be descriptive when analyzing images and helpful in answering questions about visual content. Be concise and helpful in your responses.'

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
  scrollContent: CommonStyles.scrollContent,
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
  const messagesRef = useRef<MessageType.Any[]>([])
  const [, setMessagesVersion] = useState(0) // For UI updates
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

  const buildLLMMessages = (
    currentUserContent?: Array<{
      type: 'text' | 'image_url'
      text?: string
      image_url?: { url: string }
    }>,
  ): LLMMessage[] => {
    const conversationMessages: LLMMessage[] = [
      {
        role: 'system',
        content: [
          {
            type: 'text',
            text: systemPrompt,
          },
        ],
      },
    ]

    // Add previous messages from chat history (excluding system messages)
    const recentMessages = messagesRef.current
      .filter(
        (msg): msg is MessageType.Text | MessageType.Image =>
          (msg.type === 'text' && !msg.metadata?.system) ||
          msg.type === 'image',
      )
      .reverse() // Reverse to get chronological order
      .slice(-20) // Keep more messages to account for images

    // Group consecutive messages by author and merge content
    const groupedMessages = recentMessages.reduce<LLMMessage[]>((acc, msg) => {
      const role =
        msg.author.id === user.id ? ('user' as const) : ('assistant' as const)
      const lastGroup = acc[acc.length - 1]

      // If this is a new author or we don't have a previous group, start a new group
      if (!lastGroup || lastGroup.role !== role) {
        acc.push({ role, content: [] })
      }

      const currentGroup = acc[acc.length - 1]!

      // Add message content to the current group
      if (msg.type === 'text') {
        if (Array.isArray(currentGroup.content)) {
          currentGroup.content.push({
            type: 'text',
            text: msg.text,
          })
        } else {
          currentGroup.content = [
            {
              type: 'text',
              text:
                typeof currentGroup.content === 'string'
                  ? currentGroup.content
                  : '',
            },
            {
              type: 'text',
              text: msg.text,
            },
          ]
        }
      } else if (msg.type === 'image') {
        if (Array.isArray(currentGroup.content)) {
          currentGroup.content.push({
            type: 'image_url',
            image_url: {
              url: msg.uri,
            },
          })
        } else {
          currentGroup.content = [
            {
              type: 'text',
              text:
                typeof currentGroup.content === 'string'
                  ? currentGroup.content
                  : '',
            },
            {
              type: 'image_url',
              image_url: {
                url: msg.uri,
              },
            },
          ]
        }
      }

      return acc
    }, [])

    const finalMessages = [
      ...conversationMessages,
      ...groupedMessages.slice(-10),
    ]

    // Add current user content if provided
    if (currentUserContent && currentUserContent.length > 0) {
      finalMessages.push({
        role: 'user',
        content: currentUserContent,
      })
    }

    return finalMessages
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
  }

  const handleImportMessages = (newMessages: MessageType.Any[]) => {
    // Reset messages and add system message back
    messagesRef.current = []
    setMessagesVersion((prev) => prev + 1)

    // Add the initial system message
    addSystemMessage(
      "Hello! I'm a vision-capable AI assistant. You can share images with me and I'll analyze them, answer questions about what I see, or engage in conversations about visual content. How can I help you today?",
    )

    // Add imported messages
    messagesRef.current = [...newMessages.reverse(), ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }

  const handleUpdateSystemPrompt = (newSystemPrompt: string) => {
    setSystemPrompt(newSystemPrompt)
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
          ...params,
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

  const handleSendPress = async (message: MessageType.PartialText) => {
    if (!context || isLoading) return

    try {
      setIsLoading(true)

      // Prepare current user message content for AI context (don't add to chat yet)
      const currentUserContent: Array<{
        type: 'text' | 'image_url'
        text?: string
        image_url?: { url: string }
      }> = []

      // Add text content if provided
      const textContent =
        message.text.trim() ||
        (pendingImage ? 'Describe this image in detail.' : '')
      if (textContent) {
        currentUserContent.push({
          type: 'text',
          text: textContent,
        })
      }

      // Add image content if there's a pending image
      if (pendingImage) {
        currentUserContent.push({
          type: 'image_url',
          image_url: {
            url: pendingImage,
          },
        })
      }

      // If no content to send, return early
      if (currentUserContent.length === 0) {
        return
      }

      // Build conversation messages using the reusable function, including current user content
      const conversationMessages = buildLLMMessages(currentUserContent)

      // Now add the user messages to the chat UI
      if (pendingImage) {
        const imageMessage: MessageType.Image = {
          author: user,
          createdAt: Date.now(),
          id: randId(),
          name: 'image',
          size: 0,
          type: 'image',
          uri: pendingImage,
          metadata: {
            mimeType: pendingImageMimeType || 'image/jpeg',
          },
        }
        addMessage(imageMessage)
      }

      if (message.text.trim()) {
        const userMessage: MessageType.Text = {
          author: user,
          createdAt: Date.now(),
          id: randId(),
          text: message.text,
          type: 'text',
        }
        addMessage(userMessage)
      }

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
      const completionResult = await context.completion(
        {
          messages: conversationMessages,
          ...completionParameters,
        },
        (data) => {
          const { token } = data
          response += token

          // Update message in real-time
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

      // Update final message with completion status
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

      // Clear pending image after processing
      setPendingImage(null)
      setPendingImageMimeType(null)
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
        <ScrollView
          style={styles.setupContainer}
          contentContainerStyle={styles.scrollContent}
        >
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
        messages={messagesRef.current}
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
