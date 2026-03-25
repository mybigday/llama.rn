import React, { useState, useRef, useCallback } from 'react'
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Alert,
  Image,
  TouchableOpacity,
  Platform,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { Chat } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { pick, keepLocalCopy } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { Bubble } from '../components/Bubble'
import { Menu } from '../components/Menu'
import { MessagesModal } from '../components/MessagesModal'
import { MaskedProgress } from '../components/MaskedProgress'
import SessionModal from '../components/SessionModal'
import { StopButton } from '../components/StopButton'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import {
  createThemedStyles,
  chatDarkTheme,
  chatLightTheme,
} from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import type { LLMMessage } from '../utils/llmMessages'
import { initLlama } from '../../../src' // import 'llama.rn'
import {
  useStoredCompletionParams,
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { useExampleContext } from '../hooks/useExampleContext'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import {
  CHAT_ASSISTANT,
  CHAT_USER,
  createMessageId,
  createSystemTextMessage,
} from '../features/chatHelpers'
import {
  DEFAULT_MULTIMODAL_SYSTEM_PROMPT,
  createMultimodalSystemPrompt,
  createMultimodalWelcomeMessage,
} from '../features/multimodalHelpers'
import {
  createExampleModelDefinitions,
  type ExampleModelKey,
} from '../utils/exampleModels'

const MULTIMODAL_MODELS = createExampleModelDefinitions(
  Object.entries(MODELS)
    .filter(([_key, model]) => model.mmproj)
    .map(([key]) => key as ExampleModelKey),
)

export default function MultimodalScreen({ navigation }: { navigation: any }) {
  const { isDark, theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = createStyles(theme, themedStyles)

  const messagesRef = useRef<MessageType.Any[]>([])
  const [, setMessagesVersion] = useState(0) // For UI updates
  const [isLoading, setIsLoading] = useState(false)
  const [pendingMedia, setPendingMedia] = useState<{
    data: string // base64 data URL
    mimeType: string
    type: 'image' | 'audio'
  } | null>(null)
  const [multimodalSupport, setMultimodalSupport] = useState<{
    vision: boolean
    audio: boolean
  } | null>(null)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showMessagesModal, setShowMessagesModal] = useState(false)
  const [showSessionModal, setShowSessionModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState(
    DEFAULT_MULTIMODAL_SYSTEM_PROMPT,
  )
  const [imageMaxTokens, setImageMaxTokens] = useState<string>('')
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

  const buildLLMMessages = (
    currentUserContent?: Array<{
      type: 'text' | 'image_url' | 'input_audio'
      text?: string
      image_url?: { url: string }
      input_audio?: { format: string; data: string }
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
        msg.author.id === CHAT_USER.id
          ? ('user' as const)
          : ('assistant' as const)
      const lastGroup = acc[acc.length - 1]

      // If this is a new author or we don't have a previous group, start a new group
      if (!lastGroup || lastGroup.role !== role) {
        acc.push({ role, content: [] })
      }

      const currentGroup = acc[acc.length - 1]!

      const reasoningContent = msg.metadata?.completionResult?.reasoning_content
      if (!currentGroup.reasoning_content) {
        currentGroup.reasoning_content = reasoningContent
      }

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
            setPendingMedia(null)
            const welcomeMessage =
              createMultimodalWelcomeMessage(multimodalSupport)
            addSystemMessage(welcomeMessage)
          },
        },
      ],
    )
  }, [multimodalSupport, addSystemMessage, context])

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

    // Add the initial system message based on current capabilities
    const welcomeMessage = createMultimodalWelcomeMessage(multimodalSupport)
    addSystemMessage(welcomeMessage)

    // Add imported messages
    messagesRef.current = [...newMessages.reverse(), ...messagesRef.current]
    setMessagesVersion((prev) => prev + 1)
  }

  const handleUpdateSystemPrompt = (newSystemPrompt: string) => {
    setSystemPrompt(newSystemPrompt)
  }

  const convertMediaToBase64 = async (
    mediaPath: string,
    mimeType: string,
  ): Promise<string> => {
    try {
      // Remove file:// prefix if present
      const cleanPath = mediaPath.replace(/^file:\/\//, '')

      // Read file as base64
      const base64 = await ReactNativeBlobUtil.fs.readFile(cleanPath, 'base64')

      // Return with data URL prefix
      return `data:${mimeType};base64,${base64}`
    } catch (error: any) {
      const mediaType = mimeType.startsWith('image/') ? 'image' : 'audio'
      console.error(`Error converting ${mediaType} to base64:`, error)
      throw new Error(
        `Failed to convert ${mediaType} to base64: ${error.message}`,
      )
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
      const maxTokens = imageMaxTokens
        ? parseInt(imageMaxTokens, 10)
        : undefined
      const multimodalInitialized = await llamaContext.initMultimodal({
        path: mmprojPath,
        use_gpu: true,
        image_max_tokens:
          maxTokens && !Number.isNaN(maxTokens) ? maxTokens : undefined,
      })

      if (!multimodalInitialized) {
        throw new Error('Failed to initialize multimodal support')
      }

      // Check what multimodal capabilities are supported
      setInitProgress(95)
      const support = await llamaContext.getMultimodalSupport()
      setMultimodalSupport(support)

      // Update system prompt based on actual capabilities
      const dynamicSystemPrompt = createMultimodalSystemPrompt(support)
      setSystemPrompt(dynamicSystemPrompt)

      await replaceContext(llamaContext)
      setInitProgress(100)

      // Create dynamic welcome message based on capabilities
      const welcomeMessage = createMultimodalWelcomeMessage(support)
      addSystemMessage(welcomeMessage)
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
        type: 'text' | 'image_url' | 'input_audio'
        text?: string
        image_url?: { url: string }
        input_audio?: { format: string; data: string }
      }> = []

      // Add media content if there's pending media
      if (pendingMedia) {
        if (pendingMedia.type === 'image') {
          currentUserContent.push({
            type: 'image_url',
            image_url: {
              url: pendingMedia.data,
            },
          })
        } else if (pendingMedia.type === 'audio') {
          currentUserContent.push({
            type: 'input_audio',
            input_audio: {
              format: pendingMedia.mimeType.split('/')[1] || 'wav',
              data: pendingMedia.data,
            },
          })
        }
      }

      // Add text content if provided
      const textContent =
        message.text.trim() ||
        (pendingMedia?.type === 'image'
          ? 'Describe this image in detail.'
          : '') ||
        (pendingMedia?.type === 'audio' ? 'Describe this audio in detail.' : '')

      if (textContent) {
        currentUserContent.push({
          type: 'text',
          text: textContent,
        })
      }

      // If no content to send, return early
      if (currentUserContent.length === 0) {
        return
      }

      // Build conversation messages using the reusable function, including current user content
      const conversationMessages = buildLLMMessages(currentUserContent)

      // Now add the user messages to the chat UI
      if (pendingMedia && pendingMedia.type === 'image') {
        const imageMessage: MessageType.Image = {
          author: CHAT_USER,
          createdAt: Date.now(),
          id: createMessageId(),
          name: 'image',
          size: 0,
          type: 'image',
          uri: pendingMedia.data,
          metadata: {
            mimeType: pendingMedia.mimeType,
          },
        }
        addMessage(imageMessage)
      }

      if (message.text.trim()) {
        const userMessage: MessageType.Text = {
          author: CHAT_USER,
          createdAt: Date.now(),
          id: createMessageId(),
          text: message.text,
          type: 'text',
        }
        addMessage(userMessage)
      }

      const responseId = createMessageId()
      const responseMessage: MessageType.Text = {
        author: CHAT_ASSISTANT,
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
        },
        (data) => {
          const { content = '', reasoning_content: reasoningContent } = data

          // Update message in real-time
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

      // Update final message with completion status
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

      // Clear pending media after processing
      setPendingMedia(null)
    } catch (error: any) {
      console.error('Chat error:', error)
      addSystemMessage(`Error: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleAttachmentPress = async () => {
    if (!multimodalSupport) {
      Alert.alert('Error', 'Multimodal capabilities not yet determined')
      return
    }

    try {
      const supportedTypes: string[] = []

      if (multimodalSupport.vision) {
        const imageTypes =
          Platform.OS === 'ios'
            ? ['public.jpeg', 'public.png', 'public.gif', 'com.microsoft.bmp']
            : ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
        supportedTypes.push(...imageTypes)
      }

      if (multimodalSupport.audio) {
        const audioTypes =
          Platform.OS === 'ios'
            ? ['public.audio', 'com.microsoft.waveform-audio', 'public.mp3']
            : ['audio/wav', 'audio/mpeg', 'audio/mp3']
        supportedTypes.push(...audioTypes)
      }

      if (supportedTypes.length === 0) {
        Alert.alert('Info', 'This model does not support image or audio input')
        return
      }

      const [file] = await pick({
        type: supportedTypes,
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

      if (file?.uri) {
        // Keep a local copy of the file
        const [localCopy] = await keepLocalCopy({
          files: [
            {
              uri: file.uri,
              fileName: file.name ?? 'media_file',
            },
          ],
          destination: 'documentDirectory',
        })

        if (localCopy.status === 'success') {
          const mediaFormat = getFileExtension(file.uri)
          const mimeType = getMimeType(mediaFormat)

          try {
            // Determine if it's an image or audio file and check if supported
            if (mimeType.startsWith('image/')) {
              if (!multimodalSupport.vision) {
                Alert.alert('Error', 'This model does not support image input')
                return
              }
              const mediaBase64 = await convertMediaToBase64(
                localCopy.localUri,
                mimeType,
              )
              setPendingMedia({
                data: mediaBase64,
                mimeType,
                type: 'image',
              })
            } else if (mimeType.startsWith('audio/')) {
              if (!multimodalSupport.audio) {
                Alert.alert('Error', 'This model does not support audio input')
                return
              }
              const mediaBase64 = await convertMediaToBase64(
                localCopy.localUri,
                mimeType,
              )
              setPendingMedia({
                data: mediaBase64,
                mimeType,
                type: 'audio',
              })
            } else {
              const supportedFormats = []
              if (multimodalSupport.vision) supportedFormats.push('images')
              if (multimodalSupport.audio) supportedFormats.push('audio')
              const formatText = supportedFormats.join(' or ')
              Alert.alert(
                'Error',
                `Unsupported file format. Please select ${formatText} files.`,
              )
            }
          } catch (conversionError: any) {
            Alert.alert(
              'Error',
              `Failed to process file: ${conversionError.message}`,
            )
          }
        } else {
          Alert.alert('Error', `Failed to copy file: ${localCopy.copyError}`)
        }
      }
    } catch (error: any) {
      if (!error.message.includes('user canceled the document picker')) {
        Alert.alert('Error', `Failed to pick file: ${error.message}`)
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
      <>
        <ExampleModelSetup
          description="Download a multimodal model to start analyzing images and audio. These models can understand and describe images and audio, answer questions about visual and audio content, and engage in multimodal conversations."
          defaultModels={MULTIMODAL_MODELS}
          customModels={(customModels || []).filter(
            (model) => model.mmprojFilename,
          )}
          onInitializeCustomModel={(_model, modelPath, mmprojPath) =>
            initializeModel(modelPath, mmprojPath || '')
          }
          onInitializeModel={(_model, modelPath, mmprojPath) =>
            initializeModel(modelPath, mmprojPath)
          }
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          requireMMProj
          isLoading={isLoading}
          initProgress={initProgress}
          progressText={`Initializing model... ${initProgress}%`}
        >
          <View style={styles.settingContainer}>
            <Text style={styles.settingLabel}>Max Image Tokens (optional)</Text>
            <Text style={styles.settingDescription}>
              Limit tokens for dynamic resolution models (e.g., Qwen-VL). Lower
              values (256-512) improve speed, higher values (up to 4096)
              preserve detail. Leave empty for model default.
            </Text>
            <TextInput
              style={styles.settingInput}
              value={imageMaxTokens}
              onChangeText={setImageMaxTokens}
              placeholder="e.g., 512"
              placeholderTextColor={theme.colors.textSecondary}
              keyboardType="numeric"
            />
          </View>
        </ExampleModelSetup>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />
      </>
    )
  }

  return (
    <View style={styles.container}>
      {/* @ts-ignore */}
      <Chat
        messages={messagesRef.current}
        onSendPress={handleSendPress}
        onAttachmentPress={
          multimodalSupport &&
          (multimodalSupport.vision || multimodalSupport.audio)
            ? handleAttachmentPress
            : undefined
        }
        user={CHAT_USER}
        renderBubble={renderBubble}
        theme={isDark ? chatDarkTheme : chatLightTheme}
        showUserAvatars
        showUserNames
        disableImageGallery={false}
        textInputProps={{
          placeholder: pendingMedia
            ? `Ask about the ${pendingMedia.type} above or send without text...`
            : 'Type your message...',
          keyboardType: 'ascii-capable',
        }}
      />

      {/* Pending Media Preview */}
      {pendingMedia && (
        <View
          style={[styles.pendingMediaContainer, { bottom: insets.bottom + 80 }]}
        >
          {pendingMedia.type === 'image' ? (
            <Image
              source={{ uri: pendingMedia.data }}
              style={styles.pendingMediaPreview}
            />
          ) : (
            <View style={styles.pendingMediaIcon}>
              <Text style={{ color: 'white', fontSize: 24 }}>♪</Text>
            </View>
          )}
          <Text style={styles.pendingMediaText}>
            {pendingMedia.type === 'image'
              ? 'Image ready'
              : `Audio file ready (${pendingMedia.mimeType
                  .split('/')[1]
                  ?.toUpperCase()})`}
          </Text>
          <TouchableOpacity
            style={styles.removePendingButton}
            onPress={() => setPendingMedia(null)}
          >
            <Text style={styles.removePendingText}>Remove</Text>
          </TouchableOpacity>
        </View>
      )}

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
        defaultSystemPrompt={createMultimodalSystemPrompt(multimodalSupport)}
      />

      <SessionModal
        visible={showSessionModal}
        onClose={() => setShowSessionModal(false)}
        context={context}
      />

      <MaskedProgress
        visible={isLoading}
        text="Processing..."
        progress={0}
        showProgressBar={false}
      />
    </View>
  )
}

function createStyles(
  theme: ReturnType<typeof useTheme>['theme'],
  themedStyles: ReturnType<typeof createThemedStyles>,
) {
  return StyleSheet.create({
    container: themedStyles.container,
    header: {
      ...themedStyles.header,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    headerTitle: themedStyles.headerTitle,
    setupContainer: themedStyles.setupContainer,
    scrollContent: themedStyles.scrollContent,
    setupDescription: themedStyles.setupDescription,
    loadingContainer: themedStyles.loadingContainer,
    loadingText: themedStyles.loadingText,
    progressContainer: themedStyles.progressContainer,
    progressBar: themedStyles.progressBar,
    progressFill: themedStyles.progressFill,
    pendingMediaContainer: {
      position: 'absolute',
      bottom: 80,
      left: 16,
      right: 16,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      padding: 12,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    pendingMediaText: {
      color: theme.colors.text,
      fontSize: 14,
      marginBottom: 8,
    },
    pendingMediaImage: {
      width: 60,
      height: 60,
      borderRadius: 4,
      marginRight: 8,
    },
    pendingMediaRow: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    pendingMediaInfo: {
      flex: 1,
    },
    cancelButton: {
      backgroundColor: theme.colors.error,
      paddingHorizontal: 12,
      paddingVertical: 6,
      borderRadius: 4,
    },
    cancelButtonText: {
      color: theme.colors.white,
      fontSize: 12,
      fontWeight: '600',
    },
    pendingMediaPreview: {
      width: 60,
      height: 60,
      borderRadius: 8,
      marginBottom: 4,
    },
    pendingMediaIcon: {
      width: 60,
      height: 60,
      borderRadius: 8,
      backgroundColor: theme.colors.surface,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: 4,
    },
    removePendingButton: {
      backgroundColor: theme.colors.error,
      paddingHorizontal: 16,
      paddingVertical: 8,
      borderRadius: 6,
    },
    removePendingText: {
      color: theme.colors.white,
      fontSize: 12,
      fontWeight: '500',
    },
    settingContainer: {
      marginTop: 16,
      marginBottom: 8,
      padding: 12,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    settingLabel: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 4,
    },
    settingDescription: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginBottom: 8,
    },
    settingInput: {
      backgroundColor: theme.colors.inputBackground,
      borderRadius: 6,
      paddingHorizontal: 12,
      paddingVertical: 8,
      fontSize: 14,
      color: theme.colors.text,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
  })
}
