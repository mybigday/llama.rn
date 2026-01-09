import React, {
  useState,
  useEffect,
  useRef,
  useLayoutEffect,
  useCallback,
} from 'react'
import {
  View,
  Text,
  ScrollView,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import ReactNativeBlobUtil from 'react-native-blob-util'
import { useTheme } from '../contexts/ThemeContext'
import { createThemedStyles } from '../styles/commonStyles'
import ModelDownloadCard, {
  MtmdModelDownloadCard,
} from '../components/ModelDownloadCard'
import { HeaderButton } from '../components/HeaderButton'
import { MaskedProgress } from '../components/MaskedProgress'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
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
import { initLlama, LlamaContext } from '../../../src'
import type { ParallelStatus } from '../../../src'

interface ConversationSlot {
  id: string
  prompt: string
  response: string
  status: 'idle' | 'processing' | 'completed' | 'error'
  startTime?: number
  endTime?: number
  requestId: number
  stop?: () => Promise<void>
  timings?: {
    cache_n: number
    prompt_n: number
    prompt_ms: number
    prompt_per_token_ms: number
    prompt_per_second: number
    predicted_n: number
    predicted_ms: number
    predicted_per_token_ms: number
    predicted_per_second: number
  }
}

const LLM_MODELS = Object.entries(MODELS).filter(([_key, model]) => {
  const modelWithExtras = model as typeof model & {
    vocoder?: any
    embedding?: any
    ranking?: any
  }
  return (
    !modelWithExtras.vocoder &&
    !modelWithExtras.embedding &&
    !modelWithExtras.ranking
  )
})

const SYSTEM_PROMPT =
  'You are a helpful AI assistant. Be concise and direct in your responses.'

// Helper to generate a simple hash for a question (for state file naming)
const hashString = (str: string): string => {
  let hash = 0
  for (let i = 0; i < str.length; i += 1) {
    const char = str.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash &= hash // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36)
}

// Helper to get state file path for a question (per model)
const getStatePath = (modelPath: string, prompt: string): string => {
  // Extract model filename without extension
  const modelFilename =
    modelPath
      .split('/')
      .pop()
      ?.replace(/\.[^./]+$/, '') || 'unknown'
  const questionHash = hashString(prompt.trim().toLowerCase())
  const cacheDir = ReactNativeBlobUtil.fs.dirs.CacheDir
  return `${cacheDir}/state_${modelFilename}_${questionHash}.bin`
}

const EXAMPLE_PROMPTS = [
  'What is the capital of France?',
  'Explain quantum computing in simple terms.',
  'Write a haiku about coding.',
  'What are the primary colors?',
  'What is the meaning of life?',
  'What is art?',
]

// Example image URLs from Lorem Picsum (free placeholder image service)
const EXAMPLE_IMAGE_URLS = [
  'https://picsum.photos/id/1025/300/300', // Scenic landscape
  'https://picsum.photos/id/237/300/300', // Dog image
  'https://picsum.photos/id/1/200/300', // People use laptop
]

// Multimodal example prompts with image URLs to download at runtime
const MULTIMODAL_EXAMPLE_PROMPTS = [
  {
    text: 'What animal do you see in this image? Describe it.',
    imageUrl: EXAMPLE_IMAGE_URLS[0],
  },
  {
    text: 'Describe what you see in this image in detail.',
    imageUrl: EXAMPLE_IMAGE_URLS[1],
  },
  {
    text: 'Describe what you see in this image in detail.', // Should use same state file with 2nd prompt
    imageUrl: EXAMPLE_IMAGE_URLS[2],
  },
]

export default function ParallelDecodingScreen({
  navigation,
}: {
  navigation: any
}) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const insets = useSafeAreaInsets()

  const [context, setContext] = useState<LlamaContext | null>(null)
  const modelPathRef = useRef<string>('')
  const [isModelReady, setIsModelReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [slots, setSlots] = useState<ConversationSlot[]>([])
  const [parallelSlots, setParallelSlots] = useState(2)
  const [customPrompt, setCustomPrompt] = useState('')
  const [isParallelMode, setIsParallelMode] = useState(false)
  const [isMultimodalEnabled, setIsMultimodalEnabled] = useState(false)
  const [parallelStatus, setParallelStatus] = useState<ParallelStatus | null>(
    null,
  )
  const statusSubscriptionRef = useRef<{ remove: () => void } | null>(null)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])
  const slotsRef = useRef<ConversationSlot[]>([])
  const imageCache = useRef<Map<string, string>>(new Map())

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

  const clearSlots = useCallback(async () => {
    // Cancel all active/queued requests first
    const processingSlots = slotsRef.current.filter(
      (t) => t.status === 'processing' || t.status === 'idle',
    )

    await Promise.all(
      processingSlots.map(async (slot) => {
        try {
          if (slot.stop) await slot.stop()
        } catch (err) {
          console.error(`Error cancelling request ${slot.requestId}:`, err)
        }
      }),
    )

    // Then clear all slots
    setSlots([])
    slotsRef.current = []
  }, [])

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

  // Set up header buttons
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton iconName="refresh" onPress={clearSlots} />
            <HeaderButton
              iconName="cog-outline"
              onPress={() => setShowCompletionParamsModal(true)}
            />
            <HeaderButton
              iconName="information-outline"
              onPress={() => {
                Alert.alert(
                  'Parallel Decoding',
                  [
                    `This demo showcases parallel request processing using ${parallelSlots} slots.`,
                    '',
                    'Multiple requests are processed concurrently, improving throughput and efficiency.',
                    '',
                    isMultimodalEnabled
                      ? 'Multimodal mode is enabled! Try sending multimodal prompts with images.'
                      : 'Load a multimodal model (SmolVLM, InternVL3, etc.) to enable image understanding.',
                    '',
                    'Try sending multiple prompts and watch them process in parallel!',
                  ].join('\n'),
                )
              }}
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
  }, [navigation, isModelReady, parallelSlots, isMultimodalEnabled, clearSlots])

  // Cleanup on unmount
  useEffect(
    () => () => {
      if (statusSubscriptionRef.current) {
        statusSubscriptionRef.current.remove()
        statusSubscriptionRef.current = null
      }
      if (context) {
        context.release()
      }
    },
    [context],
  )

  const initializeModel = async (modelPath: string, mmprojPath?: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const params = contextParams || (await loadContextParams())
      const llamaContext = await initLlama(
        {
          model: modelPath,
          n_parallel: 8,
          ...params,
        },
        (progress) => {
          // Progress is reported as 1 to 100
          setInitProgress(progress)
        },
      )

      // Initialize multimodal if mmproj path is provided
      if (mmprojPath) {
        console.log('Initializing multimodal support...')
        const multimodalSuccess = await llamaContext.initMultimodal({
          path: mmprojPath,
          use_gpu: true,
        })
        if (!multimodalSuccess) {
          console.warn('Failed to initialize multimodal support')
        } else {
          console.log('Multimodal initialized successfully')
        }
      }

      // Enable parallel mode with configured slot count
      const success = await llamaContext.parallel.enable({
        n_parallel: parallelSlots,
        n_batch: 512,
      })

      if (!success) {
        throw new Error('Failed to enable parallel mode')
      }

      setContext(llamaContext)
      modelPathRef.current = modelPath
      setIsModelReady(true)
      setIsParallelMode(true)
      setInitProgress(100)

      // Check if multimodal is enabled
      const multimodalEnabled = await llamaContext.isMultimodalEnabled()
      setIsMultimodalEnabled(multimodalEnabled)
      console.log(`Parallel mode enabled with ${parallelSlots} slots`)
      console.log(
        `Multimodal support: ${multimodalEnabled ? 'enabled' : 'disabled'}`,
      )

      // Subscribe to parallel status changes
      try {
        const subscription = await llamaContext.parallel.subscribeToStatus(
          (status) => {
            setParallelStatus(status)
          },
        )
        statusSubscriptionRef.current = subscription
        // Get initial status
        const initialStatus = await llamaContext.parallel.getStatus()
        setParallelStatus(initialStatus)
      } catch (err) {
        console.warn('Failed to subscribe to status:', err)
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const addSlot = (prompt: string) => {
    const newSlot: ConversationSlot = {
      id: Math.random().toString(36).substring(2, 11),
      prompt,
      response: '',
      status: 'idle',
      requestId: -1, // Will be set when request is queued
    }

    setSlots((prev) => [...prev, newSlot])
    slotsRef.current = [...slotsRef.current, newSlot]
    return newSlot
  }

  const updateSlot = (id: string, updates: Partial<ConversationSlot>) => {
    setSlots((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    )
    slotsRef.current = slotsRef.current.map((t) =>
      t.id === id ? { ...t, ...updates } : t,
    )
  }

  const startConversation = async (prompt: string, images?: string[]) => {
    if (!context || !isParallelMode) {
      Alert.alert('Error', 'Model not ready or parallel mode not enabled')
      return
    }

    const slot = addSlot(prompt)
    const slotId = slot.id
    updateSlot(slotId, { status: 'processing', startTime: Date.now() })

    try {
      // Load completion params
      const params = completionParams || (await loadCompletionParams())

      // Build user message content with images if provided
      const userContent: Array<{
        type: 'text' | 'image_url'
        text?: string
        image_url?: { url: string }
      }> = []

      // Add images first
      if (images && images.length > 0) {
        images.forEach((imageUrl) => {
          userContent.push({
            type: 'image_url',
            image_url: { url: imageUrl },
          })
        })
      }

      // Add text prompt
      userContent.push({
        type: 'text',
        text: prompt,
      })

      const messages = [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userContent },
      ]

      // Get state path for this question (model-specific)
      const statePath = getStatePath(modelPathRef.current, prompt)

      // Check if state file exists on filesystem
      const stateFileExists = await ReactNativeBlobUtil.fs.exists(statePath)
      const loadStatePath = stateFileExists ? statePath : undefined

      // Format chat to get the formatted prompt for tokenization
      const formattedChat = await context.getFormattedChat(messages, undefined)

      // Tokenize the formatted prompt to get question token count
      let questionTokenCount = 0
      try {
        const tokenizeResult = await context.tokenize(formattedChat.prompt, {
          media_paths: images && images.length > 0 ? images : undefined,
        })
        questionTokenCount = tokenizeResult.tokens.length
        console.log(
          `Question token count: ${questionTokenCount}, has saved state: ${!!loadStatePath}`,
        )
      } catch (error) {
        console.error('Error tokenizing prompt:', error)
        // Continue without state if tokenization fails
        questionTokenCount = 0
      }

      const usePromptState =
        !!context.model?.is_recurrent || !!context.model?.is_hybrid

      // Use parallel.completion for parallel processing with messages format
      const { requestId, promise, stop } = await context.parallel.completion(
        {
          messages,
          ...params,
          reasoning_format: 'auto',
          n_predict: params.n_predict || 50,
          // State management: load previous state if exists, always save
          load_state_path: loadStatePath,
          save_prompt_state_path:
            usePromptState && questionTokenCount > 0 ? statePath : undefined,
          save_state_path:
            !usePromptState && questionTokenCount > 0 ? statePath : undefined,
          save_state_size:
            !usePromptState && questionTokenCount > 0
              ? questionTokenCount
              : undefined,
        },
        (_reqId, data) => {
          const currentSlot = slotsRef.current.find((t) => t.id === slotId)
          if (currentSlot && data.token) {
            updateSlot(slotId, {
              response: data.accumulated_text,
            })
          }
        },
      )

      // Update slot with requestId and stop function
      updateSlot(slotId, { requestId, stop })

      // Await promise to get the final result
      promise
        .then((result) => {
          const finalText = result.text || ''
          updateSlot(slotId, {
            status: 'completed',
            response: finalText,
            endTime: Date.now(),
            timings: result.timings,
          })
          console.log('Timings:', result.timings)
        })
        .catch((err) => {
          console.error('Promise error:', err)
          updateSlot(slotId, {
            status: 'error',
            response: `${err}`,
            endTime: Date.now(),
          })
        })
    } catch (error) {
      console.error('Completion error:', error)
      updateSlot(slotId, {
        status: 'error',
        response: `Error: ${error}`,
        endTime: Date.now(),
      })
    }
  }

  // Helper function to download image and convert to base64 data URI (with caching)
  const downloadImageAsDataUri = async (url: string): Promise<string> => {
    // Check cache first
    const cached = imageCache.current.get(url)
    if (cached) {
      console.log(`Using cached image for ${url}`)
      return cached
    }

    try {
      console.log(`Downloading image from ${url}`)
      const response = await fetch(url)
      const blob = await response.blob()
      let dataUri = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onloadend = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(blob)
      })
      // NOTE: Replace octet-stream with jpeg on Android
      dataUri = dataUri.replace(
        'data:application/octet-stream;base64,',
        'data:image/jpeg;base64,',
      )

      // Store in cache
      imageCache.current.set(url, dataUri)
      return dataUri
    } catch (error) {
      console.error(`Failed to download image from ${url}:`, error)
      throw error
    }
  }

  const sendExamplePrompts = async () => {
    if (isMultimodalEnabled) {
      // Download all images in parallel, then send examples with delays
      const downloadPromises = MULTIMODAL_EXAMPLE_PROMPTS.map(
        async (example, index) => {
          // Skip if imageUrl is missing
          if (!example.imageUrl) {
            return { success: false, text: example.text, index }
          }

          try {
            // Download image from URL
            const imageDataUri = await downloadImageAsDataUri(example.imageUrl)
            return { success: true, text: example.text, imageDataUri, index }
          } catch (error) {
            console.error('Failed to download image for example:', error)
            // Return text-only if image download fails
            return { success: false, text: example.text, index }
          }
        },
      )

      const results = await Promise.all(downloadPromises)

      // Send all examples with small delays
      results.forEach((result) => {
        if (result.success && result.imageDataUri) {
          setTimeout(
            () => startConversation(result.text, [result.imageDataUri]),
            result.index * 100,
          )
        } else {
          setTimeout(() => startConversation(result.text), result.index * 100)
        }
      })
    } else {
      // Send text-only examples
      EXAMPLE_PROMPTS.forEach((prompt, index) => {
        setTimeout(() => startConversation(prompt), index * 100)
      })
    }
  }

  const sendCustomPrompt = () => {
    if (customPrompt.trim()) {
      startConversation(customPrompt)
      setCustomPrompt('')
    }
  }

  const cancelSlot = async (slot: ConversationSlot) => {
    try {
      // Use the stop function from queueCompletion
      if (slot.stop) await slot.stop()
      updateSlot(slot.id, {
        status: 'error',
        response: slot.response || 'Cancelled',
        endTime: Date.now(),
      })
    } catch (err) {
      console.error(`Error cancelling request ${slot.requestId}:`, err)
    }
  }

  const cancelAllRequests = async () => {
    if (!context) return

    try {
      // Cancel all processing requests using the stop function
      const processingSlots = slots.filter(
        (t) => t.status === 'processing' || t.status === 'idle',
      )

      await Promise.all(
        processingSlots.map(async (slot) => {
          await cancelSlot(slot)
        }),
      )
    } catch (error) {
      console.error('Error cancelling requests:', error)
    }
  }

  const clearStateFiles = async () => {
    try {
      const cacheDir = ReactNativeBlobUtil.fs.dirs.CacheDir

      // List all files in cache directory
      const files = await ReactNativeBlobUtil.fs.ls(cacheDir)

      // Filter state files (files starting with "state_" and ending with ".bin")
      const stateFiles = files.filter(
        (file) => file.startsWith('state_') && file.endsWith('.bin'),
      )

      if (stateFiles.length === 0) {
        Alert.alert('Info', 'No state files found')
        return
      }

      // Confirm before deleting
      Alert.alert(
        'Clear State Files',
        `Found ${stateFiles.length} state file(s). Delete all?`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Delete',
            style: 'destructive',
            onPress: async () => {
              try {
                // Delete all state files
                await Promise.all(
                  stateFiles.map((file) =>
                    ReactNativeBlobUtil.fs.unlink(`${cacheDir}/${file}`),
                  ),
                )
                console.log(`Deleted ${stateFiles.length} state files`)
                Alert.alert(
                  'Success',
                  `Deleted ${stateFiles.length} state file(s)`,
                )
              } catch (error) {
                console.error('Error deleting state files:', error)
                Alert.alert('Error', 'Failed to delete some state files')
              }
            },
          },
        ],
      )
    } catch (error) {
      console.error('Error listing state files:', error)
      Alert.alert('Error', 'Failed to list state files')
    }
  }

  const updateParallelSlots = async (newSlotCount: number) => {
    if (!context || !isParallelMode) {
      Alert.alert('Error', 'Model not ready or parallel mode not enabled')
      return
    }

    const currentActiveCount = slots.filter(
      (t) => t.status === 'processing' || t.status === 'idle',
    ).length
    if (currentActiveCount > 0) {
      Alert.alert('Error', 'Cannot change slot count while requests are active')
      return
    }

    try {
      const success = await context.parallel.configure({
        n_parallel: newSlotCount,
        n_batch: 512,
      })

      if (success) {
        setParallelSlots(newSlotCount)
        console.log(`Parallel slots updated to ${newSlotCount}`)
      } else {
        Alert.alert('Error', 'Failed to update parallel slot count')
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to update parallel slots: ${error.message}`)
    }
  }

  const getSlotDuration = (slot: ConversationSlot) => {
    if (slot.startTime && slot.endTime) {
      const duration = ((slot.endTime - slot.startTime) / 1000).toFixed(2)
      return `${duration}s`
    }
    return '-'
  }

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    content: {
      flex: 1,
    },
    section: {
      padding: 12,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    sectionTitle: {
      fontSize: 16,
      fontWeight: 'bold',
      color: theme.colors.text,
      marginBottom: 12,
    },
    statsRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: 6,
    },
    statBox: {
      flex: 1,
      padding: 12,
      marginHorizontal: 4,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      alignItems: 'center',
    },
    statLabel: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginBottom: 4,
    },
    statValue: {
      fontSize: 16,
      fontWeight: 'bold',
      color: theme.colors.primary,
    },
    controlsRow: {
      flexDirection: 'row',
      marginBottom: 12,
    },
    button: {
      flex: 1,
      padding: 4,
      height: 42,
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      marginHorizontal: 4,
      alignItems: 'center',
      justifyContent: 'center',
    },
    buttonDisabled: {
      backgroundColor: theme.colors.border,
    },
    buttonText: {
      color: theme.colors.white,
      fontWeight: '600',
      fontSize: 12,
      textAlign: 'center',
    },
    inputRow: {
      flexDirection: 'row',
      marginBottom: 4,
    },
    input: {
      flex: 1,
      padding: 12,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      marginRight: 8,
      color: theme.colors.text,
    },
    sendButton: {
      padding: 12,
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      justifyContent: 'center',
      alignItems: 'center',
      minWidth: 60,
    },
    slotsContainer: {
      flex: 1,
      padding: 16,
    },
    slot: {
      marginBottom: 16,
      padding: 12,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      borderLeftWidth: 4,
    },
    slotIdle: {
      borderLeftColor: theme.colors.border,
    },
    slotProcessing: {
      borderLeftColor: '#FFA500',
    },
    slotCompleted: {
      borderLeftColor: '#4CAF50',
    },
    slotError: {
      borderLeftColor: '#F44336',
    },
    slotHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: 8,
    },
    slotStatus: {
      fontSize: 12,
      fontWeight: '600',
      textTransform: 'uppercase',
    },
    slotDuration: {
      fontSize: 12,
      color: theme.colors.textSecondary,
    },
    cancelButton: {
      marginLeft: 8,
      paddingHorizontal: 8,
      paddingVertical: 4,
      backgroundColor: theme.colors.error,
      borderRadius: 4,
    },
    cancelButtonText: {
      color: theme.colors.white,
      fontSize: 11,
      fontWeight: '600',
    },
    slotPrompt: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 8,
    },
    slotResponse: {
      fontSize: 14,
      color: theme.colors.textSecondary,
      lineHeight: 20,
    },
    emptyState: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      padding: 32,
    },
    emptyStateText: {
      fontSize: 16,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      lineHeight: 24,
    },
  })

  if (!isModelReady) {
    return (
      <View style={styles.container}>
        <ScrollView contentContainerStyle={{ padding: 16 }}>
          <Text
            style={{ fontSize: 16, color: theme.colors.text, marginBottom: 16 }}
          >
            Select a model to start parallel decoding demo
          </Text>

          {/* Custom Models Section */}
          {customModels.length > 0 && (
            <>
              <Text style={themedStyles.modelSectionTitle}>Custom Models</Text>
              {customModels.map((model) => (
                <CustomModelCard
                  key={model.id}
                  model={model}
                  onInitialize={(modelPath, mmprojPath) => {
                    initializeModel(modelPath, mmprojPath)
                  }}
                  onModelRemoved={handleCustomModelRemoved}
                  initializeButtonText="Initialize"
                />
              ))}
            </>
          )}

          {/* Add Custom Model Button */}
          <TouchableOpacity
            style={themedStyles.addCustomModelButton}
            onPress={() => setShowCustomModelModal(true)}
          >
            <Text style={themedStyles.addCustomModelButtonText}>
              + Add Custom Model
            </Text>
          </TouchableOpacity>

          {/* Predefined Models Section */}
          <Text style={themedStyles.modelSectionTitle}>Default Models</Text>
          {LLM_MODELS.map(([key, model]) => {
            // Use MtmdModelDownloadCard for multimodal models
            if (model.mmproj) {
              return (
                <MtmdModelDownloadCard
                  key={key}
                  title={model.name}
                  repo={model.repo}
                  filename={model.filename}
                  mmproj={model.mmproj}
                  size={model.size}
                  onInitialize={(modelPath, mmprojPath) => {
                    initializeModel(modelPath, mmprojPath)
                  }}
                />
              )
            }

            // Use regular ModelDownloadCard for text-only models
            return (
              <ModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
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

  const activeCount = slots.filter(
    (t) => t.status === 'processing' || t.status === 'idle',
  ).length
  const completedCount = slots.filter((t) => t.status === 'completed').length
  const avgDuration =
    completedCount > 0
      ? (
          slots
            .filter((t) => t.startTime && t.endTime)
            .reduce((sum, t) => sum + (t.endTime! - t.startTime!) / 1000, 0) /
          completedCount
        ).toFixed(2)
      : '0'

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom }]}>
      {/* Stats Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Parallel Processing Stats</Text>
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Queued</Text>
            <Text style={styles.statValue}>{activeCount}</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Completed</Text>
            <Text style={styles.statValue}>{completedCount}</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Avg Time</Text>
            <Text style={styles.statValue}>{`${avgDuration}s`}</Text>
          </View>
        </View>

        {/* Configurable Slot Count */}
        <View style={{ marginTop: 12 }}>
          <View
            style={{
              flexDirection: 'row',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Text style={{ fontSize: 14, color: theme.colors.text }}>
              Number of Slots:
            </Text>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <TouchableOpacity
                style={[
                  {
                    backgroundColor: theme.colors.primary,
                    width: 32,
                    height: 32,
                    borderRadius: 16,
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginRight: 12,
                  },
                  (parallelSlots <= 1 || activeCount > 0) && { opacity: 0.5 },
                ]}
                onPress={() =>
                  parallelSlots > 1 && updateParallelSlots(parallelSlots - 1)
                }
                disabled={parallelSlots <= 1 || activeCount > 0}
              >
                <Text
                  style={{
                    color: theme.colors.white,
                    fontSize: 20,
                    fontWeight: 'bold',
                  }}
                >
                  -
                </Text>
              </TouchableOpacity>
              <Text
                style={{
                  fontSize: 18,
                  fontWeight: 'bold',
                  color: theme.colors.primary,
                  minWidth: 24,
                  textAlign: 'center',
                }}
              >
                {parallelSlots}
              </Text>
              <TouchableOpacity
                style={[
                  {
                    backgroundColor: theme.colors.primary,
                    width: 32,
                    height: 32,
                    borderRadius: 16,
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginLeft: 12,
                  },
                  (parallelSlots >= 8 || activeCount > 0) && { opacity: 0.5 },
                ]}
                onPress={() =>
                  parallelSlots < 8 && updateParallelSlots(parallelSlots + 1)
                }
                disabled={parallelSlots >= 8 || activeCount > 0}
              >
                <Text
                  style={{
                    color: theme.colors.white,
                    fontSize: 20,
                    fontWeight: 'bold',
                  }}
                >
                  +
                </Text>
              </TouchableOpacity>
            </View>
          </View>
          {activeCount > 0 && (
            <Text
              style={{
                fontSize: 11,
                color: theme.colors.textSecondary,
                marginTop: 4,
                textAlign: 'right',
              }}
            >
              Cannot change while requests are active
            </Text>
          )}
        </View>
      </View>

      {/* Slot Manager Status Section */}
      {parallelStatus && (
        <View
          style={[
            styles.section,
            { backgroundColor: theme.colors.surface, paddingVertical: 8 },
          ]}
        >
          <Text
            style={[styles.sectionTitle, { fontSize: 14, marginBottom: 8 }]}
          >
            Slot Manager Status
          </Text>
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 12 }}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <View
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#4CAF50',
                  marginRight: 6,
                }}
              />
              <Text style={{ fontSize: 12, color: theme.colors.text }}>
                {`Active: ${parallelStatus.active_slots}/${parallelStatus.n_parallel}`}
              </Text>
            </View>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <View
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: '#FFA500',
                  marginRight: 6,
                }}
              />
              <Text style={{ fontSize: 12, color: theme.colors.text }}>
                {`Queued: ${parallelStatus.queued_requests}`}
              </Text>
            </View>
          </View>
          {parallelStatus.requests.length > 0 && (
            <View style={{ marginTop: 8 }}>
              <Text
                style={{
                  fontSize: 11,
                  color: theme.colors.textSecondary,
                  marginBottom: 4,
                }}
              >
                Active Requests:
              </Text>
              {parallelStatus.requests
                .filter((r) => r.state !== 'queued')
                .slice(0, 3)
                .map((req) => (
                  <View
                    key={req.request_id}
                    style={{
                      flexDirection: 'row',
                      alignItems: 'center',
                      paddingVertical: 2,
                    }}
                  >
                    <Text
                      style={{
                        fontSize: 10,
                        color: (() => {
                          if (req.state === 'generating') return '#4CAF50'
                          if (req.state === 'processing_prompt')
                            return '#FFA500'
                          return theme.colors.textSecondary
                        })(),
                        width: 100,
                      }}
                    >
                      {`#${req.request_id} ${req.state}`}
                    </Text>
                    <Text
                      style={{
                        fontSize: 10,
                        color: theme.colors.textSecondary,
                      }}
                    >
                      {req.tokens_generated > 0
                        ? `${
                            req.tokens_generated
                          } tokens @ ${req.tokens_per_second.toFixed(1)} t/s`
                        : `prompt: ${req.prompt_length} tokens`}
                    </Text>
                  </View>
                ))}
              {parallelStatus.requests.filter((r) => r.state !== 'queued')
                .length > 3 && (
                <Text
                  style={{
                    fontSize: 10,
                    color: theme.colors.textSecondary,
                    fontStyle: 'italic',
                  }}
                >
                  {`+${
                    parallelStatus.requests.filter((r) => r.state !== 'queued')
                      .length - 3
                  } more...`}
                </Text>
              )}
            </View>
          )}
        </View>
      )}

      {/* Controls Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Controls</Text>
        <View style={styles.controlsRow}>
          <TouchableOpacity
            style={[styles.button, activeCount > 0 && styles.buttonDisabled]}
            onPress={sendExamplePrompts}
            disabled={activeCount > 0}
          >
            <Text style={styles.buttonText}>
              {isMultimodalEnabled
                ? `Send ${MULTIMODAL_EXAMPLE_PROMPTS.length} MM Examples`
                : `Send ${EXAMPLE_PROMPTS.length} Examples`}
            </Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={clearSlots}>
            <Text style={styles.buttonText}>Clear All</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, activeCount === 0 && styles.buttonDisabled]}
            onPress={cancelAllRequests}
            disabled={activeCount === 0}
          >
            <Text style={styles.buttonText}>Cancel All</Text>
          </TouchableOpacity>

          <TouchableOpacity style={styles.button} onPress={clearStateFiles}>
            <Text style={styles.buttonText}>Clear State Files</Text>
          </TouchableOpacity>
        </View>
        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            placeholder="Enter custom prompt..."
            placeholderTextColor={theme.colors.textSecondary}
            value={customPrompt}
            onChangeText={setCustomPrompt}
          />
          <TouchableOpacity
            style={[
              styles.sendButton,
              !customPrompt.trim() && styles.buttonDisabled,
            ]}
            onPress={sendCustomPrompt}
            disabled={!customPrompt.trim()}
          >
            <Text style={styles.buttonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Slots Section */}
      <ScrollView style={styles.slotsContainer}>
        {slots.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>
              {isMultimodalEnabled
                ? `No conversations yet.\n\nClick "Send ${MULTIMODAL_EXAMPLE_PROMPTS.length} MM Examples" to see parallel multimodal processing in action,\nor enter a custom prompt below.`
                : `No conversations yet.\n\nClick "Send ${EXAMPLE_PROMPTS.length} Examples" to see parallel processing in action,\nor enter a custom prompt below.`}
            </Text>
          </View>
        ) : (
          slots.map((slot) => (
            <View
              key={slot.id}
              style={[
                styles.slot,
                slot.status === 'idle' && styles.slotIdle,
                slot.status === 'processing' && styles.slotProcessing,
                slot.status === 'completed' && styles.slotCompleted,
                slot.status === 'error' && styles.slotError,
              ]}
            >
              <View style={styles.slotHeader}>
                <Text
                  style={[
                    styles.slotStatus,
                    {
                      color: (() => {
                        if (slot.status === 'processing') return '#FFA500'
                        if (slot.status === 'completed') return '#4CAF50'
                        if (slot.status === 'error') return '#F44336'
                        return theme.colors.textSecondary
                      })(),
                    },
                  ]}
                >
                  {slot.status}
                </Text>
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <Text style={styles.slotDuration}>
                    {getSlotDuration(slot)}
                  </Text>
                  {(slot.status === 'idle' || slot.status === 'processing') && (
                    <TouchableOpacity
                      style={styles.cancelButton}
                      onPress={() => cancelSlot(slot)}
                    >
                      <Text style={styles.cancelButtonText}>Cancel</Text>
                    </TouchableOpacity>
                  )}
                </View>
              </View>
              <Text style={styles.slotPrompt}>{slot.prompt}</Text>
              {slot.status === 'processing' && !slot.response ? (
                <ActivityIndicator color={theme.colors.primary} />
              ) : (
                <Text style={styles.slotResponse}>
                  {slot.response || 'Waiting...'}
                </Text>
              )}
              {slot.timings && slot.status === 'completed' && (
                <View
                  style={{
                    marginTop: 8,
                    paddingTop: 8,
                    borderTopWidth: 1,
                    borderTopColor: theme.colors.border,
                  }}
                >
                  <Text
                    style={{
                      fontSize: 11,
                      color: theme.colors.textSecondary,
                      marginBottom: 4,
                    }}
                  >
                    Performance:
                  </Text>
                  <View
                    style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 8 }}
                  >
                    {slot.timings.cache_n > 0 && (
                      <Text
                        style={{
                          fontSize: 10,
                          color: theme.colors.textSecondary,
                        }}
                      >
                        {`Cache: ${slot.timings.cache_n} tokens`}
                      </Text>
                    )}
                    <Text
                      style={{
                        fontSize: 10,
                        color: theme.colors.textSecondary,
                      }}
                    >
                      {`Prompt: ${slot.timings.prompt_per_second.toFixed(
                        2,
                      )} t/s`}
                    </Text>
                    <Text
                      style={{
                        fontSize: 10,
                        color: theme.colors.textSecondary,
                      }}
                    >
                      {`Generation: ${slot.timings.predicted_per_second.toFixed(
                        2,
                      )} t/s`}
                    </Text>
                  </View>
                </View>
              )}
            </View>
          ))
        )}
      </ScrollView>

      <CompletionParamsModal
        visible={showCompletionParamsModal}
        onClose={() => setShowCompletionParamsModal(false)}
        onSave={handleSaveCompletionParams}
      />
    </View>
  )
}
