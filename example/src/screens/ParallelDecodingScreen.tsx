import React, { useState, useEffect, useRef, useCallback } from 'react'
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
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import { useTheme } from '../contexts/ThemeContext'
import { createThemedStyles } from '../styles/commonStyles'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import {
  useStoredCompletionParams,
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import { buildParallelStatePath } from '../features/parallelHelpers'
import {
  createExampleModelDefinitions,
  isTextGenerationModel,
  type ExampleModelKey,
} from '../utils/exampleModels'
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

const PARALLEL_MODELS = createExampleModelDefinitions(
  Object.entries(MODELS)
    .filter(([_key, model]) => model.mmproj || isTextGenerationModel(model))
    .map(([key]) => key as ExampleModelKey),
)

const SYSTEM_PROMPT =
  'You are a helpful AI assistant. Be concise and direct in your responses.'

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
  const styles = createStyles(theme, themedStyles)
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
  const slotsRef = useRef<ConversationSlot[]>([])
  const imageCache = useRef<Map<string, string>>(new Map())
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

  const showParallelInfo = useCallback(() => {
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
  }, [parallelSlots, isMultimodalEnabled])

  useExampleScreenHeader({
    navigation,
    isModelReady,
    readyActions: [
      {
        key: 'clear-slots',
        iconName: 'refresh',
        onPress: clearSlots,
      },
      {
        key: 'completion-settings',
        iconName: 'cog-outline',
        onPress: () => setShowCompletionParamsModal(true),
      },
      {
        key: 'parallel-info',
        iconName: 'information-outline',
        onPress: showParallelInfo,
      },
    ],
    setupActions: [
      {
        key: 'context-settings',
        iconName: 'cog-outline',
        onPress: () => setShowContextParamsModal(true),
      },
    ],
  })

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

  const settleConversationResult = async (
    slotId: string,
    completionPromise: Promise<{
      text?: string
      timings?: ConversationSlot['timings']
    }>,
  ) => {
    try {
      const result = await completionPromise
      updateSlot(slotId, {
        status: 'completed',
        response: result.text || '',
        endTime: Date.now(),
        timings: result.timings,
      })
      console.log('Timings:', result.timings)
    } catch (error) {
      console.error('Promise error:', error)
      updateSlot(slotId, {
        status: 'error',
        response: `${error}`,
        endTime: Date.now(),
      })
    }
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
      const messages = buildConversationMessages(prompt, images)
      const statePath = buildParallelStatePath(
        ReactNativeBlobUtil.fs.dirs.CacheDir,
        modelPathRef.current,
        prompt,
      )

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

      void settleConversationResult(slotId, promise)
    } catch (error) {
      console.error('Completion error:', error)
      updateSlot(slotId, {
        status: 'error',
        response: `Error: ${error}`,
        endTime: Date.now(),
      })
    }
  }

  const queueConversationStart = async (prompt: string, images?: string[]) => {
    try {
      await startConversation(prompt, images)
    } catch (error) {
      handleStartConversationError(prompt, error)
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

    const tempPath = `${
      ReactNativeBlobUtil.fs.dirs.CacheDir
    }/parallel_image_${Date.now()}_${Math.random().toString(36).slice(2)}.tmp`

    try {
      console.log(`Downloading image from ${url}`)
      const response = await withTimeout(
        ReactNativeBlobUtil.config({
          fileCache: true,
          followRedirect: true,
          path: tempPath,
        }).fetch('GET', url),
        15000,
        `Timed out downloading image from ${url}`,
      )
      const statusCode = response.info().status
      if (statusCode < 200 || statusCode >= 300) {
        throw new Error(`Image download failed with status ${statusCode}`)
      }

      const base64 = await withTimeout(
        ReactNativeBlobUtil.fs.readFile(tempPath, 'base64'),
        5000,
        `Timed out reading downloaded image from ${url}`,
      )
      const contentType =
        response.info().headers['content-type'] ||
        response.info().headers['Content-Type']
      const dataUri = `data:${normalizeImageMimeType(
        contentType,
      )};base64,${base64}`

      // Store in cache
      imageCache.current.set(url, dataUri)
      return dataUri
    } catch (error) {
      console.error(`Failed to download image from ${url}:`, error)
      throw error
    } finally {
      try {
        const tempFileExists = await ReactNativeBlobUtil.fs.exists(tempPath)
        if (tempFileExists) {
          await ReactNativeBlobUtil.fs.unlink(tempPath)
        }
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary image file:', cleanupError)
      }
    }
  }

  const sendExamplePrompts = async () => {
    if (isMultimodalEnabled) {
      await runSequentially(MULTIMODAL_EXAMPLE_PROMPTS, async (example) => {
        let images: string[] | undefined

        if (example.imageUrl) {
          try {
            const imageDataUri = await downloadImageAsDataUri(example.imageUrl)
            images = [imageDataUri]
          } catch (error) {
            console.error('Failed to download image for example:', error)
          }
        }

        await queueConversationStart(example.text, images)
      })
      return
    }

    await runSequentially(EXAMPLE_PROMPTS, async (prompt) => {
      await queueConversationStart(prompt)
    })
  }

  const sendCustomPrompt = () => {
    const prompt = customPrompt.trim()
    if (!prompt) {
      return
    }

    setCustomPrompt('')
    void queueConversationStart(prompt)
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

    const currentActiveCount = countActiveSlots(slots)
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

  const activeCount = countActiveSlots(slots)
  const completedCount = countCompletedSlots(slots)
  const avgDuration = calculateAverageDuration(slots)
  const activeRequests =
    parallelStatus?.requests.filter((request) => request.state !== 'queued') ||
    []
  const hasActiveRequests = activeCount > 0

  if (!isModelReady) {
    return (
      <>
        <ExampleModelSetup
          description="Download a model to see how llama.rn queues multiple requests, reuses prompt state, and runs them across parallel slots."
          defaultModels={PARALLEL_MODELS}
          customModels={customModels || []}
          onInitializeCustomModel={(_model, modelPath, mmprojPath) =>
            initializeModel(modelPath, mmprojPath)
          }
          onInitializeModel={(_model, modelPath, mmprojPath) =>
            initializeModel(modelPath, mmprojPath)
          }
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          customModelModalTitle="Add Custom Parallel Model"
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
    <View style={[styles.container, { paddingBottom: insets.bottom }]}>
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

        <View style={styles.slotCountSection}>
          <View style={styles.slotCountHeader}>
            <Text style={styles.slotCountLabel}>Number of Slots:</Text>
            <View style={styles.slotCountControls}>
              <TouchableOpacity
                style={[
                  styles.slotAdjustButton,
                  (parallelSlots <= 1 || hasActiveRequests) &&
                    styles.slotAdjustButtonDisabled,
                ]}
                onPress={() =>
                  parallelSlots > 1 && updateParallelSlots(parallelSlots - 1)
                }
                disabled={parallelSlots <= 1 || hasActiveRequests}
              >
                <Text style={styles.slotAdjustButtonText}>-</Text>
              </TouchableOpacity>
              <Text style={styles.slotCountValue}>{parallelSlots}</Text>
              <TouchableOpacity
                style={[
                  styles.slotAdjustButton,
                  (parallelSlots >= 8 || hasActiveRequests) &&
                    styles.slotAdjustButtonDisabled,
                ]}
                onPress={() =>
                  parallelSlots < 8 && updateParallelSlots(parallelSlots + 1)
                }
                disabled={parallelSlots >= 8 || hasActiveRequests}
              >
                <Text style={styles.slotAdjustButtonText}>+</Text>
              </TouchableOpacity>
            </View>
          </View>
          {hasActiveRequests && (
            <Text style={styles.slotCountHint}>
              Cannot change while requests are active
            </Text>
          )}
        </View>
      </View>

      {parallelStatus && (
        <View style={[styles.section, styles.statusSection]}>
          <Text style={styles.statusSectionTitle}>Slot Manager Status</Text>
          <View style={styles.statusLegendRow}>
            <View style={styles.statusLegendItem}>
              <View style={[styles.statusDot, styles.statusDotActive]} />
              <Text style={styles.statusLegendText}>
                {`Active: ${parallelStatus.active_slots}/${parallelStatus.n_parallel}`}
              </Text>
            </View>
            <View style={styles.statusLegendItem}>
              <View style={[styles.statusDot, styles.statusDotQueued]} />
              <Text style={styles.statusLegendText}>
                {`Queued: ${parallelStatus.queued_requests}`}
              </Text>
            </View>
          </View>
          {activeRequests.length > 0 && (
            <View style={styles.statusRequests}>
              <Text style={styles.statusRequestsLabel}>Active Requests:</Text>
              {activeRequests.slice(0, 3).map((req) => (
                <View key={req.request_id} style={styles.statusRequestRow}>
                  <Text
                    style={[
                      styles.statusRequestState,
                      {
                        color: getParallelRequestStatusColor(req.state, theme),
                      },
                    ]}
                  >
                    {`#${req.request_id} ${req.state}`}
                  </Text>
                  <Text style={styles.statusRequestDetails}>
                    {req.tokens_generated > 0
                      ? `${
                          req.tokens_generated
                        } tokens @ ${req.tokens_per_second.toFixed(1)} t/s`
                      : `prompt: ${req.prompt_length} tokens`}
                  </Text>
                </View>
              ))}
              {activeRequests.length > 3 && (
                <Text style={styles.statusRequestOverflow}>
                  {`+${activeRequests.length - 3} more...`}
                </Text>
              )}
            </View>
          )}
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Controls</Text>
        <View style={styles.controlsRow}>
          <TouchableOpacity
            style={[styles.button, hasActiveRequests && styles.buttonDisabled]}
            onPress={sendExamplePrompts}
            // disabled={hasActiveRequests}
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
            style={[styles.button, !hasActiveRequests && styles.buttonDisabled]}
            onPress={cancelAllRequests}
            disabled={!hasActiveRequests}
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
                    { color: getSlotStatusColor(slot.status, theme) },
                  ]}
                >
                  {slot.status}
                </Text>
                <View style={styles.slotMeta}>
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
                <View style={styles.timingsSection}>
                  <Text style={styles.timingsLabel}>Performance:</Text>
                  <View style={styles.timingsRow}>
                    {slot.timings.cache_n > 0 && (
                      <Text style={styles.timingText}>
                        {`Cache: ${slot.timings.cache_n} tokens`}
                      </Text>
                    )}
                    <Text style={styles.timingText}>
                      {`Prompt: ${slot.timings.prompt_per_second.toFixed(
                        2,
                      )} t/s`}
                    </Text>
                    <Text style={styles.timingText}>
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

function buildConversationMessages(prompt: string, images?: string[]) {
  const userContent: Array<{
    type: 'text' | 'image_url'
    text?: string
    image_url?: { url: string }
  }> = []

  if (images && images.length > 0) {
    images.forEach((imageUrl) => {
      userContent.push({
        type: 'image_url',
        image_url: { url: imageUrl },
      })
    })
  }

  userContent.push({
    type: 'text',
    text: prompt,
  })

  return [
    { role: 'system' as const, content: SYSTEM_PROMPT },
    { role: 'user' as const, content: userContent },
  ]
}

function countActiveSlots(slots: ConversationSlot[]) {
  return slots.filter(
    (slot) => slot.status === 'processing' || slot.status === 'idle',
  ).length
}

function countCompletedSlots(slots: ConversationSlot[]) {
  return slots.filter((slot) => slot.status === 'completed').length
}

function calculateAverageDuration(slots: ConversationSlot[]) {
  const completedSlots = slots.filter((slot) => slot.startTime && slot.endTime)
  if (completedSlots.length === 0) {
    return '0'
  }

  const totalDuration = completedSlots.reduce(
    (sum, slot) => sum + (slot.endTime! - slot.startTime!) / 1000,
    0,
  )
  return (totalDuration / completedSlots.length).toFixed(2)
}

function runSequentially<T>(items: T[], task: (item: T) => Promise<void>) {
  return items.reduce(
    (pending, item) => pending.then(() => task(item)),
    Promise.resolve(),
  )
}

function normalizeImageMimeType(contentType?: string) {
  if (!contentType || contentType === 'application/octet-stream') {
    return 'image/jpeg'
  }
  return contentType
}

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string,
) {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      setTimeout(() => {
        reject(new Error(timeoutMessage))
      }, timeoutMs)
    }),
  ])
}

function handleStartConversationError(prompt: string, error: unknown) {
  console.error(`Failed to start conversation for prompt: ${prompt}`, error)
}

function getSlotDuration(slot: ConversationSlot) {
  if (slot.startTime && slot.endTime) {
    return `${((slot.endTime - slot.startTime) / 1000).toFixed(2)}s`
  }
  return '-'
}

function getSlotStatusColor(
  status: ConversationSlot['status'],
  theme: ReturnType<typeof useTheme>['theme'],
) {
  if (status === 'processing') return '#FFA500'
  if (status === 'completed') return '#4CAF50'
  if (status === 'error') return '#F44336'
  return theme.colors.textSecondary
}

function getParallelRequestStatusColor(
  status: string,
  theme: ReturnType<typeof useTheme>['theme'],
) {
  if (status === 'generating') return '#4CAF50'
  if (status === 'processing_prompt') return '#FFA500'
  return theme.colors.textSecondary
}

function createStyles(
  theme: ReturnType<typeof useTheme>['theme'],
  themedStyles: ReturnType<typeof createThemedStyles>,
) {
  return StyleSheet.create({
    container: themedStyles.container,
    section: {
      padding: 12,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    sectionTitle: {
      fontSize: 16,
      fontWeight: '700' as const,
      color: theme.colors.text,
      marginBottom: 12,
    },
    statsRow: {
      flexDirection: 'row' as const,
      justifyContent: 'space-between' as const,
      marginBottom: 6,
    },
    statBox: {
      flex: 1,
      padding: 12,
      marginHorizontal: 4,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      alignItems: 'center' as const,
    },
    statLabel: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginBottom: 4,
    },
    statValue: {
      fontSize: 16,
      fontWeight: '700' as const,
      color: theme.colors.primary,
    },
    slotCountSection: {
      marginTop: 12,
    },
    slotCountHeader: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
      justifyContent: 'space-between' as const,
    },
    slotCountLabel: {
      fontSize: 14,
      color: theme.colors.text,
    },
    slotCountControls: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
    },
    slotAdjustButton: {
      backgroundColor: theme.colors.primary,
      width: 32,
      height: 32,
      borderRadius: 16,
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
    },
    slotAdjustButtonDisabled: {
      opacity: 0.5,
    },
    slotAdjustButtonText: {
      color: theme.colors.white,
      fontSize: 20,
      fontWeight: '700' as const,
    },
    slotCountValue: {
      fontSize: 18,
      fontWeight: '700' as const,
      color: theme.colors.primary,
      minWidth: 48,
      textAlign: 'center' as const,
    },
    slotCountHint: {
      fontSize: 11,
      color: theme.colors.textSecondary,
      marginTop: 4,
      textAlign: 'right' as const,
    },
    statusSection: {
      backgroundColor: theme.colors.surface,
      paddingVertical: 8,
    },
    statusSectionTitle: {
      fontSize: 14,
      fontWeight: '700' as const,
      color: theme.colors.text,
      marginBottom: 8,
    },
    statusLegendRow: {
      flexDirection: 'row' as const,
      flexWrap: 'wrap' as const,
      gap: 12,
    },
    statusLegendItem: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
    },
    statusDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      marginRight: 6,
    },
    statusDotActive: {
      backgroundColor: '#4CAF50',
    },
    statusDotQueued: {
      backgroundColor: '#FFA500',
    },
    statusLegendText: {
      fontSize: 12,
      color: theme.colors.text,
    },
    statusRequests: {
      marginTop: 8,
    },
    statusRequestsLabel: {
      fontSize: 11,
      color: theme.colors.textSecondary,
      marginBottom: 4,
    },
    statusRequestRow: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
      paddingVertical: 2,
    },
    statusRequestState: {
      fontSize: 10,
      width: 100,
    },
    statusRequestDetails: {
      fontSize: 10,
      color: theme.colors.textSecondary,
    },
    statusRequestOverflow: {
      fontSize: 10,
      color: theme.colors.textSecondary,
      fontStyle: 'italic' as const,
    },
    controlsRow: {
      flexDirection: 'row' as const,
      marginBottom: 12,
    },
    button: {
      flex: 1,
      padding: 4,
      height: 42,
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      marginHorizontal: 4,
      alignItems: 'center' as const,
      justifyContent: 'center' as const,
    },
    buttonDisabled: {
      backgroundColor: theme.colors.border,
    },
    buttonText: {
      color: theme.colors.white,
      fontWeight: '600' as const,
      fontSize: 12,
      textAlign: 'center' as const,
    },
    inputRow: {
      flexDirection: 'row' as const,
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
      justifyContent: 'center' as const,
      alignItems: 'center' as const,
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
      flexDirection: 'row' as const,
      justifyContent: 'space-between' as const,
      marginBottom: 8,
    },
    slotStatus: {
      fontSize: 12,
      fontWeight: '600' as const,
      textTransform: 'uppercase' as const,
    },
    slotMeta: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
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
      fontWeight: '600' as const,
    },
    slotPrompt: {
      fontSize: 14,
      fontWeight: '600' as const,
      color: theme.colors.text,
      marginBottom: 8,
    },
    slotResponse: {
      fontSize: 14,
      color: theme.colors.textSecondary,
      lineHeight: 20,
    },
    timingsSection: {
      marginTop: 8,
      paddingTop: 8,
      borderTopWidth: 1,
      borderTopColor: theme.colors.border,
    },
    timingsLabel: {
      fontSize: 11,
      color: theme.colors.textSecondary,
      marginBottom: 4,
    },
    timingsRow: {
      flexDirection: 'row' as const,
      flexWrap: 'wrap' as const,
      gap: 8,
    },
    timingText: {
      fontSize: 10,
      color: theme.colors.textSecondary,
    },
    emptyState: {
      flex: 1,
      justifyContent: 'center' as const,
      alignItems: 'center' as const,
      padding: 32,
    },
    emptyStateText: {
      fontSize: 16,
      color: theme.colors.textSecondary,
      textAlign: 'center' as const,
      lineHeight: 24,
    },
  })
}
