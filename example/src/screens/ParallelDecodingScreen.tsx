import React, { useState, useEffect, useRef, useLayoutEffect, useCallback } from 'react'
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
import { useTheme } from '../contexts/ThemeContext'
import ModelDownloadCard from '../components/ModelDownloadCard'
import { HeaderButton } from '../components/HeaderButton'
import { MaskedProgress } from '../components/MaskedProgress'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import {
  loadContextParams,
  loadCompletionParams,
} from '../utils/storage'
import { initLlama, LlamaContext } from '../../../src'

interface ConversationThread {
  id: string
  prompt: string
  response: string
  status: 'idle' | 'processing' | 'completed' | 'error'
  startTime?: number
  endTime?: number
  requestId: number
}

const SYSTEM_PROMPT = 'You are a helpful AI assistant. Be concise and direct in your responses.'

const EXAMPLE_PROMPTS = [
  'What is the capital of France?',
  'Explain quantum computing in simple terms.',
  'Write a haiku about coding.',
  'What are the primary colors?',
]

export default function ParallelDecodingScreen({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const insets = useSafeAreaInsets()

  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [threads, setThreads] = useState<ConversationThread[]>([])
  const [parallelSlots] = useState(2)
  const [customPrompt, setCustomPrompt] = useState('')
  const [isParallelMode, setIsParallelMode] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] = useState<CompletionParams | null>(null)
  const threadsRef = useRef<ConversationThread[]>([])

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const clearThreads = useCallback(() => {
    setThreads([])
    threadsRef.current = []
  }, [])

  // Set up header buttons
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton
              iconName="refresh"
              onPress={clearThreads}
            />
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
  }, [navigation, isModelReady, parallelSlots, clearThreads])

  // Cleanup on unmount
  useEffect(
    () => () => {
      if (context) {
        context.release()
      }
    },
    [context],
  )

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
      setIsParallelMode(true)
      setInitProgress(100)
      console.log(`Parallel mode ready with ${parallelSlots} slots`)
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const addThread = (prompt: string) => {
    const newThread: ConversationThread = {
      id: Math.random().toString(36).substring(2, 11),
      prompt,
      response: '',
      status: 'idle',
      requestId: -1, // Will be set when request is queued
    }

    setThreads((prev) => [...prev, newThread])
    threadsRef.current = [...threadsRef.current, newThread]
    return newThread
  }

  const updateThread = (id: string, updates: Partial<ConversationThread>) => {
    setThreads((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...updates } : t)),
    )
    threadsRef.current = threadsRef.current.map((t) =>
      t.id === id ? { ...t, ...updates } : t,
    )
  }

  const startConversation = async (prompt: string) => {
    if (!context || !isParallelMode) {
      Alert.alert('Error', 'Model not ready or parallel mode not enabled')
      return
    }

    const thread = addThread(prompt)
    const threadId = thread.id
    updateThread(threadId, { status: 'processing', startTime: Date.now() })

    try {
      // Load completion params
      const params = completionParams || (await loadCompletionParams())

      // Use queueCompletion for parallel processing with messages format
      const requestId = await context.queueCompletion(
        {
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
          ...params,
          reasoning_format: 'auto',
          n_predict: params.n_predict || 50,
          jinja: true,
        },
        (_reqId, data) => {
          const currentThread = threadsRef.current.find((t) => t.id === threadId)
          if (currentThread && data.token) {
            updateThread(threadId, {
              response: data.accumulated_text,
            })
          }
        },
        (_reqId, result) => {
          console.log('onComplete', result)
          const finalText = result.text || ''
          updateThread(threadId, {
            status: 'completed',
            response: finalText,
            endTime: Date.now(),
          })
        },
      )

      // Update thread with requestId
      updateThread(threadId, { requestId })
    } catch (error) {
      console.error('Completion error:', error)
      updateThread(threadId, {
        status: 'error',
        response: `Error: ${error}`,
        endTime: Date.now(),
      })
    }
  }

  const sendExamplePrompts = () => {
    EXAMPLE_PROMPTS.forEach((prompt, index) => {
      setTimeout(() => startConversation(prompt), index * 100)
    })
  }

  const sendCustomPrompt = () => {
    if (customPrompt.trim()) {
      startConversation(customPrompt)
      setCustomPrompt('')
    }
  }

  const cancelAllRequests = async () => {
    if (!context) return

    try {
      // Cancel all processing requests using cancelRequest
      const processingThreads = threads.filter((t) => t.status === 'processing')

      await Promise.all(
        processingThreads.map(async (thread) => {
          try {
            if (thread.requestId >= 0) {
              await context.cancelRequest(thread.requestId)
            }
            updateThread(thread.id, {
              status: 'error',
              endTime: Date.now(),
            })
          } catch (err) {
            console.error(`Error cancelling request ${thread.requestId}:`, err)
          }
        })
      )
    } catch (error) {
      console.error('Error cancelling requests:', error)
    }
  }

  const getThreadDuration = (thread: ConversationThread) => {
    if (thread.startTime && thread.endTime) {
      const duration = ((thread.endTime - thread.startTime) / 1000).toFixed(2)
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
      padding: 16,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    sectionTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      color: theme.colors.text,
      marginBottom: 12,
    },
    statsRow: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: 8,
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
      fontSize: 20,
      fontWeight: 'bold',
      color: theme.colors.primary,
    },
    controlsRow: {
      flexDirection: 'row',
      marginBottom: 12,
    },
    button: {
      flex: 1,
      padding: 12,
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      marginHorizontal: 4,
      alignItems: 'center',
    },
    buttonDisabled: {
      backgroundColor: theme.colors.border,
    },
    buttonText: {
      color: theme.colors.white,
      fontWeight: '600',
      fontSize: 14,
    },
    inputRow: {
      flexDirection: 'row',
      marginBottom: 12,
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
    threadsContainer: {
      flex: 1,
      padding: 16,
    },
    thread: {
      marginBottom: 16,
      padding: 12,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      borderLeftWidth: 4,
    },
    threadIdle: {
      borderLeftColor: theme.colors.border,
    },
    threadProcessing: {
      borderLeftColor: '#FFA500',
    },
    threadCompleted: {
      borderLeftColor: '#4CAF50',
    },
    threadError: {
      borderLeftColor: '#F44336',
    },
    threadHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      marginBottom: 8,
    },
    threadStatus: {
      fontSize: 12,
      fontWeight: '600',
      textTransform: 'uppercase',
    },
    threadDuration: {
      fontSize: 12,
      color: theme.colors.textSecondary,
    },
    threadPrompt: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 8,
    },
    threadResponse: {
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
          <Text style={{ fontSize: 16, color: theme.colors.text, marginBottom: 16 }}>
            Select a model to start parallel decoding demo
          </Text>
          {[
            'SMOL_LM_3',
            'GEMMA_3_4B_QAT',
            'QWEN_3_4B',
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

        <MaskedProgress
          visible={isLoading}
          text={`Initializing model... ${initProgress}%`}
          progress={initProgress}
          showProgressBar={initProgress > 0}
        />
      </View>
    )
  }

  const activeCount = threads.filter((t) => t.status === 'processing').length
  const completedCount = threads.filter((t) => t.status === 'completed').length
  const avgDuration =
    completedCount > 0
      ? (
          threads
            .filter((t) => t.startTime && t.endTime)
            .reduce(
              (sum, t) => sum + (t.endTime! - t.startTime!) / 1000,
              0,
            ) / completedCount
        ).toFixed(2)
      : '0'

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom }]}>
      {/* Stats Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Parallel Processing Stats</Text>
        <View style={styles.statsRow}>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Slots</Text>
            <Text style={styles.statValue}>{parallelSlots}</Text>
          </View>
          <View style={styles.statBox}>
            <Text style={styles.statLabel}>Active</Text>
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
      </View>

      {/* Controls Section */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Controls</Text>
        <View style={styles.controlsRow}>
          <TouchableOpacity
            style={[styles.button, activeCount > 0 && styles.buttonDisabled]}
            onPress={sendExamplePrompts}
            disabled={activeCount > 0}
          >
            <Text style={styles.buttonText}>Send 4 Examples</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={clearThreads}>
            <Text style={styles.buttonText}>Clear All</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, activeCount === 0 && styles.buttonDisabled]}
            onPress={cancelAllRequests}
            disabled={activeCount === 0}
          >
            <Text style={styles.buttonText}>Cancel All</Text>
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
            style={[styles.sendButton, !customPrompt.trim() && styles.buttonDisabled]}
            onPress={sendCustomPrompt}
            disabled={!customPrompt.trim()}
          >
            <Text style={styles.buttonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Threads Section */}
      <ScrollView style={styles.threadsContainer}>
        {threads.length === 0 ? (
          <View style={styles.emptyState}>
            <Text style={styles.emptyStateText}>
              {'No conversations yet.\n\nClick "Send 4 Examples" to see parallel processing in action,\nor enter a custom prompt below.'}
            </Text>
          </View>
        ) : (
          threads.map((thread) => (
            <View
              key={thread.id}
              style={[
                styles.thread,
                thread.status === 'idle' && styles.threadIdle,
                thread.status === 'processing' && styles.threadProcessing,
                thread.status === 'completed' && styles.threadCompleted,
                thread.status === 'error' && styles.threadError,
              ]}
            >
              <View style={styles.threadHeader}>
                <Text
                  style={[
                    styles.threadStatus,
                    {
                      color: (() => {
                        if (thread.status === 'processing') return '#FFA500'
                        if (thread.status === 'completed') return '#4CAF50'
                        if (thread.status === 'error') return '#F44336'
                        return theme.colors.textSecondary
                      })(),
                    },
                  ]}
                >
                  {thread.status}
                </Text>
                <Text style={styles.threadDuration}>
                  {getThreadDuration(thread)}
                </Text>
              </View>
              <Text style={styles.threadPrompt}>{thread.prompt}</Text>
              {thread.status === 'processing' && !thread.response ? (
                <ActivityIndicator color={theme.colors.primary} />
              ) : (
                <Text style={styles.threadResponse}>
                  {thread.response || 'Waiting...'}
                </Text>
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
