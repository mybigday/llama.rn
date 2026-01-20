/* eslint-disable no-plusplus, no-await-in-loop, no-restricted-syntax */
import React, { useState, useEffect, useLayoutEffect, useRef } from 'react'
import {
  View,
  Text,
  ScrollView,
  Alert,
  TextInput,
  TouchableOpacity,
  Clipboard,
  StyleSheet,
} from 'react-native'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
import { MaskedProgress } from '../components/MaskedProgress'
import { HeaderButton } from '../components/HeaderButton'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type { ContextParams, CustomModel } from '../utils/storage'
import { loadContextParams, loadCustomModels } from '../utils/storage'
import { initLlama, LlamaContext } from '../../../src'

// Filter models to only include LLM models (no mmproj or vocoder)
const LLM_MODELS = Object.entries(MODELS).filter(([_key, model]) => {
  const modelWithExtras = model as typeof model & { vocoder?: any }
  return !modelWithExtras.vocoder
})

interface TestResult {
  name: string
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped'
  error?: string
  duration?: number
  raceConditionsCaught?: number
  totalCycles?: number // Total iterations/cycles for this test
}

// Known error codes from our null checks in JSICompletion.h
const RACE_CONDITION_ERRORS = [
  'RNLLAMA_NULL_CONTEXT',
  'RNLLAMA_NULL_COMPLETION',
  'RNLLAMA_NULL_LLAMA_CONTEXT',
  'RNLLAMA_NULL_SLOT',
]

const isRaceConditionError = (error: any): boolean => {
  // Check both message and the error itself as string
  const message = error?.message || ''
  const errorStr = String(error)
  const combined = `${message} ${errorStr}`
  return RACE_CONDITION_ERRORS.some((code) => combined.includes(code))
}

export default function StressTestScreen({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  // Track race conditions caught via global error handler (errors thrown outside Promise chains)
  const globalRaceConditionsRef = useRef(0)
  // Track if we're actively testing (to suppress error screen for expected race conditions)
  const isTestingRef = useRef(false)

  const styles = StyleSheet.create({
    container: themedStyles.container,
    setupContainer: themedStyles.setupContainer,
    scrollContent: themedStyles.scrollContent,
    setupDescription: themedStyles.setupDescription,
    testContainer: {
      flex: 1,
      padding: 16,
    },
    testButtonContainer: {
      marginBottom: 16,
      flexDirection: 'row' as const,
      flexWrap: 'wrap' as const,
      gap: 8,
    },
    testButton: {
      backgroundColor: theme.colors.primary,
      paddingHorizontal: 16,
      paddingVertical: 10,
      borderRadius: 8,
      alignItems: 'center' as const,
      minWidth: 120,
    },
    testButtonDisabled: {
      backgroundColor: theme.colors.disabled,
    },
    testButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '600' as const,
    },
    logContainer: {
      flex: 1,
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      padding: 12,
      marginTop: 16,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    logTitle: {
      fontSize: 16,
      fontWeight: '600' as const,
      color: theme.colors.text,
      marginBottom: 8,
    },
    logArea: {
      flex: 1,
      backgroundColor: theme.colors.inputBackground,
      borderRadius: 6,
      padding: 8,
      fontFamily: 'Courier',
      fontSize: 11,
      color: theme.colors.text,
      textAlignVertical: 'top' as const,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    logButton: {
      backgroundColor: theme.colors.buttonBackground,
      paddingHorizontal: 16,
      paddingVertical: 8,
      borderRadius: 6,
    },
    logButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '600' as const,
    },
    modelNameText: {
      fontSize: 18,
      fontWeight: '600' as const,
      color: theme.colors.text,
    },
    modelPathText: {
      fontSize: 14,
      color: theme.colors.textSecondary,
      marginTop: 4,
    },
    logControlsContainer: {
      flexDirection: 'row' as const,
      justifyContent: 'space-between' as const,
      marginTop: 8,
    },
    resultItem: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
      paddingVertical: 4,
    },
    resultStatus: {
      width: 20,
      textAlign: 'center' as const,
      marginRight: 8,
    },
    resultName: {
      flex: 1,
      color: theme.colors.text,
      fontSize: 13,
    },
    resultDuration: {
      color: theme.colors.textSecondary,
      fontSize: 12,
    },
    summaryContainer: {
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      padding: 12,
      marginBottom: 16,
    },
    summaryText: {
      color: theme.colors.text,
      fontSize: 14,
      fontWeight: '600' as const,
    },
    iterationsRow: {
      flexDirection: 'row' as const,
      alignItems: 'center' as const,
      marginBottom: 12,
      gap: 8,
    },
    iterationsInput: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 6,
      paddingHorizontal: 12,
      paddingVertical: 6,
      color: theme.colors.text,
      fontSize: 14,
      width: 60,
      textAlign: 'center' as const,
    },
  })

  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isTesting, setIsTesting] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [testResults, setTestResults] = useState<TestResult[]>([])
  const [modelInfo, setModelInfo] = useState<{
    name: string
    path: string
  } | null>(null)
  const [iterations, setIterations] = useState(5) // Unified iteration count for all tests

  const contextRef = useRef<LlamaContext | null>(null)
  const logInputRef = useRef<TextInput>(null)

  useEffect(() => {
    contextRef.current = context
  }, [context])

  useEffect(
    () => () => {
      if (contextRef.current) {
        contextRef.current.release()
      }
    },
    [],
  )

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

  // Set up global error handler to catch race conditions thrown outside Promise chains
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, prefer-destructuring
    const ErrorUtils = (global as any).ErrorUtils
    if (!ErrorUtils) return

    const originalHandler = ErrorUtils.getGlobalHandler()

    ErrorUtils.setGlobalHandler((error: any, isFatal: boolean) => {
      if (isRaceConditionError(error)) {
        globalRaceConditionsRef.current++
        setLogs((prev) => [
          ...prev,
          `[${new Date().toLocaleTimeString('en-US', {
            hour12: false,
          })}] 🔴 GLOBAL: Race condition caught - ${
            error?.message || String(error)
          }`,
        ])
        // During testing, suppress the error screen for expected race conditions
        // Don't change isTesting state - let the test loop continue
        if (isTestingRef.current) {
          return // Don't propagate to original handler (prevents red screen)
        }
      }
      // Call original handler for non-race-condition errors
      if (originalHandler) {
        originalHandler(error, isFatal)
      }
    })

    return () => {
      ErrorUtils.setGlobalHandler(originalHandler)
    }
  }, [])

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false })
    const logEntry = `[${timestamp}] ${message}`
    setLogs((prev) => [...prev, logEntry])
  }

  const clearLogs = () => {
    setLogs([])
    setTestResults([])
  }

  const generateReport = (
    results: TestResult[],
    _globalCaught: number,
    startTime: Date,
  ): string => {
    const endTime = new Date()
    const totalDuration = endTime.getTime() - startTime.getTime()
    const passedClean = results.filter(
      (r) => r.status === 'passed' && !r.raceConditionsCaught,
    ).length
    const passedWithRaces = results.filter(
      (r) => r.status === 'passed' && (r.raceConditionsCaught ?? 0) > 0,
    ).length
    const failed = results.filter((r) => r.status === 'failed').length
    const totalRaceCaught = results.reduce(
      (acc, r) => acc + (r.raceConditionsCaught || 0),
      0,
    )

    // Generate text report
    const lines: string[] = [
      '═══════════════════════════════════════════',
      '          STRESS TEST REPORT',
      '═══════════════════════════════════════════',
      '',
      `Timestamp: ${endTime.toISOString()}`,
      `Model: ${modelInfo?.name || 'Unknown'}`,
      `Path: ${modelInfo?.path || 'Unknown'}`,
      '',
      '───────────────────────────────────────────',
      '               SUMMARY',
      '───────────────────────────────────────────',
      '',
      `Total Tests:              ${results.length}`,
      `Passed (clean):           ${passedClean}`,
      `Passed (w/ races caught): ${passedWithRaces}`,
      `Failed:                   ${failed}`,
      `Total Duration:           ${totalDuration}ms`,
      `Race Conditions Caught:   ${totalRaceCaught}`,
      '',
      '───────────────────────────────────────────',
      '            TEST RESULTS',
      '───────────────────────────────────────────',
      '',
    ]

    const getStatusIcon = (status: string, racesCaught?: number): string => {
      if (status === 'passed' && racesCaught && racesCaught > 0) return '[WARN]'
      if (status === 'passed') return '[PASS]'
      if (status === 'failed') return '[FAIL]'
      return '[SKIP]'
    }

    for (const r of results) {
      const statusIcon = getStatusIcon(r.status, r.raceConditionsCaught)
      const durationStr = r.duration !== undefined ? `${r.duration}ms` : '-'
      // Show races as "3/10 races" so dev knows the ratio
      const raceStr = r.raceConditionsCaught
        ? `${r.raceConditionsCaught}/${r.totalCycles || '?'} races`
        : ''

      lines.push(`${statusIcon} ${r.name}`)
      const infoParts = [raceStr, durationStr].filter(Boolean)
      lines.push(`       ${infoParts.join(' | ')}`)
      if (r.error) {
        lines.push(`       Error: ${r.error}`)
      }
      lines.push('')
    }

    return lines.join('\n')
  }

  const copyLogs = () => {
    Clipboard.setString(logs.join('\n'))
  }

  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton iconName="refresh" onPress={clearLogs} />
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
  }, [navigation, isModelReady])

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleCustomModelAdded = async (_model: CustomModel) => {
    const models = await loadCustomModels()
    setCustomModels(models)
  }

  const handleCustomModelRemoved = async () => {
    const models = await loadCustomModels()
    setCustomModels(models)
  }

  const initializeModel = async (modelPath: string, modelKey?: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      let modelName: string
      if (modelKey) {
        const model = MODELS[modelKey as keyof typeof MODELS]
        modelName = model.name
      } else {
        modelName = modelPath.split('/').pop() || 'Custom Model'
      }

      setModelInfo({ name: modelName, path: modelPath })
      addLog(`Initializing model: ${modelName}`)

      const params = contextParams || (await loadContextParams())
      addLog(
        `Context params: n_ctx=${params.n_ctx}, n_gpu_layers=${params.n_gpu_layers}`,
      )

      const llamaContext = await initLlama(
        {
          model: modelPath,
          ...params,
        },
        (progress) => {
          setInitProgress(progress)
        },
      )

      setContext(llamaContext)
      setIsModelReady(true)
      setInitProgress(100)
      addLog('Model initialized successfully!')
    } catch (error: any) {
      addLog(`Failed to initialize: ${error.message}`)
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const updateTestResult = (name: string, update: Partial<TestResult>) => {
    setTestResults((prev) =>
      prev.map((r) => (r.name === name ? { ...r, ...update } : r)),
    )
  }

  const sleep = (ms: number) =>
    new Promise((resolve) => setTimeout(resolve, ms))

  // Test 1: Rapid completion start/stop
  const testRapidStartStop = async (
    ctx: LlamaContext,
  ): Promise<{ raceConditionsCaught: number; totalCycles: number }> => {
    addLog(`  Starting ${iterations} rapid completions with immediate stops...`)
    let raceConditionsCaught = 0

    for (let i = 0; i < iterations; i++) {
      try {
        const completionPromise = ctx.completion({
          prompt: 'Count to 100: 1, 2, 3,',
          n_predict: 100,
        })

        // Stop almost immediately
        await sleep(10 + Math.random() * 50)
        await ctx.stopCompletion()

        await completionPromise
        addLog(`    Iteration ${i + 1}: OK (completed)`)
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          raceConditionsCaught++
          addLog(
            `    Iteration ${i + 1}: CAUGHT RACE CONDITION - ${error.message}`,
          )
        } else if (error.message?.includes('interrupted')) {
          addLog(`    Iteration ${i + 1}: OK (interrupted)`)
        } else {
          throw error
        }
      }
    }

    if (raceConditionsCaught > 0) {
      addLog(
        `    >> Caught ${raceConditionsCaught} race condition(s) - null checks working!`,
      )
    }
    return { raceConditionsCaught, totalCycles: iterations }
  }

  // Test 2: Concurrent token callbacks with errors (no race conditions expected)
  const testTokenCallbackStress = async (ctx: LlamaContext): Promise<void> => {
    const expectedTokens = iterations * 10
    addLog(
      `  Running completion with heavy token callback (${expectedTokens} tokens)...`,
    )

    let tokenCount = 0
    let errorCount = 0

    await ctx.completion(
      {
        prompt: 'Write numbers: 1, 2, 3, 4, 5,',
        n_predict: expectedTokens,
      },
      () => {
        tokenCount++
        // Simulate some work in callback
        const arr = new Array(1000).fill(0).map((_, i) => i * i)
        if (arr.length < 0) errorCount++ // Never true, just to use arr
      },
    )

    addLog(`    Received ${tokenCount} tokens, errors: ${errorCount}`)
  }

  // Test 3: Multiple sequential completions (no race conditions expected)
  const testSequentialCompletions = async (
    ctx: LlamaContext,
  ): Promise<void> => {
    addLog(`  Running ${iterations} sequential completions...`)

    for (let i = 0; i < iterations; i++) {
      const result = await ctx.completion({
        prompt: `Say "${i}":`,
        n_predict: 10,
      })
      addLog(
        `    Completion ${i + 1}: ${result.text.trim().substring(0, 30)}...`,
      )
    }
  }

  // Test 4: Tokenize during completion (no race conditions expected)
  const testTokenizeDuringCompletion = async (
    ctx: LlamaContext,
  ): Promise<void> => {
    addLog(`  Testing ${iterations} tokenize ops during active completion...`)

    const completionPromise = ctx.completion({
      prompt: 'Write a story:',
      n_predict: 100,
    })

    // Run tokenize operations while completion is running
    const tokenizePromises = []
    for (let i = 0; i < iterations; i++) {
      await sleep(20)
      tokenizePromises.push(
        ctx.tokenize(`Test string ${i}`).catch((e) => ({ error: e.message })),
      )
    }

    const tokenizeResults = await Promise.all(tokenizePromises)
    await ctx.stopCompletion()
    await completionPromise.catch(() => {})

    const successCount = tokenizeResults.filter((r) => !('error' in r)).length
    addLog(`    Tokenize operations: ${successCount}/${iterations} succeeded`)
  }

  // Test 5: Rapid context release test (creates new context)
  const testRapidContextRelease = async (
    _ctx: LlamaContext,
    modelPath: string,
  ): Promise<{ raceConditionsCaught: number; totalCycles: number }> => {
    addLog(`  Testing ${iterations} rapid init/release cycles...`)
    let raceConditionsCaught = 0

    const params = contextParams || (await loadContextParams())

    for (let i = 0; i < iterations; i++) {
      let tempCtx: LlamaContext | null = null
      try {
        tempCtx = await initLlama({
          model: modelPath,
          ...params,
          n_ctx: 512, // Small context for faster init
        })

        // Start a completion but don't await it yet
        let completionError: any = null
        const completionPromise = tempCtx
          .completion({
            prompt: 'Hello',
            n_predict: 20,
          })
          .catch((e) => {
            completionError = e
          })

        // Release while completion might be running
        await sleep(50)
        await tempCtx.release().catch(() => {})
        tempCtx = null

        // Wait for completion to finish/fail with timeout
        await Promise.race([completionPromise, sleep(200)])

        // Check if we caught a race condition
        if (completionError && isRaceConditionError(completionError)) {
          raceConditionsCaught++
          addLog(
            `    Cycle ${i + 1}: CAUGHT RACE CONDITION - ${
              completionError.message || String(completionError)
            }`,
          )
        } else {
          addLog(`    Cycle ${i + 1}: OK`)
        }
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          raceConditionsCaught++
          addLog(`    Cycle ${i + 1}: CAUGHT RACE CONDITION - ${error.message}`)
        } else {
          addLog(`    Cycle ${i + 1}: ${error.message}`)
        }
        if (tempCtx) {
          await tempCtx.release().catch(() => {})
        }
      }
      // Small delay to let any pending async errors settle
      await sleep(100)
    }

    if (raceConditionsCaught > 0) {
      addLog(
        `    >> Caught ${raceConditionsCaught} race condition(s) - null checks working!`,
      )
    }
    return { raceConditionsCaught, totalCycles: iterations }
  }

  // Test 6: Completion with release during token callback
  const testReleaseInCallback = async (
    _ctx: LlamaContext,
    modelPath: string,
  ): Promise<{ raceConditionsCaught: number; totalCycles: number }> => {
    addLog('  Testing release during token callback...')
    let raceConditionsCaught = 0

    const params = contextParams || (await loadContextParams())
    let tempCtx: LlamaContext | null = null

    try {
      tempCtx = await initLlama({
        model: modelPath,
        ...params,
        n_ctx: 512,
      })

      let tokenCount = 0
      let released = false
      let completionError: any = null
      const ctxToRelease = tempCtx

      const completionPromise = tempCtx
        .completion(
          {
            prompt: 'Count to 100: 1, 2, 3,',
            n_predict: 50,
          },
          async () => {
            tokenCount++
            if (tokenCount === 10 && !released) {
              released = true
              addLog('    Releasing context at token 10...')
              await ctxToRelease.release().catch(() => {})
              tempCtx = null
            }
          },
        )
        .catch((e) => {
          completionError = e
        })

      // Wait for completion with timeout
      await Promise.race([
        completionPromise,
        sleep(5000), // 5 second timeout
      ])

      // Check results
      if (completionError) {
        if (isRaceConditionError(completionError)) {
          raceConditionsCaught++
          addLog(
            `    CAUGHT RACE CONDITION: ${
              completionError.message || String(completionError)
            }`,
          )
        } else {
          addLog(
            `    Completion ended: ${completionError.message || 'released'}`,
          )
        }
      }

      addLog(`    Tokens received before release: ${tokenCount}`)
    } catch (error: any) {
      if (isRaceConditionError(error)) {
        raceConditionsCaught++
        addLog(`    CAUGHT RACE CONDITION: ${error.message}`)
      } else {
        addLog(`    Error: ${error.message}`)
      }
    } finally {
      if (tempCtx) {
        await tempCtx.release().catch(() => {})
      }
      // Small delay to let any pending async errors settle
      await sleep(100)
    }

    if (raceConditionsCaught > 0) {
      addLog(
        `    >> Caught ${raceConditionsCaught} race condition(s) - null checks working!`,
      )
    }
    return { raceConditionsCaught, totalCycles: 1 }
  }

  // Test 7: Compare stopCompletion + release vs release only
  // This test tracks global errors per-section to properly measure race conditions
  // that are thrown asynchronously outside the Promise chain
  const testStopBeforeRelease = async (
    _ctx: LlamaContext,
    modelPath: string,
  ): Promise<{ raceConditionsCaught: number; totalCycles: number }> => {
    addLog('  Comparing stopCompletion+release vs release only...')
    let raceConditionsCaught = 0

    const params = contextParams || (await loadContextParams())

    // Part A: Release WITHOUT stopCompletion first
    addLog('    Part A: Release WITHOUT stopCompletion...')
    const globalBeforeA = globalRaceConditionsRef.current
    for (let i = 0; i < iterations; i++) {
      let tempCtx: LlamaContext | null = null
      try {
        tempCtx = await initLlama({
          model: modelPath,
          ...params,
          n_ctx: 512,
        })

        let completionError: any = null
        const completionPromise = tempCtx
          .completion({
            prompt: 'Write a long story about dragons and knights:',
            n_predict: 50,
          })
          .catch((e) => {
            completionError = e
          })

        // Wait a bit for completion to start generating tokens
        await sleep(100)

        // Release directly WITHOUT stopping first
        await tempCtx.release().catch(() => {})
        tempCtx = null

        // Wait for completion to finish/fail
        await Promise.race([completionPromise, sleep(200)])

        if (completionError && isRaceConditionError(completionError)) {
          addLog(`      Cycle ${i + 1}: RACE (promise)`)
        } else {
          addLog(`      Cycle ${i + 1}: OK`)
        }
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          addLog(`      Cycle ${i + 1}: RACE (thrown)`)
        } else {
          addLog(`      Cycle ${i + 1}: Error - ${error.message}`)
        }
        if (tempCtx) {
          await tempCtx.release().catch(() => {})
        }
      }
      // Wait for any async errors to be caught by global handler
      await sleep(200)
    }
    const raceWithoutStop = globalRaceConditionsRef.current - globalBeforeA

    // Part B: Release WITH stopCompletion first (but no await on promise)
    addLog('    Part B: Release WITH stopCompletion (no await)...')
    const globalBeforeB = globalRaceConditionsRef.current
    for (let i = 0; i < iterations; i++) {
      let tempCtx: LlamaContext | null = null
      try {
        tempCtx = await initLlama({
          model: modelPath,
          ...params,
          n_ctx: 512,
        })

        let completionError: any = null
        const completionPromise = tempCtx
          .completion({
            prompt: 'Write a long story about dragons and knights:',
            n_predict: 50,
          })
          .catch((e) => {
            completionError = e
          })

        // Wait a bit for completion to start generating tokens
        await sleep(100)

        // Stop completion first, then release WITHOUT waiting for promise
        await tempCtx.stopCompletion()
        await tempCtx.release().catch(() => {})
        tempCtx = null

        // Wait for completion to finish/fail
        await Promise.race([completionPromise, sleep(200)])

        if (completionError && isRaceConditionError(completionError)) {
          addLog(`      Cycle ${i + 1}: RACE (promise)`)
        } else {
          addLog(`      Cycle ${i + 1}: OK`)
        }
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          addLog(`      Cycle ${i + 1}: RACE (thrown)`)
        } else {
          addLog(`      Cycle ${i + 1}: Error - ${error.message}`)
        }
        if (tempCtx) {
          await tempCtx.release().catch(() => {})
        }
      }
      // Wait for any async errors to be caught by global handler
      await sleep(200)
    }
    const raceWithStop = globalRaceConditionsRef.current - globalBeforeB

    // Part C: Stop + await with timeout (safety net pattern)
    addLog('    Part C: Stop + await with timeout...')
    const globalBeforeC = globalRaceConditionsRef.current
    for (let i = 0; i < iterations; i++) {
      let tempCtx: LlamaContext | null = null
      try {
        tempCtx = await initLlama({
          model: modelPath,
          ...params,
          n_ctx: 512,
        })

        let completionError: any = null
        const completionPromise = tempCtx
          .completion({
            prompt: 'Write a long story about dragons and knights:',
            n_predict: 50,
          })
          .catch((e) => {
            completionError = e
          })

        // Wait a bit for completion to start generating tokens
        await sleep(100)

        // Pattern with timeout safety net
        await tempCtx.stopCompletion()
        await Promise.race([
          completionPromise,
          sleep(200), // Timeout just in case
        ])
        await tempCtx.release().catch(() => {})
        tempCtx = null

        if (completionError && isRaceConditionError(completionError)) {
          addLog(`      Cycle ${i + 1}: RACE (promise)`)
        } else {
          addLog(`      Cycle ${i + 1}: OK`)
        }
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          addLog(`      Cycle ${i + 1}: RACE (thrown)`)
        } else {
          addLog(`      Cycle ${i + 1}: Error - ${error.message}`)
        }
        if (tempCtx) {
          await tempCtx.release().catch(() => {})
        }
      }
      // Wait for any async errors to be caught by global handler
      await sleep(200)
    }
    const timeoutPatternRaces = globalRaceConditionsRef.current - globalBeforeC

    // Part D: Real-world pattern - stop + await promise directly + release
    addLog('    Part D: RECOMMENDED (stop + await promise + release)...')
    const globalBeforeD = globalRaceConditionsRef.current
    for (let i = 0; i < iterations; i++) {
      let tempCtx: LlamaContext | null = null
      let completionPromise: Promise<any> | null = null
      try {
        tempCtx = await initLlama({
          model: modelPath,
          ...params,
          n_ctx: 512,
        })

        completionPromise = tempCtx.completion({
          prompt: 'Write a long story about dragons and knights:',
          n_predict: 50,
        })

        // Wait a bit for completion to start generating tokens
        await sleep(100)

        // RECOMMENDED REAL-WORLD PATTERN:
        // 1. Stop completion
        // 2. Await the promise directly (no magic timeout)
        // 3. Release context
        await tempCtx.stopCompletion()
        if (completionPromise) {
          await completionPromise.catch(() => {}) // Wait for it to finish
        }
        completionPromise = null
        await tempCtx.release().catch(() => {})
        tempCtx = null

        addLog(`      Cycle ${i + 1}: OK`)
      } catch (error: any) {
        if (isRaceConditionError(error)) {
          addLog(`      Cycle ${i + 1}: RACE (thrown)`)
        } else {
          addLog(`      Cycle ${i + 1}: Error - ${error.message}`)
        }
        // Cleanup on error
        if (completionPromise) {
          await completionPromise.catch(() => {})
        }
        if (tempCtx) {
          await tempCtx.release().catch(() => {})
        }
      }
      // Wait for any async errors to be caught by global handler
      await sleep(200)
    }
    const recommendedPatternRaces =
      globalRaceConditionsRef.current - globalBeforeD

    // Calculate total
    raceConditionsCaught =
      raceWithoutStop +
      raceWithStop +
      timeoutPatternRaces +
      recommendedPatternRaces

    addLog('')
    addLog('    === COMPARISON RESULTS ===')
    addLog(
      `    A) Without stopCompletion:        ${raceWithoutStop} race(s) in ${iterations} cycles`,
    )
    addLog(
      `    B) With stop (no await):          ${raceWithStop} race(s) in ${iterations} cycles`,
    )
    addLog(
      `    C) Stop + await with timeout:     ${timeoutPatternRaces} race(s) in ${iterations} cycles`,
    )
    addLog(
      `    D) RECOMMENDED (stop+await+rel):  ${recommendedPatternRaces} race(s) in ${iterations} cycles`,
    )
    addLog('')
    if (
      recommendedPatternRaces === 0 &&
      (raceWithoutStop > 0 || raceWithStop > 0)
    ) {
      addLog('    ✅ RECOMMENDED PATTERN ELIMINATES RACE CONDITIONS!')
    }
    if (raceConditionsCaught > 0) {
      addLog(`    >> Total: ${raceConditionsCaught} race condition(s) caught`)
    }
    const totalCycles = iterations * 4 // 4 parts (A, B, C, D)
    return { raceConditionsCaught, totalCycles }
  }

  type TestFn = (
    ctx: LlamaContext,
    modelPath: string,
  ) => Promise<void | { raceConditionsCaught?: number; totalCycles?: number }>

  const runTest = async (name: string, testFn: TestFn) => {
    if (!context || !modelInfo) return

    updateTestResult(name, { status: 'running' })
    const startTime = Date.now()
    const globalBefore = globalRaceConditionsRef.current

    try {
      const result = await testFn(context, modelInfo.path)
      const duration = Date.now() - startTime
      // Include both test-reported and global race conditions
      const testRaces = result?.raceConditionsCaught || 0
      const globalRaces = globalRaceConditionsRef.current - globalBefore
      const totalRaces = testRaces + globalRaces
      const totalCycles = result?.totalCycles
      updateTestResult(name, {
        status: 'passed',
        duration,
        raceConditionsCaught: totalRaces,
        totalCycles,
      })
      if (totalRaces > 0) {
        addLog(
          `  PASSED (${duration}ms) - caught ${totalRaces} race condition(s)`,
        )
      } else {
        addLog(`  PASSED (${duration}ms)`)
      }
    } catch (error: any) {
      const duration = Date.now() - startTime
      const globalRaces = globalRaceConditionsRef.current - globalBefore
      if (isRaceConditionError(error)) {
        // Race condition caught at top level - this is actually a success
        const totalRaces = 1 + globalRaces
        updateTestResult(name, {
          status: 'passed',
          duration,
          raceConditionsCaught: totalRaces,
        })
        addLog(
          `  PASSED (${duration}ms) - caught ${totalRaces} race condition(s)`,
        )
      } else {
        updateTestResult(name, {
          status: 'failed',
          error: error.message,
          duration,
          raceConditionsCaught: globalRaces,
        })
        addLog(`  FAILED: ${error.message}`)
      }
    }
  }

  const runAllTests = async () => {
    if (!context || isTesting) return

    setIsTesting(true)
    isTestingRef.current = true
    clearLogs()
    globalRaceConditionsRef.current = 0 // Reset global counter
    const testStartTime = new Date()

    const tests: Array<{ name: string; fn: TestFn }> = [
      { name: 'Rapid Start/Stop', fn: testRapidStartStop },
      { name: 'Token Callback Stress', fn: testTokenCallbackStress },
      { name: 'Sequential Completions', fn: testSequentialCompletions },
      { name: 'Tokenize During Completion', fn: testTokenizeDuringCompletion },
      { name: 'Rapid Context Release', fn: testRapidContextRelease },
      { name: 'Release In Callback', fn: testReleaseInCallback },
      { name: 'Stop Before Release', fn: testStopBeforeRelease },
    ]

    setTestResults(tests.map((t) => ({ name: t.name, status: 'pending' })))

    try {
      addLog('=== Starting Stress Tests ===')
      addLog(`Model: ${modelInfo?.name}`)
      addLog('')

      for (const test of tests) {
        addLog(`[TEST] ${test.name}`)
        await runTest(test.name, test.fn)
        addLog('')

        // Small delay between tests
        await sleep(500)
      }

      addLog('=== Stress Tests Complete ===')
      addLog('')

      // Use callback to get latest state and generate report
      setTestResults((currentResults) => {
        const globalCaught = globalRaceConditionsRef.current
        const report = generateReport(
          currentResults,
          globalCaught,
          testStartTime,
        )
        // Append report to logs
        setLogs((prevLogs) => [...prevLogs, report])
        return currentResults
      })
    } catch (error: any) {
      addLog(`=== Tests aborted: ${error.message} ===`)
      if (isRaceConditionError(error)) {
        addLog('Race condition caught at top level')
      }
      addLog('')
      // Still generate report on error
      setTestResults((currentResults) => {
        const globalCaught = globalRaceConditionsRef.current
        const report = generateReport(
          currentResults,
          globalCaught,
          testStartTime,
        )
        // Append report to logs
        setLogs((prevLogs) => [...prevLogs, report])
        return currentResults
      })
    } finally {
      isTestingRef.current = false
      setIsTesting(false)
    }
  }

  const getStatusEmoji = (result: TestResult) => {
    if (result.status === 'passed' && (result.raceConditionsCaught ?? 0) > 0) {
      return '⚠️' // Passed but caught race conditions
    }
    switch (result.status) {
      case 'passed':
        return '✅'
      case 'failed':
        return '❌'
      case 'running':
        return '🔄'
      case 'skipped':
        return '⏭️'
      default:
        return '⏳'
    }
  }

  if (!isModelReady) {
    return (
      <View style={styles.container}>
        <ScrollView
          style={styles.setupContainer}
          contentContainerStyle={styles.scrollContent}
        >
          <Text style={styles.setupDescription}>
            Load a model to run stress tests. These tests help identify race
            conditions and crash scenarios in the native code.
          </Text>

          {customModels.filter((model) => !model.mmprojFilename).length > 0 && (
            <>
              <Text style={themedStyles.modelSectionTitle}>Custom Models</Text>
              {customModels
                .filter((model) => !model.mmprojFilename)
                .map((model) => (
                  <CustomModelCard
                    key={model.id}
                    model={model}
                    onInitialize={(modelPath: string) =>
                      initializeModel(modelPath)
                    }
                    onModelRemoved={handleCustomModelRemoved}
                    initializeButtonText="Test"
                  />
                ))}
            </>
          )}

          <TouchableOpacity
            style={themedStyles.addCustomModelButton}
            onPress={() => setShowCustomModelModal(true)}
          >
            <Text style={themedStyles.addCustomModelButtonText}>
              + Add Custom Model
            </Text>
          </TouchableOpacity>

          <Text style={themedStyles.modelSectionTitle}>Default Models</Text>
          {LLM_MODELS.map(([key, model]) => (
            <ModelDownloadCard
              key={key}
              title={model.name}
              repo={model.repo}
              filename={model.filename}
              size={model.size}
              initializeButtonText="Test"
              onInitialize={(modelPath: string) =>
                initializeModel(modelPath, key)
              }
            />
          ))}
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
          title="Add Custom Test Model"
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
      <View style={styles.testContainer}>
        {modelInfo && (
          <View style={{ marginBottom: 16 }}>
            <Text style={styles.modelNameText}>{modelInfo.name}</Text>
            <Text style={styles.modelPathText}>{modelInfo.path}</Text>
          </View>
        )}

        {/* Iterations input */}
        <View style={styles.iterationsRow}>
          <Text style={{ color: theme.colors.text }}>Iterations:</Text>
          <TextInput
            style={styles.iterationsInput}
            value={String(iterations)}
            onChangeText={(text) => {
              const num = parseInt(text, 10)
              if (Number.isFinite(num) && num > 0) setIterations(num)
              else if (text === '') setIterations(1)
            }}
            keyboardType="number-pad"
            editable={!isTesting}
          />
        </View>

        <View style={styles.testButtonContainer}>
          <TouchableOpacity
            style={[styles.testButton, isTesting && styles.testButtonDisabled]}
            onPress={runAllTests}
            disabled={isTesting}
          >
            <Text style={styles.testButtonText}>
              {isTesting ? 'Running...' : 'Run Tests'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Test results summary */}
        {testResults.length > 0 && (
          <View style={styles.summaryContainer}>
            <Text style={styles.summaryText}>
              {isTesting ? 'Progress:' : 'Results:'}
            </Text>
            {testResults.map((result) => (
              <View key={result.name} style={styles.resultItem}>
                <Text style={styles.resultStatus}>
                  {getStatusEmoji(result)}
                </Text>
                <Text style={styles.resultName}>{result.name}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Logs with report appended */}
        <View style={styles.logContainer}>
          <Text style={styles.logTitle}>Logs</Text>
          <TextInput
            ref={logInputRef}
            style={styles.logArea}
            value={logs.join('\n')}
            multiline
            editable={false}
            scrollEnabled
            placeholder="Test logs will appear here..."
          />
          <View style={styles.logControlsContainer}>
            <TouchableOpacity style={styles.logButton} onPress={clearLogs}>
              <Text style={styles.logButtonText}>Clear</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.logButton} onPress={copyLogs}>
              <Text style={styles.logButtonText}>Copy</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>

      <MaskedProgress
        visible={isTesting}
        text="Running stress tests..."
        progress={0}
        showProgressBar={false}
      />
    </View>
  )
}
