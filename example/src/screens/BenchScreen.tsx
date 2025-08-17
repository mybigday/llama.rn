import React, { useState, useEffect, useLayoutEffect, useRef } from 'react'
import {
  View,
  Text,
  ScrollView,
  Alert,
  TextInput,
  TouchableOpacity,
  Clipboard,
} from 'react-native'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
import { MaskedProgress } from '../components/MaskedProgress'
import { HeaderButton } from '../components/HeaderButton'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CustomModel } from '../utils/storage'
import { loadContextParams, loadCustomModels } from '../utils/storage'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

const styles = {
  container: CommonStyles.container,
  setupContainer: CommonStyles.setupContainer,
  scrollContent: CommonStyles.scrollContent,
  setupDescription: CommonStyles.setupDescription,
  benchContainer: {
    flex: 1,
    padding: 16,
  },
  benchButtonContainer: {
    marginBottom: 16,
  },
  benchButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center' as const,
  },
  benchButtonDisabled: {
    backgroundColor: '#CCCCCC',
  },
  benchButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600' as const,
  },
  logContainer: {
    flex: 1,
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    padding: 12,
    marginTop: 16,
  },
  logTitle: {
    fontSize: 16,
    fontWeight: '600' as const,
    color: '#333',
    marginBottom: 8,
  },
  logArea: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    borderRadius: 6,
    padding: 8,
    fontFamily: 'Courier',
    fontSize: 12,
    color: '#333',
    textAlignVertical: 'top' as const,
    minHeight: 200,
  },
  logControls: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between',
    marginTop: 8,
  },
  logButton: {
    backgroundColor: '#6C757D',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  logButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '600' as const,
  },
  modelNameText: {
    fontSize: 18,
    fontWeight: '600' as const,
    color: '#333',
  },
  modelPathText: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  logControlsContainer: {
    flexDirection: 'row' as const,
    justifyContent: 'space-between' as const,
    marginTop: 8,
  },
}

// Filter models to only include LLM models (no mmproj or vocoder)
const LLM_MODELS = Object.entries(MODELS).filter(([_key, model]) => {
  const modelWithExtras = model as typeof model & { vocoder?: any }
  return !modelWithExtras.vocoder
})

export default function BenchScreen({ navigation }: { navigation: any }) {
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isBenching, setIsBenching] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [modelInfo, setModelInfo] = useState<{
    name: string
    path: string
  } | null>(null)

  const logsRef = useRef<string[]>([])

  // Sync logs state with ref for better performance
  useEffect(() => {
    logsRef.current = logs
  }, [logs])

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

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString()
    const logEntry = `[${timestamp}] ${message}`
    setLogs((prev) => [...prev, logEntry])
  }

  const clearLogs = () => {
    setLogs([])
  }

  const copyLogs = () => {
    Clipboard.setString(logs.join('\n'))
  }

  // Set up navigation header buttons
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
    // Reload custom models to reflect the new addition
    const models = await loadCustomModels()
    setCustomModels(models)
  }

  const handleCustomModelRemoved = async () => {
    // Reload custom models to reflect the removal
    const models = await loadCustomModels()
    setCustomModels(models)
  }

  const initializeModel = async (modelPath: string, modelKey?: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      let modelName: string
      if (modelKey) {
        // Predefined model
        const model = MODELS[modelKey as keyof typeof MODELS]
        modelName = model.name
      } else {
        // Custom model - extract name from path
        modelName = modelPath.split('/').pop() || 'Custom Model'
      }

      setModelInfo({ name: modelName, path: modelPath })

      addLog(`Initializing model: ${modelName}`)
      addLog(`Model path: ${modelPath}`)

      const params = contextParams || (await loadContextParams())
      addLog(`Using context params: ${JSON.stringify(params)}`)

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
      addLog('Ready to run benchmarks.')
    } catch (error: any) {
      addLog(`Failed to initialize model: ${error.message}`)
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const runBenchmark = async () => {
    if (!context || isBenching) return

    try {
      setIsBenching(true)
      addLog('🚀 Starting benchmark...')

      // Heat-up phase
      addLog('Warming up the model...')
      const t0 = Date.now()
      await context.bench(8, 4, 1, 1)
      const tHeat = Date.now() - t0
      if (tHeat > 1e4) {
        addLog('Heat up time is too long, please try again.')
        return
      }
      addLog(`Heat up time: ${tHeat}ms`)

      // Main benchmark
      addLog('Benchmarking the model...')
      const { modelDesc, modelSize, modelNParams, ppAvg, ppStd, tgAvg, tgStd } =
        await context.bench(512, 128, 1, 3)

      const size = `${(modelSize / 1024.0 / 1024.0 / 1024.0).toFixed(2)} GiB`
      const nParams = `${(modelNParams / 1e9).toFixed(2)}B`
      const md =
        '| model | size | params | test | t/s |\n' +
        '| --- | --- | --- | --- | --- |\n' +
        `| ${modelDesc} | ${size} | ${nParams} | pp 512 | ${ppAvg.toFixed(
          2,
        )} ± ${ppStd.toFixed(2)} |\n` +
        `| ${modelDesc} | ${size} | ${nParams} | tg 128 | ${tgAvg.toFixed(
          2,
        )} ± ${tgStd.toFixed(2)} |`

      addLog('')
      addLog('📊 Benchmark Results:')
      addLog(md)
      addLog('')
      addLog('✅ Benchmark completed successfully!')
    } catch (error: any) {
      addLog(`❌ Benchmark failed: ${error.message}`)
      Alert.alert('Error', `Benchmark failed: ${error.message}`)
    } finally {
      setIsBenching(false)
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
            Download a language model to run performance benchmarks. Benchmarks
            measure prompt processing and text generation speeds.
          </Text>

          {/* Custom Models Section */}
          {customModels.filter((model) => !model.mmprojFilename).length > 0 && (
            <>
              <Text style={CommonStyles.modelSectionTitle}>
                Custom Models
              </Text>
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
                    initializeButtonText="Bench"
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
          {LLM_MODELS.map(([key, model]) => (
            <ModelDownloadCard
              key={key}
              title={model.name}
              repo={model.repo}
              filename={model.filename}
              size={model.size}
              initializeButtonText="Bench"
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
          title="Add Custom Benchmark Model"
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
      <ScrollView style={styles.benchContainer}>
        {modelInfo && (
          <View style={{ marginBottom: 16 }}>
            <Text style={styles.modelNameText}>{modelInfo.name}</Text>
            <Text style={styles.modelPathText}>{modelInfo.path}</Text>
          </View>
        )}

        <View style={styles.benchButtonContainer}>
          <TouchableOpacity
            style={[
              styles.benchButton,
              (isBenching || isLoading) && styles.benchButtonDisabled,
            ]}
            onPress={runBenchmark}
            disabled={isBenching || isLoading}
          >
            <Text style={styles.benchButtonText}>
              {isBenching ? 'Running Benchmark...' : 'Start Benchmark'}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.logContainer}>
          <Text style={styles.logTitle}>Benchmark Logs</Text>
          <TextInput
            style={styles.logArea}
            value={logs.join('\n')}
            multiline
            editable={false}
            placeholder="Benchmark logs will appear here..."
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
      </ScrollView>

      <MaskedProgress
        visible={isBenching}
        text="Running benchmark..."
        progress={0}
        showProgressBar={false}
      />
    </View>
  )
}
