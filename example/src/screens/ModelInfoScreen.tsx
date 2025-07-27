import React, { useState } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Modal,
  TouchableOpacity,
  Dimensions,
  Clipboard,
  Alert,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { loadLlamaModelInfo } from '../../../src'
import ModelDownloadCard, {
  TTSModelDownloadCard,
  VLMModelDownloadCard,
} from '../components/ModelDownloadCard'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'

const { width, height } = Dimensions.get('window')

const styles = StyleSheet.create({
  // Using shared styles for common patterns
  container: CommonStyles.container,
  header: CommonStyles.header,
  headerTitle: CommonStyles.headerTitle,
  description: {
    ...CommonStyles.description,
    marginHorizontal: 16,
  },
  scrollContent: CommonStyles.scrollContent,
  // Modal styles - using shared base with customizations
  modalContainer: CommonStyles.modalContainer,
  modalContent: {
    ...CommonStyles.modalContent,
    maxHeight: height * 0.8,
    maxWidth: width * 0.95,
    minWidth: width * 0.85,
  },
  modalHeader: {
    ...CommonStyles.modalHeader,
    marginBottom: 16,
    paddingBottom: 12,
  },
  modalTitle: {
    ...CommonStyles.modalTitle,
    flex: 1,
  },
  closeButton: {
    padding: 4,
  },
  closeButtonText: {
    fontSize: 18,
    color: '#007AFF',
    fontWeight: '600',
  },
  modalBody: {},
  infoContainer: {
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  infoLabelContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  infoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    flex: 1,
  },
  copyButton: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    backgroundColor: '#007AFF',
    borderRadius: 4,
    marginLeft: 8,
  },
  copyButtonText: {
    fontSize: 12,
    color: 'white',
    fontWeight: '600',
  },
  infoValue: {
    fontSize: 14,
    color: '#666',
    fontFamily: 'Courier',
  },
  loadingContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 8,
    fontSize: 14,
    color: '#666',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    textAlign: 'center',
    padding: 20,
  },
  metadataScrollView: {
    maxHeight: 300,
  },
  multiModelContainer: {
    flexDirection: 'row',
    gap: 12,
  },
  modelFileContainer: {
    flex: 1,
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
    overflow: 'hidden',
  },
  individualModelScrollView: {
    maxHeight: 280,
    padding: 8,
  },
  fileHeaderContainer: {
    backgroundColor: '#E3F2FD',
    borderLeftWidth: 4,
    borderLeftColor: '#007AFF',
    marginBottom: 0,
  },
  fileHeaderText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#007AFF',
  },
  separator: {
    height: 1,
    backgroundColor: '#E0E0E0',
    marginVertical: 16,
  },
})

interface ModelFileInfo {
  name: string
  path: string
  info: any
}

export default function ModelInfoScreen() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>([])
  const [isLoadingInfo, setIsLoadingInfo] = useState(false)
  const [infoError, setInfoError] = useState<string | null>(null)

  const loadModelInfo = async (
    modelName: string,
    files: Array<{ name: string; path: string }>,
  ) => {
    setIsLoadingInfo(true)
    setInfoError(null)
    setModelFiles([])
    setSelectedModel(modelName)

    try {
      const fileInfoPromises = files.map(async (file) => {
        try {
          const info = await loadLlamaModelInfo(file.path)
          return {
            name: file.name,
            path: file.path,
            info,
          }
        } catch (fileError) {
          console.warn(`Failed to load info for ${file.name}:`, fileError)
          // Still add an entry with error info
          return {
            name: file.name,
            path: file.path,
            info: {
              error: `Failed to load: ${
                fileError instanceof Error ? fileError.message : 'Unknown error'
              }`,
            },
          }
        }
      })

      const fileInfos = await Promise.all(fileInfoPromises)
      setModelFiles(fileInfos)
    } catch (error) {
      console.error('Failed to load model info:', error)
      setInfoError(
        error instanceof Error
          ? error.message
          : 'Failed to load model information',
      )
    } finally {
      setIsLoadingInfo(false)
    }
  }

  const handleRegularModelInitialize = async (
    modelPath: string,
    modelKey: string,
  ) => {
    const model = MODELS[modelKey as keyof typeof MODELS]
    if (!model) return

    // Show model info modal
    await loadModelInfo(model.name, [{ name: 'Model', path: modelPath }])
  }

  const handleVLMModelInitialize = async (
    modelPath: string,
    mmprojPath: string,
    modelKey: string,
  ) => {
    const model = MODELS[modelKey as keyof typeof MODELS]
    if (!model) return

    // Show model info modal for both model and mmproj
    await loadModelInfo(model.name, [
      { name: 'Vision Model', path: modelPath },
      { name: 'MMProj', path: mmprojPath },
    ])
  }

  const handleTTSModelInitialize = async (
    ttsPath: string,
    vocoderPath: string,
    modelKey: string,
  ) => {
    const model = MODELS[modelKey as keyof typeof MODELS]
    if (!model) return

    // Show model info modal for both TTS model and vocoder
    await loadModelInfo(model.name, [
      { name: 'TTS Model', path: ttsPath },
      { name: 'Vocoder', path: vocoderPath },
    ])
  }

  const closeModal = () => {
    setSelectedModel(null)
    setModelFiles([])
    setInfoError(null)
  }

  const copyToClipboard = (value: string, label: string) => {
    Clipboard.setString(value)
    Alert.alert('Copied', `${label} copied to clipboard`)
  }

  const renderSingleModelInfo = (fileInfo: ModelFileInfo) => {
    const { info } = fileInfo

    if (info.error) {
      return (
        <View style={styles.infoContainer}>
          <Text style={styles.errorText} selectable>
            {info.error}
          </Text>
        </View>
      )
    }

    return Object.entries(info).map(([key, value]) => {
      const displayValue =
        typeof value === 'object'
          ? JSON.stringify(value, null, 2)
          : String(value)

      return (
        <View key={key} style={styles.infoContainer}>
          <View style={styles.infoLabelContainer}>
            <Text style={styles.infoLabel}>{key}</Text>
            <TouchableOpacity
              style={styles.copyButton}
              onPress={() => copyToClipboard(displayValue, key)}
            >
              <Text style={styles.copyButtonText}>Copy</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.infoValue} selectable>
            {displayValue}
          </Text>
        </View>
      )
    })
  }

  const renderModelInfo = () => {
    if (isLoadingInfo) {
      return (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>Loading model information...</Text>
        </View>
      )
    }

    if (infoError) {
      return (
        <Text style={styles.errorText} selectable>
          {infoError}
        </Text>
      )
    }

    if (modelFiles.length === 0) {
      return (
        <Text style={styles.errorText} selectable>
          No model information available
        </Text>
      )
    }

    // Single model - use original layout
    if (modelFiles.length === 1) {
      return (
        <ScrollView
          style={styles.metadataScrollView}
          showsVerticalScrollIndicator
        >
          {modelFiles[0] && renderSingleModelInfo(modelFiles[0])}
        </ScrollView>
      )
    }

    // Multiple models - split into separate ScrollViews
    return (
      <View style={styles.multiModelContainer}>
        {modelFiles.map((fileInfo) => (
          <View key={fileInfo.path} style={styles.modelFileContainer}>
            <View style={[styles.infoContainer, styles.fileHeaderContainer]}>
              <Text style={styles.fileHeaderText}>{fileInfo.name}</Text>
            </View>
            <ScrollView
              style={styles.individualModelScrollView}
              showsVerticalScrollIndicator
              nestedScrollEnabled
            >
              {renderSingleModelInfo(fileInfo)}
            </ScrollView>
          </View>
        ))}
      </View>
    )
  }

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.description}>
        Download and explore detailed information about available AI models. Tap
        &quot;See&quot; on any model to view its metadata and specifications.
      </Text>

      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.scrollContent}
      >
        {Object.entries(MODELS).map(([key, model]) => {
          // Type assertion to access potential vocoder property
          const modelWithVocoder = model as typeof model & { vocoder?: any }

          if (model.mmproj && modelWithVocoder.vocoder) {
            // Multi-modal TTS model (like OUTE_TTS with vocoder) - currently no such model exists
            return (
              <TTSModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                vocoder={modelWithVocoder.vocoder}
                initializeButtonText="See"
                onInitialize={(ttsPath: string, vocoderPath: string) => {
                  handleTTSModelInitialize(ttsPath, vocoderPath, key)
                }}
              />
            )
          } else if (model.mmproj) {
            // Vision/multi-modal model
            return (
              <VLMModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                mmproj={model.mmproj}
                size={model.size}
                initializeButtonText="See"
                onInitialize={(modelPath: string, mmprojPath: string) => {
                  handleVLMModelInitialize(modelPath, mmprojPath, key)
                }}
              />
            )
          } else if (modelWithVocoder.vocoder) {
            // TTS model with vocoder
            return (
              <TTSModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                vocoder={modelWithVocoder.vocoder}
                initializeButtonText="See"
                onInitialize={(ttsPath: string, vocoderPath: string) => {
                  handleTTSModelInitialize(ttsPath, vocoderPath, key)
                }}
              />
            )
          } else {
            // Regular text model
            return (
              <ModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                initializeButtonText="See"
                onInitialize={(modelPath: string) => {
                  handleRegularModelInitialize(modelPath, key)
                }}
              />
            )
          }
        })}
      </ScrollView>

      {/* Model Info Modal */}
      <Modal
        visible={selectedModel !== null}
        transparent
        animationType="fade"
        onRequestClose={closeModal}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{selectedModel}</Text>
              <TouchableOpacity style={styles.closeButton} onPress={closeModal}>
                <Text style={styles.closeButtonText}>✕</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.modalBody}>{renderModelInfo()}</View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  )
}
