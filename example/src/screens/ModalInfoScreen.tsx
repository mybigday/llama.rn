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
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { loadLlamaModelInfo } from '../../../src'
import ModelDownloadCard, {
  TTSModelDownloadCard,
  VLMModelDownloadCard,
} from '../components/ModelDownloadCard'
import { MODELS } from '../utils/constants'

const { width, height } = Dimensions.get('window')

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
  description: {
    fontSize: 16,
    color: '#666',
    lineHeight: 24,
    marginVertical: 16,
    marginHorizontal: 16,
    textAlign: 'center',
  },
  scrollContent: {
    paddingBottom: 20,
  },
  // Modal styles
  modalContainer: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 20,
    margin: 20,
    maxHeight: height * 0.8,
    maxWidth: width * 0.95,
    minWidth: width * 0.85,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
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
  infoLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
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

export default function ModalInfoScreen() {
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

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
  }

  const formatNumber = (num: number): string => {
    if (num >= 1e9) {
      return `${(num / 1e9).toFixed(1)}B`
    } else if (num >= 1e6) {
      return `${(num / 1e6).toFixed(1)}M`
    } else if (num >= 1e3) {
      return `${(num / 1e3).toFixed(1)}K`
    }
    return num.toString()
  }

  const renderSingleModelInfo = (fileInfo: ModelFileInfo) => {
    const { info } = fileInfo

    if (info.error) {
      return (
        <View style={styles.infoContainer}>
          <Text style={styles.errorText}>{info.error}</Text>
        </View>
      )
    }

    return (
      <>
        {info.desc && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoLabel}>Description</Text>
            <Text style={styles.infoValue}>{info.desc}</Text>
          </View>
        )}

        {info.nParams && info.nParams > 0 && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoLabel}>Parameters</Text>
            <Text style={styles.infoValue}>{formatNumber(info.nParams)}</Text>
          </View>
        )}

        {info.nEmbd && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoLabel}>Embedding Dimensions</Text>
            <Text style={styles.infoValue}>{info.nEmbd.toLocaleString()}</Text>
          </View>
        )}

        {info.size > 0 && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoLabel}>Model Size</Text>
            <Text style={styles.infoValue}>{formatBytes(info.size)}</Text>
          </View>
        )}

        {info.chatTemplates && (
          <View style={styles.infoContainer}>
            <Text style={styles.infoLabel}>Chat Templates</Text>
            <Text style={styles.infoValue}>
              {`Llama Chat: ${info.chatTemplates.llamaChat ? 'Yes' : 'No'}`}
              {'\n'}
              {`Jinja Default: ${
                info.chatTemplates.minja?.default ? 'Yes' : 'No'
              }`}
              {'\n'}
              {`Jinja Tool Use: ${
                info.chatTemplates.minja?.toolUse ? 'Yes' : 'No'
              }`}
            </Text>
          </View>
        )}

        {/* Show other metadata */}
        {Object.entries(info)
          .filter(
            ([key]) =>
              ![
                'desc',
                'nEmbd',
                'nParams',
                'size',
                'chatTemplates',
                'error',
              ].includes(key),
          )
          .map(([key, value]) => (
            <View key={key} style={styles.infoContainer}>
              <Text style={styles.infoLabel}>{key}</Text>
              <Text style={styles.infoValue}>
                {typeof value === 'object'
                  ? JSON.stringify(value, null, 2)
                  : String(value)}
              </Text>
            </View>
          ))}
      </>
    )
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
      return <Text style={styles.errorText}>{infoError}</Text>
    }

    if (modelFiles.length === 0) {
      return (
        <Text style={styles.errorText}>No model information available</Text>
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
