import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Modal,
  TouchableOpacity,
  Dimensions,
  Clipboard,
} from 'react-native'
import ModelDownloadCard, {
  TTSModelDownloadCard,
  MtmdModelDownloadCard,
} from '../components/ModelDownloadCard'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import { MaskedProgress } from '../components/MaskedProgress'
import type { CustomModel } from '../utils/storage'
import { loadCustomModels } from '../utils/storage'
import { loadLlamaModelInfo } from '../../../src' // import 'llama.rn'

const { width, height } = Dimensions.get('window')

interface ModelFileInfo {
  name: string
  path: string
  info: any
}

export default function ModelInfoScreen() {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const styles = StyleSheet.create({
    container: themedStyles.container,
    header: themedStyles.header,
    headerTitle: themedStyles.headerTitle,
    description: {
      ...themedStyles.description,
      marginHorizontal: 16,
    },
    setupContainer: themedStyles.setupContainer,
    scrollContent: themedStyles.scrollContent,
    modalContainer: themedStyles.modalContainer,
    modalContent: {
      ...themedStyles.modalContent,
      maxHeight: height * 0.8,
      maxWidth: width * 0.95,
      minWidth: width * 0.85,
    },
    modalHeader: {
      ...themedStyles.modalHeader,
      marginBottom: 16,
      paddingBottom: 12,
    },
    modalTitle: {
      ...themedStyles.modalTitle,
      flex: 1,
    },
    closeButton: {
      padding: 4,
    },
    closeButtonText: {
      fontSize: 18,
      color: theme.colors.primary,
      fontWeight: '600',
    },
    modalBody: {},
    infoContainer: {
      backgroundColor: theme.colors.card,
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
      color: theme.colors.text,
      flex: 1,
    },
    copyButton: {
      paddingHorizontal: 8,
      paddingVertical: 4,
      backgroundColor: theme.colors.primary,
      borderRadius: 4,
      marginLeft: 8,
    },
    copyButtonText: {
      fontSize: 12,
      color: theme.colors.white,
      fontWeight: '600',
    },
    infoValue: {
      fontSize: 14,
      color: theme.colors.textSecondary,
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
        color: theme.colors.textSecondary,
      },
  })

  const additionalStyles = StyleSheet.create({
    errorText: {
      color: theme.colors.error,
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
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      overflow: 'hidden',
    },
    individualModelScrollView: {
      maxHeight: 280,
      padding: 8,
    },
    fileHeaderContainer: {
      backgroundColor: theme.dark ? '#1E3A8A' : '#E3F2FD',
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.primary,
      marginBottom: 0,
    },
    fileHeaderText: {
      fontSize: 16,
      fontWeight: '600',
      color: theme.colors.primary,
    },
    separator: {
      height: 1,
      backgroundColor: theme.colors.border,
      marginVertical: 16,
    },
  })

  // Component state
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>([])
  const [isLoadingInfo, setIsLoadingInfo] = useState(false)
  const [infoError, setInfoError] = useState<string | null>(null)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])

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

  const handleMultimodalModelInitialize = async (
    modelPath: string,
    mmprojPath: string,
    modelKey: string,
  ) => {
    const model = MODELS[modelKey as keyof typeof MODELS]
    if (!model) return

    // Show model info modal for both model and mmproj
    await loadModelInfo(model.name, [
      { name: 'Model', path: modelPath },
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

  const handleCustomModelInitialize = async (
    customModel: CustomModel,
    modelPath: string,
    mmprojPath?: string,
  ) => {
    if (customModel.mmprojFilename && mmprojPath) {
      // Custom multimodal model
      await loadModelInfo(`${customModel.id} (${customModel.quantization})`, [
        { name: 'Model', path: modelPath },
        { name: 'MMProj', path: mmprojPath },
      ])
    } else {
      // Custom regular model
      await loadModelInfo(`${customModel.id} (${customModel.quantization})`, [
        { name: 'Model', path: modelPath },
      ])
    }
  }

  const closeModal = () => {
    setSelectedModel(null)
    setModelFiles([])
    setInfoError(null)
  }

  const copyToClipboard = (value: string) => {
    Clipboard.setString(value)
  }

  const renderSingleModelInfo = (fileInfo: ModelFileInfo) => {
    const { info } = fileInfo

    if (info.error) {
      return (
        <View style={styles.infoContainer}>
          <Text style={additionalStyles.errorText} selectable>
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
              onPress={() => copyToClipboard(displayValue)}
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
      return null // Loading will be handled by MaskedProgress
    }

    if (infoError) {
      return (
        <Text style={additionalStyles.errorText} selectable>
          {infoError}
        </Text>
      )
    }

    if (modelFiles.length === 0) {
      return (
        <Text style={additionalStyles.errorText} selectable>
          No model information available
        </Text>
      )
    }

    // Single model - use original layout
    if (modelFiles.length === 1) {
      return (
        <ScrollView
          style={additionalStyles.metadataScrollView}
          showsVerticalScrollIndicator
        >
          {modelFiles[0] && renderSingleModelInfo(modelFiles[0])}
        </ScrollView>
      )
    }

    // Multiple models - split into separate ScrollViews
    return (
      <View style={additionalStyles.multiModelContainer}>
        {modelFiles.map((fileInfo) => (
          <View key={fileInfo.path} style={additionalStyles.modelFileContainer}>
            <View style={[styles.infoContainer, additionalStyles.fileHeaderContainer]}>
              <Text style={additionalStyles.fileHeaderText}>{fileInfo.name}</Text>
            </View>
            <ScrollView
              style={additionalStyles.individualModelScrollView}
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
    <View style={styles.container}>
      <Text style={styles.description}>
        Download and explore detailed information about available AI models. Tap
        &quot;See&quot; on any model to view its metadata and specifications.
      </Text>

      <ScrollView
        style={styles.setupContainer}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Custom Models Section */}
        {customModels.length > 0 && (
          <>
            <Text style={themedStyles.modelSectionTitle}>Custom Models</Text>
            {customModels.map((customModel) => (
              <CustomModelCard
                key={customModel.id}
                model={customModel}
                onInitialize={(modelPath: string, mmprojPath?: string) => {
                  handleCustomModelInitialize(
                    customModel,
                    modelPath,
                    mmprojPath,
                  )
                }}
                onModelRemoved={handleCustomModelRemoved}
                initializeButtonText="See"
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
            // Multi-modal model
            return (
              <MtmdModelDownloadCard
                key={key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                mmproj={model.mmproj}
                size={model.size}
                initializeButtonText="See"
                onInitialize={(modelPath: string, mmprojPath: string) => {
                  handleMultimodalModelInitialize(modelPath, mmprojPath, key)
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
                <Text style={styles.closeButtonText}>X</Text>
              </TouchableOpacity>
            </View>
            <View style={styles.modalBody}>{renderModelInfo()}</View>
          </View>
        </View>
      </Modal>

      <CustomModelModal
        visible={showCustomModelModal}
        onClose={() => setShowCustomModelModal(false)}
        onModelAdded={handleCustomModelAdded}
        title="Add Custom Model"
        enableFileSelection
      />

      <MaskedProgress
        visible={isLoadingInfo}
        text="Loading model information..."
        progress={0}
        showProgressBar={false}
      />
    </View>
  )
}
