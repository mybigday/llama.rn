import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  TextInput,
  ScrollView,
  Modal,
  Platform,
} from 'react-native'
import { pick, keepLocalCopy } from '@react-native-documents/picker'
import {
  HuggingFaceAPI,
  type CustomModelInfo,
} from '../services/HuggingFaceAPI'
import { saveCustomModel, type CustomModel } from '../utils/storage'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface CustomModelModalProps {
  visible: boolean
  onClose: () => void
  onModelAdded?: (model: CustomModel) => void
  requireMMProj?: boolean // For multimodal screens
  title?: string
  enableFileSelection?: boolean // Enable file selection mode
}

export default function CustomModelModal({
  visible,
  onClose,
  onModelAdded,
  requireMMProj = false,
  title = 'Add Custom Model',
  enableFileSelection = false,
}: CustomModelModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const styles = StyleSheet.create({
    container: {
      ...themedStyles.container,
      backgroundColor: theme.colors.background,
    },
    header: {
      ...themedStyles.modalHeader,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      paddingHorizontal: 24,
      paddingVertical: 18,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 2,
      elevation: 2,
    },
    title: {
      ...themedStyles.modalTitle,
      fontSize: 18,
      fontWeight: '700',
      color: theme.colors.text,
    },
    cancelButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.textSecondary,
      fontSize: 16,
      fontWeight: '500',
    },
    saveButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.primary,
      fontWeight: '700',
      fontSize: 16,
    },
    disabledButton: {
      ...themedStyles.disabledButton,
      color: theme.colors.textSecondary,
    },
    content: {
      flex: 1,
      paddingHorizontal: 20,
      paddingTop: 20,
      backgroundColor: theme.colors.background,
    },
    inputContainer: {
      marginBottom: 24,
      backgroundColor: theme.colors.surface,
      borderRadius: 12,
      padding: 16,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 3,
      elevation: 2,
    },
    label: {
      fontSize: 17,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 10,
      letterSpacing: -0.2,
    },
    input: {
      borderWidth: 2,
      borderColor: theme.colors.border,
      borderRadius: 10,
      paddingHorizontal: 16,
      paddingVertical: 14,
      fontSize: 16,
      backgroundColor: theme.colors.surface,
      color: theme.colors.text,
      minHeight: 48,
    },
    helpText: {
      fontSize: 13,
      color: theme.colors.textSecondary,
      marginTop: 8,
      lineHeight: 18,
    },
    loadingContainer: {
      alignItems: 'center',
      marginVertical: 24,
      backgroundColor: theme.colors.surface,
      borderRadius: 12,
      padding: 20,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 3,
      elevation: 2,
    },
    loadingText: {
      marginTop: 12,
      fontSize: 15,
      color: theme.colors.textSecondary,
      fontWeight: '500',
    },
    errorContainer: {
      backgroundColor: theme.dark ? '#2D1B1B' : '#FEF2F2',
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.error,
      borderRadius: 12,
      padding: 16,
      marginVertical: 16,
      shadowColor: theme.colors.error,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    errorText: {
      color: theme.colors.error,
      fontSize: 14,
      fontWeight: '600',
      lineHeight: 20,
    },
    infoContainer: {
      backgroundColor: theme.dark ? '#1E293B' : '#F0F9FF',
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.primary,
      borderRadius: 12,
      padding: 16,
      marginVertical: 16,
      shadowColor: theme.colors.primary,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    infoText: {
      color: theme.colors.primary,
      fontSize: 14,
      fontWeight: '600',
      lineHeight: 20,
    },
    sectionTitle: {
      fontSize: 18,
      fontWeight: '700',
      color: theme.colors.text,
      marginTop: 28,
      marginBottom: 16,
      letterSpacing: -0.2,
    },
    quantCard: {
      backgroundColor: theme.colors.surface,
      borderRadius: 12,
      padding: 16,
      marginBottom: 12,
      borderWidth: 2,
      borderColor: theme.colors.border,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.05,
      shadowRadius: 3,
      elevation: 2,
    },
    quantCardSelected: {
      borderColor: theme.colors.primary,
      backgroundColor: theme.dark ? '#1E293B' : '#EFF6FF',
      shadowColor: theme.colors.primary,
      shadowOpacity: 0.2,
      transform: [{ scale: 1.02 }],
    },
    quantName: {
      fontSize: 16,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: 4,
    },
    quantFileName: {
      fontSize: 13,
      color: theme.colors.textSecondary,
      fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
      backgroundColor: theme.colors.card,
      paddingHorizontal: 8,
      paddingVertical: 4,
      borderRadius: 6,
      overflow: 'hidden',
    },
    warningContainer: {
      backgroundColor: theme.dark ? '#2D2416' : '#FFFBEB',
      borderLeftWidth: 4,
      borderLeftColor: '#F59E0B',
      borderRadius: 12,
      padding: 16,
      marginVertical: 16,
      shadowColor: '#F59E0B',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    warningText: {
      color: theme.dark ? '#FBBF24' : '#D97706',
      fontSize: 14,
      fontWeight: '600',
      lineHeight: 20,
    },
    bottomPadding: {
      height: 40,
    },
    noteContainer: {
      backgroundColor: theme.dark ? '#1E293B' : '#F0F9FF',
      borderLeftWidth: 4,
      borderLeftColor: '#06B6D4',
      borderRadius: 12,
      padding: 16,
      marginBottom: 20,
      shadowColor: '#06B6D4',
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.1,
      shadowRadius: 2,
      elevation: 2,
    },
    noteText: {
      fontSize: 14,
      color: theme.dark ? '#CBD5E1' : '#155E75',
      lineHeight: 20,
    },
    noteTitle: {
      fontWeight: '700',
      color: theme.dark ? '#E2E8F0' : '#0E7490',
    },
  })

  const [modelId, setModelId] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [modelInfo, setModelInfo] = useState<CustomModelInfo | null>(null)
  const [selectedFile, setSelectedFile] = useState<{
    filename: string
    quantization: string
  } | null>(null)
  const [selectedMMProjFile, setSelectedMMProjFile] = useState<{
    filename: string
    quantization: string
  } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedModelFile, setSelectedModelFile] = useState<{
    uri: string
    name: string
    size?: number | null
  } | null>(null)
  const [selectedMmprojFile, setSelectedMmprojFile] = useState<{
    uri: string
    name: string
    size?: number | null
  } | null>(null)
  const [useFileSelection, setUseFileSelection] = useState(false)

  useEffect(() => {
    if (!visible) {
      // Reset state when modal closes
      setModelId('')
      setIsLoading(false)
      setModelInfo(null)
      setSelectedFile(null)
      setSelectedMMProjFile(null)
      setSelectedModelFile(null)
      setSelectedMmprojFile(null)
      setUseFileSelection(false)
      setError(null)
    }
  }, [visible])

  const handleFetchModel = async () => {
    if (!modelId.trim()) {
      setError('Please enter a model ID')
      return
    }

    setIsLoading(true)
    setError(null)
    setModelInfo(null)
    setSelectedFile(null)
    setSelectedMMProjFile(null)

    try {
      const info = await HuggingFaceAPI.fetchModelInfo(modelId.trim())
      setModelInfo(info)

      if (!info.exists) {
        setError(info.error || 'Model not found')
        return
      }

      if (info.files.length === 0) {
        setError('No GGUF files found in this model')
        return
      }

      // Auto-select default quantization
      const defaultFile = HuggingFaceAPI.getDefaultQuantization(info.files)
      if (defaultFile) {
        setSelectedFile(defaultFile)
      }

      // Auto-select default mmproj if available
      if (info.mmprojFiles.length > 0) {
        const defaultMMProj = HuggingFaceAPI.getDefaultMmproj(info.mmprojFiles)
        if (defaultMMProj) {
          setSelectedMMProjFile(defaultMMProj)
        }
      } else if (requireMMProj) {
        setError(
          'This model does not have mmproj files required for multimodal functionality',
        )
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handlePickModelFile = async () => {
    try {
      const supportedTypes = Platform.OS === 'ios'
        ? ['public.data'] // iOS: Allow all file types
        : ['*/*'] // Android: Allow all file types

      const [file] = await pick({
        type: supportedTypes,
      })

      if (file?.uri && file?.name) {
        // Check if it's a GGUF file
        if (!file.name.toLowerCase().endsWith('.gguf')) {
          Alert.alert('Invalid File', 'Please select a GGUF model file (.gguf extension)')
          return
        }

        // Keep a local copy of the file
        const [localCopy] = await keepLocalCopy({
          files: [
            {
              uri: file.uri,
              fileName: file.name,
            },
          ],
          destination: 'documentDirectory',
        })

        if (localCopy.status === 'success') {
          setSelectedModelFile({
            uri: localCopy.localUri,
            name: file.name,
            size: file.size ?? null,
          })
        } else {
          setError(`Failed to copy model file: ${localCopy.copyError}`)
          return
        }
        setError(null)
      }
    } catch (err: any) {
      if (!err.message.includes('user canceled')) {
        setError(`Failed to pick model file: ${err.message}`)
      }
    }
  }

  const handlePickMmprojFile = async () => {
    try {
      const supportedTypes = Platform.OS === 'ios'
        ? ['public.data'] // iOS: Allow all file types
        : ['*/*'] // Android: Allow all file types

      const [file] = await pick({
        type: supportedTypes,
      })

      if (file?.uri && file?.name) {
        // Check if it's a GGUF file
        if (!file.name.toLowerCase().endsWith('.gguf')) {
          Alert.alert('Invalid File', 'Please select a GGUF mmproj file (.gguf extension)')
          return
        }

        // Keep a local copy of the file
        const [localCopy] = await keepLocalCopy({
          files: [
            {
              uri: file.uri,
              fileName: file.name,
            },
          ],
          destination: 'documentDirectory',
        })

        if (localCopy.status === 'success') {
          setSelectedMmprojFile({
            uri: localCopy.localUri,
            name: file.name,
            size: file.size ?? null,
          })
        } else {
          setError(`Failed to copy mmproj file: ${localCopy.copyError}`)
          return
        }
        setError(null)
      }
    } catch (err: any) {
      if (!err.message.includes('user canceled')) {
        setError(`Failed to pick mmproj file: ${err.message}`)
      }
    }
  }

  const handleSave = async () => {
    if (useFileSelection) {
      // File selection mode
      if (!selectedModelFile) {
        setError('Please select a model file')
        return
      }

      if (requireMMProj && !selectedMmprojFile) {
        setError('Please select an mmproj file for multimodal functionality')
        return
      }

      try {
        const customModel: CustomModel = {
          id: selectedModelFile.name.replace('.gguf', ''),
          repo: 'local-file',
          filename: selectedModelFile.name,
          quantization: 'Unknown',
          mmprojFilename: selectedMmprojFile?.name,
          mmprojQuantization: selectedMmprojFile ? 'Unknown' : undefined,
          addedAt: Date.now(),
          localPath: selectedModelFile.uri,
          mmprojLocalPath: selectedMmprojFile?.uri,
        }

        await saveCustomModel(customModel)
        Alert.alert('Success', 'Custom model added successfully!')

        if (onModelAdded) {
          onModelAdded(customModel)
        }

        onClose()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save model')
      }
    } else {
      // HuggingFace download mode
      if (!modelInfo || !selectedFile) {
        setError('Please select a model file')
        return
      }

      if (requireMMProj && !selectedMMProjFile) {
        setError('Please select an mmproj file for multimodal functionality')
        return
      }

      try {
        const customModel: CustomModel = {
          id: modelInfo.id,
          repo: modelInfo.id,
          filename: selectedFile.filename,
          quantization: selectedFile.quantization,
          mmprojFilename: selectedMMProjFile?.filename,
          mmprojQuantization: selectedMMProjFile?.quantization,
          addedAt: Date.now(),
        }

        await saveCustomModel(customModel)
        Alert.alert('Success', 'Custom model added successfully!')

        if (onModelAdded) {
          onModelAdded(customModel)
        }

        onClose()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save model')
      }
    }
  }

  const canSave = useFileSelection
    ? selectedModelFile && (!requireMMProj || selectedMmprojFile)
    : modelInfo?.exists && selectedFile && (!requireMMProj || selectedMMProjFile)

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose}>
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.title}>{title}</Text>
          <TouchableOpacity onPress={handleSave} disabled={!canSave}>
            <Text
              style={[styles.saveButton, !canSave && styles.disabledButton]}
            >
              Add
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {enableFileSelection && (
            <View style={styles.inputContainer}>
              <Text style={styles.label}>Add Model From</Text>
              <View style={{ flexDirection: 'row', marginBottom: 16 }}>
                <TouchableOpacity
                  style={[
                    themedStyles.primaryButton,
                    { flex: 1, marginRight: 8 },
                    !useFileSelection && themedStyles.primaryButtonActive,
                  ]}
                  onPress={() => {
                    setUseFileSelection(false)
                    setError(null)
                  }}
                >
                  <Text style={themedStyles.primaryButtonText}>
                    HuggingFace
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[
                    themedStyles.primaryButton,
                    { flex: 1, marginLeft: 8 },
                    useFileSelection && themedStyles.primaryButtonActive,
                  ]}
                  onPress={() => {
                    setUseFileSelection(true)
                    setError(null)
                  }}
                >
                  <Text style={themedStyles.primaryButtonText}>
                    Select File
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

          {useFileSelection ? (
            <>
              {/* File Selection Mode */}
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Model File</Text>
                <TouchableOpacity
                  style={[
                    styles.input,
                    {
                      justifyContent: 'center',
                      backgroundColor: selectedModelFile ? theme.colors.card : theme.colors.surface,
                      borderColor: selectedModelFile ? theme.colors.primary : theme.colors.border,
                    },
                  ]}
                  onPress={handlePickModelFile}
                >
                  <Text
                    style={{
                      color: selectedModelFile ? theme.colors.primary : theme.colors.textSecondary,
                      fontSize: 16,
                    }}
                  >
                    {selectedModelFile
                      ? selectedModelFile.name
                      : 'Tap to select model file (.gguf)'}
                  </Text>
                </TouchableOpacity>
                <Text style={styles.helpText}>
                  Select a GGUF model file from your device
                </Text>
              </View>

              {requireMMProj && (
                <View style={styles.inputContainer}>
                  <Text style={styles.label}>MMProj File (Required)</Text>
                  <TouchableOpacity
                    style={[
                      styles.input,
                      {
                        justifyContent: 'center',
                        backgroundColor: selectedMmprojFile ? theme.colors.card : theme.colors.surface,
                        borderColor: selectedMmprojFile ? theme.colors.primary : theme.colors.border,
                      },
                    ]}
                    onPress={handlePickMmprojFile}
                  >
                    <Text
                      style={{
                        color: selectedMmprojFile ? theme.colors.primary : theme.colors.textSecondary,
                        fontSize: 16,
                      }}
                    >
                      {selectedMmprojFile
                        ? selectedMmprojFile.name
                        : 'Tap to select mmproj file (.gguf)'}
                    </Text>
                  </TouchableOpacity>
                  <Text style={styles.helpText}>
                    Select an mmproj GGUF file for multimodal support
                  </Text>
                </View>
              )}
            </>
          ) : (
            <>
              {/* HuggingFace Download Mode */}
              <View style={styles.inputContainer}>
                <Text style={styles.label}>Model ID</Text>
                <TextInput
                  style={styles.input}
                  value={modelId}
                  onChangeText={setModelId}
                  placeholder="e.g., microsoft/DialoGPT-medium"
                  placeholderTextColor={theme.colors.textSecondary}
                  autoCapitalize="none"
                  autoCorrect={false}
                  onSubmitEditing={handleFetchModel}
                />
                <Text style={styles.helpText}>
                  Enter the HuggingFace model ID (format: username/model-name)
                </Text>
              </View>

              <View style={styles.noteContainer}>
                <Text style={styles.noteText}>
                  <Text style={styles.noteTitle}>Note:</Text>
                  {' '}
                  Some models may
                  require granting access or accepting license agreements on
                  HuggingFace before they can be downloaded. If download fails with
                  access errors, visit the model page on huggingface.co to accept
                  the license.
                </Text>
              </View>

              <TouchableOpacity
                style={themedStyles.primaryButton}
                onPress={handleFetchModel}
                disabled={isLoading}
              >
                <Text style={themedStyles.primaryButtonText}>
                  {isLoading ? 'Fetching...' : 'Check Model'}
                </Text>
              </TouchableOpacity>
            </>
          )}

          {isLoading && !useFileSelection && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color={theme.colors.primary} />
              <Text style={styles.loadingText}>
                Fetching model information...
              </Text>
            </View>
          )}

          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>
                Error:
                {error}
              </Text>
            </View>
          )}

          {modelInfo?.exists && !useFileSelection && (
            <View style={styles.infoContainer}>
              <Text style={styles.infoText}>
                Model found! Select quantization below.
              </Text>
            </View>
          )}

          {requireMMProj &&
            modelInfo?.exists &&
            modelInfo.mmprojFiles.length === 0 &&
            !useFileSelection && (
              <View style={styles.warningContainer}>
                <Text style={styles.warningText}>
                  Warning: This model doesn&apos;t have mmproj files. Multimodal
                  functionality will not be available.
                </Text>
              </View>
            )}

          {modelInfo?.exists && modelInfo.files.length > 0 && !useFileSelection && (
            <>
              <Text style={styles.sectionTitle}>Select Model Quantization</Text>
              {modelInfo.files.map((file) => (
                <TouchableOpacity
                  key={file.filename}
                  style={[
                    styles.quantCard,
                    selectedFile?.filename === file.filename &&
                      styles.quantCardSelected,
                  ]}
                  onPress={() => setSelectedFile(file)}
                >
                  <Text style={styles.quantName}>{file.quantization}</Text>
                  <Text style={styles.quantFileName}>{file.filename}</Text>
                </TouchableOpacity>
              ))}
            </>
          )}

          {modelInfo?.exists && modelInfo.mmprojFiles.length > 0 && !useFileSelection && (
            <>
              <Text style={styles.sectionTitle}>
                Select MMProj File
                {' '}
                {requireMMProj ? '(Required)' : '(Optional)'}
              </Text>
              {modelInfo.mmprojFiles.map((file) => (
                <TouchableOpacity
                  key={file.filename}
                  style={[
                    styles.quantCard,
                    selectedMMProjFile?.filename === file.filename &&
                      styles.quantCardSelected,
                  ]}
                  onPress={() => setSelectedMMProjFile(file)}
                >
                  <Text style={styles.quantName}>{file.quantization}</Text>
                  <Text style={styles.quantFileName}>{file.filename}</Text>
                </TouchableOpacity>
              ))}
            </>
          )}

          <View style={styles.bottomPadding} />
        </ScrollView>
      </View>
    </Modal>
  )
}
