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
} from 'react-native'
import {
  HuggingFaceAPI,
  type CustomModelInfo,
} from '../services/HuggingFaceAPI'
import { saveCustomModel, type CustomModel } from '../utils/storage'
import { CommonStyles } from '../styles/commonStyles'

interface CustomModelModalProps {
  visible: boolean
  onClose: () => void
  onModelAdded?: (model: CustomModel) => void
  requireMMProj?: boolean // For multimodal screens
  title?: string
}

const styles = StyleSheet.create({
  container: {
    ...CommonStyles.container,
    backgroundColor: '#f8fafc',
  },
  header: {
    ...CommonStyles.modalHeader,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#ffffff',
    paddingHorizontal: 24,
    paddingVertical: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  title: {
    ...CommonStyles.modalTitle,
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
  },
  cancelButton: {
    ...CommonStyles.headerButtonText,
    color: '#64748b',
    fontSize: 16,
    fontWeight: '500',
  },
  saveButton: {
    ...CommonStyles.headerButtonText,
    color: '#2563eb',
    fontWeight: '700',
    fontSize: 16,
  },
  disabledButton: {
    ...CommonStyles.disabledButton,
    color: '#cbd5e1',
  },
  content: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 20,
    backgroundColor: '#f8fafc',
  },
  inputContainer: {
    marginBottom: 24,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  label: {
    fontSize: 17,
    fontWeight: '600',
    color: '#1e293b',
    marginBottom: 10,
    letterSpacing: -0.2,
  },
  input: {
    borderWidth: 2,
    borderColor: '#e2e8f0',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    backgroundColor: '#ffffff',
    color: '#1e293b',
    minHeight: 48,
  },
  helpText: {
    fontSize: 13,
    color: '#64748b',
    marginTop: 8,
    lineHeight: 18,
  },
  loadingContainer: {
    alignItems: 'center',
    marginVertical: 24,
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 15,
    color: '#64748b',
    fontWeight: '500',
  },
  errorContainer: {
    backgroundColor: '#fef2f2',
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
    borderRadius: 12,
    padding: 16,
    marginVertical: 16,
    shadowColor: '#ef4444',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  errorText: {
    color: '#dc2626',
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
  },
  infoContainer: {
    backgroundColor: '#f0f9ff',
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
    borderRadius: 12,
    padding: 16,
    marginVertical: 16,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  infoText: {
    color: '#1e40af',
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1e293b',
    marginTop: 28,
    marginBottom: 16,
    letterSpacing: -0.2,
  },
  quantCard: {
    backgroundColor: '#ffffff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    borderColor: '#e2e8f0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  quantCardSelected: {
    borderColor: '#2563eb',
    backgroundColor: '#eff6ff',
    shadowColor: '#2563eb',
    shadowOpacity: 0.2,
    transform: [{ scale: 1.02 }],
  },
  quantName: {
    fontSize: 16,
    fontWeight: '700',
    color: '#1e293b',
    marginBottom: 4,
  },
  quantFileName: {
    fontSize: 13,
    color: '#64748b',
    fontFamily: 'Menlo, Monaco, monospace',
    backgroundColor: '#f1f5f9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    overflow: 'hidden',
  },
  warningContainer: {
    backgroundColor: '#fffbeb',
    borderLeftWidth: 4,
    borderLeftColor: '#f59e0b',
    borderRadius: 12,
    padding: 16,
    marginVertical: 16,
    shadowColor: '#f59e0b',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  warningText: {
    color: '#d97706',
    fontSize: 14,
    fontWeight: '600',
    lineHeight: 20,
  },
  bottomPadding: {
    height: 40,
  },
  noteContainer: {
    backgroundColor: '#f0f9ff',
    borderLeftWidth: 4,
    borderLeftColor: '#06b6d4',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#06b6d4',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  noteText: {
    fontSize: 14,
    color: '#155e75',
    lineHeight: 20,
  },
  noteTitle: {
    fontWeight: '700',
    color: '#0e7490',
  },
})

export default function CustomModelModal({
  visible,
  onClose,
  onModelAdded,
  requireMMProj = false,
  title = 'Add Custom Model',
}: CustomModelModalProps) {
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

  useEffect(() => {
    if (!visible) {
      // Reset state when modal closes
      setModelId('')
      setIsLoading(false)
      setModelInfo(null)
      setSelectedFile(null)
      setSelectedMMProjFile(null)
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

  const handleSave = async () => {
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

  const canSave =
    modelInfo?.exists && selectedFile && (!requireMMProj || selectedMMProjFile)

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
          <View style={styles.inputContainer}>
            <Text style={styles.label}>Model ID</Text>
            <TextInput
              style={styles.input}
              value={modelId}
              onChangeText={setModelId}
              placeholder="e.g., microsoft/DialoGPT-medium"
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
            style={CommonStyles.primaryButton}
            onPress={handleFetchModel}
            disabled={isLoading}
          >
            <Text style={CommonStyles.primaryButtonText}>
              {isLoading ? 'Fetching...' : 'Check Model'}
            </Text>
          </TouchableOpacity>

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
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

          {modelInfo?.exists && (
            <View style={styles.infoContainer}>
              <Text style={styles.infoText}>
                Model found! Select quantization below.
              </Text>
            </View>
          )}

          {requireMMProj &&
            modelInfo?.exists &&
            modelInfo.mmprojFiles.length === 0 && (
              <View style={styles.warningContainer}>
                <Text style={styles.warningText}>
                  Warning: This model doesn&apos;t have mmproj files. Multimodal
                  functionality will not be available.
                </Text>
              </View>
            )}

          {modelInfo?.exists && modelInfo.files.length > 0 && (
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

          {modelInfo?.exists && modelInfo.mmprojFiles.length > 0 && (
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
