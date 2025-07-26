import React, { useState, useEffect } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  TextInput,
  Alert,
} from 'react-native'
import { CommonStyles } from '../styles/commonStyles'
import type { CompletionParams } from '../utils/storage'
import {
  saveCompletionParams,
  loadCompletionParams,
  resetCompletionParams,
  DEFAULT_COMPLETION_PARAMS,
} from '../utils/storage'

interface CompletionParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: CompletionParams) => void
}

const styles = StyleSheet.create({
  // Using shared styles for common patterns
  container: CommonStyles.container,
  header: CommonStyles.modalHeader,
  title: CommonStyles.modalTitle,
  cancelButton: CommonStyles.headerButtonText,
  saveButton: {
    ...CommonStyles.headerButtonText,
    fontWeight: '600',
  },
  disabledButton: CommonStyles.disabledButton,
  content: {
    flex: 1,
    paddingHorizontal: 16,
  },
  description: CommonStyles.description,
  paramGroup: CommonStyles.paramGroup,
  paramLabel: CommonStyles.paramLabel,
  paramDescription: CommonStyles.paramDescription,
  textInput: CommonStyles.textInput,
  stopSequenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  stopSequenceInput: {
    flex: 1,
    marginRight: 8,
  },
  removeButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  removeButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  addButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    borderRadius: 8,
    marginTop: 8,
  },
  addButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
  resetButton: {
    backgroundColor: '#FF3B30',
    borderRadius: 8,
    paddingVertical: 12,
    marginTop: 20,
    marginBottom: 20,
  },
  resetButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  bottomPadding: {
    height: 30,
  },
})

export default function CompletionParamsModal({
  visible,
  onClose,
  onSave,
}: CompletionParamsModalProps) {
  const [params, setParams] = useState<CompletionParams>(
    DEFAULT_COMPLETION_PARAMS,
  )
  const [isLoading, setIsLoading] = useState(false)

  const loadParams = async () => {
    try {
      const savedParams = await loadCompletionParams()
      setParams(savedParams)
    } catch (error) {
      console.error('Error loading params:', error)
    }
  }

  useEffect(() => {
    if (visible) {
      loadParams()
    }
  }, [visible])

  const handleSave = async () => {
    try {
      setIsLoading(true)
      await saveCompletionParams(params)
      onSave(params)
      Alert.alert('Success', 'Completion parameters saved successfully!')
      onClose()
    } catch (error: any) {
      Alert.alert('Error', `Failed to save parameters: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = async () => {
    Alert.alert(
      'Reset Parameters',
      'Are you sure you want to reset to default values?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              await resetCompletionParams()
              setParams(DEFAULT_COMPLETION_PARAMS)
              Alert.alert('Success', 'Parameters reset to defaults!')
            } catch (error: any) {
              Alert.alert('Error', `Failed to reset: ${error.message}`)
            }
          },
        },
      ],
    )
  }

  const updateParam = (key: keyof CompletionParams, value: any) => {
    setParams((prev) => ({ ...prev, [key]: value }))
  }

  const validateNumber = (text: string, min = 0, max = Infinity) => {
    const num = parseFloat(text)
    return !Number.isNaN(num) && num >= min && num <= max ? num : undefined
  }

  const validateInteger = (text: string, min = 0, max = Infinity) => {
    const num = parseInt(text, 10)
    return !Number.isNaN(num) && num >= min && num <= max ? num : undefined
  }

  const addStopSequence = () => {
    const newStop = [...(params.stop || []), '']
    updateParam('stop', newStop)
  }

  const removeStopSequence = (index: number) => {
    const newStop = (params.stop || []).filter((_, i) => i !== index)
    updateParam('stop', newStop)
  }

  const updateStopSequence = (index: number, value: string) => {
    const newStop = [...(params.stop || [])]
    newStop[index] = value
    updateParam('stop', newStop)
  }

  return (
    <Modal
      visible={visible}
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose}>
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Completion Parameters</Text>
          <TouchableOpacity onPress={handleSave} disabled={isLoading}>
            <Text
              style={[styles.saveButton, isLoading && styles.disabledButton]}
            >
              Save
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          <Text style={styles.description}>
            Configure completion and sampling parameters. These settings control
            how the AI generates responses during conversations.
          </Text>

          {/* Max Tokens */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Max Tokens (n_predict)</Text>
            <Text style={styles.paramDescription}>
              Maximum number of tokens to generate in response. Higher values
              allow longer responses.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_predict?.toString()}
              onChangeText={(text) => {
                const value = validateInteger(text, 1, 4096)
                if (value !== undefined) updateParam('n_predict', value)
              }}
              keyboardType="numeric"
              placeholder="512"
            />
          </View>

          {/* Temperature */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Temperature</Text>
            <Text style={styles.paramDescription}>
              Controls randomness in responses. Lower values (0.1-0.3) are more
              focused and deterministic, higher values (0.7-1.0) are more
              creative and varied.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.temperature?.toString()}
              onChangeText={(text) => {
                const value = validateNumber(text, 0.0, 2.0)
                if (value !== undefined) updateParam('temperature', value)
              }}
              keyboardType="decimal-pad"
              placeholder="0.7"
            />
          </View>

          {/* Top-p */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Top-p (Nucleus Sampling)</Text>
            <Text style={styles.paramDescription}>
              Controls diversity by considering only tokens with cumulative
              probability up to p. Lower values (0.1-0.5) are more focused,
              higher values (0.8-0.95) are more diverse.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.top_p?.toString()}
              onChangeText={(text) => {
                const value = validateNumber(text, 0.0, 1.0)
                if (value !== undefined) updateParam('top_p', value)
              }}
              keyboardType="decimal-pad"
              placeholder="0.9"
            />
          </View>

          {/* Stop Sequences */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Stop Sequences</Text>
            <Text style={styles.paramDescription}>
              Text sequences that will stop generation when encountered. Common
              examples: &lt;|im_end|&gt;, &lt;end_of_turn&gt;, User:
            </Text>

            {(params.stop || []).map((stopSeq, index) => (
              <View key={index} style={styles.stopSequenceContainer}>
                <TextInput
                  style={[styles.textInput, styles.stopSequenceInput]}
                  value={stopSeq}
                  onChangeText={(text) => updateStopSequence(index, text)}
                  placeholder="Enter stop sequence"
                />
                <TouchableOpacity
                  style={styles.removeButton}
                  onPress={() => removeStopSequence(index)}
                >
                  <Text style={styles.removeButtonText}>Remove</Text>
                </TouchableOpacity>
              </View>
            ))}

            <TouchableOpacity
              style={styles.addButton}
              onPress={addStopSequence}
            >
              <Text style={styles.addButtonText}>Add Stop Sequence</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
            <Text style={styles.resetButtonText}>Reset to Defaults</Text>
          </TouchableOpacity>

          <View style={styles.bottomPadding} />
        </ScrollView>
      </View>
    </Modal>
  )
}
