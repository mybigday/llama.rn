import React, { useState, useEffect } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  TextInput,
  Switch,
  Alert,
} from 'react-native'
import type { ContextParams } from '../utils/storage'
import {
  saveContextParams,
  loadContextParams,
  resetContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '../utils/storage'

interface ContextParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: ContextParams) => void
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: 'white',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  cancelButton: {
    fontSize: 16,
    color: '#007AFF',
  },
  saveButton: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
  disabledButton: {
    opacity: 0.5,
  },
  content: {
    flex: 1,
    paddingHorizontal: 16,
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
    marginVertical: 16,
    textAlign: 'center',
  },
  paramGroup: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  paramLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 4,
  },
  paramDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 18,
    marginBottom: 12,
  },
  textInput: {
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
    backgroundColor: '#F8F8F8',
  },
  switchRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  switchInfo: {
    flex: 1,
    marginRight: 12,
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
  warningContainer: {
    backgroundColor: '#FFFBEB',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    alignItems: 'center',
  },
  warningText: {
    color: '#D97706',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
})

export default function ContextParamsModal({
  visible,
  onClose,
  onSave,
}: ContextParamsModalProps) {
  const [params, setParams] = useState<ContextParams>(DEFAULT_CONTEXT_PARAMS)
  const [isLoading, setIsLoading] = useState(false)

  const loadParams = async () => {
    try {
      const savedParams = await loadContextParams()
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
      await saveContextParams(params)
      onSave?.(params)
      Alert.alert('Success', 'Context parameters saved successfully!')
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
              await resetContextParams()
              setParams(DEFAULT_CONTEXT_PARAMS)
              Alert.alert('Success', 'Parameters reset to defaults!')
            } catch (error: any) {
              Alert.alert('Error', `Failed to reset: ${error.message}`)
            }
          },
        },
      ],
    )
  }

  const updateParam = (key: keyof ContextParams, value: any) => {
    setParams((prev) => ({ ...prev, [key]: value }))
  }

  const validateNumber = (text: string, min = 0, max = Infinity) => {
    const num = parseInt(text, 10)
    return !Number.isNaN(num) && num >= min && num <= max ? num : undefined
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
          <Text style={styles.title}>Context Parameters</Text>
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
            Configure context initialization parameters. These settings affect
            memory usage, performance, and model behavior during loading.
          </Text>

          <View style={styles.warningContainer}>
            <Text style={styles.warningText}>
              Warning: Changing context parameters requires reinitializing the
              model, which will clear your current conversation.
            </Text>
          </View>

          {/* Context Size */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Context Size (n_ctx)</Text>
            <Text style={styles.paramDescription}>
              Maximum context length in tokens. Higher values use more memory.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_ctx.toString()}
              onChangeText={(text) => {
                const value = validateNumber(text, 512, 32768)
                if (value !== undefined) updateParam('n_ctx', value)
              }}
              keyboardType="numeric"
              placeholder="4096"
            />
          </View>

          {/* GPU Layers */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>GPU Layers (n_gpu_layers)</Text>
            <Text style={styles.paramDescription}>
              Number of layers to run on GPU. Use 99 for all layers, 0 for CPU
              only.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_gpu_layers.toString()}
              onChangeText={(text) => {
                const value = validateNumber(text, 0, 99)
                if (value !== undefined) updateParam('n_gpu_layers', value)
              }}
              keyboardType="numeric"
              placeholder="99"
            />
          </View>

          {/* Memory Lock */}
          <View style={styles.paramGroup}>
            <View style={styles.switchRow}>
              <View style={styles.switchInfo}>
                <Text style={styles.paramLabel}>Memory Lock (use_mlock)</Text>
                <Text style={styles.paramDescription}>
                  Lock model in memory to prevent swapping to disk.
                </Text>
              </View>
              <Switch
                value={params.use_mlock}
                onValueChange={(value) => updateParam('use_mlock', value)}
                trackColor={{ false: '#E0E0E0', true: '#007AFF' }}
                thumbColor={params.use_mlock ? '#FFFFFF' : '#FFFFFF'}
              />
            </View>
          </View>

          {/* Memory Map */}
          <View style={styles.paramGroup}>
            <View style={styles.switchRow}>
              <View style={styles.switchInfo}>
                <Text style={styles.paramLabel}>Memory Map (use_mmap)</Text>
                <Text style={styles.paramDescription}>
                  Use memory mapping for better performance.
                </Text>
              </View>
              <Switch
                value={params.use_mmap}
                onValueChange={(value) => updateParam('use_mmap', value)}
                trackColor={{ false: '#E0E0E0', true: '#007AFF' }}
                thumbColor={params.use_mmap ? '#FFFFFF' : '#FFFFFF'}
              />
            </View>
          </View>

          {/* Batch Size */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Batch Size (n_batch)</Text>
            <Text style={styles.paramDescription}>
              Number of tokens to process in parallel. Higher values use more
              memory.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_batch?.toString() || '512'}
              onChangeText={(text) => {
                const value = validateNumber(text, 1, 2048)
                if (value !== undefined) updateParam('n_batch', value)
              }}
              keyboardType="numeric"
              placeholder="512"
            />
          </View>

          {/* Micro Batch Size */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Micro Batch Size (n_ubatch)</Text>
            <Text style={styles.paramDescription}>
              Internal batch size for processing. Should be ≤ n_batch.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_ubatch?.toString() || '512'}
              onChangeText={(text) => {
                const value = validateNumber(text, 1, 2048)
                if (value !== undefined) updateParam('n_ubatch', value)
              }}
              keyboardType="numeric"
              placeholder="512"
            />
          </View>

          {/* Threads */}
          <View style={styles.paramGroup}>
            <Text style={styles.paramLabel}>Threads (n_threads)</Text>
            <Text style={styles.paramDescription}>
              Number of CPU threads to use. Usually set to number of CPU cores.
            </Text>
            <TextInput
              style={styles.textInput}
              value={params.n_threads?.toString()}
              onChangeText={(text) => {
                const value = validateNumber(text, 1, 32)
                if (value !== undefined) updateParam('n_threads', value)
              }}
              keyboardType="numeric"
              placeholder="8"
            />
          </View>

          {/* Context Shift */}
          <View style={styles.paramGroup}>
            <View style={styles.switchRow}>
              <View style={styles.switchInfo}>
                <Text style={styles.paramLabel}>Context Shift (ctx_shift)</Text>
                <Text style={styles.paramDescription}>
                  Enable automatic context shifting when context is full.
                </Text>
              </View>
              <Switch
                value={params.ctx_shift || false}
                onValueChange={(value) => updateParam('ctx_shift', value)}
                trackColor={{ false: '#E0E0E0', true: '#007AFF' }}
                thumbColor={params.ctx_shift ? '#FFFFFF' : '#FFFFFF'}
              />
            </View>
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
