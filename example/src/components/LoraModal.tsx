import React, { useCallback, useEffect, useState } from 'react'
import {
  ActivityIndicator,
  Alert,
  Modal,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native'
import { keepLocalCopy, pick } from '@react-native-documents/picker'
import type { LlamaContext } from '../../../src'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

type LoraAdapter = {
  path: string
  scaled?: number
}

interface LoraModalProps {
  visible: boolean
  onClose: () => void
  context: LlamaContext | null
}

const stripFilePrefix = (path: string) => path.replace(/^file:\/\//, '')

const getFileName = (path: string) => {
  const parts = path.split(/[/\\]/)
  return parts[parts.length - 1] || path
}

const formatScale = (scaled?: number) => {
  const scale = scaled ?? 1
  return Number.isInteger(scale) ? `${scale}` : scale.toFixed(2)
}

const scaleToInputValue = (scaled?: number) => `${scaled ?? 1}`

const parseScaleInput = (value: string) => {
  const normalizedValue = value.trim()
  if (!normalizedValue) {
    return null
  }

  const scale = Number(normalizedValue)
  return Number.isFinite(scale) ? scale : null
}

const getErrorMessage = (error: unknown) => {
  if (error instanceof Error) {
    return error.message
  }

  if (typeof error === 'string') {
    return error
  }

  if (
    error &&
    typeof error === 'object' &&
    'message' in error &&
    typeof error.message === 'string'
  ) {
    return error.message
  }

  return 'Unknown error'
}

const dedupeAdapters = (adapters: LoraAdapter[]) => {
  const normalized = new Map<string, LoraAdapter>()

  adapters.forEach((adapter) => {
    const path = stripFilePrefix(adapter.path)
    if (!path) return
    normalized.set(path, {
      path,
      scaled: adapter.scaled,
    })
  })

  return Array.from(normalized.values())
}

export default function LoraModal({
  visible,
  onClose,
  context,
}: LoraModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const [adapters, setAdapters] = useState<LoraAdapter[]>([])
  const [adapterScaleInputs, setAdapterScaleInputs] = useState<
    Record<string, string>
  >({})
  const [isBusy, setIsBusy] = useState(false)
  const [newAdapterScale, setNewAdapterScale] = useState('1')

  const styles = StyleSheet.create({
    container: themedStyles.container,
    header: {
      ...themedStyles.modalHeader,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      paddingHorizontal: 20,
      paddingVertical: 16,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
      ...Platform.select({
        ios: {
          shadowColor: theme.colors.shadow,
          shadowOffset: { width: 0, height: 1 },
          shadowOpacity: theme.dark ? 0.3 : 0.1,
          shadowRadius: 2,
        },
        android: {
          elevation: 2,
        },
      }),
    },
    title: {
      ...themedStyles.modalTitle,
      color: theme.colors.text,
      fontWeight: '700',
    },
    headerButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.primary,
      fontWeight: '600',
      minWidth: 60,
    },
    disabledHeaderButton: {
      color: theme.colors.textSecondary,
    },
    content: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    contentContainer: {
      paddingHorizontal: 16,
      paddingVertical: 20,
      paddingBottom: 32,
    },
    description: {
      ...themedStyles.description,
      color: theme.colors.textSecondary,
      textAlign: 'left',
      marginTop: 0,
    },
    inputSection: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 12,
      padding: 14,
      marginBottom: 16,
    },
    inputLabel: {
      color: theme.colors.text,
      fontSize: 14,
      fontWeight: '700',
      marginBottom: 8,
    },
    scaleInput: {
      ...themedStyles.textInput,
      paddingVertical: 12,
    },
    helperText: {
      color: theme.colors.textSecondary,
      fontSize: 12,
      lineHeight: 18,
      marginTop: 8,
    },
    buttonRow: {
      flexDirection: 'row',
      gap: 12,
      marginBottom: 20,
    },
    button: {
      flex: 1,
      paddingVertical: 14,
      paddingHorizontal: 16,
      borderRadius: 10,
      alignItems: 'center',
      justifyContent: 'center',
    },
    addButton: {
      backgroundColor: theme.colors.primary,
    },
    removeAllButton: {
      backgroundColor: theme.colors.error,
    },
    disabledButton: {
      backgroundColor: theme.colors.border,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: 15,
      fontWeight: '700',
    },
    sectionTitle: {
      fontSize: 16,
      fontWeight: '700',
      color: theme.colors.text,
      marginBottom: 12,
    },
    emptyState: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 12,
      padding: 16,
    },
    emptyStateText: {
      color: theme.colors.textSecondary,
      fontSize: 14,
      lineHeight: 20,
    },
    adapterCard: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 12,
      padding: 14,
      marginBottom: 12,
    },
    adapterHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      gap: 12,
    },
    adapterInfo: {
      flex: 1,
    },
    adapterTitle: {
      color: theme.colors.text,
      fontSize: 15,
      fontWeight: '700',
      marginBottom: 4,
    },
    adapterScale: {
      color: theme.colors.textSecondary,
      fontSize: 13,
      marginBottom: 10,
    },
    adapterPath: {
      color: theme.colors.textSecondary,
      fontSize: 12,
      lineHeight: 18,
    },
    adapterControls: {
      marginTop: 12,
      flexDirection: 'row',
      alignItems: 'center',
      gap: 10,
    },
    adapterScaleInput: {
      ...themedStyles.textInput,
      flex: 1,
      paddingVertical: 10,
      fontSize: 14,
    },
    updateButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      paddingHorizontal: 12,
      paddingVertical: 10,
      alignItems: 'center',
      justifyContent: 'center',
    },
    updateButtonText: {
      color: theme.colors.white,
      fontSize: 13,
      fontWeight: '700',
    },
    removeButton: {
      backgroundColor: theme.colors.error,
      borderRadius: 8,
      paddingHorizontal: 12,
      paddingVertical: 8,
    },
    removeButtonText: {
      color: theme.colors.white,
      fontSize: 13,
      fontWeight: '700',
    },
    loadingContainer: {
      paddingVertical: 24,
      alignItems: 'center',
      justifyContent: 'center',
    },
    loadingText: {
      color: theme.colors.textSecondary,
      fontSize: 14,
      marginTop: 10,
    },
  })

  const syncLoadedAdapters = useCallback((loadedAdapters: LoraAdapter[]) => {
    setAdapters(loadedAdapters)
    setAdapterScaleInputs(
      Object.fromEntries(
        loadedAdapters.map((adapter) => [
          adapter.path,
          scaleToInputValue(adapter.scaled),
        ]),
      ),
    )
  }, [])

  const refreshAdapters = useCallback(
    async (showError = true) => {
      if (!context) {
        syncLoadedAdapters([])
        return
      }

      setIsBusy(true)
      try {
        const loadedAdapters = await context.getLoadedLoraAdapters()
        syncLoadedAdapters(loadedAdapters)
      } catch (error: unknown) {
        const errorMessage = getErrorMessage(error)
        if (showError) {
          Alert.alert(
            'Error',
            `Failed to load current LoRA adapters: ${errorMessage}`,
          )
        }
      } finally {
        setIsBusy(false)
      }
    },
    [context, syncLoadedAdapters],
  )

  useEffect(() => {
    if (!visible) return
    void refreshAdapters(false)
  }, [refreshAdapters, visible])

  const handleAddAdapter = async () => {
    if (!context) {
      Alert.alert('Error', 'No active context to apply a LoRA adapter to')
      return
    }

    const scale = parseScaleInput(newAdapterScale)
    if (scale === null) {
      Alert.alert('Invalid Scale', 'Enter a valid numeric LoRA scale value.')
      return
    }

    try {
      setIsBusy(true)

      const [file] = await pick({
        type: ['*/*'],
        allowMultiSelection: false,
      })

      if (!file?.uri) {
        return
      }

      const fileName = file.name ?? getFileName(file.uri)
      if (!fileName.toLowerCase().endsWith('.gguf')) {
        Alert.alert(
          'Unsupported File',
          'Select a LoRA adapter in GGUF format.',
        )
        return
      }

      const [localCopy] = await keepLocalCopy({
        files: [
          {
            uri: file.uri,
            fileName,
          },
        ],
        destination: 'documentDirectory',
      })

      if (localCopy.status !== 'success') {
        Alert.alert(
          'Error',
          `Failed to copy LoRA adapter locally: ${localCopy.copyError}`,
        )
        return
      }

      const nextAdapters = dedupeAdapters([
        ...adapters,
        {
          path: stripFilePrefix(localCopy.localUri),
          scaled: scale,
        },
      ])

      await context.applyLoraAdapters(nextAdapters)
      const loadedAdapters = await context.getLoadedLoraAdapters()
      syncLoadedAdapters(loadedAdapters)
    } catch (error: unknown) {
      const errorMessage = getErrorMessage(error)
      if (!errorMessage.includes('user canceled the document picker')) {
        Alert.alert('Error', `Failed to apply LoRA adapter: ${errorMessage}`)
      }
    } finally {
      setIsBusy(false)
    }
  }

  const handleUpdateAdapterScale = async (path: string) => {
    if (!context) {
      Alert.alert('Error', 'No active context to update LoRA adapters on')
      return
    }

    const scale = parseScaleInput(adapterScaleInputs[path] ?? '')
    if (scale === null) {
      Alert.alert('Invalid Scale', 'Enter a valid numeric LoRA scale value.')
      return
    }

    try {
      setIsBusy(true)

      const nextAdapters = adapters.map((adapter) =>
        adapter.path === path ? { ...adapter, scaled: scale } : adapter,
      )

      await context.applyLoraAdapters(nextAdapters)
      const loadedAdapters = await context.getLoadedLoraAdapters()
      syncLoadedAdapters(loadedAdapters)
    } catch (error: unknown) {
      Alert.alert(
        'Error',
        `Failed to update LoRA scale: ${getErrorMessage(error)}`,
      )
    } finally {
      setIsBusy(false)
    }
  }

  const handleRemoveAdapter = async (path: string) => {
    if (!context) {
      Alert.alert('Error', 'No active context to remove LoRA adapters from')
      return
    }

    try {
      setIsBusy(true)

      const nextAdapters = adapters.filter((adapter) => adapter.path !== path)

      if (nextAdapters.length === 0) {
        await context.removeLoraAdapters()
        syncLoadedAdapters([])
      } else {
        await context.applyLoraAdapters(nextAdapters)
        const loadedAdapters = await context.getLoadedLoraAdapters()
        syncLoadedAdapters(loadedAdapters)
      }
    } catch (error: unknown) {
      Alert.alert(
        'Error',
        `Failed to remove LoRA adapter: ${getErrorMessage(error)}`,
      )
    } finally {
      setIsBusy(false)
    }
  }

  const handleRemoveAll = async () => {
    if (!context) {
      Alert.alert('Error', 'No active context to remove LoRA adapters from')
      return
    }

    if (adapters.length === 0) {
      return
    }

    try {
      setIsBusy(true)
      await context.removeLoraAdapters()
      syncLoadedAdapters([])
    } catch (error: unknown) {
      Alert.alert(
        'Error',
        `Failed to remove LoRA adapters: ${getErrorMessage(error)}`,
      )
    } finally {
      setIsBusy(false)
    }
  }

  let adapterContent = (
    <View style={styles.emptyState}>
      <Text style={styles.emptyStateText}>
        No LoRA adapters are currently applied.
      </Text>
    </View>
  )

  if (isBusy && adapters.length === 0) {
    adapterContent = (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="small" color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading adapters...</Text>
      </View>
    )
  } else if (adapters.length > 0) {
    adapterContent = (
      <>
        {adapters.map((adapter) => (
          <View key={adapter.path} style={styles.adapterCard}>
            <View style={styles.adapterHeader}>
              <View style={styles.adapterInfo}>
                <Text style={styles.adapterTitle}>
                  {getFileName(adapter.path)}
                </Text>
                <Text style={styles.adapterScale}>
                  {`Scale: ${formatScale(adapter.scaled)}`}
                </Text>
                <Text style={styles.adapterPath}>{adapter.path}</Text>

                <View style={styles.adapterControls}>
                  <TextInput
                    style={styles.adapterScaleInput}
                    value={adapterScaleInputs[adapter.path] ?? ''}
                    onChangeText={(value) =>
                      setAdapterScaleInputs((currentInputs) => ({
                        ...currentInputs,
                        [adapter.path]: value,
                      }))
                    }
                    editable={!isBusy}
                    placeholder="1.0"
                    placeholderTextColor={theme.colors.textSecondary}
                    autoCapitalize="none"
                    autoCorrect={false}
                    keyboardType={
                      Platform.OS === 'ios'
                        ? 'numbers-and-punctuation'
                        : 'numeric'
                    }
                  />

                  <TouchableOpacity
                    style={[styles.updateButton, isBusy && styles.disabledButton]}
                    onPress={() => void handleUpdateAdapterScale(adapter.path)}
                    disabled={isBusy}
                  >
                    <Text style={styles.updateButtonText}>Update Scale</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <TouchableOpacity
                style={[styles.removeButton, isBusy && styles.disabledButton]}
                onPress={() => void handleRemoveAdapter(adapter.path)}
                disabled={isBusy}
              >
                <Text style={styles.removeButtonText}>Remove</Text>
              </TouchableOpacity>
            </View>
          </View>
        ))}
      </>
    )
  }

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
            <Text style={styles.headerButton}>Close</Text>
          </TouchableOpacity>
          <Text style={styles.title}>LoRA Adapters</Text>
          <TouchableOpacity
            onPress={() => void refreshAdapters()}
            disabled={isBusy}
          >
            <Text
              style={[
                styles.headerButton,
                isBusy && styles.disabledHeaderButton,
              ]}
            >
              Refresh
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
        >
          <Text style={styles.description}>
            Apply LoRA GGUF files to the currently loaded context. Removing an
            adapter reapplies the remaining set for this context only.
          </Text>

          <View style={styles.inputSection}>
            <Text style={styles.inputLabel}>Scale For Next Adapter</Text>
            <TextInput
              style={styles.scaleInput}
              value={newAdapterScale}
              onChangeText={setNewAdapterScale}
              editable={!isBusy}
              placeholder="1.0"
              placeholderTextColor={theme.colors.textSecondary}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType={
                Platform.OS === 'ios' ? 'numbers-and-punctuation' : 'numeric'
              }
            />
            <Text style={styles.helperText}>
              This value is used when you add the next LoRA file.
            </Text>
          </View>

          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.addButton,
                (!context || isBusy) && styles.disabledButton,
              ]}
              onPress={handleAddAdapter}
              disabled={!context || isBusy}
            >
              <Text style={styles.buttonText}>
                {isBusy ? 'Working...' : 'Add LoRA File'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.removeAllButton,
                (!context || isBusy || adapters.length === 0) &&
                  styles.disabledButton,
              ]}
              onPress={handleRemoveAll}
              disabled={!context || isBusy || adapters.length === 0}
            >
              <Text style={styles.buttonText}>Remove All</Text>
            </TouchableOpacity>
          </View>

          <Text style={styles.sectionTitle}>Applied To Current Context</Text>

          {adapterContent}
        </ScrollView>
      </View>
    </Modal>
  )
}
