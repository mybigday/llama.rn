import React, { useState } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Platform,
} from 'react-native'
import {
  pick,
  saveDocuments,
  keepLocalCopy,
} from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import type { LlamaContext } from '../../../src'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface SessionModalProps {
  visible: boolean
  onClose: () => void
  context: LlamaContext | null
}

export default function SessionModal({
  visible,
  onClose,
  context,
}: SessionModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const [isLoading, setIsLoading] = useState(false)

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
    cancelButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.primary,
      fontWeight: '600',
    },
    content: {
      flex: 1,
      paddingHorizontal: 16,
      paddingVertical: 20,
      backgroundColor: theme.colors.background,
    },
    description: {
      ...themedStyles.description,
      color: theme.colors.textSecondary,
    },
    buttonContainer: {
      marginVertical: 10,
    },
    button: {
      backgroundColor: theme.colors.primary,
      paddingVertical: 15,
      paddingHorizontal: 20,
      borderRadius: 8,
      alignItems: 'center',
      marginVertical: 8,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
    },
    saveButton: {
      backgroundColor: theme.colors.valid,
    },
    loadButton: {
      backgroundColor: theme.colors.primary,
    },
    disabledButton: {
      backgroundColor: theme.colors.border,
    },
  })

  const handleSaveSession = async () => {
    if (!context) {
      Alert.alert('Error', 'No active context to save session from')
      return
    }

    try {
      setIsLoading(true)

      // Generate a default filename with timestamp
      const timestamp = new Date().toISOString().replace(/[.:]/g, '-')
      const defaultFilename = `llama-session-${timestamp}.bin`

      // Create a temporary file path
      const tempDir = ReactNativeBlobUtil.fs.dirs.CacheDir
      const tempFilePath = `${tempDir}/${defaultFilename}`

      // Save the session to temporary file first
      const tokensCount = await context.saveSession(tempFilePath)

      // Now use saveDocuments to let user choose where to save it
      const result = await saveDocuments({
        sourceUris: [`file://${tempFilePath}`],
        fileName: defaultFilename,
        mimeType: 'application/octet-stream',
      })

      if (result && result.length > 0) {
        // Clean up temporary file
        await ReactNativeBlobUtil.fs.unlink(tempFilePath).catch(() => {
          // Ignore cleanup errors
        })

        Alert.alert(
          'Success',
          `Session saved successfully!\nTokens saved: ${tokensCount}`,
        )
      }
    } catch (error: any) {
      if (!error.message.includes('user canceled the document picker')) {
        Alert.alert('Error', `Failed to save session: ${error.message}`)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoadSession = async () => {
    if (!context) {
      Alert.alert('Error', 'No active context to load session into')
      return
    }

    try {
      setIsLoading(true)

      const [file] = await pick({
        type: ['application/octet-stream', '*/*'],
        allowMultiSelection: false,
      })

      if (file?.uri) {
        // Keep a local copy of the file
        const [localCopy] = await keepLocalCopy({
          files: [
            {
              uri: file.uri,
              fileName: file.name ?? 'session.bin',
            },
          ],
          destination: 'documentDirectory',
        })

        if (localCopy.status === 'success') {
          // Clean the file path
          const filePath = localCopy.localUri.replace(/^file:\/\//, '')

          // Load the session
          const sessionResult = await context.loadSession(filePath)

          Alert.alert(
            'Success',
            `Session loaded successfully!\nTokens loaded: ${sessionResult.tokens_loaded}`,
          )

          onClose()
        } else {
          Alert.alert(
            'Error',
            `Failed to copy session file: ${localCopy.copyError}`,
          )
        }
      }
    } catch (error: any) {
      if (!error.message.includes('user canceled the document picker')) {
        Alert.alert('Error', `Failed to load session: ${error.message}`)
      }
    } finally {
      setIsLoading(false)
    }
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
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.title}>Session Management</Text>
          <View style={{ width: 60 }} />
        </View>

        <View style={styles.content}>
          <Text style={styles.description}>
            Save your current conversation state to a file or load a previously
            saved session. Loading a session restores the context without
            clearing your current conversation.
          </Text>

          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[
                styles.button,
                styles.saveButton,
                (!context || isLoading) && styles.disabledButton,
              ]}
              onPress={handleSaveSession}
              disabled={!context || isLoading}
            >
              <Text style={styles.buttonText}>
                {isLoading ? 'Saving...' : 'Save Session'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.loadButton,
                (!context || isLoading) && styles.disabledButton,
              ]}
              onPress={handleLoadSession}
              disabled={!context || isLoading}
            >
              <Text style={styles.buttonText}>
                {isLoading ? 'Loading...' : 'Load Session'}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  )
}
