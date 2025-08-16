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
import { CommonStyles } from '../styles/commonStyles'

const styles = StyleSheet.create({
  container: CommonStyles.container,
  header: {
    ...CommonStyles.modalHeader,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  title: CommonStyles.modalTitle,
  cancelButton: CommonStyles.headerButtonText,
  content: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 20,
  },
  description: CommonStyles.description,
  buttonContainer: {
    marginVertical: 10,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    marginVertical: 8,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  saveButton: {
    backgroundColor: '#34C759',
  },
  loadButton: {
    backgroundColor: '#007AFF',
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
})

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
  const [isLoading, setIsLoading] = useState(false)

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
