import React, { useEffect, useState } from 'react'
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  Alert,
  TouchableOpacity,
  Linking,
} from 'react-native'
import type { TTSParams } from '../utils/storage'
import {
  saveTTSParams,
  loadTTSParams,
  resetTTSParams,
  DEFAULT_TTS_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import BaseParameterModal from './BaseParameterModal'
import { CommonStyles } from '../styles/commonStyles'

const styles = StyleSheet.create({
  paramGroup: CommonStyles.paramGroup,
  paramLabel: CommonStyles.paramLabel,
  paramDescription: CommonStyles.paramDescription,
  textInput: {
    ...CommonStyles.textInput,
    minHeight: 120,
    textAlignVertical: 'top',
  },
  linkContainer: {
    backgroundColor: '#E8F4FD',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    marginBottom: 16,
  },
  linkText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 8,
  },
  linkButton: {
    backgroundColor: '#007AFF',
    borderRadius: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    alignSelf: 'flex-start',
  },
  linkButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  validationText: {
    fontSize: 12,
    marginTop: 4,
  },
  validText: {
    color: '#34C759',
  },
  invalidText: {
    color: '#FF3B30',
  },
})

interface TTSParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: TTSParams) => void
}

export default function TTSParamsModal({
  visible,
  onClose,
  onSave,
}: TTSParamsModalProps) {
  const [speakerConfigText, setSpeakerConfigText] = useState('')
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean
    message: string
  }>({ isValid: true, message: '' })

  const {
    params,
    isLoading,
    loadParamsAsync,
    handleSave,
    handleReset,
    updateParam,
  } = useParameterModal({
    loadParams: loadTTSParams,
    saveParams: saveTTSParams,
    resetParams: resetTTSParams,
    defaultParams: DEFAULT_TTS_PARAMS,
  })

  useEffect(() => {
    if (visible) {
      loadParamsAsync()
    }
  }, [loadParamsAsync, visible])

  // Update text input when params change
  useEffect(() => {
    if (params.speakerConfig) {
      try {
        setSpeakerConfigText(JSON.stringify(params.speakerConfig, null, 2))
        setValidationResult({ isValid: true, message: 'Valid JSON format' })
      } catch {
        setSpeakerConfigText('')
      }
    } else {
      setSpeakerConfigText('')
      setValidationResult({
        isValid: true,
        message: 'No speaker config set (will use default)',
      })
    }
  }, [params.speakerConfig])

  const validateAndUpdateSpeakerConfig = (text: string) => {
    setSpeakerConfigText(text)

    if (!text.trim()) {
      updateParam('speakerConfig', null)
      setValidationResult({
        isValid: true,
        message: 'No speaker config set (will use default)',
      })
      return
    }

    try {
      const parsed = JSON.parse(text)
      updateParam('speakerConfig', parsed)
      setValidationResult({ isValid: true, message: 'Valid JSON format' })
    } catch (error) {
      setValidationResult({
        isValid: false,
        message: `Invalid JSON: ${
          error instanceof Error ? error.message : 'Parse error'
        }`,
      })
    }
  }

  const onSaveHandler = () => {
    if (!validationResult.isValid) {
      Alert.alert(
        'Invalid Configuration',
        'Please fix the JSON format before saving.',
      )
      return
    }
    handleSave(onSave, onClose)
  }

  const openSpeakerCreator = () => {
    const url =
      'https://huggingface.co/spaces/BricksDisplay/OuteTTS-Speaker-Creator'
    Linking.openURL(url).catch(() => {
      Alert.alert('Error', `Could not open the link. Please visit: ${url}`)
    })
  }

  return (
    <BaseParameterModal
      visible={visible}
      onClose={onClose}
      title="TTS Parameters"
      description="Configure text-to-speech parameters including speaker configuration for OuteTTS models."
      isLoading={isLoading}
      onSave={onSaveHandler}
      onReset={handleReset}
    >
      <View style={styles.paramGroup}>
        <Text style={styles.paramLabel}>Speaker Configuration</Text>
        <Text style={styles.paramDescription}>
          Import a JSON speaker configuration for OuteTTS. This controls the
          voice characteristics, accent, and speaking style. Leave empty to use
          the default voice.
        </Text>

        <View style={styles.linkContainer}>
          <Text style={styles.linkText}>
            Generate a custom speaker configuration using the OuteTTS Speaker
            Creator:
          </Text>
          <TouchableOpacity
            style={styles.linkButton}
            onPress={openSpeakerCreator}
          >
            <Text style={styles.linkButtonText}>Open Speaker Creator</Text>
          </TouchableOpacity>
        </View>

        <TextInput
          style={styles.textInput}
          value={speakerConfigText}
          onChangeText={validateAndUpdateSpeakerConfig}
          placeholder="Paste your speaker configuration JSON here..."
          multiline
          autoCapitalize="none"
          autoCorrect={false}
        />

        <Text
          style={[
            styles.validationText,
            validationResult.isValid ? styles.validText : styles.invalidText,
          ]}
        >
          {validationResult.message}
        </Text>
      </View>
    </BaseParameterModal>
  )
}
