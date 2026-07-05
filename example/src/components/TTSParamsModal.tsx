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
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface TTSParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: TTSParams) => void
  /**
   * Optional voice-clone shortcut: runs `encodeSpeaker` on the bundled
   * Chatterbox demo clip and returns a speaker JSON the user can paste/save.
   * Caller (TTSScreen) wires this up only when a vocoder context is ready
   * and the loaded family is a voice-clone arch.
   */
  onEncodeDefaultRef?: () => Promise<object | null>
}

export default function TTSParamsModal({
  visible,
  onClose,
  onSave,
  onEncodeDefaultRef,
}: TTSParamsModalProps) {
  const [encodingDefault, setEncodingDefault] = useState(false)
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const [speakerConfigText, setSpeakerConfigText] = useState('')
  const [validationResult, setValidationResult] = useState<{
    isValid: boolean
    message: string
  }>({ isValid: true, message: '' })

  const styles = StyleSheet.create({
    textInput: {
      ...themedStyles.textInput,
      minHeight: 120,
      textAlignVertical: 'top',
    },
    linkContainer: {
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      padding: 12,
      marginTop: 8,
      marginBottom: 16,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    linkText: {
      fontSize: 14,
      color: theme.colors.text,
      marginBottom: 8,
    },
    linkButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: 6,
      paddingVertical: 8,
      paddingHorizontal: 12,
      alignSelf: 'flex-start',
    },
    linkButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '500',
    },
    validationText: {
      fontSize: 12,
      marginTop: 4,
    },
    validText: {
      color: '#34A759',
    },
    invalidText: {
      color: theme.colors.error,
    },
  })

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

  const onUseChatterboxDemo = async () => {
    if (!onEncodeDefaultRef || encodingDefault) return
    setEncodingDefault(true)
    try {
      const speakerJson = await onEncodeDefaultRef()
      if (!speakerJson) return
      const pretty = JSON.stringify(speakerJson, null, 2)
      setSpeakerConfigText(pretty)
      updateParam('speakerConfig', speakerJson)
      setValidationResult({
        isValid: true,
        message:
          'Encoded bundled Chatterbox demo voice — save to apply, or edit before saving.',
      })
    } catch (e: any) {
      Alert.alert(
        'Encode failed',
        e?.message ||
          'Could not encode the bundled reference clip. Make sure a voice-clone codec is loaded.',
      )
    } finally {
      setEncodingDefault(false)
    }
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
      <View style={themedStyles.paramGroup}>
        <Text style={themedStyles.paramLabel}>Speaker Configuration</Text>
        <Text style={themedStyles.paramDescription}>
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

        {onEncodeDefaultRef && (
          <View style={styles.linkContainer}>
            <Text style={styles.linkText}>
              Voice-clone families (Chatterbox / Qwen3-TTS / MOSS-TTSD): encode
              the bundled ResembleAI Chatterbox demo clip (en_f1, 3s) as the
              reference speaker.
            </Text>
            <TouchableOpacity
              style={[
                styles.linkButton,
                encodingDefault && { opacity: 0.6 },
              ]}
              onPress={onUseChatterboxDemo}
              disabled={encodingDefault}
            >
              <Text style={styles.linkButtonText}>
                {encodingDefault
                  ? 'Encoding…'
                  : 'Use Chatterbox demo voice'}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        <TextInput
          style={styles.textInput}
          value={speakerConfigText}
          onChangeText={validateAndUpdateSpeakerConfig}
          placeholder="Paste your speaker configuration JSON here..."
          placeholderTextColor={theme.colors.textSecondary}
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
