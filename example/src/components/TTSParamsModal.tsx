import React, { useEffect } from 'react'
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Linking,
} from 'react-native'
import { pick, keepLocalCopy } from '@react-native-documents/picker'
import type { TTSCapabilities } from '../../../src'
import type { TTSParams } from '../utils/storage'
import {
  saveTTSParams,
  loadTTSParams,
  resetTTSParams,
  DEFAULT_TTS_PARAMS,
  TTS_VOICE_AUDIO,
  TTS_VOICE_CLONE,
  TTS_VOICE_DEFAULT,
  TTS_VOICE_FILE,
  TTS_VOICE_OUTETTS_JSON,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import BaseParameterModal from './BaseParameterModal'
import { ParameterMenu } from './ParameterMenu'
import { ParameterTextInput } from './ParameterFormFields'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

const OUTETTS_SPEAKER_CREATOR =
  'https://huggingface.co/spaces/BricksDisplay/OuteTTS-Speaker-Creator'

interface TTSParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: TTSParams) => void
  // The loaded model's capabilities: its builtin voices, whether it clones a
  // voice from reference audio and whether it takes an instruct.
  caps: TTSCapabilities | null
}

// A local copy of a picked document, as a plain path.
const pickFile = async (type: string[], fallbackName: string) => {
  const [file] = await pick({ type, allowMultiSelection: false })
  if (!file?.uri) return null
  const [copy] = await keepLocalCopy({
    files: [{ uri: file.uri, fileName: file.name ?? fallbackName }],
    destination: 'documentDirectory',
  })
  return copy.status === 'success'
    ? copy.localUri.replace(/^file:\/\//, '')
    : null
}

export default function TTSParamsModal({
  visible,
  onClose,
  onSave,
  caps,
}: TTSParamsModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = StyleSheet.create({
    row: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      marginTop: 8,
    },
    button: {
      backgroundColor: theme.colors.primary,
      borderRadius: 6,
      paddingVertical: 8,
      paddingHorizontal: 12,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '500',
    },
    path: {
      flex: 1,
      fontSize: 12,
      color: theme.colors.textSecondary,
    },
    jsonInput: {
      ...themedStyles.textInput,
      minHeight: 120,
      textAlignVertical: 'top',
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

  const isOuteTTS = caps?.family === 'outetts'
  const builtinVoices = caps?.builtinVoices ?? []
  const defaultVoice = builtinVoices.find((v) => v.isDefault)?.name
  const voiceOptions = [
    ...(caps?.needsVoice ? [] : [TTS_VOICE_DEFAULT]),
    ...builtinVoices.map((v) => v.name),
    ...(caps?.canSpeakerFromAudio
      ? [TTS_VOICE_CLONE, TTS_VOICE_AUDIO, TTS_VOICE_FILE]
      : []),
    ...(isOuteTTS ? [TTS_VOICE_OUTETTS_JSON] : []),
  ]
  const selected = builtinVoices.find((v) => v.name === params.voice)

  const choose = async (
    key: 'refAudioPath' | 'speakerFilePath',
    type: string[],
    name: string,
  ) => {
    try {
      const path = await pickFile(type, name)
      if (path) updateParam(key, path)
    } catch (e: any) {
      if (e?.code !== 'OPERATION_CANCELED') Alert.alert('Error', String(e?.message ?? e))
    }
  }

  const onSaveHandler = () => {
    if (params.voice === TTS_VOICE_OUTETTS_JSON && params.outettsSpeakerJson.trim()) {
      try {
        JSON.parse(params.outettsSpeakerJson)
      } catch (e: any) {
        Alert.alert('Invalid JSON', String(e?.message ?? e))
        return
      }
    }
    handleSave(onSave, onClose)
  }

  return (
    <BaseParameterModal
      visible={visible}
      onClose={onClose}
      title="TTS Parameters"
      description="Voice and speaking style of the loaded TTS model. Sampling comes from the model's GGUF (general.sampling.*)."
      isLoading={isLoading}
      onSave={onSaveHandler}
      onReset={handleReset}
    >
      <ParameterMenu
        label="Voice"
        description={
          caps
            ? `${TTS_VOICE_DEFAULT}: the model's own${
                defaultVoice ? ` (${defaultVoice})` : ''
              }. A named voice is one of the codec's builtin voices.${
                caps.canSpeakerFromAudio
                  ? ` ${TTS_VOICE_CLONE}: the bundled clip. ${TTS_VOICE_AUDIO}: your WAV recording and what it says. ${TTS_VOICE_FILE}: a saved speaker (.gguf).`
                  : ''
              }${
                isOuteTTS
                  ? ` ${TTS_VOICE_OUTETTS_JSON}: a speaker made with the OuteTTS Speaker Creator.`
                  : ''
              }`
            : 'Load a TTS model to list its voices.'
        }
        value={params.voice}
        options={voiceOptions}
        onSelect={(value) => updateParam('voice', value)}
        placeholder={TTS_VOICE_DEFAULT}
      />
      {selected?.refText ? (
        <View style={themedStyles.paramGroup}>
          <Text style={themedStyles.paramDescription}>
            {`Reference transcript: ${selected.refText}`}
          </Text>
        </View>
      ) : null}

      {params.voice === TTS_VOICE_AUDIO ? (
        <>
          <View style={themedStyles.paramGroup}>
            <Text style={themedStyles.paramLabel}>Reference recording</Text>
            <Text style={themedStyles.paramDescription}>
              A WAV file of the voice to clone (a few seconds of clean speech).
            </Text>
            <View style={styles.row}>
              <TouchableOpacity
                style={styles.button}
                onPress={() => choose('refAudioPath', ['audio/*'], 'reference.wav')}
              >
                <Text style={styles.buttonText}>Choose WAV</Text>
              </TouchableOpacity>
              <Text style={styles.path} numberOfLines={2}>
                {params.refAudioPath || 'None'}
              </Text>
            </View>
          </View>
          <ParameterTextInput
            label="Transcript"
            description="What the recording says. Models that prompt with it (Qwen3-TTS Base, CosyVoice, NeuTTS) clone better with it; phoneme models get it phonemized."
            value={params.refText}
            onChangeText={(text) => updateParam('refText', text)}
            placeholder="What the recording says"
          />
        </>
      ) : null}

      {params.voice === TTS_VOICE_FILE ? (
        <>
          <View style={themedStyles.paramGroup}>
            <Text style={themedStyles.paramLabel}>Speaker file</Text>
            <Text style={themedStyles.paramDescription}>
              A speaker GGUF of this model (saveSpeaker, or codec.cpp tts-cli
              --save-speaker).
            </Text>
            <View style={styles.row}>
              <TouchableOpacity
                style={styles.button}
                onPress={() =>
                  choose('speakerFilePath', ['application/octet-stream', '*/*'], 'speaker.gguf')
                }
              >
                <Text style={styles.buttonText}>Choose file</Text>
              </TouchableOpacity>
              <Text style={styles.path} numberOfLines={2}>
                {params.speakerFilePath || 'None'}
              </Text>
            </View>
          </View>
          <ParameterTextInput
            label="Speaker name"
            description="The speaker in the file. Empty: the file's default."
            value={params.speakerName}
            onChangeText={(text) => updateParam('speakerName', text)}
            placeholder="default"
          />
        </>
      ) : null}

      {params.voice === TTS_VOICE_OUTETTS_JSON ? (
        <View style={themedStyles.paramGroup}>
          <Text style={themedStyles.paramLabel}>OuteTTS speaker JSON</Text>
          <Text style={themedStyles.paramDescription}>
            Create a speaker from your own recording with the OuteTTS Speaker
            Creator, then paste its JSON here.
          </Text>
          <View style={styles.row}>
            <TouchableOpacity
              style={styles.button}
              onPress={() =>
                Linking.openURL(OUTETTS_SPEAKER_CREATOR).catch(() =>
                  Alert.alert('Error', `Could not open ${OUTETTS_SPEAKER_CREATOR}`),
                )
              }
            >
              <Text style={styles.buttonText}>Open Speaker Creator</Text>
            </TouchableOpacity>
          </View>
          <TextInput
            style={[styles.jsonInput, { marginTop: 8 }]}
            value={params.outettsSpeakerJson}
            onChangeText={(text) => updateParam('outettsSpeakerJson', text)}
            placeholder="Paste the speaker JSON here..."
            placeholderTextColor={theme.colors.textSecondary}
            multiline
            autoCapitalize="none"
            autoCorrect={false}
          />
        </View>
      ) : null}

      {caps?.canTakeInstruct ? (
        <ParameterTextInput
          label="Instruct"
          description="How to speak, e.g. 'Speak slowly in a sad, quiet whisper.' Empty: none."
          value={params.instruct}
          onChangeText={(text) => updateParam('instruct', text)}
          placeholder="Speak cheerfully."
        />
      ) : null}
    </BaseParameterModal>
  )
}
