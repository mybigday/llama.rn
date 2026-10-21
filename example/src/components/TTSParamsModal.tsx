import React, { useEffect } from 'react'
import { View, Text } from 'react-native'
import type { TTSCapabilities } from '../../../src'
import type { TTSParams } from '../utils/storage'
import {
  saveTTSParams,
  loadTTSParams,
  resetTTSParams,
  DEFAULT_TTS_PARAMS,
  TTS_VOICE_CLONE,
  TTS_VOICE_DEFAULT,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import BaseParameterModal from './BaseParameterModal'
import { ParameterMenu } from './ParameterMenu'
import { ParameterTextInput } from './ParameterFormFields'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface TTSParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: TTSParams) => void
  // The loaded model's capabilities: its builtin voices, whether it clones a
  // voice from reference audio and whether it takes an instruct.
  caps: TTSCapabilities | null
}

export default function TTSParamsModal({
  visible,
  onClose,
  onSave,
  caps,
}: TTSParamsModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

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

  const builtinVoices = caps?.builtinVoices ?? []
  const defaultVoice = builtinVoices.find((v) => v.isDefault)?.name
  const voiceOptions = [
    ...(caps?.needsVoice ? [] : [TTS_VOICE_DEFAULT]),
    ...builtinVoices.map((v) => v.name),
    ...(caps?.canSpeakerFromAudio ? [TTS_VOICE_CLONE] : []),
  ]
  const selected = builtinVoices.find((v) => v.name === params.voice)

  return (
    <BaseParameterModal
      visible={visible}
      onClose={onClose}
      title="TTS Parameters"
      description="Voice and speaking style of the loaded TTS model. Sampling comes from the model's GGUF (general.sampling.*)."
      isLoading={isLoading}
      onSave={() => handleSave(onSave, onClose)}
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
                  ? ` ${TTS_VOICE_CLONE}: clones the bundled reference clip (createSpeaker).`
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
