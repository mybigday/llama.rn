import React, { useEffect, useRef, useState } from 'react'
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Linking,
  PermissionsAndroid,
  Platform,
} from 'react-native'
import { pick, keepLocalCopy } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import { AudioManager, AudioRecorder } from 'react-native-audio-api'
import type { TTSCapabilities } from '../../../src'
import type { TTSCustomVoice, TTSParams } from '../utils/storage'
import {
  saveTTSParams,
  loadTTSParams,
  resetTTSParams,
  DEFAULT_TTS_PARAMS,
  TTS_VOICE_BUNDLED,
  TTS_VOICE_DEFAULT,
} from '../utils/storage'
import { writeWavFile } from '../utils/audioUtils'
import { useParameterModal } from '../hooks/useParameterModal'
import BaseParameterModal from './BaseParameterModal'
import { ParameterTextInput } from './ParameterFormFields'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

const OUTETTS_SPEAKER_CREATOR =
  'https://huggingface.co/spaces/BricksDisplay/OuteTTS-Speaker-Creator'
const RECORD_SAMPLE_RATE = 24000

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

// A builtin voice's label; the default one is marked (or is just "Default"
// when that is its name).
const voiceTitle = (name: string, isDefault: boolean) => {
  if (!isDefault) return name
  return name.toLowerCase() === 'default' ? TTS_VOICE_DEFAULT : `${name} (default)`
}

const newId = () => `voice-${Date.now().toString(36)}`
const baseName = (path: string) =>
  (path.split('/').pop() ?? path).replace(/\.[^.]+$/, '')

export default function TTSParamsModal({
  visible,
  onClose,
  onSave,
  caps,
}: TTSParamsModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = StyleSheet.create({
    tabs: {
      flexDirection: 'row',
      borderRadius: 8,
      borderWidth: 1,
      borderColor: theme.colors.border,
      overflow: 'hidden',
      marginBottom: 16,
    },
    tab: { flex: 1, paddingVertical: 10, alignItems: 'center' },
    tabActive: { backgroundColor: theme.colors.primary },
    tabText: { fontSize: 14, fontWeight: '600', color: theme.colors.text },
    tabTextActive: { color: theme.colors.white },
    option: {
      flexDirection: 'row',
      alignItems: 'center',
      paddingVertical: 10,
      borderBottomWidth: StyleSheet.hairlineWidth,
      borderBottomColor: theme.colors.border,
    },
    radio: {
      width: 18,
      height: 18,
      borderRadius: 9,
      borderWidth: 2,
      borderColor: theme.colors.primary,
      marginRight: 10,
      alignItems: 'center',
      justifyContent: 'center',
    },
    radioDot: {
      width: 8,
      height: 8,
      borderRadius: 4,
      backgroundColor: theme.colors.primary,
    },
    optionBody: { flex: 1 },
    optionTitle: { fontSize: 15, color: theme.colors.text },
    optionNote: { fontSize: 12, color: theme.colors.textSecondary },
    remove: { color: theme.colors.error, fontSize: 13, paddingHorizontal: 8 },
    row: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 8 },
    button: {
      backgroundColor: theme.colors.primary,
      borderRadius: 6,
      paddingVertical: 8,
      paddingHorizontal: 12,
    },
    buttonDisabled: { opacity: 0.4 },
    buttonText: { color: theme.colors.white, fontSize: 14, fontWeight: '500' },
    recording: { backgroundColor: theme.colors.error },
    note: { flex: 1, fontSize: 12, color: theme.colors.textSecondary },
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

  const [tab, setTab] = useState<'select' | 'clone'>('select')
  // The Clone Voice form.
  const [cloneName, setCloneName] = useState('')
  const [cloneAudio, setCloneAudio] = useState('')
  const [cloneText, setCloneText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const recorder = useRef<AudioRecorder | null>(null)
  const recorded = useRef<Float32Array[]>([])

  useEffect(() => {
    if (visible) {
      loadParamsAsync()
      setTab('select')
    }
  }, [loadParamsAsync, visible])

  useEffect(() => () => recorder.current?.stop(), [])

  const isOuteTTS = caps?.family === 'outetts'
  const canClone = !!caps?.canSpeakerFromAudio
  const builtinVoices = caps?.builtinVoices ?? []
  // Speaker files work for any model that takes voices; clones need cloning.
  const customVoices = (params.customVoices ?? []).filter((v) =>
    v.kind === 'file' ? canClone || isOuteTTS : canClone,
  )
  // Choices of the Select tab, and whether the model offers any at all.
  // A builtin voice the codec marks as default stands for the model's own
  // voice; without one, a generic Default entry does.
  const builtinDefault = builtinVoices.find((v) => v.isDefault)
  const options: Array<{ id: string; title: string; note?: string }> = [
    ...(caps?.needsVoice || builtinDefault
      ? []
      : [
          {
            id: TTS_VOICE_DEFAULT,
            title: TTS_VOICE_DEFAULT,
            note: "The model's own voice",
          },
        ]),
    ...builtinVoices.map((v) => ({
      id: v.name,
      title: voiceTitle(v.name, v.isDefault),
      note: v.refText || undefined,
    })),
    ...(canClone
      ? [
          {
            id: TTS_VOICE_BUNDLED,
            title: TTS_VOICE_BUNDLED,
            note: 'A clone of the clip bundled with the example',
          },
        ]
      : []),
    ...customVoices.map((v) => ({
      id: v.id,
      title: v.name,
      note: {
        audio: `Clone of ${baseName(v.audioPath ?? '')}`,
        file: `Speaker file ${baseName(v.filePath ?? '')}`,
      }[v.kind],
    })),
  ]
  const canSelect = builtinVoices.length > 0 || canClone || isOuteTTS
  const showCloneTab = canClone || isOuteTTS
  const transcriptRequired = !!caps?.usesRefText

  const addVoice = (voice: TTSCustomVoice) => {
    updateParam('customVoices', [...(params.customVoices ?? []), voice])
    updateParam('voice', voice.id)
  }

  const removeVoice = (id: string) => {
    updateParam(
      'customVoices',
      (params.customVoices ?? []).filter((v) => v.id !== id),
    )
    if (params.voice === id) updateParam('voice', TTS_VOICE_DEFAULT)
  }

  const fail = (e: any) => {
    if (e?.code !== 'OPERATION_CANCELED') Alert.alert('Error', String(e?.message ?? e))
  }

  const loadSpeakerFile = async () => {
    try {
      const path = await pickFile(['application/octet-stream', '*/*'], 'speaker.gguf')
      if (path) addVoice({ id: newId(), name: baseName(path), kind: 'file', filePath: path })
    } catch (e) {
      fail(e)
    }
  }

  const chooseWav = async () => {
    try {
      const path = await pickFile(['audio/*'], 'reference.wav')
      if (path) setCloneAudio(path)
    } catch (e) {
      fail(e)
    }
  }

  const toggleRecording = async () => {
    try {
      if (isRecording) {
        recorder.current?.stop()
        recorder.current = null
        setIsRecording(false)
        const n = recorded.current.reduce((acc, c) => acc + c.length, 0)
        const pcm = new Float32Array(n)
        recorded.current.reduce((at, c) => {
          pcm.set(c, at)
          return at + c.length
        }, 0)
        const path = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/recording-${Date.now()}.wav`
        await writeWavFile(path, pcm, RECORD_SAMPLE_RATE)
        setCloneAudio(path)
        return
      }
      // react-native-audio-api's permission request never settles on
      // Android; React Native's own asks there.
      const granted =
        Platform.OS === 'android'
          ? (await PermissionsAndroid.request(
              PermissionsAndroid.PERMISSIONS.RECORD_AUDIO,
            )) === PermissionsAndroid.RESULTS.GRANTED
          : (await AudioManager.requestRecordingPermissions()) === 'Granted'
      if (!granted) {
        Alert.alert('Microphone', 'Recording needs the microphone permission.')
        return
      }
      AudioManager.setAudioSessionOptions({
        iosCategory: 'playAndRecord',
        iosMode: 'default',
        iosOptions: ['defaultToSpeaker'],
      })
      recorded.current = []
      const rec = new AudioRecorder({
        sampleRate: RECORD_SAMPLE_RATE,
        bufferLengthInSamples: RECORD_SAMPLE_RATE / 10,
      })
      rec.onAudioReady(({ buffer, numFrames }) => {
        // copyFromChannel, not getChannelData: the latter's ArrayBuffer frees
        // the recorder's own samples (a double free in react-native-audio-api 0.9).
        const chunk = new Float32Array(numFrames)
        buffer.copyFromChannel(chunk, 0)
        recorded.current.push(chunk)
      })
      rec.start()
      recorder.current = rec
      setIsRecording(true)
    } catch (e) {
      setIsRecording(false)
      fail(e)
    }
  }

  const audioLabel = cloneAudio ? baseName(cloneAudio) : 'No audio yet'
  const cloneReady =
    !!cloneAudio && (!transcriptRequired || !!cloneText.trim())

  const createVoice = () => {
    addVoice({
      id: newId(),
      name: cloneName.trim() || baseName(cloneAudio),
      kind: 'audio',
      audioPath: cloneAudio,
      refText: cloneText.trim(),
    })
    setCloneName('')
    setCloneAudio('')
    setCloneText('')
    setTab('select')
  }

  const selectTab = (
    <View style={themedStyles.paramGroup}>
      {canSelect ? (
        options.map((o) => {
          const custom = customVoices.find((v) => v.id === o.id)
          const current = params.voice || TTS_VOICE_DEFAULT
          const selected =
            current === o.id ||
            (current === TTS_VOICE_DEFAULT && builtinDefault?.name === o.id)
          return (
            <TouchableOpacity
              key={o.id}
              style={styles.option}
              onPress={() => updateParam('voice', o.id)}
            >
              <View style={styles.radio}>
                {selected ? <View style={styles.radioDot} /> : null}
              </View>
              <View style={styles.optionBody}>
                <Text style={styles.optionTitle}>{o.title}</Text>
                {o.note ? (
                  <Text style={styles.optionNote} numberOfLines={2}>
                    {o.note}
                  </Text>
                ) : null}
              </View>
              {custom ? (
                <TouchableOpacity onPress={() => removeVoice(custom.id)}>
                  <Text style={styles.remove}>Remove</Text>
                </TouchableOpacity>
              ) : null}
            </TouchableOpacity>
          )
        })
      ) : (
        <Text style={themedStyles.paramDescription}>
          {caps
            ? 'This model speaks with a single voice: it has no voices to choose from and cannot clone one.'
            : 'Load a TTS model to list its voices.'}
        </Text>
      )}
      {canClone || isOuteTTS ? (
        <View style={styles.row}>
          <TouchableOpacity style={styles.button} onPress={loadSpeakerFile}>
            <Text style={styles.buttonText}>Load speaker file</Text>
          </TouchableOpacity>
          <Text style={styles.note}>
            A speaker GGUF of this model (saveSpeaker, tts-cli --save-speaker)
          </Text>
        </View>
      ) : null}
    </View>
  )

  // OuteTTS clones a voice outside the app: the Speaker Creator exports a
  // speaker GGUF, loaded here like any speaker file.
  const outettsCloneTab = (
    <View style={themedStyles.paramGroup}>
      <Text style={themedStyles.paramLabel}>OuteTTS Speaker Creator</Text>
      <Text style={themedStyles.paramDescription}>
        Create a speaker from your recording with the OuteTTS Speaker Creator,
        export it as a speaker GGUF, then load that file here.
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
        <TouchableOpacity
          style={styles.button}
          onPress={async () => {
            await loadSpeakerFile()
            setTab('select')
          }}
        >
          <Text style={styles.buttonText}>Load speaker file</Text>
        </TouchableOpacity>
      </View>
    </View>
  )

  const cloneTab = (
    <>
      <>
        <View style={themedStyles.paramGroup}>
          <Text style={themedStyles.paramLabel}>Reference audio</Text>
          <Text style={themedStyles.paramDescription}>
            Record or choose a WAV of the voice to clone (a few seconds of
            clean speech).
          </Text>
          <View style={styles.row}>
            <TouchableOpacity
              style={[styles.button, isRecording && styles.recording]}
              onPress={toggleRecording}
            >
              <Text style={styles.buttonText}>
                {isRecording ? 'Stop' : 'Record'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.button, isRecording && styles.buttonDisabled]}
              disabled={isRecording}
              onPress={chooseWav}
            >
              <Text style={styles.buttonText}>Choose WAV</Text>
            </TouchableOpacity>
          </View>
          <Text style={[styles.note, { marginTop: 8 }]} numberOfLines={2}>
            {isRecording ? 'Recording...' : audioLabel}
          </Text>
        </View>
        <ParameterTextInput
          label={transcriptRequired ? 'Transcript (required)' : 'Transcript'}
          description={
              transcriptRequired
                ? 'What the audio says. This model prompts with it; phoneme models get it phonemized.'
                : 'What the audio says. This model does not use it.'
            }
          value={cloneText}
          onChangeText={setCloneText}
          placeholder="What the audio says"
        />
      </>
      <ParameterTextInput
        label="Name"
        description="How the voice is listed under Select loaded voice."
        value={cloneName}
        onChangeText={setCloneName}
        placeholder="My voice"
      />
      <View style={themedStyles.paramGroup}>
        <TouchableOpacity
          style={[styles.button, !cloneReady && styles.buttonDisabled]}
          disabled={!cloneReady}
          onPress={createVoice}
        >
          <Text style={styles.buttonText}>Create voice</Text>
        </TouchableOpacity>
      </View>
    </>
  )

  let activeTab = selectTab
  if (tab === 'clone' && showCloneTab) activeTab = isOuteTTS ? outettsCloneTab : cloneTab

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
      {showCloneTab ? (
        <View style={styles.tabs}>
          {(['select', 'clone'] as const).map((t) => (
            <TouchableOpacity
              key={t}
              style={[styles.tab, tab === t && styles.tabActive]}
              onPress={() => setTab(t)}
            >
              <Text style={[styles.tabText, tab === t && styles.tabTextActive]}>
                {t === 'select' ? 'Select loaded voice' : 'Clone Voice'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      ) : null}
      {activeTab}

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
