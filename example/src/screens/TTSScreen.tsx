import React, { useState, useLayoutEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { saveDocuments } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
import { toIPA } from 'phonemize/all'
import { TTSModelDownloadCard } from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import TTSParamsModal from '../components/TTSParamsModal'
import { AudioPlayer } from '../components/AudioPlayer'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type {
  ContextParams,
  CompletionParams,
  TTSParams,
} from '../utils/storage'
import {
  loadContextParams,
  loadCompletionParams,
  loadTTSParams,
} from '../utils/storage'
import { HeaderButton } from '../components/HeaderButton'
import { MaskedProgress } from '../components/MaskedProgress'
import { createWavFile, decodeBase64Pcm16, dumpTtsWavToDisk } from '../utils/audioUtils'
import { DEFAULT_REF_AUDIO } from '../assets/voices/en_f1'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

const models: (typeof MODELS.OUTE_TTS_0_3)[] = [
  MODELS.OUTE_TTS_0_3,
  MODELS.OUTE_TTS_1_0,
  MODELS.SOPRANO_1_1,
  MODELS.NEUTTS_NANO,
  MODELS.NEUTTS_AIR,
  MODELS.CSM_1B,
  MODELS.QWEN3_TTS_0_6B,
  MODELS.MOSS_TTS_REALTIME,
  MODELS.MOSS_TTSD_V07,
  MODELS.CHATTERBOX_MULTILINGUAL,
  MODELS.BLUEMAGPIE_TTS,
]

export default function TTSScreen({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const styles = StyleSheet.create({
    // Using themed styles for common patterns
    container: themedStyles.container,
    setupContainer: themedStyles.setupContainer,
    scrollContent: themedStyles.scrollContent,
    setupDescription: themedStyles.setupDescription,
    loadingContainer: themedStyles.loadingContainer,
    loadingText: themedStyles.loadingText,
    header: {
      ...themedStyles.header,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    headerTitle: {
      fontSize: 18,
      fontWeight: '600',
      color: theme.colors.text,
    },
    textInput: {
      ...themedStyles.textInput,
      height: 120,
      textAlignVertical: 'top',
    },
    button: themedStyles.primaryButton,
    buttonText: themedStyles.primaryButtonText,
    disabledButton: themedStyles.primaryButtonDisabled,
    generateButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: 12,
      paddingVertical: 16,
      alignItems: 'center',
    },
    generateButtonText: {
      color: theme.colors.white,
      fontSize: 18,
      fontWeight: '600',
    },
    audioCard: {
      backgroundColor: theme.colors.surface,
      borderRadius: 12,
      padding: 16,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    audioDescription: {
      fontSize: 16,
      color: theme.colors.text,
      marginBottom: 16,
      fontStyle: 'italic',
    },
    infoSection: {
      backgroundColor: theme.colors.surface,
      borderRadius: 12,
      padding: 16,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    infoTitle: {
      fontSize: 16,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 8,
      marginTop: 16,
    },
    infoText: {
      fontSize: 14,
      color: theme.colors.textSecondary,
      lineHeight: 20,
    },
    content: {
      flex: 1,
      padding: 16,
    },
    inputSection: {
      marginBottom: 24,
    },
    sectionTitle: {
      fontSize: 18,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 12,
    },
    characterCount: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      textAlign: 'right',
      marginTop: 4,
    },
    controlSection: {
      marginBottom: 24,
    },
    generateButtonDisabled: {
      backgroundColor: theme.colors.disabled,
    },
    audioSection: {
      marginBottom: 24,
    },
  })
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isVocoderReady, setIsVocoderReady] = useState(false)
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null)
  const [audioData, setAudioData] = useState<Float32Array | null>(null)
  const [sampleRate, setSampleRate] = useState<number>(24000)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showTTSParamsModal, setShowTTSParamsModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)
  const [ttsParams, setTtsParams] = useState<TTSParams | null>(null)

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const handleSaveTTSParams = (params: TTSParams) => {
    setTtsParams(params)
  }

  // Encode the bundled ResembleAI Chatterbox demo clip into a speaker JSON
  // for voice-clone families. Returns null if the loaded codec doesn't
  // produce a speakerEmb (i.e. the family isn't a voice-clone arch).
  const encodeDefaultRefVoice = async (): Promise<object | null> => {
    if (!context || !isVocoderReady) {
      Alert.alert(
        'Vocoder not ready',
        'Load a TTS model + vocoder before encoding a reference voice.',
      )
      return null
    }
    const pcm = decodeBase64Pcm16(DEFAULT_REF_AUDIO.pcm16Base64)
    const enc = await context.encodeSpeaker({
      refAudioPCM: pcm,
      refAudioSampleRate: DEFAULT_REF_AUDIO.sampleRate,
      refText: DEFAULT_REF_AUDIO.refText,
    })
    if (!enc.speakerEmb || enc.speakerNRows === 0) {
      Alert.alert(
        'No speaker section',
        "The loaded codec.gguf doesn't carry a speaker section — this codec isn't a voice-clone arch (try Chatterbox / Qwen3-TTS / MOSS-TTSD).",
      )
      return null
    }
    // Shape matches what getFormattedAudioCompletion's resolver lifts out
    // (speakerEmb / speakerNRows / speakerHiddenDim) plus the codec-token
    // bundle for any model whose speaker JSON expects ref_codes too.
    return {
      ref_text: enc.refText,
      ref_codes: enc.refCodes,
      n_q: enc.nQ,
      n_frames: enc.nFrames,
      sample_rate: enc.sampleRate,
      codebook_size: enc.codebookSize,
      speakerEmb: enc.speakerEmb,
      speakerNRows: enc.speakerNRows,
      speakerHiddenDim: enc.speakerHiddenDim,
    }
  }

  const saveAudioAsWav = async () => {
    if (!audioData || audioData.length === 0) {
      Alert.alert('No Audio', 'No audio data available to save.')
      return
    }

    try {
      console.log('Starting WAV file creation...')

      // Create WAV file from audio data
      const wavBuffer = createWavFile(audioData, sampleRate || 24000, 16)
      console.log('WAV buffer created, size:', wavBuffer.byteLength)

      // Convert to base64 for proper binary file handling
      const uint8Array = new Uint8Array(wavBuffer)
      // Convert Uint8Array to string for base64 encoding
      const binaryString = String.fromCharCode.apply(
        null,
        Array.from(uint8Array),
      )
      const base64Data = ReactNativeBlobUtil.base64.encode(binaryString)
      console.log('Base64 data length:', base64Data.length)

      // Generate filename with timestamp
      const timestamp = new Date().toISOString().replace(/[.:]/g, '-')
      const filename = `generated-speech-${timestamp}.wav`

      // Create temporary file path in a more accessible location
      const tempFilePath = `${ReactNativeBlobUtil.fs.dirs.CacheDir}/${filename}`

      // Write the WAV file as base64 binary data
      await ReactNativeBlobUtil.fs.writeFile(tempFilePath, base64Data, 'base64')

      // Save using documents picker with proper file URI
      await saveDocuments({
        sourceUris: [`file://${tempFilePath}`],
        fileName: filename,
        mimeType: 'audio/wav',
        copy: false,
      })

      // Clean up temp file
      try {
        await ReactNativeBlobUtil.fs.unlink(tempFilePath)
      } catch (cleanupError) {
        console.warn('Failed to cleanup temp file:', cleanupError)
      }

      Alert.alert('Success', `Audio saved as ${filename}`)
    } catch (error: any) {
      console.error('Error saving audio:', error)

      // Provide more detailed error information
      if (error.message?.includes('User cancelled')) {
        // User cancelled the save dialog, don't show error
        return
      }

      Alert.alert(
        'Error',
        `Failed to save audio: ${
          error.message || 'Unknown error'
        }\n\nPlease check the console for more details.`,
      )
    }
  }

  // Load TTS parameters on mount
  useLayoutEffect(() => {
    const loadTTSParamsAsync = async () => {
      try {
        const params = await loadTTSParams()
        setTtsParams(params)
      } catch (error) {
        console.error('Failed to load TTS params:', error)
      }
    }
    loadTTSParamsAsync()
  }, [])

  // Set up navigation header button
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton
              iconName="speaker"
              onPress={() => setShowTTSParamsModal(true)}
            />
            <HeaderButton
              iconName="cog-outline"
              onPress={() => setShowCompletionParamsModal(true)}
            />
          </View>
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            iconName="cog-outline"
            onPress={() => setShowContextParamsModal(true)}
          />
        ),
      })
    }
  }, [navigation, isModelReady])

  const initializeModels = async (ttsPath: string, vocoderPath: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const params = contextParams || (await loadContextParams())
      // Override context params with TTS-appropriate defaults so real
      // devices don't OOM.  The generic DEFAULT_CONTEXT_PARAMS assumes chat
      // workloads (n_ctx=8192, mlock on, f16 KV), which pushes voice-clone /
      // continuous-latent models like BlueMagpie over Pixel's per-process
      // budget once the codec (up to ~1.8GB) is mmap'd.
      //   - n_ctx=4096 covers real TTS prompt lengths (BlueMagpie continuous
      //     uses 1 KV position per patch → 2000 max; token-flow families
      //     stay under 4k tokens) while halving attn KV vs 8192.
      //   - use_mlock=false lets the OS demand-page cold model regions; the
      //     "failed to mlock" warning at init was a symptom of forced
      //     residency + tight RAM.
      //   - cache_type_k/v='q8_0' further halves attn KV memory at
      //     negligible TTS quality cost.
      const ttsParams = {
        ...params,
        n_ctx: Math.min(params.n_ctx ?? 8192, 4096),
        use_mlock: false,
        cache_type_k: 'q8_0' as const,
        cache_type_v: 'q8_0' as const,
      }
      // Initialize the TTS model
      const llamaContext = await initLlama(
        {
          model: ttsPath,
          ...ttsParams,
        },
        (progress) => {
          // Progress is reported as 1 to 100
          setInitProgress(progress)
        },
      )

      setContext(llamaContext)
      setIsModelReady(true)

      // Initialize vocoder directly after TTS model
      try {
        await llamaContext.initVocoder({ path: vocoderPath, n_batch: 4096 })
        setIsVocoderReady(true)
      } catch (vocoderError) {
        console.log('Vocoder initialization error:', vocoderError)
          Alert.alert(
          'TTS Model Loaded',
          'TTS model loaded successfully, but vocoder initialization failed. Audio tokens can be generated but not played.',
        )
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize models: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const generateSpeech = async () => {
    if (!inputText.trim() || !context || isLoading) return

    try {
      setIsLoading(true)

      const prompt = inputText.trim()
      const caps = await context.getTTSCapabilities()
      const {
        prompt: formattedPrompt,
        grammar,
        embedding,
        flow,
        speakerEmbPrefix,
        speakerEmbRows,
        speakerEmbHiddenDim,
      } = await context.getFormattedAudioCompletion({
        prompt,
        speaker: ttsParams?.speakerConfig || undefined,
        phonemizer: (text: string, language: string) =>
          toIPA(text, { anyAscii: true, language })
            .replace(/ɫ/g, 'l')
            .replace(/oʊ/g, 'əʊ')
            .replace(/ˈ\b/g, ''),
      })

      const params = completionParams || (await loadCompletionParams())

      const ttsSamplingDefaults: {
        temperature: number
        top_p: number
        top_k?: number
      } =
        caps.family === 'neutts'
          ? { temperature: 1.0, top_k: 50, top_p: 1.0 }
          : { temperature: 0.7, top_p: 0.9 }

      // codec_lm AR flow (CSM and similar multi-codebook RVQ models): the
      // backbone never emits text tokens — codec_lm.generateAudioCodes
      // drives the AR loop and returns the (T × n_codebook) interleaved
      // code matrix straight from the residual depth decoder.
      if (flow === 'codec_lm_ar') {
        const audioSampleRate = await context.getAudioSampleRate()
        const arResult = await context.generateAudioCodes({
          prompt: formattedPrompt,
          // codec_lm-AR frame rates are 12.5–25 Hz; 200 frames is 8–16 s
          // of audio.  Capping below the previous 500 default makes CPU
          // smoke-tests finish in reasonable wall-clock and gives caller a
          // hard ceiling — JS callers that want longer audio should bump
          // params.n_predict themselves.
          maxFrames: params?.n_predict
            ? Math.min(params.n_predict, 600)
            : 200,
          temperature: params.temperature ?? ttsSamplingDefaults.temperature,
          topP: params.top_p ?? ttsSamplingDefaults.top_p,
          topK:
            (params as { top_k?: number }).top_k ??
            ttsSamplingDefaults.top_k ??
            50,
          seed: 0,
          ...(speakerEmbPrefix
            ? {
                speakerEmbPrefix,
                speakerEmbRows,
                speakerEmbHiddenDim,
              }
            : {}),
        })

        if (arResult.codes.length === 0) {
          setGeneratedAudio(
            `No audio frames produced (codec_lm AR loop returned empty).`,
          )
          Alert.alert(
            'Generation Failed',
            'codec_lm AR loop produced no frames — check that the codec.gguf carries a `lm.*` section and the backbone matches the codec_lm host arch.',
          )
          return
        }

        setGeneratedAudio(
          `Generated ${arResult.nFrames} frames × ${arResult.nCodebook} codebooks for: "${inputText.trim()}"`,
        )

        if (isVocoderReady && context.decodeAudioTokens) {
          try {
            const decodedAudio = await context.decodeAudioTokens(
              arResult.codes,
            )
            const audioFloat32 = new Float32Array(decodedAudio)
            setAudioData(audioFloat32)
            setSampleRate(audioSampleRate)
            void dumpTtsWavToDisk(audioFloat32, audioSampleRate)
            setGeneratedAudio(
              `Generated audio data (${audioFloat32.length} samples) for: "${inputText.trim()}"`,
            )
          } catch (decodeError) {
            console.log('codec_lm AR audio decoding error:', decodeError)
          }
        }

        Alert.alert(
          'Speech Generated',
          `Successfully generated ${arResult.nFrames} audio frames${
            arResult.stoppedOnEos ? ' (stopped on EOS)' : ''
          }! ${
            isVocoderReady
              ? 'Audio data is ready for playback.'
              : 'Note: Audio playback requires vocoder setup.'
          }`,
        )
        return
      }

      const collectedTokens: number[] = []

      const result = await context.completion(
        {
          prompt: formattedPrompt,
          grammar,
          embedding,
          n_predict: params?.n_predict || 4096,
          temperature: params.temperature ?? ttsSamplingDefaults.temperature,
          top_p: params.top_p ?? ttsSamplingDefaults.top_p,
          top_k: (params as { top_k?: number }).top_k ?? ttsSamplingDefaults.top_k,
          stop: params.stop || ['<|im_end|>', '<|SPEECH_GENERATION_END|>'],
        },
        (data) => {
          // Collect tokens for potential audio processing
          if (data.token && typeof data.token === 'number') {
            collectedTokens.push(data.token)
          }
        },
      )

      const audioSampleRate = await context.getAudioSampleRate()

      if (
        result.embeddings &&
        result.embedding_dim &&
        result.embeddings.length > 0
      ) {
        setGeneratedAudio(
          `Generated ${
            result.embeddings.length / result.embedding_dim
          } audio embedding frames for: "${inputText.trim()}"`,
        )

        if (isVocoderReady && context.decodeAudioEmbeddings) {
          try {
            const decodedAudio = await context.decodeAudioEmbeddings(
              result.embeddings,
              result.embedding_dim,
            )
            console.log('Generated audio data length:', decodedAudio.length)

            const audioFloat32 = new Float32Array(decodedAudio)
            setAudioData(audioFloat32)
            setSampleRate(audioSampleRate)
            void dumpTtsWavToDisk(audioFloat32, audioSampleRate)

            setGeneratedAudio(
              `Generated audio data (${
                audioFloat32.length
              } samples) for: "${inputText.trim()}"`,
            )
          } catch (decodeError) {
            console.log('Audio embedding decoding error:', decodeError)
          }
        }

        Alert.alert(
          'Speech Generated',
          `Successfully generated ${
            result.embeddings.length / result.embedding_dim
          } audio embedding frames! ${
            isVocoderReady
              ? 'Audio data is ready for playback.'
              : 'Note: Audio playback requires vocoder setup.'
          }`,
        )
      } else if (result.audio_tokens && result.audio_tokens.length > 0) {
        setGeneratedAudio(
          `Generated ${
            result.audio_tokens.length
          } audio tokens for: "${inputText.trim()}"`,
        )

        // If vocoder is available, decode audio tokens
        if (isVocoderReady && context.decodeAudioTokens) {
          try {
            const decodedAudio = await context.decodeAudioTokens(
              result.audio_tokens,
            )
            console.log('Generated audio data length:', decodedAudio.length)

            // Convert ArrayBuffer to Float32Array for AudioPlayer
            const audioFloat32 = new Float32Array(decodedAudio)
            setAudioData(audioFloat32)
            setSampleRate(audioSampleRate)
            void dumpTtsWavToDisk(audioFloat32, audioSampleRate)

            setGeneratedAudio(
              `Generated audio data (${
                audioFloat32.length
              } samples) for: "${inputText.trim()}"`,
            )
          } catch (decodeError) {
            console.log('Audio decoding error:', decodeError)
          }
        }

        Alert.alert(
          'Speech Generated',
          `Successfully generated ${result.audio_tokens.length} audio tokens! ${
            isVocoderReady
              ? 'Audio data is ready for playback.'
              : 'Note: Audio playback requires vocoder setup.'
          }`,
        )
      } else {
        setGeneratedAudio(
          `Text processed: "${inputText.trim()}" (No audio tokens generated)`,
        )
        Alert.alert(
          'Processing Complete',
          'Text was processed but no audio tokens were generated. Check your model configuration.',
        )
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to generate speech: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  if (!isModelReady) {
    return (
      <View style={styles.container}>
        <ScrollView
          style={styles.setupContainer}
          contentContainerStyle={styles.scrollContent}
        >
          <Text style={styles.setupDescription}>
            Download a TTS model to convert text into natural-sounding speech.
            Each model includes a paired vocoder for full audio generation and
            playback.
          </Text>

          {models.map((model) => (
            <TTSModelDownloadCard
              key={model.name}
              title={model.name}
              repo={model.repo}
              filename={model.filename}
              size={model.size}
              vocoder={model.vocoder}
              onInitialize={initializeModels}
            />
          ))}
        </ScrollView>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />

        <TTSParamsModal
          visible={showTTSParamsModal}
          onClose={() => setShowTTSParamsModal(false)}
          onSave={handleSaveTTSParams}
          onEncodeDefaultRef={encodeDefaultRefVoice}
        />

        <MaskedProgress
          visible={isLoading}
          text={
            context
              ? 'Initializing vocoder...'
              : `Initializing TTS model... ${initProgress}%`
          }
          progress={initProgress}
          showProgressBar={!context && initProgress > 0}
        />
      </View>
    )
  }

  return (
    <View style={styles.container}>
      <ScrollView style={styles.content}>
        <View style={styles.inputSection}>
          <Text style={styles.sectionTitle}>Enter Text to Speak</Text>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Type the text you want to convert to speech..."
            multiline
            textAlignVertical="top"
            maxLength={500}
            keyboardType="ascii-capable"
          />
          <Text style={styles.characterCount}>
            {inputText.length}
            /500
          </Text>
        </View>

        <View style={styles.controlSection}>
          <TouchableOpacity
            style={[
              styles.generateButton,
              (!inputText.trim() || isLoading) && styles.generateButtonDisabled,
            ]}
            onPress={generateSpeech}
            disabled={!inputText.trim() || isLoading}
          >
            {isLoading ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Text style={styles.generateButtonText}>Generate Speech</Text>
            )}
          </TouchableOpacity>
        </View>

        {generatedAudio && (
          <View style={styles.audioSection}>
            <Text style={styles.sectionTitle}>Generated Audio</Text>
            <View style={styles.audioCard}>
              <Text style={styles.audioDescription}>{generatedAudio}</Text>

              {audioData && audioData.length > 0 ? (
                <>
                  <AudioPlayer audio={audioData} sr={sampleRate} />
                  <TouchableOpacity
                    style={[
                      styles.generateButton,
                      { marginTop: 12, backgroundColor: theme.colors.valid },
                    ]}
                    onPress={saveAudioAsWav}
                  >
                    <Text style={styles.generateButtonText}>Save as WAV</Text>
                  </TouchableOpacity>
                </>
              ) : (
                <Text style={styles.infoText}>
                  {isVocoderReady
                    ? 'Audio data will appear here after generation'
                    : 'Vocoder not ready — audio playback unavailable'}
                </Text>
              )}
            </View>
          </View>
        )}

        <View style={styles.infoSection}>
          <Text style={styles.infoTitle}>About Text-to-Speech</Text>
          <Text style={styles.infoText}>
            Multiple neural TTS models are supported — OuteTTS, Soprano and
            Kani TTS. Each uses a paired vocoder to decode
            audio tokens into natural-sounding speech with proper intonation
            and pronunciation.
          </Text>

          <Text style={styles.infoTitle}>Speaker Configuration</Text>
          <Text style={styles.infoText}>
            Tap the &quot;TTS&quot; button in the header to configure a custom
            speaker voice. Speaker configs are model-specific — refer to each
            model&apos;s documentation, or leave it empty to use the default
            voice.
          </Text>

          <Text style={styles.infoTitle}>Tips for Better Results</Text>
          <Text style={styles.infoText}>
            {`• Use clear, well-punctuated text
• Shorter texts often produce better quality
• Configure a custom speaker for different voices
• Avoid special characters and abbreviations`}
          </Text>
        </View>
      </ScrollView>

      <ContextParamsModal
        visible={showContextParamsModal}
        onClose={() => setShowContextParamsModal(false)}
        onSave={handleSaveContextParams}
      />

      <CompletionParamsModal
        visible={showCompletionParamsModal}
        onClose={() => setShowCompletionParamsModal(false)}
        onSave={handleSaveCompletionParams}
      />

      <TTSParamsModal
        visible={showTTSParamsModal}
        onClose={() => setShowTTSParamsModal(false)}
        onSave={handleSaveTTSParams}
      />
    </View>
  )
}
