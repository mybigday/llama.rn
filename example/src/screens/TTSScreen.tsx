import React, { useState, useLayoutEffect, useEffect, useRef } from 'react'
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Animated,
  Easing,
} from 'react-native'
import { saveDocuments } from '@react-native-documents/picker'
import ReactNativeBlobUtil from 'react-native-blob-util'
// eslint-disable-next-line import/no-unresolved -- valid package `exports` subpath the resolver can't follow
import { toIPA } from 'phonemize/all'
import { TTSModelDownloadCard } from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import TTSParamsModal from '../components/TTSParamsModal'
import { AudioPlayer } from '../components/AudioPlayer'
import { Waveform } from '../components/Waveform'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type {
  ContextParams,
  TTSParams,
} from '../utils/storage'
import {
  loadContextParams,
  loadTTSParams,
} from '../utils/storage'
import { HeaderButton } from '../components/HeaderButton'
import { MaskedProgress } from '../components/MaskedProgress'
import { createWavFile, decodeBase64Pcm16, dumpTtsWavToDisk } from '../utils/audioUtils'
// en_f1: ResembleAI's Chatterbox English demo reference clip (base64 PCM24k),
// committed so Metro can bundle it — used by the voice-clone demo below.
import { DEFAULT_REF_AUDIO } from '../assets/voices/en_f1'
import { initLlama, LlamaContext, LlamaSpeaker } from '../../../src' // import 'llama.rn'

const models: (typeof MODELS.OUTE_TTS_0_3)[] = [
  MODELS.OUTE_TTS_0_3,
  MODELS.OUTE_TTS_1_0,
  MODELS.SOPRANO_1_1,
  MODELS.NEUTTS_NANO,
  MODELS.NEUTTS_AIR,
  MODELS.CSM_1B,
  MODELS.QWEN3_TTS_0_6B,
  MODELS.MOSS_TTS_REALTIME,
  MODELS.MOSS_TTSD_V05,
  MODELS.CHATTERBOX_MULTILINGUAL,
  MODELS.CHATTERBOX_TURBO,
  MODELS.BREEZYVOICE,
  MODELS.BLUEMAGPIE_TTS,
]

interface GenerationStats {
  generationSec: number
  tokenCount: number
  tokenLabel: string
  audioSec: number | null
}

const EQ_BAR_COUNT = 5

// Looping equalizer animation shown while tokens are being generated, so the
// screen visibly "works" even for model families that don't stream tokens.
function EqualizerBars({ color }: { color: string }) {
  const anims = useRef(
    Array.from({ length: EQ_BAR_COUNT }, () => new Animated.Value(0.3)),
  ).current

  useEffect(() => {
    const loops = anims.map((v, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(v, {
            toValue: 1,
            duration: 240 + i * 80,
            easing: Easing.inOut(Easing.quad),
            useNativeDriver: true,
          }),
          Animated.timing(v, {
            toValue: 0.25,
            duration: 240 + (EQ_BAR_COUNT - i) * 70,
            easing: Easing.inOut(Easing.quad),
            useNativeDriver: true,
          }),
        ]),
      ),
    )
    loops.forEach((l) => l.start())
    return () => loops.forEach((l) => l.stop())
  }, [anims])

  return (
    <View
      style={{
        flexDirection: 'row',
        alignItems: 'center',
        gap: 3,
        height: 24,
      }}
    >
      {anims.map((v, i) => (
        <Animated.View
          key={i}
          style={{
            width: 4,
            height: 22,
            borderRadius: 2,
            backgroundColor: color,
            transform: [{ scaleY: v }],
          }}
        />
      ))}
    </View>
  )
}

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
    generatingHeader: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 12,
      marginBottom: 12,
    },
    generatingTitle: {
      fontSize: 16,
      fontWeight: '600',
      color: theme.colors.text,
    },
    statsRow: {
      flexDirection: 'row',
      marginTop: 8,
      marginBottom: 4,
    },
    statCell: {
      flex: 1,
      alignItems: 'center',
    },
    statValue: {
      fontSize: 17,
      fontWeight: '700',
      color: theme.colors.text,
      fontVariant: ['tabular-nums'],
    },
    statLabel: {
      fontSize: 11,
      color: theme.colors.textSecondary,
      marginTop: 2,
      textTransform: 'uppercase',
      letterSpacing: 0.5,
    },
  })
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [liveTokenCount, setLiveTokenCount] = useState(0)
  const [liveElapsedMs, setLiveElapsedMs] = useState(0)
  const [genStats, setGenStats] = useState<GenerationStats | null>(null)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [isVocoderReady, setIsVocoderReady] = useState(false)
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null)
  const [audioData, setAudioData] = useState<Float32Array | null>(null)
  const [sampleRate, setSampleRate] = useState<number>(24000)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showTTSParamsModal, setShowTTSParamsModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [ttsParams, setTtsParams] = useState<TTSParams | null>(null)

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveTTSParams = (params: TTSParams) => {
    setTtsParams(params)
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

  // Tick the elapsed-time readout while speech is being generated
  useEffect(() => {
    if (!isGenerating) return undefined
    const startedAt = Date.now()
    setLiveElapsedMs(0)
    const interval = setInterval(() => {
      setLiveElapsedMs(Date.now() - startedAt)
    }, 100)
    return () => clearInterval(interval)
  }, [isGenerating])

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
      //   - cache_type_k/v='q8_0' further halves attn KV memory, but ONLY for
      //     BlueMagpie (the one family whose codec pushes memory limits).
      //     Small token-flow audio LMs are sensitive to KV quantization —
      //     OuteTTS 0.3 (Qwen2 500M) produces corrupt audio with q8_0 KV.
      const isBlueMagpie = ttsPath
        .toLowerCase()
        .match(/bluemagpie|barbet/)
      // CosyVoice-family LMs (BreezyVoice) put their relative positions in a
      // KQ bias that only runs without flash attention, so keep the KV in F16.
      const isCosyVoice = ttsPath.toLowerCase().match(/breezyvoice|cosyvoice/)
      // Chatterbox T3 decodes a guidance pair (two sequences); Turbo has none.
      const isChatterboxT3 =
        /chatterbox/.test(ttsPath.toLowerCase()) &&
        !/turbo/.test(ttsPath.toLowerCase())
      const ttsParams = {
        ...params,
        n_ctx: Math.min(params.n_ctx ?? 8192, 4096),
        use_mlock: false,
        ...(isBlueMagpie
          ? {
              cache_type_k: 'q8_0' as const,
              cache_type_v: 'q8_0' as const,
            }
          : {}),
        ...(isCosyVoice
          ? {
              cache_type_k: 'f16' as const,
              cache_type_v: 'f16' as const,
            }
          : {}),
        ...(isChatterboxT3 ? { n_parallel: 2 } : {}),
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
        await llamaContext.initVocoder({
          path: vocoderPath,
          n_batch: 4096,
          // The flow decoder runs 2 U-Net passes per Euler step; 6 steps
          // (default 10) keeps phone latency reasonable.
          ...(isCosyVoice
            ? { kv_overrides: ['cosyvoice.flow.n_timesteps=int:6'] }
            : {}),
        })
        setIsVocoderReady(true)
        if (isCosyVoice && !inputText.trim()) {
          setInputText('今天天氣真好，我們一起去公園走走吧。')
        }
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
    if (!inputText.trim() || !context || isGenerating) return

    // Create a LlamaSpeaker from the bundled ref audio when speakerConfig is a
    // non-null object (voice-clone family). String values are passed as-is
    // (built-in voice name). null / undefined → no speaker.
    let spk: LlamaSpeaker | undefined
    try {
      setIsGenerating(true)
      setLiveTokenCount(0)
      setGenStats(null)
      setGeneratedAudio(null)
      setAudioData(null)

      // Each synthesis request starts at position 0. Embedding-driven TTS
      // prefills cannot safely reuse the previous generation's KV entries.
      await context.clearCache(false)

      const speakerCfg = ttsParams?.speakerConfig
      const caps = await context.getTTSCapabilities()
      // Voice-clone models without a voice of their own (Qwen3-TTS Base)
      // clone the bundled reference clip unless a voice is configured.
      const cloneDefault = caps.needsVoice && !speakerCfg
      if (
        ((speakerCfg && typeof speakerCfg === 'object') || cloneDefault) &&
        isVocoderReady
      ) {
        const pcm = decodeBase64Pcm16(DEFAULT_REF_AUDIO.pcm16Base64)
        spk = await context.createSpeaker({
          refAudio: pcm,
          refAudioSampleRate: DEFAULT_REF_AUDIO.sampleRate,
          refText: DEFAULT_REF_AUDIO.refText,
        })
      }

      const resolvedSpeaker: string | LlamaSpeaker | undefined =
        spk ??
        (typeof speakerCfg === 'string' ? speakerCfg : undefined)

      const prompt = inputText.trim()
      // completionParams are the model's sampling and constraints, reported by
      // codec.cpp; the completion runs with exactly these.
      const { completionParams: ttsCompletion } =
        await context.getFormattedAudioCompletion({
          prompt,
          speaker: resolvedSpeaker,
          // The IPA phonemizer is for the English phoneme families; Chinese
          // text (BreezyVoice's bopomofo frontend) goes in as typed.
          phonemizer: (text: string, language: string) =>
            language.startsWith('zh')
              ? text
              : toIPA(text, { anyAscii: true, language })
              .replace(/ɫ/g, 'l')
              .replace(/oʊ/g, 'əʊ')
              .replace(/ˈ\b/g, ''),
        })
      console.log('TTS completionParams:', JSON.stringify(ttsCompletion))

      // The completion's result carries the decoder input: `audio_tokens`
      // (codec codes) or `embeddings` (latents, e.g. Soprano / BlueMagpie).
      const collectedTokens: number[] = []
      const genStartTime = Date.now()

      const result = await context.completion(
        ttsCompletion,
        (data) => {
          // Token-flow families (e.g. OuteTTS) stream one callback per audio
          // token; continuous-latent families (e.g. BlueMagpie) don't stream,
          // so the live counter stays hidden for them.
          setLiveTokenCount((c) => c + 1)
          // Collect tokens for potential audio processing
          if (data.token && typeof data.token === 'number') {
            collectedTokens.push(data.token)
          }
        },
      )

      const audioSampleRate = await context.getAudioSampleRate()

      // Total time-to-audio (token generation + vocoder decode), captured
      // after decoding below so the RTF stat reflects real wall time.
      const finishStats = (tokenCount: number, tokenLabel: string) =>
        (audioFloat32: Float32Array | null) => {
          setGenStats({
            generationSec: (Date.now() - genStartTime) / 1000,
            tokenCount,
            tokenLabel,
            audioSec: audioFloat32
              ? audioFloat32.length / audioSampleRate
              : null,
          })
        }

      if (
        result.embeddings &&
        result.embedding_dim &&
        result.embeddings.length > 0
      ) {
        const frameCount = result.embeddings.length / result.embedding_dim
        const setStats = finishStats(frameCount, 'frames')
        setGeneratedAudio(`“${inputText.trim()}”`)

        let audioFloat32: Float32Array | null = null
        if (isVocoderReady && context.decodeAudioEmbeddings) {
          try {
            const decodedAudio = await context.decodeAudioEmbeddings(
              result.embeddings,
              result.embedding_dim,
            )
            console.log('Generated audio data length:', decodedAudio.length)

            audioFloat32 = new Float32Array(decodedAudio)
            setAudioData(audioFloat32)
            setSampleRate(audioSampleRate)
            void dumpTtsWavToDisk(audioFloat32, audioSampleRate)
          } catch (decodeError) {
            console.log('Audio embedding decoding error:', decodeError)
          }
        }
        setStats(audioFloat32)
      } else if (result.audio_tokens && result.audio_tokens.length > 0) {
        const setStats = finishStats(result.audio_tokens.length, 'tokens')
        setGeneratedAudio(`“${inputText.trim()}”`)

        // If vocoder is available, decode audio tokens
        let audioFloat32: Float32Array | null = null
        if (isVocoderReady && context.decodeAudioTokens) {
          try {
            const decodedAudio = await context.decodeAudioTokens(
              result.audio_tokens,
            )
            console.log('Generated audio data length:', decodedAudio.length)

            // Convert ArrayBuffer to Float32Array for AudioPlayer
            audioFloat32 = new Float32Array(decodedAudio)
            setAudioData(audioFloat32)
            setSampleRate(audioSampleRate)
            void dumpTtsWavToDisk(audioFloat32, audioSampleRate)
          } catch (decodeError) {
            console.log('Audio decoding error:', decodeError)
          }
        }
        setStats(audioFloat32)
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
      if (spk) await spk.release()
      setIsGenerating(false)
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
              (!inputText.trim() || isGenerating) &&
                styles.generateButtonDisabled,
            ]}
            onPress={generateSpeech}
            disabled={!inputText.trim() || isGenerating}
          >
            {isGenerating ? (
              <ActivityIndicator size="small" color="white" />
            ) : (
              <Text style={styles.generateButtonText}>Generate Speech</Text>
            )}
          </TouchableOpacity>
        </View>

        {isGenerating && (
          <View style={styles.audioSection}>
            <View style={styles.audioCard}>
              <View style={styles.generatingHeader}>
                <EqualizerBars color={theme.colors.primary} />
                <Text style={styles.generatingTitle}>
                  Generating on device…
                </Text>
              </View>
              <View style={styles.statsRow}>
                <View style={styles.statCell}>
                  <Text style={styles.statValue}>
                    {`${(liveElapsedMs / 1000).toFixed(1)}s`}
                  </Text>
                  <Text style={styles.statLabel}>elapsed</Text>
                </View>
                {liveTokenCount > 0 && (
                  <>
                    <View style={styles.statCell}>
                      <Text style={styles.statValue}>{liveTokenCount}</Text>
                      <Text style={styles.statLabel}>audio tokens</Text>
                    </View>
                    <View style={styles.statCell}>
                      <Text style={styles.statValue}>
                        {liveElapsedMs > 0
                          ? (liveTokenCount / (liveElapsedMs / 1000)).toFixed(0)
                          : '—'}
                      </Text>
                      <Text style={styles.statLabel}>tok/s</Text>
                    </View>
                  </>
                )}
              </View>
            </View>
          </View>
        )}

        {generatedAudio && (
          <View style={styles.audioSection}>
            <Text style={styles.sectionTitle}>Generated Audio</Text>
            <View style={styles.audioCard}>
              <Text style={styles.audioDescription}>{generatedAudio}</Text>

              {audioData && audioData.length > 0 && (
                <Waveform audio={audioData} color={theme.colors.primary} />
              )}

              {genStats && (
                <View style={styles.statsRow}>
                  <View style={styles.statCell}>
                    <Text style={styles.statValue}>
                      {`${genStats.generationSec.toFixed(2)}s`}
                    </Text>
                    <Text style={styles.statLabel}>generation</Text>
                  </View>
                  {genStats.audioSec != null && (
                    <>
                      <View style={styles.statCell}>
                        <Text style={styles.statValue}>
                          {`${genStats.audioSec.toFixed(2)}s`}
                        </Text>
                        <Text style={styles.statLabel}>audio</Text>
                      </View>
                      <View style={styles.statCell}>
                        <Text style={styles.statValue}>
                          {`${(
                            genStats.generationSec / genStats.audioSec
                          ).toFixed(2)}×`}
                        </Text>
                        <Text style={styles.statLabel}>RTF</Text>
                      </View>
                    </>
                  )}
                  <View style={styles.statCell}>
                    <Text style={styles.statValue}>{genStats.tokenCount}</Text>
                    <Text style={styles.statLabel}>{genStats.tokenLabel}</Text>
                  </View>
                </View>
              )}

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

      <TTSParamsModal
        visible={showTTSParamsModal}
        onClose={() => setShowTTSParamsModal(false)}
        onSave={handleSaveTTSParams}
      />
    </View>
  )
}
