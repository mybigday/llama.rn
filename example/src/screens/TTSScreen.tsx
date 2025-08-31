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
import { createWavFile } from '../utils/audioUtils'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

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
      // Initialize the TTS model
      const llamaContext = await initLlama(
        {
          model: ttsPath,
          ...params,
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
          'OuteTTS model loaded successfully, but vocoder initialization failed. Audio tokens can be generated but not played.',
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

      const text = inputText.trim()
      // Get formatted prompt and guide tokens using OuteTTS format
      const { prompt: formattedPrompt, grammar } =
        await context.getFormattedAudioCompletion(
          ttsParams?.speakerConfig || null,
          text,
        )

      const guideTokens: number[] = await context.getAudioCompletionGuideTokens(
        text,
      )

      const collectedTokens: number[] = []
      const params = completionParams || (await loadCompletionParams())

      const result = await context.completion(
        {
          prompt: formattedPrompt,
          grammar,
          guide_tokens: guideTokens,
          n_predict: params.n_predict || 4096,
          temperature: params.temperature || 0.7,
          top_p: params.top_p || 0.9,
          stop: params.stop || ['<|im_end|>'],
        },
        (data) => {
          // Collect tokens for potential audio processing
          if (data.token && typeof data.token === 'number') {
            collectedTokens.push(data.token)
          }
        },
      )

      // Check if we got audio tokens
      if (result.audio_tokens && result.audio_tokens.length > 0) {
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
            setSampleRate(24000) // OuteTTS default sample rate

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
          'Text was processed but no audio tokens were generated. This may require proper OuteTTS model configuration.',
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
            Download the OuteTTS model to convert text into natural-sounding
            speech. For full audio generation and playback, you&apos;ll also
            need the WavTokenizer vocoder model.
          </Text>

          <TTSModelDownloadCard
            title={MODELS.OUTE_TTS_0_3.name}
            repo={MODELS.OUTE_TTS_0_3.repo}
            filename={MODELS.OUTE_TTS_0_3.filename}
            size={MODELS.OUTE_TTS_0_3.size}
            vocoder={MODELS.OUTE_TTS_0_3.vocoder}
            onInitialize={initializeModels}
          />
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
                    : 'Download WavTokenizer for audio playback'}
                </Text>
              )}
            </View>
          </View>
        )}

        <View style={styles.infoSection}>
          <Text style={styles.infoTitle}>About OuteTTS</Text>
          <Text style={styles.infoText}>
            OuteTTS is a neural text-to-speech model that converts written text
            into natural-sounding speech. It supports various languages and can
            generate high-quality audio with proper intonation and
            pronunciation.
          </Text>

          <Text style={styles.infoTitle}>Speaker Configuration</Text>
          <Text style={styles.infoText}>
            Tap the &quot;TTS&quot; button in the header to configure a custom
            speaker voice. You can import speaker configurations from the
            OuteTTS Speaker Creator or leave it empty to use the default voice.
          </Text>

          <Text style={styles.infoTitle}>Tips for Better Results</Text>
          <Text style={styles.infoText}>
            {`• Use clear, well-punctuated text
• Shorter texts often produce better quality
• Configure custom speaker for different voices
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
