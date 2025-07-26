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
import { SafeAreaView } from 'react-native-safe-area-context'
import { initLlama, LlamaContext } from '../../../src'
import { TTSModelDownloadCard } from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { AudioPlayer } from '../components/AudioPlayer'
import { CommonStyles } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadContextParams, loadCompletionParams } from '../utils/storage'
import { HeaderButton } from '../components/HeaderButton'

// Sample speaker configuration for OuteTTS
const speakerConfig = null

const styles = StyleSheet.create({
  // Using shared styles for common patterns
  container: CommonStyles.container,
  setupContainer: CommonStyles.setupContainer,
  scrollContent: CommonStyles.scrollContent,
  setupTitle: CommonStyles.setupTitle,
  setupDescription: CommonStyles.setupDescription,
  loadingContainer: CommonStyles.loadingContainer,
  loadingText: CommonStyles.loadingText,
  header: {
    ...CommonStyles.header,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
  },
  clearButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#FF3B30',
    borderRadius: 6,
  },
  clearButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
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
    color: '#000',
    marginBottom: 12,
  },
  textInput: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    minHeight: 120,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  characterCount: {
    fontSize: 12,
    color: '#666',
    textAlign: 'right',
    marginTop: 4,
  },
  controlSection: {
    marginBottom: 24,
  },
  generateButton: {
    backgroundColor: '#007AFF',
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
  },
  generateButtonDisabled: {
    backgroundColor: '#C0C0C0',
  },
  generateButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  audioSection: {
    marginBottom: 24,
  },
  audioCard: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  audioDescription: {
    fontSize: 16,
    color: '#333',
    marginBottom: 16,
    fontStyle: 'italic',
  },

  infoSection: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
    marginBottom: 8,
    marginTop: 16,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  progressContainer: {
    marginTop: 16,
    width: '100%',
    alignItems: 'center',
  },
  progressBar: {
    width: '80%',
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
})

export default function TTSScreen({ navigation }: { navigation: any }) {
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
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  // Set up navigation header button
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            title="Params"
            onPress={() => setShowCompletionParamsModal(true)}
          />
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            title="Context Params"
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
        await llamaContext.initVocoder({ path: vocoderPath })
        setIsVocoderReady(true)
        Alert.alert(
          'Success',
          'OuteTTS model and WavTokenizer vocoder loaded successfully! You can now generate and play speech audio.',
        )
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

      // Get formatted prompt and guide tokens using OuteTTS format
      let formattedPrompt: string
      let guideTokens: number[] | undefined

      try {
        // Try to get formatted prompt if the method exists
        if (
          typeof (context as any).getFormattedAudioCompletion === 'function'
        ) {
          formattedPrompt = await (context as any).getFormattedAudioCompletion(
            speakerConfig,
            inputText.trim(),
          )
        } else {
          formattedPrompt = `[SPEECH]${inputText.trim()}[/SPEECH]`
        }

        // Try to get guide tokens if the method exists
        if (
          typeof (context as any).getAudioCompletionGuideTokens === 'function'
        ) {
          guideTokens = await (context as any).getAudioCompletionGuideTokens(
            inputText.trim(),
          )
        }
      } catch (e) {
        formattedPrompt = `[SPEECH]${inputText.trim()}[/SPEECH]`
      }

      const prompt = formattedPrompt

      const collectedTokens: number[] = []
      const params = completionParams || (await loadCompletionParams())

      const result = await context.completion(
        {
          prompt,
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
      <SafeAreaView style={styles.container}>
        <ScrollView style={styles.setupContainer} contentContainerStyle={styles.scrollContent}>
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

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>
                {context
                  ? 'Initializing vocoder...'
                  : `Initializing TTS model... ${initProgress}%`}
              </Text>
              {!context && initProgress > 0 && (
                <View style={styles.progressContainer}>
                  <View style={styles.progressBar}>
                    <View
                      style={[
                        styles.progressFill,
                        { width: `${initProgress}%` },
                      ]}
                    />
                  </View>
                </View>
              )}
            </View>
          )}
        </ScrollView>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={styles.container}>
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
                <AudioPlayer audio={audioData} sr={sampleRate} />
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

          <Text style={styles.infoTitle}>Tips for Better Results</Text>
          <Text style={styles.infoText}>
            • Use clear, well-punctuated text
            {'\n'}
            • Shorter texts often produce better quality
            {'\n'}
            • Try different speaking styles or emotions
            {'\n'}
            • Avoid special characters and abbreviations
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
    </SafeAreaView>
  )
}
