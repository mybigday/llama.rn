import React, {
  useState,
  useEffect,
  useRef,
  useLayoutEffect,
  useCallback,
} from 'react'
import {
  View,
  Text,
  TextInput,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Modal,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import type { TokenData } from '../../../src'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import CustomModelModal from '../components/CustomModelModal'
import CustomModelCard from '../components/CustomModelCard'
import { HeaderButton } from '../components/HeaderButton'
import { MaskedProgress } from '../components/MaskedProgress'
import { ParameterTextInput } from '../components/ParameterFormFields'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type {
  ContextParams,
  CompletionParams,
  CustomModel,
} from '../utils/storage'
import {
  loadContextParams,
  loadCompletionParams,
  loadCustomModels,
} from '../utils/storage'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

interface ProbabilityDropdownProps {
  token: TokenData
  position: { x: number; y: number }
  onClose: () => void
}

function ProbabilityDropdown({
  token,
  position,
  onClose,
}: ProbabilityDropdownProps) {
  const localStyles = {
    probabilityDropdown: {
      position: 'absolute' as 'absolute',
      backgroundColor: '#fff',
      borderRadius: 8,
      borderWidth: 1,
      borderColor: '#ccc',
      padding: 8,
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.2,
      shadowRadius: 4,
      elevation: 5,
      zIndex: 1000,
      minWidth: 200,
    },
    probabilityItem: {
      flexDirection: 'row' as 'row',
      justifyContent: 'space-between' as 'space-between',
      paddingVertical: 4,
      paddingHorizontal: 8,
    },
    probabilityToken: {
      fontFamily: 'monospace',
      fontSize: 12,
      flex: 1,
    },
    probabilityValue: {
      fontFamily: 'monospace',
      fontSize: 12,
      fontWeight: '600' as '600',
      marginLeft: 8,
    },
  }

  return (
    <Modal transparent visible animationType="fade" onRequestClose={onClose}>
      <TouchableOpacity style={{ flex: 1 }} activeOpacity={1} onPress={onClose}>
        <View
          style={[
            localStyles.probabilityDropdown,
            { top: position.y, left: position.x },
          ]}
        >
          {token.completion_probabilities?.[0]?.probs?.map((prob, idx) => (
            <View key={idx} style={localStyles.probabilityItem}>
              <Text style={localStyles.probabilityToken}>
                {JSON.stringify(prob.tok_str)}
              </Text>
              <Text style={localStyles.probabilityValue}>
                {`${(prob.prob * 100).toFixed(2)}%`}
              </Text>
            </View>
          ))}
        </View>
      </TouchableOpacity>
    </Modal>
  )
}

export default function TextCompletionScreen({
  navigation,
}: {
  navigation: any
}) {
  const { theme, isDark } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const styles = StyleSheet.create({
    container: themedStyles.container,
    setupContainer: themedStyles.setupContainer,
    scrollContent: themedStyles.scrollContent,
    setupDescription: themedStyles.setupDescription,
    contentContainer: {
      flex: 1,
      padding: 10,
    },
    textAreaContainer: {
      flex: 1,
      marginBottom: 10,
    },
    textInput: {
      ...themedStyles.textInput,
      height: 120,
      textAlignVertical: 'top',
    },
    outputContainer: {
      flex: 1,
      backgroundColor: theme.colors.inputBackground,
      borderColor: theme.colors.border,
      borderWidth: 1,
      borderRadius: 8,
      padding: 10,
      marginTop: 10,
    },
    outputText: {
      color: theme.colors.text,
      fontSize: 16,
      lineHeight: 22,
    },
    tokenText: {
      color: theme.colors.textSecondary,
      fontSize: 12,
      fontFamily: 'monospace',
      margin: 2,
    },
    label: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 5,
    },
    textArea: {
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 8,
      padding: 10,
      fontSize: 14,
      minHeight: 60,
      textAlignVertical: 'top',
      backgroundColor: theme.colors.inputBackground,
      color: theme.colors.text,
    },
    promptTextArea: {
      fontFamily: 'monospace',
      backgroundColor: theme.colors.inputBackground,
    },
    editButtonContainer: {
      marginTop: 10,
      alignItems: 'flex-end',
    },
    editButton: {
      backgroundColor: theme.colors.primary,
      paddingHorizontal: 12,
      paddingVertical: 6,
      borderRadius: 6,
    },
    editButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '600',
    },
    tokenContainer: {
      flexDirection: 'row',
      flexWrap: 'wrap',
    },
    token: {
      padding: 2,
      borderRadius: 2,
      margin: 1,
    },
    actionButton: {
      backgroundColor: theme.colors.primary,
      padding: 12,
      borderRadius: 8,
      alignItems: 'center',
      marginVertical: 10,
    },
    actionButtonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
    },
    stopButton: {
      backgroundColor: theme.colors.error,
    },
  })
  const [prompt, setPrompt] = useState('')
  const [grammar, setGrammar] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [completionParams, setCompletionParams] =
    useState<CompletionParams | null>(null)
  const [customModels, setCustomModels] = useState<CustomModel[]>([])
  const [tokens, setTokens] = useState<TokenData[]>([])
  const [selectedToken, setSelectedToken] = useState<{
    token: TokenData
    position: { x: number; y: number }
  } | null>(null)
  const [isEditingResult, setIsEditingResult] = useState(false)
  const [editableResult, setEditableResult] = useState('')
  const [formattedPrompt, setFormattedPrompt] = useState('')
  const [promptTokens, setPromptTokens] = useState<string[]>([])
  const abortControllerRef = useRef<AbortController | null>(null)
  const insets = useSafeAreaInsets()

  useEffect(
    () => () => {
      if (context) {
        context.release()
      }
    },
    [context],
  )

  useEffect(() => {
    const loadCustomModelsData = async () => {
      try {
        const models = await loadCustomModels()
        setCustomModels(models)
      } catch (error) {
        console.error('Error loading custom models:', error)
      }
    }
    loadCustomModelsData()
  }, [])

  useEffect(() => {
    const loadParams = async () => {
      try {
        const ctxParams = await loadContextParams()
        const compParams = await loadCompletionParams()
        setContextParams(ctxParams)
        setCompletionParams({
          ...compParams,
          n_probs: 5,
        })
      } catch (error) {
        console.error('Error loading params:', error)
      }
    }
    loadParams()
  }, [])

  const handleReset = useCallback(() => {
    Alert.alert(
      'Reset Text Completion',
      'Are you sure you want to clear all generated text and reset to default prompt? This action cannot be undone.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            setTokens([])
            setIsEditingResult(false)
            setEditableResult('')
            if (context) {
              await context.clearCache(false)
              try {
                const defaultMessages = [
                  {
                    role: 'system' as const,
                    content: 'You are a helpful AI assistant.',
                  },
                  {
                    role: 'user' as const,
                    content: 'Hello! Please introduce yourself.',
                  },
                ]
                const formatted = await context.getFormattedChat(
                  defaultMessages,
                  null,
                )
                setPrompt(formatted.prompt)
                setFormattedPrompt(formatted.prompt)
                const tokenized = await context.tokenize(formatted.prompt)
                const detokenizedTokens = await Promise.all(
                  tokenized.tokens.map((token) => context.detokenize([token])),
                )
                setPromptTokens(detokenizedTokens)
              } catch (formatError) {
                console.warn('Failed to format default prompt:', formatError)
                const fallbackPrompt = 'Hello! Please introduce yourself.'
                setPrompt(fallbackPrompt)
                setFormattedPrompt(fallbackPrompt)
                setPromptTokens([])
              }
            }
          },
        },
      ],
    )
  }, [context])

  // Set up navigation header buttons
  useLayoutEffect(() => {
    if (isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', gap: 8 }}>
            <HeaderButton iconName="refresh" onPress={handleReset} />
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
  }, [navigation, isModelReady, handleReset])

  const handleInitModel = async (modelUri: string, params?: ContextParams) => {
    setIsLoading(true)
    setInitProgress(0)
    try {
      const ctx = await initLlama(
        {
          model: modelUri,
          use_mlock: true,
          ...(params || contextParams || {}),
        },
        (progress) => {
          setInitProgress(progress)
        },
      )
      console.log('Model initialized')
      setContext(ctx)
      setIsModelReady(true)
      setInitProgress(100)

      // Set default system/user message format as initial prompt
      const defaultMessages = [
        { role: 'system' as const, content: 'You are a helpful AI assistant.' },
        { role: 'user' as const, content: 'Hello! Please introduce yourself.' },
      ]
      try {
        const formatted = await ctx.getFormattedChat(defaultMessages, null)
        setPrompt(formatted.prompt)
        setFormattedPrompt(formatted.prompt) // Store the formatted prompt
        // Tokenize the prompt to show individual tokens
        const tokenized = await ctx.tokenize(formatted.prompt)
        const detokenizedTokens = await Promise.all(
          tokenized.tokens.map((token) => ctx.detokenize([token])),
        )
        setPromptTokens(detokenizedTokens)
      } catch (formatError) {
        console.warn('Failed to format default prompt:', formatError)
        const fallbackPrompt = 'Hello! Please introduce yourself.'
        setPrompt(fallbackPrompt)
        setFormattedPrompt(fallbackPrompt)
      }
    } catch (error) {
      Alert.alert('Error', `Failed to initialize model: ${error}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams({
      ...params,
      n_probs: params.n_probs || 5,
    })
  }

  const getTokenColor = (prob?: number): string => {
    if (prob === undefined) return 'transparent'
    const normalizedProb = Math.max(0, Math.min(1, prob))
    const red = Math.round(255 * (1 - normalizedProb))
    const green = Math.round(255 * normalizedProb)
    return `rgba(${red}, ${green}, 0, 0.3)`
  }

  const handleGenerate = async () => {
    if (!context || !formattedPrompt) {
      Alert.alert('Error', 'Please enter a prompt')
      return
    }

    setIsGenerating(true)
    setIsEditingResult(false)
    const controller = new AbortController()
    abortControllerRef.current = controller

    // Use the complete text (formatted prompt + any existing generated tokens) as the prompt
    const existingGenerated = tokens.map((t) => t.token).join('')
    const fullPrompt = formattedPrompt + existingGenerated

    try {
      await context.completion(
        {
          prompt: fullPrompt,
          grammar: grammar || undefined,
          n_probs: 5,
          ...completionParams,
        },
        (tokenData) => {
          setTokens((prev) => [...prev, tokenData])
        },
      )
      console.log('Completion finished')
    } catch (error) {
      if (error !== 'aborted') {
        Alert.alert('Error', `Failed to generate: ${error}`)
      }
    } finally {
      setIsGenerating(false)
      abortControllerRef.current = null
    }
  }

  const handleStop = async () => {
    if (context) {
      try {
        await context.stopCompletion()
      } catch (error) {
        console.warn('Failed to stop completion:', error)
      }
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }

  const handleTokenPress = (event: any, token: TokenData) => {
    if (!token.completion_probabilities?.[0]?.probs) return

    event.currentTarget.measure(
      (
        _x: number,
        _y: number,
        _width: number,
        _height: number,
        pageX: number,
        pageY: number,
      ) => {
        setSelectedToken({
          token,
          position: { x: pageX, y: pageY + _height },
        })
      },
    )
  }

  const renderContent = () => {
    if (!isModelReady) {
      return (
        <View style={styles.setupContainer}>
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.setupDescription}>
              Select a model to start text completion
            </Text>
            {/* Custom Models Section */}
            {customModels.length > 0 && (
              <>
                <Text style={themedStyles.modelSectionTitle}>
                  Custom Models
                </Text>
                {customModels.map((model) => (
                  <CustomModelCard
                    key={model.id}
                    model={model}
                    onInitialize={(modelPath) => handleInitModel(modelPath)}
                    onModelRemoved={async () => {
                      const models = await loadCustomModels()
                      setCustomModels(models)
                    }}
                    initializeButtonText="Initialize"
                  />
                ))}
              </>
            )}

            {/* Add Custom Model Button */}
            <TouchableOpacity
              style={themedStyles.addCustomModelButton}
              onPress={() => setShowCustomModelModal(true)}
            >
              <Text style={themedStyles.addCustomModelButtonText}>
                + Add Custom Model
              </Text>
            </TouchableOpacity>

            {/* Predefined Models Section */}
            <Text style={themedStyles.modelSectionTitle}>Default Models</Text>
            {[
              'SMOL_LM_3',
              'GEMMA_3_4B_QAT',
              'QWEN_3_4B',
              'GEMMA_3N_E2B',
              'GEMMA_3N_E4B',
            ].map((modelKey) => {
              const model = MODELS[modelKey as keyof typeof MODELS]
              return (
                <ModelDownloadCard
                  key={modelKey}
                  title={model.name}
                  repo={model.repo}
                  filename={model.filename}
                  size={model.size}
                  onInitialize={(modelPath) => handleInitModel(modelPath)}
                />
              )
            })}
          </ScrollView>

          <MaskedProgress
            visible={isLoading}
            text={`Initializing model... ${initProgress}%`}
            progress={initProgress}
            showProgressBar={initProgress > 0}
          />
        </View>
      )
    }

    return (
      <ScrollView style={styles.contentContainer}>
        <View style={styles.textAreaContainer}>
          <Text style={styles.label}>Text Completion</Text>
          {(() => {
            if (isEditingResult) {
              return (
                <View>
                  <TextInput
                    style={[
                      styles.textArea,
                      styles.promptTextArea,
                      { minHeight: 200 },
                    ]}
                    multiline
                    scrollEnabled={false}
                    value={editableResult}
                    onChangeText={setEditableResult}
                    placeholder="Edit complete text (formatted prompt + generated)..."
                    autoFocus
                    keyboardType="ascii-capable"
                  />
                  <View style={styles.editButtonContainer}>
                    <TouchableOpacity
                      style={styles.editButton}
                      onPress={async () => {
                        // Check if the text has actually changed before re-tokenizing
                        const originalText =
                          formattedPrompt + tokens.map((t) => t.token).join('')
                        const hasChanges = editableResult !== originalText

                        if (context && editableResult && hasChanges) {
                          try {
                            // Tokenize the entire edited text to show as prompt tokens
                            const tokenized = await context.tokenize(
                              editableResult,
                            )
                            const detokenizedTokens = await Promise.all(
                              tokenized.tokens.map((token) =>
                                context.detokenize([token]),
                              ),
                            )
                            setPromptTokens(detokenizedTokens)

                            // Update the prompt and formatted prompt
                            setPrompt(editableResult)
                            setFormattedPrompt(editableResult)

                            // Clear the generated tokens since we're now showing everything as prompt
                            setTokens([])
                          } catch (error) {
                            console.warn(
                              'Failed to tokenize edited text:',
                              error,
                            )
                          }
                        }
                        setIsEditingResult(false)
                      }}
                    >
                      <Text style={styles.editButtonText}>Done</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )
            }

            return (
              <View style={[styles.textArea, styles.promptTextArea]}>
                <View style={styles.tokenContainer}>
                  {promptTokens.map((token, index) => (
                    <View
                      key={`prompt-${index}`}
                      style={[
                        styles.token,
                        { backgroundColor: isDark ? '#3C3C3C' : '#e3f2fd' },
                      ]}
                    >
                      <Text style={styles.tokenText}>
                        {token?.includes('\n')
                          ? token.replaceAll('\n', '\\n')
                          : token}
                      </Text>
                    </View>
                  ))}
                  {tokens.map((token, index) => {
                    const firstProb =
                      token.completion_probabilities?.[0]?.probs?.[0]?.prob
                    return (
                      <TouchableOpacity
                        key={`gen-${index}`}
                        onPress={(e) => handleTokenPress(e, token)}
                        style={[
                          styles.token,
                          { backgroundColor: getTokenColor(firstProb) },
                        ]}
                      >
                        <Text style={styles.tokenText}>
                          {token.token?.includes('\n')
                            ? token.token.replaceAll('\n', '\\n')
                            : token.token}
                        </Text>
                      </TouchableOpacity>
                    )
                  })}
                </View>
                <View style={styles.editButtonContainer}>
                  <TouchableOpacity
                    style={styles.editButton}
                    onPress={() => {
                      // Combine formatted prompt + generated text for editing
                      const fullText =
                        formattedPrompt + tokens.map((t) => t.token).join('')
                      setEditableResult(fullText)
                      setIsEditingResult(true)
                    }}
                  >
                    <Text style={styles.editButtonText}>Edit</Text>
                  </TouchableOpacity>
                </View>
              </View>
            )
          })()}
        </View>

        <View style={styles.textAreaContainer}>
          <Text style={styles.label}>Grammar (optional)</Text>
          <TextInput
            style={styles.textArea}
            multiline
            scrollEnabled={false}
            value={grammar}
            onChangeText={setGrammar}
            placeholder="Enter grammar rules (GBNF format)..."
            editable={!isGenerating}
            autoCorrect={false}
            autoComplete="off"
            autoCapitalize="none"
            keyboardType="ascii-capable"
          />
        </View>

        {isGenerating ? (
          <TouchableOpacity
            style={[styles.actionButton, styles.stopButton]}
            onPress={handleStop}
          >
            <Text style={styles.actionButtonText}>Stop Generation</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.actionButton}
            onPress={handleGenerate}
            disabled={!prompt}
          >
            <Text style={styles.actionButtonText}>Generate</Text>
          </TouchableOpacity>
        )}

        {/* Completion Sampling Parameters */}
        <View style={styles.textAreaContainer}>
          <Text style={styles.label}>Additional Sampling Parameters</Text>

          <ParameterTextInput
            label="Top K"
            description="Limits selection to the K most probable tokens. Lower values are more focused."
            value={completionParams?.top_k?.toString() || '40'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({ ...prev!, top_k: undefined }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({ ...prev!, top_k: value }))
                }
              }
            }}
            placeholder="40"
            keyboardType="numeric"
          />

          <ParameterTextInput
            label="Min P"
            description="Minimum probability threshold relative to the most likely token. Higher values filter more tokens."
            value={completionParams?.min_p?.toString() || '0.05'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({ ...prev!, min_p: undefined }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({ ...prev!, min_p: value }))
                }
              }
            }}
            placeholder="0.05"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Typical P"
            description="Locally typical sampling parameter. 1.0 disables, lower values increase selectivity."
            value={completionParams?.typical_p?.toString() || '1.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  typical_p: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    typical_p: value,
                  }))
                }
              }
            }}
            placeholder="1.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Repeat Penalty"
            description="Controls repetition of token sequences. 1.0 is no penalty, higher values reduce repetition."
            value={completionParams?.penalty_repeat?.toString() || '1.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  penalty_repeat: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    penalty_repeat: value,
                  }))
                }
              }
            }}
            placeholder="1.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="XTC Probability"
            description="Chance for token removal via XTC sampler. 0.0 disables, higher values increase diversity."
            value={completionParams?.xtc_probability?.toString() || '0.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  xtc_probability: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    xtc_probability: value,
                  }))
                }
              }
            }}
            placeholder="0.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="XTC Threshold"
            description="Minimum probability threshold for XTC sampler. Values > 0.5 effectively disable XTC."
            value={completionParams?.xtc_threshold?.toString() || '0.1'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  xtc_threshold: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    xtc_threshold: value,
                  }))
                }
              }
            }}
            placeholder="0.1"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Penalty Last N"
            description="Last N tokens to consider for repetition penalty. 0 disables, -1 uses context size."
            value={completionParams?.penalty_last_n?.toString() || '64'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  penalty_last_n: undefined,
                }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    penalty_last_n: value,
                  }))
                }
              }
            }}
            placeholder="64"
            keyboardType="numeric"
          />

          <ParameterTextInput
            label="Frequency Penalty"
            description="Repeat alpha frequency penalty. Higher values reduce repetition of frequent tokens."
            value={completionParams?.penalty_freq?.toString() || '0.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  penalty_freq: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    penalty_freq: value,
                  }))
                }
              }
            }}
            placeholder="0.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Presence Penalty"
            description="Repeat alpha presence penalty. Higher values reduce repetition regardless of frequency."
            value={completionParams?.penalty_present?.toString() || '0.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  penalty_present: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    penalty_present: value,
                  }))
                }
              }
            }}
            placeholder="0.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Mirostat Mode"
            description="Mirostat sampling mode. 0=disabled, 1=Mirostat v1, 2=Mirostat v2."
            value={completionParams?.mirostat?.toString() || '0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  mirostat: undefined,
                }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({ ...prev!, mirostat: value }))
                }
              }
            }}
            placeholder="0"
            keyboardType="numeric"
          />

          <ParameterTextInput
            label="Mirostat Tau"
            description="Mirostat target entropy (tau). Controls perplexity during generation."
            value={completionParams?.mirostat_tau?.toString() || '5.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  mirostat_tau: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    mirostat_tau: value,
                  }))
                }
              }
            }}
            placeholder="5.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Mirostat Eta"
            description="Mirostat learning rate (eta). Controls how quickly the algorithm adapts."
            value={completionParams?.mirostat_eta?.toString() || '0.1'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  mirostat_eta: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    mirostat_eta: value,
                  }))
                }
              }
            }}
            placeholder="0.1"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="DRY Multiplier"
            description="DRY (Don't Repeat Yourself) repetition penalty multiplier. 0.0 disables DRY sampling."
            value={completionParams?.dry_multiplier?.toString() || '0.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  dry_multiplier: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    dry_multiplier: value,
                  }))
                }
              }
            }}
            placeholder="0.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="DRY Base"
            description="DRY repetition penalty base value. Used as exponential base for penalty calculation."
            value={completionParams?.dry_base?.toString() || '1.75'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  dry_base: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({ ...prev!, dry_base: value }))
                }
              }
            }}
            placeholder="1.75"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="DRY Allowed Length"
            description="Tokens extending repetition beyond this length receive exponentially increasing penalty."
            value={completionParams?.dry_allowed_length?.toString() || '2'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  dry_allowed_length: undefined,
                }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    dry_allowed_length: value,
                  }))
                }
              }
            }}
            placeholder="2"
            keyboardType="numeric"
          />

          <ParameterTextInput
            label="DRY Penalty Last N"
            description="How many tokens to scan for DRY repetitions. -1 uses context size, 0 disables."
            value={completionParams?.dry_penalty_last_n?.toString() || '-1'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  dry_penalty_last_n: undefined,
                }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    dry_penalty_last_n: value,
                  }))
                }
              }
            }}
            placeholder="-1"
            keyboardType="numeric"
          />

          <ParameterTextInput
            label="Top N Sigma"
            description="Top-nσ sampling as described in academic paper. -1.0 disables, lower values increase selectivity."
            value={completionParams?.top_n_sigma?.toString() || '-1.0'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({
                  ...prev!,
                  top_n_sigma: undefined,
                }))
              } else {
                const value = parseFloat(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({
                    ...prev!,
                    top_n_sigma: value,
                  }))
                }
              }
            }}
            placeholder="-1.0"
            keyboardType="decimal-pad"
          />

          <ParameterTextInput
            label="Seed"
            description="Random number generator seed. -1 for random seed, any other number for reproducible results."
            value={completionParams?.seed?.toString() || '-1'}
            onChangeText={(text) => {
              if (text === '') {
                setCompletionParams((prev) => ({ ...prev!, seed: undefined }))
              } else {
                const value = parseInt(text)
                if (!Number.isNaN(value)) {
                  setCompletionParams((prev) => ({ ...prev!, seed: value }))
                }
              }
            }}
            placeholder="-1"
            keyboardType="numeric"
          />
        </View>
      </ScrollView>
    )
  }

  return (
    <View style={[styles.container, { paddingBottom: insets.bottom }]}>
      {renderContent()}

      {selectedToken && (
        <ProbabilityDropdown
          token={selectedToken.token}
          position={selectedToken.position}
          onClose={() => setSelectedToken(null)}
        />
      )}

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

      <CustomModelModal
        visible={showCustomModelModal}
        onClose={() => setShowCustomModelModal(false)}
        onModelAdded={async () => {
          const models = await loadCustomModels()
          setCustomModels(models)
        }}
        title="Add Custom Model"
        enableFileSelection
      />
    </View>
  )
}
