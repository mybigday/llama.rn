import React, { useCallback, useMemo, useState } from 'react'
import {
  Alert,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import { initLlama } from '../../../src'
import type { NativeCompletionResult } from '../../../src'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import {
  ParameterSwitch,
  ParameterTextInput,
} from '../components/ParameterFormFields'
import { useTheme } from '../contexts/ThemeContext'
import { useExampleContext } from '../hooks/useExampleContext'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import {
  useStoredCompletionParams,
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { createThemedStyles, Spacing } from '../styles/commonStyles'
import type { CompletionParams, ContextParams } from '../utils/storage'
import { createExampleModelDefinitions } from '../utils/exampleModels'

const DEFAULT_PROMPT = [
  'Explain why multi-token prediction can speed up on-device text generation.',
  'Keep the answer concise and include one practical tradeoff.',
].join('\n')

const DEFAULT_DRAFT_TOKENS = 4
const MAX_DRAFT_TOKENS = 32
const DEFAULT_MAX_TOKENS = 256
const MTP_BATCH = 1024
const MTP_UBATCH = 512

const MTP_MODELS = createExampleModelDefinitions(
  ['QWEN_3_6_35B_A3B_MTP'],
  'Initialize MTP Model',
)

function parseBoundedInteger(
  value: string,
  fallback: number,
  min: number,
  max: number,
) {
  const parsed = Number.parseInt(value, 10)
  if (Number.isNaN(parsed)) return fallback
  return Math.max(min, Math.min(max, parsed))
}

function formatRate(accepted?: number, drafted?: number) {
  if (!accepted || !drafted) return '0.0%'
  return `${((accepted / drafted) * 100).toFixed(1)}%`
}

export default function MTPSpeculativeScreen({
  navigation,
}: {
  navigation: any
}) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = createStyles(theme)
  const insets = useSafeAreaInsets()
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [output, setOutput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [draftTokensText, setDraftTokensText] = useState(
    DEFAULT_DRAFT_TOKENS.toString(),
  )
  const [maxTokensText, setMaxTokensText] = useState(
    DEFAULT_MAX_TOKENS.toString(),
  )
  const [isMTPEnabled, setIsMTPEnabled] = useState(true)
  const [lastResult, setLastResult] = useState<NativeCompletionResult | null>(
    null,
  )
  const {
    context,
    initProgress,
    isModelReady,
    replaceContext,
    setInitProgress,
  } = useExampleContext()
  const { value: contextParams, setValue: setContextParams } =
    useStoredContextParams()
  const { value: completionParams, setValue: setCompletionParams } =
    useStoredCompletionParams()
  const { value: customModels, reload: reloadCustomModels } =
    useStoredCustomModels()

  const draftTokens = useMemo(
    () =>
      parseBoundedInteger(
        draftTokensText,
        DEFAULT_DRAFT_TOKENS,
        1,
        MAX_DRAFT_TOKENS,
      ),
    [draftTokensText],
  )
  const maxTokens = useMemo(
    () => parseBoundedInteger(maxTokensText, DEFAULT_MAX_TOKENS, 1, 4096),
    [maxTokensText],
  )

  const handleReset = useCallback(async () => {
    setOutput('')
    setLastResult(null)
    setPrompt(DEFAULT_PROMPT)
    if (context) {
      await context.clearCache(false)
    }
  }, [context])

  useExampleScreenHeader({
    navigation,
    isModelReady,
    readyActions: [
      {
        key: 'reset',
        iconName: 'refresh',
        onPress: handleReset,
      },
      {
        key: 'completion-settings',
        iconName: 'cog-outline',
        onPress: () => setShowCompletionParamsModal(true),
      },
    ],
    setupActions: [
      {
        key: 'context-settings',
        iconName: 'cog-outline',
        onPress: () => setShowContextParamsModal(true),
      },
    ],
  })

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const handleInitModel = async (
    modelUri: string,
    params?: ContextParams,
  ) => {
    setIsLoading(true)
    setInitProgress(0)

    try {
      const baseParams = params || contextParams || {}
      const ctx = await initLlama(
        {
          ...baseParams,
          model: modelUri,
          use_mlock: true,
          n_ctx: baseParams.n_ctx ?? 8192,
          n_batch: MTP_BATCH,
          n_ubatch: MTP_UBATCH,
          n_gpu_layers: baseParams.n_gpu_layers ?? 99,
          speculative: {
            type: 'draft-mtp',
            draft: {
              n_max: MAX_DRAFT_TOKENS,
            },
          },
          spec_draft_n_max: MAX_DRAFT_TOKENS,
        },
        (progress) => {
          setInitProgress(progress)
        },
      )

      await replaceContext(ctx)
      setOutput('')
      setLastResult(null)
      setInitProgress(100)
    } catch (error) {
      Alert.alert('Error', `Failed to initialize MTP model: ${error}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleGenerate = async () => {
    if (!context) {
      Alert.alert('Error', 'Initialize a model before generating.')
      return
    }

    const trimmedPrompt = prompt.trim()
    if (!trimmedPrompt) {
      Alert.alert('Error', 'Please enter a prompt.')
      return
    }

    setIsGenerating(true)
    setOutput('')
    setLastResult(null)

    try {
      const result = await context.completion(
        {
          prompt: trimmedPrompt,
          ...completionParams,
          n_predict: maxTokens,
          speculative: isMTPEnabled
            ? {
                type: 'draft-mtp',
                draft: {
                  n_max: draftTokens,
                },
              }
            : false,
          spec_draft_n_max: isMTPEnabled ? draftTokens : 0,
        },
        (tokenData) => {
          setOutput((current) => current + tokenData.token)
        },
      )

      setLastResult(result)
      const finalText = result.content || result.text
      if (finalText) {
        setOutput(finalText)
      }
    } catch (error) {
      if (error !== 'aborted') {
        Alert.alert('Error', `Failed to generate: ${error}`)
      }
    } finally {
      setIsGenerating(false)
    }
  }

  const handleStop = async () => {
    if (!context) return
    try {
      await context.stopCompletion()
    } catch (error) {
      console.warn('Failed to stop completion:', error)
    }
  }

  if (!isModelReady) {
    return (
      <>
        <ExampleModelSetup
          description="Download an MTP-enabled GGUF model to test self-speculative decoding with native draft token metrics."
          defaultModels={MTP_MODELS}
          defaultModelSectionTitle="MTP Models"
          defaultModelsFirst
          customModels={customModels || []}
          customModelSectionTitle="Custom MTP Models"
          addCustomModelLabel="+ Add Custom MTP Model"
          customModelModalTitle="Add Custom MTP Model"
          onInitializeCustomModel={(_model, modelPath) =>
            handleInitModel(modelPath)
          }
          onInitializeModel={(_model, modelPath) => handleInitModel(modelPath)}
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          isLoading={isLoading}
          initProgress={initProgress}
          progressText={`Initializing MTP model... ${initProgress}%`}
        >
          <View style={styles.setupNote}>
            <Text style={styles.setupNoteTitle}>Requirements</Text>
            <Text style={styles.setupNoteText}>
              MTP works only with text-only models that include draft prediction
              layers. Multimodal prompts are intentionally disabled for this
              demo.
            </Text>
          </View>
        </ExampleModelSetup>

        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={handleSaveContextParams}
        />
      </>
    )
  }

  return (
    <View style={themedStyles.container}>
      <ScrollView
        style={styles.container}
        contentContainerStyle={[
          styles.content,
          { paddingBottom: insets.bottom + Spacing.xl },
        ]}
      >
        <View style={styles.section}>
          <Text style={styles.label}>Prompt</Text>
          <TextInput
            style={[styles.textArea, styles.promptInput]}
            multiline
            value={prompt}
            onChangeText={setPrompt}
            placeholder="Enter a text-only prompt..."
            placeholderTextColor={theme.colors.textSecondary}
            editable={!isGenerating}
            autoCorrect={false}
            autoComplete="off"
            autoCapitalize="none"
            keyboardType="ascii-capable"
          />
        </View>

        <View style={styles.controlsGrid}>
          <View style={styles.controlItem}>
            <ParameterTextInput
              label="Draft Tokens"
              description="Maximum MTP draft tokens proposed before verification."
              value={draftTokensText}
              onChangeText={setDraftTokensText}
              placeholder={DEFAULT_DRAFT_TOKENS.toString()}
              keyboardType="numeric"
            />
          </View>
          <View style={styles.controlItem}>
            <ParameterTextInput
              label="Max Tokens"
              description="Maximum generated tokens for this completion."
              value={maxTokensText}
              onChangeText={setMaxTokensText}
              placeholder={DEFAULT_MAX_TOKENS.toString()}
              keyboardType="numeric"
            />
          </View>
        </View>

        <ParameterSwitch
          label="MTP Speculation"
          description="Disable to run the same initialized model without draft tokens for comparison."
          value={isMTPEnabled}
          onValueChange={setIsMTPEnabled}
        />

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
          >
            <Text style={styles.actionButtonText}>Generate</Text>
          </TouchableOpacity>
        )}

        <View style={styles.section}>
          <Text style={styles.label}>Output</Text>
          <Text style={styles.outputText}>
            {output || 'Generated text will appear here.'}
          </Text>
        </View>

        {lastResult && (
          <View style={styles.metricsPanel}>
            <Text style={styles.metricsTitle}>MTP Metrics</Text>
            <View style={styles.metricsGrid}>
              <MetricItem
                label="Generated"
                value={`${lastResult.tokens_predicted} tokens`}
                styles={styles}
              />
              <MetricItem
                label="Drafted"
                value={`${lastResult.draft_tokens || 0} tokens`}
                styles={styles}
              />
              <MetricItem
                label="Accepted"
                value={`${lastResult.draft_tokens_accepted || 0} tokens`}
                styles={styles}
              />
              <MetricItem
                label="Accept Rate"
                value={formatRate(
                  lastResult.draft_tokens_accepted,
                  lastResult.draft_tokens,
                )}
                styles={styles}
              />
              <MetricItem
                label="Prompt"
                value={`${lastResult.timings.prompt_per_second.toFixed(
                  2,
                )} t/s`}
                styles={styles}
              />
              <MetricItem
                label="Generation"
                value={`${lastResult.timings.predicted_per_second.toFixed(
                  2,
                )} t/s`}
                styles={styles}
              />
            </View>
          </View>
        )}
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
    </View>
  )
}

function MetricItem({
  label,
  value,
  styles,
}: {
  label: string
  value: string
  styles: ReturnType<typeof createStyles>
}) {
  return (
    <View style={styles.metricItem}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  )
}

function createStyles(theme: any) {
  return StyleSheet.create({
    container: {
      flex: 1,
    },
    content: {
      padding: Spacing.lg,
    },
    setupNote: {
      backgroundColor: theme.colors.surface,
      borderColor: theme.colors.border,
      borderWidth: 1,
      borderRadius: Spacing.sm,
      padding: Spacing.lg,
      marginBottom: Spacing.xl,
    },
    setupNoteTitle: {
      color: theme.colors.text,
      fontSize: 16,
      fontWeight: '700',
      marginBottom: Spacing.xs,
    },
    setupNoteText: {
      color: theme.colors.textSecondary,
      fontSize: 14,
      lineHeight: 20,
    },
    section: {
      backgroundColor: theme.colors.surface,
      borderRadius: Spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
      padding: Spacing.lg,
      marginBottom: Spacing.md,
    },
    label: {
      color: theme.colors.text,
      fontSize: 16,
      fontWeight: '700',
      marginBottom: Spacing.sm,
    },
    textArea: {
      backgroundColor: theme.colors.inputBackground,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: Spacing.sm,
      color: theme.colors.text,
      fontSize: 15,
      lineHeight: 21,
      padding: Spacing.md,
      textAlignVertical: 'top',
    },
    promptInput: {
      minHeight: 140,
    },
    controlsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: Spacing.md,
    },
    controlItem: {
      flexGrow: 1,
      flexBasis: 220,
    },
    actionButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: Spacing.sm,
      paddingVertical: Spacing.md,
      alignItems: 'center',
      marginBottom: Spacing.md,
    },
    stopButton: {
      backgroundColor: theme.colors.error,
    },
    actionButtonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '700',
    },
    outputText: {
      color: theme.colors.text,
      fontSize: 15,
      lineHeight: 22,
      minHeight: 120,
    },
    metricsPanel: {
      backgroundColor: theme.colors.surface,
      borderRadius: Spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
      padding: Spacing.lg,
    },
    metricsTitle: {
      color: theme.colors.text,
      fontSize: 16,
      fontWeight: '700',
      marginBottom: Spacing.md,
    },
    metricsGrid: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: Spacing.sm,
    },
    metricItem: {
      backgroundColor: theme.colors.card,
      borderRadius: Spacing.sm,
      paddingHorizontal: Spacing.md,
      paddingVertical: Spacing.sm,
      minWidth: 120,
      flexGrow: 1,
    },
    metricLabel: {
      color: theme.colors.textSecondary,
      fontSize: 12,
      marginBottom: 2,
    },
    metricValue: {
      color: theme.colors.text,
      fontSize: 14,
      fontWeight: '700',
    },
  })
}
