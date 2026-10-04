import React, { useState } from 'react'
import {
  View,
  Text,
  ScrollView,
  Alert,
  TextInput,
  TouchableOpacity,
  StyleSheet,
} from 'react-native'
import Icon from '@react-native-vector-icons/material-design-icons'
import ContextParamsModal from '../components/ContextParamsModal'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import { QuestionEditor } from '../components/decision/QuestionEditor'
import { AnswerCard } from '../components/decision/AnswerCard'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import { loadContextParams } from '../utils/storage'
import {
  initLlama,
  type DecisionRequest,
  type DecisionResult,
  type NativeLlamaContext,
} from '../../../src' // import 'llama.rn'
import {
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { useExampleContext } from '../hooks/useExampleContext'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import {
  createExampleModelDefinitions,
  isDecisionModel,
  type ExampleModelKey,
} from '../utils/exampleModels'
import {
  DECISION_PRESETS,
  buildDecisionRequest,
  createQuestion,
  type EditableQuestion,
} from '../features/decisionHelpers'

const DECISION_MODELS = createExampleModelDefinitions(
  (Object.keys(MODELS) as ExampleModelKey[]).filter((key) =>
    isDecisionModel(MODELS[key]),
  ),
)

const N_PARALLEL = 2

type RunMode = 'single' | 'parallel'
type Tab = 'questions' | 'results'

type RunOutcome = {
  mode: RunMode
  ms: number
  request: DecisionRequest
  result?: DecisionResult
  error?: string
}

type DecisionInfo = NonNullable<NativeLlamaContext['model']['decision']>

export default function DecisionScreen({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = createStyles(theme, themedStyles)

  const [isLoading, setIsLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [modelName, setModelName] = useState('')
  const [decisionInfo, setDecisionInfo] = useState<DecisionInfo | null>(null)

  const [preset] = useState(() => DECISION_PRESETS[0]!())
  const [presetName, setPresetName] = useState(preset.name)
  const [state, setState] = useState(preset.state)
  const [questions, setQuestions] = useState<EditableQuestion[]>(
    preset.questions,
  )
  const [mode, setMode] = useState<RunMode>('single')
  const [tab, setTab] = useState<Tab>('questions')
  const [outcome, setOutcome] = useState<RunOutcome | null>(null)

  const { context, initProgress, isModelReady, replaceContext, setInitProgress } =
    useExampleContext()
  const { setValue: setContextParams } = useStoredContextParams()
  const { value: customModels, reload: reloadCustomModels } =
    useStoredCustomModels()

  useExampleScreenHeader({
    navigation,
    isModelReady,
    readyActions: [],
    setupActions: [
      {
        key: 'context-settings',
        iconName: 'cog-outline',
        onPress: () => setShowContextParamsModal(true),
      },
    ],
  })

  const initializeModel = async (modelPath: string, title?: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)
      const params = await loadContextParams()
      const llamaContext = await initLlama(
        { ...params, model: modelPath, n_parallel: N_PARALLEL },
        (progress) => setInitProgress(progress),
      )
      const { decision } = llamaContext.model
      if (!decision || decision.type === 'unknown') {
        await llamaContext.release()
        Alert.alert(
          'Not a usable decision model',
          decision
            ? 'This model declares a decision type that this build does not support.'
            : 'This model has no decision metadata (<arch>.decision.type).',
        )
        return
      }
      await replaceContext(llamaContext)
      setModelName(title || modelPath.split('/').pop() || 'Model')
      setDecisionInfo(decision)
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const applyPreset = (index: number) => {
    const next = DECISION_PRESETS[index]!()
    setPresetName(next.name)
    setState(next.state)
    setQuestions(next.questions)
  }

  const clearQuestions = () => {
    if (questions.length === 0) return
    Alert.alert('Clear all questions?', undefined, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear',
        style: 'destructive',
        onPress: () => {
          setQuestions([])
          setPresetName('')
        },
      },
    ])
  }

  const run = async () => {
    if (!context || isRunning) return
    let request: DecisionRequest
    try {
      request = buildDecisionRequest(state, questions)
    } catch (error: any) {
      Alert.alert('Incomplete request', error.message)
      return
    }

    setIsRunning(true)
    const t0 = Date.now()
    try {
      let result: DecisionResult
      if (mode === 'parallel') {
        await context.parallel.enable({ n_parallel: N_PARALLEL })
        try {
          const { promise } = await context.parallel.decide(request)
          result = await promise
        } finally {
          await context.parallel.disable()
        }
      } else {
        result = await context.decide(request)
      }
      const ms = Date.now() - t0
      console.log(`[Decision] ${mode} ${ms}ms ${JSON.stringify(result)}`)
      setOutcome({ mode, ms, request, result })
    } catch (error: any) {
      console.log(`[Decision] error ${error.message}`)
      setOutcome({ mode, ms: Date.now() - t0, request, error: error.message })
    } finally {
      setIsRunning(false)
      setTab('results')
    }
  }

  if (!isModelReady) {
    return (
      <>
        <ExampleModelSetup
          description="Typed decision models answer choice, score and yes/no questions about a state with calibrated probabilities, in one forward pass per question instead of generating text."
          defaultModels={DECISION_MODELS}
          customModels={customModels || []}
          onInitializeCustomModel={(model, modelPath) =>
            initializeModel(modelPath, model.filename)
          }
          onInitializeModel={(model, modelPath) =>
            initializeModel(modelPath, model.title)
          }
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          customModelModalTitle="Add Custom Decision Model"
          isLoading={isLoading}
          initProgress={initProgress}
          progressText={`Initializing model... ${initProgress}%`}
        />
        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={setContextParams}
        />
      </>
    )
  }

  const renderSegmented = <T extends string>(
    items: Array<{ value: T; label: string }>,
    value: T,
    onChange: (value: T) => void,
  ) => (
    <View style={styles.segmented}>
      {items.map((item) => (
        <TouchableOpacity
          key={item.value}
          style={[styles.segment, item.value === value && styles.segmentSelected]}
          onPress={() => onChange(item.value)}
        >
          <Text
            style={[
              styles.segmentText,
              item.value === value && styles.segmentTextSelected,
            ]}
          >
            {item.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  )

  const renderQuestionsTab = () => (
    <>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.presets}
        contentContainerStyle={styles.presetsContent}
      >
        {DECISION_PRESETS.map((makePreset, index) => {
          const { name } = makePreset()
          const selected = name === presetName
          return (
            <TouchableOpacity
              key={name}
              style={[styles.presetChip, selected && styles.presetChipSelected]}
              onPress={() => applyPreset(index)}
            >
              <Text
                style={[
                  styles.presetText,
                  selected && styles.presetTextSelected,
                ]}
              >
                {name}
              </Text>
            </TouchableOpacity>
          )
        })}
      </ScrollView>

      <Text style={styles.sectionTitle}>State</Text>
      <TextInput
        style={styles.stateInput}
        value={state}
        onChangeText={setState}
        placeholder="What the questions are about: text, or JSON"
        placeholderTextColor={theme.colors.textSecondary}
        multiline
        autoCorrect={false}
      />

      <View style={styles.questionsHeader}>
        <Text style={styles.sectionTitle}>{`Questions (${questions.length})`}</Text>
        <TouchableOpacity
          style={styles.textButton}
          onPress={clearQuestions}
          disabled={questions.length === 0}
        >
          <Icon
            name="playlist-remove"
            size={18}
            color={
              questions.length === 0 ? theme.colors.disabled : theme.colors.error
            }
          />
          <Text
            style={[
              styles.textButtonLabel,
              {
                color:
                  questions.length === 0
                    ? theme.colors.disabled
                    : theme.colors.error,
              },
            ]}
          >
            Clear all
          </Text>
        </TouchableOpacity>
      </View>

      {questions.length === 0 && (
        <Text style={styles.emptyText}>
          No questions yet. Pick an example above or add one.
        </Text>
      )}
      {questions.map((question, index) => (
        <QuestionEditor
          key={question.uid}
          index={index}
          question={question}
          onChange={(next) =>
            setQuestions((prev) =>
              prev.map((q) => (q.uid === next.uid ? next : q)),
            )
          }
          onRemove={() =>
            setQuestions((prev) => prev.filter((q) => q.uid !== question.uid))
          }
        />
      ))}
      <TouchableOpacity
        style={styles.addQuestionButton}
        onPress={() => setQuestions((prev) => [...prev, createQuestion()])}
      >
        <Icon name="plus" size={18} color={theme.colors.primary} />
        <Text style={styles.addQuestionText}>Add question</Text>
      </TouchableOpacity>
    </>
  )

  const renderResultsTab = () => {
    if (!outcome) {
      return (
        <Text style={styles.emptyText}>
          Tap Decide to answer the questions.
        </Text>
      )
    }
    const { result, error, request } = outcome
    return (
      <>
        <View style={styles.summary}>
          <View style={styles.summaryItem}>
            <Icon name="timer-outline" size={16} color={theme.colors.textSecondary} />
            <Text style={styles.summaryText}>{`${outcome.ms} ms`}</Text>
          </View>
          {result && (
            <View style={styles.summaryItem}>
              <Icon name="text-box-outline" size={16} color={theme.colors.textSecondary} />
              <Text style={styles.summaryText}>
                {`${result.usage.input_tokens} tokens`}
              </Text>
            </View>
          )}
          <View style={styles.summaryItem}>
            <Icon
              name={outcome.mode === 'parallel' ? 'call-split' : 'arrow-right'}
              size={16}
              color={theme.colors.textSecondary}
            />
            <Text style={styles.summaryText}>
              {outcome.mode === 'parallel' ? 'parallel.decide()' : 'decide()'}
            </Text>
          </View>
        </View>
        {error && (
          <View style={styles.errorCard}>
            <Icon name="alert-circle-outline" size={20} color={theme.colors.error} />
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        {result &&
          Object.entries(result.answers).map(([id, answer]) => {
            const question = request.questions[id]!
            return (
              <AnswerCard
                key={id}
                id={id}
                instructions={
                  typeof question.instructions === 'string'
                    ? question.instructions
                    : JSON.stringify(question.instructions)
                }
                answer={answer}
                optionDescriptions={
                  question.type === 'choice' ? question.criteria : undefined
                }
              />
            )
          })}
      </>
    )
  }

  return (
    <View style={styles.container}>
      <View style={styles.modelBar}>
        <Icon name="bullseye-arrow" size={18} color={theme.colors.primary} />
        <Text style={styles.modelName} numberOfLines={1}>
          {modelName}
        </Text>
        {decisionInfo && (
          <Text style={styles.modelChip}>
            {`${decisionInfo.type} · ≤${decisionInfo.nOptionsMax} options`}
          </Text>
        )}
      </View>

      <View style={styles.tabs}>
        {renderSegmented<Tab>(
          [
            { value: 'questions', label: 'Questions' },
            { value: 'results', label: 'Results' },
          ],
          tab,
          setTab,
        )}
      </View>

      <ScrollView
        style={styles.content}
        contentContainerStyle={styles.contentContainer}
        keyboardShouldPersistTaps="handled"
      >
        {tab === 'questions' ? renderQuestionsTab() : renderResultsTab()}
      </ScrollView>

      <View style={styles.footer}>
        <View style={styles.modeToggle}>
          {renderSegmented<RunMode>(
            [
              { value: 'single', label: 'Single' },
              { value: 'parallel', label: 'Parallel' },
            ],
            mode,
            setMode,
          )}
        </View>
        <TouchableOpacity
          style={[styles.decideButton, isRunning && styles.decideButtonDisabled]}
          onPress={run}
          disabled={isRunning}
        >
          <Icon name="lightning-bolt" size={18} color={theme.colors.white} />
          <Text style={styles.decideText}>
            {isRunning ? 'Deciding…' : 'Decide'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  )
}

function createStyles(
  theme: ReturnType<typeof useTheme>['theme'],
  themedStyles: ReturnType<typeof createThemedStyles>,
) {
  const { colors } = theme
  return StyleSheet.create({
    container: themedStyles.container,
    modelBar: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 8,
      paddingHorizontal: 16,
      paddingVertical: 10,
      backgroundColor: colors.surface,
      borderBottomWidth: 1,
      borderBottomColor: colors.border,
    },
    modelName: { flex: 1, fontSize: 14, fontWeight: '600', color: colors.text },
    modelChip: {
      fontSize: 12,
      color: colors.textSecondary,
      backgroundColor: colors.card,
      borderRadius: 10,
      paddingHorizontal: 8,
      paddingVertical: 2,
      overflow: 'hidden',
    },
    tabs: { paddingHorizontal: 16, paddingTop: 12 },
    segmented: {
      flexDirection: 'row',
      borderRadius: 8,
      borderWidth: 1,
      borderColor: colors.border,
      overflow: 'hidden',
      backgroundColor: colors.surface,
    },
    segment: { flex: 1, paddingVertical: 8, alignItems: 'center' },
    segmentSelected: { backgroundColor: colors.primary },
    segmentText: { fontSize: 14, fontWeight: '600', color: colors.text },
    segmentTextSelected: { color: colors.white },
    content: { flex: 1 },
    contentContainer: { padding: 16, paddingBottom: 32 },
    presets: { marginBottom: 12, flexGrow: 0 },
    presetsContent: { gap: 8 },
    presetChip: {
      borderRadius: 16,
      borderWidth: 1,
      borderColor: colors.border,
      backgroundColor: colors.surface,
      paddingHorizontal: 14,
      paddingVertical: 6,
    },
    presetChipSelected: {
      borderColor: colors.primary,
      backgroundColor: colors.primary,
    },
    presetText: { fontSize: 13, fontWeight: '600', color: colors.text },
    presetTextSelected: { color: colors.white },
    sectionTitle: {
      fontSize: 15,
      fontWeight: '700',
      color: colors.text,
      marginBottom: 8,
    },
    stateInput: {
      minHeight: 80,
      maxHeight: 180,
      borderWidth: 1,
      borderColor: colors.border,
      borderRadius: 12,
      padding: 10,
      fontSize: 14,
      color: colors.text,
      backgroundColor: colors.surface,
      textAlignVertical: 'top',
      marginBottom: 16,
    },
    questionsHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    textButton: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 4,
      marginBottom: 8,
    },
    textButtonLabel: { fontSize: 13, fontWeight: '600' },
    emptyText: {
      fontSize: 14,
      color: colors.textSecondary,
      textAlign: 'center',
      marginVertical: 24,
    },
    addQuestionButton: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 6,
      paddingVertical: 12,
      borderRadius: 12,
      borderWidth: 1,
      borderStyle: 'dashed',
      borderColor: colors.primary,
    },
    addQuestionText: { color: colors.primary, fontSize: 14, fontWeight: '600' },
    summary: {
      flexDirection: 'row',
      flexWrap: 'wrap',
      gap: 14,
      marginBottom: 12,
    },
    summaryItem: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    summaryText: { fontSize: 13, color: colors.textSecondary },
    errorCard: {
      flexDirection: 'row',
      gap: 8,
      padding: 12,
      borderRadius: 12,
      borderWidth: 1,
      borderColor: colors.error,
      marginBottom: 12,
    },
    errorText: { flex: 1, fontSize: 14, color: colors.error },
    footer: {
      flexDirection: 'row',
      alignItems: 'center',
      gap: 12,
      padding: 12,
      paddingBottom: 20,
      borderTopWidth: 1,
      borderTopColor: colors.border,
      backgroundColor: colors.surface,
    },
    modeToggle: { width: 170 },
    decideButton: {
      flex: 1,
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 6,
      backgroundColor: colors.primary,
      borderRadius: 10,
      paddingVertical: 12,
    },
    decideButtonDisabled: { opacity: 0.5 },
    decideText: { color: colors.white, fontSize: 16, fontWeight: '700' },
  })
}
