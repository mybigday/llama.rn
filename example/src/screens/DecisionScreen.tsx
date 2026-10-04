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
import ContextParamsModal from '../components/ContextParamsModal'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import { MaskedProgress } from '../components/MaskedProgress'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import { loadContextParams } from '../utils/storage'
import {
  initLlama,
  type DecisionAnswer,
  type DecisionRequest,
  type DecisionResult,
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

const DECISION_MODELS = createExampleModelDefinitions(
  (Object.keys(MODELS) as ExampleModelKey[]).filter((key) =>
    isDecisionModel(MODELS[key]),
  ),
)

// A drive-thru turn: one state, several typed questions answered in one pass each
const EXAMPLE_REQUEST: DecisionRequest = {
  state: {
    utterance:
      'uh can I get a medium fries and, actually make that a large cola too',
    stage: 'menu',
    cart: [],
  },
  questions: {
    is_ordering: {
      type: 'noul',
      instructions: 'The customer is placing an order',
    },
    ordered_product: {
      type: 'choice',
      instructions: 'Which product did the customer order first?',
      criteria: {
        'fries-m': 'Fries M (medium fries)',
        'cola-l': 'Cola L (large cola)',
        burger: 'Mos Burger',
        none: 'Nothing ordered',
      },
    },
    mood: {
      type: 'score',
      instructions: 'How impatient does the customer sound?',
      criteria: ['Calm', 'Neutral', 'Impatient'],
    },
    wants_drink: {
      type: 'noul',
      instructions: 'The customer wants a drink',
      criteria: { true: 'a drink is mentioned', false: 'no drink' },
    },
  },
}

const N_PARALLEL = 2

const formatAnswer = (answer: DecisionAnswer) => {
  switch (answer.type) {
    case 'noul':
      return `P(true) = ${answer.noul.toFixed(4)}`
    case 'choice':
      return `${answer.choice} (confidence ${answer.confidence.toFixed(3)})\n${Object.entries(answer.probabilities)
        .map(([key, p]) => `  ${key}: ${p.toFixed(4)}`)
        .join('\n')}`
    case 'score':
      return `score ${answer.score.toFixed(3)} (confidence ${answer.confidence.toFixed(3)})\n${Object.entries(answer.probabilities)
        .map(([level, p]) => `  ${answer.legend[level]}: ${p.toFixed(4)}`)
        .join('\n')}`
    default:
      return JSON.stringify(answer)
  }
}

export default function DecisionScreen({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = createStyles(theme, themedStyles)
  const [isLoading, setIsLoading] = useState(false)
  const [isRunning, setIsRunning] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const [requestText, setRequestText] = useState(
    JSON.stringify(EXAMPLE_REQUEST, null, 2),
  )
  const [output, setOutput] = useState('')
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

  const initializeModel = async (modelPath: string) => {
    try {
      setIsLoading(true)
      setInitProgress(0)
      const params = await loadContextParams()
      const llamaContext = await initLlama(
        { ...params, model: modelPath, n_parallel: N_PARALLEL },
        (progress) => setInitProgress(progress),
      )
      const { decision } = llamaContext.model
      console.log(`[Decision] model.decision = ${JSON.stringify(decision)}`)
      if (!decision) {
        await llamaContext.release()
        Alert.alert('Not a decision model', modelPath.split('/').pop())
        return
      }
      await replaceContext(llamaContext)
      setOutput(`model.decision = ${JSON.stringify(decision, null, 2)}`)
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  // Answers the request with decide(), then with parallel.decide() on a slot
  const run = async () => {
    if (!context || isRunning) return
    let request: DecisionRequest
    try {
      request = JSON.parse(requestText)
    } catch (error: any) {
      Alert.alert('Invalid JSON', error.message)
      return
    }

    setIsRunning(true)
    const lines: string[] = []
    const report = (label: string, result: DecisionResult, ms: number) => {
      console.log(`[Decision] ${label} ${ms}ms ${JSON.stringify(result)}`)
      lines.push(
        `== ${label}: ${ms} ms, ${result.usage.input_tokens} input tokens`,
        ...Object.entries(result.answers).map(
          ([id, answer]) => `${id}: ${formatAnswer(answer)}`,
        ),
        '',
      )
    }
    try {
      let t0 = Date.now()
      report('sync', await context.decide(request), Date.now() - t0)

      await context.parallel.enable({ n_parallel: N_PARALLEL })
      try {
        t0 = Date.now()
        const { promise } = await context.parallel.decide(request)
        report('parallel', await promise, Date.now() - t0)
      } finally {
        await context.parallel.disable()
      }
    } catch (error: any) {
      console.log(`[Decision] error ${error.message}`)
      lines.push(`Error: ${error.message}`)
    } finally {
      setOutput(lines.join('\n'))
      setIsRunning(false)
    }
  }

  if (!isModelReady) {
    return (
      <>
        <ExampleModelSetup
          description="Typed decision models answer choice / score / yes-no questions about a state in one forward pass each, with calibrated probabilities instead of generated text."
          defaultModels={DECISION_MODELS}
          customModels={customModels || []}
          onInitializeCustomModel={(_model, modelPath) =>
            initializeModel(modelPath)
          }
          onInitializeModel={(_model, modelPath) => initializeModel(modelPath)}
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

  return (
    <View style={styles.container}>
      <ScrollView style={styles.content}>
        <Text style={styles.label}>Request</Text>
        <TextInput
          style={styles.requestInput}
          value={requestText}
          onChangeText={setRequestText}
          multiline
          autoCapitalize="none"
          autoCorrect={false}
        />
        <TouchableOpacity
          style={[styles.button, isRunning && styles.buttonDisabled]}
          onPress={run}
          disabled={isRunning}
        >
          <Text style={styles.buttonText}>
            {isRunning ? 'Deciding...' : 'Decide'}
          </Text>
        </TouchableOpacity>
        <Text style={styles.output} selectable>
          {output}
        </Text>
      </ScrollView>
      <MaskedProgress
        visible={isRunning}
        text="Deciding..."
        progress={0}
        showProgressBar={false}
      />
    </View>
  )
}

function createStyles(
  theme: ReturnType<typeof useTheme>['theme'],
  themedStyles: ReturnType<typeof createThemedStyles>,
) {
  return StyleSheet.create({
    container: themedStyles.container,
    content: {
      flex: 1,
      padding: 16,
    },
    label: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: 8,
    },
    requestInput: {
      minHeight: 180,
      maxHeight: 280,
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 8,
      padding: 8,
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.colors.text,
      backgroundColor: theme.colors.surface,
      textAlignVertical: 'top',
    },
    button: {
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      paddingVertical: 12,
      alignItems: 'center',
      marginVertical: 16,
    },
    buttonDisabled: {
      opacity: 0.5,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
    },
    output: {
      fontFamily: 'monospace',
      fontSize: 12,
      color: theme.colors.text,
      marginBottom: 32,
    },
  })
}
