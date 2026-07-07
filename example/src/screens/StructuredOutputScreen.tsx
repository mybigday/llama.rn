import React, { useState, useRef } from 'react'
import {
  View,
  Text,
  TextInput,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import ContextParamsModal from '../components/ContextParamsModal'
import CompletionParamsModal from '../components/CompletionParamsModal'
import { ExampleModelSetup } from '../components/ExampleModelSetup'
import { StopButton } from '../components/StopButton'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import type { ContextParams, CompletionParams } from '../utils/storage'
import { loadCompletionParams } from '../utils/storage'
import { initLlama } from '../../../src'
import {
  useStoredCompletionParams,
  useStoredContextParams,
  useStoredCustomModels,
} from '../hooks/useStoredSetting'
import { useExampleContext } from '../hooks/useExampleContext'
import { useExampleScreenHeader } from '../hooks/useExampleScreenHeader'
import { createExampleModelDefinitions } from '../utils/exampleModels'

const STRUCTURED_OUTPUT_MODELS = createExampleModelDefinitions([
  'SMOL_LM_3',
  'GEMMA_3_4B_QAT',
  'QWEN_3_4B',
  'GEMMA_3N_E2B',
  'GEMMA_3N_E4B',
])

const DEFAULT_PROMPT =
  'Extract the task details from this request: Remind me to call Alex tomorrow at 9 AM about the quarterly launch checklist. Mark it as high priority.'

const DEFAULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    title: {
      type: 'string',
      description: 'Short task title',
    },
    assignee: {
      type: 'string',
      description: 'Person responsible for the task',
    },
    due_date: {
      type: 'string',
      description: 'Natural language due date or time',
    },
    priority: {
      type: 'string',
      enum: ['low', 'medium', 'high'],
    },
    tags: {
      type: 'array',
      items: {
        type: 'string',
      },
    },
  },
  required: ['title', 'assignee', 'due_date', 'priority', 'tags'],
}

const SYSTEM_PROMPT =
  'Return only a JSON object that satisfies the provided schema.'

const formatJson = (value: unknown) => JSON.stringify(value, null, 2)

const tryFormatJsonText = (text: string) => {
  try {
    return formatJson(JSON.parse(text))
  } catch {
    return text
  }
}

const createStyles = (theme: ReturnType<typeof useTheme>['theme']) =>
  StyleSheet.create({
    container: {
      flex: 1,
    },
    contentContainer: {
      padding: 16,
    },
    section: {
      marginBottom: 16,
    },
    label: {
      color: theme.colors.text,
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 8,
    },
    textArea: {
      backgroundColor: theme.colors.inputBackground,
      borderColor: theme.colors.border,
      borderRadius: 8,
      borderWidth: 1,
      color: theme.colors.text,
      fontFamily: 'Menlo',
      fontSize: 13,
      padding: 12,
      textAlignVertical: 'top',
    },
    promptInput: {
      minHeight: 96,
    },
    schemaInput: {
      minHeight: 260,
    },
    actionButton: {
      alignItems: 'center',
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      marginBottom: 16,
      padding: 14,
    },
    actionButtonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '700',
    },
    disabledButton: {
      backgroundColor: theme.colors.disabled,
    },
    outputText: {
      backgroundColor: theme.colors.card,
      borderColor: theme.colors.border,
      borderRadius: 8,
      borderWidth: 1,
      color: theme.colors.text,
      fontFamily: 'Menlo',
      fontSize: 13,
      lineHeight: 18,
      minHeight: 96,
      padding: 12,
    },
  })

export default function StructuredOutputScreen({
  navigation,
}: {
  navigation: any
}) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const styles = createStyles(theme)
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [schemaText, setSchemaText] = useState(formatJson(DEFAULT_SCHEMA))
  const [rawOutput, setRawOutput] = useState('')
  const [parsedOutput, setParsedOutput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [showCompletionParamsModal, setShowCompletionParamsModal] =
    useState(false)
  const [showCustomModelModal, setShowCustomModelModal] = useState(false)
  const streamedTextRef = useRef('')
  const insets = useSafeAreaInsets()
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

  const handleSaveContextParams = (params: ContextParams) => {
    setContextParams(params)
  }

  const handleSaveCompletionParams = (params: CompletionParams) => {
    setCompletionParams(params)
  }

  const handleReset = () => {
    setPrompt(DEFAULT_PROMPT)
    setSchemaText(formatJson(DEFAULT_SCHEMA))
    setRawOutput('')
    setParsedOutput('')
  }

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

  const initializeModel = async (modelPath: string, params?: ContextParams) => {
    try {
      setIsLoading(true)
      setInitProgress(0)

      const llamaContext = await initLlama(
        {
          model: modelPath,
          use_mlock: true,
          ...(params || contextParams || {}),
        },
        (progress) => {
          setInitProgress(progress)
        },
      )

      await replaceContext(llamaContext)
      setInitProgress(100)
    } catch (error: any) {
      Alert.alert('Error', `Failed to initialize model: ${error.message}`)
    } finally {
      setIsLoading(false)
      setInitProgress(0)
    }
  }

  const handleGenerate = async () => {
    if (!context || isGenerating) return

    let schema: object
    try {
      schema = JSON.parse(schemaText)
    } catch (error: any) {
      Alert.alert('Invalid JSON Schema', error.message)
      return
    }

    setRawOutput('')
    setParsedOutput('')
    streamedTextRef.current = ''
    setIsGenerating(true)

    try {
      const completionParameters =
        completionParams || (await loadCompletionParams())
      const messages = [
        {
          role: 'system' as const,
          content: SYSTEM_PROMPT,
        },
        {
          role: 'user' as const,
          content: prompt,
        },
      ]
      const responseFormat = {
        type: 'json_schema' as const,
        json_schema: {
          strict: true,
          schema,
        },
      }

      const result = await context.completion(
        {
          ...completionParameters,
          messages,
          response_format: responseFormat,
          temperature: completionParameters.temperature ?? 0.2,
          enable_thinking: false,
        },
        (data) => {
          const text = data.content || data.token || ''
          streamedTextRef.current = data.content || streamedTextRef.current + text
          setRawOutput(streamedTextRef.current)
        },
      )

      const output = result.content || result.text || streamedTextRef.current
      setRawOutput(output)
      setParsedOutput(tryFormatJsonText(output))
    } catch (error: any) {
      Alert.alert('Error', `Failed to generate structured output: ${error}`)
    } finally {
      setIsGenerating(false)
    }
  }

  if (!isModelReady) {
    return (
      <>
        <ExampleModelSetup
          description="Select a model to generate JSON constrained by a JSON Schema."
          defaultModels={STRUCTURED_OUTPUT_MODELS}
          customModels={customModels || []}
          onInitializeCustomModel={(_model, modelPath) =>
            initializeModel(modelPath)
          }
          onInitializeModel={(_model, modelPath) => initializeModel(modelPath)}
          onReloadCustomModels={reloadCustomModels}
          showCustomModelModal={showCustomModelModal}
          onOpenCustomModelModal={() => setShowCustomModelModal(true)}
          onCloseCustomModelModal={() => setShowCustomModelModal(false)}
          isLoading={isLoading}
          initProgress={initProgress}
          progressText={`Initializing model... ${initProgress}%`}
        />

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
          styles.contentContainer,
          { paddingBottom: insets.bottom + 24 },
        ]}
      >
        <View style={styles.section}>
          <Text style={styles.label}>Prompt</Text>
          <TextInput
            style={[styles.textArea, styles.promptInput]}
            multiline
            value={prompt}
            onChangeText={setPrompt}
            editable={!isGenerating}
            autoCorrect={false}
            autoCapitalize="sentences"
          />
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>JSON Schema</Text>
          <TextInput
            style={[styles.textArea, styles.schemaInput]}
            multiline
            value={schemaText}
            onChangeText={setSchemaText}
            editable={!isGenerating}
            autoCorrect={false}
            autoComplete="off"
            autoCapitalize="none"
            keyboardType="ascii-capable"
          />
        </View>

        <TouchableOpacity
          style={[
            styles.actionButton,
            (!prompt || isGenerating) && styles.disabledButton,
          ]}
          onPress={handleGenerate}
          disabled={!prompt || isGenerating}
        >
          <Text style={styles.actionButtonText}>
            {isGenerating ? 'Generating...' : 'Generate JSON'}
          </Text>
        </TouchableOpacity>

        <View style={styles.section}>
          <Text style={styles.label}>Parsed JSON</Text>
          <Text style={styles.outputText}>
            {parsedOutput || 'Generated JSON will appear here.'}
          </Text>
        </View>

        <View style={styles.section}>
          <Text style={styles.label}>Raw Output</Text>
          <Text style={styles.outputText}>
            {rawOutput || 'Streaming output will appear here.'}
          </Text>
        </View>
      </ScrollView>

      <StopButton context={context} insets={insets} isLoading={isGenerating} />

      <CompletionParamsModal
        visible={showCompletionParamsModal}
        onClose={() => setShowCompletionParamsModal(false)}
        onSave={handleSaveCompletionParams}
      />
    </View>
  )
}
