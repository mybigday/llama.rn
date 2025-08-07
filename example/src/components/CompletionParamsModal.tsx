import React, { useEffect } from 'react'
import { Alert } from 'react-native'
import type { CompletionParams } from '../utils/storage'
import {
  saveCompletionParams,
  loadCompletionParams,
  resetCompletionParams,
  DEFAULT_COMPLETION_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import {
  ParameterTextInput,
  ParameterSwitch,
  StopSequenceField,
} from './ParameterFormFields'
import BaseParameterModal from './BaseParameterModal'

interface CompletionParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: CompletionParams) => void
}

export default function CompletionParamsModal({
  visible,
  onClose,
  onSave,
}: CompletionParamsModalProps) {
  const {
    params,
    isLoading,
    loadParamsAsync,
    handleSave,
    handleReset,
    updateParam,
  } = useParameterModal({
    loadParams: loadCompletionParams,
    saveParams: saveCompletionParams,
    resetParams: resetCompletionParams,
    defaultParams: DEFAULT_COMPLETION_PARAMS,
  })

  useEffect(() => {
    if (visible) loadParamsAsync()
  }, [loadParamsAsync, visible])

  const handleTextInput = (text: string, paramKey: keyof CompletionParams) => {
    if (text === '') {
      updateParam(paramKey, undefined)
    } else {
      const parsedInt = parseInt(text, 10)
      const parsedFloat = parseFloat(text)

      // For integer fields
      if (paramKey === 'n_predict') {
        updateParam(paramKey, Number.isNaN(parsedInt) ? text : parsedInt)
      } else {
        // For float fields (temperature, top_p)
        updateParam(paramKey, Number.isNaN(parsedFloat) ? text : parsedFloat)
      }
    }
  }

  const validateIntegerParam = (
    value: any,
    min: number,
    max: number,
    fieldName: string,
  ): string | null => {
    if (value === undefined || value === null) return null

    const num = typeof value === 'string' ? parseInt(value, 10) : value
    if (Number.isNaN(num) || num < min || num > max) {
      return `${fieldName} must be between ${min} and ${max}`
    }
    return null
  }

  const validateNumberParam = (
    value: any,
    min: number,
    max: number,
    fieldName: string,
  ): string | null => {
    if (value === undefined || value === null) return null

    const num = typeof value === 'string' ? parseFloat(value) : value
    if (Number.isNaN(num) || num < min || num > max) {
      return `${fieldName} must be between ${min} and ${max}`
    }
    return null
  }

  const validateParams = (): { isValid: boolean; errors: string[] } => {
    const validations = [
      validateIntegerParam(
        params.n_predict,
        -1,
        4096,
        'Max Tokens (-1 for no limit)',
      ),
      validateNumberParam(params.temperature, 0.0, 2.0, 'Temperature'),
      validateNumberParam(params.top_p, 0.0, 1.0, 'Top-p'),
    ]

    const errors = validations.filter(
      (error): error is string => error !== null,
    )
    return { isValid: errors.length === 0, errors }
  }

  const convertStringParamsToNumbers = (
    stringParams: CompletionParams,
  ): CompletionParams => {
    const converted = { ...stringParams }

    if (typeof converted.n_predict === 'string') {
      const num = parseInt(converted.n_predict, 10)
      converted.n_predict = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.temperature === 'string') {
      const num = parseFloat(converted.temperature)
      converted.temperature = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.top_p === 'string') {
      const num = parseFloat(converted.top_p)
      converted.top_p = Number.isNaN(num) ? undefined : num
    }

    return converted
  }

  const addStopSequence = () => {
    const newStop = [...(params.stop || []), '']
    updateParam('stop', newStop)
  }

  const removeStopSequence = (index: number) => {
    const newStop = (params.stop || []).filter((_, i) => i !== index)
    updateParam('stop', newStop)
  }

  const updateStopSequence = (index: number, value: string) => {
    const newStop = [...(params.stop || [])]
    newStop[index] = value
    updateParam('stop', newStop)
  }

  const onSaveHandler = () => {
    const validation = validateParams()
    if (!validation.isValid) {
      Alert.alert(
        'Validation Error',
        `Please fix the following errors:\n\n${validation.errors.join('\n')}`,
        [{ text: 'OK' }],
      )
      return
    }

    const convertedParams = convertStringParamsToNumbers(params)
    handleSave((_params) => onSave(convertedParams), onClose)
  }

  return (
    <BaseParameterModal
      visible={visible}
      onClose={onClose}
      title="Completion Parameters"
      description="Configure completion and sampling parameters. These settings control how the AI generates responses during conversations."
      isLoading={isLoading}
      onSave={onSaveHandler}
      onReset={handleReset}
    >
      {/* Max Tokens */}
      <ParameterTextInput
        label="Max Tokens (n_predict)"
        description="Maximum number of tokens to generate in response. Higher values allow longer responses."
        value={params.n_predict?.toString()}
        onChangeText={(text) => handleTextInput(text, 'n_predict')}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Temperature */}
      <ParameterTextInput
        label="Temperature"
        description="Controls randomness in responses. Lower values (0.1-0.3) are more focused and deterministic, higher values (0.7-1.0) are more creative and varied."
        value={params.temperature?.toString()}
        onChangeText={(text) => handleTextInput(text, 'temperature')}
        keyboardType="decimal-pad"
        placeholder="0.7"
      />

      {/* Top-p */}
      <ParameterTextInput
        label="Top-p (Nucleus Sampling)"
        description="Controls diversity by considering only tokens with cumulative probability up to p. Lower values (0.1-0.5) are more focused, higher values (0.8-0.95) are more diverse."
        value={params.top_p?.toString()}
        onChangeText={(text) => handleTextInput(text, 'top_p')}
        keyboardType="decimal-pad"
        placeholder="0.9"
      />

      {/* Enable Thinking */}
      <ParameterSwitch
        label="Enable Thinking"
        description="Enable thinking in the response if the model supports it."
        value={params.enable_thinking || false}
        onValueChange={(value) => updateParam('enable_thinking', value)}
      />

      {/* Stop Sequences */}
      <StopSequenceField
        stopSequences={params.stop || []}
        onUpdateStopSequence={updateStopSequence}
        onRemoveStopSequence={removeStopSequence}
        onAddStopSequence={addStopSequence}
      />
    </BaseParameterModal>
  )
}
