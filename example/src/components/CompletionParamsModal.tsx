import React, { useEffect } from 'react'
import type { CompletionParams } from '../utils/storage'
import {
  saveCompletionParams,
  loadCompletionParams,
  resetCompletionParams,
  DEFAULT_COMPLETION_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import { validateNumber, validateInteger } from '../utils/validation'
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

  const onSaveHandler = () => {
    handleSave(onSave, onClose)
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
        onChangeText={(text) => {
          const value = validateInteger(text, 1, 4096)
          if (value !== undefined) updateParam('n_predict', value)
        }}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Temperature */}
      <ParameterTextInput
        label="Temperature"
        description="Controls randomness in responses. Lower values (0.1-0.3) are more focused and deterministic, higher values (0.7-1.0) are more creative and varied."
        value={params.temperature?.toString()}
        onChangeText={(text) => {
          const value = validateNumber(text, 0.0, 2.0)
          if (value !== undefined) updateParam('temperature', value)
        }}
        keyboardType="decimal-pad"
        placeholder="0.7"
      />

      {/* Top-p */}
      <ParameterTextInput
        label="Top-p (Nucleus Sampling)"
        description="Controls diversity by considering only tokens with cumulative probability up to p. Lower values (0.1-0.5) are more focused, higher values (0.8-0.95) are more diverse."
        value={params.top_p?.toString()}
        onChangeText={(text) => {
          const value = validateNumber(text, 0.0, 1.0)
          if (value !== undefined) updateParam('top_p', value)
        }}
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
