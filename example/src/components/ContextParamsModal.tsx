import React, { useEffect } from 'react'
import { Alert } from 'react-native'
import type { ContextParams } from '../utils/storage'
import {
  saveContextParams,
  loadContextParams,
  resetContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import { ParameterTextInput, ParameterSwitch } from './ParameterFormFields'
import { ParameterMenu } from './ParameterMenu'
import BaseParameterModal from './BaseParameterModal'

interface ContextParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: ContextParams) => void
}

const CACHE_TYPE_OPTIONS = ['f16', 'f32', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']

export default function ContextParamsModal({
  visible,
  onClose,
  onSave,
}: ContextParamsModalProps) {
  const {
    params,
    isLoading,
    loadParamsAsync,
    handleSave,
    handleReset,
    updateParam,
  } = useParameterModal({
    loadParams: loadContextParams,
    saveParams: saveContextParams,
    resetParams: resetContextParams,
    defaultParams: DEFAULT_CONTEXT_PARAMS,
  })

  useEffect(() => {
    if (visible) loadParamsAsync()
  }, [loadParamsAsync, visible])

  const handleTextInput = (text: string, paramKey: keyof ContextParams) => {
    if (text === '') {
      updateParam(paramKey, undefined)
    } else {
      const parsedValue = parseInt(text, 10)
      updateParam(paramKey, Number.isNaN(parsedValue) ? text : parsedValue)
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

  const validateParams = (): { isValid: boolean; errors: string[] } => {
    const validations = [
      validateIntegerParam(params.n_ctx, 128, 99999, 'Context Size'),
      validateIntegerParam(params.n_gpu_layers, 0, 99, 'GPU Layers'),
      validateIntegerParam(params.n_batch, 1, 99999, 'Batch Size'),
      validateIntegerParam(params.n_ubatch, 1, 99999, 'Micro Batch Size'),
      validateIntegerParam(params.n_threads, 1, 32, 'Threads'),
    ]

    const errors = validations.filter((error): error is string => error !== null)
    return { isValid: errors.length === 0, errors }
  }

  const convertStringParamsToNumbers = (
    stringParams: ContextParams,
  ): ContextParams => {
    const converted = { ...stringParams }

    if (typeof converted.n_ctx === 'string') {
      const num = parseInt(converted.n_ctx, 10)
      converted.n_ctx = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_gpu_layers === 'string') {
      const num = parseInt(converted.n_gpu_layers, 10)
      converted.n_gpu_layers = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_batch === 'string') {
      const num = parseInt(converted.n_batch, 10)
      converted.n_batch = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_ubatch === 'string') {
      const num = parseInt(converted.n_ubatch, 10)
      converted.n_ubatch = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_threads === 'string') {
      const num = parseInt(converted.n_threads, 10)
      converted.n_threads = Number.isNaN(num) ? undefined : num
    }

    return converted
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
      title="Context Parameters"
      description="Configure context initialization parameters. These settings affect memory usage, performance, and model behavior during loading."
      isLoading={isLoading}
      onSave={onSaveHandler}
      onReset={handleReset}
      showWarning
      warningText="Warning: Changing context parameters requires reinitializing the model, which will clear your current conversation."
    >
      {/* Context Size */}
      <ParameterTextInput
        label="Context Size (n_ctx)"
        description="Maximum context length in tokens. Higher values use more memory."
        value={params.n_ctx?.toString()}
        onChangeText={(text) => {
          // Allow any text input, validation happens on save
          if (text === '') {
            updateParam('n_ctx', undefined)
          } else {
            const parsedValue = parseInt(text, 10)
            updateParam('n_ctx', Number.isNaN(parsedValue) ? text : parsedValue)
          }
        }}
        keyboardType="numeric"
        placeholder="8192"
      />

      {/* GPU Layers */}
      <ParameterTextInput
        label="GPU Layers (n_gpu_layers)"
        description="Number of layers to run on GPU. Use 99 for all layers, 0 for CPU only."
        value={params.n_gpu_layers?.toString()}
        onChangeText={(text) => {
          // Allow any text input, validation happens on save
          if (text === '') {
            updateParam('n_gpu_layers', undefined)
          } else {
            const parsedValue = parseInt(text, 10)
            updateParam(
              'n_gpu_layers',
              Number.isNaN(parsedValue) ? text : parsedValue,
            )
          }
        }}
        keyboardType="numeric"
        placeholder="99"
      />

      {/* Memory Lock */}
      <ParameterSwitch
        label="Memory Lock (use_mlock)"
        description="Lock model in memory to prevent swapping to disk."
        value={params.use_mlock || false}
        onValueChange={(value) => updateParam('use_mlock', value)}
      />

      {/* Memory Map */}
      <ParameterSwitch
        label="Memory Map (use_mmap)"
        description="Use memory mapping for better performance."
        value={params.use_mmap || false}
        onValueChange={(value) => updateParam('use_mmap', value)}
      />

      {/* Batch Size */}
      <ParameterTextInput
        label="Batch Size (n_batch)"
        description="Number of tokens to process in parallel. Higher values use more memory."
        value={params.n_batch?.toString() || '512'}
        onChangeText={(text) => handleTextInput(text, 'n_batch')}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Micro Batch Size */}
      <ParameterTextInput
        label="Micro Batch Size (n_ubatch)"
        description="Internal batch size for processing. Should be ≤ n_batch."
        value={params.n_ubatch?.toString() || '512'}
        onChangeText={(text) => handleTextInput(text, 'n_ubatch')}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Threads */}
      <ParameterTextInput
        label="Threads (n_threads)"
        description="Number of CPU threads to use. Usually set to number of CPU cores."
        value={params.n_threads?.toString()}
        onChangeText={(text) => handleTextInput(text, 'n_threads')}
        keyboardType="numeric"
        placeholder="8"
      />

      {/* Context Shift */}
      <ParameterSwitch
        label="Context Shift (ctx_shift)"
        description="Enable automatic context shifting when context is full."
        value={params.ctx_shift || false}
        onValueChange={(value) => updateParam('ctx_shift', value)}
      />

      {/* Flash Attention */}
      <ParameterSwitch
        label="Flash Attention (flash_attn)"
        description="Only recommended in GPU device."
        value={params.flash_attn || false}
        onValueChange={(value) => updateParam('flash_attn', value)}
      />

      {/* Cache Type K */}
      <ParameterMenu
        label="Cache Type K (cache_type_k)"
        description="KV cache data type for the K. Need enable flash_attn to change this."
        value={params.cache_type_k}
        options={CACHE_TYPE_OPTIONS}
        onSelect={(value) => updateParam('cache_type_k', value)}
        placeholder="f16"
      />

      {/* Cache Type V */}
      <ParameterMenu
        label="Cache Type V (cache_type_v)"
        description="KV cache data type for the V. Need enable flash_attn if change this."
        value={params.cache_type_v}
        options={CACHE_TYPE_OPTIONS}
        onSelect={(value) => updateParam('cache_type_v', value)}
        placeholder="f16"
      />

      {/* KV Unified */}
      <ParameterSwitch
        label="KV Unified (kv_unified)"
        description="Use unified key-value store for better performance."
        value={params.kv_unified || false}
        onValueChange={(value) => updateParam('kv_unified', value)}
      />

      {/* SWA Full */}
      <ParameterSwitch
        label="SWA Full (swa_full)"
        description="Use full-size SWA cache. May improve performance for multiple sequences but uses more memory."
        value={params.swa_full || false}
        onValueChange={(value) => updateParam('swa_full', value)}
      />
    </BaseParameterModal>
  )
}
