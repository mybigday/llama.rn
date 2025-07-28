import React, { useEffect } from 'react'
import type { ContextParams } from '../utils/storage'
import {
  saveContextParams,
  loadContextParams,
  resetContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import { validateInteger } from '../utils/validation'
import { ParameterTextInput, ParameterSwitch } from './ParameterFormFields'
import BaseParameterModal from './BaseParameterModal'

interface ContextParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: ContextParams) => void
}

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

  const onSaveHandler = () => {
    handleSave(onSave, onClose)
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
          const value = validateInteger(text, 512, 32768)
          if (value !== undefined) updateParam('n_ctx', value)
        }}
        keyboardType="numeric"
        placeholder="4096"
      />

      {/* GPU Layers */}
      <ParameterTextInput
        label="GPU Layers (n_gpu_layers)"
        description="Number of layers to run on GPU. Use 99 for all layers, 0 for CPU only."
        value={params.n_gpu_layers?.toString()}
        onChangeText={(text) => {
          const value = validateInteger(text, 0, 99)
          if (value !== undefined) updateParam('n_gpu_layers', value)
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
        onChangeText={(text) => {
          const value = validateInteger(text, 1, 2048)
          if (value !== undefined) updateParam('n_batch', value)
        }}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Micro Batch Size */}
      <ParameterTextInput
        label="Micro Batch Size (n_ubatch)"
        description="Internal batch size for processing. Should be ≤ n_batch."
        value={params.n_ubatch?.toString() || '512'}
        onChangeText={(text) => {
          const value = validateInteger(text, 1, 2048)
          if (value !== undefined) updateParam('n_ubatch', value)
        }}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Threads */}
      <ParameterTextInput
        label="Threads (n_threads)"
        description="Number of CPU threads to use. Usually set to number of CPU cores."
        value={params.n_threads?.toString()}
        onChangeText={(text) => {
          const value = validateInteger(text, 1, 32)
          if (value !== undefined) updateParam('n_threads', value)
        }}
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
      <ParameterTextInput
        label="Cache Type K (cache_type_k)"
        description="KV cache data type for the K. Need enable flash_attn to change this. Available values: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
        value={params.cache_type_k}
        onChangeText={(text) => updateParam('cache_type_k', text)}
        placeholder="f16"
      />

      {/* Cache Type V */}
      <ParameterTextInput
        label="Cache Type V (cache_type_v)"
        description="KV cache data type for the V. Need enable flash_attn if change this. Available values: f16, f32, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
        value={params.cache_type_v}
        onChangeText={(text) => updateParam('cache_type_v', text)}
        placeholder="f16"
      />

      {/* KV Unified */}
      <ParameterSwitch
        label="KV Unified (kv_unified)"
        description="Use unified key-value store for better performance."
        value={params.kv_unified || false}
        onValueChange={(value) => updateParam('kv_unified', value)}
      />
    </BaseParameterModal>
  )
}
