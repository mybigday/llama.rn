import AsyncStorage from '@react-native-async-storage/async-storage'
import type {
  ContextParams as LlamaContextParams,
  CompletionParams as LlamaCompletionParams,
} from '../../../src'

export type ContextParams = Omit<LlamaContextParams, 'model'>
export type CompletionParams = Omit<LlamaCompletionParams, 'prompt'>

// Storage keys
const CONTEXT_PARAMS_KEY = '@llama_context_params'
const COMPLETION_PARAMS_KEY = '@llama_completion_params'

// Default parameter values
export const DEFAULT_CONTEXT_PARAMS: ContextParams = {
  n_ctx: 8192,
  n_gpu_layers: 99,
  use_mlock: true,
  use_mmap: true,
  n_batch: 512,
  n_ubatch: 512,
  ctx_shift: false,
  cache_type_k: 'f16',
  cache_type_v: 'f16',
  kv_unified: true,
}

export const DEFAULT_COMPLETION_PARAMS: CompletionParams = {
  n_predict: 512,
  temperature: 0.7,
  top_p: 0.9,
  stop: ['<|im_end|>', '<end_of_turn>'],
}

// Storage functions for context parameters
export const saveContextParams = async (
  params: ContextParams,
): Promise<void> => {
  try {
    const jsonValue = JSON.stringify(params)
    await AsyncStorage.setItem(CONTEXT_PARAMS_KEY, jsonValue)
  } catch (error) {
    console.error('Error saving context params:', error)
    throw error
  }
}

export const loadContextParams = async (): Promise<ContextParams> => {
  try {
    const jsonValue = await AsyncStorage.getItem(CONTEXT_PARAMS_KEY)
    if (jsonValue != null) {
      const params = JSON.parse(jsonValue)
      // Merge with defaults to ensure all required fields exist
      return { ...DEFAULT_CONTEXT_PARAMS, ...params }
    }
    return DEFAULT_CONTEXT_PARAMS
  } catch (error) {
    console.error('Error loading context params:', error)
    return DEFAULT_CONTEXT_PARAMS
  }
}

// Storage functions for completion parameters
export const saveCompletionParams = async (
  params: CompletionParams,
): Promise<void> => {
  try {
    const jsonValue = JSON.stringify(params)
    await AsyncStorage.setItem(COMPLETION_PARAMS_KEY, jsonValue)
  } catch (error) {
    console.error('Error saving completion params:', error)
    throw error
  }
}

export const loadCompletionParams = async (): Promise<CompletionParams> => {
  try {
    const jsonValue = await AsyncStorage.getItem(COMPLETION_PARAMS_KEY)
    if (jsonValue != null) {
      const params = JSON.parse(jsonValue)
      // Merge with defaults to ensure all required fields exist
      return { ...DEFAULT_COMPLETION_PARAMS, ...params }
    }
    return DEFAULT_COMPLETION_PARAMS
  } catch (error) {
    console.error('Error loading completion params:', error)
    return DEFAULT_COMPLETION_PARAMS
  }
}

// Reset functions
export const resetContextParams = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(CONTEXT_PARAMS_KEY)
  } catch (error) {
    console.error('Error resetting context params:', error)
    throw error
  }
}

export const resetCompletionParams = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(COMPLETION_PARAMS_KEY)
  } catch (error) {
    console.error('Error resetting completion params:', error)
    throw error
  }
}
