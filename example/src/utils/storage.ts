import AsyncStorage from '@react-native-async-storage/async-storage'

// Storage keys
const CONTEXT_PARAMS_KEY = '@llama_context_params'
const COMPLETION_PARAMS_KEY = '@llama_completion_params'

// Default parameter values
export const DEFAULT_CONTEXT_PARAMS = {
  n_ctx: 8192,
  n_gpu_layers: 99,
  use_mlock: true,
  use_mmap: true,
  n_batch: 512,
  n_ubatch: 512,
  // n_threads: 4,
  ctx_shift: false,
}

export const DEFAULT_COMPLETION_PARAMS = {
  n_predict: 512,
  temperature: 0.7,
  top_p: 0.9,
  stop: ['<|im_end|>', '<end_of_turn>'],
}

// Context parameters interface
export interface ContextParams {
  n_ctx: number
  n_gpu_layers: number
  use_mlock: boolean
  use_mmap: boolean
  n_batch?: number
  n_ubatch?: number
  n_threads?: number
  ctx_shift?: boolean
}

// Completion parameters interface
export interface CompletionParams {
  n_predict: number
  temperature: number
  top_p: number
  stop: string[]
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
