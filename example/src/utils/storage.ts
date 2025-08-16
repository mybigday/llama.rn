import AsyncStorage from '@react-native-async-storage/async-storage'
import type {
  ContextParams as LlamaContextParams,
  CompletionParams as LlamaCompletionParams,
} from '../../../src'

export type ContextParams = Omit<LlamaContextParams, 'model'>
export type CompletionParams = Omit<LlamaCompletionParams, 'prompt'>

export interface TTSParams {
  speakerConfig: any | null
}

export interface CustomModel {
  id: string
  repo: string
  filename: string
  quantization: string
  mmprojFilename?: string
  mmprojQuantization?: string
  addedAt: number
  localPath?: string
  mmprojLocalPath?: string
}

// Storage keys
const CONTEXT_PARAMS_KEY = '@llama_context_params'
const COMPLETION_PARAMS_KEY = '@llama_completion_params'
const TTS_PARAMS_KEY = '@llama_tts_params'
const CUSTOM_MODELS_KEY = '@llama_custom_models'

// Default parameter values
export const DEFAULT_CONTEXT_PARAMS: ContextParams = {
  n_ctx: 8192,
  n_gpu_layers: 99,
  use_mlock: true,
  use_mmap: true,
  n_batch: 512,
  n_ubatch: 512,
  ctx_shift: false,
  flash_attn: false,
  cache_type_k: 'f16',
  cache_type_v: 'f16',
  kv_unified: false,
  swa_full: false,
}

export const DEFAULT_COMPLETION_PARAMS: CompletionParams = {
  enable_thinking: true,
  n_predict: 1024,
  temperature: 0.7,
  top_p: 0.9,
  stop: [],
}

export const DEFAULT_TTS_PARAMS: TTSParams = {
  speakerConfig: null,
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

// Storage functions for TTS parameters
export const saveTTSParams = async (params: TTSParams): Promise<void> => {
  try {
    const jsonValue = JSON.stringify(params)
    await AsyncStorage.setItem(TTS_PARAMS_KEY, jsonValue)
  } catch (error) {
    console.error('Error saving TTS params:', error)
    throw error
  }
}

export const loadTTSParams = async (): Promise<TTSParams> => {
  try {
    const jsonValue = await AsyncStorage.getItem(TTS_PARAMS_KEY)
    if (jsonValue != null) {
      const params = JSON.parse(jsonValue)
      // Merge with defaults to ensure all required fields exist
      return { ...DEFAULT_TTS_PARAMS, ...params }
    }
    return DEFAULT_TTS_PARAMS
  } catch (error) {
    console.error('Error loading TTS params:', error)
    return DEFAULT_TTS_PARAMS
  }
}

export const resetTTSParams = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(TTS_PARAMS_KEY)
  } catch (error) {
    console.error('Error resetting TTS params:', error)
    throw error
  }
}

export const loadCustomModels = async (): Promise<CustomModel[]> => {
  try {
    const jsonValue = await AsyncStorage.getItem(CUSTOM_MODELS_KEY)
    if (jsonValue != null) {
      return JSON.parse(jsonValue)
    }
    return []
  } catch (error) {
    console.error('Error loading custom models:', error)
    return []
  }
}

// Storage functions for custom models
export const saveCustomModel = async (model: CustomModel): Promise<void> => {
  try {
    const existingModels = await loadCustomModels()
    const updatedModels = [
      ...existingModels.filter((m) => m.id !== model.id),
      model,
    ]
    const jsonValue = JSON.stringify(updatedModels)
    await AsyncStorage.setItem(CUSTOM_MODELS_KEY, jsonValue)
  } catch (error) {
    console.error('Error saving custom model:', error)
    throw error
  }
}

export const deleteCustomModel = async (modelId: string): Promise<void> => {
  try {
    const existingModels = await loadCustomModels()
    const updatedModels = existingModels.filter((m) => m.id !== modelId)
    const jsonValue = JSON.stringify(updatedModels)
    await AsyncStorage.setItem(CUSTOM_MODELS_KEY, jsonValue)
  } catch (error) {
    console.error('Error deleting custom model:', error)
    throw error
  }
}

export const getCustomModel = async (
  modelId: string,
): Promise<CustomModel | null> => {
  try {
    const models = await loadCustomModels()
    return models.find((m) => m.id === modelId) || null
  } catch (error) {
    console.error('Error getting custom model:', error)
    return null
  }
}
