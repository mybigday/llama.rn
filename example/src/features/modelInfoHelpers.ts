import { MODELS } from '../utils/constants'
import type { CustomModel } from '../utils/storage'

export interface ModelFileDescriptor {
  name: string
  path: string
}

export const formatModelInfoValue = (value: unknown) =>
  typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)

export const buildRegularModelFiles = (modelPath: string): ModelFileDescriptor[] => [
  { name: 'Model', path: modelPath },
]

export const buildMultimodalModelFiles = (
  modelPath: string,
  mmprojPath: string,
): ModelFileDescriptor[] => [
  { name: 'Model', path: modelPath },
  { name: 'MMProj', path: mmprojPath },
]

export const buildTTSModelFiles = (
  ttsPath: string,
  vocoderPath: string,
): ModelFileDescriptor[] => [
  { name: 'TTS Model', path: ttsPath },
  { name: 'Vocoder', path: vocoderPath },
]

export const buildCustomModelFiles = (
  model: CustomModel,
  modelPath: string,
  mmprojPath?: string,
): ModelFileDescriptor[] => {
  if (model.mmprojFilename && mmprojPath) {
    return buildMultimodalModelFiles(modelPath, mmprojPath)
  }

  return buildRegularModelFiles(modelPath)
}

export const getModelInfoDisplayName = (modelKey: keyof typeof MODELS) =>
  MODELS[modelKey].name
