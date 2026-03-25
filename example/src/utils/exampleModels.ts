import { MODELS } from './constants'
import type {
  ExampleModelDefinition,
  ExampleTTSModelDefinition,
} from '../types/example'

export type ExampleModelKey = keyof typeof MODELS

type ModelConfig = (typeof MODELS)[ExampleModelKey]

type ModelWithVocoder = ModelConfig & {
  vocoder?: ExampleTTSModelDefinition['vocoder']
}

type ModelWithCapabilities = ModelWithVocoder & {
  embedding?: unknown
  ranking?: unknown
}

export const createExampleModelDefinition = (
  key: ExampleModelKey,
  initializeButtonText = 'Initialize',
): ExampleModelDefinition => {
  const model = MODELS[key]
  const baseDefinition = {
    key,
    title: model.name,
    size: model.size,
    initializeButtonText,
  }

  if (model.mmproj) {
    return {
      ...baseDefinition,
      kind: 'multimodal',
      repo: model.repo,
      filename: model.filename,
      mmproj: model.mmproj,
    }
  }

  const modelWithVocoder = model as ModelWithVocoder
  if (modelWithVocoder.vocoder) {
    return {
      ...baseDefinition,
      kind: 'tts',
      repo: model.repo,
      filename: model.filename,
      vocoder: modelWithVocoder.vocoder,
    }
  }

  return {
    ...baseDefinition,
    kind: 'text',
    repo: model.repo,
    filename: model.filename,
  }
}

export const createExampleModelDefinitions = (
  keys: ExampleModelKey[],
  initializeButtonText?: string,
) => keys.map((key) => createExampleModelDefinition(key, initializeButtonText))

export const isTextGenerationModel = (model: ModelConfig) => {
  const modelWithCapabilities = model as ModelWithCapabilities
  return (
    !model.mmproj &&
    !modelWithCapabilities.embedding &&
    !modelWithCapabilities.ranking &&
    !modelWithCapabilities.vocoder
  )
}

export const isEmbeddingModel = (model: ModelConfig) =>
  Boolean((model as ModelWithCapabilities).embedding)

export const isRankingModel = (model: ModelConfig) =>
  Boolean((model as ModelWithCapabilities).ranking)

export const getAllExampleModelDefinitions = (
  initializeButtonText?: string,
) =>
  createExampleModelDefinitions(
    Object.keys(MODELS) as ExampleModelKey[],
    initializeButtonText,
  )
