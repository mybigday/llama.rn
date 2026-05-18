import type { ComponentType } from 'react'
import type { LlamaContext } from '../../../src'

export type ExampleRouteName =
  | 'SimpleChat'
  | 'TextCompletion'
  | 'StructuredOutput'
  | 'ParallelDecoding'
  | 'Multimodal'
  | 'ToolCalling'
  | 'Embeddings'
  | 'TTS'
  | 'ModelInfo'
  | 'Bench'
  | 'StressTest'

export interface ExampleScreenDefinition {
  routeName: ExampleRouteName
  title: string
  homeLabel: string
  emoji: string
  component: ComponentType<any>
}

interface ExampleModelDefinitionBase {
  key: string
  title: string
  size: string
  initializeButtonText?: string
}

export interface ExampleTextModelDefinition
  extends ExampleModelDefinitionBase {
  kind: 'text'
  repo: string
  filename: string
}

export interface ExampleMultimodalModelDefinition
  extends ExampleModelDefinitionBase {
  kind: 'multimodal'
  repo: string
  filename: string
  mmproj: string
}

export interface ExampleTTSModelDefinition
  extends ExampleModelDefinitionBase {
  kind: 'tts'
  repo: string
  filename: string
  vocoder: {
    repo: string
    filename: string
    size: string
  }
}

export type ExampleModelDefinition =
  | ExampleTextModelDefinition
  | ExampleMultimodalModelDefinition
  | ExampleTTSModelDefinition

export interface ExampleInitState {
  isLoading: boolean
  isModelReady: boolean
  initProgress: number
}

export interface ExampleContextState extends ExampleInitState {
  context: LlamaContext | null
}
