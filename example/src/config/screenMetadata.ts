import type { ExampleScreenDefinition } from '../types/example'

export const EXAMPLE_SCREEN_METADATA: Array<
  Omit<ExampleScreenDefinition, 'component'>
> = [
  {
    routeName: 'SimpleChat',
    title: 'Simple Chat',
    homeLabel: 'Simple Chat',
    emoji: '💬',
  },
  {
    routeName: 'TextCompletion',
    title: 'Text Completion',
    homeLabel: 'Text Completion',
    emoji: '✏️',
  },
  {
    routeName: 'ParallelDecoding',
    title: 'Parallel Decoding',
    homeLabel: 'Parallel Decoding',
    emoji: '⚡',
  },
  {
    routeName: 'Multimodal',
    title: 'Multimodal Chat',
    homeLabel: 'Multimodal',
    emoji: '👁️',
  },
  {
    routeName: 'ToolCalling',
    title: 'Tool Calling & MCP',
    homeLabel: 'Tool Calling & MCP',
    emoji: '🛠️',
  },
  {
    routeName: 'Embeddings',
    title: 'Vector Search (in-memory) & Rerank',
    homeLabel: 'Vector Search (in-memory) & Rerank',
    emoji: '🔍',
  },
  {
    routeName: 'TTS',
    title: 'Text-to-Speech',
    homeLabel: 'Text-to-Speech (OuteTTS)',
    emoji: '🔊',
  },
  {
    routeName: 'ModelInfo',
    title: 'Model Info',
    homeLabel: 'Model Info',
    emoji: '📊',
  },
  {
    routeName: 'Bench',
    title: 'Bench',
    homeLabel: 'Bench',
    emoji: '🏋️',
  },
  {
    routeName: 'StressTest',
    title: 'Stress Test',
    homeLabel: 'Stress Test',
    emoji: '🔥',
  },
]
