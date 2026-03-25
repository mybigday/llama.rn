import { EXAMPLE_SCREEN_METADATA } from '../config/screenMetadata'

describe('example screen registry', () => {
  it('contains a unique entry for each demo screen', () => {
    const routeNames = EXAMPLE_SCREEN_METADATA.map((screen) => screen.routeName)
    expect(new Set(routeNames).size).toBe(routeNames.length)
    expect(routeNames).toEqual([
      'SimpleChat',
      'TextCompletion',
      'ParallelDecoding',
      'Multimodal',
      'ToolCalling',
      'Embeddings',
      'TTS',
      'ModelInfo',
      'Bench',
      'StressTest',
    ])
  })

  it('defines home metadata for every screen', () => {
    EXAMPLE_SCREEN_METADATA.forEach((screen) => {
      expect(screen.title.length).toBeGreaterThan(0)
      expect(screen.homeLabel.length).toBeGreaterThan(0)
      expect(screen.emoji.length).toBeGreaterThan(0)
    })
  })
})
