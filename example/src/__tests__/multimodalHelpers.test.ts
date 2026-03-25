import {
  createMultimodalSystemPrompt,
  createMultimodalWelcomeMessage,
} from '../features/multimodalHelpers'

describe('multimodal helpers', () => {
  it('builds prompts for multimodal capabilities', () => {
    expect(
      createMultimodalSystemPrompt({ vision: true, audio: false }),
    ).toContain('vision capabilities')
    expect(
      createMultimodalWelcomeMessage({ vision: true, audio: true }),
    ).toContain('share images and audio files')
  })

  it('falls back to the text-only message when no capabilities exist', () => {
    expect(createMultimodalSystemPrompt(null)).toContain('helpful AI assistant')
    expect(createMultimodalWelcomeMessage(null)).toContain(
      'ready to help with text conversations',
    )
  })
})
