import {
  buildGeneratedCompletionText,
  getDefaultTextCompletionMessages,
  getTokenHeatmapColor,
} from '../features/textCompletionHelpers'

describe('text completion helpers', () => {
  it('returns the default chat seed messages', () => {
    expect(getDefaultTextCompletionMessages()).toHaveLength(2)
  })

  it('builds the editable combined prompt text', () => {
    expect(
      buildGeneratedCompletionText('Hello', [
        { token: ' world' },
        { token: '!' },
      ]),
    ).toBe('Hello world!')
  })

  it('maps probabilities to a color ramp', () => {
    expect(getTokenHeatmapColor(undefined)).toBe('transparent')
    expect(getTokenHeatmapColor(1)).toContain('0, 255')
  })
})
