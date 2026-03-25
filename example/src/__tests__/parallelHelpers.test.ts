import {
  buildParallelStatePath,
  formatParallelModeLabel,
  hashPrompt,
} from '../features/parallelHelpers'

describe('parallel helpers', () => {
  it('hashes prompts deterministically', () => {
    expect(hashPrompt('hello')).toBe(hashPrompt('hello'))
    expect(hashPrompt('hello')).not.toBe(hashPrompt('world'))
  })

  it('builds state cache paths from the model filename and prompt hash', () => {
    expect(
      buildParallelStatePath('/tmp/cache', '/models/demo.gguf', 'Hello world'),
    ).toMatch(/^\/tmp\/cache\/state_demo_[\da-z]+\.bin$/)
  })

  it('formats the parallel mode label', () => {
    expect(formatParallelModeLabel(true)).toBe('⚡ Parallel')
    expect(formatParallelModeLabel(false)).toBe('🔄 Single')
  })
})
