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

  it('keys state cache paths by media so identical prompts do not collide', () => {
    const text = buildParallelStatePath(
      '/tmp/cache',
      '/models/demo.gguf',
      'Hello world',
    )
    const withImage = buildParallelStatePath(
      '/tmp/cache',
      '/models/demo.gguf',
      'Hello world',
      ['/tmp/a.jpg'],
    )
    const withOtherImage = buildParallelStatePath(
      '/tmp/cache',
      '/models/demo.gguf',
      'Hello world',
      ['/tmp/b.jpg'],
    )
    expect(withImage).not.toBe(text)
    expect(withImage).not.toBe(withOtherImage)
    // No media keeps the legacy text-only path
    expect(
      buildParallelStatePath('/tmp/cache', '/models/demo.gguf', 'Hello world', []),
    ).toBe(text)
  })

  it('formats the parallel mode label', () => {
    expect(formatParallelModeLabel(true)).toBe('⚡ Parallel')
    expect(formatParallelModeLabel(false)).toBe('🔄 Single')
  })
})
