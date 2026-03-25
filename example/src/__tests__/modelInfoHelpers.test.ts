import {
  buildCustomModelFiles,
  buildTTSModelFiles,
  formatModelInfoValue,
} from '../features/modelInfoHelpers'

describe('model info helpers', () => {
  it('formats object values as pretty JSON', () => {
    expect(formatModelInfoValue({ a: 1 })).toContain('"a": 1')
  })

  it('builds multimodal custom model file groups when mmproj exists', () => {
    expect(
      buildCustomModelFiles(
        {
          id: 'demo',
          repo: 'repo',
          filename: 'model.gguf',
          quantization: 'Q4',
          mmprojFilename: 'mmproj.gguf',
          addedAt: 0,
        },
        '/tmp/model.gguf',
        '/tmp/mmproj.gguf',
      ),
    ).toHaveLength(2)
  })

  it('builds TTS model file groups', () => {
    expect(buildTTSModelFiles('/tmp/model.gguf', '/tmp/vocoder.gguf')).toEqual([
      { name: 'TTS Model', path: '/tmp/model.gguf' },
      { name: 'Vocoder', path: '/tmp/vocoder.gguf' },
    ])
  })
})
