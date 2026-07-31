import { NativeModules } from 'react-native'
import { initLlama, LlamaSpeaker } from '..'

jest.mock('..', () => require('../../jest/mock'))

it('createSpeaker returns a LlamaSpeaker handle backed by the native id', async () => {
  // install() populates globals; capture the jest.fn reference before initLlama
  // moves it into jsiBindings (both code paths call the same function object).
  await NativeModules.RNLlama.install()
  const { llamaReleaseSpeaker } = global as typeof globalThis & {
    llamaReleaseSpeaker: jest.Mock
  }
  const ctx = await initLlama({ model: 'x.gguf' })
  const spk = await ctx.createSpeaker({
    refAudio: new Float32Array([0, 0.1, -0.1]),
    refAudioSampleRate: 24000,
  })
  expect(spk).toBeInstanceOf(LlamaSpeaker)
  expect(spk.id).toBe(1)
  expect(spk.baked).toBe(false)
  await spk.bake()
  expect(spk.baked).toBe(true)
  expect(spk.rows).toBe(34)
  await spk.release()
  expect(llamaReleaseSpeaker).toHaveBeenCalledWith(ctx.id, 1)
})

