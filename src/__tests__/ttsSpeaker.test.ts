import { NativeModules } from 'react-native'
import { initLlama, LlamaContext, LlamaSpeaker } from '..'

jest.mock('..', () => require('../../jest/mock'))

// Capture mock function references from globals BEFORE initLlama's installJsi()
// moves them into jsiBindings and deletes them from global.
let mockReleaseSpeaker: jest.Mock
let mockGetFormattedAudioCompletion: jest.Mock
let ctx: LlamaContext

beforeAll(async () => {
  // install() populates the globals via ensureJSIFunctions
  await NativeModules.RNLlama.install()
  // Grab mock refs while they still live on global (before initLlama deletes them)
  const g = global as typeof globalThis & {
    llamaReleaseSpeaker: jest.Mock
    llamaGetFormattedAudioCompletion: jest.Mock
  }
  mockReleaseSpeaker = g.llamaReleaseSpeaker
  mockGetFormattedAudioCompletion = g.llamaGetFormattedAudioCompletion
  // initLlama calls installJsi → bindJsiFromGlobal → deletes keys from global
  ctx = await initLlama({ model: 'x.gguf' })
})

it('createSpeaker returns a LlamaSpeaker handle backed by the native id', async () => {
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
  expect(mockReleaseSpeaker).toHaveBeenCalledWith(ctx.id, 1)
})

it('getFormattedAudioCompletion forwards speaker.id when passed a LlamaSpeaker and returns no speakerEmb* fields', async () => {
  mockGetFormattedAudioCompletion.mockClear()

  const spk = await ctx.createSpeaker({
    refAudio: new Float32Array([0, 0.1, -0.1]),
    refAudioSampleRate: 24000,
  })

  const result = await ctx.getFormattedAudioCompletion({
    prompt: 'hello',
    speaker: spk,
  })

  // Native call must receive empty speakerStr and the speaker id as 4th arg
  expect(mockGetFormattedAudioCompletion).toHaveBeenCalledWith(
    ctx.id,
    '',
    'hello',
    spk.id,
  )

  // Return value must not carry any speakerEmb* fields
  expect(result).not.toHaveProperty('speakerEmbPrefix')
  expect(result).not.toHaveProperty('speakerEmbRows')
  expect(result).not.toHaveProperty('speakerEmbHiddenDim')
  expect(result).toHaveProperty('prompt')
  expect(result).toHaveProperty('flow')
})

it('encodeSpeaker does not exist on LlamaContext', () => {
  expect((ctx as any).encodeSpeaker).toBeUndefined()
})

