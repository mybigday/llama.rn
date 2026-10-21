import { NativeModules } from 'react-native'
import { initLlama, LlamaContext, LlamaSpeaker } from '..'

jest.mock('..', () => require('../../jest/mock'))

// Capture mock function references from globals BEFORE initLlama's installJsi()
// moves them into jsiBindings and deletes them from global.
let mockReleaseSpeaker: jest.Mock
let mockGetFormattedAudioCompletion: jest.Mock
let mockGetTTSCapabilities: jest.Mock
let ctx: LlamaContext

beforeAll(async () => {
  // install() populates the globals via ensureJSIFunctions
  await NativeModules.RNLlama.install()
  // Grab mock refs while they still live on global (before initLlama deletes them)
  const g = global as typeof globalThis & {
    llamaReleaseSpeaker: jest.Mock
    llamaGetFormattedAudioCompletion: jest.Mock
    llamaGetTTSCapabilities: jest.Mock
  }
  mockReleaseSpeaker = g.llamaReleaseSpeaker
  mockGetFormattedAudioCompletion = g.llamaGetFormattedAudioCompletion
  mockGetTTSCapabilities = g.llamaGetTTSCapabilities
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

  // Native call must receive a null speaker payload and the speaker id as 4th arg
  expect(mockGetFormattedAudioCompletion).toHaveBeenCalledWith(
    ctx.id,
    null,
    'hello',
    spk.id,
    undefined,
    {},
  )

  // Return value must not carry any speakerEmb* fields
  expect(result).not.toHaveProperty('speakerEmbPrefix')
  expect(result).not.toHaveProperty('speakerEmbRows')
  expect(result).not.toHaveProperty('speakerEmbHiddenDim')
  expect(result).toHaveProperty('prompt')
  expect(result).toHaveProperty('flow')
})

it('getFormattedAudioCompletion rejects structured speaker payloads and unknown voices', async () => {
  const payload = { ref_phones: 'h ə l oʊ', ref_codes: [11, 22, 33] }
  await expect(
    ctx.getFormattedAudioCompletion({
      prompt: 'hello',
      speaker: payload as unknown as string,
    }),
  ).rejects.toThrow(TypeError)
  await expect(
    ctx.getFormattedAudioCompletion({ prompt: 'hello', speaker: 'nobody' }),
  ).rejects.toThrow("Unknown voice 'nobody'")
})

it('getFormattedAudioCompletion forwards the request options in snake_case and returns completionParams', async () => {
  mockGetFormattedAudioCompletion.mockClear()

  const f = await ctx.getFormattedAudioCompletion({
    prompt: 'hello',
    request: { seed: 7, topP: 0.9, maxFrames: 100, lmGuidance: 0 },
  })

  expect(mockGetFormattedAudioCompletion).toHaveBeenCalledWith(
    ctx.id,
    null,
    'hello',
    -1,
    undefined,
    { seed: 7, top_p: 0.9, max_frames: 100, lm_guidance: 0 },
  )
  expect(f.completionParams.prompt).toBe('hello')
})

it('getFormattedAudioCompletion resolves a NeuTTS default to the language default codec voice', async () => {
  const voice = (name: string, isDefault = false) => ({
    name,
    refText: '',
    isDefault,
  })
  mockGetTTSCapabilities.mockResolvedValueOnce({
    type: 5,
    promptKind: 'neutts',
    family: 'neutts',
    requiresPhonemes: true,
    defaultLanguage: 'en-us',
    builtinVoices: [voice('jo', true), voice('dave'), voice('greta')],
    canSpeakerFromAudio: true,
    samplingDefaults: {},
  })
  mockGetFormattedAudioCompletion.mockClear()

  await ctx.getFormattedAudioCompletion({ prompt: 'hallo', language: 'de' })

  // A builtin codec voice beats the JS preset of the same name, and 'de'
  // picks greta rather than the codec's own default (jo).
  expect(mockGetFormattedAudioCompletion).toHaveBeenCalledWith(
    ctx.id,
    { codec_voice: 'greta' },
    'hallo',
    -1,
    undefined,
    {},
  )
})

it('getFormattedAudioCompletion forwards instruct', async () => {
  mockGetFormattedAudioCompletion.mockClear()

  await ctx.getFormattedAudioCompletion({
    prompt: 'hello',
    instruct: 'Speak cheerfully',
  })

  expect(mockGetFormattedAudioCompletion).toHaveBeenCalledWith(
    ctx.id,
    null,
    'hello',
    -1,
    'Speak cheerfully',
    {},
  )
})

it('encodeSpeaker does not exist on LlamaContext', () => {
  expect((ctx as any).encodeSpeaker).toBeUndefined()
})

