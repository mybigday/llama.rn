import { NativeModules } from 'react-native'
import { initLlama, LlamaContext } from '..'

jest.mock('..', () => require('../../jest/mock'))

// Large numeric payloads cross JSI as typed arrays (one ArrayBuffer copy
// instead of one call per element) while the public API keeps number[].
let mockDecodeAudioTokens: jest.Mock
let mockDecodeAudioEmbeddings: jest.Mock
let ctx: LlamaContext

beforeAll(async () => {
  await NativeModules.RNLlama.install()
  const g = global as typeof globalThis & {
    llamaDecodeAudioTokens: jest.Mock
    llamaDecodeAudioEmbeddings: jest.Mock
  }
  mockDecodeAudioTokens = g.llamaDecodeAudioTokens
  mockDecodeAudioEmbeddings = g.llamaDecodeAudioEmbeddings
  ctx = await initLlama({ model: 'x.gguf' })
})

it('decodeAudioTokens sends an Int32Array and returns a plain array', async () => {
  const pcm = await ctx.decodeAudioTokens([1, 2, 3])
  const sent = mockDecodeAudioTokens.mock.calls[0]![1]
  expect(sent).toBeInstanceOf(Int32Array)
  expect(Array.from(sent)).toEqual([1, 2, 3])
  expect(Array.isArray(pcm)).toBe(true)
  expect(pcm).toEqual([0.25, -0.5])
})

it('decodeAudioTokens passes an Int32Array through untouched', async () => {
  const tokens = new Int32Array([7, 8])
  await ctx.decodeAudioTokens(tokens)
  expect(mockDecodeAudioTokens.mock.calls[1]![1]).toBe(tokens)
})

it('decodeAudioEmbeddings sends a Float32Array and returns a plain array', async () => {
  const pcm = await ctx.decodeAudioEmbeddings([0.5, 1.5], 2)
  const sent = mockDecodeAudioEmbeddings.mock.calls[0]![1]
  expect(sent).toBeInstanceOf(Float32Array)
  expect(Array.from(sent)).toEqual([0.5, 1.5])
  expect(mockDecodeAudioEmbeddings.mock.calls[0]![2]).toBe(2)
  expect(pcm).toEqual([0.125])
})

it('embedding results are plain arrays for both single and parallel paths', async () => {
  const single = await ctx.embedding('hello')
  expect(Array.isArray(single.embedding)).toBe(true)
  expect(single.embedding.length).toBeGreaterThan(0)

  await ctx.parallel.enable()
  const { promise } = await ctx.parallel.embedding('hello')
  const queued = await promise
  expect(Array.isArray(queued.embedding)).toBe(true)
  expect(queued.embedding).toEqual(single.embedding)
})
