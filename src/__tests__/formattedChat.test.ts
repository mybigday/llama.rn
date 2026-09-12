import { NativeModules } from 'react-native'
import { initLlama, LlamaContext } from '..'

jest.mock('..', () => require('../../jest/mock'))

// Grab the JSI mock before initLlama's installJsi() moves it off global.
let mockGetFormattedChat: jest.Mock
let ctx: LlamaContext

beforeAll(async () => {
  await NativeModules.RNLlama.install()
  const g = global as typeof globalThis & { llamaGetFormattedChat: jest.Mock }
  mockGetFormattedChat = g.llamaGetFormattedChat
  ctx = await initLlama({ model: 'x.gguf' })
})

beforeEach(() => {
  mockGetFormattedChat.mockClear()
  mockGetFormattedChat.mockResolvedValueOnce({ prompt: '', chat_format: 0 })
})

it('passes chat_template_kwargs and parallel_tool_calls to native as raw values', async () => {
  await ctx.getFormattedChat([{ role: 'user', content: 'hi' }], null, {
    jinja: true,
    parallel_tool_calls: true,
    chat_template_kwargs: { enable_thinking: false, style: 'brief', n: 2 },
    now: 1700000000,
  })

  const [, messages, , opts] = mockGetFormattedChat.mock.calls[0]
  // messages still cross as a JSON string
  expect(JSON.parse(messages)).toEqual([{ role: 'user', content: 'hi' }])
  // options cross as-is: no pre-stringified kwargs, boolean stays boolean
  expect(opts.parallel_tool_calls).toBe(true)
  expect(opts.chat_template_kwargs).toEqual({
    enable_thinking: false,
    style: 'brief',
    n: 2,
  })
  expect(opts.now).toBe(1700000000)
})

it('defaults parallel_tool_calls to false and leaves optional fields undefined', async () => {
  await ctx.getFormattedChat([{ role: 'user', content: 'hi' }], null, {
    jinja: true,
  })
  const opts = mockGetFormattedChat.mock.calls[0][3]
  expect(opts.parallel_tool_calls).toBe(false)
  expect(opts.chat_template_kwargs).toBeUndefined()
  expect(opts.tools).toBeUndefined()
  expect(opts.json_schema).toBeUndefined()
})
