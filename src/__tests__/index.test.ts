import { initLlama, releaseAllLlama } from '..'
import type { TokenData } from '..'

jest.mock('..', () => require('../../jest/mock'))

Math.random = () => 0.5

test('Mock', async () => {
  const context = await initLlama({
    model: 'test.bin',
  })
  expect(context.id).toBe(0)
  const events: TokenData[] = []
  const completionResult = await context.completion({
    prompt: 'Test',
  }, data => {
    events.push(data)
  })
  expect(events).toMatchSnapshot()
  expect(completionResult).toMatchSnapshot('completion result')

  expect(await context.bench(512, 128, 1, 3)).toMatchSnapshot('bench')

  await context.release()
  await releaseAllLlama()
})
