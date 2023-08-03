import { initLlama, releaseAllLlama } from '..'
import type { TokenData } from '..'

jest.mock('..', () => require('../../jest/mock'))

Math.random = () => 0.5

test('Mock', async () => {
  const context = await initLlama({
    model: 'test.bin',
  })
  expect(context.id).toBe(1)
  const events: TokenData[] = []
  const completionResult = await context.completion({
    prompt: 'Test',
  }, data => {
    events.push(data)
  })
  expect(events).toMatchSnapshot()
  expect(completionResult).toMatchSnapshot()

  await context.release()
  await releaseAllLlama()
})
