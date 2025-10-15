import { initLlama, releaseAllLlama } from '..'
import type { TokenData } from '..'

jest.mock('..', () => require('../../jest/mock'))

Math.random = () => 0.5

test('Mock', async () => {
  const context = await initLlama({
    model: 'test.gguf',
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

  await context.initMultimodal({
    path: 'mmproj-test.gguf',
  })
  expect(await context.isMultimodalEnabled()).toBe(true)

  await context.releaseMultimodal()
  expect(await context.isMultimodalEnabled()).toBe(false)

  await context.initVocoder({
    path: 'vocoder-test.gguf',
  })
  expect(await context.isVocoderEnabled()).toBe(true)

  expect(await context.getFormattedAudioCompletion(null, 'Hello world')).toHaveProperty('prompt')

  await context.releaseVocoder()
  expect(await context.isVocoderEnabled()).toBe(false)

  await context.release()
  await releaseAllLlama()
})

test('Parallel APIs - completion', async () => {
  const context = await initLlama({
    model: 'test.gguf',
    n_parallel: 4,
  })

  // Enable parallel mode
  await context.parallel.enable({ n_parallel: 2 })

  // Queue a completion
  const tokens: TokenData[] = []
  const { requestId, promise, stop } = await context.parallel.completion(
    { prompt: 'Hello' },
    (_reqId, data) => {
      tokens.push(data)
    },
  )

  expect(typeof requestId).toBe('number')
  expect(typeof stop).toBe('function')

  // Wait for completion
  const result = await promise
  expect(result).toMatchSnapshot('parallel completion result')
  expect(tokens.length).toBeGreaterThan(0)
  expect(tokens).toMatchSnapshot('parallel completion tokens')

  await context.release()
})

test('Parallel APIs - multiple completions', async () => {
  const context = await initLlama({
    model: 'test.gguf',
    n_parallel: 4,
  })

  await context.parallel.enable({ n_parallel: 3 })

  // Queue multiple completions in parallel
  const req1 = await context.parallel.completion({ prompt: 'First' })
  const req2 = await context.parallel.completion({ prompt: 'Second' })
  const req3 = await context.parallel.completion({ prompt: 'Third' })

  expect(req1.requestId).not.toBe(req2.requestId)
  expect(req2.requestId).not.toBe(req3.requestId)

  // Wait for all to complete
  const [result1, result2, result3] = await Promise.all([
    req1.promise,
    req2.promise,
    req3.promise,
  ])

  expect(result1).toHaveProperty('text')
  expect(result2).toHaveProperty('text')
  expect(result3).toHaveProperty('text')

  await context.release()
})

test('Parallel APIs - completion with stop', async () => {
  const context = await initLlama({
    model: 'test.gguf',
    n_parallel: 2,
  })

  await context.parallel.enable()

  const { requestId, stop } = await context.parallel.completion({
    prompt: 'Test',
  })

  expect(typeof requestId).toBe('number')

  // Stop the completion
  await stop()

  await context.release()
})

test('Parallel APIs - embedding', async () => {
  const context = await initLlama({
    model: 'test.gguf',
  })

  const { requestId, promise } = await context.parallel.embedding('Test text')

  expect(typeof requestId).toBe('number')

  const result = await promise
  expect(result).toHaveProperty('embedding')
  expect(Array.isArray(result.embedding)).toBe(true)
  expect(result.embedding.length).toBe(768)

  await context.release()
})

test('Parallel APIs - multiple embeddings', async () => {
  const context = await initLlama({
    model: 'test.gguf',
  })

  // Queue multiple embeddings in parallel
  const req1 = await context.parallel.embedding('First text')
  const req2 = await context.parallel.embedding('Second text')
  const req3 = await context.parallel.embedding('Third text')

  expect(req1.requestId).not.toBe(req2.requestId)
  expect(req2.requestId).not.toBe(req3.requestId)

  // Wait for all to complete
  const [result1, result2, result3] = await Promise.all([
    req1.promise,
    req2.promise,
    req3.promise,
  ])

  expect(result1.embedding.length).toBe(768)
  expect(result2.embedding.length).toBe(768)
  expect(result3.embedding.length).toBe(768)

  await context.release()
})

test('Parallel APIs - rerank', async () => {
  const context = await initLlama({
    model: 'test.gguf',
  })

  const query = 'What is the capital of France?'
  const documents = [
    'Paris is the capital of France.',
    'London is the capital of England.',
    'Berlin is the capital of Germany.',
  ]

  const { requestId, promise } = await context.parallel.rerank(query, documents)

  expect(typeof requestId).toBe('number')

  const results = await promise
  expect(Array.isArray(results)).toBe(true)
  expect(results.length).toBe(3)

  // Check that results are sorted by score (descending)
  for (let i = 0; i < results.length - 1; i += 1) {
    expect(results[i]?.score).toBeGreaterThanOrEqual(results[i + 1]?.score ?? 0)
  }

  // Each result should have index, score, and document
  results.forEach((result) => {
    expect(result).toHaveProperty('index')
    expect(result).toHaveProperty('score')
    expect(result).toHaveProperty('document')
    expect(typeof result.index).toBe('number')
    expect(typeof result.score).toBe('number')
    expect(typeof result.document).toBe('string')
  })

  await context.release()
})

test('Parallel APIs - configure and disable', async () => {
  const context = await initLlama({
    model: 'test.gguf',
    n_parallel: 8,
  })

  // Configure (enables if not already enabled)
  await context.parallel.configure({ n_parallel: 4, n_batch: 256 })

  // Reconfigure with different settings
  await context.parallel.configure({ n_parallel: 6 })

  // Disable parallel mode
  await context.parallel.disable()

  await context.release()
})
