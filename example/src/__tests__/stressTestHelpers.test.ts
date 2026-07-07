import {
  formatStressTestReport,
  isRaceConditionError,
} from '../features/stressTestHelpers'

describe('stress test helpers', () => {
  it('detects known race condition markers', () => {
    expect(
      isRaceConditionError(new Error('RNLLAMA_NULL_CONTEXT happened')),
    ).toBe(true)
    expect(isRaceConditionError(new Error('other failure'))).toBe(false)
  })

  it('formats a readable stress test report', () => {
    const report = formatStressTestReport({
      results: [
        {
          name: 'Rapid Start/Stop',
          status: 'passed',
          duration: 12,
          raceConditionsCaught: 1,
          totalCycles: 3,
        },
      ],
      modelName: 'Demo',
      modelPath: '/tmp/demo.gguf',
      startTime: new Date('2025-01-01T00:00:00.000Z'),
      endTime: new Date('2025-01-01T00:00:01.000Z'),
    })

    expect(report).toContain('STRESS TEST REPORT')
    expect(report).toContain('Rapid Start/Stop')
    expect(report).toContain('1/3 races')
  })
})
