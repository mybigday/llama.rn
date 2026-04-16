export interface StressTestResult {
  name: string
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped'
  error?: string
  duration?: number
  raceConditionsCaught?: number
  totalCycles?: number
}

export const RACE_CONDITION_ERRORS = [
  'RNLLAMA_NULL_CONTEXT',
  'RNLLAMA_NULL_COMPLETION',
  'RNLLAMA_NULL_LLAMA_CONTEXT',
  'RNLLAMA_NULL_SLOT',
]

export const isRaceConditionError = (error: any): boolean => {
  const message = error?.message || ''
  const errorText = String(error)
  const combined = `${message} ${errorText}`
  return RACE_CONDITION_ERRORS.some((code) => combined.includes(code))
}

export const formatStressTestReport = ({
  results,
  modelName,
  modelPath,
  startTime,
  endTime,
}: {
  results: StressTestResult[]
  modelName: string
  modelPath: string
  startTime: Date
  endTime: Date
}) => {
  const totalDuration = endTime.getTime() - startTime.getTime()
  const passedClean = results.filter(
    (result) => result.status === 'passed' && !result.raceConditionsCaught,
  ).length
  const passedWithRaces = results.filter(
    (result) =>
      result.status === 'passed' && (result.raceConditionsCaught ?? 0) > 0,
  ).length
  const failed = results.filter((result) => result.status === 'failed').length
  const totalRaceCaught = results.reduce(
    (accumulator, result) => accumulator + (result.raceConditionsCaught || 0),
    0,
  )

  const getStatusIcon = (
    status: StressTestResult['status'],
    racesCaught?: number,
  ) => {
    if (status === 'passed' && racesCaught && racesCaught > 0) return '[WARN]'
    if (status === 'passed') return '[PASS]'
    if (status === 'failed') return '[FAIL]'
    return '[SKIP]'
  }

  const lines: string[] = [
    '═══════════════════════════════════════════',
    '          STRESS TEST REPORT',
    '═══════════════════════════════════════════',
    '',
    `Timestamp: ${endTime.toISOString()}`,
    `Model: ${modelName}`,
    `Path: ${modelPath}`,
    '',
    '───────────────────────────────────────────',
    '               SUMMARY',
    '───────────────────────────────────────────',
    '',
    `Total Tests:              ${results.length}`,
    `Passed (clean):           ${passedClean}`,
    `Passed (w/ races caught): ${passedWithRaces}`,
    `Failed:                   ${failed}`,
    `Total Duration:           ${totalDuration}ms`,
    `Race Conditions Caught:   ${totalRaceCaught}`,
    '',
    '───────────────────────────────────────────',
    '            TEST RESULTS',
    '───────────────────────────────────────────',
    '',
  ]

  results.forEach((result) => {
    const statusIcon = getStatusIcon(
      result.status,
      result.raceConditionsCaught,
    )
    const durationText =
      result.duration !== undefined ? `${result.duration}ms` : '-'
    const raceText = result.raceConditionsCaught
      ? `${result.raceConditionsCaught}/${result.totalCycles || '?'} races`
      : ''

    lines.push(`${statusIcon} ${result.name}`)
    lines.push(`       ${[raceText, durationText].filter(Boolean).join(' | ')}`)
    if (result.error) {
      lines.push(`       Error: ${result.error}`)
    }
    lines.push('')
  })

  return lines.join('\n')
}
