import {
  DECISION_PRESETS,
  buildDecisionRequest,
  createQuestion,
} from '../features/decisionHelpers'

describe('buildDecisionRequest', () => {
  it('builds every preset into a valid request', () => {
    DECISION_PRESETS.forEach((makePreset) => {
      const preset = makePreset()
      const request = buildDecisionRequest(preset.state, preset.questions)
      expect(Object.keys(request.questions)).toEqual(
        preset.questions.map((q) => q.id),
      )
    })
  })

  it('sends a JSON-looking state as JSON and anything else as text', () => {
    const questions = [createQuestion({ id: 'q', type: 'noul', instructions: 'x' })]
    expect(buildDecisionRequest('{"a": 1}', questions).state).toEqual({ a: 1 })
    expect(buildDecisionRequest('hello', questions).state).toBe('hello')
    expect(() => buildDecisionRequest('{oops', questions)).toThrow('does not parse')
  })

  it('maps each question type to its criteria', () => {
    const request = buildDecisionRequest('s', [
      createQuestion({
        id: 'c',
        type: 'choice',
        instructions: 'pick',
        options: [
          { key: 'a', description: 'first' },
          { key: 'b', description: '' },
          { key: ' ', description: 'ignored' },
        ],
      }),
      createQuestion({ id: 's', type: 'score', instructions: 'rate', levels: ['lo', '', 'hi'] }),
      createQuestion({ id: 'n', type: 'noul', instructions: 'yes?', noulTrue: 'yes means' }),
    ])
    expect(request.questions).toEqual({
      c: { type: 'choice', instructions: 'pick', criteria: { a: 'first', b: null } },
      s: { type: 'score', instructions: 'rate', criteria: ['lo', 'hi'] },
      n: { type: 'noul', instructions: 'yes?', criteria: { true: 'yes means' } },
    })
  })

  it('names the incomplete question', () => {
    expect(() => buildDecisionRequest('s', [])).toThrow('at least one question')
    expect(() =>
      buildDecisionRequest('s', [
        createQuestion({ id: 'a', type: 'noul', instructions: 'x' }),
        createQuestion({ id: 'a', type: 'noul', instructions: 'y' }),
      ]),
    ).toThrow('Question a: the ID is used twice')
    expect(() =>
      buildDecisionRequest('s', [
        createQuestion({ id: 'r', type: 'score', instructions: 'x', levels: ['only'] }),
      ]),
    ).toThrow('Question r: a score needs 2 to 10 levels')
  })
})
