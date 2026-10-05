import type { DecisionQuestion, DecisionRequest } from '../../../src'

export type DecisionQuestionType = DecisionQuestion['type']

export type EditableOption = { key: string; description: string }

// One question as the editor holds it: every type's criteria are kept, so
// switching the type back and forth does not lose what was typed
export type EditableQuestion = {
  uid: string
  id: string
  type: DecisionQuestionType
  instructions: string
  options: EditableOption[] // choice
  levels: string[] // score, lowest first
  noulTrue: string // noul, optional
  noulFalse: string
}

export type DecisionPreset = {
  name: string
  state: string
  questions: EditableQuestion[]
}

let nextUid = 0
const newUid = () => `q${(nextUid += 1)}`

export const createQuestion = (
  fields: Partial<Omit<EditableQuestion, 'uid'>> = {},
): EditableQuestion => ({
  uid: newUid(),
  id: '',
  type: 'choice',
  instructions: '',
  options: [{ key: '', description: '' }],
  levels: ['', ''],
  noulTrue: '',
  noulFalse: '',
  ...fields,
})

const choice = (
  id: string,
  instructions: string,
  options: Record<string, string>,
) =>
  createQuestion({
    id,
    type: 'choice',
    instructions,
    options: Object.entries(options).map(([key, description]) => ({
      key,
      description,
    })),
  })

const score = (id: string, instructions: string, levels: string[]) =>
  createQuestion({ id, type: 'score', instructions, levels })

const noul = (id: string, instructions: string, yes = '', no = '') =>
  createQuestion({
    id,
    type: 'noul',
    instructions,
    noulTrue: yes,
    noulFalse: no,
  })

export const DECISION_PRESETS: Array<() => DecisionPreset> = [
  () => ({
    name: 'Drive-thru',
    state: JSON.stringify(
      {
        utterance:
          'uh can I get a medium fries and, actually make that a large cola too',
        stage: 'menu',
        cart: [],
      },
      null,
      2,
    ),
    questions: [
      noul('is_ordering', 'The customer is placing an order'),
      choice('ordered_product', 'Which product did the customer order first?', {
        'fries-m': 'Fries M (medium fries)',
        'cola-l': 'Cola L (large cola)',
        burger: 'Mos Burger',
        none: 'Nothing ordered',
      }),
      score('mood', 'How impatient does the customer sound?', [
        'Calm',
        'Neutral',
        'Impatient',
      ]),
      noul(
        'wants_drink',
        'The customer wants a drink',
        'a drink is mentioned',
        'no drink',
      ),
    ],
  }),
  () => ({
    name: '點餐（中文）',
    state: '嗯我要一份中薯，啊還有一杯大杯可樂',
    questions: [
      noul('is_ordering', '顧客正在點餐'),
      choice('first_item', '顧客第一個點的是什麼？', {
        中薯: '中份薯條',
        大可樂: '大杯可樂',
        漢堡: '摩斯漢堡',
        沒有: '沒有點餐',
      }),
      score('mood', '顧客聽起來有多不耐煩？', ['平靜', '普通', '不耐煩']),
    ],
  }),
  () => ({
    name: 'Support triage',
    state:
      'I was charged twice for my order last week and nobody has replied to my emails. I want my money back today.',
    questions: [
      choice('route', 'Which team should handle this?', {
        billing: 'Payments, charges, refunds',
        shipping: 'Delivery and tracking',
        technical: 'App or website problems',
        account: 'Login and profile',
      }),
      score('urgency', 'How urgent is this?', [
        'Can wait',
        'This week',
        'Today',
        'Right now',
      ]),
      noul('angry', 'The customer is angry'),
      noul('wants_refund', 'The customer asks for a refund'),
    ],
  }),
  () => ({
    name: 'Review',
    state:
      'The noodles were great and arrived hot, but the delivery took almost an hour and it was pricey for the portion.',
    questions: [
      score('sentiment', 'Overall sentiment of the review', [
        'Very negative',
        'Negative',
        'Mixed',
        'Positive',
        'Very positive',
      ]),
      choice('main_complaint', 'What is the main complaint?', {
        food: 'Taste or quality of the food',
        delivery: 'Speed or condition of the delivery',
        price: 'Price or value',
        none: 'No complaint',
      }),
      noul('mentions_price', 'The review mentions the price'),
    ],
  }),
  () => ({
    name: 'Moderation',
    state: 'Click here to claim your FREE prize!!! Limited time only, act now',
    questions: [
      noul('is_spam', 'This message is spam or a scam'),
      score('toxicity', 'How toxic is this message?', [
        'Not toxic',
        'Mildly toxic',
        'Very toxic',
      ]),
      choice('category', 'What kind of message is this?', {
        promotion: 'Advertising or promotion',
        question: 'A question',
        chat: 'Casual conversation',
        harassment: 'Harassment or insult',
      }),
    ],
  }),
]

// A state that looks like JSON is sent as JSON, anything else as text
const parseState = (state: string) => {
  const trimmed = state.trim()
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      return JSON.parse(trimmed)
    } catch {
      throw new Error('The state looks like JSON but does not parse')
    }
  }
  return state
}

// Throws an Error that says which question is incomplete
export const buildDecisionRequest = (
  state: string,
  questions: EditableQuestion[],
): DecisionRequest => {
  if (!state.trim()) throw new Error('The state is empty')
  if (questions.length === 0) throw new Error('Add at least one question')

  // entries and Sets rather than plain-object lookups: an ID or a key such as
  // `constructor` or `__proto__` is just a name here
  const built: Array<[string, DecisionQuestion]> = []
  const ids = new Set<string>()
  questions.forEach((q, index) => {
    const name = q.id.trim() || `#${index + 1}`
    const fail = (message: string) => {
      throw new Error(`Question ${name}: ${message}`)
    }
    const id = q.id.trim()
    if (!id) fail('the ID is empty')
    if (ids.has(id)) fail('the ID is used twice')
    ids.add(id)
    if (!q.instructions.trim()) fail('the question is empty')

    if (q.type === 'choice') {
      const options = q.options.filter((o) => o.key.trim())
      if (options.length === 0) fail('add at least one option')
      const keys = new Set<string>()
      const criteria = Object.fromEntries(
        options.map((o) => {
          const key = o.key.trim()
          if (keys.has(key)) fail(`option "${key}" is listed twice`)
          keys.add(key)
          return [key, o.description.trim() || null]
        }),
      )
      built.push([id, { type: 'choice', instructions: q.instructions, criteria }])
    } else if (q.type === 'score') {
      const levels = q.levels.map((l) => l.trim()).filter(Boolean)
      if (levels.length < 2 || levels.length > 10) fail('a score needs 2 to 10 levels')
      built.push([id, { type: 'score', instructions: q.instructions, criteria: levels }])
    } else {
      const criteria: { true?: string; false?: string } = {}
      if (q.noulTrue.trim()) criteria.true = q.noulTrue.trim()
      if (q.noulFalse.trim()) criteria.false = q.noulFalse.trim()
      built.push([
        id,
        {
          type: 'noul',
          instructions: q.instructions,
          ...(Object.keys(criteria).length > 0 ? { criteria } : {}),
        },
      ])
    }
  })

  return { state: parseState(state), questions: Object.fromEntries(built) }
}
