import {
  buildConversationMessages,
  createSystemTextMessage,
} from '../features/chatHelpers'

describe('chat helpers', () => {
  it('prepends the system prompt and preserves recent conversation order', () => {
    const messages = [
      {
        type: 'text',
        author: { id: 'assistant' },
        text: 'Second response',
        metadata: {},
      },
      {
        type: 'text',
        author: { id: 'user' },
        text: 'Second prompt',
        metadata: {},
      },
      {
        type: 'text',
        author: { id: 'assistant' },
        text: 'First response',
        metadata: {},
      },
      {
        type: 'text',
        author: { id: 'user' },
        text: 'First prompt',
        metadata: {},
      },
    ] as any

    expect(buildConversationMessages(messages, 'system prompt', 'user')).toEqual(
      [
        { role: 'system', content: 'system prompt' },
        { role: 'user', content: 'First prompt', reasoning_content: undefined },
        {
          role: 'assistant',
          content: 'First response',
          reasoning_content: undefined,
        },
        { role: 'user', content: 'Second prompt', reasoning_content: undefined },
        {
          role: 'assistant',
          content: 'Second response',
          reasoning_content: undefined,
        },
      ],
    )
  })

  it('creates system chat messages with system metadata', () => {
    expect(createSystemTextMessage('hello').metadata).toMatchObject({
      system: true,
    })
  })
})
