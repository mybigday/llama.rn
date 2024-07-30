import { formatChat } from '../chat'

describe('formatChat', () => {
  it('should format chat messages', () => {
    const messages = [
      {
        role: 'user',
        content: 'Hello, world!',
      },
      {
        role: 'bot',
        content: [
          {
            text: 'Hello, user!',
          },
          {
            text: 'How are you?',
          },
        ],
      },
    ]

    const expected = [
      {
        role: 'user',
        content: 'Hello, world!',
      },
      {
        role: 'bot',
        content: 'Hello, user!\nHow are you?',
      },
    ]

    expect(formatChat(messages)).toEqual(expected)
  })

  it('should throw an error if the content is missing', () => {
    const messages = [
      {
        role: 'user',
      },
    ]

    expect(() => formatChat(messages)).toThrowError(
      "Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
    )
  })

  it('should throw an error if the content type is invalid', () => {
    const messages = [
      {
        role: 'user',
        content: 42,
      },
    ]

    expect(() => formatChat(messages)).toThrowError(
      "Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
    )
  })
})
