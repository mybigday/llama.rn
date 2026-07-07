import type { MessageType } from '@flyerhq/react-native-chat-ui'
import type { LLMMessage } from '../utils/llmMessages'

export const CHAT_USER = { id: 'user' }
export const CHAT_ASSISTANT = { id: 'assistant' }

export const createMessageId = () => Math.random().toString(36).substr(2, 9)

export const buildConversationMessages = (
  messages: MessageType.Any[],
  systemPrompt: string,
  userId: string,
  limit = 10,
): LLMMessage[] => {
  const conversationMessages: LLMMessage[] = [
    {
      role: 'system',
      content: systemPrompt,
    },
  ]

  const recentMessages = messages
    .filter(
      (message): message is MessageType.Text =>
        message.type === 'text' && !message.metadata?.system,
    )
    .reverse()
    .slice(-limit)
    .map((message) => ({
      role: message.author.id === userId ? ('user' as const) : ('assistant' as const),
      content: message.text,
      reasoning_content: message.metadata?.completionResult?.reasoning_content,
    }))

  return [...conversationMessages, ...recentMessages]
}

export const createSystemTextMessage = (
  text: string,
  metadata: Record<string, unknown> = {},
): MessageType.Text => ({
  author: CHAT_ASSISTANT,
  createdAt: Date.now(),
  id: createMessageId(),
  text,
  type: 'text',
  metadata: { system: true, ...metadata },
})

export const createUserTextMessage = (text: string): MessageType.Text => ({
  author: CHAT_USER,
  createdAt: Date.now(),
  id: createMessageId(),
  text,
  type: 'text',
})
