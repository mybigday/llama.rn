import type { NativeLlamaChatMessage } from './NativeRNLlama'

export type RNLlamaMessagePart = {
  text?: string
}

export type RNLlamaOAICompatibleMessage = {
  role: string
  content?: string | RNLlamaMessagePart[] | any // any for check invalid content type
}

export function formatChat(
  messages: RNLlamaOAICompatibleMessage[],
): NativeLlamaChatMessage[] {
  const chat: NativeLlamaChatMessage[] = []

  messages.forEach((currMsg) => {
    const role: string = currMsg.role || ''

    let content: string = ''
    if ('content' in currMsg) {
      if (typeof currMsg.content === 'string') {
        ;({ content } = currMsg)
      } else if (Array.isArray(currMsg.content)) {
        currMsg.content.forEach((part) => {
          if ('text' in part) {
            content += `${content ? '\n' : ''}${part.text}`
          }
        })
      } else {
        throw new TypeError(
          "Invalid 'content' type (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
        )
      }
    } else {
      throw new Error(
        "Missing 'content' (ref: https://github.com/ggerganov/llama.cpp/issues/8367)",
      )
    }

    chat.push({ role, content })
  })
  return chat
}
