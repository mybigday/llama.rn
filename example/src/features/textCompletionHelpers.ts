interface TokenLike {
  token: string
}

export const getDefaultTextCompletionMessages = () => [
  {
    role: 'system' as const,
    content: 'You are a helpful AI assistant.',
  },
  {
    role: 'user' as const,
    content: 'Hello! Please introduce yourself.',
  },
]

export const buildGeneratedCompletionText = (
  formattedPrompt: string,
  tokens: TokenLike[],
) => formattedPrompt + tokens.map((token) => token.token).join('')

export const getTokenHeatmapColor = (prob?: number): string => {
  if (prob === undefined) return 'transparent'
  const normalizedProb = Math.max(0, Math.min(1, prob))
  const red = Math.round(255 * (1 - normalizedProb))
  const green = Math.round(255 * normalizedProb)
  return `rgba(${red}, ${green}, 0, 0.3)`
}
