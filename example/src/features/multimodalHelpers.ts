export interface MultimodalSupport {
  vision: boolean
  audio: boolean
}

export const DEFAULT_MULTIMODAL_SYSTEM_PROMPT =
  'You are a helpful AI assistant. Be concise and helpful in your responses.'

export const createMultimodalSystemPrompt = (
  multimodalSupport: MultimodalSupport | null,
) => {
  if (!multimodalSupport) {
    return DEFAULT_MULTIMODAL_SYSTEM_PROMPT
  }

  const capabilities = []
  if (multimodalSupport.vision) capabilities.push('vision')
  if (multimodalSupport.audio) capabilities.push('audio')

  if (capabilities.length === 0) {
    return DEFAULT_MULTIMODAL_SYSTEM_PROMPT
  }

  const capabilityText =
    capabilities.length > 1
      ? `${capabilities.slice(0, -1).join(', ')} and ${
          capabilities[capabilities.length - 1]
        }`
      : capabilities[0]

  const mediaTypes = []
  if (multimodalSupport.vision) mediaTypes.push('images')
  if (multimodalSupport.audio) mediaTypes.push('audio')

  const mediaText =
    mediaTypes.length > 1
      ? `${mediaTypes.slice(0, -1).join(', ')} and ${
          mediaTypes[mediaTypes.length - 1]
        }`
      : mediaTypes[0]

  let analysisText
  if (multimodalSupport.vision && multimodalSupport.audio) {
    analysisText = 'see and analyze images and listen to and analyze audio'
  } else if (multimodalSupport.vision) {
    analysisText = 'see and analyze images'
  } else {
    analysisText = 'listen to and analyze audio'
  }

  return `You are a helpful AI assistant with ${capabilityText} capabilities. You can ${analysisText} that users share. Be descriptive when analyzing ${mediaText} and helpful in answering questions about multimedia content. Be concise and helpful in your responses.`
}

export const createMultimodalWelcomeMessage = (
  multimodalSupport: MultimodalSupport | null,
) => {
  if (!multimodalSupport) {
    return "Hello! I'm an AI assistant ready to help with text conversations. How can I help you today?"
  }

  const capabilities = []
  if (multimodalSupport.vision) capabilities.push('images')
  if (multimodalSupport.audio) capabilities.push('audio files')

  if (capabilities.length === 0) {
    return "Hello! I'm an AI assistant ready to help with text conversations. How can I help you today?"
  }

  const capabilityText =
    capabilities.length > 1
      ? `${capabilities.slice(0, -1).join(', ')} and ${
          capabilities[capabilities.length - 1]
        }`
      : capabilities[0]

  let senseText
  if (multimodalSupport.vision && multimodalSupport.audio) {
    senseText = 'see or hear'
  } else if (multimodalSupport.vision) {
    senseText = 'see'
  } else {
    senseText = 'hear'
  }

  const contentType =
    capabilities.length > 1
      ? 'multimedia'
      : capabilities[0]?.replace(' files', '')

  return `Hello! I'm a multimodal AI assistant. You can share ${capabilityText} with me and I'll analyze them, answer questions about what I ${senseText}, or engage in conversations about ${contentType} content. How can I help you today?`
}
