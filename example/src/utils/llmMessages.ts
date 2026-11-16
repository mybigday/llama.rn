/**
 * Shared types for LLM messages across all screens.
 */

export interface LLMContentPart {
  type: 'text' | 'image_url' | 'input_audio' | 'video_url'
  text?: string
  image_url?: { url: string }
  input_audio?: { format: string; data: string }
  video_url?: { url: string }
}

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | LLMContentPart[]
  reasoning_content?: string
  tool_call_id?: string
  tool_calls?: any[]
}
