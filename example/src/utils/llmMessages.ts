/**
 * Shared types for LLM messages across all screens.
 */

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | Array<{
    type: 'text' | 'image_url' | 'input_audio'
    text?: string
    image_url?: { url: string }
    input_audio?: { format: string; data: string }
  }>
  reasoning_content?: string
  tool_call_id?: string
  tool_calls?: any[]
}
