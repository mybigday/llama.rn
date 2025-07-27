/**
 * Shared types for LLM messages across all screens.
 */

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | Array<{
    type: 'text' | 'image_url'
    text?: string
    image_url?: { url: string }
  }>
  tool_call_id?: string
  tool_calls?: any[]
}
