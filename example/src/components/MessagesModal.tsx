import React, { useState, useRef, useEffect } from 'react'
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  Clipboard,
  Platform,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import type { LlamaContext } from '../../../src'
import { CommonStyles } from '../styles/commonStyles'
import type { LLMMessage } from '../utils/llmMessages'

interface MessagesModalProps {
  visible: boolean
  onClose: () => void
  messages: LLMMessage[]
  tools?: any[]
  context: LlamaContext | null
  onImportMessages?: (messages: MessageType.Any[]) => void
  onUpdateSystemPrompt?: (systemPrompt: string) => void
  defaultSystemPrompt?: string
}

const styles = StyleSheet.create({
  // Using common styles where possible
  container: {
    ...CommonStyles.container,
    backgroundColor: '#f8f9fa',
  },
  header: {
    ...CommonStyles.header,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  headerTitle: {
    ...CommonStyles.headerTitle,
    fontSize: 20,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  closeButton: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    backgroundColor: '#f1f3f4',
  },
  closeButtonText: {
    fontSize: 16,
    color: '#007AFF',
    fontWeight: '600',
  },
  content: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  contentContainer: {
    ...CommonStyles.scrollContent,
    paddingHorizontal: 20,
    paddingVertical: 24,
  },
  section: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.08,
        shadowRadius: 8,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 16,
    letterSpacing: -0.2,
  },
  codeBlock: {
    backgroundColor: '#1e1e1e',
    borderRadius: 10,
    padding: 20,
    maxHeight: 320,
    borderWidth: 1,
    borderColor: '#333',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.15,
        shadowRadius: 4,
      },
      android: {
        elevation: 2,
      },
    }),
  },
  codeText: {
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 13,
    color: '#f8f8f2',
    lineHeight: 18,
    letterSpacing: 0.2,
  },
  buttonRow: {
    flexDirection: 'row',
    marginTop: 16,
    gap: 12,
  },
  copyButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 48,
    ...Platform.select({
      ios: {
        shadowColor: '#007AFF',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  secondaryButton: {
    backgroundColor: '#5856D6',
    ...Platform.select({
      ios: {
        shadowColor: '#5856D6',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  copyButtonText: {
    color: 'white',
    fontSize: 15,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  textArea: {
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 18,
    borderWidth: 2,
    borderColor: '#e9ecef',
    minHeight: 140,
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    color: '#1a1a1a',
    lineHeight: 20,
    textAlignVertical: 'top',
  },
  textAreaFocused: {
    borderColor: '#007AFF',
    backgroundColor: 'white',
  },
  importButton: {
    backgroundColor: '#34C759',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    minHeight: 52,
    ...Platform.select({
      ios: {
        shadowColor: '#34C759',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
    }),
  },
  disabledButton: {
    backgroundColor: '#d1d5db',
    ...Platform.select({
      ios: {
        shadowColor: 'transparent',
      },
      android: {
        elevation: 0,
      },
    }),
  },
  importButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  disabledText: {
    color: '#9ca3af',
  },
  messageCountBadge: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    marginLeft: 8,
  },
  messageCountText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  divider: {
    height: 1,
    backgroundColor: '#e9ecef',
    marginVertical: 4,
  },
})

const MessagesModal: React.FC<MessagesModalProps> = ({
  visible,
  onClose,
  messages,
  tools,
  context,
  onImportMessages,
  onUpdateSystemPrompt,
  defaultSystemPrompt = '',
}) => {
  const [importText, setImportText] = useState('')
  const [isImporting, setIsImporting] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState('')
  const scrollViewRef = useRef<ScrollView>(null)

  // Extract current system prompt from messages
  useEffect(() => {
    const systemMessage = messages.find(msg => msg.role === 'system')
    if (systemMessage && typeof systemMessage.content === 'string') {
      setSystemPrompt(systemMessage.content)
    } else if (systemMessage && Array.isArray(systemMessage.content)) {
      // Handle multimodal system messages
      const textContent = systemMessage.content.find(item => item.type === 'text')
      if (textContent && textContent.text) {
        setSystemPrompt(textContent.text)
      }
    }
  }, [messages])

  // Handle real-time system prompt updates
  const handleSystemPromptChange = (text: string) => {
    setSystemPrompt(text)
    if (onUpdateSystemPrompt) {
      onUpdateSystemPrompt(text)
    }
  }

  // Handle reset to default system prompt
  const handleResetSystemPrompt = () => {
    if (defaultSystemPrompt) {
      setSystemPrompt(defaultSystemPrompt)
      if (onUpdateSystemPrompt) {
        onUpdateSystemPrompt(defaultSystemPrompt)
      }
    }
  }

  // Helper function to omit image base64 from display
  const omitImageBase64ForDisplay = (msgs: LLMMessage[]) =>
    msgs.map((msg) => {
      if (Array.isArray(msg.content)) {
        return {
          ...msg,
          content: msg.content.map((item) => {
            if (item.type === 'image_url' && item.image_url?.url?.startsWith('data:')) {
              return {
                ...item,
                image_url: { url: '[Base64 Image Data Omitted]' }
              }
            }
            return item
          })
        }
      }
      return msg
    })

  // Helper function to get full messages with base64 for copying
  const getFullMessagesForCopy = (msgs: LLMMessage[]) => msgs

  // Helper function to convert LLM messages to chat UI messages
  const convertLLMMessagesToChatMessages = (llmMessages: LLMMessage[]): MessageType.Any[] => {
    const user = { id: 'user' }
    const assistant = { id: 'assistant' }
    const randId = () => Math.random().toString(36).substr(2, 9)

    const chatMessages: MessageType.Any[] = []

    llmMessages.forEach((llmMsg) => {
      // Skip system messages - they're not displayed in chat UI
      if (llmMsg.role === 'system') {
        return
      }

      // Handle tool messages - convert to assistant messages with metadata
      if (llmMsg.role === 'tool') {
        const toolMessage: MessageType.Text = {
          author: assistant,
          createdAt: Date.now(),
          id: randId(),
          text: `Tool Result: ${llmMsg.content}`,
          type: 'text',
          metadata: {
            toolResult: true,
            toolCallId: llmMsg.tool_call_id,
          },
        }
        chatMessages.push(toolMessage)
        return
      }

      const author = llmMsg.role === 'user' ? user : assistant

      // Handle text content
      if (typeof llmMsg.content === 'string') {
        const textMessage: MessageType.Text = {
          author,
          createdAt: Date.now(),
          id: randId(),
          text: llmMsg.content,
          type: 'text',
          metadata: llmMsg.tool_calls ? {
            toolCalls: true,
            storedToolCalls: llmMsg.tool_calls
          } : {},
        }
        chatMessages.push(textMessage)
      }
      // Handle multimodal content (array of text/image items)
      else if (Array.isArray(llmMsg.content)) {
        llmMsg.content.forEach((item) => {
          if (item.type === 'text' && item.text) {
            const textMessage: MessageType.Text = {
              author,
              createdAt: Date.now(),
              id: randId(),
              text: item.text,
              type: 'text',
              metadata: llmMsg.tool_calls ? {
                toolCalls: true,
                storedToolCalls: llmMsg.tool_calls
              } : {},
            }
            chatMessages.push(textMessage)
          } else if (item.type === 'image_url' && item.image_url?.url) {
            const imageMessage: MessageType.Image = {
              author,
              createdAt: Date.now(),
              id: randId(),
              name: 'imported-image',
              size: 0,
              type: 'image',
              uri: item.image_url.url,
              metadata: {
                imported: true,
              },
            }
            chatMessages.push(imageMessage)
          }
        })
      }
    })

    return chatMessages
  }

  const displayMessages = omitImageBase64ForDisplay(messages)
  const jsonContent = JSON.stringify(displayMessages, null, 2)
  const fullJsonContent = JSON.stringify(getFullMessagesForCopy(messages), null, 2)

  const copyToClipboard = async (content: string, type: string) => {
    try {
      if (Platform.OS === 'ios') {
        await Clipboard.setString(content)
      } else {
        // For Android, we'll use the same method
        await Clipboard.setString(content)
      }
    } catch (error) {
      Alert.alert('Error', `Failed to copy ${type}`)
    }
  }

  const handleCopyRawJson = () => {
    copyToClipboard(fullJsonContent, 'Raw JSON')
  }

  const handleCopyFormattedChat = async () => {
    if (!context) {
      Alert.alert('Error', 'Context not available')
      return
    }

    try {
      // Try to get formatted chat - the API might expect different input
      const parsedMessages = JSON.parse(fullJsonContent)
      const result = await context.getFormattedChat(parsedMessages, null, {
        jinja: true,
        tools,
        tool_choice: tools ? 'auto' : undefined,
      })
      if (result && typeof result === 'object' && 'prompt' in result) {
        copyToClipboard(result.prompt, 'Formatted chat')
      } else {
        copyToClipboard(String(result), 'Formatted chat')
      }
    } catch (error: any) {
      Alert.alert('Error', `Failed to format chat: ${error.message}`)
    }
  }

  const handleImportMessages = async () => {
    if (!importText.trim()) {
      Alert.alert('Error', 'Please enter JSON content to import')
      return
    }

    if (!context) {
      Alert.alert('Error', 'Context not available for validation')
      return
    }

    setIsImporting(true)
    try {
      // Parse JSON
      const parsedMessages = JSON.parse(importText.trim())

      // Validate with context - simplified validation
      if (Array.isArray(parsedMessages)) {
        await context.getFormattedChat(parsedMessages)
      } else {
        throw new TypeError('Messages must be an array')
      }

      // If validation passes, convert and import the messages
      if (onImportMessages) {
        const chatMessages = convertLLMMessagesToChatMessages(parsedMessages)
        onImportMessages(chatMessages)
        setImportText('')
        onClose()
      }
    } catch (error: any) {
      if (error.message.includes('JSON')) {
        Alert.alert('Error', 'Invalid JSON format')
      } else {
        Alert.alert('Error', `Validation failed: ${error.message}`)
      }
    } finally {
      setIsImporting(false)
    }
  }

  const [textAreaFocused, setTextAreaFocused] = useState(false)

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View style={{ flexDirection: 'row', alignItems: 'center' }}>
            <Text style={styles.headerTitle}>Messages</Text>
            <View style={styles.messageCountBadge}>
              <Text style={styles.messageCountText}>{messages.length}</Text>
            </View>
          </View>
          <TouchableOpacity onPress={onClose} style={styles.closeButton}>
            <Text style={styles.closeButtonText}>Done</Text>
          </TouchableOpacity>
        </View>

        {/* Content */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          showsVerticalScrollIndicator={false}
        >
          {/* System Prompt Editor */}
          <View style={styles.section}>
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <Text style={styles.sectionTitle}>System Prompt</Text>
              <TouchableOpacity
                onPress={handleResetSystemPrompt}
                style={[styles.copyButton, { backgroundColor: '#6b7280', flex: 0, paddingHorizontal: 12 }]}
                disabled={!defaultSystemPrompt || systemPrompt === defaultSystemPrompt}
              >
                <Text style={[styles.copyButtonText, (!defaultSystemPrompt || systemPrompt === defaultSystemPrompt) && styles.disabledText]}>
                  Reset
                </Text>
              </TouchableOpacity>
            </View>
            <TextInput
              style={[styles.textArea, { minHeight: 120 }]}
              placeholder="Enter system prompt..."
              placeholderTextColor="#9ca3af"
              value={systemPrompt}
              onChangeText={handleSystemPromptChange}
              multiline
              textAlignVertical="top"
            />
          </View>

          {/* JSON Display */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Current Messages</Text>
            <View style={styles.codeBlock}>
              <ScrollView
                showsVerticalScrollIndicator
                showsHorizontalScrollIndicator
                nestedScrollEnabled
                bounces={false}
              >
                <Text style={styles.codeText} selectable>
                  {jsonContent}
                </Text>
              </ScrollView>
            </View>

            {/* Copy buttons */}
            <View style={styles.buttonRow}>
              <TouchableOpacity
                style={styles.copyButton}
                onPress={handleCopyRawJson}
                activeOpacity={0.8}
              >
                <Text style={styles.copyButtonText}>Copy Raw JSON</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.copyButton, styles.secondaryButton]}
                onPress={handleCopyFormattedChat}
                disabled={!context}
                activeOpacity={0.8}
              >
                <Text style={[styles.copyButtonText, !context && styles.disabledText]}>
                  Copy Formatted
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Import Section */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Import Messages</Text>
            <TextInput
              style={[
                styles.textArea,
                textAreaFocused && styles.textAreaFocused
              ]}
              placeholder="Paste JSON messages here..."
              placeholderTextColor="#9ca3af"
              value={importText}
              onChangeText={setImportText}
              onFocus={() => setTextAreaFocused(true)}
              onBlur={() => setTextAreaFocused(false)}
              multiline
              textAlignVertical="top"
            />
            <TouchableOpacity
              style={[
                styles.importButton,
                (!importText.trim() || !context || isImporting) && styles.disabledButton,
              ]}
              onPress={handleImportMessages}
              disabled={!importText.trim() || !context || isImporting}
              activeOpacity={0.8}
            >
              <Text
                style={[
                  styles.importButtonText,
                  (!importText.trim() || !context || isImporting) && styles.disabledText,
                ]}
              >
                {isImporting ? 'Importing...' : 'Import Messages'}
              </Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </SafeAreaView>
    </Modal>
  )
}

export { MessagesModal }
