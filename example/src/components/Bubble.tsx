import React, { useContext, useState } from 'react'
import type { ReactNode } from 'react'
import { View, Text, TouchableOpacity, Image } from 'react-native'
import Clipboard from '@react-native-clipboard/clipboard'
import { ThemeContext, UserContext } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { useTheme } from '../contexts/ThemeContext'

export const Bubble = ({
  child,
  message,
}: {
  child: ReactNode
  message: MessageType.Any
}) => {
  const { isDark } = useTheme()
  const theme = useContext(ThemeContext)
  const user = useContext(UserContext)
  const currentUserIsAuthor = user?.id === message.author.id
  const { copyable, timings, completionResult, partialCompletionResult } =
    message.metadata || {}

  const [showReasoning, setShowReasoning] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)

  const Container = copyable ? TouchableOpacity : View

  // Theme-aware colors
  const overlayBackground = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
  const sectionBackground = isDark
    ? 'rgba(255,255,255,0.05)'
    : 'rgba(0,0,0,0.05)'
  const borderColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
  const textColor = (() => {
    if (currentUserIsAuthor) {
      return 'rgba(255,255,255,0.8)'
    }
    return isDark ? 'rgba(255,255,255,0.8)' : 'rgba(0,0,0,0.6)'
  })()
  const contentTextColor = (() => {
    if (currentUserIsAuthor) {
      return 'rgba(255,255,255,0.9)'
    }
    return isDark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.8)'
  })()
  const toolCallBackground = isDark
    ? 'rgba(255,255,255,0.1)'
    : 'rgba(0,0,0,0.1)'
  const timingTextColor = isDark ? '#999' : '#ccc'

  // Use partial data during streaming, fall back to final result
  const currentResult = partialCompletionResult || completionResult
  const hasReasoningContent = currentResult?.reasoning_content
  const hasToolCalls =
    currentResult?.tool_calls && currentResult.tool_calls.length > 0
  // Check if we're in a streaming state (partial data available but no final content yet)
  const isStreamingReasoning =
    partialCompletionResult && partialCompletionResult?.reasoning_content
  const isStreamingToolCalls =
    partialCompletionResult && partialCompletionResult?.tool_calls

  return (
    <Container
      style={{
        backgroundColor:
          !currentUserIsAuthor || message.type === 'image'
            ? theme.colors.secondary
            : theme.colors.primary,
        borderBottomLeftRadius: currentUserIsAuthor
          ? theme.borders.messageBorderRadius
          : 0,
        borderBottomRightRadius: currentUserIsAuthor
          ? 0
          : theme.borders.messageBorderRadius,
        borderColor: 'transparent',
        borderRadius: theme.borders.messageBorderRadius,
        overflow: 'hidden',
      }}
      onPress={() => {
        if (message.type !== 'text') return
        Clipboard.setString(message.text)
      }}
    >
      {/* Show toggle button for reasoning if available */}
      {(hasReasoningContent || isStreamingReasoning) && (
        <TouchableOpacity
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            paddingHorizontal: 12,
            paddingVertical: 4,
            backgroundColor: overlayBackground,
          }}
          onPress={() => setShowReasoning(!showReasoning)}
        >
          <Text
            style={{
              fontSize: 12,
              color: textColor,
              fontWeight: '600',
            }}
          >
            {showReasoning ? '💭 Hide Reasoning' : '💭 Show Reasoning'}
          </Text>
        </TouchableOpacity>
      )}

      {/* Show reasoning content or progress if toggled on */}
      {showReasoning && (hasReasoningContent || isStreamingReasoning) && (
        <View
          style={{
            paddingHorizontal: 12,
            paddingVertical: 8,
            backgroundColor: sectionBackground,
            borderBottomWidth: 1,
            borderBottomColor: borderColor,
          }}
        >
          <Text
            style={{
              fontSize: 13,
              color: contentTextColor,
              fontFamily: 'monospace',
              lineHeight: 18,
            }}
          >
            {currentResult.reasoning_content}
          </Text>
        </View>
      )}

      {/* Show main content */}
      <View>{child}</View>

      {/* Show toggle button for tool calls if available */}
      {(hasToolCalls || isStreamingToolCalls) && (
        <TouchableOpacity
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            paddingHorizontal: 12,
            paddingVertical: 4,
            backgroundColor: overlayBackground,
          }}
          onPress={() => setShowToolCalls(!showToolCalls)}
        >
          <Text
            style={{
              fontSize: 12,
              color: textColor,
              fontWeight: '600',
            }}
          >
            {showToolCalls ? '🛠️ Hide Tool Calls' : '🛠️ Show Tool Calls'}
          </Text>
        </TouchableOpacity>
      )}

      {/* Show tool calls or progress if toggled on */}
      {showToolCalls && (hasToolCalls || isStreamingToolCalls) && (
        <View
          style={{
            paddingHorizontal: 12,
            paddingVertical: 8,
            backgroundColor: sectionBackground,
            borderTopWidth: 1,
            borderTopColor: borderColor,
          }}
        >
          {currentResult.tool_calls.map((toolCall: any, index: number) => (
            <View
              key={index}
              style={{
                marginBottom: 4,
                padding: 8,
                backgroundColor: toolCallBackground,
                borderRadius: 4,
              }}
            >
              <Text
                style={{
                  fontSize: 12,
                  fontWeight: '600',
                  color: contentTextColor,
                }}
              >
                {toolCall.function?.name || toolCall.name}
              </Text>
              <Text
                style={{
                  fontSize: 11,
                  color: textColor,
                  fontFamily: 'monospace',
                  marginTop: 2,
                }}
              >
                {toolCall.function?.arguments || toolCall.arguments}
              </Text>
            </View>
          ))}
        </View>
      )}

      {message?.metadata?.mediaPath && (
        <Image
          source={{ uri: message.metadata.mediaPath }}
          resizeMode="cover"
          style={{
            width: 100,
            height: 100,
            alignSelf: 'flex-end',
            marginRight: 12,
            marginBottom: 12,
          }}
        />
      )}
      {timings && (
        <Text
          style={{
            textAlign: 'right',
            color: timingTextColor,
            paddingRight: 12,
            paddingBottom: 12,
            marginTop: -8,
            fontSize: 10,
          }}
        >
          {timings}
        </Text>
      )}
    </Container>
  )
}
