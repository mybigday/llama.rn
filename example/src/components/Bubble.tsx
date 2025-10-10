import React, { useContext, useState } from 'react'
import type { ReactNode } from 'react'
import { View, Text, TouchableOpacity, Image } from 'react-native'
import Clipboard from '@react-native-clipboard/clipboard'
import { ThemeContext, UserContext } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import type { NativeCompletionResultTimings } from '../../../src'
import { useTheme } from '../contexts/ThemeContext'

const isPositiveFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

const formatDuration = (ms?: number) => {
  if (!isPositiveFiniteNumber(ms)) return undefined
  if (ms >= 1000) {
    const seconds = ms / 1000
    return `${seconds >= 10 ? seconds.toFixed(1) : seconds.toFixed(2)} s`
  }
  return `${Math.round(ms)} ms`
}

const formatRate = (value?: number) => {
  if (!isPositiveFiniteNumber(value)) return undefined
  if (value >= 100) return `${Math.round(value)} tok/s`
  return `${value.toFixed(1)} tok/s`
}

const formatMsPerToken = (value?: number) => {
  if (!isPositiveFiniteNumber(value)) return undefined
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ms/tok`
}

const formatTokenCount = (value?: number) => {
  if (typeof value !== 'number' || value < 0) return undefined
  if (value === 0) return '0 tok'
  return `${value} tok`
}

const buildTimingLine = (
  label: string,
  data?: {
    tokens?: number
    ms?: number
    perSecond?: number
    perTokenMs?: number
  },
) => {
  if (!data) return undefined
  const parts: string[] = []
  const tokensText = formatTokenCount(data.tokens)
  if (tokensText) parts.push(tokensText)
  const durationText = formatDuration(data.ms)
  if (durationText) parts.push(durationText)
  const rateText = formatRate(data.perSecond)
  if (rateText) parts.push(rateText)
  const perTokenText = formatMsPerToken(data.perTokenMs)
  if (perTokenText) parts.push(perTokenText)
  if (parts.length === 0) return undefined
  return `${label}: ${parts.join(' | ')}`
}

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
  const [showTimings, setShowTimings] = useState(false)

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
  const timingTextColor = isDark ? '#999' : '#666'
  const timingBackground = isDark
    ? 'rgba(255,255,255,0.08)'
    : 'rgba(0,0,0,0.08)'
  const timingBorderColor = isDark
    ? 'rgba(255,255,255,0.15)'
    : 'rgba(0,0,0,0.12)'

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

  const completionTimings: NativeCompletionResultTimings | undefined =
    completionResult?.timings ||
    (typeof timings === 'object' && timings !== null
      ? (timings as NativeCompletionResultTimings)
      : undefined)

  const promptTimingLine = completionTimings
    ? buildTimingLine('Prompt', {
        tokens: completionTimings.prompt_n,
        ms: completionTimings.prompt_ms,
        perSecond: completionTimings.prompt_per_second,
        perTokenMs: completionTimings.prompt_per_token_ms,
      })
    : undefined

  const generationTimingLine = completionTimings
    ? buildTimingLine('Generation', {
        tokens: completionTimings.predicted_n,
        ms: completionTimings.predicted_ms,
        perSecond: completionTimings.predicted_per_second,
        perTokenMs: completionTimings.predicted_per_token_ms,
      })
    : undefined

  const timingLines = [promptTimingLine, generationTimingLine].filter(
    (line): line is string => !!line,
  )

  const fallbackTimingText =
    timingLines.length === 0 && typeof timings === 'string'
      ? timings
      : undefined

  const canToggleTimings = timingLines.length > 0
  const generationRate = formatRate(completionTimings?.predicted_per_second)
  const generationSummary =
    completionTimings && generationRate ? generationRate : undefined
  const summaryLabel = generationSummary || 'Timings'
  const summaryArrow = showTimings ? '▾' : '▸'

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
      {(generationSummary || canToggleTimings) && (
        <View
          style={{
            paddingHorizontal: 12,
            paddingTop: 6,
            paddingBottom: 6,
            borderTopWidth: 1,
            borderTopColor: borderColor,
            alignItems: 'flex-end',
            width: '100%',
          }}
        >
          <TouchableOpacity
            activeOpacity={canToggleTimings ? 0.6 : 1}
            disabled={!canToggleTimings}
            onPress={() => setShowTimings((prev) => !prev)}
            style={{
              alignSelf: 'flex-end',
              backgroundColor: timingBackground,
              borderRadius: 999,
              borderWidth: 1,
              borderColor: timingBorderColor,
              paddingHorizontal: 12,
              paddingVertical: 2,
              flexDirection: 'row',
              alignItems: 'center',
              gap: summaryArrow ? 6 : 0,
            }}
          >
            <Text
              style={{
                color: timingTextColor,
                fontSize: 11,
                fontWeight: generationSummary ? '600' : '500',
                fontFamily: 'monospace',
                textAlign: 'right',
              }}
            >
              {summaryLabel}
            </Text>
            {summaryArrow && (
              <Text
                style={{
                  color: timingTextColor,
                  fontSize: 10,
                  fontWeight: '600',
                }}
              >
                {summaryArrow}
              </Text>
            )}
          </TouchableOpacity>
          {showTimings && timingLines.length > 0 && (
            <View
              style={{
                marginTop: 8,
                gap: 4,
                alignItems: 'flex-end',
                backgroundColor: sectionBackground,
                borderRadius: 8,
                paddingHorizontal: 10,
                paddingVertical: 6,
                borderWidth: 1,
                borderColor,
                alignSelf: 'flex-end',
              }}
            >
              {timingLines.map((line, index) => (
                <Text
                  key={index}
                  style={{
                    color: timingTextColor,
                    fontSize: 10,
                    textAlign: 'right',
                    fontFamily: 'monospace',
                  }}
                >
                  {line}
                </Text>
              ))}
            </View>
          )}
        </View>
      )}
      {fallbackTimingText && (
        <Text
          style={{
            textAlign: 'right',
            color: timingTextColor,
            paddingHorizontal: 12,
            paddingBottom: 12,
            marginTop: -8,
            fontSize: 10,
          }}
        >
          {fallbackTimingText}
        </Text>
      )}

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
    </Container>
  )
}
