import React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import Icon from '@react-native-vector-icons/material-design-icons'
import { useTheme } from '../../contexts/ThemeContext'
import type { DecisionAnswer } from '../../../../src'

const YES_COLOR = '#34A759'
const NO_COLOR = '#8E8E93'

const percent = (p: number) => `${(p * 100).toFixed(p >= 0.995 || p < 0.1 ? 1 : 0)}%`

interface AnswerCardProps {
  id: string
  instructions: string
  answer: DecisionAnswer
  // choice: option key -> description, in the request's order
  optionDescriptions?: Record<string, string | null>
}

export function AnswerCard({
  id,
  instructions,
  answer,
  optionDescriptions,
}: AnswerCardProps) {
  const { theme } = useTheme()
  const styles = createStyles(theme.colors)

  const renderBar = (
    label: string,
    p: number,
    highlighted: boolean,
    detail?: string | null,
  ) => (
    <View key={label} style={styles.barRow}>
      <View style={styles.barLabelRow}>
        <Text
          style={[styles.barLabel, highlighted && styles.barLabelHighlighted]}
          numberOfLines={1}
        >
          {highlighted ? '● ' : ''}
          {label}
          {detail ? <Text style={styles.barDetail}>{`  ${detail}`}</Text> : null}
        </Text>
        <Text style={[styles.barValue, highlighted && styles.barLabelHighlighted]}>
          {percent(p)}
        </Text>
      </View>
      <View style={styles.track}>
        <View
          style={[
            styles.fill,
            {
              width: `${Math.max(p * 100, 0.5)}%`,
              backgroundColor: highlighted
                ? theme.colors.primary
                : theme.colors.textSecondary,
              opacity: highlighted ? 1 : 0.45,
            },
          ]}
        />
      </View>
    </View>
  )

  const renderBody = () => {
    switch (answer.type) {
      case 'noul': {
        const yes = answer.noul >= 0.5
        return (
          <View>
            <View style={styles.verdictRow}>
              <Icon
                name={yes ? 'check-circle' : 'close-circle'}
                size={22}
                color={yes ? YES_COLOR : NO_COLOR}
              />
              <Text style={[styles.verdict, { color: yes ? YES_COLOR : NO_COLOR }]}>
                {yes ? 'Yes' : 'No'}
              </Text>
              <Text style={styles.verdictDetail}>
                {`P(yes) ${percent(answer.noul)}`}
              </Text>
            </View>
            <View style={styles.splitTrack}>
              <View
                style={{ flex: answer.noul, backgroundColor: YES_COLOR }}
              />
              <View
                style={{ flex: 1 - answer.noul, backgroundColor: NO_COLOR, opacity: 0.35 }}
              />
            </View>
            <View style={styles.splitLabels}>
              <Text style={styles.splitLabel}>yes</Text>
              <Text style={styles.splitLabel}>no</Text>
            </View>
          </View>
        )
      }
      case 'choice':
        return (
          <View>
            {Object.entries(answer.probabilities).map(([key, p]) =>
              renderBar(key, p, key === answer.choice, optionDescriptions?.[key]),
            )}
          </View>
        )
      case 'score': {
        const levels = Object.keys(answer.probabilities)
        const n = levels.length
        const best = levels.reduce((a, b) =>
          answer.probabilities[a]! >= answer.probabilities[b]! ? a : b,
        )
        return (
          <View>
            <View style={styles.scaleTrack}>
              <View
                style={[
                  styles.scaleMarker,
                  { left: `${(answer.score / Math.max(n - 1, 1)) * 100}%` },
                ]}
              />
            </View>
            <View style={styles.splitLabels}>
              <Text style={styles.splitLabel}>{answer.legend[levels[0]!]}</Text>
              <Text style={styles.scoreValue}>
                {`${answer.score.toFixed(2)} / ${n - 1}`}
              </Text>
              <Text style={styles.splitLabel}>{answer.legend[levels[n - 1]!]}</Text>
            </View>
            {levels.map((level) =>
              renderBar(
                `${level} · ${answer.legend[level]}`,
                answer.probabilities[level]!,
                level === best,
              ),
            )}
          </View>
        )
      }
      default:
        return null
    }
  }

  const confidence =
    answer.type === 'noul' ? undefined : (answer as { confidence: number }).confidence

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <Text style={styles.id}>{id}</Text>
        <Text style={styles.typeChip}>{answer.type}</Text>
        {confidence !== undefined && (
          <Text style={styles.confidence}>{`confidence ${percent(confidence)}`}</Text>
        )}
      </View>
      <Text style={styles.instructions}>{instructions}</Text>
      {renderBody()}
    </View>
  )
}

function createStyles(colors: ReturnType<typeof useTheme>['theme']['colors']) {
  return StyleSheet.create({
    card: {
      backgroundColor: colors.surface,
      borderRadius: 12,
      padding: 14,
      marginBottom: 12,
      borderWidth: 1,
      borderColor: colors.border,
    },
    header: { flexDirection: 'row', alignItems: 'center', gap: 8 },
    id: {
      fontFamily: 'monospace',
      fontSize: 14,
      fontWeight: '700',
      color: colors.text,
      flexShrink: 1,
    },
    typeChip: {
      fontSize: 11,
      fontWeight: '600',
      color: colors.primary,
      borderWidth: 1,
      borderColor: colors.primary,
      borderRadius: 10,
      paddingHorizontal: 8,
      paddingVertical: 1,
    },
    confidence: { marginLeft: 'auto', fontSize: 12, color: colors.textSecondary },
    instructions: {
      fontSize: 13,
      color: colors.textSecondary,
      marginTop: 4,
      marginBottom: 12,
    },
    verdictRow: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8 },
    verdict: { fontSize: 20, fontWeight: '700' },
    verdictDetail: { marginLeft: 'auto', fontSize: 13, color: colors.textSecondary },
    splitTrack: {
      flexDirection: 'row',
      height: 10,
      borderRadius: 5,
      overflow: 'hidden',
    },
    splitLabels: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginTop: 4,
      marginBottom: 6,
    },
    splitLabel: { fontSize: 11, color: colors.textSecondary },
    barRow: { marginBottom: 8 },
    barLabelRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 3 },
    barLabel: { fontSize: 13, color: colors.text, flexShrink: 1, marginRight: 8 },
    barLabelHighlighted: { fontWeight: '700', color: colors.primary },
    barDetail: { fontSize: 12, fontWeight: '400', color: colors.textSecondary },
    barValue: { fontSize: 13, color: colors.text, fontVariant: ['tabular-nums'] },
    track: {
      height: 8,
      borderRadius: 4,
      backgroundColor: colors.card,
      overflow: 'hidden',
    },
    fill: { height: 8, borderRadius: 4 },
    scaleTrack: {
      height: 8,
      borderRadius: 4,
      backgroundColor: colors.card,
      marginTop: 4,
      marginHorizontal: 6,
    },
    scaleMarker: {
      position: 'absolute',
      top: -4,
      width: 16,
      height: 16,
      marginLeft: -8,
      borderRadius: 8,
      backgroundColor: colors.primary,
      borderWidth: 2,
      borderColor: colors.white,
    },
    scoreValue: { fontSize: 13, fontWeight: '700', color: colors.primary },
  })
}
