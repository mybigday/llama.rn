/* eslint-disable react/require-default-props */
import React from 'react'
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native'
import { FontSizes, Spacing } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface MaskedProgressProps {
  visible: boolean
  text?: string
  progress?: number // 0-100
  showProgressBar?: boolean
}

export function MaskedProgress(props: MaskedProgressProps) {
  const { visible, text = 'Loading...', progress = 0, showProgressBar = false } = props
  const { theme } = useTheme()

  const styles = StyleSheet.create({
    overlay: {
      position: 'absolute',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: theme.dark ? 'rgba(0, 0, 0, 0.7)' : 'rgba(0, 0, 0, 0.5)',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000,
    },
    container: {
      backgroundColor: theme.colors.surface,
      borderRadius: Spacing.lg,
      padding: Spacing.xxl,
      alignItems: 'center',
      minWidth: 200,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: theme.dark ? 0.4 : 0.25,
      shadowRadius: theme.dark ? 10 : 8,
      elevation: 10,
    },
    text: {
      marginTop: Spacing.lg,
      fontSize: FontSizes.large,
      color: theme.colors.text,
      textAlign: 'center',
      fontWeight: '500',
    },
    progressContainer: {
      marginTop: Spacing.lg,
      width: 180,
      alignItems: 'center',
    },
    progressBar: {
      width: '100%',
      height: 8,
      backgroundColor: theme.colors.border,
      borderRadius: 4,
      overflow: 'hidden',
    },
    progressFill: {
      height: '100%',
      backgroundColor: theme.colors.primary,
      borderRadius: 4,
    },
  })

  if (!visible) return null

  return (
    <View style={styles.overlay}>
      <View style={styles.container}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
        <Text style={styles.text}>{text}</Text>
        {showProgressBar && progress > 0 && (
          <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${Math.min(100, Math.max(0, progress))}%` },
                ]}
              />
            </View>
          </View>
        )}
      </View>
    </View>
  )
}
