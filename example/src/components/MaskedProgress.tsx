/* eslint-disable react/require-default-props */
import React from 'react'
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native'
import { Colors, FontSizes, Spacing } from '../styles/commonStyles'

interface MaskedProgressProps {
  visible: boolean
  text?: string
  progress?: number // 0-100
  showProgressBar?: boolean
}

const styles = StyleSheet.create({
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  container: {
    backgroundColor: Colors.white,
    borderRadius: Spacing.lg,
    padding: Spacing.xxl,
    alignItems: 'center',
    minWidth: 200,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 10,
  },
  text: {
    marginTop: Spacing.lg,
    fontSize: FontSizes.large,
    color: Colors.text,
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
    backgroundColor: Colors.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: Colors.primary,
    borderRadius: 4,
  },
})

export function MaskedProgress(props: MaskedProgressProps) {
  const { visible, text = 'Loading...', progress = 0, showProgressBar = false } = props

  if (!visible) return null

  return (
    <View style={styles.overlay}>
      <View style={styles.container}>
        <ActivityIndicator size="large" color={Colors.primary} />
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
