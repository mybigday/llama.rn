import React from 'react'
import { TouchableOpacity, Text, StyleSheet } from 'react-native'
import type { EdgeInsets } from 'react-native-safe-area-context'
import type { LlamaContext } from '../../../src'
import { useTheme } from '../contexts/ThemeContext'

interface StopButtonProps {
  context: LlamaContext | null
  insets: EdgeInsets
  isLoading: boolean
}

export const StopButton: React.FC<StopButtonProps> = ({
  context,
  insets,
  isLoading,
}) => {
  const { theme } = useTheme()

  const styles = StyleSheet.create({
    stopButtonContainer: {
      position: 'absolute',
      bottom: 80,
      left: '50%',
      transform: [{ translateX: -50 }],
      backgroundColor: `${theme.colors.error}e6`,
      paddingHorizontal: 20,
      paddingVertical: 12,
      borderRadius: 25,
      alignItems: 'center',
      justifyContent: 'center',
      minWidth: 100,
    },
    stopButtonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
    },
  })

  if (!isLoading) {
    return null
  }

  return (
    <TouchableOpacity
      style={[styles.stopButtonContainer, { bottom: insets.bottom + 80 }]}
      onPress={() => {
        if (context) {
          context.stopCompletion()
        }
      }}
    >
      <Text style={styles.stopButtonText}>Stop</Text>
    </TouchableOpacity>
  )
}
