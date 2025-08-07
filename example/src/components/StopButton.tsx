import React from 'react'
import { TouchableOpacity, Text, StyleSheet } from 'react-native'
import type { EdgeInsets } from 'react-native-safe-area-context'
import type { LlamaContext } from '../../../src'

interface StopButtonProps {
  context: LlamaContext | null
  insets: EdgeInsets
  isLoading: boolean
}

const styles = StyleSheet.create({
  stopButtonContainer: {
    position: 'absolute',
    bottom: 80,
    left: '50%',
    transform: [{ translateX: -50 }],
    backgroundColor: 'rgba(255, 59, 48, 0.9)',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 25,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 100,
  },
  stopButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
})

export const StopButton: React.FC<StopButtonProps> = ({
  context,
  insets,
  isLoading,
}) => {
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
