import React from 'react'
import { StyleSheet } from 'react-native'
import { TouchableOpacity } from 'react-native-gesture-handler'
import Icon from '@react-native-vector-icons/material-design-icons'
import { useTheme } from '../contexts/ThemeContext'

type IconName =
  | 'refresh'
  | 'cog-outline'
  | 'speaker'
  | 'close'
  | 'timer-sand-empty'
  | 'information-outline'

interface HeaderButtonProps {
  onPress: () => void
  iconName: IconName
  size?: number
}

const styles = StyleSheet.create({
  button: {
    width: 36,
    justifyContent: 'center',
    alignItems: 'center',
  },
})

export function HeaderButton({
  onPress,
  iconName,
  size = 24,
}: HeaderButtonProps) {
  const { theme } = useTheme()

  return (
    <TouchableOpacity
      style={styles.button}
      onPress={onPress}
    >
      <Icon name={iconName} size={size} color={theme.colors.primary} />
    </TouchableOpacity>
  )
}
