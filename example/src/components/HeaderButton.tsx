import React from 'react'
import { TouchableOpacity } from 'react-native-gesture-handler'
import Icon from '@react-native-vector-icons/material-design-icons'
import { useTheme } from '../contexts/ThemeContext'

type IconName =
  | 'refresh'
  | 'cog-outline'
  | 'speaker'
  | 'close'
  | 'timer-sand-empty'

interface HeaderButtonProps {
  onPress: () => void
  iconName: IconName
  size?: number
}

export function HeaderButton({
  onPress,
  iconName,
  size = 24,
}: HeaderButtonProps) {
  const { theme } = useTheme()
  
  return (
    <TouchableOpacity style={{ marginRight: 4 }} onPress={onPress}>
      <Icon name={iconName} size={size} color={theme.colors.primary} />
    </TouchableOpacity>
  )
}
