import React from 'react'
import { TouchableOpacity } from 'react-native-gesture-handler'
import Icon from '@react-native-vector-icons/material-design-icons'
import { CommonStyles } from '../styles/commonStyles'

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
  return (
    <TouchableOpacity style={CommonStyles.headerButton} onPress={onPress}>
      <Icon name={iconName} size={size} color="#007AFF" />
    </TouchableOpacity>
  )
}
