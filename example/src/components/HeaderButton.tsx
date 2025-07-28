import React from 'react'
import { TouchableOpacity, Text } from 'react-native'
import { CommonStyles } from '../styles/commonStyles'

interface HeaderButtonProps {
  onPress: () => void
  title: string
}

export function HeaderButton({ onPress, title }: HeaderButtonProps) {
  return (
    <TouchableOpacity style={CommonStyles.headerButton} onPress={onPress}>
      <Text style={CommonStyles.headerButtonText}>{title}</Text>
    </TouchableOpacity>
  )
}
