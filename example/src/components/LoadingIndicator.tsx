/* eslint-disable react/require-default-props */
import React from 'react'
import { View, Text, ActivityIndicator } from 'react-native'
import { CommonStyles, Colors } from '../styles/commonStyles'

interface LoadingIndicatorProps {
  text?: string
  size?: 'small' | 'large'
  color?: string
}

export function LoadingIndicator(props: LoadingIndicatorProps) {
  const { text = 'Loading...', size = 'large', color = Colors.primary } = props

  return (
    <View style={CommonStyles.loadingContainer}>
      <ActivityIndicator size={size} color={color} />
      <Text style={CommonStyles.loadingText}>{text}</Text>
    </View>
  )
}
