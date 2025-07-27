/* eslint-disable react/require-default-props */
import React from 'react'
import { View } from 'react-native'
import type { DimensionValue } from 'react-native'
import { CommonStyles } from '../styles/commonStyles'

interface ProgressBarProps {
  progress: number // 0-100
  width?: DimensionValue
  height?: number
}

export function ProgressBar(props: ProgressBarProps) {
  const { progress, width = '80%', height = 8 } = props

  return (
    <View style={CommonStyles.progressContainer}>
      <View style={[CommonStyles.progressBar, { width, height }]}>
        <View
          style={[
            CommonStyles.progressFill,
            { width: `${Math.min(100, Math.max(0, progress))}%` },
          ]}
        />
      </View>
    </View>
  )
}
