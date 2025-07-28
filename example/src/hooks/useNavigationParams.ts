import { useLayoutEffect } from 'react'

interface NavigationParamsConfig {
  navigation: any
  isModelReady: boolean
  onShowContextSettings?: () => void
  onShowCompletionSettings?: () => void
}

export function useNavigationParams({
  navigation,
  isModelReady,
  onShowContextSettings,
  onShowCompletionSettings,
}: NavigationParamsConfig) {
  useLayoutEffect(() => {
    if (isModelReady && onShowCompletionSettings) {
      navigation.setParams({
        showCompletionSettings: onShowCompletionSettings,
        showContextSettings: null,
      })
    } else if (!isModelReady && onShowContextSettings) {
      navigation.setParams({
        showContextSettings: onShowContextSettings,
        showCompletionSettings: null,
      })
    }
  }, [
    navigation,
    isModelReady,
    onShowContextSettings,
    onShowCompletionSettings,
  ])
}
