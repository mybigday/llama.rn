import { useCallback, useState } from 'react'
import { Alert } from 'react-native'

interface UseParameterModalOptions<T> {
  loadParams: () => Promise<T>
  saveParams: (params: T) => Promise<void>
  resetParams: () => Promise<void>
  defaultParams: T
  successMessage?: string
}

export function useParameterModal<T>({
  loadParams,
  saveParams,
  resetParams,
  defaultParams,
}: UseParameterModalOptions<T>) {
  const [params, setParams] = useState<T>(defaultParams)
  const [isLoading, setIsLoading] = useState(false)

  const loadParamsAsync = useCallback(async () => {
    try {
      const savedParams = await loadParams()
      setParams(savedParams)
    } catch (error) {
      console.error('Error loading params:', error)
    }
  }, [loadParams])

  const handleSave = async (
    onSave?: (params: T) => void,
    onClose?: () => void,
  ) => {
    try {
      setIsLoading(true)
      await saveParams(params)
      onSave?.(params)
      onClose?.()
    } catch (error: any) {
      Alert.alert('Error', `Failed to save parameters: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = async () => {
    Alert.alert(
      'Reset Parameters',
      'Are you sure you want to reset to default values?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              await resetParams()
              setParams(defaultParams)
              Alert.alert('Success', 'Parameters reset to defaults!')
            } catch (error: any) {
              Alert.alert('Error', `Failed to reset: ${error.message}`)
            }
          },
        },
      ],
    )
  }

  const updateParam = (key: keyof T, value: any) => {
    setParams((prev) => ({ ...prev, [key]: value }))
  }

  return {
    params,
    setParams,
    isLoading,
    loadParamsAsync,
    handleSave,
    handleReset,
    updateParam,
  }
}
