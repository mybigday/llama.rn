import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  ActivityIndicator,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Platform,
} from 'react-native'
import type { ContextParams } from '../utils/storage'
import {
  saveContextParams,
  loadContextParams,
  resetContextParams,
  DEFAULT_CONTEXT_PARAMS,
} from '../utils/storage'
import { useParameterModal } from '../hooks/useParameterModal'
import { ParameterTextInput, ParameterSwitch } from './ParameterFormFields'
import { ParameterMenu } from './ParameterMenu'
import BaseParameterModal from './BaseParameterModal'
import { useTheme } from '../contexts/ThemeContext'
import { createThemedStyles } from '../styles/commonStyles'
import { getBackendDevicesInfo } from '../../../src'
import type { NativeBackendDeviceInfo } from '../../../src'

interface ContextParamsModalProps {
  visible: boolean
  onClose: () => void
  onSave: (params: ContextParams) => void
}

const CACHE_TYPE_OPTIONS = [
  'f16',
  'f32',
  'q8_0',
  'q4_0',
  'q4_1',
  'iq4_nl',
  'q5_0',
  'q5_1',
]

export default function ContextParamsModal({
  visible,
  onClose,
  onSave,
}: ContextParamsModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const deviceStyles = useMemo(
    () =>
      StyleSheet.create({
        devicesSection: {
          marginTop: 6,
        },
        deviceItem: {
          borderWidth: 1,
          borderColor: theme.colors.border,
          borderRadius: 10,
          paddingHorizontal: 12,
          paddingVertical: 8,
          marginBottom: 6,
          backgroundColor: theme.colors.inputBackground,
        },
        deviceItemSelected: {
          borderColor: theme.colors.primary,
          backgroundColor: theme.dark
            ? 'rgba(17, 122, 255, 0.18)'
            : 'rgba(17, 122, 255, 0.08)',
        },
        deviceHeader: {
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
        },
        deviceRight: {
          flexDirection: 'row',
          alignItems: 'center',
          gap: 6,
        },
        deviceName: {
          flex: 1,
          fontSize: 15,
          fontWeight: '600',
          color: theme.colors.text,
          marginRight: 6,
        },
        deviceMeta: {
          fontSize: 11,
          color: theme.colors.textSecondary,
        },
        deviceMetaRow: {
          flexDirection: 'row',
          alignItems: 'center',
          marginTop: 4,
          flexWrap: 'wrap',
          gap: 8,
        },
        deviceBadge: {
          paddingHorizontal: 8,
          paddingVertical: 2,
          borderRadius: 10,
          backgroundColor: theme.colors.card,
          borderWidth: 1,
          borderColor: theme.colors.border,
        },
        deviceBadgeText: {
          color: theme.colors.text,
          fontSize: 10,
          fontWeight: '600',
        },
        selectionIndicator: {
          width: 16,
          height: 16,
          borderRadius: 8,
          borderWidth: 2,
          borderColor: theme.colors.border,
        },
        selectionIndicatorSelected: {
          borderColor: theme.colors.primary,
          backgroundColor: theme.colors.primary,
        },
        helperText: {
          fontSize: 11,
          color: theme.colors.textSecondary,
          marginTop: 2,
        },
        loadingRow: {
          flexDirection: 'row',
          alignItems: 'center',
          marginTop: 6,
        },
        loadingText: {
          marginLeft: 8,
          color: theme.colors.textSecondary,
          fontSize: 13,
        },
        errorText: {
          marginTop: 6,
          color: theme.colors.error,
          fontSize: 12,
        },
        emptyText: {
          marginTop: 6,
          color: theme.colors.textSecondary,
          fontSize: 12,
        },
      }),
    [theme],
  )
  const {
    params,
    isLoading,
    loadParamsAsync,
    handleSave,
    handleReset,
    updateParam,
  } = useParameterModal({
    loadParams: loadContextParams,
    saveParams: saveContextParams,
    resetParams: resetContextParams,
    defaultParams: DEFAULT_CONTEXT_PARAMS,
  })
  const [availableDevices, setAvailableDevices] = useState<
    NativeBackendDeviceInfo[]
  >([])
  const [isLoadingDevices, setIsLoadingDevices] = useState(false)
  const [devicesError, setDevicesError] = useState<string | null>(null)

  useEffect(() => {
    if (visible) loadParamsAsync()
  }, [loadParamsAsync, visible])

  useEffect(() => {
    let isMounted = true
    const fetchDevices = async () => {
      if (!visible || availableDevices.length > 0) return

      try {
        setIsLoadingDevices(true)
        const devices = await getBackendDevicesInfo()
        if (isMounted) {
          setAvailableDevices(devices)
          setDevicesError(null)
        }
      } catch (error: any) {
        console.error('Error loading backend devices:', error)
        if (isMounted) {
          setDevicesError(error?.message ?? 'Failed to load devices')
        }
      } finally {
        if (isMounted) {
          setIsLoadingDevices(false)
        }
      }
    }

    fetchDevices()
    return () => {
      isMounted = false
    }
  }, [availableDevices.length, visible])

  const handleTextInput = (text: string, paramKey: keyof ContextParams) => {
    if (text === '') {
      updateParam(paramKey, undefined)
    } else {
      const parsedValue = parseInt(text, 10)
      updateParam(paramKey, Number.isNaN(parsedValue) ? text : parsedValue)
    }
  }

  const validateIntegerParam = (
    value: any,
    min: number,
    max: number,
    fieldName: string,
  ): string | null => {
    if (value === undefined || value === null) return null

    const num = typeof value === 'string' ? parseInt(value, 10) : value
    if (Number.isNaN(num) || num < min || num > max) {
      return `${fieldName} must be between ${min} and ${max}`
    }
    return null
  }

  const validateParams = (): { isValid: boolean; errors: string[] } => {
    const validations = [
      validateIntegerParam(params.n_ctx, 128, 999999, 'Context Size'),
      validateIntegerParam(params.n_gpu_layers, 0, 99, 'GPU Layers'),
      validateIntegerParam(params.n_batch, 1, 99999, 'Batch Size'),
      validateIntegerParam(params.n_ubatch, 1, 99999, 'Micro Batch Size'),
      validateIntegerParam(params.n_threads, 1, 32, 'Threads'),
      validateIntegerParam(params.n_cpu_moe, 0, 99, 'CPU MoE Layers'),
    ]

    const errors = validations.filter(
      (error): error is string => error !== null,
    )
    return { isValid: errors.length === 0, errors }
  }

  const convertStringParamsToNumbers = (
    stringParams: ContextParams,
  ): ContextParams => {
    const converted = { ...stringParams }

    if (typeof converted.n_ctx === 'string') {
      const num = parseInt(converted.n_ctx, 10)
      converted.n_ctx = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_gpu_layers === 'string') {
      const num = parseInt(converted.n_gpu_layers, 10)
      converted.n_gpu_layers = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_batch === 'string') {
      const num = parseInt(converted.n_batch, 10)
      converted.n_batch = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_ubatch === 'string') {
      const num = parseInt(converted.n_ubatch, 10)
      converted.n_ubatch = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_threads === 'string') {
      const num = parseInt(converted.n_threads, 10)
      converted.n_threads = Number.isNaN(num) ? undefined : num
    }

    if (typeof converted.n_cpu_moe === 'string') {
      const num = parseInt(converted.n_cpu_moe, 10)
      converted.n_cpu_moe = Number.isNaN(num) ? undefined : num
    }

    return converted
  }

  const onSaveHandler = () => {
    const validation = validateParams()
    if (!validation.isValid) {
      Alert.alert(
        'Validation Error',
        `Please fix the following errors:\n\n${validation.errors.join('\n')}`,
        [{ text: 'OK' }],
      )
      return
    }

    const convertedParams = convertStringParamsToNumbers(params)
    handleSave((_params) => onSave(convertedParams), onClose)
  }

  const formatBytes = (bytes?: number) => {
    if (!bytes || bytes <= 0) return 'Unknown memory'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    const i = Math.min(
      Math.floor(Math.log(bytes) / Math.log(k)),
      sizes.length - 1,
    )
    const formatted = bytes / Math.pow(k, i)
    return `${formatted.toFixed(formatted >= 100 || i === 0 ? 0 : 1)} ${
      sizes[i]
    }`
  }

  const isAllDevicesSelected = !params.devices || params.devices.length === 0

  const toggleDeviceSelection = (deviceName: string) => {
    const existing = params.devices ?? []
    const isSelected = existing.includes(deviceName)
    if (isSelected) {
      const filtered = existing.filter((dev) => dev !== deviceName)
      updateParam('devices', filtered.length > 0 ? filtered : undefined)
    } else {
      updateParam('devices', [...existing, deviceName])
    }
  }

  return (
    <BaseParameterModal
      visible={visible}
      onClose={onClose}
      title="Context Parameters"
      description="Configure context initialization parameters. These settings affect memory usage, performance, and model behavior during loading."
      isLoading={isLoading}
      onSave={onSaveHandler}
      onReset={handleReset}
      showWarning
      warningText="Warning: Changing context parameters requires reinitializing the model, which will clear your current conversation."
    >
      {/* Context Size */}
      <ParameterTextInput
        label="Context Size (n_ctx)"
        description="Maximum context length in tokens. Higher values use more memory."
        value={params.n_ctx?.toString()}
        onChangeText={(text) => {
          // Allow any text input, validation happens on save
          if (text === '') {
            updateParam('n_ctx', undefined)
          } else {
            const parsedValue = parseInt(text, 10)
            updateParam('n_ctx', Number.isNaN(parsedValue) ? text : parsedValue)
          }
        }}
        keyboardType="numeric"
        placeholder="8192"
      />

      {/* GPU Layers */}
      <ParameterTextInput
        label="GPU Layers (n_gpu_layers)"
        description="Number of layers to run on GPU. Use 99 for all layers, 0 for CPU only."
        value={params.n_gpu_layers?.toString()}
        onChangeText={(text) => {
          // Allow any text input, validation happens on save
          if (text === '') {
            updateParam('n_gpu_layers', undefined)
          } else {
            const parsedValue = parseInt(text, 10)
            updateParam(
              'n_gpu_layers',
              Number.isNaN(parsedValue) ? text : parsedValue,
            )
          }
        }}
        keyboardType="numeric"
        placeholder="99"
      />

      <View style={themedStyles.paramGroup}>
        <Text style={themedStyles.paramLabel}>Devices</Text>
        <Text style={themedStyles.paramDescription}>
          Select the backend devices to use when initializing the context. Leave
          it on &quot;All available devices&quot; to let llama.rn pick
          automatically.
        </Text>

        <View style={deviceStyles.devicesSection}>
          <TouchableOpacity
            style={[
              deviceStyles.deviceItem,
              isAllDevicesSelected && deviceStyles.deviceItemSelected,
            ]}
            onPress={() => updateParam('devices', undefined)}
          >
            <View style={deviceStyles.deviceHeader}>
              <View>
                <Text style={deviceStyles.deviceName}>
                  All available devices
                </Text>
                <Text style={deviceStyles.deviceMeta}>
                  Use default backend selection
                </Text>
              </View>
              <View
                style={[
                  deviceStyles.selectionIndicator,
                  isAllDevicesSelected &&
                    deviceStyles.selectionIndicatorSelected,
                ]}
              />
            </View>
          </TouchableOpacity>

          {isLoadingDevices && (
            <View style={deviceStyles.loadingRow}>
              <ActivityIndicator color={theme.colors.primary} size="small" />
              <Text style={deviceStyles.loadingText}>Loading devices...</Text>
            </View>
          )}

          {devicesError && !isLoadingDevices && (
            <Text style={deviceStyles.errorText}>{devicesError}</Text>
          )}

          {!isLoadingDevices &&
            availableDevices.length === 0 &&
            !devicesError && (
              <Text style={deviceStyles.emptyText}>
                No backend devices reported. Defaults will be used.
              </Text>
            )}

          {availableDevices.map((device) => {
            const isSelected =
              !isAllDevicesSelected &&
              (params.devices?.includes(device.deviceName) ?? false)
            return (
              <TouchableOpacity
                key={device.deviceName}
                style={[
                  deviceStyles.deviceItem,
                  isSelected && deviceStyles.deviceItemSelected,
                ]}
                onPress={() => toggleDeviceSelection(device.deviceName)}
              >
                <View style={deviceStyles.deviceHeader}>
                  <Text style={deviceStyles.deviceName}>
                    {device.deviceName}
                  </Text>
                  <View style={deviceStyles.deviceRight}>
                    <View style={deviceStyles.deviceBadge}>
                      <Text style={deviceStyles.deviceBadgeText}>
                        {device.type.toUpperCase()}
                      </Text>
                    </View>
                    <View
                      style={[
                        deviceStyles.selectionIndicator,
                        isSelected && deviceStyles.selectionIndicatorSelected,
                      ]}
                    />
                  </View>
                </View>
                <View style={deviceStyles.deviceMetaRow}>
                  <Text style={deviceStyles.deviceMeta}>
                    {`Backend: ${device.backend}`}
                  </Text>
                  <Text style={deviceStyles.deviceMeta}>
                    {`Memory: ${formatBytes(device.maxMemorySize)}`}
                  </Text>
                </View>
              </TouchableOpacity>
            )
          })}

          {Platform.OS === 'android' && (
            <Text style={deviceStyles.helperText}>
              Tip: On Android, selecting devices starting with &quot;HTP&quot;
              enables Hexagon acceleration. Wildcards such as HTP* are
              supported.
            </Text>
          )}
        </View>
      </View>

      {/* Memory Lock */}
      <ParameterSwitch
        label="Memory Lock (use_mlock)"
        description="Lock model in memory to prevent swapping to disk."
        value={params.use_mlock || false}
        onValueChange={(value) => updateParam('use_mlock', value)}
      />

      {/* Memory Map */}
      <ParameterSwitch
        label="Memory Map (use_mmap)"
        description="Use memory mapping for better performance."
        value={params.use_mmap || false}
        onValueChange={(value) => updateParam('use_mmap', value)}
      />

      {/* Batch Size */}
      <ParameterTextInput
        label="Batch Size (n_batch)"
        description="Number of tokens to process in parallel. Higher values use more memory."
        value={params.n_batch?.toString() || '512'}
        onChangeText={(text) => handleTextInput(text, 'n_batch')}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Micro Batch Size */}
      <ParameterTextInput
        label="Micro Batch Size (n_ubatch)"
        description="Internal batch size for processing. Should be ≤ n_batch."
        value={params.n_ubatch?.toString() || '512'}
        onChangeText={(text) => handleTextInput(text, 'n_ubatch')}
        keyboardType="numeric"
        placeholder="512"
      />

      {/* Threads */}
      <ParameterTextInput
        label="Threads (n_threads)"
        description="Number of CPU threads to use. Usually set to number of CPU cores."
        value={params.n_threads?.toString()}
        onChangeText={(text) => handleTextInput(text, 'n_threads')}
        keyboardType="numeric"
        placeholder="8"
      />

      {/* CPU MoE Layers */}
      <ParameterTextInput
        label="CPU MoE Layers (n_cpu_moe)"
        description="Number of MoE layers to keep on CPU. Use 0 to disable, higher values for more CPU processing."
        value={params.n_cpu_moe?.toString() || '0'}
        onChangeText={(text) => handleTextInput(text, 'n_cpu_moe')}
        keyboardType="numeric"
        placeholder="0"
      />

      {/* Context Shift */}
      <ParameterSwitch
        label="Context Shift (ctx_shift)"
        description="Enable automatic context shifting when context is full."
        value={params.ctx_shift || false}
        onValueChange={(value) => updateParam('ctx_shift', value)}
      />

      {/* Flash Attention Type */}
      <ParameterMenu
        label="Flash Attention (flash_attn_type)"
        description="Flash attention type. Only recommended on GPU devices."
        value={params.flash_attn_type}
        options={['auto', 'on', 'off']}
        onSelect={(value) => updateParam('flash_attn_type', value)}
        placeholder="auto"
      />

      {/* Cache Type K */}
      <ParameterMenu
        label="Cache Type K (cache_type_k)"
        description="KV cache data type for the K. Need flash_attn_type set to 'on' to change this."
        value={params.cache_type_k}
        options={CACHE_TYPE_OPTIONS}
        onSelect={(value) => updateParam('cache_type_k', value)}
        placeholder="f16"
      />

      {/* Cache Type V */}
      <ParameterMenu
        label="Cache Type V (cache_type_v)"
        description="KV cache data type for the V. Need flash_attn_type set to 'on' to change this."
        value={params.cache_type_v}
        options={CACHE_TYPE_OPTIONS}
        onSelect={(value) => updateParam('cache_type_v', value)}
        placeholder="f16"
      />

      {/* KV Unified */}
      <ParameterSwitch
        label="KV Unified (kv_unified)"
        description="Use unified key-value store for better performance."
        value={params.kv_unified || false}
        onValueChange={(value) => updateParam('kv_unified', value)}
      />

      {/* SWA Full */}
      <ParameterSwitch
        label="SWA Full (swa_full)"
        description="Use full-size SWA cache. May improve performance for multiple sequences but uses more memory."
        value={params.swa_full || false}
        onValueChange={(value) => updateParam('swa_full', value)}
      />
    </BaseParameterModal>
  )
}
