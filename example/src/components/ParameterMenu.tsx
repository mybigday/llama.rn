import React from 'react'
import { View, Text, TouchableOpacity, Platform } from 'react-native'
import { MenuView } from '@react-native-menu/menu'
import Icon from '@react-native-vector-icons/material-design-icons'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'

interface ParameterMenuProps {
  label: string
  description?: string
  value?: string
  options: string[]
  onSelect: (value: string) => void
  placeholder?: string
}

export function ParameterMenu({
  label,
  description,
  value,
  options,
  onSelect,
  placeholder = 'Select...',
}: ParameterMenuProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const menuActions = options.map((option) => ({
    id: option,
    title: option,
    ...(Platform.OS === 'ios' && value === option && {
      state: 'on' as const,
    }),
  }))

  const handleMenuAction = ({ nativeEvent }: { nativeEvent: { event: string } }) => {
    onSelect(nativeEvent.event)
  }

  return (
    <View style={themedStyles.paramGroup}>
      <Text style={themedStyles.paramLabel}>{label}</Text>
      {description && (
        <Text style={themedStyles.paramDescription}>{description}</Text>
      )}

      <MenuView
        onPressAction={handleMenuAction}
        actions={menuActions}
        shouldOpenOnLongPress={false}
      >
        <TouchableOpacity
          style={{
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderWidth: 1,
            borderColor: theme.colors.border,
            borderRadius: 8,
            paddingHorizontal: 12,
            paddingVertical: 16,
            backgroundColor: theme.colors.inputBackground,
          }}
        >
          <Text
            style={{
              fontSize: 16,
              color: value ? theme.colors.text : theme.colors.textSecondary,
              flex: 1,
            }}
          >
            {value || placeholder}
          </Text>
          <Icon name="menu-down" size={24} color={theme.colors.textSecondary} />
        </TouchableOpacity>
      </MenuView>
    </View>
  )
}
