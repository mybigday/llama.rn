import React from 'react'
import { View, Text, TouchableOpacity, Platform } from 'react-native'
import { MenuView } from '@react-native-menu/menu'
import Icon from '@react-native-vector-icons/material-design-icons'
import { CommonStyles } from '../styles/commonStyles'

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
    <View style={{ marginVertical: 8 }}>
      <Text style={CommonStyles.paramLabel}>{label}</Text>
      {description && (
        <Text style={CommonStyles.paramDescription}>{description}</Text>
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
            borderColor: '#ddd',
            borderRadius: 8,
            paddingHorizontal: 12,
            paddingVertical: 16,
            backgroundColor: '#fff',
          }}
        >
          <Text
            style={{
              fontSize: 16,
              color: value ? '#333' : '#999',
              flex: 1,
            }}
          >
            {value || placeholder}
          </Text>
          <Icon name="menu-down" size={24} color="#666" />
        </TouchableOpacity>
      </MenuView>
    </View>
  )
}
