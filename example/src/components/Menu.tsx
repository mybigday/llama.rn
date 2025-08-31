import React from 'react'
import { Platform } from 'react-native'
import { MenuView } from '@react-native-menu/menu'
import Icon from '@react-native-vector-icons/material-design-icons'
import { useTheme } from '../contexts/ThemeContext'

interface MenuAction {
  id: string
  title: string
  onPress: () => void
  systemIcon?: string
}

interface MenuProps {
  actions: MenuAction[]
  icon?: 'dots-vertical' | 'theme-light-dark'
}

export function Menu({ icon = 'dots-vertical', actions }: MenuProps) {
  const { theme } = useTheme()

  const menuActions = actions.map((action) => ({
    id: action.id,
    title: action.title,
    image: Platform.select({
      ios: action.systemIcon || 'ellipsis.circle',
      android: 'ic_menu_more',
    }),
  }))

  const handleMenuAction = ({ nativeEvent }: { nativeEvent: { event: string } }) => {
    const action = actions.find(a => a.id === nativeEvent.event)
    action?.onPress()
  }

  return (
    <MenuView
      onPressAction={handleMenuAction}
      actions={menuActions}
    >
      <Icon name={icon} size={24} color={theme.colors.primary} />
    </MenuView>
  )
}
