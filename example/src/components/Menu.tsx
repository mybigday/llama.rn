import React from 'react'
import { Platform } from 'react-native'
import { MenuView } from '@react-native-menu/menu'
import Icon from '@react-native-vector-icons/material-design-icons'

interface MenuAction {
  id: string
  title: string
  onPress: () => void
  systemIcon?: string
}

interface MenuProps {
  actions: MenuAction[]
}

export function Menu({ actions }: MenuProps) {
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
      <Icon name="dots-vertical" size={24} color="#007AFF" />
    </MenuView>
  )
}
