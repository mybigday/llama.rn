import React from 'react'
import { Modal, View, Text, TouchableOpacity, ScrollView } from 'react-native'
import { CommonStyles } from '../styles/commonStyles'

interface BaseModalProps {
  visible: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  onSave?: () => void
  saveDisabled?: boolean
  saveText?: string
  cancelText?: string
}

export function BaseModal(props: BaseModalProps) {
  const {
    visible,
    onClose,
    title,
    children,
    onSave,
    saveDisabled,
    saveText,
    cancelText,
  } = props
  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={CommonStyles.container}>
        <View style={CommonStyles.modalHeader}>
          <TouchableOpacity onPress={onClose}>
            <Text style={CommonStyles.headerButtonText}>{cancelText}</Text>
          </TouchableOpacity>
          <Text style={CommonStyles.modalTitle}>{title}</Text>
          {onSave && (
            <TouchableOpacity
              onPress={onSave}
              disabled={saveDisabled}
              style={saveDisabled ? CommonStyles.disabledButton : undefined}
            >
              <Text
                style={[CommonStyles.headerButtonText, { fontWeight: '600' }]}
              >
                {saveText}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        <ScrollView
          style={CommonStyles.flex1}
          contentContainerStyle={{ paddingHorizontal: 16 }}
          showsVerticalScrollIndicator={false}
        >
          {children}
        </ScrollView>
      </View>
    </Modal>
  )
}
