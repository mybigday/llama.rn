import React, { type ReactNode } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
} from 'react-native'
import { CommonStyles } from '../styles/commonStyles'

const styles = StyleSheet.create({
  container: CommonStyles.container,
  header: CommonStyles.modalHeader,
  title: CommonStyles.modalTitle,
  cancelButton: CommonStyles.headerButtonText,
  saveButton: {
    ...CommonStyles.headerButtonText,
    fontWeight: '600',
  },
  disabledButton: CommonStyles.disabledButton,
  content: {
    flex: 1,
    paddingHorizontal: 16,
  },
  description: CommonStyles.description,
  resetButton: {
    backgroundColor: '#FF3B30',
    borderRadius: 8,
    paddingVertical: 12,
    marginTop: 20,
    marginBottom: 20,
  },
  resetButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  bottomPadding: {
    height: 30,
  },
  warningContainer: {
    backgroundColor: '#FFFBEB',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
    alignItems: 'center',
  },
  warningText: {
    color: '#D97706',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
})

interface BaseParameterModalProps {
  visible: boolean
  onClose: () => void
  title: string
  description: string
  isLoading: boolean
  onSave: () => void
  onReset: () => void
  showWarning?: boolean
  warningText?: string
  children: ReactNode
}

export default function BaseParameterModal({
  visible,
  onClose,
  title,
  description,
  isLoading,
  onSave,
  onReset,
  showWarning = false,
  warningText = '',
  children,
}: BaseParameterModalProps) {
  return (
    <Modal
      visible={visible}
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onClose}>
            <Text style={styles.cancelButton}>Cancel</Text>
          </TouchableOpacity>
          <Text style={styles.title}>{title}</Text>
          <TouchableOpacity onPress={onSave} disabled={isLoading}>
            <Text
              style={[styles.saveButton, isLoading && styles.disabledButton]}
            >
              Save
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          <Text style={styles.description}>{description}</Text>

          {showWarning && (
            <View style={styles.warningContainer}>
              <Text style={styles.warningText}>{warningText}</Text>
            </View>
          )}

          {children}

          <TouchableOpacity style={styles.resetButton} onPress={onReset}>
            <Text style={styles.resetButtonText}>Reset to Defaults</Text>
          </TouchableOpacity>

          <View style={styles.bottomPadding} />
        </ScrollView>
      </View>
    </Modal>
  )
}
