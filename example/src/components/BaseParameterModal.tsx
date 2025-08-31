import React, { type ReactNode } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Platform,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'


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
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  
  const styles = StyleSheet.create({
    container: themedStyles.container,
    header: {
      ...themedStyles.modalHeader,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      backgroundColor: theme.colors.surface,
      paddingHorizontal: 20,
      paddingVertical: 16,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
      ...Platform.select({
        ios: {
          shadowColor: theme.colors.shadow,
          shadowOffset: { width: 0, height: 1 },
          shadowOpacity: 0.1,
          shadowRadius: 2,
        },
        android: {
          elevation: 2,
        },
      }),
    },
    title: themedStyles.modalTitle,
    cancelButton: {
      color: theme.colors.primary,
      fontSize: 16,
      fontWeight: '500',
    },
    saveButton: {
      color: theme.colors.primary,
      fontSize: 16,
      fontWeight: '600',
    },
    disabledButton: themedStyles.disabledButton,
    content: {
      flex: 1,
      paddingHorizontal: 16,
      backgroundColor: theme.colors.background,
    },
    description: themedStyles.description,
    resetButton: {
      backgroundColor: theme.colors.error,
      borderRadius: 8,
      paddingVertical: 12,
      marginTop: 20,
      marginBottom: 20,
    },
    resetButtonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
      textAlign: 'center',
    },
    bottomPadding: {
      height: 30,
    },
    warningContainer: {
      backgroundColor: theme.dark ? '#3F2A00' : '#FFFBEB',
      borderRadius: 8,
      padding: 12,
      marginBottom: 16,
      alignItems: 'center',
    },
    warningText: {
      color: theme.dark ? '#FBBF24' : '#D97706',
      fontSize: 14,
      fontWeight: '600',
      textAlign: 'center',
    },
  })
  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="pageSheet"
      onRequestClose={onClose}
    >
      <SafeAreaView style={styles.container}>
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
      </SafeAreaView>
    </Modal>
  )
}
