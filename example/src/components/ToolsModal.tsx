import React, { useState, useEffect } from 'react'
import {
  Modal,
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  TextInput,
  Alert,
  Platform,
} from 'react-native'
import { SafeAreaView } from 'react-native-safe-area-context'
import { CommonStyles } from '../styles/commonStyles'

interface Tool {
  type: string
  function: {
    name: string
    description: string
    parameters: any
  }
}

interface ToolsModalProps {
  visible: boolean
  onClose: () => void
  tools: Tool[]
  onSave: (tools: Tool[], mockResponses: Record<string, string>) => void
  mockResponses?: Record<string, string>
}

const styles = StyleSheet.create({
  container: CommonStyles.container,
  header: {
    ...CommonStyles.modalHeader,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#e9ecef',
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
      },
      android: {
        elevation: 2,
      },
    }),
  },
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
  textArea: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    minHeight: 400,
    textAlignVertical: 'top',
    backgroundColor: '#f8f9fa',
  },
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
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    marginTop: 8,
    fontWeight: '500',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
    color: '#1a1a1a',
  },
  toolCard: {
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#e9ecef',
  },
  toolName: {
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 4,
    color: '#495057',
  },
  toolDescription: {
    fontSize: 12,
    color: '#6c757d',
    marginBottom: 8,
  },
  mockResponseInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 6,
    padding: 8,
    fontSize: 14,
    minHeight: 60,
    textAlignVertical: 'top',
    backgroundColor: 'white',
  },
  mockResponseLabel: {
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 4,
    color: '#495057',
  },
})

export default function ToolsModal({
  visible,
  onClose,
  tools,
  onSave,
  mockResponses = {},
}: ToolsModalProps) {
  const [toolsJson, setToolsJson] = useState('')
  const [currentMockResponses, setCurrentMockResponses] = useState<
    Record<string, string>
  >({})
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (visible) {
      setToolsJson(JSON.stringify(tools, null, 2))
      setCurrentMockResponses(mockResponses)
      setError('')
    }
  }, [visible, tools, mockResponses])

  const handleSave = () => {
    setIsLoading(true)
    setError('')

    try {
      const parsedTools = JSON.parse(toolsJson)

      // Validate the structure
      if (!Array.isArray(parsedTools)) {
        throw new TypeError('Tools must be an array')
      }

      parsedTools.forEach((tool) => {
        if (!tool.type || tool.type !== 'function') {
          throw new TypeError('Each tool must have type: "function"')
        }
        if (
          !tool.function ||
          !tool.function.name ||
          !tool.function.description
        ) {
          throw new TypeError(
            'Each tool must have function.name and function.description',
          )
        }
        if (!tool.function.parameters) {
          throw new TypeError('Each tool must have function.parameters')
        }
      })

      onSave(parsedTools, currentMockResponses)
      onClose()
    } catch (parseError: any) {
      setError(`Invalid JSON: ${parseError.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    Alert.alert(
      'Reset Tools',
      'Are you sure you want to reset to default tools? This will lose all custom changes.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            setToolsJson(JSON.stringify(tools, null, 2))
            setCurrentMockResponses(mockResponses)
            setError('')
          },
        },
      ],
    )
  }

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
          <Text style={styles.title}>Edit Tools</Text>
          <TouchableOpacity onPress={handleSave} disabled={isLoading}>
            <Text
              style={[styles.saveButton, isLoading && styles.disabledButton]}
            >
              Save
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          <Text style={styles.description}>
            Edit the JSON schema for available tools and configure mock
            responses for testing.
          </Text>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Tools JSON Schema</Text>
            <TextInput
              style={styles.textArea}
              value={toolsJson}
              onChangeText={(text) => {
                setToolsJson(text)
                setError('')
              }}
              multiline
              placeholder="Enter tools JSON schema..."
              placeholderTextColor="#999"
            />
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Mock Responses</Text>
            {(() => {
              try {
                const parsedTools = JSON.parse(toolsJson)
                return parsedTools.map((tool: Tool, index: number) => (
                  <View
                    key={`${tool.function.name}-${index}`}
                    style={styles.toolCard}
                  >
                    <Text style={styles.toolName}>{tool.function.name}</Text>
                    <Text style={styles.toolDescription}>
                      {tool.function.description}
                    </Text>
                    <Text style={styles.mockResponseLabel}>Mock Response:</Text>
                    <TextInput
                      style={styles.mockResponseInput}
                      value={currentMockResponses[tool.function.name] || ''}
                      onChangeText={(text) => {
                        setCurrentMockResponses((prev) => ({
                          ...prev,
                          [tool.function.name]: text,
                        }))
                      }}
                      multiline
                      placeholder={`Enter mock response for ${tool.function.name}...`}
                      placeholderTextColor="#999"
                    />
                  </View>
                ))
              } catch (e: any) {
                return (
                  <Text style={styles.errorText}>
                    {`Invalid JSON (${e.message}) - fix tools JSON below to see mock response inputs`}
                  </Text>
                )
              }
            })()}
          </View>

          {error ? <Text style={styles.errorText}>{error}</Text> : null}

          <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
            <Text style={styles.resetButtonText}>Reset to Defaults</Text>
          </TouchableOpacity>

          <View style={styles.bottomPadding} />
        </ScrollView>
      </SafeAreaView>
    </Modal>
  )
}
