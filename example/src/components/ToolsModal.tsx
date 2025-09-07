import React, { useState, useEffect, useCallback } from 'react'
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
import { createThemedStyles } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import type { MCPConfig } from '../utils/storage'
import { loadMCPConfig, saveMCPConfig } from '../utils/storage'
import { mcpClientManager, type MCPConnection } from '../utils/mcpClient'

interface OpenAITool {
  type: string
  function: {
    name: string
    description: string
    parameters: any
  }
  mock_response?: string
}

interface AnthropicTool {
  name: string
  description: string
  input_schema: any
  mock_response?: string
}

type Tool = OpenAITool | AnthropicTool

interface ToolsModalProps {
  visible: boolean
  onClose: () => void
  tools: OpenAITool[]
  onSave: (tools: OpenAITool[], mockResponses: Record<string, string>) => void
  mockResponses?: Record<string, string>
  onMCPConfigSave?: (config: MCPConfig) => void
  disabledTools?: Set<string>
  onDisabledToolsChange?: (disabledTools: Set<string>) => void
}

// Schema converter functions
function convertOpenAIToAnthropic(openAITools: OpenAITool[]): AnthropicTool[] {
  return openAITools
    .filter(tool => tool.type === 'function')
    .map(tool => ({
      name: tool.function.name,
      description: tool.function.description,
      input_schema: tool.function.parameters,
      ...(tool.mock_response && { mock_response: tool.mock_response })
    }))
}

function convertAnthropicToOpenAI(anthropicTools: AnthropicTool[]): OpenAITool[] {
  return anthropicTools.map(tool => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema
    },
    ...(tool.mock_response && { mock_response: tool.mock_response })
  }))
}

function detectSchemaFormat(tools: any[]): 'openai' | 'anthropic' | 'unknown' {
  if (tools.length === 0) return 'openai' // default

  const firstTool = tools[0]
  if (firstTool.type === 'function' && firstTool.function) {
    return 'openai'
  }
  if (firstTool.name && firstTool.description && firstTool.input_schema) {
    return 'anthropic'
  }
  return 'unknown'
}

function getToolName(tool: Tool): string {
  if ('function' in tool) {
    return tool.function.name
  }
  return tool.name
}

function getToolDescription(tool: Tool): string {
  if ('function' in tool) {
    return tool.function.description
  }
  return tool.description
}

function extractMockResponsesFromTools(tools: any[]): Record<string, string> {
  const mockResponses: Record<string, string> = {}

  tools.forEach(tool => {
    const toolName = getToolName(tool)
    const mockResponse = tool.mock_response || (tool.function && tool.function.mock_response)

    if (mockResponse) {
      mockResponses[toolName] = mockResponse
    }
  })

  return mockResponses
}

export default function ToolsModal({
  visible,
  onClose,
  tools,
  onSave,
  mockResponses = {},
  onMCPConfigSave,
  disabledTools = new Set(),
  onDisabledToolsChange,
}: ToolsModalProps) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  // Tab and MCP state
  const [activeTab, setActiveTab] = useState<'custom' | 'mcp'>('custom')
  const [mcpConfigJson, setMcpConfigJson] = useState('')
  const [mcpConfig, setMcpConfig] = useState<MCPConfig>({ mcpServers: {} })
  const [mcpError, setMcpError] = useState('')
  const [mcpConnections, setMcpConnections] = useState<MCPConnection[]>([])
  const [mcpConnecting, setMcpConnecting] = useState(false)

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
          shadowOpacity: theme.dark ? 0.3 : 0.1,
          shadowRadius: 2,
        },
        android: {
          elevation: 2,
        },
      }),
    },
    title: {
      ...themedStyles.modalTitle,
      color: theme.colors.text,
      fontWeight: '700',
    },
    cancelButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.primary,
      fontWeight: '600',
    },
    saveButton: {
      ...themedStyles.headerButtonText,
      color: theme.colors.primary,
      fontWeight: '700',
    },
    disabledButton: {
      ...themedStyles.disabledButton,
      color: theme.colors.textSecondary,
    },
    content: {
      flex: 1,
      paddingHorizontal: 16,
      backgroundColor: theme.colors.background,
    },
    description: {
      ...themedStyles.description,
      color: theme.colors.textSecondary,
    },
    textArea: {
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 8,
      padding: 12,
      fontSize: 14,
      fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
      minHeight: 200,
      textAlignVertical: 'top',
      backgroundColor: theme.colors.surface,
      color: theme.colors.text,
    },
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
    errorText: {
      color: theme.colors.error,
      fontSize: 14,
      marginTop: 8,
      fontWeight: '500',
    },
    formatToggle: {
      flexDirection: 'row',
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      padding: 4,
      marginBottom: 16,
    },
    formatButton: {
      flex: 1,
      paddingVertical: 8,
      paddingHorizontal: 16,
      borderRadius: 6,
      alignItems: 'center',
    },
    formatButtonActive: {
      backgroundColor: theme.colors.primary,
    },
    formatButtonText: {
      fontSize: 14,
      fontWeight: '500',
      color: theme.colors.textSecondary,
    },
    formatButtonTextActive: {
      color: theme.colors.white,
    },
    section: {
      marginBottom: 24,
    },
    sectionTitle: {
      fontSize: 16,
      fontWeight: '600',
      marginBottom: 12,
      color: theme.colors.text,
    },
    toolCard: {
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      padding: 12,
      marginBottom: 12,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    toolName: {
      fontSize: 14,
      fontWeight: '600',
      marginBottom: 4,
      color: theme.colors.text,
    },
    toolDescription: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginBottom: 8,
    },
    mockResponseInput: {
      borderWidth: 1,
      borderColor: theme.colors.border,
      borderRadius: 6,
      padding: 8,
      fontSize: 14,
      minHeight: 60,
      textAlignVertical: 'top',
      backgroundColor: theme.colors.card,
      color: theme.colors.text,
    },
    mockResponseLabel: {
      fontSize: 12,
      fontWeight: '500',
      marginBottom: 4,
      color: theme.colors.text,
    },
    autoFillSingleButton: {
      backgroundColor: theme.colors.card,
      borderWidth: 1,
      borderColor: theme.colors.primary,
      borderRadius: 6,
      paddingVertical: 6,
      paddingHorizontal: 12,
      marginTop: 8,
      alignSelf: 'flex-start',
    },
    autoFillSingleButtonText: {
      color: theme.colors.primary,
      fontSize: 12,
      fontWeight: '500',
    },
    autoFillSchemaButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: 8,
      paddingVertical: 10,
      paddingHorizontal: 16,
      marginTop: 12,
      alignSelf: 'center',
    },
    autoFillSchemaButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '500',
      textAlign: 'center',
    },
    tabContainer: {
      flexDirection: 'row',
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      padding: 4,
      marginHorizontal: 16,
      marginVertical: 16,
    },
    tab: {
      flex: 1,
      paddingVertical: 8,
      paddingHorizontal: 16,
      borderRadius: 6,
      alignItems: 'center',
    },
    tabActive: {
      backgroundColor: theme.colors.primary,
    },
    tabText: {
      fontSize: 14,
      fontWeight: '500',
      color: theme.colors.textSecondary,
    },
    tabTextActive: {
      color: theme.colors.white,
    },
    connectionStatus: {
      backgroundColor: theme.colors.surface,
      borderRadius: 8,
      padding: 12,
      marginBottom: 12,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    connectionHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 8,
    },
    connectionName: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
    },
    connectionStatusText: {
      fontSize: 12,
      fontWeight: '500',
    },
    statusConnected: {
      color: '#4CAF50',
    },
    statusDisconnected: {
      color: theme.colors.error,
    },
    statusConnecting: {
      color: '#FF9800',
    },
    connectionDetails: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginBottom: 4,
    },
    toolsList: {
      marginTop: 8,
    },
    toolItem: {
      backgroundColor: theme.colors.card,
      borderRadius: 4,
      padding: 8,
      marginBottom: 4,
      borderLeftWidth: 3,
      borderLeftColor: theme.colors.primary,
    },
    toolItemDisabled: {
      opacity: 0.5,
      borderLeftColor: theme.colors.textSecondary,
    },
    toolItemName: {
      fontSize: 12,
      fontWeight: '600',
      color: theme.colors.text,
    },
    toolItemDescription: {
      fontSize: 11,
      color: theme.colors.textSecondary,
      marginTop: 2,
    },
    connectButton: {
      backgroundColor: theme.colors.primary,
      borderRadius: 6,
      paddingVertical: 8,
      paddingHorizontal: 16,
      marginTop: 12,
    },
    connectButtonText: {
      color: theme.colors.white,
      fontSize: 14,
      fontWeight: '500',
      textAlign: 'center',
    },
    connectButtonDisabled: {
      backgroundColor: theme.colors.textSecondary,
    },
  })
  const [toolsJson, setToolsJson] = useState('')
  const [currentMockResponses, setCurrentMockResponses] = useState<
    Record<string, string>
  >({})
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentFormat, setCurrentFormat] = useState<'openai' | 'anthropic'>('openai')

  const updateToolsDisplay = useCallback(() => {
    const displayTools = currentFormat === 'anthropic'
      ? convertOpenAIToAnthropic(tools)
      : tools
    setToolsJson(JSON.stringify(displayTools, null, 2))
  }, [currentFormat, tools])

  const autoFillEmptyMockResponses = useCallback((parsedTools: any[]) => {
    const extractedMockResponses = extractMockResponsesFromTools(parsedTools)

    if (Object.keys(extractedMockResponses).length > 0) {
      setCurrentMockResponses(prev => {
        const updated = { ...prev }
        // Only fill if the field is empty
        Object.entries(extractedMockResponses).forEach(([toolName, mockResponse]) => {
          if (!updated[toolName] || updated[toolName]?.trim() === '') {
            updated[toolName] = mockResponse
          }
        })
        return updated
      })
    }
  }, [])

  const connectToMCPServers = useCallback(async (config: MCPConfig) => {
    setMcpConnecting(true)
    setMcpError('')

    try {
      mcpClientManager.updateConfig(config)
      await mcpClientManager.connectToServers()
      setMcpConnections(mcpClientManager.getConnections())
    } catch (err: any) {
      setMcpError(`Failed to connect to MCP servers: ${err.message}`)
    } finally {
      setMcpConnecting(false)
    }
  }, [])

  const loadMCPConfiguration = useCallback(async () => {
    try {
      const config = await loadMCPConfig()
      setMcpConfig(config)
      setMcpConfigJson(JSON.stringify(config, null, 2))
      setMcpError('')
    } catch (err: any) {
      setMcpError(`Failed to load MCP config: ${err.message}`)
    }
  }, [])

  useEffect(() => {
    if (visible) {
      updateToolsDisplay()
      setCurrentMockResponses(mockResponses)
      setError('')

      // Load MCP configuration
      loadMCPConfiguration()

      // Auto-fill mock responses from tools schema
      try {
        const displayTools = currentFormat === 'anthropic'
          ? convertOpenAIToAnthropic(tools)
          : tools
        autoFillEmptyMockResponses(displayTools)
      } catch (e) {
        // Ignore JSON parsing errors
      }
    }
  }, [updateToolsDisplay, visible, tools, mockResponses, currentFormat, autoFillEmptyMockResponses, loadMCPConfiguration])

  const handleCustomToolsSave = () => {
    setIsLoading(true)
    setError('')

    try {
      const parsedTools = JSON.parse(toolsJson)

      // Validate the structure
      if (!Array.isArray(parsedTools)) {
        throw new TypeError('Tools must be an array')
      }

      // Convert to OpenAI format for saving (since the app expects OpenAI format)
      let toolsToSave: OpenAITool[]
      const detectedFormat = detectSchemaFormat(parsedTools)

      if (detectedFormat === 'anthropic') {
        toolsToSave = convertAnthropicToOpenAI(parsedTools)
      } else if (detectedFormat === 'openai') {
        toolsToSave = parsedTools
      } else {
        throw new TypeError('Invalid tool schema format')
      }

      // Validate OpenAI format
      toolsToSave.forEach((tool) => {
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

      onSave(toolsToSave, currentMockResponses)
      onClose()
    } catch (parseErr: any) {
      setError(`Invalid JSON: ${parseErr.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  const handleMCPSave = useCallback(async () => {
    setIsLoading(true)
    setMcpError('')

    try {
      const parsedConfig = JSON.parse(mcpConfigJson)

      // Basic validation
      if (!parsedConfig.mcpServers || typeof parsedConfig.mcpServers !== 'object') {
        throw new TypeError('Config must have mcpServers object')
      }

      // Validate server configurations
      Object.entries(parsedConfig.mcpServers).forEach(([name, server]: [string, any]) => {
        if (!server.type || !server.url) {
          throw new TypeError(`Server "${name}" must have type and url`)
        }
        if (!['streamable-http', 'sse'].includes(server.type)) {
          throw new TypeError(`Server "${name}" type must be "streamable-http" or "sse"`)
        }
      })

      await saveMCPConfig(parsedConfig)
      setMcpConfig(parsedConfig)

      if (onMCPConfigSave) {
        onMCPConfigSave(parsedConfig)
      }

      // Don't close modal
    } catch (parseErr: any) {
      setMcpError(`Invalid MCP config: ${parseErr.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [onMCPConfigSave, mcpConfigJson])

  const handleSave = () => {
    if (activeTab === 'custom') {
      handleCustomToolsSave()
    } else {
      handleMCPSave()
    }
  }


  const handleReset = () => {
    const resetTitle = activeTab === 'custom' ? 'Reset Tools' : 'Reset MCP Config'
    const resetMessage = activeTab === 'custom'
      ? 'Are you sure you want to reset to default tools? This will lose all custom changes.'
      : 'Are you sure you want to reset MCP configuration? This will lose all MCP server settings.'

    Alert.alert(
      resetTitle,
      resetMessage,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: () => {
            if (activeTab === 'custom') {
              setCurrentFormat('openai')
              updateToolsDisplay()
              setCurrentMockResponses(mockResponses)
              setError('')
            } else {
              const defaultConfig = { mcpServers: {} }
              setMcpConfig(defaultConfig)
              setMcpConfigJson(JSON.stringify(defaultConfig, null, 2))
              setMcpError('')
            }
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
            <Text style={styles.cancelButton}>Close</Text>
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

        {/* Tab Navigation */}
        <View style={styles.tabContainer}>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'custom' && styles.tabActive]}
            onPress={() => setActiveTab('custom')}
          >
            <Text style={[styles.tabText, activeTab === 'custom' && styles.tabTextActive]}>
              Custom
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.tab, activeTab === 'mcp' && styles.tabActive]}
            onPress={() => setActiveTab('mcp')}
          >
            <Text style={[styles.tabText, activeTab === 'mcp' && styles.tabTextActive]}>
              MCP
            </Text>
          </TouchableOpacity>
        </View>

        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {activeTab === 'custom' ? (
            <>
              <Text style={styles.description}>
                Edit the JSON schema for available tools and configure mock
                responses for testing. Toggle between OpenAI and Anthropic formats using the buttons above.
              </Text>
            </>
          ) : (
            <>
              <Text style={styles.description}>
                Configure MCP (Model Context Protocol) servers. Add server connections to extend tool capabilities.
              </Text>
            </>
          )}

          {activeTab === 'custom' && (
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
                placeholderTextColor={theme.colors.textSecondary}
                keyboardType="ascii-capable"
              />
              <TouchableOpacity
                style={styles.autoFillSchemaButton}
                onPress={() => {
                  try {
                    const parsedTools = JSON.parse(toolsJson)
                    autoFillEmptyMockResponses(parsedTools)
                  } catch (e) {
                    // Invalid JSON - ignore
                  }
                }}
              >
                <Text style={styles.autoFillSchemaButtonText}>Auto Fill Mock Responses from Schema</Text>
              </TouchableOpacity>
            </View>
          )}

          {activeTab === 'mcp' && (
            <>
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>MCP Configuration</Text>

                <TextInput
                  style={styles.textArea}
                  value={mcpConfigJson}
                  onChangeText={(text) => {
                    setMcpConfigJson(text)
                    setMcpError('')
                  }}
                  multiline
                  placeholder={`{
  "mcpServers": {
    "example": {
      "type": "streamable-http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer XXX"
      }
    }
  }
}`}
                  placeholderTextColor={theme.colors.textSecondary}
                  keyboardType="ascii-capable"
                />

                {Object.keys(mcpConfig.mcpServers).length > 0 && (
                  <TouchableOpacity
                    style={[styles.connectButton, (mcpConnecting || isLoading) && styles.connectButtonDisabled]}
                    onPress={() => {
                      connectToMCPServers(mcpConfig)
                    }}
                    disabled={mcpConnecting || isLoading}
                  >
                    <Text style={styles.connectButtonText}>
                      {mcpConnecting ? 'Connecting...' : 'Test Connections'}
                    </Text>
                  </TouchableOpacity>
                )}
              </View>

              {mcpConnections.length > 0 && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Connection Status</Text>
                  {mcpConnections.map((connection) => (
                    <View key={connection.serverId} style={styles.connectionStatus}>
                      <View style={styles.connectionHeader}>
                        <Text style={styles.connectionName}>{connection.serverId}</Text>
                        <Text
                          style={[
                            styles.connectionStatusText,
                            connection.connected ? styles.statusConnected : styles.statusDisconnected,
                          ]}
                        >
                          {(() => {
                            if (connection.connected) return 'Connected'
                            if (connection.error) return 'Error'
                            return 'Disconnected'
                          })()}
                        </Text>
                      </View>
                      {mcpConfig.mcpServers[connection.serverId] && (
                        <>
                          <Text style={styles.connectionDetails}>
                            Type:
                            {' '}
                            {mcpConfig.mcpServers[connection.serverId]?.type}
                          </Text>
                          <Text style={styles.connectionDetails}>
                            URL:
                            {' '}
                            {mcpConfig.mcpServers[connection.serverId]?.url}
                          </Text>
                        </>
                      )}
                      {connection.error && (
                        <Text style={styles.errorText}>{connection.error}</Text>
                      )}
                      {connection.connected && connection.tools.length > 0 && (
                        <View style={styles.toolsList}>
                          <Text style={styles.mockResponseLabel}>Available Tools:</Text>
                          {connection.tools.map((tool) => {
                            const isDisabled = disabledTools.has(tool.name)
                            return (
                              <TouchableOpacity
                                key={tool.name}
                                style={[styles.toolItem, isDisabled && styles.toolItemDisabled]}
                                onPress={() => {
                                  if (onDisabledToolsChange) {
                                    const newDisabledTools = new Set(disabledTools)
                                    if (isDisabled) {
                                      newDisabledTools.delete(tool.name)
                                    } else {
                                      newDisabledTools.add(tool.name)
                                    }
                                    onDisabledToolsChange(newDisabledTools)
                                  }
                                }}
                              >
                                <Text style={styles.toolItemName}>
                                  {tool.name}
                                  {isDisabled ? ' (disabled)' : ''}
                                </Text>
                                <Text style={styles.toolItemDescription}>{tool.description}</Text>
                              </TouchableOpacity>
                            )
                          })}
                        </View>
                      )}
                    </View>
                  ))}
                </View>
              )}
            </>
          )}

          {activeTab === 'custom' && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Mock Responses</Text>
              {(() => {
                try {
                  const parsedTools = JSON.parse(toolsJson)
                  return parsedTools.map((tool: any, index: number) => {
                    const toolName = getToolName(tool)
                    const toolDescription = getToolDescription(tool)

                    return (
                      <View
                        key={`${toolName}-${index}`}
                        style={styles.toolCard}
                      >
                        <Text style={styles.toolName}>{toolName}</Text>
                        <Text style={styles.toolDescription}>
                          {toolDescription}
                        </Text>
                        <Text style={styles.mockResponseLabel}>Mock Response:</Text>
                        <TextInput
                          style={styles.mockResponseInput}
                          value={currentMockResponses[toolName] || ''}
                          onChangeText={(text) => {
                            setCurrentMockResponses((prev) => ({
                              ...prev,
                              [toolName]: text,
                            }))
                          }}
                          multiline
                          placeholder={`Enter mock response for ${toolName}...`}
                          placeholderTextColor={theme.colors.textSecondary}
                          keyboardType="ascii-capable"
                        />
                      </View>
                    )
                  })
                } catch (e: any) {
                  return (
                    <Text style={styles.errorText}>
                      {`Invalid JSON (${e.message}) - fix tools JSON below to see mock response inputs`}
                    </Text>
                  )
                }
              })()}
            </View>
          )}

          {(activeTab === 'custom' && error) ? <Text style={styles.errorText}>{error}</Text> : null}
          {(activeTab === 'mcp' && mcpError) ? <Text style={styles.errorText}>{mcpError}</Text> : null}

          <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
            <Text style={styles.resetButtonText}>
              {activeTab === 'custom' ? 'Reset to Defaults' : 'Reset MCP Config'}
            </Text>
          </TouchableOpacity>

          <View style={styles.bottomPadding} />
        </ScrollView>
      </SafeAreaView>
    </Modal>
  )
}
