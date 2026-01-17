/* eslint-disable jsx-a11y/accessible-emoji */
import * as React from 'react'
import { View, Text, StyleSheet, ScrollView, Linking, Alert, Modal, TouchableOpacity as RNTouchableOpacity } from 'react-native'
import {
  GestureHandlerRootView,
  TouchableOpacity,
} from 'react-native-gesture-handler'
import { enableScreens } from 'react-native-screens'
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import { toggleNativeLog, addNativeLogListener, BuildInfo, getBackendDevicesInfo } from '../../src'
import SimpleChatScreen from './screens/SimpleChatScreen'
import MultimodalScreen from './screens/MultimodalScreen'
import TTSScreen from './screens/TTSScreen'
import ToolCallsScreen from './screens/ToolCallsScreen'
import ModelInfoScreen from './screens/ModelInfoScreen'
import BenchScreen from './screens/BenchScreen'
import TextCompletionScreen from './screens/TextCompletionScreen'
import ParallelDecodingScreen from './screens/ParallelDecodingScreen'
import EmbeddingScreen from './screens/EmbeddingScreen'
import StressTestScreen from './screens/StressTestScreen'
import { ThemeProvider, useTheme } from './contexts/ThemeContext'
import { createThemedStyles } from './styles/commonStyles'
import { Menu } from './components/Menu'

// Catch logs from llama.cpp
toggleNativeLog(true)
addNativeLogListener((level, text) => {
  console.log(['[rnllama]', level ? `[${level}]` : '', text].filter(Boolean).join(' '))
})

enableScreens()

function HomeScreenComponent({ navigation }: { navigation: any }) {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)
  const [deviceInfo, setDeviceInfo] = React.useState<any[]>([])
  const [showDeviceInfo, setShowDeviceInfo] = React.useState(false)

  const styles = StyleSheet.create({
    container: themedStyles.centerContainer,
    scrollContainer: {
      flexGrow: 1,
      justifyContent: 'center',
      paddingVertical: 20,
    },
    button: {
      margin: 10,
      padding: 10,
      backgroundColor: '#333',
      borderRadius: 5,
    },
    buttonText: {
      color: theme.colors.white,
      fontSize: 16,
      fontWeight: '600',
      textAlign: 'center',
    },
    title: {
      fontSize: 28,
      fontWeight: 'bold',
      color: theme.colors.text,
      marginBottom: 10,
      textAlign: 'center',
    },
    description: {
      fontSize: 16,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      paddingHorizontal: 32,
      lineHeight: 22,
      marginTop: 4,
    },
    repoLink: {
      marginTop: 8,
      marginBottom: 24,
      paddingHorizontal: 32,
      alignItems: 'center',
    },
    repoLinkText: {
      fontSize: 16,
      color: theme.colors.primary,
      textAlign: 'center',
    },
    modalContainer: {
      flex: 1,
      backgroundColor: 'rgba(0, 0, 0, 0.5)',
      justifyContent: 'center',
      alignItems: 'center',
    },
    modalContent: {
      backgroundColor: theme.colors.background,
      borderRadius: 12,
      padding: 20,
      width: '90%',
      maxHeight: '80%',
    },
    modalHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 16,
      paddingBottom: 12,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    modalTitle: {
      fontSize: 18,
      fontWeight: 'bold',
      color: theme.colors.text,
    },
    closeButton: {
      padding: 4,
    },
    closeButtonText: {
      fontSize: 18,
      color: theme.colors.primary,
      fontWeight: '600',
    },
    deviceCard: {
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      padding: 12,
      marginBottom: 12,
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.primary,
    },
    deviceCardHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: 8,
    },
    deviceName: {
      fontSize: 14,
      fontWeight: '600',
      color: theme.colors.text,
      flex: 1,
    },
    deviceBadge: {
      paddingHorizontal: 8,
      paddingVertical: 4,
      borderRadius: 4,
      backgroundColor: theme.colors.primary,
    },
    deviceBadgeText: {
      fontSize: 12,
      fontWeight: '600',
      color: theme.colors.white,
    },
    deviceDetail: {
      fontSize: 12,
      color: theme.colors.textSecondary,
      marginTop: 4,
    },
  })

  const openRepo = () => {
    Linking.openURL('https://github.com/mybigday/llama.rn/tree/main/example')
  }

  const openLlamaCppRepo = () => {
    Linking.openURL(
      `https://github.com/ggml-org/llama.cpp/releases/b${BuildInfo.number}`,
    )
  }

  const loadDeviceInfo = async () => {
    try {
      const devices = await getBackendDevicesInfo()
      console.log('Backend Devices Info:')
      console.log(JSON.stringify(devices, null, 2))
      setDeviceInfo(devices)
    } catch (error) {
      console.error('Error getting device info:', error)
      Alert.alert('Error', `${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const toggleDeviceInfo = async () => {
    if (!showDeviceInfo && deviceInfo.length === 0) {
      // Load device info if not already loaded
      await loadDeviceInfo()
    }
    setShowDeviceInfo(!showDeviceInfo)
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`
  }

  return (
    <ScrollView contentContainerStyle={styles.scrollContainer}>
      <View style={styles.container}>
        <Text style={styles.title}>🦙 llama.rn</Text>
        <Text style={styles.description}>
          Experience the power of large language models running locally on your
          mobile device. Explore different AI capabilities including chat,
          vision, tool calling, and text-to-speech.
        </Text>
        <View style={styles.repoLink}>
          <TouchableOpacity onPress={openRepo}>
            <Text style={styles.repoLinkText}>
              📂 https://github.com/mybigday/llama.rn/tree/main/example
            </Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('SimpleChat')}
        >
          <Text style={styles.buttonText}>💬 Simple Chat</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('TextCompletion')}
        >
          <Text style={styles.buttonText}>✏️ Text Completion</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ParallelDecoding')}
        >
          <Text style={styles.buttonText}>⚡ Parallel Decoding</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('Multimodal')}
        >
          <Text style={styles.buttonText}>👁️ Multimodal</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ToolCalling')}
        >
          <Text style={styles.buttonText}>🛠️ Tool Calling & MCP</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('Embeddings')}
        >
          <Text style={styles.buttonText}>🔍 Vector Search (in-memory) & Rerank</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('TTS')}
        >
          <Text style={styles.buttonText}>🔊 Text-to-Speech (OuteTTS)</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ModelInfo')}
        >
          <Text style={styles.buttonText}>📊 Model Info</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('Bench')}
        >
          <Text style={styles.buttonText}>🏋️ Bench</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('StressTest')}
        >
          <Text style={styles.buttonText}>🔥 Stress Test</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={toggleDeviceInfo}
        >
          <Text style={styles.buttonText}>🖥️ Device Info</Text>
        </TouchableOpacity>
        <View style={styles.repoLink}>
          <TouchableOpacity onPress={openLlamaCppRepo}>
            <Text style={styles.repoLinkText}>
              {`llama.cpp b${BuildInfo.number}`}
            </Text>
          </TouchableOpacity>
        </View>

      </View>

      {/* Device Info Modal */}
      <Modal
        visible={showDeviceInfo}
        transparent
        animationType="fade"
        onRequestClose={() => setShowDeviceInfo(false)}
      >
        <View style={styles.modalContainer}>
          <RNTouchableOpacity
            style={StyleSheet.absoluteFill}
            activeOpacity={1}
            onPress={() => setShowDeviceInfo(false)}
          />
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>🖥️ Device Information</Text>
              <RNTouchableOpacity
                style={styles.closeButton}
                onPress={() => setShowDeviceInfo(false)}
              >
                <Text style={styles.closeButtonText}>✕</Text>
              </RNTouchableOpacity>
            </View>
            <ScrollView showsVerticalScrollIndicator>
              {deviceInfo.map((device, index) => (
                <View key={index} style={styles.deviceCard}>
                  <View style={styles.deviceCardHeader}>
                    <Text style={styles.deviceName}>{device.deviceName}</Text>
                    <View style={styles.deviceBadge}>
                      <Text style={styles.deviceBadgeText}>{device.backend}</Text>
                    </View>
                  </View>
                  <Text style={styles.deviceDetail}>
                    {`Type: ${device.type.toUpperCase()}`}
                  </Text>
                  <Text style={styles.deviceDetail}>
                    {`Memory: ${formatBytes(device.maxMemorySize)}`}
                  </Text>
                  {device.metadata && Object.keys(device.metadata).length > 0 && (
                    <Text style={styles.deviceDetail}>
                      {`Metadata: ${Object.entries(device.metadata)
                        .filter(([_, v]) => v === true)
                        .map(([k]) => k)
                        .join(', ')}`}
                    </Text>
                  )}
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScrollView>
  )
}

const Stack = createNativeStackNavigator()

function AppContent() {
  const { theme, setThemeMode } = useTheme()

  // Create a wrapper component that has access to theme context
  const HomeScreen = (props: any) => <HomeScreenComponent {...props} />

  const navigationTheme = theme.dark ? DarkTheme : DefaultTheme

  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: theme.colors.background }}>
      <NavigationContainer theme={navigationTheme}>
        <Stack.Navigator
          screenOptions={{
            headerStyle: {
              backgroundColor: theme.colors.surface,
            },
            headerTitleStyle: {
              color: theme.colors.text,
            },
            headerTintColor: theme.colors.primary,
            headerRight: () => (
              <Menu
                icon="theme-light-dark"
                actions={[
                  {
                    id: 'system',
                    title: '📱 System',
                    onPress: () => setThemeMode('system'),
                  },
                  {
                    id: 'light',
                    title: '☀️ Light',
                    onPress: () => setThemeMode('light'),
                  },
                  {
                    id: 'dark',
                    title: '🌙 Dark',
                    onPress: () => setThemeMode('dark'),
                  },
                ]}
              />
            ),
          }}
        >
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen
            name="SimpleChat"
            component={SimpleChatScreen}
            options={{
              title: 'Simple Chat',
            }}
          />
          <Stack.Screen
            name="TextCompletion"
            component={TextCompletionScreen}
            options={{
              title: 'Text Completion',
            }}
          />
          <Stack.Screen
            name="ParallelDecoding"
            component={ParallelDecodingScreen}
            options={{
              title: 'Parallel Decoding',
            }}
          />
          <Stack.Screen
            name="Multimodal"
            component={MultimodalScreen}
            options={{
              title: 'Multimodal Chat',
            }}
          />
          <Stack.Screen
            name="ToolCalling"
            component={ToolCallsScreen}
            options={{
              title: 'Tool Calling & MCP',
            }}
          />
          <Stack.Screen
            name="Embeddings"
            component={EmbeddingScreen}
            options={{
              title: 'Vector Search (in-memory) & Rerank',
            }}
          />
          <Stack.Screen
            name="TTS"
            component={TTSScreen}
            options={{
              title: 'Text-to-Speech',
            }}
          />
          <Stack.Screen name="ModelInfo" component={ModelInfoScreen} />
          <Stack.Screen name="Bench" component={BenchScreen} />
          <Stack.Screen
            name="StressTest"
            component={StressTestScreen}
            options={{
              title: 'Stress Test',
            }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  )
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  )
}

export default App
