/* eslint-disable jsx-a11y/accessible-emoji */
import * as React from 'react'
import { View, Text, StyleSheet, ScrollView, Linking } from 'react-native'
import {
  GestureHandlerRootView,
  TouchableOpacity,
} from 'react-native-gesture-handler'
import { enableScreens } from 'react-native-screens'
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import { toggleNativeLog, addNativeLogListener, BuildInfo } from '../../src'
import SimpleChatScreen from './screens/SimpleChatScreen'
import MultimodalScreen from './screens/MultimodalScreen'
import TTSScreen from './screens/TTSScreen'
import ToolCallsScreen from './screens/ToolCallsScreen'
import ModelInfoScreen from './screens/ModelInfoScreen'
import BenchScreen from './screens/BenchScreen'
import TextCompletionScreen from './screens/TextCompletionScreen'
import EmbeddingScreen from './screens/EmbeddingScreen'
import { ThemeProvider, useTheme } from './contexts/ThemeContext'
import { createThemedStyles } from './styles/commonStyles'

// Example: Catch logs from llama.cpp
toggleNativeLog(true)
addNativeLogListener((level, text) => {
  // eslint-disable-next-line prefer-const
  let log = (t: string) => t // noop
  // Uncomment to test:
  // ;({log} = console)
  log(['[rnllama]', level ? `[${level}]` : '', text].filter(Boolean).join(' '))
})

enableScreens()

function HomeScreenComponent({ navigation }: { navigation: any }) {
  const { theme, themeMode, setThemeMode } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

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
    themeToggle: {
      alignItems: 'center',
      marginVertical: 8,
      flexDirection: 'row',
    },
    themeToggleButton: {
      backgroundColor: theme.colors.card,
      borderRadius: 8,
      paddingHorizontal: 20,
      paddingVertical: 12,
      marginVertical: 4,
      marginHorizontal: 4,
      borderWidth: 1,
      borderColor: theme.colors.border,
      minWidth: 120,
    },
    themeToggleButtonText: {
      color: theme.colors.text,
      fontSize: 16,
      textAlign: 'center',
      fontWeight: '600',
    },
    activeThemeButton: {
      backgroundColor: theme.colors.primary,
      borderColor: theme.colors.primary,
    },
    activeThemeButtonText: {
      color: theme.colors.white,
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
          onPress={() => navigation.navigate('Multimodal')}
        >
          <Text style={styles.buttonText}>👁️ Multimodal</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('ToolCalling')}
        >
          <Text style={styles.buttonText}>🛠️ Tool Calling</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate('Embeddings')}
        >
          <Text style={styles.buttonText}>🔍 Vector Search (in-memory)</Text>
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
        <View style={styles.repoLink}>
          <TouchableOpacity onPress={openLlamaCppRepo}>
            <Text style={styles.repoLinkText}>
              {`llama.cpp b${BuildInfo.number}`}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.themeToggle}>
          <TouchableOpacity
            style={[
              styles.themeToggleButton,
              themeMode === 'system' && styles.activeThemeButton,
            ]}
            onPress={() => setThemeMode('system')}
          >
            <Text
              style={[
                styles.themeToggleButtonText,
                themeMode === 'system' && styles.activeThemeButtonText,
              ]}
            >
              📱 System
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.themeToggleButton,
              themeMode === 'light' && styles.activeThemeButton,
            ]}
            onPress={() => setThemeMode('light')}
          >
            <Text
              style={[
                styles.themeToggleButtonText,
                themeMode === 'light' && styles.activeThemeButtonText,
              ]}
            >
              ☀️ Light
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.themeToggleButton,
              themeMode === 'dark' && styles.activeThemeButton,
            ]}
            onPress={() => setThemeMode('dark')}
          >
            <Text
              style={[
                styles.themeToggleButtonText,
                themeMode === 'dark' && styles.activeThemeButtonText,
              ]}
            >
              🌙 Dark
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </ScrollView>
  )
}

const Stack = createNativeStackNavigator()

function AppContent() {
  const { theme } = useTheme()

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
              title: 'Tool Calling',
            }}
          />
          <Stack.Screen
            name="Embeddings"
            component={EmbeddingScreen}
            options={{
              title: 'Vector Search (in-memory)',
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
