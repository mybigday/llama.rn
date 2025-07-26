/* eslint-disable jsx-a11y/accessible-emoji */
import * as React from 'react'
import { View, Text, StyleSheet } from 'react-native'
import {
  GestureHandlerRootView,
  TouchableOpacity,
} from 'react-native-gesture-handler'
import { enableScreens } from 'react-native-screens'
import { NavigationContainer } from '@react-navigation/native'
import { createNativeStackNavigator } from '@react-navigation/native-stack'
import { toggleNativeLog, addNativeLogListener } from '../../src'
import SimpleChatScreen from './screens/SimpleChatScreen'
import MultimodalScreen from './screens/MultimodalScreen'
import TTSScreen from './screens/TTSScreen'
import ToolCallsScreen from './screens/ToolCallsScreen'
import ModalInfoScreen from './screens/ModalInfoScreen'

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

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  button: {
    margin: 10,
    padding: 10,
    backgroundColor: '#333',
    borderRadius: 5,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
  },
  headerButton: {
    marginRight: 15,
  },
  headerButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: '500',
  },
})

// Header button component
function HeaderButton({
  onPress,
  title,
}: {
  onPress: () => void
  title: string
}) {
  return (
    <TouchableOpacity style={styles.headerButton} onPress={onPress}>
      <Text style={styles.headerButtonText}>{title}</Text>
    </TouchableOpacity>
  )
}

function HomeScreen({ navigation }: { navigation: any }) {
  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('SimpleChat')}
      >
        <Text style={styles.buttonText}>💬 Simple Chat</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('Multimodal')}
      >
        <Text style={styles.buttonText}>👁️ Vision/Multimodal</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('ToolCalling')}
      >
        <Text style={styles.buttonText}>🛠️ Tool Calling</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('TTS')}
      >
        <Text style={styles.buttonText}>🔊 Text-to-Speech (OuteTTS)</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.button}
        onPress={() => navigation.navigate('ModalInfo')}
      >
        <Text style={styles.buttonText}>📊 Modal Info</Text>
      </TouchableOpacity>
    </View>
  )
}

const Stack = createNativeStackNavigator()

function App() {
  return (
    <GestureHandlerRootView>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen name="Home" component={HomeScreen} />
          <Stack.Screen
            name="SimpleChat"
            component={SimpleChatScreen}
            options={({ route }) => ({
              title: 'Simple Chat',
              headerRight: () => {
                const params = route.params as any
                if (params?.showContextSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showContextSettings}
                      title="Context"
                    />
                  )
                } else if (params?.showCompletionSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showCompletionSettings}
                      title="Chat"
                    />
                  )
                }
                return null
              },
            })}
          />
          <Stack.Screen
            name="Multimodal"
            component={MultimodalScreen}
            options={({ route }) => ({
              title: 'Vision Chat',
              headerRight: () => {
                const params = route.params as any
                if (params?.showContextSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showContextSettings}
                      title="Context"
                    />
                  )
                } else if (params?.showCompletionSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showCompletionSettings}
                      title="Chat"
                    />
                  )
                }
                return null
              },
            })}
          />
          <Stack.Screen
            name="ToolCalling"
            component={ToolCallsScreen}
            options={({ route }) => ({
              title: 'Tool Calling',
              headerRight: () => {
                const params = route.params as any
                if (params?.showContextSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showContextSettings}
                      title="Context"
                    />
                  )
                } else if (params?.showCompletionSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showCompletionSettings}
                      title="Chat"
                    />
                  )
                }
                return null
              },
            })}
          />
          <Stack.Screen
            name="TTS"
            component={TTSScreen}
            options={({ route }) => ({
              title: 'Text-to-Speech',
              headerRight: () => {
                const params = route.params as any
                if (params?.showContextSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showContextSettings}
                      title="Context"
                    />
                  )
                } else if (params?.showCompletionSettings) {
                  return (
                    <HeaderButton
                      onPress={params.showCompletionSettings}
                      title="Chat"
                    />
                  )
                }
                return null
              },
            })}
          />
          <Stack.Screen name="ModalInfo" component={ModalInfoScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  )
}

export default App
