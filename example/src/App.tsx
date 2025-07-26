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
import ModelInfoScreen from './screens/ModelInfoScreen'
import { CommonStyles } from './styles/commonStyles'

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
  container: CommonStyles.centerContainer,
  button: CommonStyles.button,
  buttonText: CommonStyles.buttonText,
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
    paddingHorizontal: 32,
    lineHeight: 22,
  },
})

function HomeScreen({ navigation }: { navigation: any }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>🦙 llama.rn</Text>
      <Text style={styles.description}>
        Experience the power of large language models running locally on your
        mobile device. Explore different AI capabilities including chat, vision,
        tool calling, and text-to-speech.
      </Text>
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
        onPress={() => navigation.navigate('ModelInfo')}
      >
        <Text style={styles.buttonText}>📊 Model Info</Text>
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
            options={{
              title: 'Simple Chat',
            }}
          />
          <Stack.Screen
            name="Multimodal"
            component={MultimodalScreen}
            options={{
              title: 'Vision Chat',
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
            name="TTS"
            component={TTSScreen}
            options={{
              title: 'Text-to-Speech',
            }}
          />
          <Stack.Screen name="ModelInfo" component={ModelInfoScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  )
}

export default App
