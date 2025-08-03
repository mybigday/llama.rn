/* eslint-disable jsx-a11y/accessible-emoji */
import * as React from 'react'
import { View, Text, StyleSheet, ScrollView, Linking } from 'react-native'
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
import BenchScreen from './screens/BenchScreen'
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
  scrollContainer: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingVertical: 20,
  },
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
    paddingHorizontal: 32,
    lineHeight: 22,
  },
  repoLink: {
    marginTop: 8,
    marginBottom: 24,
    paddingHorizontal: 32,
    alignItems: 'center',
  },
  repoLinkText: {
    fontSize: 16,
    color: '#007AFF',
    textAlign: 'center',
  },
})

function HomeScreen({ navigation }: { navigation: any }) {
  const openRepo = () => {
    Linking.openURL('https://github.com/mybigday/llama.rn/tree/main/example')
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
      </View>
    </ScrollView>
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

export default App
