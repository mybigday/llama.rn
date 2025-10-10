import { AppRegistry } from 'react-native'
import NativeCustomEvent from 'react-native/src/private/webapis/dom/events/CustomEvent'
import App from './src/App.tsx'
import { name as appName } from './app.json'

// Setup for `mcp-sdk-client-ssejs` package
window.CustomEvent = class CustomEvent extends NativeCustomEvent {
  constructor(type, eventInitDict = {}) {
    super(type, eventInitDict)
  }
}

AppRegistry.registerComponent(appName, () => App)
