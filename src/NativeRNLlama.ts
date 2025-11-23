import type { TurboModule } from 'react-native'
import { TurboModuleRegistry } from 'react-native'

export interface Spec extends TurboModule {
  install(): Promise<boolean>
}

export default TurboModuleRegistry.get<Spec>('RNLlama') as Spec
