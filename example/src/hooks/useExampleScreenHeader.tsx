import React, { useLayoutEffect, type ComponentProps } from 'react'
import { View } from 'react-native'
import { HeaderButton } from '../components/HeaderButton'

interface HeaderAction {
  key?: string
  iconName: ComponentProps<typeof HeaderButton>['iconName']
  onPress: () => void
}

interface UseExampleScreenHeaderOptions {
  navigation: any
  isModelReady: boolean
  readyActions?: HeaderAction[]
  setupActions?: HeaderAction[]
  renderReadyExtras?: () => React.ReactNode
  renderSetupExtras?: () => React.ReactNode
}

const HeaderActionsRow = ({
  actions,
  renderExtras,
}: {
  actions: HeaderAction[]
  renderExtras?: () => React.ReactNode
}) => (
  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
    {actions.map((action) => (
      <HeaderButton
        key={action.key || `${action.iconName}-${action.onPress.toString()}`}
        iconName={action.iconName}
        onPress={action.onPress}
      />
    ))}
    {renderExtras?.()}
  </View>
)

export function useExampleScreenHeader({
  navigation,
  isModelReady,
  readyActions = [],
  setupActions = [],
  renderReadyExtras,
  renderSetupExtras,
}: UseExampleScreenHeaderOptions) {
  useLayoutEffect(() => {
    const actions = isModelReady ? readyActions : setupActions
    const renderExtras = isModelReady ? renderReadyExtras : renderSetupExtras

    navigation.setOptions({
      headerRight: () => (
        <HeaderActionsRow actions={actions} renderExtras={renderExtras} />
      ),
    })
  }, [
    isModelReady,
    navigation,
    readyActions,
    renderReadyExtras,
    renderSetupExtras,
    setupActions,
  ])
}
