import React, { useContext } from 'react'
import type { ReactNode } from 'react'
import { View, Text } from 'react-native'
import { ThemeContext, UserContext } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'

export const Bubble = ({
  child,
  message,
  nextMessageInGroup,
}: {
  child: ReactNode
  message: MessageType.Any
  nextMessageInGroup: boolean
}) => {
  const theme = useContext(ThemeContext)
  const user = useContext(UserContext)
  const currentUserIsAuthor = user?.id === message.author.id
  const roundBorder = nextMessageInGroup ? theme.borders.messageBorderRadius : 0
  return (
    <View
      style={{
        backgroundColor:
          !currentUserIsAuthor || message.type === 'image'
            ? theme.colors.secondary
            : theme.colors.primary,
        borderBottomLeftRadius: currentUserIsAuthor ? roundBorder : 0,
        borderBottomRightRadius: currentUserIsAuthor
          ? roundBorder
          : theme.borders.messageBorderRadius,
        borderColor: 'transparent',
        borderRadius: theme.borders.messageBorderRadius,
        overflow: 'hidden',
      }}
    >
      {child}
      {message.metadata?.timings && (
        <Text
          style={{
            textAlign: 'right',
            color: '#ccc',
            paddingRight: 12,
            paddingBottom: 12,
            marginTop: -8,
            fontSize: 10,
          }}
        >
          {message.metadata.timings}
        </Text>
      )}
    </View>
  )
}
