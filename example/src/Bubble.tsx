import React, { useContext, useState, useEffect, useRef, useMemo } from 'react'
import type { ReactNode } from 'react'
import { View, Text, TouchableOpacity, Image, StyleSheet, Button } from 'react-native'
import Clipboard from '@react-native-clipboard/clipboard'
import { ThemeContext, UserContext } from '@flyerhq/react-native-chat-ui'
import type { MessageType } from '@flyerhq/react-native-chat-ui'
import { AudioContext } from 'react-native-audio-api'

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginVertical: 12,
  },
  text: {
    fontSize: 12,
    color: '#ccc',
    marginLeft: 12,
  },
  button: {
    padding: 5,
    borderRadius: 5,
    backgroundColor: '#ccc',
    justifyContent: 'center',
    alignItems: 'center',
  },
})

const AudioPlayer = ({ audio, sr }: { audio: Float32Array; sr: number }) => {
  const ctxRef = useRef<AudioContext | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const duration = useMemo(() => audio.length / sr, [audio, sr])

  useEffect(() => {
    if (isPlaying) {
      setProgress(0)
      const interval = setInterval(() => {
        setProgress(ctxRef.current?.currentTime ?? 0)
      }, 10)
      ctxRef.current ??= new AudioContext()
      const audioBuffer = ctxRef.current.createBuffer(1, audio.length, sr)
      audioBuffer.copyToChannel(new Float32Array(audio), 0)
      const source = ctxRef.current.createBufferSource()
      source.buffer = audioBuffer
      source.connect(ctxRef.current.destination)
      source.start()
      source.onended = () => {
        clearInterval(interval)
        setIsPlaying(false)
        setProgress(duration)
      }
      return () => clearInterval(interval)
    } else {
      ctxRef.current?.close()
      ctxRef.current = null
      return () => {}
    }
  }, [isPlaying, audio, sr, duration])

  return (
    <View style={styles.container}>
      <Button title={isPlaying ? '⏹️' : '▶️'} onPress={() => setIsPlaying((v) => !v)} />
      <Text style={styles.text}>
        {progress.toFixed(2)}
        {' '}
        /
        {' '}
        {duration.toFixed(2)}
        {' '}
        s
      </Text>
    </View>
  )
}

export const Bubble = ({
  child,
  message,
}: {
  child: ReactNode
  message: MessageType.Any
}) => {
  const theme = useContext(ThemeContext)
  const user = useContext(UserContext)
  const currentUserIsAuthor = user?.id === message.author.id
  const { copyable, timings } = message.metadata || {}

  const Container = copyable ? TouchableOpacity : View

  return (
    <Container
      style={{
        backgroundColor:
          !currentUserIsAuthor || message.type === 'image'
            ? theme.colors.secondary
            : theme.colors.primary,
        borderBottomLeftRadius: currentUserIsAuthor
          ? theme.borders.messageBorderRadius
          : 0,
        borderBottomRightRadius: currentUserIsAuthor
          ? 0
          : theme.borders.messageBorderRadius,
        borderColor: 'transparent',
        borderRadius: theme.borders.messageBorderRadius,
        overflow: 'hidden',
      }}
      onPress={() => {
        if (message.type !== 'text') return
        Clipboard.setString(message.text)
      }}
    >
      {child}
      {message?.metadata?.audio && (
        <AudioPlayer audio={message.metadata.audio} sr={message.metadata.sr} />
      )}
      {message?.metadata?.mediaPath && (
        <Image
          source={{ uri: message.metadata.mediaPath }}
          resizeMode="cover"
          style={{
            width: 100,
            height: 100,
            alignSelf: 'flex-end',
            marginRight: 12,
            marginBottom: 12,
          }}
        />
      )}
      {timings && (
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
          {timings}
        </Text>
      )}
    </Container>
  )
}
