import React, { useState, useEffect, useRef, useMemo } from 'react'
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native'
import { AudioContext } from 'react-native-audio-api'

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 16,
    marginVertical: 12,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  playerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  playButton: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#007AFF',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 4,
  },
  playButtonPressed: {
    backgroundColor: '#0056CC',
    transform: [{ scale: 0.95 }],
  },
  buttonText: {
    fontSize: 20,
    color: '#FFFFFF',
  },
  timeContainer: {
    flex: 1,
    alignItems: 'flex-end',
    marginLeft: 16,
  },
  timeText: {
    fontSize: 14,
    color: '#CCCCCC',
    fontWeight: '500',
    fontVariant: ['tabular-nums'],
  },
  progressContainer: {
    height: 4,
    backgroundColor: '#333333',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 10,
    color: '#888888',
    textAlign: 'center',
    marginTop: 4,
    fontVariant: ['tabular-nums'],
  },
})

export const AudioPlayer = ({
  audio,
  sr,
}: {
  audio: Float32Array
  sr: number
}) => {
  const ctxRef = useRef<AudioContext | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [isPressed, setIsPressed] = useState(false)
  const duration = useMemo(() => audio.length / sr, [audio, sr])
  const progressPercentage = useMemo(
    () => (progress / duration) * 100,
    [progress, duration],
  )

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

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <View style={styles.container}>
      <View style={styles.playerRow}>
        <TouchableOpacity
          style={[styles.playButton, isPressed && styles.playButtonPressed]}
          onPress={() => setIsPlaying((v) => !v)}
          onPressIn={() => setIsPressed(true)}
          onPressOut={() => setIsPressed(false)}
          activeOpacity={0.8}
        >
          <Text style={styles.buttonText}>{isPlaying ? '⏹' : '▶'}</Text>
        </TouchableOpacity>
        <View style={styles.timeContainer}>
          <Text style={styles.timeText}>
            {formatTime(progress)}
            {' / '}
            {formatTime(duration)}
          </Text>
        </View>
      </View>
      <View style={styles.progressContainer}>
        <View
          style={[styles.progressBar, { width: `${progressPercentage}%` }]}
        />
      </View>
      <Text style={styles.progressText}>
        {isPlaying ? 'Playing...' : 'Ready to play'}
      </Text>
    </View>
  )
}
