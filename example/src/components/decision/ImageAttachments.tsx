import React from 'react'
import {
  View,
  Text,
  Image,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native'
import Icon from '@react-native-vector-icons/material-design-icons'
import { pick, keepLocalCopy } from '@react-native-documents/picker'
import { useTheme } from '../../contexts/ThemeContext'

const MAX_IMAGES = 8 // what a decision request takes

// Images as local file URIs (file://...), which is also what decide() accepts once the scheme is dropped
interface ImageAttachmentsProps {
  images: string[]
  onChange: (images: string[]) => void
  // why images cannot be added, e.g. no projector loaded
  disabledReason?: string
}

// a file URI is percent-encoded, a path with spaces or non-ASCII characters is not
export const toImagePath = (uri: string) =>
  decodeURIComponent(uri.replace(/^file:\/\//, ''))

export function ImageAttachments({
  images,
  onChange,
  disabledReason,
}: ImageAttachmentsProps) {
  const { theme } = useTheme()
  const styles = createStyles(theme.colors)

  const addImage = async () => {
    try {
      const [file] = await pick({ type: ['image/*'] })
      if (!file?.uri) return
      const [copy] = await keepLocalCopy({
        files: [{ uri: file.uri, fileName: file.name ?? 'image.jpg' }],
        destination: 'documentDirectory',
      })
      if (copy.status !== 'success') {
        throw new Error(copy.copyError || 'could not copy the image')
      }
      onChange([...images, copy.localUri])
    } catch (error: any) {
      if (!String(error?.message).includes('canceled')) {
        Alert.alert('Error', `Failed to add the image: ${error.message}`)
      }
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>{`Images (${images.length})`}</Text>
      {disabledReason ? (
        <Text style={styles.hint}>{disabledReason}</Text>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View style={styles.row}>
            {images.map((uri, i) => (
              <View key={uri} style={styles.thumbWrapper}>
                <Image source={{ uri }} style={styles.thumb} />
                <TouchableOpacity
                  style={styles.removeButton}
                  onPress={() => onChange(images.filter((_, j) => j !== i))}
                  hitSlop={8}
                >
                  <Icon name="close" size={14} color={theme.colors.white} />
                </TouchableOpacity>
              </View>
            ))}
            {images.length < MAX_IMAGES && (
              <TouchableOpacity style={styles.addTile} onPress={addImage}>
                <Icon
                  name="image-plus"
                  size={26}
                  color={theme.colors.primary}
                />
                <Text style={styles.addText}>Add image</Text>
              </TouchableOpacity>
            )}
          </View>
        </ScrollView>
      )}
    </View>
  )
}

function createStyles(colors: ReturnType<typeof useTheme>['theme']['colors']) {
  return StyleSheet.create({
    container: { marginBottom: 16 },
    title: {
      fontSize: 15,
      fontWeight: '700',
      color: colors.text,
      marginBottom: 8,
    },
    hint: { fontSize: 13, color: colors.textSecondary },
    row: { flexDirection: 'row', gap: 8 },
    thumbWrapper: { width: 84, height: 84 },
    thumb: {
      width: 84,
      height: 84,
      borderRadius: 10,
      backgroundColor: colors.card,
    },
    removeButton: {
      position: 'absolute',
      top: 4,
      right: 4,
      width: 22,
      height: 22,
      borderRadius: 11,
      backgroundColor: 'rgba(0,0,0,0.6)',
      alignItems: 'center',
      justifyContent: 'center',
    },
    addTile: {
      width: 84,
      height: 84,
      borderRadius: 10,
      borderWidth: 1,
      borderStyle: 'dashed',
      borderColor: colors.primary,
      alignItems: 'center',
      justifyContent: 'center',
      gap: 2,
    },
    addText: { fontSize: 11, color: colors.primary, fontWeight: '600' },
  })
}
