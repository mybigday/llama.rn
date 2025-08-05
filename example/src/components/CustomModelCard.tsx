import React, { useState } from 'react'
import {
  View,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native'
import Icon from '@react-native-vector-icons/material-icons'
import ModelDownloadCard, { MtmdModelDownloadCard } from './ModelDownloadCard'
import { deleteCustomModel, type CustomModel } from '../utils/storage'
import { ModelDownloader } from '../services/ModelDownloader'

interface CustomModelCardProps {
  model: CustomModel
  onInitialize: (modelPath: string, mmprojPath?: string) => void
  onModelRemoved: () => void
  initializeButtonText?: string
  showRemoveButton?: boolean
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  removeButton: {
    position: 'absolute',
    top: 8,
    right: 12,
    backgroundColor: '#ef4444',
    borderRadius: 20,
    width: 24,
    height: 24,
    zIndex: 10,
    shadowColor: '#ef4444',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 3,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modelCard: {
    marginTop: 0,
    backgroundColor: 'transparent',
    borderRadius: 0,
    shadowColor: 'transparent',
    elevation: 0,
  },
})

export default function CustomModelCard({
  model,
  onInitialize,
  onModelRemoved,
  initializeButtonText = 'Initialize',
  showRemoveButton = true,
}: CustomModelCardProps) {
  const [isRemoving, setIsRemoving] = useState(false)

  const handleRemoveModel = async () => {
    if (isRemoving) return

    Alert.alert(
      'Remove Custom Model',
      `Are you sure you want to remove "${model.id}" from your custom models? This will also delete the downloaded files if they exist.`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              setIsRemoving(true)

              // Delete downloaded model files
              try {
                await ModelDownloader.deleteModel(model.filename)
                if (model.mmprojFilename) {
                  await ModelDownloader.deleteModel(model.mmprojFilename)
                }
              } catch (fileError) {
                console.warn('Failed to delete model files:', fileError)
                // Continue with removing from storage even if file deletion fails
              }

              // Remove from custom models storage
              await deleteCustomModel(model.id)

              Alert.alert('Success', `"${model.id}" has been removed from your custom models.`)
              onModelRemoved()
            } catch (error) {
              console.error('Error removing custom model:', error)
              Alert.alert(
                'Error',
                `Failed to remove "${model.id}". Please try again.`,
              )
            } finally {
              setIsRemoving(false)
            }
          },
        },
      ],
    )
  }

  const handleInitialize = (modelPath: string, mmprojPath?: string) => {
    onInitialize(modelPath, mmprojPath)
  }

  return (
    <View style={styles.container}>
      {showRemoveButton && (
        <TouchableOpacity
          style={styles.removeButton}
          onPress={handleRemoveModel}
          disabled={isRemoving}
        >
          <Icon
            name={isRemoving ? 'hourglass-empty' : 'close'}
            size={16}
            color="#ffffff"
          />
        </TouchableOpacity>
      )}

      <View style={styles.modelCard}>
        {model.mmprojFilename ? (
          <MtmdModelDownloadCard
            title={`${model.id} (${model.quantization})`}
            repo={model.repo}
            filename={model.filename}
            mmproj={model.mmprojFilename}
            size="Size unknown"
            initializeButtonText={initializeButtonText}
            onInitialize={(modelPath: string, mmprojPath: string) => {
              handleInitialize(modelPath, mmprojPath)
            }}
          />
        ) : (
          <ModelDownloadCard
            title={`${model.id} (${model.quantization})`}
            repo={model.repo}
            filename={model.filename}
            size="Size unknown"
            initializeButtonText={initializeButtonText}
            onInitialize={(modelPath: string) => {
              handleInitialize(modelPath)
            }}
          />
        )}
      </View>
    </View>
  )
}
