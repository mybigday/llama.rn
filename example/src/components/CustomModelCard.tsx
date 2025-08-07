import React, { useState, useEffect } from 'react'
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
  const [modelSize, setModelSize] = useState<string>('Size unknown')
  const [mmprojSize, setMmprojSize] = useState<string>('Size unknown')
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  useEffect(() => {
    const calculateSizes = async () => {
      try {
        // Check if model is downloaded first
        const isDownloaded = await ModelDownloader.isModelDownloaded(model.filename)
        
        if (isDownloaded) {
          // Calculate model size for downloaded files
          const modelSizeFormatted = await ModelDownloader.getModelSizeFormatted(model.filename)
          if (modelSizeFormatted) {
            setModelSize(modelSizeFormatted)
          }

          // Calculate mmproj size if exists
          if (model.mmprojFilename) {
            const mmprojSizeFormatted = await ModelDownloader.getModelSizeFormatted(model.mmprojFilename)
            if (mmprojSizeFormatted) {
              setMmprojSize(mmprojSizeFormatted)
            }
          }
        } else {
          // For undownloaded models, try to get remote size first
          try {
            const remoteSizeFormatted = await ModelDownloader.getModelSizeFromRemoteFormatted(model.repo, model.filename)
            if (remoteSizeFormatted) {
              const splitInfo = await ModelDownloader.getSplitFileInfo(model.filename)
              if (splitInfo) {
                setModelSize(`${remoteSizeFormatted} (${splitInfo.totalParts} parts)`)
              } else {
                setModelSize(remoteSizeFormatted)
              }
            } else {
              // Fallback to split info only
              const splitInfo = await ModelDownloader.getSplitFileInfo(model.filename)
              if (splitInfo) {
                setModelSize(`Split model (${splitInfo.totalParts} parts)`)
              } else {
                setModelSize('Size unknown')
              }
            }
          } catch {
            // If remote size fails, fallback to split info
            const splitInfo = await ModelDownloader.getSplitFileInfo(model.filename)
            if (splitInfo) {
              setModelSize(`Split model (${splitInfo.totalParts} parts)`)
            } else {
              setModelSize('Size unknown')
            }
          }
          
          // Same for mmproj if exists
          if (model.mmprojFilename) {
            try {
              const mmprojRemoteSizeFormatted = await ModelDownloader.getModelSizeFromRemoteFormatted(model.repo, model.mmprojFilename)
              if (mmprojRemoteSizeFormatted) {
                const mmprojSplitInfo = await ModelDownloader.getSplitFileInfo(model.mmprojFilename)
                if (mmprojSplitInfo) {
                  setMmprojSize(`${mmprojRemoteSizeFormatted} (${mmprojSplitInfo.totalParts} parts)`)
                } else {
                  setMmprojSize(mmprojRemoteSizeFormatted)
                }
              } else {
                const mmprojSplitInfo = await ModelDownloader.getSplitFileInfo(model.mmprojFilename)
                if (mmprojSplitInfo) {
                  setMmprojSize(`Split file (${mmprojSplitInfo.totalParts} parts)`)
                } else {
                  setMmprojSize('Size unknown')
                }
              }
            } catch {
              const mmprojSplitInfo = await ModelDownloader.getSplitFileInfo(model.mmprojFilename)
              if (mmprojSplitInfo) {
                setMmprojSize(`Split file (${mmprojSplitInfo.totalParts} parts)`)
              } else {
                setMmprojSize('Size unknown')
              }
            }
          }
        }
      } catch (error) {
        console.warn('Failed to calculate model sizes:', error)
      }
    }

    calculateSizes()
  }, [model.filename, model.mmprojFilename, model.repo, refreshTrigger])

  // Function to refresh sizes (can be called after download completion)
  const refreshSizes = () => {
    setRefreshTrigger(prev => prev + 1)
  }

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
            size={`Model: ${modelSize} + MMProj: ${mmprojSize}`}
            initializeButtonText={initializeButtonText}
            onInitialize={(modelPath: string, mmprojPath: string) => {
              handleInitialize(modelPath, mmprojPath)
            }}
            onDownloaded={() => {
              refreshSizes()
            }}
          />
        ) : (
          <ModelDownloadCard
            title={`${model.id} (${model.quantization})`}
            repo={model.repo}
            filename={model.filename}
            size={modelSize}
            initializeButtonText={initializeButtonText}
            onInitialize={(modelPath: string) => {
              handleInitialize(modelPath)
            }}
            onDownloaded={() => {
              refreshSizes()
            }}
          />
        )}
      </View>
    </View>
  )
}
