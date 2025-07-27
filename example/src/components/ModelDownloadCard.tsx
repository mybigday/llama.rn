/* eslint-disable react/require-default-props */
import React, { useState, useEffect } from 'react'
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native'
import { ModelDownloader } from '../services/ModelDownloader'
import type { DownloadProgress } from '../services/ModelDownloader'

// Common interfaces and types
interface ModelFile {
  repo: string
  filename: string
  size?: string
  label?: string // For display purposes (e.g., "TTS model", "vocoder")
}

interface BaseModelDownloadCardProps {
  title: string
  size: string
  files: ModelFile[] // Array of files to download
  onInitialize?: (...paths: string[]) => void
  downloadButtonText?: string
  initializeButtonText?: string
}

interface ModelDownloadCardProps {
  title: string
  repo: string
  filename: string
  size: string
  onDownloaded?: (path: string) => void
  onInitialize?: (path: string) => void
  initializeButtonText?: string
}

interface TTSModelDownloadCardProps {
  title: string
  repo: string
  filename: string
  size: string
  vocoder: {
    repo: string
    filename: string
    size: string
  }
  onInitialize: (ttsPath: string, vocoderPath: string) => void
  initializeButtonText?: string
}

interface VLMModelDownloadCardProps {
  title: string
  repo: string
  filename: string
  mmproj: string
  size: string
  onInitialize: (modelPath: string, mmprojPath: string) => void
  initializeButtonText?: string
}

// Common styles
const styles = StyleSheet.create({
  card: {
    backgroundColor: 'white',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  header: {
    marginBottom: 8,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerColumn: {
    flexDirection: 'column',
    alignItems: 'flex-start',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    flexShrink: 1,
  },
  size: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  sizeColumn: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
    marginTop: 4,
  },
  description: {
    fontSize: 14,
    color: '#666',
    marginBottom: 16,
    lineHeight: 20,
  },
  progressContainer: {
    marginBottom: 16,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#E0E0E0',
    borderRadius: 2,
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  downloadButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    flex: 1,
  },
  downloadButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  downloadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
  },
  downloadingText: {
    marginLeft: 8,
    fontSize: 16,
    color: '#007AFF',
  },
  downloadedContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flex: 1,
  },
  downloadedIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  checkmark: {
    fontSize: 20,
    color: '#4CAF50',
    marginRight: 8,
  },
  downloadedText: {
    fontSize: 16,
    color: '#4CAF50',
    fontWeight: '500',
  },
  deleteButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
  },
  deleteButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  initializeButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    marginLeft: 8,
  },
  initializeButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  actionButtonsContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
})

// Common utility functions
const formatSize = (bytes: number): string => {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

// Base component with shared logic
function BaseModelDownloadCard({
  title,
  size,
  files,
  onInitialize,
  downloadButtonText = 'Download',
  initializeButtonText = 'Initialize',
}: BaseModelDownloadCardProps) {
  const [isDownloaded, setIsDownloaded] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [progress, setProgress] = useState<DownloadProgress | null>(null)
  const [filePaths, setFilePaths] = useState<string[]>([])
  const [downloadStatus, setDownloadStatus] = useState<string>('')
  const [useRowLayout, setUseRowLayout] = useState(true)

  const downloader = new ModelDownloader()

  const checkIfDownloaded = React.useCallback(async () => {
    try {
      const downloadStatuses = await Promise.all(
        files.map((file) => ModelDownloader.isModelDownloaded(file.filename)),
      )

      const allDownloaded = downloadStatuses.every((status) => status)
      setIsDownloaded(allDownloaded)

      if (allDownloaded) {
        const pathPromises = files.map((file) =>
          ModelDownloader.getModelPath(file.filename),
        )
        const paths = await Promise.all(pathPromises)
        // Filter out any null paths
        const validPaths = paths.filter((path): path is string => path !== null)
        if (validPaths.length === files.length) {
          setFilePaths(validPaths)
        }
      }
    } catch (error) {
      console.error('Error checking model status:', error)
    }
  }, [files])

  useEffect(() => {
    checkIfDownloaded()
  }, [checkIfDownloaded])

  const handleDownload = async () => {
    if (isDownloading) return

    try {
      setIsDownloading(true)
      setProgress({ written: 0, total: 0, percentage: 0 })

      const paths: string[] = []
      const progressWeight = 1 / files.length

      // ESLint disable for intentional sequential downloads
      /* eslint-disable no-await-in-loop */
      for (let i = 0; i < files.length; i += 1) {
        const file = files[i]
        if (file) {
          const statusText = file.label || `file ${i + 1}`
          setDownloadStatus(`Downloading ${statusText}...`)

          const path = await downloader.downloadModel(
            file.repo,
            file.filename,
            (prog) => {
              const baseProgress = i * progressWeight * 100
              const currentProgress = prog.percentage * progressWeight
              setProgress({
                ...prog,
                percentage: Math.round(baseProgress + currentProgress),
              })
            },
          )

          paths.push(path)
        }
      }
      /* eslint-enable no-await-in-loop */

      setFilePaths(paths)
      setIsDownloaded(true)
      setProgress(null)
      setDownloadStatus('')

      Alert.alert('Success', `${title} downloaded successfully!`)
    } catch (error: any) {
      Alert.alert(
        'Download Failed',
        error.message || 'Failed to download model(s)',
      )
      setProgress(null)
      setDownloadStatus('')
    } finally {
      setIsDownloading(false)
    }
  }

  const handleInitialize = async () => {
    if (!isDownloaded || filePaths.length !== files.length) {
      Alert.alert('Error', 'Model(s) not downloaded yet.')
      return
    }

    if (onInitialize) {
      onInitialize(...filePaths)
    } else {
      Alert.alert('Error', 'No initialization handler provided.')
    }
  }

  const handleDelete = async () => {
    const modelText = files.length > 1 ? 'Models' : 'Model'
    Alert.alert(
      `Delete ${modelText}`,
      `Are you sure you want to delete ${title}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await Promise.all(
                files.map((file) => ModelDownloader.deleteModel(file.filename)),
              )
              setIsDownloaded(false)
              setFilePaths([])
            } catch (error: any) {
              Alert.alert(
                'Error',
                `Failed to delete ${modelText.toLowerCase()}`,
              )
            }
          },
        },
      ],
    )
  }

  const repoDisplay =
    files.length === 1 && files[0] ? files[0].repo : `${files.length} files`

  const handleLayout = (event: any) => {
    const { width } = event.nativeEvent.layout
    // Switch to column layout if width is less than 300px (adjusted for Android)
    const shouldUseRow = width < 300
    if (shouldUseRow !== useRowLayout) {
      setUseRowLayout(shouldUseRow)
    }
  }

  return (
    <View style={styles.card} onLayout={handleLayout}>
      <View
        style={[
          styles.header,
          useRowLayout ? styles.headerRow : styles.headerColumn,
        ]}
      >
        <Text style={styles.title}>{title}</Text>
        <Text style={useRowLayout ? styles.size : styles.sizeColumn}>
          {size}
        </Text>
      </View>

      <Text style={styles.description}>{repoDisplay}</Text>

      {progress && (
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View
              style={[
                styles.progressFill,
                { width: `${progress.percentage}%` },
              ]}
            />
          </View>
          <Text style={styles.progressText}>
            {downloadStatus} 
            {' '}
            {`${progress.percentage}%`}
          </Text>
          {progress.total > 0 && (
            <Text style={styles.progressText}>
              (
              {formatSize(progress.written)}
              {' / '}
              {formatSize(progress.total)}
              )
            </Text>
          )}
        </View>
      )}

      <View style={styles.buttonContainer}>
        {!isDownloaded && !isDownloading && (
          <TouchableOpacity
            style={styles.downloadButton}
            onPress={handleDownload}
          >
            <Text style={styles.downloadButtonText}>{downloadButtonText}</Text>
          </TouchableOpacity>
        )}

        {isDownloading && (
          <View style={styles.downloadingContainer}>
            <ActivityIndicator size="small" color="#007AFF" />
            <Text style={styles.downloadingText}>Downloading...</Text>
          </View>
        )}

        {isDownloaded && !isDownloading && (
          <View style={styles.downloadedContainer}>
            <View style={styles.downloadedIndicator}>
              <Text style={styles.checkmark}>✓</Text>
              <Text style={styles.downloadedText}>Downloaded</Text>
            </View>
            <View style={styles.actionButtonsContainer}>
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={handleDelete}
              >
                <Text style={styles.deleteButtonText}>Delete</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.initializeButton}
                onPress={handleInitialize}
              >
                <Text style={styles.initializeButtonText}>
                  {initializeButtonText}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </View>
  )
}

// Simple single-model download card
function ModelDownloadCard({
  title,
  repo,
  filename,
  size,
  onDownloaded: _onDownloaded,
  onInitialize,
  initializeButtonText,
}: ModelDownloadCardProps) {
  const files: ModelFile[] = [{ repo, filename }]

  return (
    <BaseModelDownloadCard
      title={title}
      size={size}
      files={files}
      onInitialize={onInitialize}
      downloadButtonText="Download"
      initializeButtonText={initializeButtonText}
    />
  )
}

// TTS-specific download card that handles both TTS model and vocoder together
export function TTSModelDownloadCard({
  title,
  repo,
  filename,
  size,
  vocoder,
  onInitialize,
  initializeButtonText,
}: TTSModelDownloadCardProps) {
  const files: ModelFile[] = [
    { repo, filename, label: 'TTS model' },
    { repo: vocoder.repo, filename: vocoder.filename, label: 'vocoder' },
  ]

  return (
    <BaseModelDownloadCard
      title={title}
      size={size}
      files={files}
      onInitialize={onInitialize}
      downloadButtonText="Download Both Models"
      initializeButtonText={initializeButtonText}
    />
  )
}

// VLM-specific download card that handles both model and mmproj files
export function VLMModelDownloadCard({
  title,
  repo,
  filename,
  mmproj,
  size,
  onInitialize,
  initializeButtonText,
}: VLMModelDownloadCardProps) {
  const files: ModelFile[] = [
    { repo, filename, label: 'VLM model' },
    { repo, filename: mmproj, label: 'mmproj' },
  ]

  return (
    <BaseModelDownloadCard
      title={title}
      size={size}
      files={files}
      onInitialize={onInitialize}
      downloadButtonText="Download VLM & MMProj"
      initializeButtonText={initializeButtonText}
    />
  )
}

export default ModelDownloadCard
