import RNBlobUtil from 'react-native-blob-util'
import { getModelDownloadUrl } from '../utils/constants'

export interface DownloadProgress {
  written: number
  total: number
  percentage: number
}

export class ModelDownloader {
  private activeDownloads = new Map<string, boolean>()

  async downloadModel(
    repo: string,
    filename: string,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<string> {
    const downloadKey = `${repo}/${filename}`

    if (this.activeDownloads.get(downloadKey)) {
      throw new Error('Download already in progress')
    }

    this.activeDownloads.set(downloadKey, true)

    try {
      const url = getModelDownloadUrl(repo, filename)
      const downloadDir = `${RNBlobUtil.fs.dirs.DocumentDir}/models`
      const filePath = `${downloadDir}/${filename}`

      // Create models directory if it doesn't exist
      if (!await RNBlobUtil.fs.exists(downloadDir)) {
        await RNBlobUtil.fs.mkdir(downloadDir)
      }

      // Check if file already exists
      const exists = await RNBlobUtil.fs.exists(filePath)
      if (exists) {
        this.activeDownloads.delete(downloadKey)
        return filePath
      }

      // Start download
      const config = RNBlobUtil.config({
        path: `${filePath}.tmp`,
        fileCache: true,
      })

      await config.fetch('GET', url).progress((written, total) => {
        if (onProgress && Number(total) > 0) {
          onProgress({
            written: Number(written),
            total: Number(total),
            percentage: Math.round((Number(written) / Number(total)) * 100),
          })
        }
      })

      // Move temp file to final location
      await RNBlobUtil.fs.mv(`${filePath}.tmp`, filePath)

      this.activeDownloads.delete(downloadKey)
      return filePath
    } catch (error) {
      this.activeDownloads.delete(downloadKey)

      // Clean up temp file if it exists
      try {
        await RNBlobUtil.fs.unlink(
          `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}.tmp`,
        )
      } catch {
        // Ignore cleanup errors
      }

      throw error
    }
  }

  static async isModelDownloaded(filename: string): Promise<boolean> {
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    return RNBlobUtil.fs.exists(filePath)
  }

  static async getModelPath(filename: string): Promise<string | null> {
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    const exists = await RNBlobUtil.fs.exists(filePath)
    return exists ? filePath : null
  }

  static async deleteModel(filename: string): Promise<void> {
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    const exists = await RNBlobUtil.fs.exists(filePath)
    if (exists) {
      await RNBlobUtil.fs.unlink(filePath)
    }
  }
}
