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
      if (!(await RNBlobUtil.fs.exists(downloadDir))) {
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

      const response = await config
        .fetch('GET', url)
        .progress((written, total) => {
          if (onProgress && Number(total) > 0) {
            onProgress({
              written: Number(written),
              total: Number(total),
              percentage: Math.round((Number(written) / Number(total)) * 100),
            })
          }
        })

      // Check response status
      const statusCode = response.info().status
      if (statusCode !== 200) {
        throw new Error(
          ModelDownloader.getHttpErrorMessage(statusCode, repo, filename),
        )
      }

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

      // Re-throw with enhanced error message
      if (error && typeof error === 'object' && 'message' in error) {
        throw new Error(
          ModelDownloader.enhanceErrorMessage(error as Error, repo, filename),
        )
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

  /**
   * Get appropriate error message based on HTTP status code
   */
  private static getHttpErrorMessage(
    statusCode: number,
    repo: string,
    filename: string,
  ): string {
    switch (statusCode) {
      case 401:
        return `Access denied (401): The model "${repo}" requires authentication or login. You may need to accept the model license on HuggingFace before downloading.`
      case 403:
        return `Forbidden (403): Access to "${repo}/${filename}" is forbidden. The model may require special permissions or approval.`
      case 404:
        return `Not found (404): The file "${filename}" was not found in repository "${repo}". Please check the model ID and filename.`
      case 429:
        return `Rate limited (429): Too many download requests. Please wait a moment and try again.`
      case 500:
      case 502:
      case 503:
      case 504:
        return `Server error (${statusCode}): HuggingFace servers are experiencing issues. Please try again later.`
      default:
        return `Download failed with HTTP ${statusCode}: Unable to download "${filename}" from "${repo}".`
    }
  }

  /**
   * Enhance error messages with more context
   */
  private static enhanceErrorMessage(
    error: Error,
    repo: string,
    filename: string,
  ): string {
    const originalMessage = error.message.toLowerCase()

    // Network-related errors
    if (
      originalMessage.includes('network') ||
      originalMessage.includes('timeout')
    ) {
      return `Network error: Unable to download "${filename}" from "${repo}". Please check your internet connection and try again.`
    }

    // Permission/access errors
    if (
      originalMessage.includes('permission') ||
      originalMessage.includes('access')
    ) {
      return `Access error: Cannot access "${repo}/${filename}". The model may require login or license acceptance on HuggingFace.`
    }

    // Space/storage errors
    if (
      originalMessage.includes('space') ||
      originalMessage.includes('storage') ||
      originalMessage.includes('disk')
    ) {
      return `Storage error: Not enough space to download "${filename}". Please free up storage space and try again.`
    }

    // Generic enhancement
    return `Download failed: ${error.message}. Model: "${repo}/${filename}".`
  }
}
