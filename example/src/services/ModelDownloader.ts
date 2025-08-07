import RNBlobUtil from 'react-native-blob-util'
import { getModelDownloadUrl } from '../utils/constants'

export interface DownloadProgress {
  written: number
  total: number
  percentage: number
}

export class ModelDownloader {
  private activeDownloads = new Map<string, boolean>()

  /**
   * Detect if a filename follows the split GGUF pattern: xx-00001-of-00003.gguf
   * Returns split info if it's a split file, null otherwise
   */
  private static detectSplitFile(filename: string): {
    baseName: string
    currentPart: number
    totalParts: number
    pattern: string
  } | null {
    const splitPattern = /^(.+)-(\d+)-of-(\d+)\.gguf$/
    const match = filename.match(splitPattern)

    if (!match || match.length < 4) return null

    const baseName = match[1]!
    const currentPartStr = match[2]!
    const totalPartsStr = match[3]!
    const currentPart = parseInt(currentPartStr, 10)
    const totalParts = parseInt(totalPartsStr, 10)

    return {
      baseName,
      currentPart,
      totalParts,
      pattern: `${baseName}-*-of-${totalParts.toString().padStart(currentPartStr.length, '0')}.gguf`
    }
  }

  /**
   * Generate all split filenames based on the first split file
   */
  private static generateSplitFilenames(filename: string): string[] {
    const splitInfo = this.detectSplitFile(filename)
    if (!splitInfo) return [filename]

    const filenames = []
    const { baseName, totalParts } = splitInfo
    const paddingLength = Math.max(5, totalParts.toString().length) // Ensure at least 5 digits

    for (let i = 1; i <= totalParts; i += 1) {
      const partNumber = i.toString().padStart(paddingLength, '0')
      const generatedFilename = `${baseName}-${partNumber}-of-${totalParts.toString().padStart(paddingLength, '0')}.gguf`
      filenames.push(generatedFilename)
    }

    return filenames
  }

  async downloadModel(
    repo: string,
    filename: string,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<string> {
    // Check if this is a split file and download all parts
    const splitFilenames = ModelDownloader.generateSplitFilenames(filename)


    if (splitFilenames.length > 1) {
      return this.downloadSplitModel(repo, splitFilenames, onProgress)
    }

    // Single file download (existing logic)
    return this.downloadSingleModel(repo, filename, onProgress)
  }

  private async downloadSplitModel(
    repo: string,
    filenames: string[],
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<string> {
    const downloadKey = `${repo}/${filenames[0]}`

    if (this.activeDownloads.get(downloadKey)) {
      throw new Error('Download already in progress')
    }

    this.activeDownloads.set(downloadKey, true)

    try {
      const downloadDir = `${RNBlobUtil.fs.dirs.DocumentDir}/models`

      // Create models directory if it doesn't exist
      if (!(await RNBlobUtil.fs.exists(downloadDir))) {
        await RNBlobUtil.fs.mkdir(downloadDir)
      }

      // Check if all files already exist
      const existsChecks = await Promise.all(
        filenames.map(filename => RNBlobUtil.fs.exists(`${downloadDir}/${filename}`))
      )

      if (existsChecks.every(exists => exists)) {
        this.activeDownloads.delete(downloadKey)
        return `${downloadDir}/${filenames[0]}` // Return first file path for model loading
      }

      // First, get the total size of all split files by fetching headers
      let totalExpectedSize = 0
      const fileSizes: number[] = []

      /* eslint-disable no-await-in-loop */
      for (let i = 0; i < filenames.length; i += 1) {
        const filename = filenames[i]!
        const filePath = `${downloadDir}/${filename}`

        // Check if file already exists and get its size
        // eslint-disable-next-line no-await-in-loop
        if (await RNBlobUtil.fs.exists(filePath)) {
          // eslint-disable-next-line no-await-in-loop
          const stat = await RNBlobUtil.fs.stat(filePath)
          const existingSize = parseInt(stat.size.toString(), 10)
          fileSizes.push(existingSize)
          totalExpectedSize += existingSize
        } else {
          // Get file size from server headers using a range request
          try {
            const url = getModelDownloadUrl(repo, filename)
            // eslint-disable-next-line no-await-in-loop
            const rangeResponse = await RNBlobUtil.fetch('GET', url, {
              'Range': 'bytes=0-0'
            })
            const contentRange = rangeResponse.info().headers['content-range'] || rangeResponse.info().headers['Content-Range']
            if (contentRange) {
              // Extract total size from Content-Range: bytes 0-0/1234567
              const match = contentRange.match(/\/(\d+)$/)
              if (match) {
                const fileSize = parseInt(match[1], 10)
                fileSizes.push(fileSize)
                totalExpectedSize += fileSize
              } else {
                fileSizes.push(0)
              }
            } else {
              // Fallback: try content-length if available
              const contentLength = rangeResponse.info().headers['content-length'] || rangeResponse.info().headers['Content-Length']
              const fileSize = contentLength ? parseInt(contentLength, 10) : 0
              fileSizes.push(fileSize)
              totalExpectedSize += fileSize
            }
          } catch {
            // If request fails, use 0 as placeholder (will be updated during download)
            fileSizes.push(0)
          }
        }
      }

      // Download all split files with real byte tracking
      let totalBytesDownloaded = 0

      for (let i = 0; i < filenames.length; i += 1) {
        const filename = filenames[i]!
        const filePath = `${downloadDir}/${filename}`

        // Skip if file already exists (already counted in totalBytesDownloaded)
        // eslint-disable-next-line no-await-in-loop
        if (await RNBlobUtil.fs.exists(filePath)) {
          totalBytesDownloaded += fileSizes[i]!
          if (onProgress && totalExpectedSize > 0) {
            onProgress({
              written: totalBytesDownloaded,
              total: totalExpectedSize,
              percentage: Math.round((totalBytesDownloaded / totalExpectedSize) * 100),
            })
          }
          // eslint-disable-next-line no-continue
          continue
        }

        const url = getModelDownloadUrl(repo, filename)
        const config = RNBlobUtil.config({
          path: `${filePath}.tmp`,
          fileCache: true,
        })

        // eslint-disable-next-line no-await-in-loop
        const response = await config
          .fetch('GET', url)
          .progress((written, total) => {
            if (onProgress && Number(total) > 0) {
              // Update file size if we didn't get it from HEAD request
              if (fileSizes[i] === 0 && Number(total) > 0) {
                fileSizes[i] = Number(total)
                // Recalculate total expected size
                totalExpectedSize = fileSizes.reduce((sum, size) => sum + size, 0)
              }

              const currentFileBytes = Number(written)
              const totalDownloadedSoFar = totalBytesDownloaded + currentFileBytes

              if (totalExpectedSize > 0) {
                onProgress({
                  written: totalDownloadedSoFar,
                  total: totalExpectedSize,
                  percentage: Math.round((totalDownloadedSoFar / totalExpectedSize) * 100),
                })
              }
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
        // eslint-disable-next-line no-await-in-loop
        await RNBlobUtil.fs.mv(`${filePath}.tmp`, filePath)

        // Update total bytes downloaded with this file's size
        totalBytesDownloaded += fileSizes[i]!
      }
      /* eslint-enable no-await-in-loop */

      this.activeDownloads.delete(downloadKey)
      return `${downloadDir}/${filenames[0]}` // Return first file path for model loading
    } catch (error) {
      this.activeDownloads.delete(downloadKey)

      // Clean up temp files if they exist
      await Promise.all(
        filenames.map(async filename => {
          try {
            await RNBlobUtil.fs.unlink(`${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}.tmp`)
          } catch {
            // Ignore cleanup errors
          }
        })
      )

      // Re-throw with enhanced error message
      if (error && typeof error === 'object' && 'message' in error) {
        throw new Error(
          ModelDownloader.enhanceErrorMessage(error as Error, repo, filenames[0]!),
        )
      }
      throw error
    }
  }

  private async downloadSingleModel(
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
    const splitFilenames = this.generateSplitFilenames(filename)

    if (splitFilenames.length > 1) {
      // For split files, check if all parts exist
      const existsChecks = await Promise.all(
        splitFilenames.map(splitFilename => {
          const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${splitFilename}`
          return RNBlobUtil.fs.exists(filePath)
        })
      )
      return existsChecks.every(exists => exists)
    }

    // Single file check
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    return RNBlobUtil.fs.exists(filePath)
  }

  static async getModelPath(filename: string): Promise<string | null> {
    const splitFilenames = this.generateSplitFilenames(filename)

    if (splitFilenames.length > 1) {
      // For split files, return the first file path if all parts exist
      const existsChecks = await Promise.all(
        splitFilenames.map(splitFilename => {
          const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${splitFilename}`
          return RNBlobUtil.fs.exists(filePath)
        })
      )

      if (existsChecks.every(exists => exists)) {
        return `${RNBlobUtil.fs.dirs.DocumentDir}/models/${splitFilenames[0]}`
      }
      return null
    }

    // Single file check
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    const exists = await RNBlobUtil.fs.exists(filePath)
    return exists ? filePath : null
  }

  /**
   * Get split file information without checking if files exist on disk
   * Useful for displaying info about split models before they're downloaded
   */
  static async getSplitFileInfo(filename: string): Promise<{
    baseName: string
    currentPart: number
    totalParts: number
    pattern: string
  } | null> {
    return this.detectSplitFile(filename)
  }

  /**
   * Get the total size of a model by checking remote file sizes (for undownloaded models)
   * Returns size in bytes, or null if unable to determine
   */
  static async getModelSizeFromRemote(repo: string, filename: string): Promise<number | null> {
    try {
      const splitFilenames = this.generateSplitFilenames(filename)
      let totalSize = 0

      // Use Promise.all to avoid no-await-in-loop and for-of issues
      const sizePromises = splitFilenames.map(async splitFilename => {
        const url = getModelDownloadUrl(repo, splitFilename)
        try {
          const rangeResponse = await RNBlobUtil.fetch('GET', url, {
            'Range': 'bytes=0-0'
          })
          const contentRange = rangeResponse.info().headers['content-range'] || rangeResponse.info().headers['Content-Range']
          if (contentRange) {
            // Extract total size from Content-Range: bytes 0-0/1234567
            const match = contentRange.match(/\/(\d+)$/)
            if (match) {
              return parseInt(match[1], 10)
            }
            return null
          }
          // Fallback: try content-length if available
          const contentLength = rangeResponse.info().headers['content-length'] || rangeResponse.info().headers['Content-Length']
          return contentLength ? parseInt(contentLength, 10) : null
        } catch {
          return null // If any request fails, return null
        }
      })

      const sizes = await Promise.all(sizePromises)

      // If any size is null, return null
      if (sizes.some(size => size === null))  return null

      totalSize = sizes.reduce((sum, size) => sum! + (size ?? 0), 0)!

      return totalSize
    } catch {
      return null
    }
  }

  /**
   * Get formatted remote size string for undownloaded models
   */
  static async getModelSizeFromRemoteFormatted(repo: string, filename: string): Promise<string | null> {
    const size = await this.getModelSizeFromRemote(repo, filename)
    if (size === null) return null

    return this.formatBytes(size)
  }

  static async deleteModel(filename: string): Promise<void> {
    const splitFilenames = this.generateSplitFilenames(filename)

    if (splitFilenames.length > 1) {
      // For split files, delete all parts
      await Promise.all(
        splitFilenames.map(async splitFilename => {
          const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${splitFilename}`
          const exists = await RNBlobUtil.fs.exists(filePath)
          if (exists) {
            await RNBlobUtil.fs.unlink(filePath)
          }
        })
      )
      return
    }

    // Single file deletion
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    const exists = await RNBlobUtil.fs.exists(filePath)
    if (exists) {
      await RNBlobUtil.fs.unlink(filePath)
    }
  }

  /**
   * Get the total size of a model (single file or sum of all split parts)
   * Returns size in bytes, or null if model is not downloaded
   */
  static async getModelSize(filename: string): Promise<number | null> {
    const splitFilenames = this.generateSplitFilenames(filename)

    if (splitFilenames.length > 1) {
      // For split files, sum all parts
      try {
        const statPromises = splitFilenames.map(async splitFilename => {
          const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${splitFilename}`
          const stat = await RNBlobUtil.fs.stat(filePath)
          return parseInt(stat.size.toString(), 10)
        })

        const sizes = await Promise.all(statPromises)
        return sizes.reduce((total, size) => total + size, 0)
      } catch {
        // If any part is missing, return null
        return null
      }
    }

    // Single file size
    const filePath = `${RNBlobUtil.fs.dirs.DocumentDir}/models/${filename}`
    try {
      const stat = await RNBlobUtil.fs.stat(filePath)
      return parseInt(stat.size.toString(), 10)
    } catch {
      return null
    }
  }

  /**
   * Get formatted size string (e.g., "1.2 GB")
   */
  static async getModelSizeFormatted(filename: string): Promise<string | null> {
    const size = await this.getModelSize(filename)
    if (size === null) return null

    return this.formatBytes(size)
  }

  /**
   * Format bytes to human readable string
   */
  private static formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B'

    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))

    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
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
