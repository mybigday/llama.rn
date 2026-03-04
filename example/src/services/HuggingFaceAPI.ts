interface HuggingFaceModel {
  id: string
  siblings: Array<{
    rfilename: string
  }>
}

interface HuggingFaceSearchModel {
  id: string
  downloads?: number
  likes?: number
  pipeline_tag?: string
  tags?: string[]
  lastModified?: string
}

interface ModelFile {
  filename: string
  quantization: string
}

export interface HuggingFaceSearchResult {
  id: string
  downloads: number
  likes: number
  pipelineTag?: string
  tags: string[]
  lastModified?: string
}

export interface CustomModelInfo {
  id: string
  exists: boolean
  files: ModelFile[]
  mmprojFiles: ModelFile[]
  error?: string
}

export class HuggingFaceAPI {
  private static readonly baseUrl = 'https://huggingface.co/api/models'

  private static readonly modelSearchFilter = 'gguf'

  /**
   * Search HuggingFace Hub models tagged with GGUF.
   */
  static async searchModels(
    query: string,
    limit = 12,
  ): Promise<HuggingFaceSearchResult[]> {
    try {
      const trimmedQuery = query.trim()
      const params: Array<[string, string]> = [
        ['filter', this.modelSearchFilter],
        ['limit', String(limit)],
        ['sort', 'downloads'],
        ['direction', '-1'],
      ]

      if (trimmedQuery) {
        params.push(['search', trimmedQuery])
      }

      const queryString = params
        .map(
          ([key, value]) =>
            `${encodeURIComponent(key)}=${encodeURIComponent(value)}`,
        )
        .join('&')

      const response = await fetch(`${this.baseUrl}?${queryString}`)

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const models: HuggingFaceSearchModel[] = await response.json()

      return models
        .filter((model) => typeof model.id === 'string' && model.id.length > 0)
        .map((model) => ({
          id: model.id,
          downloads: model.downloads ?? 0,
          likes: model.likes ?? 0,
          pipelineTag: model.pipeline_tag,
          tags: model.tags ?? [],
          lastModified: model.lastModified,
        }))
    } catch (error) {
      console.error('Error searching HuggingFace models:', error)
      throw error
    }
  }

  /**
   * Fetch model information from HuggingFace Hub API
   */
  static async fetchModelInfo(modelId: string): Promise<CustomModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/${modelId}`)

      if (!response.ok) {
        if (response.status === 404) {
          return {
            id: modelId,
            exists: false,
            files: [],
            mmprojFiles: [],
            error: 'Model not found on HuggingFace Hub',
          }
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const model: HuggingFaceModel = await response.json()

      // Extract GGUF files
      const ggufFiles = model.siblings
        .filter((sibling) => sibling.rfilename.endsWith('.gguf'))
        .map((sibling) => ({
          filename: sibling.rfilename,
          quantization: this.extractQuantization(sibling.rfilename),
        }))

      // Separate regular model files from mmproj files
      const files = ggufFiles.filter(
        (file) => !file.filename.startsWith('mmproj-'),
      )
      const mmprojFiles = ggufFiles.filter((file) =>
        file.filename.startsWith('mmproj-'),
      )

      // Sort files by quantization priority
      files.sort((a, b) =>
        this.compareQuantizations(a.quantization, b.quantization),
      )
      mmprojFiles.sort((a, b) =>
        this.compareQuantizations(a.quantization, b.quantization),
      )

      return {
        id: modelId,
        exists: true,
        files,
        mmprojFiles,
      }
    } catch (error) {
      console.error('Error fetching model info:', error)
      return {
        id: modelId,
        exists: false,
        files: [],
        mmprojFiles: [],
        error:
          error instanceof Error ? error.message : 'Unknown error occurred',
      }
    }
  }

  /**
   * Extract quantization type from filename
   */
  private static extractQuantization(filename: string): string {
    // Common quantization patterns in GGUF files
    const patterns = [
      /[_-](iq\d+_[a-z]+)/i, // IQ4_NL, IQ4_XS, IQ3_XXS, etc.
      /[_-](q\d+_k_[lmsx]+)/i, // Q4_K_M, Q3_K_L, etc.
      /[_-](q\d+_\d+)/i, // Q4_0, Q5_1, etc.
      /[_-](q\d+)/i, // Q4, Q8, etc.
      /[_-](mxfp\d+)/i, // MXFP4, MXFP8, etc.
      /[_-](f\d+)/i, // F16, F32
      /[_-](bf\d+)/i, // BF16
    ]

    const match = patterns
      .map((pattern) => filename.match(pattern))
      .find((result) => result && result[1])

    return match && match[1] ? match[1].toUpperCase() : 'UNKNOWN'
  }

  /**
   * Compare quantizations for sorting (preferred order)
   */
  private static compareQuantizations(a: string, b: string): number {
    const priority = [
      'IQ4_NL',
      'Q4_K_M',
      'IQ4_XS',
      'Q3_K_M',
      'IQ3_M',
      'Q4_0',
      'IQ3_S',
      'Q5_K_M',
      'IQ3_XS',
      'Q5_0',
      'IQ3_XXS',
      'Q3_K_L',
      'Q3_K_S',
      'Q4_K_S',
      'IQ2_M',
      'IQ2_S',
      'IQ2_XS',
      'Q6_K',
      'IQ2_XXS',
      'IQ1_M',
      'IQ1_S',
      'MXFP4',
      'Q8_0',
      'F16',
      'F32',
      'BF16',
    ]

    const indexA = priority.indexOf(a)
    const indexB = priority.indexOf(b)

    // If both found, use priority order
    if (indexA !== -1 && indexB !== -1) {
      return indexA - indexB
    }

    // If only one found, prioritize it
    if (indexA !== -1) return -1
    if (indexB !== -1) return 1

    // If neither found, alphabetical order
    return a.localeCompare(b)
  }

  /**
   * Get the default quantization file based on priority
   */
  static getDefaultQuantization(files: ModelFile[]): ModelFile | null {
    if (files.length === 0) return null

    // Files are already sorted by priority, so return the first one
    return files[0] || null
  }

  /**
   * Get the default mmproj file
   */
  static getDefaultMmproj(mmprojFiles: ModelFile[]): ModelFile | null {
    if (mmprojFiles.length === 0) return null

    // For mmproj, prefer higher precision quantizations
    return mmprojFiles[0] || null
  }
}
