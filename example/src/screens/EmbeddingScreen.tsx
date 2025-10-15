import React, { useState, useEffect, useLayoutEffect, useCallback } from 'react'
import {
  View,
  Text,
  ScrollView,
  Alert,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  FlatList,
  ActivityIndicator,
} from 'react-native'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import { MaskedProgress } from '../components/MaskedProgress'
import { HeaderButton } from '../components/HeaderButton'
import { createThemedStyles, Spacing, FontSizes } from '../styles/commonStyles'
import { useTheme } from '../contexts/ThemeContext'
import { MODELS } from '../utils/constants'
import type { ContextParams } from '../utils/storage'
import { loadContextParams } from '../utils/storage'
import { initLlama, LlamaContext } from '../../../src' // import 'llama.rn'

interface EmbeddingData {
  id: string
  text: string
  embedding: number[]
}

interface SearchResult {
  id: string
  text: string
  similarity: number
}

interface RerankResult {
  text: string
  score: number
  index: number
}

const calculateCosineSimilarity = (vecA: number[], vecB: number[]): number => {
  if (vecA.length !== vecB.length) return 0

  let dotProduct = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < vecA.length; i += 1) {
    const a = vecA[i] || 0
    const b = vecB[i] || 0
    dotProduct += a * b
    normA += a * a
    normB += b * b
  }

  const normProduct = Math.sqrt(normA) * Math.sqrt(normB)
  return normProduct === 0 ? 0 : dotProduct / normProduct
}

const embeddingModels = Object.keys(MODELS)
  .map((key) => ({
    key,
    ...MODELS[key as keyof typeof MODELS],
  }))
  .filter((model) => (model as any).embedding)

const rerankModels = Object.keys(MODELS)
  .map((key) => ({
    key,
    ...MODELS[key as keyof typeof MODELS],
  }))
  .filter((model) => (model as any).ranking)

const EXAMPLE_TEXTS = [
  'Artificial intelligence is transforming the way we work and live by automating complex tasks and providing intelligent insights.',
  'Climate change poses significant challenges to global ecosystems, requiring urgent action from governments and individuals worldwide.',
  'Machine learning algorithms can process vast amounts of data to identify patterns and make predictions with remarkable accuracy.',
  'Renewable energy sources like solar and wind power are becoming increasingly cost-effective alternatives to fossil fuels.',
  'The human brain contains approximately 86 billion neurons that communicate through trillions of synaptic connections.',
]

const RERANK_EXAMPLE_DOCUMENTS = [
  'The capital of France is Paris, which is known for the Eiffel Tower and the Louvre Museum.',
  'Python is a high-level programming language widely used for web development, data analysis, and machine learning.',
  'The Great Wall of China is one of the most iconic landmarks in the world, stretching over 13,000 miles.',
  'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.',
  'Tokyo is the capital city of Japan, famous for its blend of traditional culture and modern technology.',
]

const EmbeddingScreen = ({ navigation }: { navigation: any }) => {
  const { theme } = useTheme()
  const themedStyles = createThemedStyles(theme.colors)

  const styles = StyleSheet.create({
    container: themedStyles.container,
    headerInfo: {
      backgroundColor: theme.colors.surface,
      padding: Spacing.lg,
      marginBottom: Spacing.sm,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    modelInfo: {
      fontSize: FontSizes.large,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: Spacing.xs,
    },
    embeddingCount: {
      fontSize: FontSizes.medium,
      color: theme.colors.textSecondary,
    },
    modelsContainer: {
      marginTop: Spacing.lg,
    },
    section: {
      backgroundColor: theme.colors.surface,
      margin: Spacing.sm,
      padding: Spacing.lg,
      borderRadius: Spacing.md,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: theme.dark ? 0.3 : 0.1,
      shadowRadius: theme.dark ? 6 : 4,
      elevation: 3,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    sectionHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: Spacing.md,
    },
    sectionTitle: {
      fontSize: FontSizes.xlarge,
      fontWeight: '600',
      color: theme.colors.text,
      marginBottom: Spacing.md,
    },
    textInput: {
      ...themedStyles.textInput,
      minHeight: 80,
      textAlignVertical: 'top',
      marginBottom: Spacing.md,
    },
    embeddingItem: {
      backgroundColor: theme.colors.inputBackground,
      padding: Spacing.md,
      marginBottom: Spacing.sm,
      borderRadius: Spacing.sm,
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.primary,
    },
    embeddingText: {
      fontSize: FontSizes.medium,
      color: theme.colors.text,
      lineHeight: 20,
      marginBottom: Spacing.xs,
    },
    embeddingDimension: {
      fontSize: FontSizes.small,
      color: theme.colors.textSecondary,
    },
    searchResult: {
      backgroundColor: theme.colors.card,
      padding: Spacing.md,
      marginBottom: Spacing.sm,
      borderRadius: Spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
    },
    searchResultHeader: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: Spacing.xs,
    },
    searchResultRank: {
      fontSize: FontSizes.medium,
      fontWeight: '600',
      color: theme.colors.primary,
    },
    similarityScore: {
      fontSize: FontSizes.small,
      fontWeight: '500',
      backgroundColor: theme.colors.primary,
      color: theme.colors.white,
      paddingHorizontal: Spacing.sm,
      paddingVertical: 2,
      borderRadius: 12,
    },
    searchResultText: {
      fontSize: FontSizes.medium,
      color: theme.colors.text,
      lineHeight: 20,
    },
    importButton: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderColor: theme.colors.primary,
      borderRadius: Spacing.sm,
      paddingHorizontal: Spacing.md,
      paddingVertical: Spacing.xs,
    },
    importButtonText: {
      color: theme.colors.primary,
      fontSize: FontSizes.medium,
      fontWeight: '500',
    },
    documentInput: {
      backgroundColor: theme.colors.inputBackground,
      padding: Spacing.md,
      marginBottom: Spacing.sm,
      borderRadius: Spacing.sm,
      borderWidth: 1,
      borderColor: theme.colors.border,
      minHeight: 60,
      textAlignVertical: 'top',
      color: theme.colors.text,
    },
    documentItem: {
      backgroundColor: theme.colors.inputBackground,
      padding: Spacing.md,
      marginBottom: Spacing.sm,
      borderRadius: Spacing.sm,
      borderLeftWidth: 4,
      borderLeftColor: theme.colors.textSecondary,
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
    },
    documentText: {
      flex: 1,
      fontSize: FontSizes.medium,
      color: theme.colors.text,
      lineHeight: 20,
      marginRight: Spacing.sm,
    },
    removeButton: {
      padding: Spacing.xs,
    },
    removeButtonText: {
      color: theme.colors.error,
      fontSize: FontSizes.medium,
      fontWeight: '600',
    },
    addDocumentButton: {
      backgroundColor: theme.colors.surface,
      borderWidth: 1,
      borderStyle: 'dashed',
      borderColor: theme.colors.primary,
      borderRadius: Spacing.sm,
      padding: Spacing.md,
      alignItems: 'center',
      marginBottom: Spacing.md,
    },
    addDocumentText: {
      color: theme.colors.primary,
      fontSize: FontSizes.medium,
      fontWeight: '500',
    },
  })

  const [context, setContext] = useState<LlamaContext | null>(null)
  const [embeddings, setEmbeddings] = useState<EmbeddingData[]>([])
  const [inputText, setInputText] = useState('')
  const [queryText, setQueryText] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isEmbedding, setIsEmbedding] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)

  // Rerank state
  const [rerankQuery, setRerankQuery] = useState('')
  const [documents, setDocuments] = useState<string[]>([])
  const [rerankResults, setRerankResults] = useState<RerankResult[]>([])
  const [isReranking, setIsReranking] = useState(false)
  const [newDocument, setNewDocument] = useState('')

  // Setup and navigation
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)
  const [modelType, setModelType] = useState<'embedding' | 'ranking' | null>(null)
  const [useParallelMode, setUseParallelMode] = useState(true)

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const ctxParams = await loadContextParams()
        setContextParams(ctxParams)
      } catch (error) {
        console.error('Failed to load initial data:', error)
      }
    }
    loadInitialData()
  }, [])

  const handleReleaseContext = useCallback(async () => {
    if (context) {
      try {
        await context.release()
        setContext(null)
        setEmbeddings([])
        setSearchResults([])
        setDocuments([])
        setRerankResults([])
      } catch (error) {
        console.error('Context release error:', error)
      }
    }
  }, [context])

  const handleInitializeModel = useCallback(async (
    modelConfig: any,
    type: 'embedding' | 'ranking',
  ) => {
    if (context) {
      await handleReleaseContext()
    }

    setIsLoading(true)
    setIsModelReady(false)
    setInitProgress(0)

    try {
      const poolingType = type === 'ranking' ? 'rank' : 'cls'

      const newContext = await initLlama(
        {
          ...modelConfig,
          n_parallel: useParallelMode ? 4 : undefined,
          pooling_type: poolingType,
        },
        (progress) => setInitProgress(progress),
      )

      // Enable parallel mode if requested
      if (useParallelMode) {
        const success = await newContext.parallel.enable({
          n_parallel: 4,
          n_batch: 512,
        })

        if (!success) {
          throw new Error('Failed to enable parallel mode')
        }
      }

      setContext(newContext)
      setIsModelReady(true)
      setInitProgress(100)
      setModelType(type)
      console.log(
        `${useParallelMode ? 'Parallel' : 'Single'} mode enabled with pooling_type: ${poolingType} (${type} model)`,
      )
    } catch (error) {
      console.error('Model initialization error:', error)
      setIsModelReady(false)
      setInitProgress(0)
      Alert.alert('Error', `Failed to load model: ${error}`)
    } finally {
      setIsLoading(false)
    }
  }, [context, handleReleaseContext, useParallelMode])

  useLayoutEffect(() => {
    if (!isModelReady) {
      navigation.setOptions({
        headerRight: () => (
          <HeaderButton
            iconName="cog-outline"
            onPress={() => setShowContextParamsModal(true)}
          />
        ),
      })
    } else {
      navigation.setOptions({
        headerRight: () => (
          <View style={{ flexDirection: 'row', alignItems: 'center', marginRight: 8 }}>
            <TouchableOpacity
              style={{
                flexDirection: 'row',
                alignItems: 'center',
                backgroundColor: theme.colors.surface,
                borderRadius: 16,
                paddingHorizontal: 12,
                paddingVertical: 6,
                marginRight: 8,
                borderWidth: 1,
                borderColor: useParallelMode ? theme.colors.primary : theme.colors.border,
              }}
              onPress={() => {
                Alert.alert(
                  'Change Mode',
                  `Switch to ${useParallelMode ? 'Single' : 'Parallel'} mode? This will reinitialize the model.`,
                  [
                    { text: 'Cancel', style: 'cancel' },
                    {
                      text: 'Switch',
                      onPress: async () => {
                        setUseParallelMode(!useParallelMode)
                        // Trigger reinitialization if model is loaded
                        if (context && modelType) {
                          const {modelPath} = (context as any)
                          if (modelPath) {
                            await handleInitializeModel(
                              {
                                model: modelPath,
                                embedding: true,
                                ...contextParams,
                              },
                              modelType,
                            )
                          }
                        }
                      },
                    },
                  ],
                )
              }}
            >
              <Text
                style={{
                  fontSize: 12,
                  fontWeight: '600',
                  color: useParallelMode ? theme.colors.primary : theme.colors.text,
                  marginRight: 4,
                }}
              >
                {useParallelMode ? '⚡ Parallel' : '🔄 Single'}
              </Text>
            </TouchableOpacity>
            <HeaderButton
              iconName="information-outline"
              onPress={() => {
                Alert.alert(
                  'Embedding & Rerank',
                  [
                    'This demo showcases embedding and reranking operations with optional parallel mode.',
                    '',
                    `Mode: ${useParallelMode ? 'Parallel (faster batch operations)' : 'Single (one at a time)'}`,
                    '',
                    'Vector Search: Create embeddings and search semantically similar content.',
                    '',
                    'Rerank: Score and rank documents by relevance to a query.',
                    '',
                    'Tap the mode toggle to switch between single and parallel processing!',
                  ].join('\n'),
                )
              }}
            />
          </View>
        ),
      })
    }
  }, [navigation, isModelReady, useParallelMode, theme, context, modelType, contextParams, handleInitializeModel])

  const handleAddEmbedding = async () => {
    if (!context || !inputText.trim()) return

    setIsEmbedding(true)
    try {
      let result
      if (useParallelMode) {
        const { promise } = await context.parallel.embedding(inputText.trim())
        result = await promise
      } else {
        result = await context.embedding(inputText.trim())
      }

      const newEmbedding: EmbeddingData = {
        id: Date.now().toString() + Math.random().toString(36).substring(2, 11),
        text: inputText.trim(),
        embedding: result.embedding,
      }

      setEmbeddings((prev) => [...prev, newEmbedding])
      setInputText('')
      Alert.alert('Success', 'Text embedded and added to memory!')
    } catch (error: any) {
      console.error('Embedding error:', error)
      const errorMessage =
        error?.message || error?.toString() || 'Unknown error occurred'
      Alert.alert(
        'Embedding Error',
        `Failed to create embedding: ${errorMessage}`,
      )
    } finally {
      setIsEmbedding(false)
    }
  }

  const handleSearch = async () => {
    if (!context || !queryText.trim() || embeddings.length === 0) return

    setIsSearching(true)
    try {
      let queryResult
      if (useParallelMode) {
        const { promise } = await context.parallel.embedding(queryText.trim())
        queryResult = await promise
      } else {
        queryResult = await context.embedding(queryText.trim())
      }
      const queryEmbedding = queryResult.embedding

      const similarities = embeddings.map((item) => ({
        id: item.id,
        text: item.text,
        similarity: calculateCosineSimilarity(queryEmbedding, item.embedding),
      }))

      const topResults = similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 3)

      setSearchResults(topResults)
    } catch (error: any) {
      console.error('Search error:', error)
      const errorMessage =
        error?.message || error?.toString() || 'Unknown error occurred'
      Alert.alert(
        'Search Error',
        `Search failed: ${errorMessage}`,
      )
    } finally {
      setIsSearching(false)
    }
  }

  const clearEmbeddings = () => {
    Alert.alert('Clear All', 'Are you sure you want to clear all embeddings?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear',
        style: 'destructive',
        onPress: () => {
          setEmbeddings([])
          setSearchResults([])
        },
      },
    ])
  }

  const handleImportExamples = async () => {
    if (!context) return

    setIsImporting(true)
    try {
      let newEmbeddings: EmbeddingData[]

      if (useParallelMode) {
        // Queue all embedding requests in parallel
        const embeddingPromises = await Promise.all(
          EXAMPLE_TEXTS.map(async (exampleText) => {
            const { promise } = await context.parallel.embedding(exampleText)
            return promise.then((result) => ({
              id: Date.now().toString() + Math.random().toString(36).substring(2, 11),
              text: exampleText,
              embedding: result.embedding,
            }))
          })
        )
        newEmbeddings = await Promise.all(embeddingPromises)
      } else {
        // Process sequentially in single mode
        newEmbeddings = await EXAMPLE_TEXTS.reduce(
          async (acc: Promise<EmbeddingData[]>, exampleText) => {
            const embds = await acc
            const result = await context.embedding(exampleText)
            return [
              ...embds,
              {
                id: Date.now().toString() + Math.random().toString(36).substring(2, 11),
                text: exampleText,
                embedding: result.embedding,
              },
            ]
          },
          Promise.resolve([]),
        )
      }

      setEmbeddings((prev) => [...prev, ...newEmbeddings])
      Alert.alert(
        'Success',
        `Imported ${EXAMPLE_TEXTS.length} example texts using ${useParallelMode ? 'parallel' : 'single'} mode!`,
      )
    } catch (error: any) {
      console.error('Import examples error:', error)
      const errorMessage =
        error?.message || error?.toString() || 'Unknown error occurred'
      Alert.alert(
        'Import Error',
        `Failed to import examples: ${errorMessage}`,
      )
    } finally {
      setIsImporting(false)
    }
  }

  const handleAddDocument = () => {
    if (newDocument.trim()) {
      setDocuments((prev) => [...prev, newDocument.trim()])
      setNewDocument('')
    }
  }

  const handleRemoveDocument = (index: number) => {
    setDocuments((prev) => prev.filter((_, i) => i !== index))
  }

  const handleImportRerankExamples = () => {
    setDocuments(RERANK_EXAMPLE_DOCUMENTS)
    setRerankQuery('What is machine learning?')
    Alert.alert('Success', 'Example documents and query imported!')
  }

  const handleRerank = async () => {
    if (!context || !rerankQuery.trim() || documents.length === 0) return

    setIsReranking(true)
    try {
      let results
      if (useParallelMode) {
        const { promise } = await context.parallel.rerank(rerankQuery.trim(), documents)
        results = await promise
      } else {
        results = await context.rerank(rerankQuery.trim(), documents)
      }

      const rerankResultsWithText: RerankResult[] = results.map((item: any) => ({
        text: documents[item.index] || '',
        score: item.score,
        index: item.index,
      }))

      // Sort by score descending
      rerankResultsWithText.sort((a, b) => b.score - a.score)

      setRerankResults(rerankResultsWithText)
    } catch (error: any) {
      console.error('Rerank error:', error)
      const errorMessage =
        error?.message || error?.toString() || 'Unknown error occurred'
      Alert.alert(
        'Reranking Error',
        `Reranking failed: ${errorMessage}`,
      )
    } finally {
      setIsReranking(false)
    }
  }

  const renderEmbeddingItem = ({ item }: { item: EmbeddingData }) => (
    <View style={styles.embeddingItem}>
      <Text style={styles.embeddingText} numberOfLines={2}>
        {item.text}
      </Text>
      <Text style={styles.embeddingDimension}>
        {`Dimension: ${item.embedding.length}`}
      </Text>
    </View>
  )

  const renderSearchResult = ({
    item,
    index,
  }: {
    item: SearchResult
    index: number
  }) => {
    let backgroundColor = theme.colors.card
    if (index < 3) {
      backgroundColor = theme.dark ? '#1a365d' : '#f0f8ff'
    }

    return (
      <View style={[styles.searchResult, { backgroundColor }]}>
        <View style={styles.searchResultHeader}>
          <Text style={styles.searchResultRank}>{`#${index + 1}`}</Text>
          <Text style={styles.similarityScore}>
            {`${(item.similarity * 100).toFixed(1)}% match`}
          </Text>
        </View>
        <Text style={styles.searchResultText}>{item.text}</Text>
      </View>
    )
  }

  const renderRerankResult = ({
    item,
    index,
  }: {
    item: RerankResult
    index: number
  }) => {
    let backgroundColor = theme.colors.card
    if (index < 3) {
      backgroundColor = theme.dark ? '#1a365d' : '#f0f8ff'
    }

    return (
      <View style={[styles.searchResult, { backgroundColor }]}>
        <View style={styles.searchResultHeader}>
          <Text style={styles.searchResultRank}>{`#${index + 1}`}</Text>
          <Text style={styles.similarityScore}>
            {`Score: ${item.score.toFixed(4)}`}
          </Text>
        </View>
        <Text style={styles.searchResultText}>{item.text}</Text>
      </View>
    )
  }

  if (!context) {
    return (
      <View style={styles.container}>
        <ScrollView
          style={themedStyles.setupContainer}
          contentContainerStyle={themedStyles.scrollContent}
        >
          <Text style={themedStyles.setupDescription}>
            Demonstration of embedding and rerank operations. Supports both single
            and parallel modes for flexible processing.
          </Text>

          <View style={styles.modelsContainer}>
            <Text style={themedStyles.modelSectionTitle}>Embedding Models</Text>
            {embeddingModels.map((model) => (
              <ModelDownloadCard
                key={model.key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                onInitialize={(path) =>
                  handleInitializeModel(
                    {
                      model: path,
                      embedding: true,
                      ...contextParams,
                    },
                    'embedding',
                  )
                }
              />
            ))}
          </View>

          <View style={styles.modelsContainer}>
            <Text style={themedStyles.modelSectionTitle}>Rerank Models</Text>
            {rerankModels.map((model) => (
              <ModelDownloadCard
                key={model.key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                onInitialize={(path) =>
                  handleInitializeModel(
                    {
                      model: path,
                      embedding: true,
                      ...contextParams,
                    },
                    'ranking',
                  )
                }
              />
            ))}
          </View>
        </ScrollView>

        {/* Modals */}
        <ContextParamsModal
          visible={showContextParamsModal}
          onClose={() => setShowContextParamsModal(false)}
          onSave={(params) => setContextParams(params)}
        />

        <MaskedProgress
          visible={isLoading}
          text={`Initializing model... ${initProgress}%`}
          progress={initProgress}
          showProgressBar={initProgress > 0}
        />
      </View>
    )
  }

  return (
    <View style={themedStyles.container}>
      {/* Header Info */}
      <View style={styles.headerInfo}>
        <Text style={styles.modelInfo}>
          {`Model: ${
            (context.model.metadata as any)?.general?.name ||
            context.model.desc ||
            'Unknown'
          } (${modelType === 'embedding' ? 'Embedding' : 'Rerank'})`}
        </Text>
        <Text style={styles.embeddingCount}>
          {modelType === 'embedding'
            ? `Embeddings in memory: ${embeddings.length}`
            : `Documents: ${documents.length}`}
        </Text>
      </View>

      <ScrollView style={styles.container}>
        {modelType === 'embedding' ? (
          <>
            {/* Add Embedding Section */}
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Add Text to Embeddings</Text>
                <TouchableOpacity
                  style={[
                    styles.importButton,
                    isImporting && themedStyles.disabledButton,
                  ]}
                  onPress={handleImportExamples}
                  disabled={isImporting}
                >
                  {isImporting ? (
                    <ActivityIndicator color={theme.colors.primary} size="small" />
                  ) : (
                    <Text style={styles.importButtonText}>Import Examples</Text>
                  )}
                </TouchableOpacity>
              </View>
              <TextInput
                style={styles.textInput}
                placeholder="Enter text to embed..."
                placeholderTextColor={theme.colors.textSecondary}
                value={inputText}
                onChangeText={setInputText}
                multiline
                numberOfLines={3}
              />
              <TouchableOpacity
                style={[
                  themedStyles.primaryButton,
                  (!inputText.trim() || isEmbedding) &&
                    themedStyles.disabledButton,
                ]}
                onPress={handleAddEmbedding}
                disabled={!inputText.trim() || isEmbedding}
              >
                {isEmbedding ? (
                  <ActivityIndicator color={theme.colors.white} size="small" />
                ) : (
                  <Text style={themedStyles.primaryButtonText}>
                    Add to Memory
                  </Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Search Section */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Search Embeddings</Text>
              <TextInput
                style={styles.textInput}
                placeholder="Enter search query..."
                placeholderTextColor={theme.colors.textSecondary}
                value={queryText}
                onChangeText={setQueryText}
                multiline
                numberOfLines={2}
              />
              <TouchableOpacity
                style={[
                  themedStyles.primaryButton,
                  (!queryText.trim() ||
                    embeddings.length === 0 ||
                    isSearching) &&
                    themedStyles.disabledButton,
                ]}
                onPress={handleSearch}
                disabled={
                  !queryText.trim() || embeddings.length === 0 || isSearching
                }
              >
                {isSearching ? (
                  <ActivityIndicator color={theme.colors.white} size="small" />
                ) : (
                  <Text style={themedStyles.primaryButtonText}>
                    Search (Top 3)
                  </Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Search Results */}
            {searchResults.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Search Results</Text>
                <FlatList
                  data={searchResults}
                  renderItem={renderSearchResult}
                  keyExtractor={(item) => item.id}
                  scrollEnabled={false}
                />
              </View>
            )}

            {/* Embeddings List */}
            {embeddings.length > 0 && (
              <View style={styles.section}>
                <View style={styles.sectionHeader}>
                  <Text style={styles.sectionTitle}>All Embeddings</Text>
                  <TouchableOpacity
                    style={themedStyles.secondaryButton}
                    onPress={clearEmbeddings}
                  >
                    <Text style={themedStyles.secondaryButtonText}>
                      Clear All
                    </Text>
                  </TouchableOpacity>
                </View>
                <FlatList
                  data={embeddings}
                  renderItem={renderEmbeddingItem}
                  keyExtractor={(item) => item.id}
                  scrollEnabled={false}
                />
              </View>
            )}
          </>
        ) : (
          <>
            {/* Rerank Query Section */}
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Query</Text>
                <TouchableOpacity
                  style={styles.importButton}
                  onPress={handleImportRerankExamples}
                >
                  <Text style={styles.importButtonText}>Import Examples</Text>
                </TouchableOpacity>
              </View>
              <TextInput
                style={styles.textInput}
                placeholder="Enter your query..."
                placeholderTextColor={theme.colors.textSecondary}
                value={rerankQuery}
                onChangeText={setRerankQuery}
                multiline
                numberOfLines={2}
              />
            </View>

            {/* Documents Section */}
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Documents to Rerank</Text>
              <TextInput
                style={styles.documentInput}
                placeholder="Enter a document..."
                placeholderTextColor={theme.colors.textSecondary}
                value={newDocument}
                onChangeText={setNewDocument}
                multiline
                numberOfLines={3}
              />
              <TouchableOpacity
                style={[
                  themedStyles.primaryButton,
                  !newDocument.trim() && themedStyles.disabledButton,
                  { marginBottom: Spacing.md },
                ]}
                onPress={handleAddDocument}
                disabled={!newDocument.trim()}
              >
                <Text style={themedStyles.primaryButtonText}>Add Document</Text>
              </TouchableOpacity>

              {documents.length > 0 && (
                <>
                  <Text
                    style={[
                      styles.embeddingDimension,
                      { marginBottom: Spacing.sm },
                    ]}
                  >
                    {`${documents.length} document(s)`}
                  </Text>
                  {documents.map((doc, index) => (
                    <View key={index} style={styles.documentItem}>
                      <Text style={styles.documentText} numberOfLines={2}>
                        {doc}
                      </Text>
                      <TouchableOpacity
                        style={styles.removeButton}
                        onPress={() => handleRemoveDocument(index)}
                      >
                        <Text style={styles.removeButtonText}>✕</Text>
                      </TouchableOpacity>
                    </View>
                  ))}
                </>
              )}

              <TouchableOpacity
                style={[
                  themedStyles.primaryButton,
                  { marginTop: Spacing.md },
                  (!rerankQuery.trim() ||
                    documents.length === 0 ||
                    isReranking) &&
                    themedStyles.disabledButton,
                ]}
                onPress={handleRerank}
                disabled={
                  !rerankQuery.trim() || documents.length === 0 || isReranking
                }
              >
                {isReranking ? (
                  <ActivityIndicator color={theme.colors.white} size="small" />
                ) : (
                  <Text style={themedStyles.primaryButtonText}>Rerank Documents</Text>
                )}
              </TouchableOpacity>
            </View>

            {/* Rerank Results */}
            {rerankResults.length > 0 && (
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Ranked Results</Text>
                <Text style={[styles.embeddingDimension, { marginBottom: Spacing.md }]}>
                  Documents ranked by relevance to query
                </Text>
                <FlatList
                  data={rerankResults}
                  renderItem={renderRerankResult}
                  keyExtractor={(item, index) => `${item.index}-${index}`}
                  scrollEnabled={false}
                />
              </View>
            )}
          </>
        )}
      </ScrollView>
    </View>
  )
}

export default EmbeddingScreen
