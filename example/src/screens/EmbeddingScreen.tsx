import React, { useState, useEffect, useLayoutEffect } from 'react'
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

const availableModels = Object.keys(MODELS)
  .map((key) => ({
    key,
    ...MODELS[key as keyof typeof MODELS],
  }))
  .filter((model) => (model as any).embedding)

const EXAMPLE_TEXTS = [
  'Artificial intelligence is transforming the way we work and live by automating complex tasks and providing intelligent insights.',
  'Climate change poses significant challenges to global ecosystems, requiring urgent action from governments and individuals worldwide.',
  'Machine learning algorithms can process vast amounts of data to identify patterns and make predictions with remarkable accuracy.',
  'Renewable energy sources like solar and wind power are becoming increasingly cost-effective alternatives to fossil fuels.',
  'The human brain contains approximately 86 billion neurons that communicate through trillions of synaptic connections.',
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
  })
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [embeddings, setEmbeddings] = useState<EmbeddingData[]>([])
  const [inputText, setInputText] = useState('')
  const [queryText, setQueryText] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isEmbedding, setIsEmbedding] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [isModelReady, setIsModelReady] = useState(false)
  const [initProgress, setInitProgress] = useState(0)

  // Setup and navigation
  const [contextParams, setContextParams] = useState<ContextParams | null>(null)
  const [showContextParamsModal, setShowContextParamsModal] = useState(false)

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
      navigation.setOptions({ headerRight: () => null })
    }
  }, [navigation, isModelReady])

  const handleReleaseContext = async () => {
    if (context) {
      try {
        await context.release()
        setContext(null)
        setEmbeddings([])
        setSearchResults([])
      } catch (error) {
        console.error('Context release error:', error)
      }
    }
  }

  const handleInitializeModel = async (modelConfig: any) => {
    if (context) {
      await handleReleaseContext()
    }

    setIsModelReady(false)
    setInitProgress(0)

    try {
      const newContext = await initLlama(modelConfig, (progress) =>
        setInitProgress(progress),
      )
      setContext(newContext)
      setIsModelReady(true)
      setInitProgress(100)
    } catch (error) {
      console.error('Model initialization error:', error)
      setIsModelReady(false)
      setInitProgress(0)
      Alert.alert('Error', `Failed to load model: ${error}`)
    }
  }

  const handleAddEmbedding = async () => {
    if (!context || !inputText.trim()) return

    setIsEmbedding(true)
    try {
      const result = await context.embedding(inputText.trim())
      const newEmbedding: EmbeddingData = {
        id: Date.now().toString() + Math.random().toString(36).substring(2, 11),
        text: inputText.trim(),
        embedding: result.embedding,
      }

      setEmbeddings((prev) => [...prev, newEmbedding])
      setInputText('')
      Alert.alert('Success', 'Text embedded and added to memory!')
    } catch (error) {
      console.error('Embedding error:', error)
      Alert.alert('Error', `Failed to create embedding: ${error}`)
    } finally {
      setIsEmbedding(false)
    }
  }

  const handleSearch = async () => {
    if (!context || !queryText.trim() || embeddings.length === 0) return

    setIsSearching(true)
    try {
      const queryResult = await context.embedding(queryText.trim())
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
    } catch (error) {
      console.error('Search error:', error)
      Alert.alert('Error', `Search failed: ${error}`)
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
      const newEmbeddings = await EXAMPLE_TEXTS.reduce(
        async (acc: Promise<EmbeddingData[]>, exampleText) => {
          const embds = await acc
          const result = await context.embedding(exampleText)
          return [
            ...embds,
            {
              id:
                Date.now().toString() +
                Math.random().toString(36).substring(2, 11),
              text: exampleText,
              embedding: result.embedding,
            },
          ]
        },
        Promise.resolve([]),
      )
      setEmbeddings((prev) => [...prev, ...newEmbeddings])
      Alert.alert(
        'Success',
        `Imported ${EXAMPLE_TEXTS.length} example texts to the database!`,
      )
    } catch (error) {
      console.error('Import examples error:', error)
      Alert.alert('Error', `Failed to import examples: ${error}`)
    } finally {
      setIsImporting(false)
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

  if (!context) {
    return (
      <View style={styles.container}>
        <ScrollView
          style={themedStyles.setupContainer}
          contentContainerStyle={themedStyles.scrollContent}
        >
          <Text style={themedStyles.setupDescription}>
            Very simple example to show how to use vector embeddings and
            semantic search in memory.
          </Text>

          <View style={styles.modelsContainer}>
            <Text style={themedStyles.modelSectionTitle}>Available Models</Text>
            {availableModels.map((model) => (
              <ModelDownloadCard
                key={model.key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                onInitialize={(path) =>
                  handleInitializeModel({
                    model: path,
                    embedding: true,
                    ...contextParams,
                  })
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
          visible={!isModelReady && initProgress > 0}
          text={`Initializing model... ${initProgress}%`}
          progress={initProgress}
          showProgressBar={initProgress > 0}
        />
      </View>
    )
  }

  return (
    <View style={themedStyles.container}>
      <ScrollView style={styles.container}>
        {/* Header Info */}
        <View style={styles.headerInfo}>
          <Text style={styles.modelInfo}>
            {`Model: ${
              (context.model.metadata as any)?.general?.name ||
              context.model.desc ||
              'Unknown'
            }`}
          </Text>
          <Text style={styles.embeddingCount}>
            {`Embeddings in memory: ${embeddings.length}`}
          </Text>
        </View>

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
              (!inputText.trim() || isEmbedding) && themedStyles.disabledButton,
            ]}
            onPress={handleAddEmbedding}
            disabled={!inputText.trim() || isEmbedding}
          >
            {isEmbedding ? (
              <ActivityIndicator color={theme.colors.white} size="small" />
            ) : (
              <Text style={themedStyles.primaryButtonText}>Add to Memory</Text>
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
              (!queryText.trim() || embeddings.length === 0 || isSearching) &&
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
              <Text style={themedStyles.primaryButtonText}>Search (Top 3)</Text>
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
                <Text style={themedStyles.secondaryButtonText}>Clear All</Text>
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
      </ScrollView>
    </View>
  )
}

export default EmbeddingScreen
