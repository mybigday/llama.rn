import React, {
  useState,
  useEffect,
  useLayoutEffect,
} from 'react'
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
import { useSafeAreaInsets } from 'react-native-safe-area-context'
import ModelDownloadCard from '../components/ModelDownloadCard'
import ContextParamsModal from '../components/ContextParamsModal'
import { HeaderButton } from '../components/HeaderButton'
import { CommonStyles, Colors, Spacing, FontSizes } from '../styles/commonStyles'
import { MODELS } from '../utils/constants'
import type {
  ContextParams,
} from '../utils/storage'
import {
  loadContextParams,
} from '../utils/storage'
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

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  headerInfo: {
    backgroundColor: Colors.white,
    padding: Spacing.lg,
    marginBottom: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  modelInfo: {
    fontSize: FontSizes.large,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: Spacing.xs,
  },
  embeddingCount: {
    fontSize: FontSizes.medium,
    color: Colors.textSecondary,
  },
  modelsContainer: {
    marginTop: Spacing.lg,
  },
  section: {
    backgroundColor: Colors.white,
    margin: Spacing.sm,
    padding: Spacing.lg,
    borderRadius: Spacing.md,
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
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
    color: Colors.text,
    marginBottom: Spacing.md,
  },
  textInput: {
    ...CommonStyles.textInput,
    minHeight: 80,
    textAlignVertical: 'top',
    marginBottom: Spacing.md,
  },
  embeddingItem: {
    backgroundColor: Colors.inputBackground,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderRadius: Spacing.sm,
    borderLeftWidth: 4,
    borderLeftColor: Colors.primary,
  },
  embeddingText: {
    fontSize: FontSizes.medium,
    color: Colors.text,
    lineHeight: 20,
    marginBottom: Spacing.xs,
  },
  embeddingDimension: {
    fontSize: FontSizes.small,
    color: Colors.textSecondary,
  },
  searchResult: {
    backgroundColor: Colors.white,
    padding: Spacing.md,
    marginBottom: Spacing.sm,
    borderRadius: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
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
    color: Colors.primary,
  },
  similarityScore: {
    fontSize: FontSizes.small,
    fontWeight: '500',
    backgroundColor: Colors.primary,
    color: Colors.white,
    paddingHorizontal: Spacing.sm,
    paddingVertical: 2,
    borderRadius: 12,
  },
  searchResultText: {
    fontSize: FontSizes.medium,
    color: Colors.text,
    lineHeight: 20,
  },
  importButton: {
    backgroundColor: Colors.white,
    borderWidth: 1,
    borderColor: Colors.primary,
    borderRadius: Spacing.sm,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
  },
  importButtonText: {
    color: Colors.primary,
    fontSize: FontSizes.medium,
    fontWeight: '500',
  },
})


const availableModels = Object.keys(MODELS).map(key => ({
  key,
  ...MODELS[key as keyof typeof MODELS],
})).filter(model => (model as any).embedding)

const EXAMPLE_TEXTS = [
  "Artificial intelligence is transforming the way we work and live by automating complex tasks and providing intelligent insights.",
  "Climate change poses significant challenges to global ecosystems, requiring urgent action from governments and individuals worldwide.",
  "Machine learning algorithms can process vast amounts of data to identify patterns and make predictions with remarkable accuracy.",
  "Renewable energy sources like solar and wind power are becoming increasingly cost-effective alternatives to fossil fuels.",
  "The human brain contains approximately 86 billion neurons that communicate through trillions of synaptic connections."
]

const EmbeddingScreen = ({ navigation }: { navigation: any }) => {
  const insets = useSafeAreaInsets()
  const [context, setContext] = useState<LlamaContext | null>(null)
  const [embeddings, setEmbeddings] = useState<EmbeddingData[]>([])
  const [inputText, setInputText] = useState('')
  const [queryText, setQueryText] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isEmbedding, setIsEmbedding] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [isImporting, setIsImporting] = useState(false)

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
    if (!context) {
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
  }, [navigation, context])


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
    try {
      const newContext = await initLlama(modelConfig)
      setContext(newContext)
      Alert.alert('Success', 'Model loaded successfully!')
    } catch (error) {
      console.error('Model initialization error:', error)
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

      setEmbeddings(prev => [...prev, newEmbedding])
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

      const similarities = embeddings.map(item => ({
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
    Alert.alert(
      'Clear All',
      'Are you sure you want to clear all embeddings?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: () => {
            setEmbeddings([])
            setSearchResults([])
          }
        }
      ]
    )
  }

  const handleImportExamples = async () => {
    if (!context) return

    setIsImporting(true)
    try {
      const newEmbeddings = await EXAMPLE_TEXTS.reduce(async (acc: Promise<EmbeddingData[]>, exampleText) => {
        const embds = await acc
        const result = await context.embedding(exampleText)
        return [...embds, {
          id: Date.now().toString() + Math.random().toString(36).substring(2, 11),
          text: exampleText,
          embedding: result.embedding,
        }]
      }, Promise.resolve([]))
      setEmbeddings(prev => [...prev, ...newEmbeddings])
      Alert.alert('Success', `Imported ${EXAMPLE_TEXTS.length} example texts to the database!`)
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

  const renderSearchResult = ({ item, index }: { item: SearchResult; index: number }) => (
    <View style={[styles.searchResult, { backgroundColor: index < 3 ? '#f0f8ff' : Colors.white }]}>
      <View style={styles.searchResultHeader}>
        <Text style={styles.searchResultRank}>{`#${index + 1}`}</Text>
        <Text style={styles.similarityScore}>
          {`${(item.similarity * 100).toFixed(1)}% match`}
        </Text>
      </View>
      <Text style={styles.searchResultText}>
        {item.text}
      </Text>
    </View>
  )

  if (!context) {
    return (
      <ScrollView style={[CommonStyles.container, { paddingTop: insets.top }]}>
        <View style={CommonStyles.setupContainer}>
          <Text style={CommonStyles.setupDescription}>
            Very simple example to show how to use vector embeddings and semantic search in memory.
          </Text>

          <View style={styles.modelsContainer}>
            <Text style={CommonStyles.modelSectionTitle}>Available Models</Text>
            {availableModels.map(model => (
              <ModelDownloadCard
                key={model.key}
                title={model.name}
                repo={model.repo}
                filename={model.filename}
                size={model.size}
                onInitialize={(path) => handleInitializeModel({
                  model: path,
                  embedding: true,
                  ...contextParams,
                })}
              />
            ))}
          </View>
        </View>
      </ScrollView>
    )
  }

  return (
    <View style={[CommonStyles.container, { paddingTop: insets.top }]}>
      <ScrollView style={styles.container}>
        {/* Header Info */}
        <View style={styles.headerInfo}>
          <Text style={styles.modelInfo}>
            {`Model: ${(context.model.metadata as any)?.general?.name || context.model.desc || 'Unknown'}`}
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
                isImporting && CommonStyles.disabledButton
              ]}
              onPress={handleImportExamples}
              disabled={isImporting}
            >
              {isImporting ? (
                <ActivityIndicator color={Colors.primary} size="small" />
              ) : (
                <Text style={styles.importButtonText}>Import Examples</Text>
              )}
            </TouchableOpacity>
          </View>
          <TextInput
            style={styles.textInput}
            placeholder="Enter text to embed..."
            value={inputText}
            onChangeText={setInputText}
            multiline
            numberOfLines={3}
          />
          <TouchableOpacity
            style={[
              CommonStyles.primaryButton,
              (!inputText.trim() || isEmbedding) && CommonStyles.disabledButton
            ]}
            onPress={handleAddEmbedding}
            disabled={!inputText.trim() || isEmbedding}
          >
            {isEmbedding ? (
              <ActivityIndicator color={Colors.white} size="small" />
            ) : (
              <Text style={CommonStyles.primaryButtonText}>Add to Memory</Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Search Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Search Embeddings</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Enter search query..."
            value={queryText}
            onChangeText={setQueryText}
            multiline
            numberOfLines={2}
          />
          <TouchableOpacity
            style={[
              CommonStyles.primaryButton,
              (!queryText.trim() || embeddings.length === 0 || isSearching) && CommonStyles.disabledButton
            ]}
            onPress={handleSearch}
            disabled={!queryText.trim() || embeddings.length === 0 || isSearching}
          >
            {isSearching ? (
              <ActivityIndicator color={Colors.white} size="small" />
            ) : (
              <Text style={CommonStyles.primaryButtonText}>
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
              keyExtractor={item => item.id}
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
                style={CommonStyles.secondaryButton}
                onPress={clearEmbeddings}
              >
                <Text style={CommonStyles.secondaryButtonText}>Clear All</Text>
              </TouchableOpacity>
            </View>
            <FlatList
              data={embeddings}
              renderItem={renderEmbeddingItem}
              keyExtractor={item => item.id}
              scrollEnabled={false}
            />
          </View>
        )}
      </ScrollView>

      {/* Modals */}
      <ContextParamsModal
        visible={showContextParamsModal}
        onClose={() => setShowContextParamsModal(false)}
        onSave={(params) => setContextParams(params)}
      />
    </View>
  )
}

export default EmbeddingScreen
