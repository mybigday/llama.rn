export interface EmbeddingItem {
  id: string
  text: string
  embedding: number[]
}

export interface SearchResult {
  id: string
  text: string
  similarity: number
}

export interface RerankResponseItem {
  index: number
  score: number
}

export interface RerankResult {
  text: string
  score: number
  index: number
}

export const calculateCosineSimilarity = (
  vecA: number[],
  vecB: number[],
): number => {
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

export const rankEmbeddingSearchResults = (
  queryEmbedding: number[],
  items: EmbeddingItem[],
  limit = 3,
): SearchResult[] =>
  items
    .map((item) => ({
      id: item.id,
      text: item.text,
      similarity: calculateCosineSimilarity(queryEmbedding, item.embedding),
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit)

export const mapRerankResults = (
  documents: string[],
  results: RerankResponseItem[],
): RerankResult[] =>
  results
    .map((item) => ({
      text: documents[item.index] || '',
      score: item.score,
      index: item.index,
    }))
    .sort((a, b) => b.score - a.score)
