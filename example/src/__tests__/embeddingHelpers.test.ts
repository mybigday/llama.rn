import {
  calculateCosineSimilarity,
  mapRerankResults,
  rankEmbeddingSearchResults,
} from '../features/embeddingHelpers'

describe('embedding helpers', () => {
  it('calculates cosine similarity safely', () => {
    expect(calculateCosineSimilarity([1, 0], [1, 0])).toBe(1)
    expect(calculateCosineSimilarity([1, 0], [0, 1])).toBe(0)
  })

  it('ranks embedding search results by similarity', () => {
    const results = rankEmbeddingSearchResults(
      [1, 0],
      [
        { id: 'a', text: 'A', embedding: [1, 0] },
        { id: 'b', text: 'B', embedding: [0, 1] },
      ],
      1,
    )

    expect(results).toEqual([{ id: 'a', text: 'A', similarity: 1 }])
  })

  it('maps rerank results back to documents in score order', () => {
    expect(
      mapRerankResults(['first', 'second'], [
        { index: 0, score: 0.1 },
        { index: 1, score: 0.9 },
      ]),
    ).toEqual([
      { text: 'second', score: 0.9, index: 1 },
      { text: 'first', score: 0.1, index: 0 },
    ])
  })
})
