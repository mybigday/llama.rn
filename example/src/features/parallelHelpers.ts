export const hashPrompt = (value: string): string => {
  let hash = 0
  for (let i = 0; i < value.length; i += 1) {
    const char = value.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash &= hash
  }
  return Math.abs(hash).toString(36)
}

export const buildParallelStatePath = (
  cacheDir: string,
  modelPath: string,
  prompt: string,
) => {
  const modelFilename =
    modelPath
      .split('/')
      .pop()
      ?.replace(/\.[^./]+$/, '') || 'unknown'
  const questionHash = hashPrompt(prompt.trim().toLowerCase())
  return `${cacheDir}/state_${modelFilename}_${questionHash}.bin`
}

export const formatParallelModeLabel = (isParallelMode: boolean) =>
  isParallelMode ? '⚡ Parallel' : '🔄 Single'
