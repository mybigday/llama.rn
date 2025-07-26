export const MODELS = {
  SMOL_LM: {
    name: 'SmolLM3 3B',
    repo: 'ggml-org/SmolLM3-3B-GGUF',
    filename: 'SmolLM3-Q4_K_M.gguf',
    size: '1.78GB',
    description: 'Fast, efficient text generation model',
  },
  SMOL_VLM: {
    name: 'SmolVLM2 2.2B Instruct',
    repo: 'ggml-org/SmolVLM2-2.2B-Instruct-GGUF',
    filename: 'SmolVLM2-2.2B-Instruct-Q4_K_M.gguf',
    mmproj: 'mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf',
    size: '1.8GB (model) + 565MB (mmproj)',
    description: 'Vision-language model for image understanding',
  },
  OUTE_TTS: {
    name: 'OuteTTS 0.3 500M + WavTokenizer',
    repo: 'OuteAI/OuteTTS-0.3-500M-GGUF',
    filename: 'OuteTTS-0.3-500M-Q4_K_M.gguf',
    size: '454MB (model) + 70MB (vocoder)',
    description: 'Text-to-speech generation model with WavTokenizer vocoder',
    vocoder: {
      repo: 'ggml-org/WavTokenizer',
      filename: 'WavTokenizer-Large-75-Q5_1.gguf',
      size: '70MB',
    },
  },
} as const;


export const HUGGINGFACE_BASE_URL = 'https://huggingface.co';

export const getModelDownloadUrl = (repo: string, filename: string) =>
  `${HUGGINGFACE_BASE_URL}/${repo}/resolve/main/${filename}?download=true`;
