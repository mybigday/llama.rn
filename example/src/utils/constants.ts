export const MODELS = {
  SMOL_LM_3: {
    name: 'SmolLM3 3B (Q4_K_M)',
    repo: 'ggml-org/SmolLM3-3B-GGUF',
    filename: 'SmolLM3-Q4_K_M.gguf',
    mmproj: undefined,
    size: '1.78GB',
  },
  GEMMA_3N_E2B: {
    name: 'Gemma 3N E2B IT (Q3_K_M)',
    repo: 'unsloth/gemma-3n-E2B-it-GGUF',
    filename: 'gemma-3n-E2B-it-Q3_K_M.gguf',
    mmproj: undefined,
    size: '2.31GB',
  },
  GEMMA_3N_E4B: {
    name: 'Gemma 3N E4B IT (Q3_K_M)',
    repo: 'unsloth/gemma-3n-E4B-it-GGUF',
    filename: 'gemma-3n-E4B-it-Q3_K_M.gguf',
    mmproj: undefined,
    size: '3.44GB',
  },
  QWEN_3_4B: {
    name: 'Qwen 3 4B (Q3_K_M)',
    repo: 'unsloth/Qwen3-4B-GGUF',
    filename: 'Qwen3-4B-Q3_K_M.gguf',
    mmproj: undefined,
    size: '1.93GB',
  },
  SMOL_VLM_500M: {
    name: 'SmolVLM 500M Instruct (Q8_0)',
    repo: 'ggml-org/SmolVLM-500M-Instruct-GGUF',
    filename: 'SmolVLM-500M-Instruct-Q8_0.gguf',
    mmproj: 'mmproj-SmolVLM-500M-Instruct-Q8_0.gguf',
    size: '416MB',
  },
  SMOL_VLM_2_2B: {
    name: 'SmolVLM2 2.2B Instruct (Q4_K_M)',
    repo: 'ggml-org/SmolVLM2-2.2B-Instruct-GGUF',
    filename: 'SmolVLM2-2.2B-Instruct-Q4_K_M.gguf',
    mmproj: 'mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf',
    size: '1.8GB (model) + 565MB (mmproj)',
  },
  INTERNVL3_2B: {
    name: 'InternVL3 2B (Q8_0)',
    repo: 'ggml-org/InternVL3-2B-Instruct-GGUF',
    filename: 'InternVL3-2B-Instruct-Q8_0.gguf',
    mmproj: 'mmproj-InternVL3-2B-Instruct-Q8_0.gguf',
    size: '1.89GB (model) + 337MB (mmproj)',
  },
  QWEN2_5_VL_3B: {
    name: 'Qwen2.5 VL 3B (Q4_K_M)',
    repo: 'ggml-org/Qwen2.5-VL-3B-Instruct-GGUF',
    filename: 'Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf',
    mmproj: 'mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf',
    size: '1.93GB (model) + 845MB (mmproj)',
  },
  GEMMA_3_4B_QAT: {
    name: 'Gemma 3 4B IT QAT (Q3_K_M)',
    repo: 'unsloth/gemma-3-4b-it-qat-GGUF',
    filename: 'gemma-3-4b-it-qat-Q3_K_M.gguf',
    mmproj: 'mmproj-BF16.gguf',
    size: '1.95GB (model) + 881MB (mmproj)',
  },
  OUTE_TTS_0_3: {
    name: 'OuteTTS 0.3 500M (Q4_K_M) + WavTokenizer (Q5_1)',
    repo: 'OuteAI/OuteTTS-0.3-500M-GGUF',
    filename: 'OuteTTS-0.3-500M-Q4_K_M.gguf',
    mmproj: undefined,
    size: '454MB (model) + 70MB (vocoder)',
    vocoder: {
      repo: 'ggml-org/WavTokenizer',
      filename: 'WavTokenizer-Large-75-Q5_1.gguf',
      size: '70MB',
    },
  },
}

export const HUGGINGFACE_BASE_URL = 'https://huggingface.co'

export const getModelDownloadUrl = (repo: string, filename: string) =>
  `${HUGGINGFACE_BASE_URL}/${repo}/resolve/main/${filename}?download=true`
