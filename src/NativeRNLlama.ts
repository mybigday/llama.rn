import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type NativeContextParams = {
  model: string
  is_model_asset?: boolean

  embedding?: boolean

  n_ctx?: number
  n_batch?: number

  n_threads?: number
  n_gpu_layers?: number

  use_mlock?: boolean
  use_mmap?: boolean

  lora?: string // lora_adaptor
  lora_scaled?: number
  lora_base?: string

  rope_freq_base?: number
  rope_freq_scale?: number
}

export type NativeCompletionParams = {
  prompt: string
  grammar?: string
  stop?: Array<string> // -> antiprompt

  n_threads?: number
  n_predict?: number
  n_probs?: number
  top_k?: number
  top_p?: number
  min_p?: number
  tfs_z?: number
  typical_p?: number
  temperature?: number // -> temp
  penalty_last_n?: number
  penalty_repeat?: number
  penalty_freq?: number
  penalty_present?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  penalize_nl?: boolean
  seed?: number

  ignore_eos?: boolean
  logit_bias?: Array<Array<number>>

  emit_partial_completion: boolean
}

export type NativeCompletionTokenProbItem = {
  tok_str: string
  prob: number
}

export type NativeCompletionTokenProb = {
  content: string
  probs: Array<NativeCompletionTokenProbItem>
}

export type NativeCompletionResultTimings = {
  prompt_n: number
  prompt_ms: number
  prompt_per_token_ms: number
  prompt_per_second: number
  predicted_n: number
  predicted_ms: number
  predicted_per_token_ms: number
  predicted_per_second: number
}

export type NativeCompletionResult = {
  text: string

  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  stopped_eos: boolean
  stopped_word: string
  stopped_limit: number
  stopping_word: string
  tokens_cached: number
  timings: NativeCompletionResultTimings

  completion_probabilities?: Array<NativeCompletionTokenProb>
}

export type NativeTokenizeResult = {
  tokens: Array<number>
}

export type NativeEmbeddingResult = {
  embedding: Array<number>
}

export type NativeLlamaContext = {
  contextId: number
  gpu: boolean
  reasonNoGPU: string
}

export type NativeSessionLoadResult = {
  tokens_loaded: number
  prompt: string
}

export interface Spec extends TurboModule {
  setContextLimit(limit: number): Promise<void>;
  initContext(params: NativeContextParams): Promise<NativeLlamaContext>;

  loadSession(contextId: number, filepath: string): Promise<NativeSessionLoadResult>;
  saveSession(contextId: number, filepath: string, size: number): Promise<number>;
  completion(contextId: number, params: NativeCompletionParams): Promise<NativeCompletionResult>;
  stopCompletion(contextId: number): Promise<void>;
  tokenize(contextId: number, text: string): Promise<NativeTokenizeResult>;
  detokenize(contextId: number, tokens: number[]): Promise<string>;
  embedding(contextId: number, text: string): Promise<NativeEmbeddingResult>;
  bench(contextId: number, pp: number, tg: number, pl: number, nr: number): Promise<string>;

  releaseContext(contextId: number): Promise<void>;

  releaseAllContexts(): Promise<void>;
}

export default TurboModuleRegistry.get<Spec>('RNLlama') as Spec;
