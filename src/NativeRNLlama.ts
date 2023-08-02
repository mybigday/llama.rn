import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type NativeContextParams = {
  model: string
  is_model_asset?: boolean

  n_ctx?: number
  n_batch?: number

  n_threads?: number
  n_gpu_layers?: number

  use_mlock?: boolean
  use_mmap?: boolean

  memory_f16?: number

  lora?: string // lora_adaptor
  lora_base?: string

  n_gqa?: number
  rms_norm_eps?: number
  rope_freq_base?: number
  rope_freq_scale?: number
}

export type NativeCompletionParams = {
  prompt: string
  stop?: Array<string> // -> antiprompt

  n_predict?: number
  n_threads?: number
  n_probs?: number

  temperature?: number // -> temp

  repeat_last_n?: number
  repeat_penalty?: number
  presence_penalty?: number
  frequency_penalty?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  top_k?: number
  top_p?: number
  tfs_z?: number
  typical_p?: number

  ignore_eos?: boolean
  logit_bias?: Array<Array<number>>
}

export type NativeCompletionTokenProbItem = {
  tok_str: string
  prob: number
}

export type NativeCompletionTokenProb = {
  content: string
  probs: Array<NativeCompletionTokenProbItem>
}

export type NativeCompletionResult = {
  text: string
  completion_probabilities?: Array<NativeCompletionTokenProb>
}

export interface Spec extends TurboModule {
  setContextLimit(limit: number): Promise<void>;
  initContext(params: NativeContextParams): Promise<number>;

  completion(contextId: number, params: NativeCompletionParams): Promise<NativeCompletionResult>;
  stopCompletion(contextId: number): Promise<void>;
  releaseContext(contextId: number): Promise<void>;

  releaseAllContexts(): Promise<void>;
}

export default TurboModuleRegistry.get<Spec>('Llama') as Spec;
