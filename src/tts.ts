// Native-side TTS capabilities snapshot. The native layer is the single
// source of truth — this module only declares the shape and exposes the
// JSI bridge.

/** A named decoder speaker: a builtin voice of a codec GGUF or an entry in a speaker file. */
export interface TTSSpeakerInfo {
  name: string
  /** Transcript of the reference audio, when known. */
  refText: string
  isDefault: boolean
}

export interface TTSCapabilities {
  /** Numeric tts_type enum value (matches cpp/rn-tts.h). */
  type: number
  /** Prompt assembly family — drives default-voice selection on the JS side. */
  promptKind:
    | 'outetts_legacy'
    | 'outetts_v0_3'
    | 'outetts_v1_0'
    | 'soprano'
    | 'neutts'
    | 'csm'
    | 'qwen3_tts'
    | 'moss_tts_realtime'
    | 'moss_ttsd'
    | 'chatterbox'
    | 'chatterbox_multilingual'
    | 'chatterbox_turbo'
    | 'bluemagpie'
    | 'cosyvoice'
    | ''
  family:
    | 'outetts'
    | 'soprano'
    | 'neutts'
    | 'csm'
    | 'qwen3_tts'
    | 'moss_tts'
    | 'moss_ttsd'
    | 'chatterbox'
    | 'bluemagpie'
    | 'cosyvoice'
    | ''
  /** True when the model was trained on phonemes — caller should provide a phonemizer hook. */
  requiresPhonemes: boolean
  /** Suggested language for the phonemizer hook ("en-us" today). */
  defaultLanguage: string
  /**
   * Builtin decoder voices of the codec GGUF (Chatterbox S3G, CosyVoice).
   * Pass a voice's `name` as `speaker` to `getFormattedAudioCompletion`.
   */
  builtinVoices: TTSSpeakerInfo[]
  /**
   * True when a `LlamaSpeaker`'s reference audio also conditions the decoder
   * (voice cloning), which also lets `saveSpeaker` write it to a file.
   */
  canSpeakerFromAudio: boolean
  /**
   * True when the model takes an `instruct` (how to speak, e.g. "Speak
   * cheerfully") in `getFormattedAudioCompletion` — Qwen3-TTS CustomVoice
   * 1.7B, CosyVoice.
   */
  canTakeInstruct: boolean
  /**
   * True when the model has no voice of its own: pass a `LlamaSpeaker`
   * (reference audio) as `speaker` (e.g. Qwen3-TTS Base).
   */
  needsVoice: boolean
  /**
   * Always empty: a text-to-speech completion samples with the model's own
   * settings (codec.cpp).
   * @deprecated
   */
  samplingDefaults: {
    temperature?: number
    top_k?: number
    top_p?: number
    min_p?: number
    penalty_repeat?: number
  }
}


/**
 * codec.cpp's request overrides for `getFormattedAudioCompletion`. Unset
 * fields keep the model's defaults. The sampling fields apply to the samplers
 * codec.cpp runs itself (codec-side heads) and to the backbone sampling it
 * reports in `completionParams`.
 */
export interface TTSRequestOptions {
  /** Seed of codec.cpp's samplers (and `completionParams.seed`). */
  seed?: number
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repetitionPenalty?: number
  /** Shortest generation, in audio frames, before the stop token may end it. */
  minFrames?: number
  /** Longest generation, in audio frames (`completionParams.n_predict`). */
  maxFrames?: number
  /** Classifier-free guidance weight of the backbone pair; `0` turns it off (one sequence). */
  lmGuidance?: number
  /** Guidance of a flow / diffusion decoder. */
  flowGuidance?: number
  /** Steps of a flow / diffusion decoder. */
  flowSteps?: number
  /** A sampling preset the model defines. */
  preset?: string
}

/**
 * The `completion()` parameters a text-to-speech request runs with, as
 * `getFormattedAudioCompletion` reports them: the model's backbone sampling
 * and constraints. Pass them to `completion()` (and override any of them);
 * nothing is applied implicitly.
 */
export interface TTSCompletionParams {
  /** The armed text; `completion()` must be given this prompt. */
  prompt: string
  /** The model reads the backbone's hidden states (the context returns them). */
  embedding: boolean
  /** Step budget: the model's frame limit, or `-1`. */
  n_predict: number
  seed: number
  temperature: number
  top_k: number
  top_p: number
  min_p: number
  penalty_last_n: number
  penalty_repeat: number
  penalty_freq: number
  penalty_present: number
  penalize_prompt: boolean
  dry_multiplier: number
  dry_base: number
  dry_allowed_length: number
  dry_penalty_last_n: number
  /** Sampler order (llama.cpp `--samplers` names). */
  samplers: string
  grammar?: string
  allowed_tokens?: Array<[number, number]>
}
