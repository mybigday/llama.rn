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
 * codec.cpp's request overrides for `getFormattedAudioCompletion`. Sampling
 * is the `completion()`'s (llama.rn hands it to codec.cpp's own samplers too);
 * model graph settings (guidance, flow steps) are `initVocoder`
 * `kv_overrides`.
 */
export interface TTSRequestOptions {
  /** Shortest generation, in audio frames, before the end token may end it (in the grammar). Default: the model's. */
  minFrames?: number
}

/**
 * The `completion()` parameters a text-to-speech request's format needs, as
 * `getFormattedAudioCompletion` reports them. Sampling (the backbone GGUF's
 * `general.sampling.*` unless set) and `n_predict` are the caller's.
 */
export interface TTSCompletionParams {
  /** The armed text; `completion()` must be given this prompt. */
  prompt: string
  /** The model reads the backbone's hidden states (the context returns them). */
  embedding: boolean
  /** The tokens the model may emit and its minimum length. */
  grammar?: string
  /** One bias per vocabulary token (`-Infinity` for tokens the model never samples), when the model restricts tokens without a grammar. */
  logit_bias?: Float32Array
}
