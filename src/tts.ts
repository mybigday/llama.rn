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
   * Reference sampling settings stored in the codec GGUF, in completion
   * param names. Only keys the model specifies are present.
   */
  samplingDefaults: {
    temperature?: number
    top_k?: number
    top_p?: number
    min_p?: number
    penalty_repeat?: number
  }
}

