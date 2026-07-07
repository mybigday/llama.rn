// Native-side TTS capabilities snapshot. The native layer is the single
// source of truth — this module only declares the shape and exposes the
// JSI bridge.

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
    | 'bluemagpie'
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
    | ''
  /** True when the model was trained on phonemes — caller should provide a phonemizer hook. */
  requiresPhonemes: boolean
  /** Suggested language for the phonemizer hook ("en-us" today). */
  defaultLanguage: string
}
