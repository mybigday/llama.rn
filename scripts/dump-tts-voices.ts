/* eslint-disable no-restricted-syntax */
// Dump the default speaker payload for every (family, language) combo the
// JS-side voice tables know about, so the C++ tts_probe can feed them
// verbatim into getFormattedAudioCompletion.  Output shape:
//
//   {
//     "outetts:en-us:default": {...OuteTTSSpeaker},
//     "neutts:en-us:default":  {...NeuTTSSpeaker},
//     ...
//   }
//
// A missing entry means "no default voice for this family/language" —
// probe callers should omit --speaker-json for those.

import { lookupVoice, listVoices, listLanguages } from '../src/tts-voices'

type Family =
  | 'outetts'
  | 'soprano'
  | 'neutts'
  | 'csm'
  | 'qwen3_tts'
  | 'moss_tts'
  | 'moss_ttsd'
  | 'chatterbox'
  | 'bluemagpie'

const FAMILIES: Family[] = [
  'outetts',
  'soprano',
  'neutts',
  'csm',
  'qwen3_tts',
  'moss_tts',
  'moss_ttsd',
  'chatterbox',
  'bluemagpie',
]

const out: Record<string, unknown> = {}
for (const family of FAMILIES) {
  const langs = listLanguages(family)
  for (const lang of langs) {
    for (const name of listVoices(family, lang)) {
      const v = lookupVoice(family, name, lang)
      if (v) out[`${family}:${lang}:${name}`] = v
    }
  }
}
process.stdout.write(JSON.stringify(out, null, 2))
