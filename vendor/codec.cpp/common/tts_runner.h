#ifndef CODEC_TTS_RUNNER_H
#define CODEC_TTS_RUNNER_H

// codec_common::tts_runner — OPTIONAL reference host loop for TTS.
//
// codec_common (audio_lm.*) provides per-step hooks so a host that owns
// its own `llama_decode` loop (llama.rn's rn-tts) can drive an audio-LM
// without inheriting a loop it can't control.  That contract is
// unchanged — see docs/codec_common_api.md §Boundary.
//
// This header adds a SEPARATE, opt-in convenience layer for hosts that
// DO want a complete, ready-made reference loop: it LINKS the isolated
// llama.cpp backbone (libttsbackbone, cmake/SetupTtsBackbone.cmake) and
// owns the whole flow — backbone load, tokenize/prefill, every per-model
// flow (continuous CFM, residual depth-AR, parallel-heads delay,
// streaming interleave, sequential text→audio, Chatterbox CFG dual-seq,
// and the self-contained Pocket-TTS FlowLM), sampling, EOS handling, and
// codes→PCM decode.  `examples/tts-cli` is a thin driver over this; a
// future server example can link the same library.
//
// This library is built only when CODEC_TTS_BACKBONE=ON.  When it is
// OFF, tts-cli falls back to the self-contained (no-backbone) FlowLM path
// exposed by `tts_runner_synthesize_selfcontained`.

#include <cstdint>
#include <string>
#include <vector>

// llama_logit_bias / llama_token — replicated inline so this header is
// includable by TUs that do NOT have llama.h on their include path (e.g.
// tts_runner_flow.cpp, which lives in the always-built codec_common library).
// Layouts are identical to llama.h's typedefs.  Gated on !LLAMA_H so that
// TUs that already included llama.h don't see a redeclaration.
#ifndef LLAMA_H
typedef struct llama_logit_bias {
    int32_t token;
    float   bias;
} llama_logit_bias;
typedef int32_t llama_token;
#endif  // !LLAMA_H

// Forward-declare llama_vocab so the detection helpers below can accept a
// pointer without requiring callers to include llama.h.
struct llama_vocab;

namespace codec_common {

// ─────────────────────────────────────────────────────────────────────
// Backbone-sampler constraint.  Exactly one field is populated for a
// constrained model; both empty = unconstrained.
//   grammar    — GBNF for structured models (OuteTTS / MOSS-TTSD)
//   logit_bias — id-level mask for flat direct-audio Type A models
// ─────────────────────────────────────────────────────────────────────
struct tts_constraint {
    std::string                   grammar;
    std::vector<llama_logit_bias> logit_bias;
};

// Mask every token OUTSIDE [offset, offset+count) ∪ {eos_id} to -INFINITY.
// eos_id < 0 = no sentinel.  offset < 0 → returns {} (Type A disabled).
std::vector<llama_logit_bias> build_flat_audio_mask(int32_t offset, int32_t count,
                                                    int32_t eos_id, int32_t n_vocab);

// ── OuteTTS V3 prompt builder ─────────────────────────────────────────────────
//
// Speaker profile: per-word text, acoustic features, and c1/c2 code frames.
// Populated by load_outetts_speaker from a JSON file (outetts 0.4.4 schema).

struct OutettsSpeakerWord {
    std::string          word;                // raw word text
    float                duration   = 0.0f;  // duration in seconds
    std::vector<int32_t> c1;                  // DAC codebook-0 indices, 0..1023
    std::vector<int32_t> c2;                  // DAC codebook-1 indices, 0..1023
    int32_t energy             = 0;           // quantized 0..100
    int32_t spectral_centroid  = 0;
    int32_t pitch              = 0;
};

struct OutettsSpeaker {
    std::string                      text;    // reference text for this speaker
    std::vector<OutettsSpeakerWord>  words;
};

// Parse an OuteTTS V3 speaker JSON file (outetts 0.4.4 schema) into *out.
// Returns true on success, false on parse failure (prints to stderr).
bool load_outetts_speaker(const std::string & json_path, OutettsSpeaker * out);

// Assemble the exact OuteTTS V3 prefill token sequence (outetts10-facts §Step 1):
//
//   <|im_start|>\n
//   <|text_start|>SPEAKER_TEXT. TARGET_TEXT<|text_end|>\n
//   <|audio_start|>\n
//   <speaker word blocks>\n
//   <|word_start|>
//
// Each speaker word block is one line:
//   <|word_start|>WORD<|features|><|t_D.DD|><|energy_N|><|spectral_centroid_N|><|pitch_N|>
//   <|code|><|c1_X|><|c2_X|>...<|word_end|>
//
// Code tokens are appended as raw llama_token ids (c1_base+code, c2_base+code).
// Valid code range: 0..1023.  Returns empty on any error (null vocab, empty text,
// out-of-range code).
std::vector<llama_token> build_outetts_v3_prompt(const llama_vocab      * vocab,
                                                 const std::string      & text,
                                                 const OutettsSpeaker   & speaker);

// ── NeuTTS detection ─────────────────────────────────────────────────────────
//
// Identifies a NeuTTS backbone from its vocab signature tokens.
// Returns true when the vocab contains the three marker pieces that together
// uniquely identify a NeuTTS backbone (nano or air).
//
// Both helpers are non-static so they can be called from the test binary
// (separate TU).  The llama_vocab* pointer type is forward-declared above;
// callers that also include llama.h see the same concrete type.

bool detect_neutts(const llama_vocab * vocab);

// Derives the audio-token range from the NeuTTS backbone vocab.
// On success: *offset = id of <|speech_0|>, *count = contiguous <|speech_N|>
// block length, *eos_id = id of <|SPEECH_GENERATION_END|>.  Returns false if
// any of the marker tokens are absent.
bool neutts_audio_token_range(const llama_vocab * vocab,
                              int32_t * offset,
                              int32_t * count,
                              int32_t * eos_id);

// Derive the multi-codebook audio-token ranges from an OuteTTS V3 backbone
// vocab.  Resolves from vocab pieces (robust across the 0.6B/1B backbones):
//   offsets[0] = <|c1_0|>, offsets[1] = <|c2_0|>, counts = {1024, 1024},
//   *sentinel = <|code|> (per-frame reset), *eos = <|audio_end|>.
// Returns false if any marker token is absent.
bool outetts_audio_token_ranges(const llama_vocab * vocab,
                                int32_t offsets[2],
                                int32_t counts[2],
                                int32_t * sentinel,
                                int32_t * eos);

// Build the full NeuTTS ICL prefill token sequence.
//
// Assembles the verbatim prompt string (phonemes mode):
//   "user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phonemes} {input_phonemes}"
//   "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
// tokenized with parse_special=true so the <|...|> markers resolve to their
// special ids, then appends one vocab id (offset + c) for each code in
// ref_codes.
//
// Guards: returns an empty vector if vocab is null, if llama_tokenize fails,
// or if any ref_code is out of [0, count).
std::vector<llama_token> build_neutts_prompt(
    const llama_vocab          * vocab,
    const std::string          & ref_text_phonemes,
    const std::string          & input_phonemes,
    const std::vector<int32_t> & ref_codes);

// ─────────────────────────────────────────────────────────────────────
// Params — everything the reference loop needs.  Unset optionals fall
// back to the model's training-time defaults (read from GGUF metadata).
// ─────────────────────────────────────────────────────────────────────
struct tts_runner_params {
    std::string codec_path;          // codec / codec_lm GGUF (required)
    std::string backbone_path;       // llama.cpp backbone GGUF; empty for
                                     // self-contained models (Pocket FlowLM)
    std::string text;                // synthesis target (required)
    std::string ref_audio_path;      // optional WAV for voice conditioning
    std::string ref_text;            // optional transcript of ref_audio (NeuTTS phonemization)

    int32_t  n_threads = 0;
    bool     use_gpu   = false;
    uint32_t seed      = 0xC0DEC1AB;
    int32_t  max_frames = 0;         // 0 → per-model default

    // Sampler overrides — has_* gates whether the value is applied.
    bool  has_temp = false;   float   temp  = 0.0f;
    bool  has_top_p = false;  float   top_p = 0.0f;
    bool  has_top_k = false;  int32_t top_k = 0;

    // Continuous-CFM (BlueMagpie) / FlowLM knobs.
    float   cfg       = 2.0f;
    int32_t timesteps = 10;
    int32_t min_len   = -1;          // -1 → model default

    // Chatterbox T3 knobs (has_* gates the override).
    bool has_cfg_weight = false;         float cfg_weight = 0.0f;
    bool has_min_p = false;              float min_p = 0.0f;
    bool has_rep_penalty = false;        float repetition_penalty = 0.0f;

    // Grammar (GBNF) constraint for the BACKBONE-logits sampler.  When
    // non-empty this GBNF is attached to the common_sampler that samples
    // backbone tokens (cb0-from-backbone / text warmup).  It never applies
    // to codec_lm audio-codebook heads (those are arbitrary float arrays
    // not tied to the backbone vocab).  A parse failure surfaces as a clean
    // error, not a crash.  Empty → the model's auto-grammar (if any) is used
    // instead; see codec_common::tts_auto_grammar.
    std::string grammar;
};

// ─────────────────────────────────────────────────────────────────────
// Result — PCM + stats.  The caller writes the WAV / marshals output.
// ─────────────────────────────────────────────────────────────────────
struct tts_runner_result {
    std::vector<float> pcm;          // interleaved if n_channels > 1
    int32_t sample_rate = 0;
    int32_t n_channels  = 1;

    int32_t     n_frames = 0;        // audio frames generated
    std::string stop_reason;         // "eos_code_c0" / "stop_head" / …

    std::string error;               // set + returns false on failure
};

// Self-contained (no-backbone) FlowLM synthesize (Pocket-TTS,
// codec.lm.kind="flow_lm").  Everything runs through the codec C API +
// codec_lm_flow_* helpers — no llama.cpp backbone.  Lives in the always-
// built codec_common library so the no-backbone tts-cli build keeps the
// FlowLM path.  Return value:
//   * 1  → handled the request (out populated on success; out->error set
//          on failure — check out->error.empty()).
//   * 0  → this is NOT a FlowLM model; the caller should fall through to
//          the backbone-driven flows.
// (Mirrors the old run_flow_lm_synthesize's -1 sentinel, remapped.)
int tts_runner_synthesize_selfcontained(const tts_runner_params & params,
                                        tts_runner_result * out);

// Run the full reference host loop and synthesize `params.text` → PCM.
// Returns true on success (result.pcm populated); false with
// result.error set on failure.  Tries the self-contained FlowLM path
// first, then chooses a backbone flow from the codec GGUF metadata
// (host_arch / codec.lm.kind) and loads the backbone.  Requires the
// isolated llama backbone (CODEC_TTS_BACKBONE=ON); only compiled into
// the codec_tts_runner library.
//
// Diagnostic progress is printed to stdout/stderr (the same lines
// examples/tts-cli emitted); hosts that want silence can capture them.
bool tts_runner_synthesize(const tts_runner_params & params,
                           tts_runner_result * out);

}  // namespace codec_common

#endif  // CODEC_TTS_RUNNER_H
