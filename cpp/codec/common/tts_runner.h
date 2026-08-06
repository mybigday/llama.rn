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

namespace codec_common {

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
