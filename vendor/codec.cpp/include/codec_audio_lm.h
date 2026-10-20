#ifndef CODEC_AUDIO_LM_H
#define CODEC_AUDIO_LM_H

#include "codec.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// codec_audio_lm — how to run an audio LM whose backbone is a llama.cpp
// model and whose audio side is a codec.cpp GGUF.  The analogue of
// llama.cpp's common + MTMD: this library describes the inference (what to
// feed the backbone, how to sample, when to stop) and bridges the backbone
// and the codec; the host owns every llama_* call (decode, sampler, KV).
//
// Shape of a host loop (TTS):
//
//   ctx = codec_audio_lm_init(&params)
//   codec_audio_lm_tts_set_voice(ctx, &voice)                  // once per request
//   n = codec_audio_lm_tts_split(ctx, text, 0)                 // optional segmentation
//   for each segment i (codec_audio_lm_tts_segment(ctx, i)):
//       codec_audio_lm_tts_prompt(ctx, &{segment_text, instruct, preset}, &vocab)
//       codec_audio_lm_get_completion(ctx, &c)
//       if c.flow == SELF_CONTAINED: codec_audio_lm_generate_self_contained(ctx)
//       else:
//           build a sampler from c.sampler (+ c.grammar), clear the KV cache
//           for each sequence s: decode chunk 0..c.n_chunks-1 of s in order
//           loop until c.max_frames audio frames (or the context is full):
//               tok = verdict.sample_backbone ? sample(backbone logits) : -1
//               codec_audio_lm_observe(ctx, toks[], hiddens[], &verdict)
//               STOP -> break; FEED_TOKEN -> decode (verdict.token >= 0 ? verdict.token : tok);
//               FEED_EMBD -> decode codec_audio_lm_next_embd(ctx, s) on every sequence s
//       codec_audio_lm_decode(ctx, &pcm)       // or stream_poll / stream_finish
//       gap = codec_audio_lm_tts_join(ctx, &pcm, i, n)   // then gap samples of silence
//
// Everything model-specific (prompt formats, conditioning, codec heads,
// their sampling, stop rules) lives behind these calls; the API carries no
// model names.  Text-to-speech is the first prompt builder
// (codec_audio_lm_tts_prompt); other tasks (audio-to-audio) add their own
// builders that produce the same completion description.
//
// Strings / arrays returned by the library are owned by the context and
// stay valid until the next call that refills them (or free).
// =====================================================================

struct codec_audio_lm;

// ── Lifecycle ────────────────────────────────────────────────────────

struct codec_audio_lm_params {
    const char * codec_path;                               // codec GGUF (required)
    const char * backbone_path;                            // backbone GGUF, read-only, for models
                                                           // that need its embedding table (may be NULL)
    const struct codec_model_kv_override * kv_overrides;   // codec metadata overrides (may be NULL)
    int32_t      n_kv_overrides;
    int32_t      n_threads;                                // 0 = default
    bool         use_gpu;
};

struct codec_audio_lm_params codec_audio_lm_default_params(void);

// NULL on failure; the reason goes to `err` (may be NULL).
struct codec_audio_lm * codec_audio_lm_init(const struct codec_audio_lm_params * params,
                                            char * err, size_t err_size);
void         codec_audio_lm_free      (struct codec_audio_lm * ctx);
const char * codec_audio_lm_last_error(const struct codec_audio_lm * ctx);

// Raw codec GGUF metadata value (stringified), or NULL.
const char * codec_audio_lm_meta_str(const struct codec_audio_lm * ctx, const char * key);

// ── Capabilities ─────────────────────────────────────────────────────

enum codec_audio_lm_modality {
    CODEC_AUDIO_LM_INPUT_TEXT   = 1u << 0,
    CODEC_AUDIO_LM_INPUT_AUDIO  = 1u << 1,
    CODEC_AUDIO_LM_OUTPUT_TEXT  = 1u << 2,
    CODEC_AUDIO_LM_OUTPUT_AUDIO = 1u << 3,
};

struct codec_audio_lm_caps {
    uint32_t modality;              // codec_audio_lm_modality bits
    bool     needs_backbone;        // false: self-contained (no llama.cpp model)
    bool     needs_backbone_path;   // params.backbone_path is required
    bool     text_needs_phonemes;   // input text must be phonemes (IPA)
    bool     takes_ref_audio;       // ref_pcm clones the voice
    bool     takes_ref_text;        // ref_text (the clip's transcript) is used
    bool     takes_instruct;        // instruct text is used
    bool     needs_voice;           // a voice / ref_pcm / speaker_file is required
    int32_t  sample_rate;           // output PCM rate
    int32_t  segment_bytes;         // segment size codec_audio_lm_tts_split uses by default
                                    // (0 = the model takes whole texts)
    const char * presets;           // sampling presets the model defines, ';'-separated ("" = none)
};

void codec_audio_lm_get_caps(const struct codec_audio_lm * ctx, struct codec_audio_lm_caps * out);

// Builtin voices (decoder speakers) of the codec, or those in a speaker
// file (path may be NULL).  Fills a per-context list; returns its size or -1.
int32_t codec_audio_lm_list_voices(struct codec_audio_lm * ctx, const char * speaker_file);
// Entry i of the last listing; any out-pointer may be NULL.
bool    codec_audio_lm_voice      (const struct codec_audio_lm * ctx, int32_t i,
                                   const char ** name, const char ** ref_text, bool * is_default);

// ── Backbone vocabulary access (host-provided) ───────────────────────
// Prompt builders look up the backbone's vocabulary (tokenize text, find
// special / audio tokens) through these; the library never decodes.  With
// llama.cpp: llama_tokenize / llama_token_to_piece / llama_vocab_n_tokens.

struct codec_audio_lm_vocab {
    void *  user_data;
    // Returns the number of tokens written (negative: -needed capacity).
    int32_t (*tokenize)(void * user_data, const char * text, int32_t len, int32_t * out, int32_t cap,
                        bool add_special, bool parse_special);
    // Piece text of `token` (special tokens rendered), written NUL-free;
    // returns its length (negative: -needed capacity).
    int32_t (*token_to_piece)(void * user_data, int32_t token, char * buf, int32_t cap);
    int32_t n_tokens;
    int32_t eos;                    // end-of-generation token (-1 = none)
    // Optional: true for every end-of-generation token (llama_vocab_is_eog);
    // NULL = only `eos`.
    bool    (*is_eog)(void * user_data, int32_t token);
};

// What the backbone's llama context needs, given its vocabulary (some models
// are recognized from it) and the longest generation the host allows (0 = the
// model default).  Call after loading the backbone model, before creating the
// context.  With llama.cpp: llama_context_params n_ctx / n_seq_max /
// embeddings (pooling none) / flash_attn_type.
struct codec_audio_lm_backbone_req {
    int32_t n_ctx;           // minimum context length
    int32_t n_ubatch;        // physical batch (0 = n_ctx)
    int32_t n_seq_max;       // sequences decoded side by side (2: guidance pair)
    bool    embeddings;      // the context returns hidden states (no pooling)
    bool    flash_attn;      // false: the model needs flash attention disabled
};

bool codec_audio_lm_backbone_requirements(struct codec_audio_lm * ctx, const struct codec_audio_lm_vocab * vocab,
                                          int32_t max_frames, struct codec_audio_lm_backbone_req * out);

// ── Text to speech ───────────────────────────────────────────────────

// Voice for the following tts_prompt calls (a request); conditioning that
// comes from reference audio is computed here, once.
struct codec_audio_lm_tts_voice {
    const char *  voice;            // builtin voice name (NULL = default)
    const char *  speaker_file;     // speaker GGUF (overrides voice)
    const float * ref_pcm;          // reference clip (mono), overrides voice / speaker_file
    int32_t       ref_n_samples;
    int32_t       ref_sample_rate;
    const char *  ref_text;         // transcript of ref_pcm
    const char *  save_speaker;     // write the decoder speaker in use here (may be NULL)
};

struct codec_audio_lm_tts_voice codec_audio_lm_default_tts_voice(void);

bool codec_audio_lm_tts_set_voice(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_voice * voice);

// Overrides of the model's defaults use "unset" sentinels: negative numbers
// (seed 0).  They apply to every sampler of the generation, the backbone's
// (completion.sampler) and the codec-side heads codec_audio_lm samples itself.
struct codec_audio_lm_tts_request {
    const char * text;          // required: one text (one segment)
    const char * instruct;      // instruction text, for models that take it (may be NULL)
    const char * preset;        // a sampling preset the model defines (caps.presets; NULL = default)
    uint32_t     seed;          // 0 = default
    float        temperature;
    float        top_p;
    int32_t      top_k;
    float        min_p;
    float        repetition_penalty;
    int32_t      min_frames;    // shortest generation before the stop token may end it
    int32_t      max_frames;    // longest generation (<= 0: the model's default)
    float        lm_guidance;   // classifier-free guidance weight of the backbone pair
    float        flow_guidance; // guidance of a flow / diffusion decoder
    int32_t      flow_steps;    // steps of a flow / diffusion decoder
};

struct codec_audio_lm_tts_request codec_audio_lm_default_tts_request(void);

// Prompt for one request.  `vocab` may be NULL for self-contained models.
// Resets the per-generation state; the voice stays.
bool codec_audio_lm_tts_prompt(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_request * req,
                               const struct codec_audio_lm_vocab * vocab);

// ── TTS segmentation ─────────────────────────────────────────────────
// Splits long text into segments generated one by one, at sentence and
// clause boundaries, each at most `max_bytes` (0 = caps.segment_bytes; a
// model that takes whole texts returns one segment).  Fills a per-context
// list; returns its size or -1.
int32_t      codec_audio_lm_tts_split  (struct codec_audio_lm * ctx, const char * text, int32_t max_bytes);
const char * codec_audio_lm_tts_segment(const struct codec_audio_lm * ctx, int32_t i);

// Prepare segment i of n's decoded audio for joining (the model's edge
// handling: trimming, fades), in place; returns the samples of silence to
// insert before segment i + 1 (0 for the last), or -1.
int32_t codec_audio_lm_tts_join(struct codec_audio_lm * ctx, struct codec_pcm_buffer * pcm, int32_t i, int32_t n);

// ── Completion description (all prompt builders) ─────────────────────

enum codec_audio_lm_chunk_type {
    CODEC_AUDIO_LM_CHUNK_TEXT   = 0,   // text: tokenize with the backbone vocab
    CODEC_AUDIO_LM_CHUNK_TOKENS = 1,   // backbone token ids
    CODEC_AUDIO_LM_CHUNK_EMBD   = 2,   // input embedding rows (n_rows x n_embd)
};

struct codec_audio_lm_chunk {
    enum codec_audio_lm_chunk_type type;
    const char *    text;              // TEXT
    bool            add_special;       // TEXT: tokenizer add_special
    bool            parse_special;     // TEXT: tokenizer parse_special
    const int32_t * tokens;            // TOKENS
    const float *   embd;              // EMBD, row-major
    int32_t         n;                 // tokens / rows
};

// How the backbone output turns into audio; also what observe() needs.
enum codec_audio_lm_flow {
    CODEC_AUDIO_LM_FLOW_TOKENS         = 0,  // audio codes are backbone tokens: observe(token)
    CODEC_AUDIO_LM_FLOW_CODEC_HEAD     = 1,  // codec heads read the backbone hidden: observe(hidden)
    CODEC_AUDIO_LM_FLOW_HIDDEN         = 2,  // backbone tokens drive, hiddens are the audio latent
    CODEC_AUDIO_LM_FLOW_SELF_CONTAINED = 3,  // no backbone: codec_audio_lm_generate_self_contained
};

// Backbone sampler settings, named after llama.cpp's common_params_sampling
// so a host maps them field by field.  "off" values disable a stage.
struct codec_audio_lm_sampler {
    uint32_t seed;
    float    temp;                 // <= 0: greedy
    int32_t  top_k;                // <= 0: off
    float    top_p;                // >= 1: off
    float    min_p;                // <= 0: off
    int32_t  penalty_last_n;       // penalty window; 0: whole history
    float    penalty_repeat;       // 1: off
    float    penalty_freq;         // 0: off
    float    penalty_present;      // 0: off
    bool     penalize_prompt;      // the prompt tokens count for the penalties
    // Tokens the penalties never touch (end / pause tokens a model must stay
    // free to emit).  Needs a llama.cpp with patches/penalty-exempt-llamacpp.patch;
    // a stock llama.cpp ignores the list (the tokens are then penalized too).
    const int32_t * penalty_exempt;
    int32_t         n_penalty_exempt;
    float    dry_multiplier;       // 0: off
    float    dry_base;
    int32_t  dry_allowed_length;
    int32_t  dry_penalty_last_n;   // -1: context size
    // Sampler order, as llama.cpp's --samplers ("penalties;dry;top_k;top_p;
    // min_p;temperature"); NULL / "" = llama.cpp's default order.
    const char * samplers;
};

struct codec_audio_lm_token_range { int32_t begin, end; };   // [begin, end)

struct codec_audio_lm_completion {
    enum codec_audio_lm_flow flow;
    int32_t n_seq;                         // backbone sequences (2: guidance pair)
    int32_t n_embd;                        // width of EMBD chunks / next_embd rows
    // Prompt of sequence s: chunks[s * n_chunks .. (s + 1) * n_chunks - 1],
    // decoded in order on sequence s (pad chunks have n == 0).
    const struct codec_audio_lm_chunk * chunks;
    int32_t n_chunks;
    // Backbone sampling (used when a verdict asks for a backbone token).
    struct codec_audio_lm_sampler sampler;
    const char * grammar;                  // GBNF ("" = none); equivalent to allowed below
    const struct codec_audio_lm_token_range * allowed;   // allowed backbone tokens
    int32_t n_allowed;                     // 0 = no restriction
    // Host requirements.
    bool    needs_embeddings;              // the backbone context returns hidden states
    bool    needs_prompt_hidden;           // pass every prompt position's hidden to codec_audio_lm_prefill
    bool    sample_first;                  // the first step samples a backbone token
    bool    streamable;                    // stream_poll works during this generation (else
                                           // only the whole generation decodes)
    int32_t max_frames;                    // audio frames to stop after (verdicts with is_audio)
    // Minimum length: while fewer than min_frames audio frames exist, a
    // sampled stop_token is replaced by one draw from a second sampler with
    // the same temp / top_k / top_p, no penalties, no grammar, restricted to
    // `allowed` minus stop_token (sample with the primary sampler first, so
    // its random stream advances every step).
    int32_t min_frames;
    int32_t stop_token;
};

// The completion for the prepared prompt.
bool codec_audio_lm_get_completion(struct codec_audio_lm * ctx, struct codec_audio_lm_completion * out);

// Every prompt position's hidden state of sequence 0, row-major
// [n_pos x n_embd] (only when needs_prompt_hidden).
bool codec_audio_lm_prefill(struct codec_audio_lm * ctx, const float * hiddens, int32_t n_pos);

// ── Per step ─────────────────────────────────────────────────────────

enum codec_audio_lm_action {
    CODEC_AUDIO_LM_FEED_TOKEN = 0,     // decode the sampled token next
    CODEC_AUDIO_LM_FEED_EMBD  = 1,     // decode codec_audio_lm_next_embd(ctx, s) on each sequence
    CODEC_AUDIO_LM_STOP       = 2,     // generation finished
};

struct codec_audio_lm_verdict {
    enum codec_audio_lm_action action;
    bool sample_backbone;              // sample a backbone token before the next observe
    bool is_audio;                     // this step produced audio (for progress / streaming)
    int32_t token;                     // FEED_TOKEN: the token to decode when the model picked it
                                       // itself (-1 = the token the host sampled)
};

// Report one step: `tokens` has n_seq sampled backbone tokens (or NULL when
// none was sampled), `hiddens` n_seq rows of the last position's hidden
// state, row-major (or NULL when the flow does not need them).
bool codec_audio_lm_observe(struct codec_audio_lm * ctx, const int32_t * tokens, const float * hiddens,
                            struct codec_audio_lm_verdict * out);

// Next input row of sequence `seq` after FEED_EMBD (n_embd floats).
const float * codec_audio_lm_next_embd(const struct codec_audio_lm * ctx, int32_t seq);

// Why the generation ended ("eos", "eos_code_c0", "stop_head", ...), or
// "max_frames" when the host's cap ended it before a STOP verdict.
const char * codec_audio_lm_stop_reason(const struct codec_audio_lm * ctx);

// Self-contained models: run the whole generation of the prepared prompt.
bool codec_audio_lm_generate_self_contained(struct codec_audio_lm * ctx);

// ── Audio out ────────────────────────────────────────────────────────

// Audio of the prompt's generation.  Free with codec_pcm_buffer_free.
bool codec_audio_lm_decode(struct codec_audio_lm * ctx, struct codec_pcm_buffer * out);

// Streaming.  Call codec_audio_lm_stream_poll after every observe whose
// verdict has is_audio; it returns (in `out`, possibly empty) the audio that
// became final since the last call, at most every `chunk_ms` of new audio.
// It re-decodes the generation from its first frame (causal decoders need the
// whole left context), holds back the right edge the decoder may still
// revise and crossfades the joins.  codec_audio_lm_stream_finish returns
// the rest; codec_audio_lm_decode is then not needed.
// Free each `out` with codec_pcm_buffer_free.
bool codec_audio_lm_stream_poll  (struct codec_audio_lm * ctx, int32_t chunk_ms, struct codec_pcm_buffer * out);
bool codec_audio_lm_stream_finish(struct codec_audio_lm * ctx, struct codec_pcm_buffer * out);

// Lower level: frames of the current generation that can be decoded, the codec
// hop (samples per frame) and the PCM of frames [frame_begin, frame_end).
int32_t codec_audio_lm_decodable_frames(const struct codec_audio_lm * ctx);
int32_t codec_audio_lm_hop(const struct codec_audio_lm * ctx);
bool    codec_audio_lm_decode_range(struct codec_audio_lm * ctx, int32_t frame_begin, int32_t frame_end,
                                    struct codec_pcm_buffer * out);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_AUDIO_LM_H
