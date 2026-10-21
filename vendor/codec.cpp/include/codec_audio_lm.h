#ifndef CODEC_AUDIO_LM_H
#define CODEC_AUDIO_LM_H

#include "codec.h"
#include "codec_speaker.h"

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
//   voice = codec_audio_lm_get_voice / create_voice / load_voice   // NULL = default
//   n = codec_audio_lm_tts_split(ctx, text, 0)                 // optional segmentation
//   for each segment i (codec_audio_lm_tts_segment(ctx, i)):
//       req = {segment_text, voice, instruct}
//       codec_audio_lm_tts_format(ctx, &req, &vocab, &f)      // the prompt as text (+ constraints)
//       codec_audio_lm_tts_build(ctx, &req, f.prompt, &vocab, &c)   // its chunks (MTMD-style)
//       codec_audio_lm_set_seed(ctx, seed)                    // the model's own noise
//       if c.flow == SELF_CONTAINED: codec_audio_lm_generate_self_contained(ctx)
//       else:
//           build the host's llama.cpp sampler (+ c.grammar / c.allowed / c.penalty_exempt)
//           and one per c.head_groups[g], clear the KV cache
//           for each sequence s: decode chunk 0..c.n_chunks-1 of s in order
//           loop until c.max_frames audio frames (or the context is full):
//               tok = verdict.sample_backbone ? sample(backbone logits) : -1
//               codec_audio_lm_observe(ctx, toks[], hiddens[], &verdict)
//               while SAMPLE_HEAD: id = sample(group[v.head_group], head_logits(ctx), v.n_vocab);
//                                  codec_audio_lm_head_token(ctx, id, &verdict)
//               STOP -> break; FEED_TOKEN -> decode (verdict.token >= 0 ? verdict.token : tok);
//               FEED_EMBD -> decode codec_audio_lm_next_embd(ctx, s) on every sequence s
//       codec_audio_lm_decode(ctx, &pcm)       // or stream_poll / stream_finish
//       gap = codec_audio_lm_tts_join(ctx, &pcm, i, n)   // then gap samples of silence
//
// Everything model-specific (prompt formats, conditioning, codec heads,
// their sampling, stop rules) lives behind these calls; the API carries no
// model names.  Text-to-speech is the first prompt builder
// (codec_audio_lm_tts_format / tts_build); other tasks (audio-to-audio) add
// their own builders that produce the same completion description.
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
};

void codec_audio_lm_get_caps(const struct codec_audio_lm * ctx, struct codec_audio_lm_caps * out);

// ── Voices ───────────────────────────────────────────────────────────
// A voice is a struct codec_speaker (codec.h) holding all of this model's
// voice conditioning, whatever it conditions: the backbone prompt (x-vector,
// reference tokens / latents / transcript), the decoder, or both.  Builtin
// voices belong to the model; created and loaded voices to the caller
// (codec_speaker_free).  A voice is read-only once made, so one voice can
// serve any number of requests.

// Builtin voices of the codec, or those in a speaker file (path may be
// NULL).  Fills a per-context list; returns its size or -1.
int32_t codec_audio_lm_list_voices(struct codec_audio_lm * ctx, const char * speaker_file);
// Entry i of the last listing; any out-pointer may be NULL.
bool    codec_audio_lm_voice      (const struct codec_audio_lm * ctx, int32_t i,
                                   const char ** name, const char ** ref_text, bool * is_default);
// A builtin voice (NULL name = the model's default, set by `<arch>.spk.default`);
// NULL when absent.
const struct codec_speaker * codec_audio_lm_get_voice(struct codec_audio_lm * ctx, const char * name);
// Clone a voice from a reference clip (mono F32, any rate) and its transcript
// (`ref_text`, may be NULL; models that prompt with it use it). caps.takes_ref_audio
// says whether the model can.
struct codec_speaker * codec_audio_lm_create_voice(struct codec_audio_lm * ctx, const float * pcm, int32_t n_samples,
                                                   int32_t sample_rate, const char * ref_text);
// Voice `name` (NULL = the file's default) from a speaker GGUF made by
// codec_audio_lm_save_voice for this model; files of another model are refused.
struct codec_speaker * codec_audio_lm_load_voice(struct codec_audio_lm * ctx, const char * path, const char * name);
bool codec_audio_lm_save_voice(struct codec_audio_lm * ctx, const struct codec_speaker * voice, const char * path,
                               const char * name);

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

// Sampling is the host's: every token of a generation, the backbone's and the
// codec-side heads' (SAMPLE_HEAD verdicts), is drawn by the host with
// llama.cpp's sampler and its settings (the backbone GGUF's general.sampling.*
// recommends them).  codec_audio_lm only suggests constraints (grammar,
// allowed ranges, penalty_exempt, head groups) and masks a head's logits.
#define CODEC_AUDIO_LM_DEFAULT_SEED 0xFFFFFFFFu   // random (also llama.cpp's LLAMA_DEFAULT_SEED)

struct codec_audio_lm_tts_request {
    const char * text;          // required: one text (one segment)
    // The voice (NULL = the model's default).  Borrowed: keep it alive until
    // the generation is decoded (decode / stream_finish) or the next prompt.
    const struct codec_speaker * voice;
    const char * instruct;      // instruction text, for models that take it (may be NULL)
    // Generation length: at least min_frames audio frames before the model's
    // end (< 0: the model's own minimum), at most max_frames (<= 0: the
    // model's default).
    int32_t      min_frames;
    int32_t      max_frames;
    // Decoder graph settings; < 0: the model's (GGUF metadata).
    float        lm_guidance;   // classifier-free guidance weight of the backbone pair
    float        flow_guidance; // guidance of a flow / diffusion decoder
    int32_t      flow_steps;    // steps of a flow / diffusion decoder
};

struct codec_audio_lm_tts_request codec_audio_lm_default_tts_request(void);

// Seed of the model's own randomness in the prepared generation: the init
// noise of a codec-side flow / CFM head (Pocket-TTS, BlueMagpie), not token
// sampling.  Call after tts_build, before the first prefill / observe (or
// generate_self_contained); not called (or CODEC_AUDIO_LM_DEFAULT_SEED) = random.
// Hosts pass the seed the user gave the completion.
bool codec_audio_lm_set_seed(struct codec_audio_lm * ctx, uint32_t seed);

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

struct codec_audio_lm_token_range { int32_t begin, end; };   // [begin, end)

// Codec-side token heads (SAMPLE_HEAD verdicts) come in groups; build one host
// sampler per group (one random stream and penalty history; seed group g with
// seed + 1 + g so groups draw independently).  All groups use the same
// llama.cpp settings as the backbone; a group differs only as below.
struct codec_audio_lm_head_group {
    bool            penalties;          // false: this group's sampler takes no penalties
    const int32_t * history;            // ids to accept into the penalty history first
    int32_t         n_history;
    const int32_t * penalty_exempt;     // ids the penalties never touch
    int32_t         n_penalty_exempt;
};

struct codec_audio_lm_completion {
    enum codec_audio_lm_flow flow;
    int32_t n_seq;                         // backbone sequences (2: guidance pair)
    int32_t n_embd;                        // width of EMBD chunks / next_embd rows
    // Prompt of sequence s: chunks[s * n_chunks .. (s + 1) * n_chunks - 1],
    // decoded in order on sequence s (pad chunks have n == 0).
    const struct codec_audio_lm_chunk * chunks;
    int32_t n_chunks;
    // Backbone token sampling (the host's sampler), suggested constraints:
    // GBNF ("" = none) for the tokens the model may emit and its minimum length
    // (min_frames frames before the end token), tokens the penalties should
    // never touch, and the allowed ranges below.
    const char * grammar;
    const int32_t * penalty_exempt;
    int32_t n_penalty_exempt;
    const struct codec_audio_lm_token_range * allowed;   // allowed backbone tokens
    int32_t n_allowed;                     // 0 = no restriction
    // Host requirements.
    bool    needs_embeddings;              // the backbone context returns hidden states
    bool    needs_prompt_hidden;           // pass every prompt position's hidden to codec_audio_lm_prefill
    bool    sample_first;                  // the first step samples a backbone token
    bool    streamable;                    // stream_poll works during this generation (else
                                           // only the whole generation decodes)
    int32_t max_frames;                    // audio frames to stop after (verdicts with frame_done)
    // Codec-side head groups (n_head_groups = 0: no SAMPLE_HEAD verdicts).
    const struct codec_audio_lm_head_group * head_groups;
    int32_t n_head_groups;
};

// The completion for the built prompt.
bool codec_audio_lm_get_completion(struct codec_audio_lm * ctx, struct codec_audio_lm_completion * out);

// ── Text to speech: the prompt ───────────────────────────────────────

// The prompt is text, as llama.cpp's MTMD has it: the model's template with
// the request filled in (roles, instruction, the normalized text, text-form
// voices such as reference tokens written as their vocab pieces), tokenized by
// the backbone vocab with parse_special, and markers where the backbone takes
// input the vocab cannot spell:
//   <__voice__>         the voice's embedding-form conditioning (x-vector,
//                       reference frames, speaker conditioning); a model may
//                       take it at more than one place
//   <__codec:NAME__>    a model row (an embedding row or a token without a
//                       piece), or <__codec:text__>: the text to speak starts
//                       here (models that tokenize it apart from what precedes)
// docs/codec_common_api.md lists each model's prompt.
#define CODEC_AUDIO_LM_MARKER_VOICE "<__voice__>"

// What codec_audio_lm_tts_format returns: the prompt and what the host's
// completion needs before the prompt is built (the same values the
// completion of tts_build carries).
struct codec_audio_lm_format {
    const char * prompt;
    enum codec_audio_lm_flow flow;
    int32_t n_seq;                         // backbone sequences (2: guidance pair)
    const char * grammar;                  // GBNF for the backbone's tokens ("" = none)
    const int32_t * penalty_exempt;
    int32_t n_penalty_exempt;
    const struct codec_audio_lm_token_range * allowed;
    int32_t n_allowed;
    bool    needs_embeddings;
    bool    needs_prompt_hidden;
    bool    sample_first;
    bool    streamable;
    int32_t max_frames;
};

// Format one request (text in, text out; no generation state).  `vocab` may
// be NULL for self-contained models.  The result is valid until the next
// format.
bool codec_audio_lm_tts_format(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_request * req,
                               const struct codec_audio_lm_vocab * vocab, struct codec_audio_lm_format * out);

// Build a formatted prompt (format's, or the host's edit of it) for `req`, the
// request it was formatted from: its chunks, sampling plan and the generation
// state.  Everything the generation needs comes from the prompt, the voice and
// the decoder settings; req->text / instruct may be NULL (a host that kept only
// the prompt), and then the completion leaves out the constraints that depend
// on the text (the format's grammar; take that one from the format).  Resets the per-generation state: a context holds one built prompt.
// `out` is what codec_audio_lm_get_completion returns.
bool codec_audio_lm_tts_build(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_request * req,
                              const char * prompt, const struct codec_audio_lm_vocab * vocab,
                              struct codec_audio_lm_completion * out);

// Every prompt position's hidden state of sequence 0, row-major
// [n_pos x n_embd] (only when needs_prompt_hidden).
bool codec_audio_lm_prefill(struct codec_audio_lm * ctx, const float * hiddens, int32_t n_pos);

// ── Per step ─────────────────────────────────────────────────────────

enum codec_audio_lm_action {
    CODEC_AUDIO_LM_FEED_TOKEN = 0,     // decode the sampled token next
    CODEC_AUDIO_LM_FEED_EMBD  = 1,     // decode codec_audio_lm_next_embd(ctx, s) on each sequence
    CODEC_AUDIO_LM_STOP       = 2,     // generation finished
    // Sample a codec-side head: draw an id from codec_audio_lm_head_logits with
    // group `head_group`'s sampler, report it with codec_audio_lm_head_token.
    CODEC_AUDIO_LM_SAMPLE_HEAD = 3,
};

struct codec_audio_lm_verdict {
    enum codec_audio_lm_action action;
    bool sample_backbone;              // sample a backbone token before the next observe
    bool is_audio;                     // this step's token belongs to the audio (codes, and the markup
                                       // between the model's audio start and end); the host emits the
                                       // fed-back tokens of the other steps as text
    bool frame_done;                   // this step completed an audio frame (progress / max_frames / streaming)
    int32_t token;                     // FEED_TOKEN: the token to decode when the model picked it
                                       // itself (-1 = the token the host sampled)
    int32_t head_group;                // SAMPLE_HEAD: the group to sample with
    int32_t n_vocab;                   // SAMPLE_HEAD: the head's logits (ids 0..n_vocab-1)
};

// Report one step: `tokens` has n_seq sampled backbone tokens (or NULL when
// none was sampled), `hiddens` n_seq rows of the last position's hidden
// state, row-major (or NULL when the flow does not need them).
bool codec_audio_lm_observe(struct codec_audio_lm * ctx, const int32_t * tokens, const float * hiddens,
                            struct codec_audio_lm_verdict * out);

// After a SAMPLE_HEAD verdict: the head's logits (verdict.n_vocab floats, the
// model's constraints already applied as -inf, guidance already combined),
// valid until the next call.
const float * codec_audio_lm_head_logits(const struct codec_audio_lm * ctx);
// The id the host drew from them; fills the next verdict (another SAMPLE_HEAD,
// FEED_* or STOP), as observe does.
bool codec_audio_lm_head_token(struct codec_audio_lm * ctx, int32_t id, struct codec_audio_lm_verdict * out);

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

// A generation's decoder input, to keep and decode later (out of order, more
// than once).  Self-contained: decoding needs nothing of the generation but
// this and the decoder speaker.  Frame-major [n_frames x width].
enum codec_audio_lm_data_kind {
    CODEC_AUDIO_LM_DATA_CODES   = 0,   // codec quantizer codes; width = codebooks per frame
    CODEC_AUDIO_LM_DATA_LATENTS = 1,   // continuous latents; width = latent dim
};
struct codec_audio_lm_generation {
    enum codec_audio_lm_data_kind kind;
    int32_t         n_frames;
    int32_t         width;
    const int32_t * codes;     // CODES, else NULL
    const float   * latents;   // LATENTS, else NULL
};
// The current generation's decoder input: what codec_audio_lm_decode decodes.
// The arrays belong to ctx and stay valid until the next call of this function
// or codec_audio_lm_free; copy them to keep the generation.
bool codec_audio_lm_get_generation(struct codec_audio_lm * ctx, struct codec_audio_lm_generation * out);
// Decode a generation of this model (from get_generation, possibly of an
// earlier prompt) with the voice it was generated with (NULL = the default
// voice; some models' decoders take no voice).  The in-progress generation is
// left untouched.  Free `out` with codec_pcm_buffer_free.
bool codec_audio_lm_decode_generation(struct codec_audio_lm * ctx, const struct codec_audio_lm_generation * gen,
                                      const struct codec_speaker * voice, struct codec_pcm_buffer * out);

// Streaming.  Call codec_audio_lm_stream_poll after every observe whose
// verdict has frame_done; it returns (in `out`, possibly empty) the audio that
// became final since the last call, at most every `chunk_ms` of new audio.
// It re-decodes the generation from its first frame (causal decoders need the
// whole left context), holds back the right edge the decoder may still
// revise and crossfades the joins.  codec_audio_lm_stream_finish returns
// the rest; codec_audio_lm_decode is then not needed.
// Free each `out` with codec_pcm_buffer_free.
bool codec_audio_lm_stream_poll  (struct codec_audio_lm * ctx, int32_t chunk_ms, struct codec_pcm_buffer * out);
bool codec_audio_lm_stream_finish(struct codec_audio_lm * ctx, struct codec_pcm_buffer * out);

// Frames of the current generation that can be decoded.
int32_t codec_audio_lm_decodable_frames(const struct codec_audio_lm * ctx);

#ifdef __cplusplus
}
#endif

#endif  // CODEC_AUDIO_LM_H
