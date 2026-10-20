#ifndef CODEC_H
#define CODEC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CODEC_DEFAULT_SEED (int32_t)0xFFFFFFFFu

enum codec_arch {
    CODEC_ARCH_UNKNOWN = 0,
    CODEC_ARCH_WAVTOKENIZER_LARGE = 1,
    CODEC_ARCH_DAC = 2,
    CODEC_ARCH_MIMI = 3,
    CODEC_ARCH_QWEN3_TTS_TOKENIZER = 4,
    CODEC_ARCH_SOPRANO = 5,
    CODEC_ARCH_NEMO_NANO_CODEC = 6,
    CODEC_ARCH_NEUCODEC = 7,
    CODEC_ARCH_DISTILL_NEUCODEC = 8,
    CODEC_ARCH_CHATTERBOX_S3T = 9,
    CODEC_ARCH_CHATTERBOX_S3G = 10,
    CODEC_ARCH_XCODEC2 = 11,
    CODEC_ARCH_SNAC = 12,
    CODEC_ARCH_MOSS_AUDIO = 13,
    CODEC_ARCH_XY_TOKENIZER = 14,
    CODEC_ARCH_BLUEMAGPIE_AUDIOVAE = 15,
    CODEC_ARCH_POCKET_MIMI = 16,
    CODEC_ARCH_COSYVOICE = 17,
};

enum codec_status {
    CODEC_STATUS_SUCCESS = 0,
    CODEC_STATUS_INVALID_ARG = 1,
    CODEC_STATUS_INVALID_STATE = 2,
    CODEC_STATUS_IO_ERROR = 3,
    CODEC_STATUS_NOT_SUPPORTED = 4,
    CODEC_STATUS_INTERNAL_ERROR = 5,
};

enum codec_pcm_type {
    CODEC_PCM_TYPE_F32 = 0,
    CODEC_PCM_TYPE_I16 = 1,
};

enum codec_batch_mode {
    CODEC_BATCH_MODE_CODES = 0,
    CODEC_BATCH_MODE_LATENT = 1,
};

struct codec_model;
struct codec_context;

struct codec_model_params {
    bool use_gpu;
    int32_t n_threads;
};

// GGUF metadata override applied at load time, mirroring llama.cpp's
// llama_model_kv_override. Lets published GGUFs that predate a metadata key
// opt into it without re-conversion (e.g. chatterbox_s3g.flow.n_timesteps).
enum codec_model_kv_override_type {
    CODEC_KV_OVERRIDE_TYPE_INT,
    CODEC_KV_OVERRIDE_TYPE_FLOAT,
    CODEC_KV_OVERRIDE_TYPE_BOOL,
    CODEC_KV_OVERRIDE_TYPE_STR,
};

struct codec_model_kv_override {
    enum codec_model_kv_override_type tag;
    char key[128];
    union {
        int64_t val_i64;
        double  val_f64;
        bool    val_bool;
        char    val_str[128];
    };
};

struct codec_context_params {
    int32_t seed;
};

struct codec_encode_params {
    int32_t n_threads;
    int32_t frame_size;
    int32_t hop_size;
    int32_t n_q;
};

struct codec_speaker;

struct codec_decode_params {
    int32_t n_threads;
    int32_t n_q;
    // Reference speaker for decoders conditioned on one (e.g. Chatterbox-S3G).
    // NULL = the model's default speaker. Passing a speaker to a model without
    // speaker support returns CODEC_STATUS_NOT_SUPPORTED.
    const struct codec_speaker * speaker;
};

struct codec_audio {
    const void * data;
    int32_t n_samples;
    int32_t sample_rate;
    int32_t n_channels;
    enum codec_pcm_type pcm_type;
};

struct codec_token_buffer {
    int32_t * data;
    int32_t n_tokens;
    int32_t n_frames;
    int32_t n_q;
    int32_t codebook_size;
    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_pcm_buffer {
    float * data;
    int32_t n_samples;
    int32_t sample_rate;
    int32_t n_channels;
};

struct codec_latent_buffer {
    float * data;
    int32_t latent_dim;
    int32_t n_frames;
    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_batch {
    int32_t n_seq;
    int32_t n_seq_alloc;
    int32_t n_seq_max;

    int32_t * seq_id;
    int32_t * n_frames;
    int32_t * n_q;

    enum codec_batch_mode mode;

    int32_t * codes;
    int32_t codes_size;
    int32_t codes_used;

    float * latent;
    int32_t latent_dim;
    int32_t latent_size;
    int32_t latent_used;

    int32_t * codes_offset;
    int32_t * latent_offset;

    int32_t sample_rate;
    int32_t hop_size;
};

struct codec_gguf_kv {
    const char * key;
    const char * value;
};

struct codec_gguf_metadata {
    struct codec_gguf_kv * items;
    size_t n_items;
};

struct codec_model_params codec_model_default_params(void);
struct codec_context_params codec_context_default_params(void);
struct codec_encode_params codec_encode_default_params(void);
struct codec_decode_params codec_decode_default_params(void);

struct codec_model * codec_model_load_from_file(const char * path_model, struct codec_model_params params);
// kv_overrides: NULL, or an array terminated by an entry with key[0] == 0.
struct codec_model * codec_model_load_from_file_ext(
    const char * path_model,
    struct codec_model_params params,
    const struct codec_model_kv_override * kv_overrides);
void codec_model_free(struct codec_model * model);

struct codec_context * codec_init_from_model(struct codec_model * model, struct codec_context_params params);
void codec_free(struct codec_context * ctx);

enum codec_status codec_encode(struct codec_context * ctx, const struct codec_audio * audio, struct codec_token_buffer * out_tokens, struct codec_encode_params params);
enum codec_status codec_encode_latent(
    struct codec_context * ctx,
    const struct codec_audio * audio,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params);
enum codec_status codec_decode(struct codec_context * ctx, const struct codec_token_buffer * tokens, struct codec_pcm_buffer * out_pcm, struct codec_decode_params params);
enum codec_status codec_decode_quantized_representation(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
struct codec_batch codec_batch_init_codes(int32_t n_seq_alloc, int32_t codes_alloc_total, int32_t n_seq_max);
struct codec_batch codec_batch_init_latent(int32_t n_seq_alloc, int32_t latent_dim, int32_t latent_alloc_total, int32_t n_seq_max);
void codec_batch_free(struct codec_batch batch);
int32_t codec_batch_add_seq_codes(struct codec_batch * batch, int32_t seq_id, int32_t n_frames, int32_t n_q, const int32_t * codes);
int32_t codec_batch_add_seq_latent(struct codec_batch * batch, int32_t seq_id, int32_t n_frames, const float * latent, int32_t latent_dim);
enum codec_status codec_decode_batch(
    struct codec_context * ctx,
    const struct codec_batch * batch,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params);
void codec_token_buffer_free(struct codec_token_buffer * tokens);
void codec_pcm_buffer_free(struct codec_pcm_buffer * pcm);
void codec_latent_buffer_free(struct codec_latent_buffer * latent);

const char * codec_get_last_error(const struct codec_context * ctx);

enum codec_arch codec_model_arch(const struct codec_model * model);
const char * codec_model_name(const struct codec_model * model);
int32_t codec_model_n_tensors(const struct codec_model * model);

int32_t codec_model_sample_rate(const struct codec_model * model);
bool codec_model_has_encoder(const struct codec_model * model);
bool codec_model_has_decoder(const struct codec_model * model);
int32_t codec_model_n_q(const struct codec_model * model);
int32_t codec_model_codebook_size(const struct codec_model * model);
int32_t codec_model_hop_size(const struct codec_model * model);
int32_t codec_model_n_fft(const struct codec_model * model);
int32_t codec_model_win_length(const struct codec_model * model);
int32_t codec_model_n_mels(const struct codec_model * model);
int32_t codec_model_latent_dim(const struct codec_model * model);

const struct codec_gguf_metadata * codec_model_metadata(const struct codec_model * model);
void codec_metadata_free(struct codec_gguf_metadata * meta);

const char * codec_arch_name(enum codec_arch arch);

// ─── Speakers ───────────────────────────────────────────────────────
// A speaker is the reference conditioning a model derives from a voice
// sample: a set of named tensors ("slots") whose layout each architecture
// defines. Builtin speakers live in the model GGUF as `spk.<name>.<slot>`
// tensors, with `<arch>.spk.default` naming the default one.
//
// Speaker files written by codec_speaker_save carry general.type="speaker",
// general.architecture, `<arch>.spk.format_version` and
// `<arch>.spk.encoder_hash`; codec_speaker_load rejects a file whose
// architecture, format version or encoder hash differs from the model's.

// Builtin speakers. Owned by the model (valid for its lifetime; do not free).
int32_t codec_model_n_speakers(const struct codec_model * model);
const char * codec_model_speaker_name(const struct codec_model * model, int32_t i);
// name = NULL returns the default speaker. Returns NULL when not found.
const struct codec_speaker * codec_get_speaker(const struct codec_model * model, const char * name);

// True when codec_speaker_from_audio is available for this model.
bool codec_model_can_speaker_from_audio(const struct codec_model * model);
// Derive a speaker from reference audio (any sample rate; mono or
// interleaved multi-channel is downmixed). Caller frees with codec_speaker_free.
enum codec_status codec_speaker_from_audio(
    struct codec_context * ctx,
    const struct codec_audio * ref,
    struct codec_speaker ** out);

enum codec_status codec_speaker_save(
    const struct codec_model * model,
    const struct codec_speaker * speaker,
    const char * name,
    const char * path);
// name = NULL loads the file's default speaker. Caller frees with codec_speaker_free.
enum codec_status codec_speaker_load(
    const struct codec_model * model,
    const char * path,
    const char * name,
    struct codec_speaker ** out);
void codec_speaker_free(struct codec_speaker * speaker);

// Speaker files may hold several speakers. Opening checks the file and every
// speaker in it against the model, like codec_speaker_load. Speakers returned
// by codec_speaker_file_get_speaker are owned by the file handle.
struct codec_speaker_file;
enum codec_status codec_speaker_file_open(
    const struct codec_model * model,
    const char * path,
    struct codec_speaker_file ** out);
int32_t codec_speaker_file_n_speakers(const struct codec_speaker_file * file);
const char * codec_speaker_file_speaker_name(const struct codec_speaker_file * file, int32_t i);
// name = NULL returns the file's default speaker. Returns NULL when not found.
const struct codec_speaker * codec_speaker_file_get_speaker(
    const struct codec_speaker_file * file, const char * name);
void codec_speaker_file_free(struct codec_speaker_file * file);

const char * codec_speaker_name(const struct codec_speaker * speaker);
// Transcript of the reference audio, or NULL when unknown.
const char * codec_speaker_ref_text(const struct codec_speaker * speaker);
// Attach the transcript of the reference audio (saved with the speaker).
enum codec_status codec_speaker_set_ref_text(struct codec_speaker * speaker, const char * text);

enum codec_speaker_slot_type {
    CODEC_SPEAKER_SLOT_F32 = 0,
    CODEC_SPEAKER_SLOT_I32 = 1,
};

// Read one slot (model-specific name, e.g. "lm_cond"). `ne` receives 4 dims
// (unused = 1, ggml order: ne[0] fastest). `*data` stays valid for the
// speaker's lifetime. Returns false when the slot is absent.
bool codec_speaker_get_slot(
    const struct codec_speaker * speaker,
    const char * slot,
    enum codec_speaker_slot_type * type,
    int64_t ne[4],
    const void ** data);

// Logging.  Everything codec.cpp reports (and the log of the ggml backends
// libcodec runs on) goes to one callback, as llama.cpp's llama_log_set; the
// levels have ggml_log_level's values, so a ggml / llama.cpp log callback can
// be passed as is.  The default prints INFO / DEBUG to stdout and WARN /
// ERROR to stderr.  `callback` NULL restores it.  Not thread-safe against
// concurrent logging: set it before using the library.
enum codec_log_level {
    CODEC_LOG_LEVEL_NONE  = 0,
    CODEC_LOG_LEVEL_INFO  = 1,
    CODEC_LOG_LEVEL_WARN  = 2,
    CODEC_LOG_LEVEL_ERROR = 3,
    CODEC_LOG_LEVEL_DEBUG = 4,
    CODEC_LOG_LEVEL_CONT  = 5,  // continues the previous message
};

typedef void (*codec_log_callback)(enum codec_log_level level, const char * text, void * user_data);

void codec_log_set(codec_log_callback callback, void * user_data);

// Emit through the current callback (printf-style); for hosts and the
// layers above libcodec (codec_audio_lm, codec_tts) so one callback sees all.
void codec_log(enum codec_log_level level, const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#endif // CODEC_H
