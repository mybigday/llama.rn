#ifndef CODEC_INTERNAL_H
#define CODEC_INTERNAL_H

#include "codec.h"

#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>

#include <string>
#include <unordered_map>
#include <cstdint>
#include <vector>

// One named tensor of a speaker (`spk.<name>.<slot>` in GGUF). Data is host
// memory; decoders upload what they need as graph inputs.
struct codec_speaker_slot {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    std::vector<uint8_t> data;

    int64_t nelements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

struct codec_speaker {
    std::string arch;          // general.architecture of the producing model
    std::string encoder_hash;  // <arch>.spk.encoder_hash of the producing model
    std::string name;
    std::string ref_text;
    std::vector<codec_speaker_slot> slots;

    const codec_speaker_slot * slot(const char * slot_name) const {
        for (const auto & s : slots) {
            if (s.name == slot_name) return &s;
        }
        return nullptr;
    }
};

struct codec_model {
    struct gguf_context * gguf;
    struct ggml_context * weights;
    // name -> tensor of `weights`; ggml_get_tensor is a linear strcmp scan and
    // graph builders look weights up tens of thousands of times per build.
    std::unordered_map<std::string, ggml_tensor *> weight_index;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    // Persistent CPU worker pool shared by the model's contexts (sized to the
    // largest thread count asked for). Without one, ggml-cpu builds without
    // OpenMP create and join a fresh pool on every graph compute.
    struct ggml_threadpool * cpu_threadpool = nullptr;
    int32_t cpu_threadpool_n = 0;

    struct codec_gguf_metadata metadata;

    enum codec_arch arch;
    std::string name;
    int32_t n_tensors;
    bool use_gpu;
    int32_t n_threads;

    int32_t sample_rate;
    int32_t encode_sample_rate;
    bool has_encoder;
    bool has_decoder;
    int32_t hop_size;
    int32_t n_q;
    int32_t codebook_size;
    int32_t n_fft;
    int32_t win_length;
    int32_t n_mels;
    int32_t latent_dim;
    // 0 means "single-channel only" (legacy default).  Multi-channel codecs
    // (e.g. MOSS-Audio-Tokenizer in stereo mode) set this to the expected
    // number of input channels; codec_encode_impl validates against it.
    int32_t expected_channels = 1;
    const struct codec_model_vtable * vtable = nullptr;
    void * impl = nullptr;

    std::string arch_name;              // general.architecture as written in the GGUF
    std::string speaker_encoder_hash;   // <arch>.spk.encoder_hash ("" when absent)
    std::vector<codec_speaker> speakers;
    int32_t default_speaker = -1;
    bool can_speaker_from_audio = false;  // set by arch init when the encoders are bundled
};

// Graph cache key with named fields to avoid ambiguous p0..p3 usage.
// Only a subset of fields are used depending on kind; helper constructors must zero the unused fields.
struct codec_graph_cache_key {
    int32_t kind = 0;

    // Common decode shapes
    int32_t n_frames = 0;   // token frames / latent frames
    int32_t n_q = 0;        // number of quantizers used (codes input)
    int32_t hop = 0;        // hop size / stride used by the model

    // Encode shapes
    int32_t n_in = 0;       // input PCM samples

    // Latent decode shapes
    int32_t latent_dim = 0; // DAC latent dimension
};

typedef bool (*codec_graph_build_fn)(ggml_context * ctx_eval, void * user_data, ggml_tensor ** out);

struct codec_graph_cache_entry {
    codec_graph_cache_key key;
    size_t required_mem_size = 0;
    codec_graph_build_fn build_fn = nullptr;
    std::vector<uint8_t> build_user_data;
    int32_t last_graph_size = 0; // exact ggml graph capacity: max(n_nodes, n_leafs)
    int32_t last_sched_graph_size = 0; // exact scheduler base size: n_nodes + n_leafs
};

// Host data for a constant graph input (DFT basis, resampler kernel, ...),
// registered while a graph is built and written after it is allocated.
struct codec_graph_const_input {
    ggml_tensor * tensor = nullptr;
    std::vector<uint8_t> data;
};

struct codec_context {
    struct codec_model * model;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    struct codec_context_params params;
    int32_t n_threads = 1; // threads for the current public call (its params.n_threads, else the model's)
    std::string last_error;
    std::vector<codec_graph_cache_entry> graph_cache;
    void * eval_arena_buf = nullptr;
    size_t eval_arena_size = 0;
    ggml_context * eval_ctx = nullptr;
    ggml_cgraph * eval_graph = nullptr;
    ggml_tensor * eval_output = nullptr;
    codec_graph_cache_entry * eval_entry = nullptr;
    std::vector<codec_graph_const_input> eval_consts; // constants of the current eval graph
    std::vector<ggml_tensor *> eval_cpu_nodes;         // nodes pinned to the CPU backend
    std::unordered_map<const ggml_tensor *, ggml_tensor *> eval_weight_casts; // weight -> its F32 cast in this graph
    bool eval_graph_allocated = false; // sched_alloc_graph done for current eval_ctx
    int32_t sched_reserved_graph_size = 0;
};

struct codec_model_vtable {
    enum codec_arch arch;
    const char * name;
    void * (*create_impl)();
    void (*destroy_impl)(void *);
    enum codec_status (*init)(struct codec_model * model);
    size_t (*graph_size)(
        const struct codec_model * model,
        const struct codec_graph_cache_key * key,
        const void * user_data,
        struct ggml_tensor * out);
    enum codec_status (*encode)(
        struct codec_context * ctx,
        const std::vector<float> & pcm,
        struct codec_token_buffer * out_tokens,
        struct codec_latent_buffer * out_latent,
        struct codec_encode_params params);
    enum codec_status (*decode)(
        struct codec_context * ctx,
        const struct codec_token_buffer * tokens,
        struct codec_pcm_buffer * out_pcm,
        struct codec_decode_params params);
    enum codec_status (*decode_latent)(
        struct codec_context * ctx,
        const float * quantized_representation,
        int32_t latent_dim,
        int32_t n_frames,
        struct codec_pcm_buffer * out_pcm,
        struct codec_decode_params params);
    // Speaker support. `speaker_validate` checks that the required slots exist
    // with valid shapes (nullptr = the arch takes no speaker). `speaker_from_audio`
    // derives a speaker from mono PCM at `sample_rate` (nullptr = unsupported).
    bool (*speaker_validate)(
        const struct codec_model * model,
        const struct codec_speaker * speaker,
        std::string * err);
    enum codec_status (*speaker_from_audio)(
        struct codec_context * ctx,
        const std::vector<float> & pcm,
        int32_t sample_rate,
        struct codec_speaker * out);
};

// Loads `spk.<name>.<slot>` tensors + `<arch>.spk.*` metadata from the model
// GGUF into model->speakers (after arch init, which may add legacy speakers).
bool codec_model_load_speakers(codec_model * model, std::string * err);
// Appends a slot holding a copy of `data`.
void codec_speaker_add_slot(
    codec_speaker * spk, const char * slot, enum ggml_type type,
    const int64_t * ne, int n_dims, const void * data);

enum codec_arch codec_arch_from_string(const std::string & arch);

enum codec_status codec_model_init_arch(codec_model * model);

void codec_context_set_error(codec_context * ctx, const std::string & error);
bool codec_prepare_mono_f32(const codec_audio * audio, std::vector<float> * mono, std::string * error);
void codec_token_buffer_reset(codec_token_buffer * tokens);
void codec_pcm_buffer_reset(codec_pcm_buffer * pcm);
void codec_latent_buffer_reset(codec_latent_buffer * latent);

const float * codec_tensor_data_f32(const ggml_tensor * t);
int64_t codec_ne(const ggml_tensor * t, int dim);
bool codec_tensor_as_vec_f32(const ggml_tensor * t, std::vector<float> * out);

char * codec_strdup(const char * s);
void codec_metadata_add(codec_gguf_metadata * meta, const char * key, const std::string & value);
std::string codec_gguf_value_to_string(gguf_context * gf, int key_id);
void codec_collect_gguf_metadata(codec_model * model);
int32_t codec_read_i32_kv(gguf_context * gf, const char * key, int32_t fallback);
int32_t codec_read_i32_kv_any(gguf_context * gf, const char * const * keys, size_t n_keys, int32_t fallback);
void codec_read_i32_array_kv(gguf_context * gf, const char * key, int32_t * dst, int32_t dst_n);
void codec_read_i32_array_kv_vec(gguf_context * gf, const char * key, std::vector<int32_t> * dst);
void codec_read_f32_array_kv(gguf_context * gf, const char * key, float * dst, int32_t dst_n);
float codec_read_f32_kv(gguf_context * gf, const char * key, float fallback);
bool codec_read_bool_kv(gguf_context * gf, const char * key, bool fallback);
std::string codec_read_str_kv(gguf_context * gf, const char * key, const char * fallback);
int codec_count_tensors_with_prefix(const codec_model * model, const char * prefix);
int32_t codec_infer_n_q_from_tensor_names(const codec_model * model);

bool codec_safe_add_i32(int32_t a, int32_t b, int32_t * out);
bool codec_safe_mul_i32(int32_t a, int32_t b, int32_t * out);
size_t codec_graph_size_exact(
    const struct codec_model * model,
    const struct codec_graph_cache_key * key,
    const void * user_data,
    struct ggml_tensor * out);

#endif
