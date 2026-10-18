#ifndef CODEC_RUNTIME_GRAPH_H
#define CODEC_RUNTIME_GRAPH_H

#include "../codec_internal.h"

enum codec_graph_kind {
    CODEC_GRAPH_WT_DECODE = 1,
    CODEC_GRAPH_WT_ENCODE = 2,
    CODEC_GRAPH_DAC_DECODE = 3,
    CODEC_GRAPH_DAC_ENCODE = 4,
    CODEC_GRAPH_DAC_DECODE_LATENT = 5,
    CODEC_GRAPH_MIMI_DECODE = 6,
    CODEC_GRAPH_MIMI_ENCODE = 7,
    CODEC_GRAPH_Q3T_DECODE = 8,
    CODEC_GRAPH_SOPRANO_DECODE = 9,
    CODEC_GRAPH_NEMO_NANO_DECODE = 10,
    CODEC_GRAPH_NEMO_NANO_ENCODE = 11,
    CODEC_GRAPH_NEUCODEC_DECODE = 12,
    CODEC_GRAPH_NEUCODEC_ENCODE = 13,
    CODEC_GRAPH_CHATTERBOX_S3T_ENCODE = 14,
    CODEC_GRAPH_CHATTERBOX_S3G_DECODE = 17,
    CODEC_GRAPH_XCODEC2_DECODE = 18,
    CODEC_GRAPH_XCODEC2_ENCODE = 19,
    CODEC_GRAPH_SNAC_ENCODE    = 20,
    CODEC_GRAPH_SNAC_DECODE    = 21,
    CODEC_GRAPH_MOSS_AUDIO_ENCODE = 22,
    CODEC_GRAPH_MOSS_AUDIO_DECODE = 23,
    CODEC_GRAPH_XY_TOKENIZER_ENCODE = 24,
    CODEC_GRAPH_XY_TOKENIZER_DECODE = 25,

    // codec_lm graph kinds (auxiliary, per-state).
    CODEC_GRAPH_LM_PARALLEL_HEADS_LOGITS  = 26,
    CODEC_GRAPH_LM_PARALLEL_HEADS_COMPOSE = 27,

    CODEC_GRAPH_LM_RDA_C0_HEAD        = 28,
    CODEC_GRAPH_LM_RDA_DEPTH_STEP     = 29,
    CODEC_GRAPH_LM_RDA_COMPOSE        = 30,
    CODEC_GRAPH_LM_RDA_DEPTH_STEP_KV  = 31,  // incremental, llama.cpp-style KV cache
    CODEC_GRAPH_LM_RDA_TEXT_PROJ      = 34,  // Qwen3-TTS talker text_projection MLP
    CODEC_GRAPH_LM_COSYVOICE_TEXT     = 35,  // CosyVoice v1 text encoder + speaker affine rows

    CODEC_GRAPH_LM_SPEAKER_CHATTERBOX = 32,  // [VoiceEncoder +] cond_enc + perceiver
    CODEC_GRAPH_LM_SPEAKER_QWEN3_TTS  = 33,  // ECAPA-TDNN x-vector

    CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_DECODE = 50,  // VoxCPM/BlueMagpie continuous-latent VAE decode
    CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_ENCODE = 55,  // AudioVAE encoder (audio → latent mu)
    CODEC_GRAPH_BLUEMAGPIE_CFM             = 53,  // LocDiT + unrolled CFM Euler (codec_bluemagpie_cfm_eval, e2e test)
    CODEC_GRAPH_BLUEMAGPIE_CFM_STEP        = 56,  // continuous_latent_cfm adaptor — unified per-step graph
    CODEC_GRAPH_BLUEMAGPIE_CFM_PREFILL     = 57,  // continuous_latent_cfm adaptor — RALM full-prefix prefill

    CODEC_GRAPH_POCKET_MIMI_DECODE = 60,  // Pocket-TTS Mimi-variant: latent [32,T] → PCM 24 kHz
    CODEC_GRAPH_POCKET_MIMI_ENCODE = 61,  // Pocket-TTS Mimi-variant: PCM 24 kHz → latent [32,T]

    CODEC_GRAPH_FLOW_LM_PREFILL = 62,  // Pocket-TTS FlowLM: text+voice prefix → KV cache
    CODEC_GRAPH_FLOW_LM_STEP    = 63,  // Pocket-TTS FlowLM: one AR frame (transformer + LSD flow + EOS)
    CODEC_GRAPH_FLOW_LM_SPKPROJ = 64,  // Pocket-TTS FlowLM: mu [ldim,T] → speaker_proj rows

    CODEC_GRAPH_CHATTERBOX_S3G_SPEAKER = 65,
    CODEC_GRAPH_COSYVOICE_DECODE = 66,   // CosyVoice tokens → flow → HiFT PCM
    CODEC_GRAPH_COSYVOICE_SPEAKER = 67,  // reference PCM → CosyVoice speaker slots  // reference PCM → S3G speaker slots (S3T + mel + CAMPPlus [+ x-vector])
};

bool codec_runtime_init(codec_context * ctx, std::string * error);
void codec_runtime_free(codec_context * ctx);

bool codec_graph_cache_get_or_build(
    codec_context * ctx,
    codec_graph_cache_key key,
    codec_graph_build_fn build_fn,
    const void * user_data,
    size_t user_data_size,
    codec_graph_cache_entry ** out_entry,
    std::string * error);

bool codec_graph_compute(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    int32_t n_threads,
    std::string * error);

bool codec_graph_prepare_io(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    std::string * error);

void codec_graph_release(codec_context * ctx);

// A constant input of the graph being built (call from a build_fn only):
// creates `name` in `ctx_eval` with a copy of `data`, written automatically by
// codec_graph_prepare_io. Lets reusable blocks carry their own lookup tables.
// A second call with the same name returns the existing tensor.
ggml_tensor * codec_graph_const(
    ggml_context * ctx_eval,
    const char * name,
    ggml_type type,
    int n_dims,
    const int64_t * ne,
    const void * data);
// Run `node` on the CPU backend even when the graph is offloaded (call from a
// build_fn only). For precision-critical ops: ggml-vulkan's F32 matmul stages
// tiles in fp16 shared memory, too coarse for framed DFTs whose quiet bins
// come from cancelling large terms. Returns `node`.
ggml_tensor * codec_graph_pin_cpu(ggml_tensor * node);
// Whether the backend the current graph is built for runs `node` natively
// (true outside a graph build).  Ops in src/ops use it to fall back to a
// composition of more common ops instead of letting the scheduler split the
// graph to the CPU (ggml-opencl has no SIN / COS, ggml-vulkan no ELU, ...).
// CODEC_FORCE_COMPOSITE_OPS=1 answers false, to test those fallbacks on CPU.
bool codec_graph_backend_supports(const ggml_tensor * node);
// Whether the graph being built targets a non-CPU backend (false outside a
// build).  For layout / algorithm choices that are measurably faster on GPUs
// and slower on CPUs.
bool codec_graph_backend_is_gpu();
// Registry name of that backend's device ("CPU", "MTL" for Metal, "Vulkan",
// "CUDA", ...); "" outside a build.
const char * codec_graph_backend_reg_name();
// Slot memoizing the F32 cast of stored weight `w` within the graph being
// built (nullptr outside a build), so unrolled passes (20 CFM estimator calls)
// share one cast node instead of re-casting every bias per pass.
ggml_tensor ** codec_graph_weight_cast_slot(ggml_context * ctx_eval, const ggml_tensor * w);

// Pins every op created in `ctx_eval` while the scope is alive. Views of
// pinned tensors cannot cross backends, so a scope should end on a non-view
// tensor (ggml_cont or a compute op) that later ops consume.
struct codec_graph_cpu_scope {
    explicit codec_graph_cpu_scope(ggml_context * ctx_eval);
    ~codec_graph_cpu_scope();
    codec_graph_cpu_scope(const codec_graph_cpu_scope &) = delete;
    codec_graph_cpu_scope & operator=(const codec_graph_cpu_scope &) = delete;

    ggml_context * ctx;
    ggml_tensor * last_before;
};
ggml_tensor * codec_graph_const_f32(
    ggml_context * ctx_eval, const char * name, const std::vector<float> & data,
    int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1);
ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name);

// RAII cleanup for the per-call eval graph.  By default the guard releases the
// eval graph (freeing eval_ctx + clearing the galloc allocation) when it goes
// out of scope — this is the behaviour every model relies on.
//
// When `persist` is true the guard leaves the eval graph alive so the next
// public call can hit the consecutive-call fast path in
// codec_graph_cache_get_or_build (same entry + identical user_data → skip
// rebuild).  Any intervening call for a DIFFERENT entry takes the slow path,
// which releases the stale graph before rebuilding, so persisting is safe.
// Only callers that issue back-to-back identical graph calls (e.g. the
// BlueMagpie per-step CFM loop) should set persist=true.
struct codec_graph_eval_guard {
    explicit codec_graph_eval_guard(codec_context * ctx_, bool persist_ = false)
        : ctx(ctx_), persist(persist_) {}
    ~codec_graph_eval_guard() {
        if (!persist) {
            codec_graph_release(ctx);
        }
    }
    codec_context * ctx;
    bool persist;
};

#endif
