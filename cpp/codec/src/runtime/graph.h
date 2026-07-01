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

    CODEC_GRAPH_LM_SPEAKER_CHATTERBOX = 32,  // cond_enc + perceiver

    CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_DECODE = 50,  // VoxCPM/BlueMagpie continuous-latent VAE decode
    CODEC_GRAPH_BLUEMAGPIE_AUDIOVAE_ENCODE = 55,  // AudioVAE encoder (audio → latent mu)
    CODEC_GRAPH_BLUEMAGPIE_CFM             = 53,  // LocDiT + unrolled CFM Euler (codec_bluemagpie_cfm_eval, e2e test)
    CODEC_GRAPH_BLUEMAGPIE_CFM_STEP        = 56,  // continuous_latent_cfm adaptor — unified per-step graph
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
lm_ggml_tensor * codec_graph_get_tensor(codec_context * ctx, codec_graph_cache_entry * entry, const char * name);

struct codec_graph_eval_guard {
    explicit codec_graph_eval_guard(codec_context * ctx_) : ctx(ctx_) {}
    ~codec_graph_eval_guard() {
        codec_graph_release(ctx);
    }
    codec_context * ctx;
};

#endif
