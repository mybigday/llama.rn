#pragma once

#include "llama.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-vocab.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <vector>

// available models
// TODO: this enum does not follow the enum naming convention
enum llm_type {
    MODEL_UNKNOWN,
    MODEL_14M,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_60M,
    MODEL_70M,
    MODEL_80M,
    MODEL_109M,
    MODEL_137M,
    MODEL_160M,
    MODEL_220M,
    MODEL_250M,
    MODEL_270M,
    MODEL_335M,
    MODEL_410M,
    MODEL_450M,
    MODEL_770M,
    MODEL_780M,
    MODEL_0_5B,
    MODEL_1B,
    MODEL_1_3B,
    MODEL_1_4B,
    MODEL_1_5B,
    MODEL_1_6B,
    MODEL_2B,
    MODEL_2_8B,
    MODEL_3B,
    MODEL_4B,
    MODEL_6B,
    MODEL_6_9B,
    MODEL_7B,
    MODEL_8B,
    MODEL_9B,
    MODEL_11B,
    MODEL_12B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_16B,
    MODEL_20B,
    MODEL_30B,
    MODEL_32B,
    MODEL_34B,
    MODEL_35B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_236B,
    MODEL_314B,
    MODEL_671B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_A1_7B,
    MODEL_A2_7B,
    MODEL_8x7B,
    MODEL_8x22B,
    MODEL_16x12B,
    MODEL_10B_128x3_66B,
    MODEL_57B_A14B,
    MODEL_27B,
};

struct llama_layer_posnet {
    // resnet
    struct lm_ggml_tensor * norm1   = nullptr;
    struct lm_ggml_tensor * norm1_b = nullptr;

    struct lm_ggml_tensor * conv1   = nullptr;
    struct lm_ggml_tensor * conv1_b = nullptr;

    struct lm_ggml_tensor * norm2   = nullptr;
    struct lm_ggml_tensor * norm2_b = nullptr;

    struct lm_ggml_tensor * conv2   = nullptr;
    struct lm_ggml_tensor * conv2_b = nullptr;

    // attention
    struct lm_ggml_tensor * attn_norm   = nullptr;
    struct lm_ggml_tensor * attn_norm_b = nullptr;

    struct lm_ggml_tensor * attn_q   = nullptr;
    struct lm_ggml_tensor * attn_q_b = nullptr;

    struct lm_ggml_tensor * attn_k   = nullptr;
    struct lm_ggml_tensor * attn_k_b = nullptr;

    struct lm_ggml_tensor * attn_v   = nullptr;
    struct lm_ggml_tensor * attn_v_b = nullptr;

    struct lm_ggml_tensor * attn_o   = nullptr;
    struct lm_ggml_tensor * attn_o_b = nullptr;

    // normalize
    struct lm_ggml_tensor * norm   = nullptr;
    struct lm_ggml_tensor * norm_b = nullptr;
};

struct llama_layer_convnext {
    struct lm_ggml_tensor * dw   = nullptr;
    struct lm_ggml_tensor * dw_b = nullptr;

    struct lm_ggml_tensor * norm   = nullptr;
    struct lm_ggml_tensor * norm_b = nullptr;

    struct lm_ggml_tensor * pw1   = nullptr;
    struct lm_ggml_tensor * pw1_b = nullptr;

    struct lm_ggml_tensor * pw2   = nullptr;
    struct lm_ggml_tensor * pw2_b = nullptr;

    struct lm_ggml_tensor * gamma = nullptr;
};

struct llama_layer {
    // normalization
    struct lm_ggml_tensor * attn_norm       = nullptr;
    struct lm_ggml_tensor * attn_norm_b     = nullptr;
    struct lm_ggml_tensor * attn_norm_2     = nullptr;
    struct lm_ggml_tensor * attn_norm_2_b   = nullptr;
    struct lm_ggml_tensor * attn_q_norm     = nullptr;
    struct lm_ggml_tensor * attn_q_norm_b   = nullptr;
    struct lm_ggml_tensor * attn_k_norm     = nullptr;
    struct lm_ggml_tensor * attn_k_norm_b   = nullptr;
    struct lm_ggml_tensor * attn_out_norm   = nullptr;
    struct lm_ggml_tensor * attn_out_norm_b = nullptr;
    struct lm_ggml_tensor * attn_q_a_norm   = nullptr;
    struct lm_ggml_tensor * attn_kv_a_norm  = nullptr;
    struct lm_ggml_tensor * attn_sub_norm   = nullptr;
    struct lm_ggml_tensor * attn_post_norm  = nullptr;
    struct lm_ggml_tensor * ffn_sub_norm    = nullptr;
    struct lm_ggml_tensor * attn_norm_cross = nullptr;
    struct lm_ggml_tensor * attn_norm_enc   = nullptr;

    // attention
    struct lm_ggml_tensor * wq        = nullptr;
    struct lm_ggml_tensor * wk        = nullptr;
    struct lm_ggml_tensor * wv        = nullptr;
    struct lm_ggml_tensor * wo        = nullptr;
    struct lm_ggml_tensor * wqkv      = nullptr;
    struct lm_ggml_tensor * wq_a      = nullptr;
    struct lm_ggml_tensor * wq_b      = nullptr;
    struct lm_ggml_tensor * wkv_a_mqa = nullptr;
    struct lm_ggml_tensor * wkv_b     = nullptr;
    struct lm_ggml_tensor * wq_cross  = nullptr;
    struct lm_ggml_tensor * wk_cross  = nullptr;
    struct lm_ggml_tensor * wv_cross  = nullptr;
    struct lm_ggml_tensor * wo_cross  = nullptr;
    struct lm_ggml_tensor * wq_enc    = nullptr;
    struct lm_ggml_tensor * wk_enc    = nullptr;
    struct lm_ggml_tensor * wv_enc    = nullptr;
    struct lm_ggml_tensor * wo_enc    = nullptr;

    // attention bias
    struct lm_ggml_tensor * bq   = nullptr;
    struct lm_ggml_tensor * bk   = nullptr;
    struct lm_ggml_tensor * bv   = nullptr;
    struct lm_ggml_tensor * bo   = nullptr;
    struct lm_ggml_tensor * bqkv = nullptr;

    // relative position bias
    struct lm_ggml_tensor * attn_rel_b       = nullptr;
    struct lm_ggml_tensor * attn_rel_b_enc   = nullptr;
    struct lm_ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct lm_ggml_tensor * ffn_norm         = nullptr;
    struct lm_ggml_tensor * ffn_norm_b       = nullptr;
    struct lm_ggml_tensor * ffn_post_norm    = nullptr;
    struct lm_ggml_tensor * layer_out_norm   = nullptr;
    struct lm_ggml_tensor * layer_out_norm_b = nullptr;
    struct lm_ggml_tensor * ffn_norm_exps    = nullptr;
    struct lm_ggml_tensor * ffn_norm_enc     = nullptr;

    // ff
    struct lm_ggml_tensor * ffn_gate     = nullptr; // w1
    struct lm_ggml_tensor * ffn_down     = nullptr; // w2
    struct lm_ggml_tensor * ffn_up       = nullptr; // w3
    struct lm_ggml_tensor * ffn_gate_enc = nullptr;
    struct lm_ggml_tensor * ffn_down_enc = nullptr;
    struct lm_ggml_tensor * ffn_up_enc   = nullptr;

    // ff MoE
    struct lm_ggml_tensor * ffn_gate_inp  = nullptr;
    struct lm_ggml_tensor * ffn_gate_exps = nullptr;
    struct lm_ggml_tensor * ffn_down_exps = nullptr;
    struct lm_ggml_tensor * ffn_up_exps   = nullptr;

    // ff shared expert (shexp)
    struct lm_ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct lm_ggml_tensor * ffn_gate_shexp     = nullptr;
    struct lm_ggml_tensor * ffn_down_shexp     = nullptr;
    struct lm_ggml_tensor * ffn_up_shexp       = nullptr;

    // ff bias
    struct lm_ggml_tensor * ffn_gate_b = nullptr;
    struct lm_ggml_tensor * ffn_down_b = nullptr; // b2
    struct lm_ggml_tensor * ffn_up_b   = nullptr; // b3
    struct lm_ggml_tensor * ffn_act    = nullptr;
    struct lm_ggml_tensor * ffn_exp_probs_b = nullptr;

    // mamba proj
    struct lm_ggml_tensor * ssm_in  = nullptr;
    struct lm_ggml_tensor * ssm_x   = nullptr;
    struct lm_ggml_tensor * ssm_dt  = nullptr;
    struct lm_ggml_tensor * ssm_out = nullptr;

    // mamba
    struct lm_ggml_tensor * ssm_conv1d = nullptr;
    struct lm_ggml_tensor * ssm_a      = nullptr;
    struct lm_ggml_tensor * ssm_d      = nullptr;

    // mamba bias
    struct lm_ggml_tensor * ssm_conv1d_b = nullptr;
    struct lm_ggml_tensor * ssm_dt_b     = nullptr;

    // rwkv
    struct lm_ggml_tensor * time_mix_w1         = nullptr;
    struct lm_ggml_tensor * time_mix_w2         = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_x     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_w     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_k     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_v     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_r     = nullptr;
    struct lm_ggml_tensor * time_mix_lerp_g     = nullptr;

    struct lm_ggml_tensor * time_mix_first      = nullptr;
    struct lm_ggml_tensor * time_mix_decay      = nullptr;
    struct lm_ggml_tensor * time_mix_decay_w1   = nullptr;
    struct lm_ggml_tensor * time_mix_decay_w2   = nullptr;
    struct lm_ggml_tensor * time_mix_key        = nullptr;
    struct lm_ggml_tensor * time_mix_value      = nullptr;
    struct lm_ggml_tensor * time_mix_receptance = nullptr;
    struct lm_ggml_tensor * time_mix_gate       = nullptr;

    struct lm_ggml_tensor * time_mix_ln     = nullptr;
    struct lm_ggml_tensor * time_mix_ln_b   = nullptr;
    struct lm_ggml_tensor * time_mix_output = nullptr;

    struct lm_ggml_tensor * channel_mix_lerp_k = nullptr;
    struct lm_ggml_tensor * channel_mix_lerp_r = nullptr;

    struct lm_ggml_tensor * channel_mix_key        = nullptr;
    struct lm_ggml_tensor * channel_mix_receptance = nullptr;
    struct lm_ggml_tensor * channel_mix_value      = nullptr;

    // long rope factors
    struct lm_ggml_tensor * rope_long  = nullptr;
    struct lm_ggml_tensor * rope_short = nullptr;
    struct lm_ggml_tensor * rope_freqs = nullptr;

    // bitnet scale
    struct lm_ggml_tensor * wq_scale       = nullptr;
    struct lm_ggml_tensor * wk_scale       = nullptr;
    struct lm_ggml_tensor * wv_scale       = nullptr;
    struct lm_ggml_tensor * wo_scale       = nullptr;
    struct lm_ggml_tensor * ffn_gate_scale = nullptr;
    struct lm_ggml_tensor * ffn_up_scale   = nullptr;
    struct lm_ggml_tensor * ffn_down_scale = nullptr;

    struct llama_layer_posnet posnet;

    struct llama_layer_convnext convnext;
};

struct llama_model {
    llm_type type = MODEL_UNKNOWN;
    llm_arch arch = LLM_ARCH_UNKNOWN;

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct lm_ggml_tensor * tok_embd   = nullptr;
    struct lm_ggml_tensor * type_embd  = nullptr;
    struct lm_ggml_tensor * pos_embd   = nullptr;
    struct lm_ggml_tensor * tok_norm   = nullptr;
    struct lm_ggml_tensor * tok_norm_b = nullptr;

    struct lm_ggml_tensor * output_norm     = nullptr;
    struct lm_ggml_tensor * output_norm_b   = nullptr;
    struct lm_ggml_tensor * output          = nullptr;
    struct lm_ggml_tensor * output_b        = nullptr;
    struct lm_ggml_tensor * output_norm_enc = nullptr;

    // classifier
    struct lm_ggml_tensor * cls       = nullptr;
    struct lm_ggml_tensor * cls_b     = nullptr;
    struct lm_ggml_tensor * cls_out   = nullptr;
    struct lm_ggml_tensor * cls_out_b = nullptr;

    struct lm_ggml_tensor * conv1d   = nullptr;
    struct lm_ggml_tensor * conv1d_b = nullptr;

    std::vector<llama_layer> layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> lm_gguf_kv;

    llama_split_mode split_mode;
    int main_gpu;
    int n_gpu_layers;

    std::vector<std::string> rpc_servers;

    // list of devices used in this model
    std::vector<lm_ggml_backend_dev_t> devices;


    // lists of buffer types used for each layer
    using buft_list_t = std::vector<std::pair<lm_ggml_backend_dev_t, lm_ggml_backend_buffer_type_t>>;
    buft_list_t cpu_buft_list;
    std::map<lm_ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        lm_ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    // contexts where the model tensors metadata is stored
    std::vector<lm_ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<lm_ggml_backend_buffer_ptr> bufs;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct lm_ggml_tensor *>> tensors_by_name;

    int64_t t_load_us  = 0;
    int64_t t_start_us = 0;

    // total number of parameters in the model
    uint64_t n_elements = 0;

    // total size of all the tensors in the model in bytes
    size_t  n_bytes     = 0;
};

const char * llm_type_name(llm_type type);

std::string llama_model_arch_name (const llama_model & model);
std::string llama_model_type_name (const llama_model & model);
std::string llama_model_ftype_name(const llama_model & model);

// used by llama_adapter_cvec
lm_ggml_backend_buffer_type_t llama_model_select_buft(const llama_model & model, int il);

// used by llama_adapter_lora
struct lm_ggml_tensor * llama_model_get_tensor(const struct llama_model & model, const char * name);

size_t llama_model_max_nodes(const llama_model & model);

struct llama_model_loader;

// TODO: become llama_model methods
void llm_load_stats     (llama_model_loader & ml, llama_model & model);
void llm_load_arch      (llama_model_loader & ml, llama_model & model);
void llm_load_hparams   (llama_model_loader & ml, llama_model & model);
void llm_load_vocab     (llama_model_loader & ml, llama_model & model);
void llm_load_print_meta(llama_model_loader & ml, llama_model & model);