#pragma once

#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"

#include <array>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cmath>

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t projection_dim = 0;
    int32_t n_head = 0;
    int32_t n_layer = 0;
    // idefics3
    int32_t image_longest_edge = 0;
    int32_t image_min_pixels = -1;
    int32_t image_max_pixels = -1;
    int32_t n_merge = 0; // number of patch merges **per-side**

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM everytime they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps = 1e-6;
    float rope_theta = 0.0;

    std::vector<clip_image_size> image_res_candidates; // for llava-uhd style models
    int32_t image_crop_resolution;
    std::unordered_set<int32_t> vision_feature_layer;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    std::unordered_set<int32_t> wa_layer_indexes; // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t proj_stack_factor = 0; // ultravox

    // audio-to-mel preprocessor params
    int32_t audio_chunk_len   = -1; // in seconds
    int32_t audio_sample_rate = -1;
    int32_t audio_n_fft       = -1;
    int32_t audio_window_len  = -1;
    int32_t audio_hop_len     = -1;

    // legacy
    bool has_llava_projector = false;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number

    // custom value provided by user, can be undefined if not set
    int32_t custom_image_min_tokens = -1;
    int32_t custom_image_max_tokens = -1;

    void set_limit_image_tokens(int n_tokens_min, int n_tokens_max) {
        const int cur_merge = n_merge == 0 ? 1 : n_merge;
        const int patch_area = patch_size * patch_size * cur_merge * cur_merge;
        image_min_pixels = (custom_image_min_tokens > 0 ? custom_image_min_tokens : n_tokens_min) * patch_area;
        image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;
        warmup_image_size = static_cast<int>(std::sqrt(image_max_pixels));
    }

    void set_warmup_n_tokens(int n_tokens) {
        int n_tok_per_side = static_cast<int>(std::sqrt(n_tokens));
        LM_GGML_ASSERT(n_tok_per_side * n_tok_per_side == n_tokens && "n_tokens must be n*n");
        const int cur_merge = n_merge == 0 ? 1 : n_merge;
        warmup_image_size = n_tok_per_side * patch_size * cur_merge;
        // TODO: support warmup size for custom token numbers
    }
};

struct clip_layer {
    // attention
    lm_ggml_tensor * k_w = nullptr;
    lm_ggml_tensor * k_b = nullptr;
    lm_ggml_tensor * q_w = nullptr;
    lm_ggml_tensor * q_b = nullptr;
    lm_ggml_tensor * v_w = nullptr;
    lm_ggml_tensor * v_b = nullptr;
    lm_ggml_tensor * qkv_w = nullptr;
    lm_ggml_tensor * qkv_b = nullptr;

    lm_ggml_tensor * o_w = nullptr;
    lm_ggml_tensor * o_b = nullptr;

    lm_ggml_tensor * k_norm = nullptr;
    lm_ggml_tensor * q_norm = nullptr;

    // layernorm 1
    lm_ggml_tensor * ln_1_w = nullptr;
    lm_ggml_tensor * ln_1_b = nullptr;

    lm_ggml_tensor * ff_up_w = nullptr;
    lm_ggml_tensor * ff_up_b = nullptr;
    lm_ggml_tensor * ff_gate_w = nullptr;
    lm_ggml_tensor * ff_gate_b = nullptr;
    lm_ggml_tensor * ff_down_w = nullptr;
    lm_ggml_tensor * ff_down_b = nullptr;

    // layernorm 2
    lm_ggml_tensor * ln_2_w = nullptr;
    lm_ggml_tensor * ln_2_b = nullptr;

    // layer scale (no bias)
    lm_ggml_tensor * ls_1_w = nullptr;
    lm_ggml_tensor * ls_2_w = nullptr;

    // qwen3vl deepstack merger
    lm_ggml_tensor * deepstack_norm_w = nullptr;
    lm_ggml_tensor * deepstack_norm_b = nullptr;
    lm_ggml_tensor * deepstack_fc1_w = nullptr;
    lm_ggml_tensor * deepstack_fc1_b = nullptr;
    lm_ggml_tensor * deepstack_fc2_w = nullptr;
    lm_ggml_tensor * deepstack_fc2_b = nullptr;

    // lfm2
    lm_ggml_tensor * ff_norm_w     = nullptr;
    lm_ggml_tensor * ff_norm_b     = nullptr;
    lm_ggml_tensor * ff_norm_1_w   = nullptr;
    lm_ggml_tensor * ff_norm_1_b   = nullptr;
    lm_ggml_tensor * ff_up_1_w     = nullptr;
    lm_ggml_tensor * ff_up_1_b     = nullptr;
    lm_ggml_tensor * ff_down_1_w   = nullptr;
    lm_ggml_tensor * ff_down_1_b   = nullptr;
    lm_ggml_tensor * pos_bias_u    = nullptr;
    lm_ggml_tensor * pos_bias_v    = nullptr;
    lm_ggml_tensor * norm_conv_w   = nullptr;
    lm_ggml_tensor * norm_conv_b   = nullptr;
    lm_ggml_tensor * linear_pos_w  = nullptr;

    lm_ggml_tensor * conv_norm_w   = nullptr;
    lm_ggml_tensor * conv_norm_b   = nullptr;
    lm_ggml_tensor * conv_dw_w     = nullptr;
    lm_ggml_tensor * conv_dw_b     = nullptr;
    lm_ggml_tensor * conv_pw1_w    = nullptr;
    lm_ggml_tensor * conv_pw1_b    = nullptr;
    lm_ggml_tensor * conv_pw2_w    = nullptr;
    lm_ggml_tensor * conv_pw2_b    = nullptr;

    bool has_deepstack() const {
        return deepstack_fc1_w != nullptr;
    }
};

// Expanded MobileNetV5 block structure for Gemma3n vision encoder
struct mobilenetv5_block {
    // Stage 0 (Edge Residual)
    lm_ggml_tensor * s0_conv_exp_w = nullptr;
    lm_ggml_tensor * s0_bn1_w      = nullptr;
    lm_ggml_tensor * s0_conv_pwl_w = nullptr;
    lm_ggml_tensor * s0_bn2_w      = nullptr;

    // Stage 1+ (Universal Inverted Residual)
    lm_ggml_tensor * dw_start_w    = nullptr;
    lm_ggml_tensor * dw_start_bn_w = nullptr;

    lm_ggml_tensor * pw_exp_w      = nullptr;
    lm_ggml_tensor * pw_exp_bn_w   = nullptr;

    lm_ggml_tensor * dw_mid_w      = nullptr;
    lm_ggml_tensor * dw_mid_bn_w   = nullptr;

    lm_ggml_tensor * pw_proj_w     = nullptr;
    lm_ggml_tensor * pw_proj_bn_w  = nullptr;

    lm_ggml_tensor * layer_scale_w = nullptr;

    // Attention (MQA) components
    lm_ggml_tensor * attn_q_w = nullptr;
    lm_ggml_tensor * attn_k_w = nullptr;
    lm_ggml_tensor * attn_v_w = nullptr;
    lm_ggml_tensor * attn_o_w = nullptr;

    // Optional downsampling/norm in attention
    lm_ggml_tensor * attn_k_dw_w   = nullptr;
    lm_ggml_tensor * attn_k_norm_w = nullptr;
    lm_ggml_tensor * attn_v_dw_w   = nullptr;
    lm_ggml_tensor * attn_v_norm_w = nullptr;

    // Block norm (often present in attention blocks)
    lm_ggml_tensor * attn_norm_w   = nullptr;
};

struct clip_model {
    clip_modality modality = CLIP_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_MLP;
    clip_hparams hparams;

    // embeddings
    lm_ggml_tensor * class_embedding = nullptr;
    lm_ggml_tensor * patch_embeddings_0 = nullptr;
    lm_ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
    lm_ggml_tensor * patch_bias = nullptr;
    lm_ggml_tensor * position_embeddings = nullptr;
    lm_ggml_tensor * norm_embd_w = nullptr;
    lm_ggml_tensor * norm_embd_b = nullptr;

    lm_ggml_tensor * pre_ln_w = nullptr;
    lm_ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    int32_t n_deepstack_layers = 0; // used by Qwen3-VL, calculated from clip_layer

    lm_ggml_tensor * post_ln_w;
    lm_ggml_tensor * post_ln_b;

    lm_ggml_tensor * projection; // TODO: rename it to fc (fully connected layer)
    lm_ggml_tensor * mm_fc_w;
    lm_ggml_tensor * mm_fc_b;
    lm_ggml_tensor * mm_ffn_up_w = nullptr;
    lm_ggml_tensor * mm_ffn_up_b = nullptr;
    lm_ggml_tensor * mm_ffn_gate_w = nullptr;
    lm_ggml_tensor * mm_ffn_gate_b = nullptr;
    lm_ggml_tensor * mm_ffn_down_w = nullptr;
    lm_ggml_tensor * mm_ffn_down_b = nullptr;
    lm_ggml_tensor * mm_post_norm_w = nullptr;
    lm_ggml_tensor * mm_post_norm_b = nullptr;

    // LLaVA projection
    lm_ggml_tensor * mm_input_norm_w = nullptr;
    lm_ggml_tensor * mm_input_norm_b = nullptr;
    lm_ggml_tensor * mm_0_w = nullptr;
    lm_ggml_tensor * mm_0_b = nullptr;
    lm_ggml_tensor * mm_2_w = nullptr;
    lm_ggml_tensor * mm_2_b = nullptr;

    lm_ggml_tensor * image_newline = nullptr;

    // Yi type models with mlp+normalization projection
    lm_ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    lm_ggml_tensor * mm_1_b = nullptr;
    lm_ggml_tensor * mm_3_w = nullptr;
    lm_ggml_tensor * mm_3_b = nullptr;
    lm_ggml_tensor * mm_4_w = nullptr;
    lm_ggml_tensor * mm_4_b = nullptr;

    // GLMV-Edge projection
    lm_ggml_tensor * mm_model_adapter_conv_w = nullptr;
    lm_ggml_tensor * mm_model_adapter_conv_b = nullptr;

    // MobileVLM projection
    lm_ggml_tensor * mm_model_mlp_1_w = nullptr;
    lm_ggml_tensor * mm_model_mlp_1_b = nullptr;
    lm_ggml_tensor * mm_model_mlp_3_w = nullptr;
    lm_ggml_tensor * mm_model_mlp_3_b = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_0_0_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_0_1_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_0_1_b = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_2_0_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_2_1_w = nullptr;
    lm_ggml_tensor * mm_model_block_1_block_2_1_b = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_0_0_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_0_1_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_0_1_b = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_2_0_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_2_1_w = nullptr;
    lm_ggml_tensor * mm_model_block_2_block_2_1_b = nullptr;

    // MobileVLM_V2 projection
    lm_ggml_tensor * mm_model_mlp_0_w = nullptr;
    lm_ggml_tensor * mm_model_mlp_0_b = nullptr;
    lm_ggml_tensor * mm_model_mlp_2_w = nullptr;
    lm_ggml_tensor * mm_model_mlp_2_b = nullptr;
    lm_ggml_tensor * mm_model_peg_0_w = nullptr;
    lm_ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    lm_ggml_tensor * mm_model_pos_embed_k = nullptr;
    lm_ggml_tensor * mm_model_query = nullptr;
    lm_ggml_tensor * mm_model_proj = nullptr;
    lm_ggml_tensor * mm_model_kv_proj = nullptr;
    lm_ggml_tensor * mm_model_attn_q_w = nullptr;
    lm_ggml_tensor * mm_model_attn_q_b = nullptr;
    lm_ggml_tensor * mm_model_attn_k_w = nullptr;
    lm_ggml_tensor * mm_model_attn_k_b = nullptr;
    lm_ggml_tensor * mm_model_attn_v_w = nullptr;
    lm_ggml_tensor * mm_model_attn_v_b = nullptr;
    lm_ggml_tensor * mm_model_attn_o_w = nullptr;
    lm_ggml_tensor * mm_model_attn_o_b = nullptr;
    lm_ggml_tensor * mm_model_ln_q_w = nullptr;
    lm_ggml_tensor * mm_model_ln_q_b = nullptr;
    lm_ggml_tensor * mm_model_ln_kv_w = nullptr;
    lm_ggml_tensor * mm_model_ln_kv_b = nullptr;
    lm_ggml_tensor * mm_model_ln_post_w = nullptr;
    lm_ggml_tensor * mm_model_ln_post_b = nullptr;

    // gemma3
    lm_ggml_tensor * mm_input_proj_w = nullptr;
    lm_ggml_tensor * mm_soft_emb_norm_w = nullptr;

    // mobilenetv5 for gemma3n
    std::vector<mobilenetv5_block> mobilenet_blocks;
    std::vector<int> mobilenet_stage_ends;
    lm_ggml_tensor * mobilenet_stem_conv_w = nullptr;
    lm_ggml_tensor * mobilenet_stem_conv_b = nullptr;
    lm_ggml_tensor * mobilenet_stem_norm_w = nullptr;
    lm_ggml_tensor * mm_post_proj_norm_w = nullptr;

    // Multi-Scale Fusion Adapter (MSFA) components
    lm_ggml_tensor * msfa_concat_conv_w = nullptr;
    lm_ggml_tensor * msfa_concat_norm_w = nullptr;
    lm_ggml_tensor * msfa_ffn_expand_w = nullptr;
    lm_ggml_tensor * msfa_ffn_project_w = nullptr;
    lm_ggml_tensor * msfa_ffn_expand_bn = nullptr;
    lm_ggml_tensor * msfa_ffn_project_bn = nullptr;


    // pixtral, glm4v
    lm_ggml_tensor * token_embd_img_break = nullptr;
    lm_ggml_tensor * mm_patch_merger_w = nullptr;
    lm_ggml_tensor * mm_patch_merger_b = nullptr;

    // ultravox / whisper encoder
    lm_ggml_tensor * conv1d_1_w = nullptr;
    lm_ggml_tensor * conv1d_1_b = nullptr;
    lm_ggml_tensor * conv1d_2_w = nullptr;
    lm_ggml_tensor * conv1d_2_b = nullptr;
    lm_ggml_tensor * mm_norm_pre_w = nullptr;
    lm_ggml_tensor * mm_norm_pre_b = nullptr;
    lm_ggml_tensor * mm_norm_mid_w = nullptr;

    // cogvlm
    lm_ggml_tensor * mm_post_fc_norm_w = nullptr;
    lm_ggml_tensor * mm_post_fc_norm_b = nullptr;
    lm_ggml_tensor * mm_h_to_4h_w = nullptr;
    lm_ggml_tensor * mm_gate_w = nullptr;
    lm_ggml_tensor * mm_4h_to_h_w = nullptr;
    lm_ggml_tensor * mm_boi = nullptr;
    lm_ggml_tensor * mm_eoi = nullptr;

    // lfm2 audio
    std::array<lm_ggml_tensor *, 7> pre_encode_conv_X_w = {nullptr};
    std::array<lm_ggml_tensor *, 7> pre_encode_conv_X_b = {nullptr};
    lm_ggml_tensor * pre_encode_out_w = nullptr;
    lm_ggml_tensor * pre_encode_out_b = nullptr;

    bool audio_has_avgpool() const {
        return proj_type == PROJECTOR_TYPE_QWEN2A
            || proj_type == PROJECTOR_TYPE_VOXTRAL
            || proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO;
    }

    bool audio_has_stack_frames() const {
        return proj_type == PROJECTOR_TYPE_ULTRAVOX
            || proj_type == PROJECTOR_TYPE_VOXTRAL;
    }
};

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx);
