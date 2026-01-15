#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <vector>
#include <functional>

#define DEFAULT_INTERPOLATION_MODE (LM_GGML_SCALE_MODE_BILINEAR | LM_GGML_SCALE_FLAG_ANTIALIAS)

struct clip_graph {
    const clip_model & model;
    const clip_hparams & hparams;
    projector_type proj_type;

    // we only support single image per batch
    const clip_image_f32 & img;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    const float kq_scale;
    const clip_flash_attn_type flash_attn_type;

    lm_ggml_context_ptr ctx0_ptr;
    lm_ggml_context * ctx0;
    lm_ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img);

    virtual ~clip_graph() = default;
    virtual lm_ggml_cgraph * build() = 0;

    //
    // utility functions
    //
    void cb(lm_ggml_tensor * cur0, const char * name, int il) const;

    // siglip2 naflex
    lm_ggml_tensor * resize_position_embeddings(uint32_t interpolation_mode = DEFAULT_INTERPOLATION_MODE);

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    lm_ggml_tensor * build_vit(
                lm_ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                lm_ggml_tensor * learned_pos_embd,
                std::function<lm_ggml_tensor *(lm_ggml_tensor *, const clip_layer &)> add_pos);

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    lm_ggml_tensor * build_inp();

    lm_ggml_tensor * build_inp_raw(int channels = 3);

    lm_ggml_tensor * build_norm(
            lm_ggml_tensor * cur,
            lm_ggml_tensor * mw,
            lm_ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const;

    lm_ggml_tensor * build_ffn(
            lm_ggml_tensor * cur,
            lm_ggml_tensor * up,
            lm_ggml_tensor * up_b,
            lm_ggml_tensor * gate,
            lm_ggml_tensor * gate_b,
            lm_ggml_tensor * down,
            lm_ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const;

    lm_ggml_tensor * build_attn(
            lm_ggml_tensor * wo,
            lm_ggml_tensor * wo_b,
            lm_ggml_tensor * q_cur,
            lm_ggml_tensor * k_cur,
            lm_ggml_tensor * v_cur,
            lm_ggml_tensor * kq_mask,
            float kq_scale,
            int il) const;

    // implementation of the 2D RoPE without adding a new op in ggml
    // this is not efficient (use double the memory), but works on all backends
    // TODO: there was a more efficient which relies on lm_ggml_view and lm_ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
    lm_ggml_tensor * build_rope_2d(
        lm_ggml_context * ctx0,
        lm_ggml_tensor * cur,
        lm_ggml_tensor * pos_a, // first half
        lm_ggml_tensor * pos_b, // second half
        const float freq_base,
        const bool interleave_freq
    );

    // aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
    // support dynamic resolution
    lm_ggml_tensor * build_patch_merge_permute(lm_ggml_tensor * cur, int scale_factor);

    // Generic function to stack frames for audio processing
    // Abstracts out the StackAudioFrames logic used by ultravox
    lm_ggml_tensor * build_stack(lm_ggml_tensor * cur, int32_t stack_factor, int32_t n_embed);
};
