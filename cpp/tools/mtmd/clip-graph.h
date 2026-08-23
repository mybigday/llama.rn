#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <vector>
#include <functional>

#define DEFAULT_INTERPOLATION_MODE (LM_GGML_SCALE_MODE_BILINEAR | LM_GGML_SCALE_FLAG_ANTIALIAS)

struct build_vit_opts {
    lm_ggml_tensor * attn_mask = nullptr;
    // TODO @ngxson : merge attn_mask and attn_mask_layers into one call
    std::vector<lm_ggml_tensor *> attn_mask_layers; // one per layer

    // hook at layer output embeddings
    std::function<void(lm_ggml_tensor * cur, int il)> callback_layer_out = nullptr;

    // whether to skip the automatic post-layernorm (model.post_ln_w) applied at the end
    bool skip_post_ln = false;
};

struct clip_graph {
    const clip_model & model;
    const clip_hparams & hparams;
    projector_type proj_type;

    const clip_image_f32 & img; // for backward compat
    const clip_image_f32_batch * img_batch = nullptr;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int n_head_kv;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    float kq_scale; // TODO: maybe move this to hparams
    const clip_flash_attn_type flash_attn_type;

    // TODO [QWEN_VIDEO]: improve this in the future
    int n_batch = 1;

    lm_ggml_context_ptr ctx0_ptr;
    lm_ggml_context * ctx0;
    lm_ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img);

    // build sub-graph, reuse buf from parent
    clip_graph(const clip_graph & parent);

    virtual ~clip_graph() = default;
    virtual lm_ggml_cgraph * build() = 0;

    // wrapper around lm_ggml_mul_mat, allow hooking (e.g. LoRA, clamping) depending on the model
    // tensor w should be the weight matrix, and tensor x should be the input
    virtual lm_ggml_tensor * build_mm(lm_ggml_tensor * w, lm_ggml_tensor * x) const;
    // TODO: build_mm(w, b, x) to support bias

    virtual bool support_batch() const {
        return false;
    }

    //
    // utility functions
    //
    void cb(lm_ggml_tensor * cur0, const char * name, int il) const;

    const clip_image_f32 & get_img(size_t idx) const {
        LM_GGML_ASSERT(img_batch);
        LM_GGML_ASSERT(idx < img_batch->entries.size());
        return img_batch->entries[idx];
    }

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
                std::function<lm_ggml_tensor *(lm_ggml_tensor *, const clip_layer &)> add_pos,
                const build_vit_opts & opts = {});

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

    lm_ggml_tensor * build_moe_ffn(
            lm_ggml_tensor * cur,
            const clip_layer & layer,
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
            int il,
            lm_ggml_tensor * sinks = nullptr) const;

    // implementation of the 2D RoPE using two lm_ggml_rope_ext calls
    //
    // unlike LM_GGML_ROPE_TYPE_VISION which forces NEOX ordering, this rotates adjacent pairs (normal ordering)
    //
    // example:
    //  given a single head with size = 8 --> [00000000]
    //  dims [0, 4) rotate with pos_a, dims [4, 8) rotate with pos_b --> [aaaabbbb]
    //  interleave_freq = false --> both halves use the same inv_freq set (like LM_GGML_ROPE_TYPE_VISION)
    //  interleave_freq = true  --> first half uses even inv_freq, second half uses odd inv_freq (used by pixtral)
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
