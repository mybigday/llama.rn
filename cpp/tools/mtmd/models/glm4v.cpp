#include "models.h"

lm_ggml_cgraph * clip_graph_glm4v::build() {
    LM_GGML_ASSERT(model.patch_bias != nullptr);
    LM_GGML_ASSERT(model.position_embeddings != nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size = 1;

    norm_type norm_t = NORM_TYPE_RMS;

    lm_ggml_tensor * inp_raw = build_inp_raw();
    lm_ggml_tensor * inp = lm_ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};
    lm_ggml_tensor * positions = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_patches * 4);
    lm_ggml_set_name(positions, "positions");
    lm_ggml_set_input(positions);

    LM_GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    LM_GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = lm_ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = lm_ggml_add(ctx0, inp, inp_1);

        inp = lm_ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = lm_ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = lm_ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = lm_ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = lm_ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    inp = lm_ggml_add(ctx0, inp, model.patch_bias);
    cb(inp, "patch_bias", -1);

    // pos-conv norm
    inp = build_norm(inp, model.norm_embd_w, model.norm_embd_b, norm_t, eps, -1);

    // calculate absolute position embedding and apply
    lm_ggml_tensor * learned_pos_embd = resize_position_embeddings(LM_GGML_SCALE_MODE_BICUBIC);
    learned_pos_embd = lm_ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = lm_ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = lm_ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = lm_ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);
    cb(learned_pos_embd, "learned_pos_embd", -1);

    auto add_pos = [&](lm_ggml_tensor * cur, const clip_layer &) {
        return lm_ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, LM_GGML_ROPE_TYPE_VISION,
                    32768, hparams.rope_theta, 1, 0, 1, 32, 1);
    };

    lm_ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            norm_t,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);
    // cb(lm_ggml_sum(ctx0, cur), "vit_out_sum", -1);

    // GLM4V projector
    // ref: https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/glm4v/modeling_glm4v.py#L116-L130

    // patch merger (downsample)
    {
        int n_merge = hparams.n_merge;
        LM_GGML_ASSERT(n_merge > 0);

        int n_token_out = n_patches / n_merge / n_merge;
        cur = lm_ggml_reshape_4d(ctx0, cur, n_embd, n_merge, n_merge, n_token_out);
        cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, cur, 2, 0, 1, 3)); // [n_merge, n_merge, n_embd, n_token_out]
        cur = lm_ggml_conv_2d(ctx0, model.mm_patch_merger_w, cur, n_merge, n_merge, 0, 0, 1, 1);
        cur = lm_ggml_reshape_2d(ctx0, cur, cur->ne[2], n_token_out); // [n_embd_out, n_token_out]

        cur = lm_ggml_add(ctx0, cur, model.mm_patch_merger_b);
    }

    // FC projector
    {
        cur = lm_ggml_mul_mat(ctx0, model.projection, cur);
        // default LayerNorm (post_projection_norm)
        cur = build_norm(cur, model.mm_post_norm_w, model.mm_post_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = lm_ggml_gelu_erf(ctx0, cur);
        cb(cur, "after_fc_proj", -1);
    }

    // FFN projector
    {
        cur = build_ffn(cur,
            model.mm_ffn_up_w, model.mm_ffn_up_b,
            model.mm_ffn_gate_w, model.mm_ffn_gate_b,
            model.mm_ffn_down_w, model.mm_ffn_down_b,
            hparams.ffn_op, -1);
        cb(cur, "after_ffn_proj", -1);
        // cb(lm_ggml_sum(ctx0, cur), "merged_sum", -1);
    }

    // build the graph
    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
