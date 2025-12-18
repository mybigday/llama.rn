#include "models.h"

lm_ggml_cgraph * clip_graph_kimivl::build() {
    // 2D input positions
    lm_ggml_tensor * pos_h = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_patches);
    lm_ggml_set_name(pos_h, "pos_h");
    lm_ggml_set_input(pos_h);

    lm_ggml_tensor * pos_w = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_patches);
    lm_ggml_set_name(pos_w, "pos_w");
    lm_ggml_set_input(pos_w);

    lm_ggml_tensor * learned_pos_embd = resize_position_embeddings();

    // build ViT with 2D position embeddings
    auto add_pos = [&](lm_ggml_tensor * cur, const clip_layer &) {
        // first half is X axis and second half is Y axis
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    lm_ggml_tensor * inp = build_inp();
    lm_ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // patch_merger
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        // projection norm
        int proj_inp_dim = cur->ne[0];
        cur = lm_ggml_view_2d(ctx0, cur,
            n_embd, cur->ne[1] * scale_factor * scale_factor,
            lm_ggml_row_size(cur->type, n_embd), 0);
        cur = lm_ggml_norm(ctx0, cur, 1e-5); // default nn.LayerNorm
        cur = lm_ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = lm_ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = lm_ggml_view_2d(ctx0, cur,
            proj_inp_dim, cur->ne[1] / scale_factor / scale_factor,
            lm_ggml_row_size(cur->type, proj_inp_dim), 0);
        cb(cur, "proj_inp_normed", -1);

        // projection mlp
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);
        cb(cur, "proj_out", -1);
    }

    // build the graph
    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
