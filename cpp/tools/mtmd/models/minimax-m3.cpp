#include "models.h"

lm_ggml_tensor * clip_graph_minimax_m3::apply_rope(
        lm_ggml_tensor * x, lm_ggml_tensor * pos_h, lm_ggml_tensor * pos_w) {
    const int dh  = (int) x->ne[0];
    const int axd = 2 * ((2 * (dh / 2) / 3) / 2);

    LM_GGML_ASSERT(3 * axd <= dh);

    const float th  = hparams.rope_theta;

    // layout of x is [t, h, w, pad]
    // t is unrotated, h and w are rotated, pad is unrotated
    x = lm_ggml_rope_ext(ctx0, x, pos_h, nullptr, axd, LM_GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    x = lm_ggml_rope_set_offset(x, axd);

    x = lm_ggml_rope_ext(ctx0, x, pos_w, nullptr, axd, LM_GGML_ROPE_TYPE_NEOX, 0, th, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    x = lm_ggml_rope_set_offset(x, 2 * axd);

    return x;
}

lm_ggml_cgraph * clip_graph_minimax_m3::build() {
    LM_GGML_ASSERT(model.patch_bias     == nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);
    LM_GGML_ASSERT(model.patch_embeddings_0 && model.patch_embeddings_1);
    LM_GGML_ASSERT(model.mm_1_w && model.mm_2_w);
    LM_GGML_ASSERT(model.mm_merger_fc1_w && model.mm_merger_fc2_w);

    const int batch_size = 1;
    const int n_pos      = n_patches;
    const int merge      = hparams.n_merge;

    // patch embedding
    lm_ggml_tensor * inp_raw = build_inp_raw();
    lm_ggml_tensor * inp = lm_ggml_add(ctx0,
        lm_ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1),
        lm_ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1));

    // spatial merge
    {
        inp = lm_ggml_permute(ctx0, inp, 1, 2, 0, 3);
        inp = lm_ggml_cont_4d(ctx0, inp, n_embd * merge, n_patches_x / merge, n_patches_y, batch_size);
        inp = lm_ggml_reshape_4d(ctx0, inp, n_embd * merge, n_patches_x / merge, merge, batch_size * (n_patches_y / merge));
        inp = lm_ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = lm_ggml_cont_3d(ctx0, inp, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // t (time axis) is always 0 for now, so we leave it unrotated
    lm_ggml_tensor * pos_h = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos);
    lm_ggml_set_name(pos_h, "minimax_pos_h"); lm_ggml_set_input(pos_h);
    lm_ggml_tensor * pos_w = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos);
    lm_ggml_set_name(pos_w, "minimax_pos_w"); lm_ggml_set_input(pos_w);

    lm_ggml_tensor * inpL = build_vit(
        inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_ERF, nullptr,
        [&](lm_ggml_tensor * c, const clip_layer &) {
            return apply_rope(c, pos_h, pos_w);
        });

    // projector
    lm_ggml_tensor * emb = inpL;
    emb = build_ffn(emb, model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b, FFN_GELU_ERF, -1);

    const int64_t proj = emb->ne[0];
    emb = lm_ggml_reshape_2d(ctx0, emb, proj * merge * merge, n_pos / (merge * merge));

    emb = build_ffn(emb, model.mm_merger_fc1_w, model.mm_merger_fc1_b,
                    nullptr, nullptr,
                    model.mm_merger_fc2_w, model.mm_merger_fc2_b, FFN_GELU_ERF, -1);

    lm_ggml_build_forward_expand(gf, emb);
    return gf;
}
