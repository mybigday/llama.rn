#include "models.h"

lm_ggml_cgraph * clip_graph_internvl::build() {
    LM_GGML_ASSERT(model.class_embedding != nullptr);
    LM_GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1;
    lm_ggml_tensor * inp = build_inp();

    // add CLS token
    inp = lm_ggml_concat(ctx0, inp, model.class_embedding, 1);

    // The larger models use a different ViT, which uses RMS norm instead of layer norm
    // ref: https://github.com/ggml-org/llama.cpp/pull/13443#issuecomment-2869786188
    norm_type norm_t = (hparams.n_embd == 3200 && hparams.n_layer == 45)
        ? NORM_TYPE_RMS // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
        : NORM_TYPE_NORMAL; // 300M ViT (Used by all smaller InternVL models)

    lm_ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            norm_t,
                            hparams.ffn_op,
                            model.position_embeddings,
                            nullptr);

    // remove CLS token
    cur = lm_ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        lm_ggml_row_size(cur->type, n_embd), 0);

    // pixel shuffle
    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz    = 1; // batch size, always 1 for now since we don't support batching
        const int height = n_patches_y;
        const int width  = n_patches_x;
        LM_GGML_ASSERT(scale_factor > 0);
        cur = lm_ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
        cur = lm_ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = lm_ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            height / scale_factor,
            width / scale_factor,
            bsz);
        cur = lm_ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = lm_ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            cur->ne[1] * cur->ne[2]);
    }

    // projector (always using GELU activation)
    {
        // projector LayerNorm uses pytorch's default eps = 1e-5
        // ref: https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/blob/a34d3e4e129a5856abfd6aa6de79776484caa14e/modeling_internvl_chat.py#L79
        cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_3_w, model.mm_3_b,
            FFN_GELU,
            -1);
    }

    // build the graph
    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
