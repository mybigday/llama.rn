#include "models.h"

lm_ggml_cgraph * clip_graph_nemotron_v2_vl::build() {
    LM_GGML_ASSERT(model.class_embedding != nullptr);
    LM_GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_registers = model.class_embedding->ne[1];
    const int n_pos = n_patches + n_registers;

    lm_ggml_tensor * inp = build_inp();

    // add position embeddings (pre-downsampled during GGUF conversion for fixed 512x512 input)
    inp = lm_ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "inp_pos", -1);

    inp = lm_ggml_concat(ctx0, model.class_embedding, inp, 1);

    lm_ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);

    cur = lm_ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        lm_ggml_row_size(cur->type, n_embd),
        n_registers * lm_ggml_row_size(cur->type, n_embd));

    cur = build_patch_merge_permute(cur, model.hparams.n_merge);

    {
        cur = build_norm(cur, model.mm_0_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
        cur = build_ffn(cur, model.mm_1_w, nullptr, nullptr, nullptr, model.mm_3_w, nullptr, FFN_RELU_SQR, -1);
    }

    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
