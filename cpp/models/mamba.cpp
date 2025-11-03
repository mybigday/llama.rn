#include "models.h"


llm_build_mamba::llm_build_mamba(const llama_model & model, const llm_graph_params & params) : llm_graph_context_mamba(params) {
    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    lm_ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (model.arch == LLM_ARCH_MAMBA2) {
            cur = build_mamba2_layer(rs_inp, cur, model, ubatch, il);
        } else {
            cur = build_mamba_layer(rs_inp, cur, model, ubatch, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = lm_ggml_get_rows(ctx0, cur, inp_out_ids);
            inpL = lm_ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // residual
        cur = lm_ggml_add(ctx0, cur, inpL);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    // final rmsnorm
    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}

