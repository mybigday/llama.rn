#include "models.h"

// Barbet — Open-Formosa R2 text-semantic LM (BlueMagpie-TTS backbone).
// Mamba2 + attention hybrid: motif `global, sliding, sliding, mamba2` × 28.
// Layer type detection follows the nemotron_h convention: n_head_kv(i)==0
// marks a mamba2 layer.  Attention layers carry per-head q/k RMSNorm.
// Sliding-window attention (window=8192) is approximated by full causal
// attention — identical for any sequence <= 8192 tokens, which covers all
// TTS use.  Ported from codec.cpp's barbet-llamacpp.patch.

llm_build_barbet::llm_build_barbet(const llama_model & model, const llm_graph_params & params) :
    llm_build_mamba_base(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    LM_GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp = build_inp_mem_hybrid();
    lm_ggml_tensor * inp_pos     = build_inp_pos();
    lm_ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        lm_ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (hparams.is_recurrent(il)) {
            cur = build_mamba2_layer(inp->get_recr(), cur, model, ubatch, il);
        } else {
            cur = build_attention_layer(cur, inp_pos, inp->get_attn(), model, n_embd_head, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = lm_ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = lm_ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        lm_ggml_tensor * ffn_inp = lm_ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // SwiGLU FFN
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = lm_ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}

lm_ggml_tensor * llm_build_barbet::build_attention_layer(lm_ggml_tensor *          cur,
                                                        lm_ggml_tensor *          inp_pos,
                                                        llm_graph_input_attn_kv * inp_attn,
                                                        const llama_model &       model,
                                                        int64_t                   n_embd_head,
                                                        int                       il) {
    lm_ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur, "Qcur", il);
    lm_ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);
    lm_ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    Qcur = lm_ggml_reshape_3d(ctx0, Qcur, n_embd_head, hparams.n_head(il),    n_tokens);
    Kcur = lm_ggml_reshape_3d(ctx0, Kcur, n_embd_head, hparams.n_head_kv(il), n_tokens);
    Vcur = lm_ggml_reshape_3d(ctx0, Vcur, n_embd_head, hparams.n_head_kv(il), n_tokens);

    // Per-head q/k RMSNorm, then NEOX-style RoPE.
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
    Qcur = lm_ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
    Kcur = lm_ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "Qcur_normed", il);
    cb(Kcur, "Kcur_normed", il);

    cur = build_attn(inp_attn,
                     model.layers[il].wo, NULL,
                     Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     1.0f/sqrtf(float(n_embd_head)), il);
    cb(cur, "attn_out", il);
    return cur;
}
