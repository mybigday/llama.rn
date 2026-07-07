#include "models.h"

// Barbet — Open-Formosa R2 text-semantic LM (BlueMagpie-TTS backbone).
// Mamba2 + attention hybrid: motif `global, sliding, sliding, mamba2` × 28.
// Layer type detection follows the nemotron_h convention: n_head_kv(i)==0
// marks a mamba2 layer.  Attention layers carry per-head q/k RMSNorm.
// Sliding-window attention (window=8192) is approximated by full causal
// attention — identical for any sequence <= 8192 tokens, which covers all
// TTS use.  Ported from codec.cpp's barbet-llamacpp.patch.

void llama_model_barbet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // A layer is recurrent (mamba2) IFF its kv-head count is 0.  Barbet always
    // carries FFN, so the extra n_ff==0 guard nemotron-h uses would flip
    // everything false.
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = (hparams.n_head_kv(i) == 0);
    }

    switch (hparams.n_layer()) {
        case 28: type = LLM_TYPE_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_barbet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // mamba2 Mixer SSM params (int64_t for tensor dimensions)
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = hparams.ssm_d_inner;
    const int64_t d_state    = hparams.ssm_d_state;
    const int64_t n_ssm_head = hparams.ssm_dt_rank;
    const int64_t n_group    = hparams.ssm_n_group;
    const int64_t d_in_proj  = 2*d_inner + 2*n_group*d_state + n_ssm_head;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        // Tied embeddings: reuse the token embed for the head.
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            layer.ssm_in       = create_tensor(tn(LLM_TENSOR_SSM_IN,     "weight", i), {n_embd, d_in_proj}, 0);
            layer.ssm_conv1d   = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner + 2*n_group*d_state}, 0);
            layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias",   i), {d_inner + 2*n_group*d_state}, 0);
            layer.ssm_dt_b     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {n_ssm_head}, 0);
            layer.ssm_a        = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_ssm_head}, 0);
            layer.ssm_d        = create_tensor(tn(LLM_TENSOR_SSM_D, i), {1, n_ssm_head}, 0);
            layer.ssm_norm     = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / n_group, n_group}, 0);
            layer.ssm_out      = create_tensor(tn(LLM_TENSOR_SSM_OUT,  "weight", i), {d_inner, n_embd}, 0);
        } else {
            const int64_t n_head_i       = hparams.n_head(i);
            const int64_t n_embd_k_gqa_i = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v_gqa_i = hparams.n_embd_v_gqa(i);
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head_i}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa_i}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa_i}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head_i, n_embd}, 0);
            // qk_norm always present; head_dim scalar per Qwen3 convention.
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_barbet::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_barbet::graph::graph(const llama_model & model, const llm_graph_params & params) :
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

        if (hparams.is_recr(il)) {
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

lm_ggml_tensor * llama_model_barbet::graph::build_attention_layer(lm_ggml_tensor *             cur,
                                                               lm_ggml_tensor *             inp_pos,
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
                     model.layers[il].wo, NULL, model.layers[il].wo_s,
                     Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     1.0f/sqrtf(float(n_embd_head)), il);
    cb(cur, "attn_out", il);
    return cur;
}
