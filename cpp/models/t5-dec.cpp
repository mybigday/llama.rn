#include "models.h"

llm_build_t5_dec::llm_build_t5_dec(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    //const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    LM_GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    lm_ggml_tensor * embd_enc       = build_inp_cross_embd();
    lm_ggml_tensor * pos_bucket_dec = build_inp_pos_bucket_dec();

    const int64_t n_outputs_enc = embd_enc->ne[1];

    auto * inp_attn_self  = build_attn_inp_kv();
    auto * inp_attn_cross = build_attn_inp_cross();

    lm_ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t dec_n_layer = hparams.dec_n_layer;

    for (int il = 0; il < dec_n_layer; ++il) {
        lm_ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            lm_ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            lm_ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            lm_ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            lm_ggml_tensor * attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
            lm_ggml_tensor * kq_b = build_pos_bias(pos_bucket_dec, attn_rel_b);

            cur = build_attn(inp_attn_self,
                    model.layers[il].wo, model.layers[il].bo,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, 1.0f, il);
            cb(cur, "kqv_out", il);
        }
        cur = lm_ggml_add(ctx0, cur, inpSA);
        cb(cur, "cross_inp", il);

        lm_ggml_tensor * inpCA = cur;

        // norm
        cur = build_norm(cur,
                model.layers[il].attn_norm_cross, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm_cross", il);

        // cross-attention
        {
            lm_ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq_cross, cur);
            cb(Qcur, "Qcur", il);

            lm_ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk_cross, embd_enc);
            cb(Kcur, "Kcur", il);

            lm_ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv_cross, embd_enc);
            cb(Vcur, "Vcur", il);

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_outputs_enc);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_outputs_enc);

            cur = build_attn(inp_attn_cross,
                    model.layers[il].wo_cross, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f, il);
            cb(cur, "kqv_out", il);

            //lm_ggml_tensor * q =                 lm_ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            //lm_ggml_tensor * k = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            //lm_ggml_tensor * kq = lm_ggml_mul_mat(ctx0, k, q);
            //cb(kq, "kq", il);

            //kq = lm_ggml_soft_max_ext(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
            //cb(kq, "kq_soft_max_ext", il);

            //lm_ggml_tensor * v = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_reshape_2d(ctx0, Vcur, n_embd_gqa, n_outputs_enc)));
            //cb(v, "v", il);

            //lm_ggml_tensor * kqv = lm_ggml_mul_mat(ctx0, lm_ggml_reshape_3d(ctx0, v, n_outputs_enc, n_embd_head, n_head_kv), kq);
            //cb(kqv, "kqv", il);

            //lm_ggml_tensor * kqv_merged = lm_ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            //cb(kqv_merged, "kqv_merged", il);

            //cur = lm_ggml_cont_2d(ctx0, kqv_merged, n_embd_gqa, n_tokens);
            //cb(cur, "kqv_merged_cont", il);

            //lm_ggml_build_forward_expand(gf, cur);

            //cur = build_lora_mm(model.layers[il].wo_cross, cur);
            //cb(cur, "kqv_out", il);
        }
        if (il == dec_n_layer - 1 && inp_out_ids) {
            cur   = lm_ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpCA = lm_ggml_get_rows(ctx0, inpCA, inp_out_ids);
        }
        lm_ggml_tensor * ffn_inp = lm_ggml_add(ctx0, cur, inpCA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        {
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            // T5 uses relu, flan-T5 uses gelu-gated
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    il);
            cb(cur, "ffn_out", il);
        }
        cur = lm_ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    cb(cur, "result_embd", -1);

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}
