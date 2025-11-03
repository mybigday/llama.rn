#include "models.h"



llm_build_gemma3n_iswa::llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model),
    n_embd_head(model.hparams.n_embd_head_k),
    n_embd_altup(model.hparams.n_embd_altup),
    n_altup(model.hparams.n_altup),
    i_altup_act(model.hparams.i_altup_act) {
    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    if (ubatch.token) {
        inpL = lm_ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);
    }
    // inp_pos - contains the positions
    lm_ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    auto * inp_attn = build_attn_inp_kv_iswa();

    // inp_per_layer shape: [n_embd_altup, n_tokens, n_layer]
    lm_ggml_tensor * inp_per_layer = project_per_layer_inputs(inpL, get_per_layer_inputs());

    // inpL now has only 1 altup, project it to the rest of the altups
    // these "added" altups will be concat to the last dim of inpL
    {
        lm_ggml_tensor * target_magnitude = calc_magnitude(inpL);
        lm_ggml_tensor * inp_repeated     = lm_ggml_repeat_4d(ctx0, inpL, n_embd, n_tokens, n_altup - 1, 1);
        lm_ggml_tensor * altup_added =
            lm_ggml_mul_mat(ctx0, model.altup_proj, inp_repeated);  // shape: [n_embd, n_tokens, n_altup - 1]
        lm_ggml_tensor * new_magnitude = calc_magnitude(altup_added);
        altup_added                 = lm_ggml_div(ctx0, lm_ggml_mul(ctx0, altup_added, target_magnitude), new_magnitude);
        inpL                        = lm_ggml_concat(ctx0, inpL, altup_added, 2);  // shape: [n_embd, n_tokens, n_altup]
        cb(inpL, "inp_stacked", -1);
    }
    // inpL now has shape:          [n_embd,       n_tokens, n_altup]
    // inp_per_layer now has shape: [n_embd_altup, n_tokens, n_layer]

    for (int il = 0; il < n_layer; ++il) {
        // this block is made to be closely resemble Gemma3p5DecoderLayer on python code
        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        lm_ggml_tensor * cur         = inpL;                    // [n_embd, n_tokens, n_altup]
        lm_ggml_tensor * predictions = altup_predict(cur, il);  // [n_embd, n_tokens, n_altup]

        // predicted value will go through self-attention and laurel
        lm_ggml_tensor * active_prediction = view_2d_slice(predictions, i_altup_act);  // [n_embd, n_tokens]
        cur                             = active_prediction;
        cb(cur, "active_prediction", il);

        // norm
        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // laurel
        lm_ggml_tensor * laurel_out = laurel(cur, il);  // [n_embd, n_tokens]

        // self-attention
        if (hparams.has_kv(il)) {
            // compute Q and K and RoPE them
            lm_ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            lm_ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);

            lm_ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Vcur = lm_ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Qcur, "Qcur_normed", il);
            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Qcur = lm_ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = lm_ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur_pos", il);
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        } else {
            // reuse KV cache of earlier layers
            lm_ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = lm_ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }
        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = lm_ggml_add(ctx0, cur, active_prediction);  // [n_embd, n_tokens]
        cb(cur, "attn_gated", il);

        lm_ggml_tensor * attn_laurel = lm_ggml_scale(ctx0, lm_ggml_add(ctx0, cur, laurel_out),
                                               1.0f / sqrtf(2.0f));  // [n_embd, n_tokens]
        cb(attn_laurel, "attn_laurel", il);

        cur = build_norm(attn_laurel, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // feed-forward network
        {
            lm_ggml_tensor * up_proj   = build_lora_mm(model.layers[il].ffn_up, cur);
            lm_ggml_tensor * gate_proj = build_lora_mm(model.layers[il].ffn_gate, cur);

            if (il < n_layer_sparsity) {
                // apply activation sparsity
                gate_proj = gaussian_topk(gate_proj);
            }
            gate_proj = lm_ggml_gelu(ctx0, gate_proj);

            cur = lm_ggml_mul(ctx0, up_proj, gate_proj);
            cur = build_lora_mm(model.layers[il].ffn_down, cur);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        lm_ggml_tensor * attn_ffw_laurel_gated = lm_ggml_add(ctx0, cur, attn_laurel);  // [n_embd, n_tokens]
        cb(attn_ffw_laurel_gated, "attn_ffw_laurel_gated", il);

        lm_ggml_tensor * corrected = altup_correct(predictions, attn_ffw_laurel_gated, il);  // [n_embd, n_tokens, n_altup]

        lm_ggml_tensor * first_prediction;                                                   // [n_embd, n_tokens]
        {
            first_prediction = view_2d_slice(corrected, i_altup_act);                     // [n_embd, n_tokens]
            first_prediction = lm_ggml_mul(ctx0, first_prediction, model.layers[il].altup_correct_scale);
            first_prediction = build_lora_mm(model.layers[il].per_layer_inp_gate, first_prediction);
            first_prediction = lm_ggml_gelu(ctx0, first_prediction);                 // [n_embd_altup, n_tokens]
            cb(first_prediction, "first_prediction_gated", il);
            lm_ggml_tensor * inp_this_layer = view_2d_slice(inp_per_layer, il);      // [n_embd_altup, n_tokens]
            first_prediction = lm_ggml_mul(ctx0, first_prediction, inp_this_layer);  // [n_embd_altup, n_tokens]
            cb(first_prediction, "first_prediction_scaled", il);

            first_prediction = build_lora_mm(model.layers[il].per_layer_proj, first_prediction);  // [n_embd, n_tokens]
            first_prediction =
                build_norm(first_prediction, model.layers[il].per_layer_post_norm, NULL, LLM_NORM_RMS, il);
            cb(first_prediction, "first_prediction_out", il);
        }
        // equivalent to python code: corrected_predictions[1:] += first_prediction
        {
            lm_ggml_tensor * slice_first = view_2d_slice(corrected, 0);
            lm_ggml_tensor * slice_rest  = lm_ggml_view_3d(
                ctx0, corrected, n_embd, n_tokens, n_altup - 1, lm_ggml_row_size(corrected->type, n_embd),
                lm_ggml_row_size(corrected->type, n_embd * n_tokens), n_embd * n_tokens * lm_ggml_element_size(corrected));
            lm_ggml_tensor * tmp = lm_ggml_add(ctx0, slice_rest, first_prediction);  // [n_embd, n_tokens, n_altup - 1]
            corrected         = lm_ggml_concat(ctx0, slice_first, tmp, 2);        // [n_embd, n_tokens, n_altup]
        }
        cur = corrected;                                                       // [n_embd, n_tokens, n_altup]
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;  // [n_embd, n_tokens, n_altup]

    // cur now has multiple altup(s), we want to merge them back to 1 altup
    {
        lm_ggml_tensor * target_magnitude = calc_magnitude(view_2d_slice(cur, i_altup_act));  // [n_embd, n_tokens]
        // do a view to skip the first slice (active altup)
        lm_ggml_tensor * alt_slice =
            lm_ggml_view_3d(ctx0, cur, n_embd, n_tokens, n_altup - 1, lm_ggml_row_size(cur->type, n_embd),
                         lm_ggml_row_size(cur->type, n_embd * n_tokens), n_embd * n_tokens * lm_ggml_element_size(cur));
        lm_ggml_tensor * altup_unembd =
            lm_ggml_mul_mat(ctx0, model.altup_unembd_proj, alt_slice);  // shape: [n_embd, n_tokens, n_altup - 1]
        lm_ggml_tensor * new_magnitude = calc_magnitude(altup_unembd);
        altup_unembd                = lm_ggml_div(ctx0, lm_ggml_mul(ctx0, altup_unembd, target_magnitude), new_magnitude);
        cb(altup_unembd, "altup_unembd", -1);

        // equivalent to torch.mean(hidden_states, dim=0)
        cur = view_2d_slice(cur, 0);  // [n_embd, n_tokens]
        for (int i = 0; i < n_altup - 1; ++i) {
            cur = lm_ggml_add(ctx0, cur, view_2d_slice(altup_unembd, i));
        }
        cur = lm_ggml_scale(ctx0, cur, 1.0f / float(n_altup));  // [n_embd, n_tokens]
        cb(cur, "unembd_merged", -1);
    }
    // cur now has shape: [n_embd, n_tokens]

    // TODO: move this to right after the last KV layer
    {
        // skip computing output for unused tokens
        lm_ggml_tensor * inp_out_ids = build_inp_out_ids();
        cur                       = lm_ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    {
        // final logit soft-capping
        cur = lm_ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = lm_ggml_tanh(ctx0, cur);
        cur = lm_ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}

lm_ggml_tensor * llm_build_gemma3n_iswa::calc_magnitude(lm_ggml_tensor * x) {
    return lm_ggml_sqrt(ctx0, lm_ggml_sum_rows(ctx0, lm_ggml_sqr(ctx0, x)));
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
lm_ggml_tensor * llm_build_gemma3n_iswa::view_2d_slice(lm_ggml_tensor * x, int idx) {
    LM_GGML_ASSERT(idx < (int) x->ne[2]);
    return lm_ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], lm_ggml_row_size(x->type, x->ne[0]),
                        idx * x->ne[0] * x->ne[1] * lm_ggml_element_size(x));
}

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_altup, n_layer, n_tokens]
lm_ggml_tensor * llm_build_gemma3n_iswa::get_per_layer_inputs() {
    auto          inp = std::make_unique<llm_graph_input_embd>();
    lm_ggml_tensor * inp_per_layer;
    if (ubatch.token) {
        inp->tokens = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, ubatch.n_tokens);
        lm_ggml_set_input(inp->tokens);
        res->t_tokens = inp->tokens;
        inp_per_layer = lm_ggml_get_rows(ctx0, model.tok_embd_per_layer, inp->tokens);
        inp_per_layer = lm_ggml_reshape_3d(ctx0, inp_per_layer, n_embd_altup, n_layer, n_tokens);
        inp_per_layer = lm_ggml_scale(ctx0, inp_per_layer, sqrtf((float) n_embd_altup));
        cb(inp_per_layer, "inp_per_layer_selected", -1);
    } else {
        LM_GGML_ABORT("TODO: support embd input");
    }
    res->add_input(std::move(inp));
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// output shape: [n_embd_altup, n_tokens, n_layer]
lm_ggml_tensor * llm_build_gemma3n_iswa::project_per_layer_inputs(lm_ggml_tensor * inputs_embeds, lm_ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    lm_ggml_tensor * per_layer_proj = lm_ggml_mul_mat(ctx0, model.per_layer_model_proj, inputs_embeds);
    per_layer_proj               = lm_ggml_scale(ctx0, per_layer_proj, per_layer_projection_scale);
    per_layer_proj               = lm_ggml_reshape_3d(ctx0, per_layer_proj, n_embd_altup, n_layer, n_tokens);
    per_layer_proj               = build_norm(per_layer_proj, model.per_layer_proj_norm, NULL, LLM_NORM_RMS,
                                              -1);  // [n_embd_altup, n_layer, n_tokens]
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = lm_ggml_add(ctx0, inp_per_layer, per_layer_proj);
    inp_per_layer = lm_ggml_scale(ctx0, inp_per_layer, per_layer_input_scale);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_altup, n_tokens, n_layer]
    inp_per_layer = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3));
    return inp_per_layer;
}

// input cur shape: [n_altup, n_tokens]
// output    shape: [n_altup, n_tokens]
lm_ggml_tensor * llm_build_gemma3n_iswa::laurel(lm_ggml_tensor * cur, int il) {
    lm_ggml_tensor * tmp = cur;
    tmp               = build_lora_mm(model.layers[il].laurel_l, tmp);
    tmp               = build_lora_mm(model.layers[il].laurel_r, tmp);
    tmp               = build_norm(tmp, model.layers[il].laurel_post_norm, NULL, LLM_NORM_RMS, il);
    tmp               = lm_ggml_add(ctx0, tmp, cur);
    cb(tmp, "laurel_out", il);
    return tmp;
}

// input x shape: [n_embd, n_tokens]
// output  shape: [n_embd, n_tokens]
lm_ggml_tensor * llm_build_gemma3n_iswa::gaussian_topk(lm_ggml_tensor * x) {
    lm_ggml_tensor * mean = lm_ggml_mean(ctx0, x);
    lm_ggml_tensor * std  = lm_ggml_sqrt(ctx0, lm_ggml_scale(ctx0, lm_ggml_sum_rows(ctx0, lm_ggml_sqr(ctx0, lm_ggml_sub(ctx0, x, mean))),
                                                    1.0f / (float) (x->ne[0] - 1)));
    lm_ggml_tensor * cutoff_x = lm_ggml_add(ctx0, mean, lm_ggml_scale(ctx0, std, f_sparsity_std_mul));
    return lm_ggml_relu(ctx0, lm_ggml_sub(ctx0, x, cutoff_x));
}

//
// altup functions
//

// equivalent to compute_router_modalities() in python code
// input x shape: [n_embd,  n_tokens]
// output  shape: [n_altup, n_tokens]
lm_ggml_tensor * llm_build_gemma3n_iswa::altup_compute_router_modalities(lm_ggml_tensor * x, int il) {
    lm_ggml_tensor * router_inputs = build_norm(x, model.layers[il].altup_router_norm, NULL, LLM_NORM_RMS, il);

    // router_input_scale
    router_inputs = lm_ggml_scale(ctx0, router_inputs, 1.0f / (float) n_embd);

    lm_ggml_tensor * output = lm_ggml_mul_mat(ctx0, model.layers[il].altup_router, router_inputs);
    return lm_ggml_tanh(ctx0, output);  // [n_altup, n_tokens]
}

// input cur shape: [n_embd, n_tokens, n_altup]
// output    shape: [n_embd, n_tokens, n_altup]
lm_ggml_tensor * llm_build_gemma3n_iswa::altup_predict(lm_ggml_tensor * cur, int il) {
    lm_ggml_tensor * activated  = view_2d_slice(cur, i_altup_act);                 // [n_embd, n_tokens]
    lm_ggml_tensor * modalities = altup_compute_router_modalities(activated, il);  // [n_altup, n_tokens]
    cb(modalities, "modalities", il);

    lm_ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_predict_coef, modalities);
    cb(all_coefs, "all_coefs", il);
    // first dim now having n_altup^2 elements, we reshape it to 2D (so we end up with 3D tensor)
    all_coefs = lm_ggml_reshape_3d(ctx0, all_coefs, n_altup, n_altup, n_tokens);

    // permute to [n_altup, n_embd, n_tokens]
    lm_ggml_tensor * cur_permuted = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, cur, 1, 2, 0, 3));
    lm_ggml_tensor * predictions  = lm_ggml_mul_mat(ctx0, cur_permuted, all_coefs);  // [n_altup, n_embd, n_tokens]

    // final shape must be the same as cur: [n_embd, n_tokens, n_altup]
    predictions = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, predictions, 0, 2, 1, 3));
    predictions = lm_ggml_add(ctx0, predictions, cur);
    cb(predictions, "predictions", il);

    return predictions;
}

// input predictions       shape: [n_embd, n_tokens, n_altup]
// input activated         shape: [n_embd, n_tokens]
// output                  shape: [n_embd, n_tokens, n_altup]
lm_ggml_tensor * llm_build_gemma3n_iswa::altup_correct(lm_ggml_tensor * predictions, lm_ggml_tensor * activated, int il) {
    lm_ggml_tensor * modalities = altup_compute_router_modalities(activated, il);  // [n_altup, n_tokens]
    cb(modalities, "modalities", il);

    lm_ggml_tensor * active_prediction = view_2d_slice(predictions, i_altup_act);
    lm_ggml_tensor * innovation        = lm_ggml_sub(ctx0, activated, active_prediction);  // [n_embd, n_tokens]
    cb(innovation, "innovation", il);

    lm_ggml_tensor * all_coefs = build_lora_mm(model.layers[il].altup_correct_coef, modalities);  // [n_altup, n_tokens]
    all_coefs               = lm_ggml_scale_bias(ctx0, all_coefs, 1.0f, 1.0f);                    // + 1.0
    cb(all_coefs, "all_coefs", il);
    all_coefs = lm_ggml_transpose(ctx0, all_coefs);                                               // [n_tokens, n_altup]
    all_coefs = lm_ggml_cont_3d(ctx0, all_coefs, 1, n_tokens, n_altup);                           // [1, n_tokens, n_altup]

    innovation              = lm_ggml_repeat_4d(ctx0, innovation, n_embd, n_tokens, n_altup, 1);
    lm_ggml_tensor * corrected = lm_ggml_mul(ctx0, innovation, all_coefs);   // [n_embd, n_tokens, n_altup]
    corrected               = lm_ggml_add(ctx0, corrected, predictions);  // [n_embd, n_tokens, n_altup]
    cb(corrected, "corrected", il);

    return corrected;
}
