#include "models.h"

llm_build_rwkv7_base::llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {}

lm_ggml_tensor * llm_build_rwkv7_base::build_rwkv7_channel_mix(const llama_layer * layer,
                                                            lm_ggml_tensor *       cur,
                                                            lm_ggml_tensor *       x_prev,
                                                            llm_arch            arch) const {
    lm_ggml_tensor * sx = lm_ggml_sub(ctx0, x_prev, cur);
    switch (arch) {
        case LLM_ARCH_RWKV7:
            {
                lm_ggml_tensor * xk = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);

                lm_ggml_tensor * k = lm_ggml_sqr(ctx0, lm_ggml_relu(ctx0, build_lora_mm(layer->channel_mix_key, xk)));

                cur = build_lora_mm(layer->channel_mix_value, k);
            }
            break;
        default:
            LM_GGML_ABORT("fatal error");
    }
    return cur;
}

lm_ggml_tensor * llm_build_rwkv7_base::build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                                         lm_ggml_tensor *        cur,
                                                         lm_ggml_tensor *        x_prev,
                                                         lm_ggml_tensor *&       first_layer_value,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto n_tokens     = ubatch.n_tokens;
    const auto n_seqs       = ubatch.n_seqs;
    const auto n_embd       = hparams.n_embd;
    const auto head_size    = hparams.wkv_head_size;
    const auto head_count   = n_embd / head_size;
    const auto n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    bool has_gating = layer.time_mix_g1 && layer.time_mix_g2;

    lm_ggml_tensor * sx    = lm_ggml_sub(ctx0, x_prev, cur);
    lm_ggml_tensor * dummy = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, n_embd, n_seq_tokens, n_seqs, has_gating ? 6 : 5);
    sx                  = lm_ggml_repeat(ctx0, sx, dummy);

    lm_ggml_tensor * xxx = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, sx, layer.time_mix_lerp_fused), cur);

    lm_ggml_tensor * xr = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
    lm_ggml_tensor * xw = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
    lm_ggml_tensor * xk = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
    lm_ggml_tensor * xv = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
    lm_ggml_tensor * xa = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    lm_ggml_tensor * xg =
        has_gating ? lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 5 * sizeof(float)) :
                     nullptr;

    lm_ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
    lm_ggml_tensor * w = lm_ggml_add(
        ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_w2, lm_ggml_tanh(ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_w1, xw))),
        layer.time_mix_w0);
    w = lm_ggml_exp(ctx0, lm_ggml_scale(ctx0, lm_ggml_sigmoid(ctx0, w), -0.606531));

    lm_ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    lm_ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
    if (first_layer_value == nullptr) {
        first_layer_value = v;
    } else {
        // Add the first layer value as a residual connection.
        v = lm_ggml_add(ctx0, v,
                     lm_ggml_mul(ctx0, lm_ggml_sub(ctx0, first_layer_value, v),
                              lm_ggml_sigmoid(ctx0, lm_ggml_add(ctx0,
                                                          lm_ggml_mul_mat(ctx0, layer.time_mix_v2,
                                                                       lm_ggml_mul_mat(ctx0, layer.time_mix_v1, xv)),
                                                          layer.time_mix_v0))));
    }
    lm_ggml_tensor * g = nullptr;
    if (layer.time_mix_g1 && layer.time_mix_g2) {
        g = lm_ggml_mul_mat(ctx0, layer.time_mix_g2, lm_ggml_sigmoid(ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_g1, xg)));
    }
    lm_ggml_tensor * a = lm_ggml_sigmoid(
        ctx0, lm_ggml_add(ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_a2, lm_ggml_mul_mat(ctx0, layer.time_mix_a1, xa)),
                       layer.time_mix_a0));

    lm_ggml_tensor * kk = lm_ggml_reshape_3d(ctx0, lm_ggml_mul(ctx0, k, layer.time_mix_k_k), head_size, head_count, n_tokens);
    kk               = lm_ggml_l2_norm(ctx0, kk, 1e-12);

    lm_ggml_tensor * ka = lm_ggml_mul(ctx0, k, layer.time_mix_k_a);
    k                = lm_ggml_add(ctx0, k, lm_ggml_sub(ctx0, lm_ggml_mul(ctx0, a, ka), ka));

    r = lm_ggml_reshape_3d(ctx0, r, head_size, head_count, n_tokens);
    w = lm_ggml_reshape_3d(ctx0, w, head_size, head_count, n_tokens);
    k = lm_ggml_reshape_3d(ctx0, k, head_size, head_count, n_tokens);
    v = lm_ggml_reshape_3d(ctx0, v, head_size, head_count, n_tokens);
    a = lm_ggml_reshape_3d(ctx0, a, head_size, head_count, n_tokens);

    lm_ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    lm_ggml_tensor * wkv_output = lm_ggml_rwkv_wkv7(ctx0, r, w, k, v, lm_ggml_neg(ctx0, kk), lm_ggml_mul(ctx0, kk, a), wkv_state);
    cur                      = lm_ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = lm_ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    lm_ggml_build_forward_expand(
        gf, lm_ggml_cpy(ctx0, wkv_state,
                     lm_ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * lm_ggml_element_size(mctx_cur->get_s_l(il)))));

    if (layer.time_mix_ln && layer.time_mix_ln_b) {
        // group norm with head_count groups
        cur = lm_ggml_reshape_3d(ctx0, cur, n_embd / head_count, head_count, n_tokens);
        cur = lm_ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }
    lm_ggml_tensor * rk = lm_ggml_sum_rows(
        ctx0, lm_ggml_mul(ctx0, lm_ggml_mul(ctx0, k, r), lm_ggml_reshape_2d(ctx0, layer.time_mix_r_k, head_size, head_count)));
    cur = lm_ggml_add(ctx0, cur, lm_ggml_reshape_2d(ctx0, lm_ggml_mul(ctx0, v, rk), n_embd, n_tokens));

    if (has_gating) {
        cur = lm_ggml_mul(ctx0, cur, g);
    }
    cur = build_lora_mm(layer.time_mix_output, cur);

    return lm_ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
}
