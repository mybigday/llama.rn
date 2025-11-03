#include "models.h"

llm_build_rwkv6_base::llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params),
    model(model) {}

lm_ggml_tensor * llm_build_rwkv6_base::build_rwkv6_channel_mix(const llama_layer * layer,
                                                            lm_ggml_tensor *       cur,
                                                            lm_ggml_tensor *       x_prev,
                                                            llm_arch            arch) const {
    lm_ggml_tensor * sx = lm_ggml_sub(ctx0, x_prev, cur);
    switch (arch) {
        case LLM_ARCH_RWKV6:
            {
                lm_ggml_tensor * xk = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, sx, layer->channel_mix_lerp_k), cur);
                lm_ggml_tensor * xr = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, sx, layer->channel_mix_lerp_r), cur);

                lm_ggml_tensor * r = lm_ggml_sigmoid(ctx0, build_lora_mm(layer->channel_mix_receptance, xr));
                lm_ggml_tensor * k = lm_ggml_sqr(ctx0, lm_ggml_relu(ctx0, build_lora_mm(layer->channel_mix_key, xk)));
                cur             = lm_ggml_mul(ctx0, r, build_lora_mm(layer->channel_mix_value, k));
            }
            break;
        default:
            LM_GGML_ABORT("fatal error");
    }
    return cur;
}

lm_ggml_tensor * llm_build_rwkv6_base::build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                                         lm_ggml_tensor *        cur,
                                                         lm_ggml_tensor *        x_prev,
                                                         const llama_ubatch & ubatch,
                                                         int                  il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto n_tokens     = ubatch.n_tokens;
    const auto n_seqs       = ubatch.n_seqs;
    const auto n_seq_tokens = ubatch.n_seq_tokens;
    const auto n_embd       = hparams.n_embd;
    const auto head_size    = hparams.wkv_head_size;
    const auto n_head       = n_embd / head_size;
    const auto n_head_kv    = hparams.n_head_kv(il);

    const auto kv_head = mctx_cur->get_head();

    const auto & layer = model.layers[il];

    bool is_qrwkv = layer.time_mix_first == nullptr;

    lm_ggml_tensor * sx = lm_ggml_sub(ctx0, x_prev, cur);

    sx  = lm_ggml_reshape_2d(ctx0, sx, n_embd, n_tokens);
    cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);

    lm_ggml_tensor * xxx = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, sx, layer.time_mix_lerp_x), cur);

    xxx = lm_ggml_reshape_4d(ctx0, lm_ggml_tanh(ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_w1, xxx)),
                          layer.time_mix_w1->ne[1] / 5, 1, 5, n_tokens);

    xxx = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, xxx, 0, 1, 3, 2));

    xxx = lm_ggml_mul_mat(
        ctx0, lm_ggml_reshape_4d(ctx0, layer.time_mix_w2, layer.time_mix_w2->ne[0], layer.time_mix_w2->ne[1], 1, 5), xxx);

    lm_ggml_tensor *xw, *xk, *xv, *xr, *xg;
    if (layer.time_mix_lerp_fused) {
        // fusing these weights makes some performance improvement
        sx  = lm_ggml_reshape_3d(ctx0, sx, n_embd, 1, n_tokens);
        cur = lm_ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
        xxx = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xxx, layer.time_mix_lerp_fused), sx), cur);
        xw  = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk  = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv  = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr  = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg  = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));
    } else {
        // for backward compatibility
        xw = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], 0);
        xk = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * sizeof(float));
        xv = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 2 * sizeof(float));
        xr = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 3 * sizeof(float));
        xg = lm_ggml_view_2d(ctx0, xxx, n_embd, n_tokens, xxx->nb[1], n_embd * n_tokens * 4 * sizeof(float));

        xw = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xw, layer.time_mix_lerp_w), sx), cur);
        xk = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xk, layer.time_mix_lerp_k), sx), cur);
        xv = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xv, layer.time_mix_lerp_v), sx), cur);
        xr = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xr, layer.time_mix_lerp_r), sx), cur);
        xg = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, lm_ggml_add(ctx0, xg, layer.time_mix_lerp_g), sx), cur);
    }
    lm_ggml_tensor * r = build_lora_mm(layer.time_mix_receptance, xr);
    lm_ggml_tensor * k = build_lora_mm(layer.time_mix_key, xk);
    lm_ggml_tensor * v = build_lora_mm(layer.time_mix_value, xv);
    if (layer.time_mix_receptance_b) {
        r = lm_ggml_add(ctx0, r, layer.time_mix_receptance_b);
    }
    if (layer.time_mix_key_b) {
        k = lm_ggml_add(ctx0, k, layer.time_mix_key_b);
    }
    if (layer.time_mix_value_b) {
        v = lm_ggml_add(ctx0, v, layer.time_mix_value_b);
    }
    lm_ggml_tensor * g = build_lora_mm(layer.time_mix_gate, xg);
    if (is_qrwkv) {
        g = lm_ggml_sigmoid(ctx0, g);
    } else {
        g = lm_ggml_silu(ctx0, g);
    }
    if (n_head_kv != 0 && n_head_kv != n_head) {
        LM_GGML_ASSERT(n_head % n_head_kv == 0);
        k                 = lm_ggml_reshape_4d(ctx0, k, head_size, 1, n_head_kv, n_tokens);
        v                 = lm_ggml_reshape_4d(ctx0, v, head_size, 1, n_head_kv, n_tokens);
        lm_ggml_tensor * tmp = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, head_size, n_head / n_head_kv, n_head_kv, n_tokens);
        k                 = lm_ggml_repeat(ctx0, k, tmp);
        v                 = lm_ggml_repeat(ctx0, v, tmp);
    }
    k = lm_ggml_reshape_3d(ctx0, k, head_size, n_head, n_tokens);
    v = lm_ggml_reshape_3d(ctx0, v, head_size, n_head, n_tokens);
    r = lm_ggml_reshape_3d(ctx0, r, head_size, n_head, n_tokens);

    lm_ggml_tensor * w =
        lm_ggml_mul_mat(ctx0, layer.time_mix_decay_w2, lm_ggml_tanh(ctx0, lm_ggml_mul_mat(ctx0, layer.time_mix_decay_w1, xw)));

    w = lm_ggml_add(ctx0, w, layer.time_mix_decay);
    w = lm_ggml_exp(ctx0, lm_ggml_neg(ctx0, lm_ggml_exp(ctx0, w)));
    w = lm_ggml_reshape_3d(ctx0, w, head_size, n_head, n_tokens);

    if (is_qrwkv) {
        // k = k * (1 - w)
        k = lm_ggml_sub(ctx0, k, lm_ggml_mul(ctx0, k, w));
    }
    lm_ggml_tensor * wkv_state = build_rs(inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

    lm_ggml_tensor * wkv_output;
    if (is_qrwkv) {
        wkv_output = lm_ggml_gated_linear_attn(ctx0, k, v, r, w, wkv_state, pow(head_size, -0.5f));
    } else {
        wkv_output = lm_ggml_rwkv_wkv6(ctx0, k, v, r, layer.time_mix_first, w, wkv_state);
    }
    cur       = lm_ggml_view_1d(ctx0, wkv_output, n_embd * n_tokens, 0);
    wkv_state = lm_ggml_view_1d(ctx0, wkv_output, n_embd * head_size * n_seqs, n_embd * n_tokens * sizeof(float));

    lm_ggml_build_forward_expand(
        gf, lm_ggml_cpy(ctx0, wkv_state,
                     lm_ggml_view_1d(ctx0, mctx_cur->get_s_l(il), hparams.n_embd_s() * n_seqs,
                                  hparams.n_embd_s() * kv_head * lm_ggml_element_size(mctx_cur->get_s_l(il)))));

    if (!is_qrwkv) {
        // group norm with head_count groups
        cur = lm_ggml_reshape_3d(ctx0, cur, n_embd / n_head, n_head, n_tokens);
        cur = lm_ggml_norm(ctx0, cur, 64e-5f);

        // Convert back to regular vectors.
        cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
        cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.time_mix_ln), layer.time_mix_ln_b);
    } else {
        cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    }
    cur = lm_ggml_mul(ctx0, cur, g);
    cur = build_lora_mm(layer.time_mix_output, cur);

    return lm_ggml_reshape_3d(ctx0, cur, n_embd, n_seq_tokens, n_seqs);
}
