#include "ggml.h"
#include "models.h"

#define CHUNK_SIZE 64

llm_build_qwen3next::llm_build_qwen3next(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context_mamba(params), model(model) {
    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp = build_inp_mem_hybrid();

    lm_ggml_tensor * inp_pos     = build_inp_pos();
    lm_ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        lm_ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = lm_ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = lm_ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = lm_ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        lm_ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        lm_ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // FFN layer (MoE or dense) - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = lm_ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static lm_ggml_tensor * get_slice_2d(lm_ggml_context * ctx0, lm_ggml_tensor * t, int64_t c) {
    return lm_ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

std::pair<lm_ggml_tensor *, lm_ggml_tensor *> llm_build_qwen3next::build_delta_net_chunking(
        lm_ggml_tensor * q,
        lm_ggml_tensor * k,
        lm_ggml_tensor * v,
        lm_ggml_tensor * g,
        lm_ggml_tensor * b,
        lm_ggml_tensor * s,
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    LM_GGML_ASSERT(S_k == S_v);
    LM_GGML_ASSERT(H_v % H_k == 0);

    LM_GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    LM_GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    LM_GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    LM_GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    LM_GGML_ASSERT(b->ne[0] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    LM_GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = lm_ggml_scale(ctx0, q, scale);

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    q = lm_ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = lm_ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = lm_ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
    g = lm_ggml_permute(ctx0, g, 2, 1, 3, 0); // [  1, n_tokens, H_v, n_seqs]
    b = lm_ggml_permute(ctx0, b, 2, 0, 1, 3); // [  1, n_tokens, H_v, n_seqs]

    const int CS = CHUNK_SIZE;

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = lm_ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = lm_ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = lm_ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = lm_ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = lm_ggml_pad(ctx0, b, 0, pad, 0, 0);

    lm_ggml_tensor * v_b = lm_ggml_mul(ctx0, v, b);
    lm_ggml_tensor * k_b = lm_ggml_mul(ctx0, k, b);

    cb(v_b, "v_b", il);
    cb(k_b, "k_b", il);

    q   = lm_ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = lm_ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = lm_ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = lm_ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = lm_ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = lm_ggml_reshape_4d(ctx0, g, CS, 1, n_chunks, H_v * n_seqs);
    b = lm_ggml_reshape_4d(ctx0, b, 1, CS, n_chunks, H_v * n_seqs);

    // [CS, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_cs = lm_ggml_cumsum(ctx0, g);
    cb(g_cs, "g_cs", il);

    lm_ggml_tensor * g_cs_i = g_cs;
    lm_ggml_tensor * g_cs_j = lm_ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);

    g_cs_j = lm_ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);

    // [CS, CS, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * decay_mask;
    decay_mask = lm_ggml_sub(ctx0, g_cs_j, g_cs_i);
    decay_mask = lm_ggml_tri(ctx0, decay_mask, LM_GGML_TRI_TYPE_LOWER_DIAG);
    decay_mask = lm_ggml_exp(ctx0, decay_mask);
    cb(decay_mask, "decay_mask", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * kb;
    kb = lm_ggml_mul_mat(ctx0, k,  k_b);
    kb = lm_ggml_mul    (ctx0, kb, decay_mask);

    // [CS, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * attn;
    attn = lm_ggml_tri(ctx0, kb, LM_GGML_TRI_TYPE_LOWER);

    lm_ggml_tensor * identity;
    identity = lm_ggml_view_1d(ctx0, attn, CS, 0);
    identity = lm_ggml_fill   (ctx0, identity, 1.0f);
    identity = lm_ggml_diag   (ctx0, identity);

    lm_ggml_tensor * lhs = lm_ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = lm_ggml_neg(ctx0, attn);

    lm_ggml_tensor * lin_solve = lm_ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = lm_ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = lm_ggml_mul_mat(ctx0, lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_exp = lm_ggml_exp(ctx0, g_cs);

    k_b = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * kbg = lm_ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * k_cd = lm_ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * g_exp_t = lm_ggml_transpose(ctx0, g_exp);
    lm_ggml_tensor * q_g_exp = lm_ggml_mul(ctx0, q, g_exp_t);

    // [CS, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * kq = lm_ggml_mul_mat(ctx0, k, q);
    kq = lm_ggml_mul(ctx0, kq, decay_mask);
    kq = lm_ggml_tri(ctx0, kq, LM_GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_last = lm_ggml_view_4d(ctx0, g_cs, 1, 1, g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            lm_ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = lm_ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_last_exp = lm_ggml_exp(ctx0, g_last);
    cb(g_last_exp, "g_last_exp", il);

    // [CS, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_diff = lm_ggml_neg(ctx0, lm_ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    lm_ggml_tensor * g_diff_exp   = lm_ggml_exp(ctx0, g_diff);
    lm_ggml_tensor * g_diff_exp_t = lm_ggml_transpose(ctx0, g_diff_exp);

    // [S_k, CS, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * kg = lm_ggml_mul(ctx0, k, g_diff_exp_t);
    cb(kg, "key_gdiff", il);

    // [CS, S_k, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * kg_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, kg));
    cb(kg_t, "key_gdiff_t", il);

    lm_ggml_tensor * s_t = lm_ggml_transpose(ctx0, s);
    s_t = lm_ggml_cont_4d(ctx0, s_t, S_v, S_v, 1, H_v * n_seqs);
    cb(s_t, "dnet_add_ch_state", il);

    // [CS, S_v, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * v_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        lm_ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk); // [S_k,  CS, 1, H_k * n_seqs]
        lm_ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk); // [ CS, S_v, 1, H_v * n_seqs]
        lm_ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk); // [ CS,  CS, 1, H_k * n_seqs]
        lm_ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk); // [S_k,  CS, 1, H_k * n_seqs]
        lm_ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk); // [ CS, S_k, 1, H_v * n_seqs]

        // [CS, S_v, 1, H_v * n_seqs]
        lm_ggml_tensor * v_t_p = lm_ggml_mul_mat(ctx0, ch_k_cd, s_t);
        cb(v_t_p, "v_prime", il);

        // [CS, S_v, 1, H_v * n_seqs]
        lm_ggml_tensor * v_t_new = lm_ggml_sub(ctx0, ch_v_t, v_t_p);
        cb(v_t_new, "v_t_new", il);

        // [S_v, CS, 1, H_v * n_seqs]
        lm_ggml_tensor * v_attn = lm_ggml_mul_mat(ctx0, v_t_new, ch_kq);
        cb(v_attn, "v_attn", il);

        // [S_v, CS, 1, H_v * n_seqs]
        lm_ggml_tensor * attn_inter = lm_ggml_mul_mat(ctx0, s_t, ch_q_g_exp);
        cb(attn_inter, "attn_inter", il);

        // [S_v, CS, 1, H_v * n_seqs]
        lm_ggml_tensor * o_ch = lm_ggml_add(ctx0, attn_inter, v_attn);
        cb(o_ch, "dnet_add_ch_attn_out", il);

        v = lm_ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
        // TODO: head broadcast might not work here - probably will need a transpose
        lm_ggml_tensor * kgv = lm_ggml_mul_mat(ctx0, ch_kg_t, v_t_new); // [S_k, S_v, 1, H_k * n_seqs]

        // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
        lm_ggml_tensor * ch_g_last_exp = get_slice_2d(ctx0, g_last_exp, chunk);
        s_t = lm_ggml_mul(ctx0, s_t, ch_g_last_exp);
        s_t = lm_ggml_add(ctx0, s_t, kgv);
        cb(s_t, "dnet_add_ch_state", il);
    }

    s_t = lm_ggml_reshape_4d(ctx0, s_t, S_v, S_v, H_v, n_seqs);

    // truncate padded tokens
    lm_ggml_tensor * o = lm_ggml_view_4d(ctx0, v,
            S_v, n_tokens, H_v, n_seqs,
            lm_ggml_row_size(v->type, S_v),
            lm_ggml_row_size(v->type, S_v * CS * n_chunks),
            lm_ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);

    o = lm_ggml_permute  (ctx0, o, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = lm_ggml_transpose(ctx0, s_t);           // [S_v, S_v, H_v, n_seqs]

    return {o, s};
}

std::pair<lm_ggml_tensor *, lm_ggml_tensor *> llm_build_qwen3next::build_delta_net_autoregressive(
        lm_ggml_tensor * q,
        lm_ggml_tensor * k,
        lm_ggml_tensor * v,
        lm_ggml_tensor * g,
        lm_ggml_tensor * b, // beta
        lm_ggml_tensor * s, // state
        int           il) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];

    LM_GGML_ASSERT(n_tokens == 1);

    LM_GGML_ASSERT(S_k == S_v);
    LM_GGML_ASSERT(H_v % H_k == 0);

    LM_GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    LM_GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    LM_GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    LM_GGML_ASSERT(g->ne[0] == H_v && g->ne[1] == n_tokens && g->ne[2] == n_seqs);
    LM_GGML_ASSERT(b->ne[0] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    LM_GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v && s->ne[3] == n_seqs);

    const float scale = 1.0f / sqrtf(S_k);

    q = lm_ggml_scale(ctx0, q, scale);

    q = lm_ggml_permute(ctx0, q, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    k = lm_ggml_permute(ctx0, k, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
    v = lm_ggml_permute(ctx0, v, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

    cb(q, "q_in", il);
    cb(k, "k_in", il);
    cb(v, "v_in", il);
    cb(b, "b_in", il);
    cb(g, "g_in", il);

    g = lm_ggml_reshape_4d(ctx0, g, 1, 1, H_v, n_seqs);
    b = lm_ggml_reshape_4d(ctx0, b, 1, 1, H_v, n_seqs);

    // [S_v, S_v, H_v, n_seqs]
    g = lm_ggml_exp(ctx0, g);
    s = lm_ggml_mul(ctx0, s, g);

    lm_ggml_tensor * s_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, s));

    // [1, S_v, H_v, n_seqs]
    lm_ggml_tensor * sk;
    sk = lm_ggml_mul     (ctx0, s_t, k);
    sk = lm_ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    lm_ggml_tensor * d;
    d = lm_ggml_sub(ctx0, v, lm_ggml_transpose(ctx0, sk));
    d = lm_ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    lm_ggml_tensor * d_t;
    d_t = lm_ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    lm_ggml_tensor * kd;
    k  = lm_ggml_repeat(ctx0, k, s);
    kd = lm_ggml_mul   (ctx0, k, d_t);

    s_t = lm_ggml_add(ctx0, s_t, kd);

    cb(s_t, "dnet_add_ar_state", il);

    lm_ggml_tensor * s_q = lm_ggml_mul     (ctx0, s_t, q);
    lm_ggml_tensor * o   = lm_ggml_sum_rows(ctx0, s_q);

    o = lm_ggml_permute  (ctx0, o, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
    s = lm_ggml_transpose(ctx0, s_t);           // [S_v, S_v, H_v, n_seqs]

    return {o, s};
}

lm_ggml_tensor * llm_build_qwen3next::build_norm_gated(
        lm_ggml_tensor * input,
        lm_ggml_tensor * weights,
        lm_ggml_tensor * gate,
        int           layer) {
    lm_ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    lm_ggml_tensor * gated_silu = lm_ggml_silu(ctx0, gate);

    return lm_ggml_mul(ctx0, normalized, gated_silu);
}

lm_ggml_tensor * llm_build_qwen3next::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        lm_ggml_tensor *             cur,
        lm_ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    LM_GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    lm_ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    Qcur_full = lm_ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);

    // Split Q projection into query and gate
    // The split should be along dimension 0 (the feature dimension)
    lm_ggml_tensor * Qcur = lm_ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                                            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    cb(Qcur, "Qcur_view", il);

    lm_ggml_tensor * gate =
        lm_ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                     Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * lm_ggml_element_size(Qcur_full));
    cb(gate, "gate", il);

    lm_ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    lm_ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    Kcur = lm_ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = lm_ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    Qcur = lm_ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = lm_ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    // TODO: CUDA is missing non-contiguous unary ops. when implemented: remove this cont
    gate = lm_ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    gate = lm_ggml_sigmoid(ctx0, gate);
    cb(gate, "gate_sigmoid", il);

    gate = lm_ggml_reshape_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    cur = lm_ggml_mul(ctx0, cur, gate);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

std::pair<lm_ggml_tensor *, lm_ggml_tensor *> llm_build_qwen3next::build_qkvz(
                lm_ggml_tensor * input,
                        int   il) {
    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv) {
        // optimized path
        lm_ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
        qkv_mixed = lm_ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        lm_ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input);
        cb(z, "z", il);

        return { qkv_mixed, z };
    } else {
        // legacy (slower) path
        lm_ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, input);
        cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

        int64_t       qkvz_new_dim        = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
        lm_ggml_tensor * mixed_qkvz_reshaped = lm_ggml_reshape_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

        // Split mixed_qkvz into query, key, value, z
        int64_t split_sizes_qkvz[4] = {
            head_k_dim,                              // query size
            head_k_dim,                              // key size
            head_v_dim * num_v_heads / num_k_heads,  // value size
            head_v_dim * num_v_heads / num_k_heads   // z size
        };

        lm_ggml_tensor * query =
            lm_ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
        cb(query, "q", il);

        lm_ggml_tensor * key = lm_ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_seq_tokens, n_seqs,
                                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                        split_sizes_qkvz[0] * lm_ggml_element_size(mixed_qkvz_reshaped));
        cb(key, "k", il);

        lm_ggml_tensor * value =
            lm_ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * lm_ggml_element_size(mixed_qkvz_reshaped));
        cb(value, "v", il);

        lm_ggml_tensor * z = lm_ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_seq_tokens, n_seqs,
                                    mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                    (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * lm_ggml_element_size(mixed_qkvz_reshaped));
        z = lm_ggml_cont(ctx0, z);
        cb(z, "z", il);

        // After creating query, key, and value_reshaped, reshape each to flatten the head dimensions
        // query: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
        lm_ggml_tensor * query_flat = lm_ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        cb(query_flat, "query_flat", il);

        // key: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
        lm_ggml_tensor * key_flat = lm_ggml_cont_3d(ctx0, key, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        cb(key_flat, "key_flat", il);

        // value_reshaped: [head_v_dim, num_v_heads, n_tokens, n_seqs] -> [head_v_dim * num_v_heads, n_tokens, n_seqs]
        lm_ggml_tensor * value_flat = lm_ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
        cb(value_flat, "value_flat", il);

        // Now concatenate along the feature dimension (dim 0) to get [conv_dim, n_tokens, n_seqs]
        lm_ggml_tensor * qkv_mixed = lm_ggml_concat(ctx0, query_flat, key_flat, 0);
        qkv_mixed               = lm_ggml_concat(ctx0, qkv_mixed, value_flat, 0);
        cb(qkv_mixed, "qkv_mixed", il);

        return { qkv_mixed, z };
    }
}

lm_ggml_tensor * llm_build_qwen3next::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        lm_ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    LM_GGML_ASSERT(n_seqs != 0);
    LM_GGML_ASSERT(ubatch.equal_seqs());
    LM_GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    lm_ggml_tensor * qkv_mixed = qkvz.first;
    lm_ggml_tensor * z         = qkvz.second;

    lm_ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
    cb(mixed_ba, "linear_attn_mixed_ba", il);

    // Reshape mixed_ba: [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    int64_t       ba_new_dim        = 2 * num_v_heads / num_k_heads;
    lm_ggml_tensor * mixed_ba_reshaped = lm_ggml_reshape_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Split mixed_ba into b and a (beta and alpha parameters)
    int64_t split_sizes_ba[2] = {
        num_v_heads / num_k_heads,  // beta size
        num_v_heads / num_k_heads   // alpha size
    };

    lm_ggml_tensor * b = lm_ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
    cb(b, "b", il);

    lm_ggml_tensor * a = lm_ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3],
                                   split_sizes_ba[0] * lm_ggml_element_size(mixed_ba_reshaped));
    cb(a, "a", il);

    // TODO: CUDA is missing non-contiguous unary ops. when implemented: remove this cont
    b = lm_ggml_cont(ctx0, b);

    lm_ggml_tensor * beta = lm_ggml_sigmoid(ctx0, b);

    beta = lm_ggml_reshape_4d(ctx0, beta, num_v_heads, 1, n_seq_tokens, n_seqs);

    // Reshape a to merge head dimensions: [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    lm_ggml_tensor * alpha = lm_ggml_cont_3d(ctx0, a, num_v_heads, n_seq_tokens, n_seqs);

    lm_ggml_tensor * alpha_biased   = lm_ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    lm_ggml_tensor * alpha_softplus = lm_ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    lm_ggml_tensor * gate = lm_ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    // Get convolution states from cache
    lm_ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    lm_ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    lm_ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    lm_ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = lm_ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = lm_ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    lm_ggml_tensor * conv_input = lm_ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    lm_ggml_tensor * last_conv_states =
        lm_ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * lm_ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    lm_ggml_tensor * state_update_target =
        lm_ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * lm_ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    lm_ggml_build_forward_expand(gf, lm_ggml_cpy(ctx0, last_conv_states, state_update_target));
    cb(conv_states_all, "conv_states_updated", il);

    lm_ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = lm_ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    lm_ggml_tensor * conv_output_proper = lm_ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    lm_ggml_tensor * conv_output_silu = lm_ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    lm_ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = lm_ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    lm_ggml_tensor * q_conv = lm_ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            lm_ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    lm_ggml_tensor * k_conv = lm_ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            lm_ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * lm_ggml_element_size(conv_qkv_mix));

    lm_ggml_tensor * v_conv = lm_ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            lm_ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            lm_ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = lm_ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = lm_ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = lm_ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = lm_ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = lm_ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        LM_GGML_ASSERT(num_v_heads % num_k_heads == 0);
        int64_t repeat_factor = num_v_heads / num_k_heads;

        // repeat interleave: reshape to (repeat part, 1, remaining part), do repeat, then reshape back
        lm_ggml_tensor * q_reshaped = lm_ggml_reshape_3d(ctx0, q_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);
        lm_ggml_tensor * k_reshaped = lm_ggml_reshape_3d(ctx0, k_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);

        // Repeat along the third dimension (the new dimension with size 1)
        lm_ggml_tensor * q_repeated =
            lm_ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);
        lm_ggml_tensor * k_repeated =
            lm_ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);

        // Reshape back to merge the head and repeat dimensions
        // From [head_dim, num_k_heads, repeat_factor, n_seq_tokens * n_seqs]
        // Back to [head_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs]
        q_conv = lm_ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        k_conv = lm_ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
    std::pair<lm_ggml_tensor *, lm_ggml_tensor *> attn_out; // pair of (output, new_state)
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, il);
    }
    lm_ggml_tensor * output    = attn_out.first;
    lm_ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // Update the recurrent states
    lm_ggml_build_forward_expand(gf,
            lm_ggml_cpy(ctx0, new_state,
                lm_ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * lm_ggml_element_size(ssm_states_all))));

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    lm_ggml_tensor * z_2d = lm_ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    lm_ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    lm_ggml_tensor * final_output = lm_ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = lm_ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

lm_ggml_tensor * llm_build_qwen3next::build_layer_ffn(lm_ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    if (model.layers[il].ffn_gate_inp != nullptr) {
        // MoE branch
        lm_ggml_tensor * moe_out =
            build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used, LLM_FFN_SILU,
                true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        cb(moe_out, "ffn_moe_out", il);

        // Add shared experts if present - following Qwen3Next reference implementation
        if (model.layers[il].ffn_up_shexp != nullptr) {
            lm_ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // Apply shared expert gating as in the reference implementation
            // The shared expert has its own gate that is sigmoided
            // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
            lm_ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(shared_gate, "shared_expert_gate", il);

            shared_gate = lm_ggml_sigmoid(ctx0, shared_gate);
            cb(shared_gate, "shared_expert_gate_sigmoid", il);

            ffn_shexp = lm_ggml_mul(ctx0, ffn_shexp, shared_gate);
            cb(ffn_shexp, "ffn_shexp_gated", il);

            cur = lm_ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    } else {
        // Dense FFN branch (not currently used I believe)
        cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    }
    return cur;
}
