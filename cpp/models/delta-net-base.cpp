#include "models.h"

#define CHUNK_SIZE 64

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static lm_ggml_tensor * get_slice_2d(lm_ggml_context * ctx0, lm_ggml_tensor * t, int64_t c) {
    return lm_ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params & params) : llm_graph_context(params) {}

std::pair<lm_ggml_tensor *, lm_ggml_tensor *> llm_build_delta_net_base::build_delta_net_chunking(
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
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    LM_GGML_ASSERT(S_k == S_v);
    LM_GGML_ASSERT(H_v % H_k == 0);

    LM_GGML_ASSERT(q->ne[0] == S_k && q->ne[1] == H_k && q->ne[2] == n_tokens && q->ne[3] == n_seqs);
    LM_GGML_ASSERT(k->ne[0] == S_k && k->ne[1] == H_k && k->ne[2] == n_tokens && k->ne[3] == n_seqs);
    LM_GGML_ASSERT(v->ne[0] == S_v && v->ne[1] == H_v && v->ne[2] == n_tokens && v->ne[3] == n_seqs);

    LM_GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    LM_GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    LM_GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    LM_GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

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
    g = lm_ggml_permute(ctx0, g, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
    b = lm_ggml_permute(ctx0, b, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

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

    g = lm_ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = lm_ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    // [CS, g_0, n_chunks, H_v * n_seqs]
    // TODO: extend lm_ggml_cumsum with axis parameter to avoid transpose
    lm_ggml_tensor * g_cs = lm_ggml_cumsum(ctx0, lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, g)));
    cb(g_cs, "g_cs", il);

    lm_ggml_tensor * kb = nullptr;
    lm_ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;

        lm_ggml_tensor * g_cs_i = lm_ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);  // [chunk_size, 1, S_k, CHB]
        lm_ggml_tensor * g_cs_j = lm_ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB]

        g_cs_j = lm_ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);  // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

        // decay_mask [chunk_size,chunk_size,S_k,CHB]
        lm_ggml_tensor * decay_mask;
        decay_mask = lm_ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = lm_ggml_tri(ctx0, decay_mask, LM_GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = lm_ggml_exp(ctx0, decay_mask);
        cb(decay_mask, "decay_mask", il);

        // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
        decay_mask = lm_ggml_cont_4d(ctx0, lm_ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);

        lm_ggml_tensor * k_b_i = lm_ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        lm_ggml_tensor * k_j   = lm_ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        lm_ggml_tensor * q_i   = lm_ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);

        lm_ggml_tensor * decay_k_b_i = lm_ggml_mul(ctx0, decay_mask, k_b_i);
        lm_ggml_tensor * decay_q_i   = lm_ggml_mul(ctx0, decay_mask, q_i);

        // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
        kb = lm_ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = lm_ggml_mul_mat(ctx0, decay_q_i,   k_j);

        kb = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
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
        kb = lm_ggml_mul_mat(ctx0, k,  k_b);
        kb = lm_ggml_mul    (ctx0, kb, decay_mask);

        // [CS, CS, n_chunks, H_k * n_seqs]
        kq = lm_ggml_mul_mat(ctx0, k, q);
        kq = lm_ggml_mul(ctx0, kq, decay_mask);
    }

    kq = lm_ggml_tri(ctx0, kq, LM_GGML_TRI_TYPE_LOWER_DIAG);
    cb(kq, "kq", il);

    // [CS, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * attn;
    attn = lm_ggml_tri(ctx0, kb, LM_GGML_TRI_TYPE_LOWER);
    cb(attn, "attn", il);

    lm_ggml_tensor * identity;
    identity = lm_ggml_view_1d(ctx0, attn, CS, 0);
    identity = lm_ggml_fill   (ctx0, identity, 1.0f);
    identity = lm_ggml_diag   (ctx0, identity);

    lm_ggml_tensor * lhs = lm_ggml_add(ctx0, attn, identity);
    cb(lhs, "dnet_add_ch_lhs", il);

    attn = lm_ggml_neg(ctx0, attn);
    cb(attn, "attn_pre_solve", il);

    lm_ggml_tensor * lin_solve = lm_ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = lm_ggml_add(ctx0, lin_solve, identity);
    cb(attn, "dnet_add_ch_attn_solved", il); // [CS, CS, n_chunks, H_k * n_seqs]

    // [S_v, CS, n_chunks, H_v * n_seqs]
    v = lm_ggml_mul_mat(ctx0, lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, v_b)), attn);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_exp = lm_ggml_exp(ctx0, g_cs);

    k_b = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, k_b));

    // [CS, S_k, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * kbg = lm_ggml_mul(ctx0, k_b, g_exp);
    cb(kbg, "k_beta_g_exp", il);

    // [S_k, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * k_cd = lm_ggml_mul_mat(ctx0, kbg, attn);
    cb(k_cd, "k_cumdecay", il);

    // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
    lm_ggml_tensor * g_exp_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, g_exp));
    lm_ggml_tensor * q_g_exp = lm_ggml_mul(ctx0, q, g_exp_t);

    // vectorized calculation of key_gdiff
    // improved from the chunked version:
    //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
    //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
    //   key_gdiff = key * g_diff.unsqueeze(-1)
    //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

    // get last element in g_cumsum along CS dimension (ne0)
    // example: [[x, y, z, ..., last], ...] -> [[last], ...]
    // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_last = lm_ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1],
            g_cs->nb[2],
            g_cs->nb[3],
            lm_ggml_row_size(g_cs->type, g_cs->ne[0] - 1));
    cb(g_last, "g_last", il);

    // TODO: remove this cont when CUDA supports non-cont unary ops
    g_last = lm_ggml_cont(ctx0, g_last);

    // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_last_exp_t = lm_ggml_transpose(ctx0, lm_ggml_exp(ctx0, g_last));
    cb(g_last_exp_t, "g_last_exp_t", il);

    // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
    lm_ggml_tensor * g_diff = lm_ggml_neg(ctx0, lm_ggml_sub(ctx0, g_cs, g_last));
    cb(g_diff, "g_diff", il);

    lm_ggml_tensor * g_diff_exp_t = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, lm_ggml_exp(ctx0, g_diff)));

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
        lm_ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);

        s_t = lm_ggml_mul(ctx0, s_t, ch_g_last_exp_t);
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
    s = lm_ggml_transpose(ctx0, s_t);
    cb(s, "output_state", il);

    return {o, s};
}

std::pair<lm_ggml_tensor *, lm_ggml_tensor *> llm_build_delta_net_base::build_delta_net_autoregressive(
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

    LM_GGML_ASSERT(g->ne[0] == 1   || g->ne[0] == S_v);
    LM_GGML_ASSERT(                   g->ne[1] == H_v && g->ne[2] == n_tokens && g->ne[3] == n_seqs);
    LM_GGML_ASSERT(b->ne[0] == 1   && b->ne[1] == H_v && b->ne[2] == n_tokens && b->ne[3] == n_seqs);
    LM_GGML_ASSERT(s->ne[0] == S_v && s->ne[1] == S_v && s->ne[2] == H_v      && s->ne[3] == n_seqs);

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

    // GDA: [1,  1,  H_v, n_seqs]
    // KDA: [1, S_k, H_v, n_seqs]
    g = lm_ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = lm_ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

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
