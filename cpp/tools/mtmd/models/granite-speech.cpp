#include "models.h"

lm_ggml_cgraph * clip_graph_granite_speech::build() {
    const int n_frames     = img.nx;
    const int context_size = hparams.audio_chunk_size;
    const int ctc_layer    = n_layer / 2;
    const int conv_kernel  = hparams.audio_conv_kernel_size;
    const int conv_pad     = conv_kernel / 2;

    const int num_blocks   = (n_frames + context_size - 1) / context_size;
    const int padded_len   = num_blocks * context_size;
    const int remainder    = n_frames % context_size;

    lm_ggml_tensor * attn_dists = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, context_size * context_size);
    lm_ggml_set_name(attn_dists, "attn_dists");
    lm_ggml_set_input(attn_dists);

    lm_ggml_tensor * attn_mask = nullptr;
    if (remainder > 0) {
        attn_mask = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32,
            context_size, context_size, 1, num_blocks);
        lm_ggml_set_name(attn_mask, "attn_mask");
        lm_ggml_set_input(attn_mask);
    }

    lm_ggml_tensor * inp = build_inp_raw(1);
    auto * cur = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inp));
    cb(cur, "inp_transposed", -1);

    cur = build_mm(model.inp_proj_w, cur);
    cur = lm_ggml_add(ctx0, cur, model.inp_proj_b);
    cb(cur, "inp_linear", -1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // ffn1 (half-step)
        {
            auto * ffn1 = build_norm(cur, layer.ff_norm_w, layer.ff_norm_b,
                                     NORM_TYPE_NORMAL, eps, il);
            cb(ffn1, "ffn1_norm", il);

            ffn1 = build_ffn(ffn1,
                layer.ff_up_w, layer.ff_up_b,
                nullptr, nullptr,
                layer.ff_down_w, layer.ff_down_b,
                FFN_SILU, il);
            cb(ffn1, "ffn1_out", il);

            residual = lm_ggml_add(ctx0, residual, lm_ggml_scale(ctx0, ffn1, 0.5f));
            cb(residual, "ffn1_residual", il);
        }

        // build_attn not used here: Shaw RPE needs pos_attn = mul_mat(pos_emb, Q)
        // injected between KQ product and softmax, which build_attn doesn't support
        {
            auto * normed = build_norm(residual, layer.ln_1_w, layer.ln_1_b,
                                       NORM_TYPE_NORMAL, eps, il);
            cb(normed, "attn_norm", il);

            if (n_frames < padded_len) {
                normed = lm_ggml_pad(ctx0, normed, 0, padded_len - n_frames, 0, 0);
            }

            lm_ggml_tensor * Q = build_mm(layer.q_w, normed);
            lm_ggml_tensor * K = build_mm(layer.k_w, normed);
            lm_ggml_tensor * V = build_mm(layer.v_w, normed);

            Q = lm_ggml_reshape_4d(ctx0, Q, d_head, n_head, context_size, num_blocks);
            K = lm_ggml_reshape_4d(ctx0, K, d_head, n_head, context_size, num_blocks);
            V = lm_ggml_reshape_4d(ctx0, V, d_head, n_head, context_size, num_blocks);

            lm_ggml_tensor * Q_perm = lm_ggml_permute(ctx0, Q, 0, 2, 1, 3);
            lm_ggml_tensor * K_perm = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, K, 0, 2, 1, 3));

            lm_ggml_tensor * kq = lm_ggml_mul_mat(ctx0, K_perm, Q_perm);

            // Shaw RPE: pos_emb ne[2]=1 broadcasts against Q ne[2]=num_blocks in mul_mat
            lm_ggml_tensor * pos_emb = lm_ggml_get_rows(ctx0, layer.attn_rel_pos_emb, attn_dists);
            pos_emb = lm_ggml_reshape_3d(ctx0, pos_emb, d_head, context_size, context_size);
            pos_emb = lm_ggml_reshape_4d(ctx0, pos_emb, d_head, context_size, 1, context_size);

            lm_ggml_tensor * Q_shaw = lm_ggml_permute(ctx0, Q, 0, 1, 3, 2);
            lm_ggml_tensor * pos_attn = lm_ggml_mul_mat(ctx0, pos_emb, Q_shaw);
            pos_attn = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, pos_attn, 0, 2, 3, 1));

            lm_ggml_tensor * scores = lm_ggml_add(ctx0, kq, pos_attn);
            lm_ggml_tensor * attn_weights = lm_ggml_soft_max_ext(ctx0, scores, attn_mask,
                                                            kq_scale, 0.0f);

            lm_ggml_tensor * V_perm = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, V, 1, 2, 0, 3));
            lm_ggml_tensor * attn_out = lm_ggml_mul_mat(ctx0, V_perm, attn_weights);

            attn_out = lm_ggml_permute(ctx0, attn_out, 0, 2, 1, 3);
            attn_out = lm_ggml_cont_2d(ctx0, attn_out, n_embd, padded_len);

            if (n_frames < padded_len) {
                attn_out = lm_ggml_view_2d(ctx0, attn_out,
                    n_embd, n_frames, attn_out->nb[1], 0);
            }

            cur = build_mm(layer.o_w, attn_out);
            cur = lm_ggml_add(ctx0, cur, layer.o_b);
            cb(cur, "attn_out", il);
        }

        residual = lm_ggml_add(ctx0, residual, cur);

        // conv module
        {
            cur = build_norm(residual, layer.norm_conv_w, layer.norm_conv_b,
                             NORM_TYPE_NORMAL, eps, il);
            cb(cur, "conv_norm", il);

            auto * x = build_mm(layer.conv_pw1_w, cur);
            x = lm_ggml_add(ctx0, x, layer.conv_pw1_b);
            cb(x, "conv_pw1", il);

            // GLU: ggml has no fused op, manual split + sigmoid gate
            {
                int64_t d = x->ne[0] / 2;
                lm_ggml_tensor * gate = lm_ggml_sigmoid(ctx0,
                    lm_ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x = lm_ggml_mul(ctx0,
                    lm_ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, x));
            }
            cb(x, "conv_glu", il);

            x = lm_ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = lm_ggml_roll(ctx0, x, conv_pad, 0, 0, 0);
            x = lm_ggml_pad(ctx0, x, conv_pad, 0, 0, 0);
            x = lm_ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            cb(x, "conv_dw", il);

            // folded batch norm
            x = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, x, layer.conv_norm_w), layer.conv_norm_b);
            x = lm_ggml_silu(ctx0, x);
            cb(x, "conv_bn_silu", il);

            x = build_mm(layer.conv_pw2_w, x);
            x = lm_ggml_add(ctx0, x, layer.conv_pw2_b);
            cb(x, "conv_pw2", il);

            cur = x;
        }

        residual = lm_ggml_add(ctx0, residual, cur);

        // ffn2 (half-step)
        {
            auto * ffn2 = build_norm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b,
                                     NORM_TYPE_NORMAL, eps, il);
            cb(ffn2, "ffn2_norm", il);

            ffn2 = build_ffn(ffn2,
                layer.ff_up_1_w, layer.ff_up_1_b,
                nullptr, nullptr,
                layer.ff_down_1_w, layer.ff_down_1_b,
                FFN_SILU, il);
            cb(ffn2, "ffn2_out", il);

            residual = lm_ggml_add(ctx0, residual, lm_ggml_scale(ctx0, ffn2, 0.5f));
        }

        cur = build_norm(residual, layer.ln_2_w, layer.ln_2_b,
                         NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_out", il);

        // CTC branch
        if (il + 1 == ctc_layer) {
            auto * mid = build_mm(model.ctc_out_w, cur);
            mid = lm_ggml_add(ctx0, mid, model.ctc_out_b);
            mid = lm_ggml_soft_max(ctx0, mid);
            mid = build_mm(model.ctc_out_mid_w, mid);
            mid = lm_ggml_add(ctx0, mid, model.ctc_out_mid_b);
            cur = lm_ggml_add(ctx0, cur, mid);
            cb(cur, "ctc_branch", il);
        }
    }

    cb(cur, "encoder_out", -1);

    // QFormer projector
    {
        const int window_size     = hparams.audio_proj_window_size;
        const int num_queries     = window_size / hparams.audio_proj_downsample_rate;
        const int proj_n_head     = hparams.audio_proj_head_count;
        const int proj_d_head     = n_embd / proj_n_head;
        const float proj_kq_scale = 1.0f / sqrtf((float)proj_d_head);
        const float proj_eps      = 1e-12f;
        const int nblocks_proj    = (n_frames + window_size - 1) / window_size;
        const int padded_proj     = nblocks_proj * window_size;

        if (n_frames < padded_proj) {
            cur = lm_ggml_pad(ctx0, cur, 0, padded_proj - n_frames, 0, 0);
        }

        lm_ggml_tensor * enc_windows = lm_ggml_reshape_3d(ctx0, cur, n_embd, window_size, nblocks_proj);

        lm_ggml_tensor * queries = build_norm(model.qf_proj_query,
            model.qf_proj_norm_w, model.qf_proj_norm_b,
            NORM_TYPE_NORMAL, proj_eps, -1);
        {
            lm_ggml_tensor * q_3d    = lm_ggml_reshape_3d(ctx0, queries, n_embd, num_queries, 1);
            lm_ggml_tensor * q_shape = lm_ggml_new_tensor_3d(ctx0, LM_GGML_TYPE_F32,
                n_embd, num_queries, nblocks_proj);
            queries = lm_ggml_repeat(ctx0, q_3d, q_shape);
        }

        for (int il = 0; il < (int)model.qf_proj_layers.size(); il++) {
            const auto & pl = model.qf_proj_layers[il];

            // self-attention
            {
                lm_ggml_tensor * Q = lm_ggml_add(ctx0, build_mm(pl.q_w, queries), pl.q_b);
                lm_ggml_tensor * K = lm_ggml_add(ctx0, build_mm(pl.k_w, queries), pl.k_b);
                lm_ggml_tensor * V = lm_ggml_add(ctx0, build_mm(pl.v_w, queries), pl.v_b);

                Q = lm_ggml_reshape_4d(ctx0, Q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                K = lm_ggml_reshape_4d(ctx0, K, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                V = lm_ggml_reshape_4d(ctx0, V, proj_d_head, proj_n_head, num_queries, nblocks_proj);

                lm_ggml_tensor * sa_out = build_attn(pl.o_w, pl.o_b,
                    Q, K, V, nullptr, proj_kq_scale, il);
                sa_out = lm_ggml_reshape_3d(ctx0, sa_out, n_embd, num_queries, nblocks_proj);

                queries = build_norm(lm_ggml_add(ctx0, sa_out, queries),
                    pl.ln_1_w, pl.ln_1_b,
                    NORM_TYPE_NORMAL, proj_eps, il);
            }

            // cross-attention
            {
                lm_ggml_tensor * Q = lm_ggml_add(ctx0, build_mm(pl.cross_attn_q_w, queries), pl.cross_attn_q_b);
                lm_ggml_tensor * K = lm_ggml_add(ctx0, build_mm(pl.cross_attn_k_w, enc_windows), pl.cross_attn_k_b);
                lm_ggml_tensor * V = lm_ggml_add(ctx0, build_mm(pl.cross_attn_v_w, enc_windows), pl.cross_attn_v_b);

                Q = lm_ggml_reshape_4d(ctx0, Q, proj_d_head, proj_n_head, num_queries, nblocks_proj);
                K = lm_ggml_reshape_4d(ctx0, K, proj_d_head, proj_n_head, window_size, nblocks_proj);
                V = lm_ggml_reshape_4d(ctx0, V, proj_d_head, proj_n_head, window_size, nblocks_proj);

                lm_ggml_tensor * ca_out = build_attn(pl.cross_attn_o_w, pl.cross_attn_o_b,
                    Q, K, V, nullptr, proj_kq_scale, il);
                ca_out = lm_ggml_reshape_3d(ctx0, ca_out, n_embd, num_queries, nblocks_proj);

                queries = build_norm(lm_ggml_add(ctx0, ca_out, queries),
                    pl.cross_attn_norm_w, pl.cross_attn_norm_b,
                    NORM_TYPE_NORMAL, proj_eps, il);
            }

            // ffn
            {
                lm_ggml_tensor * ffn_out = build_ffn(queries,
                    pl.ff_up_w, pl.ff_up_b,
                    nullptr, nullptr,
                    pl.ff_down_w, pl.ff_down_b,
                    FFN_GELU, il);

                queries = build_norm(lm_ggml_add(ctx0, ffn_out, queries),
                    pl.ln_2_w, pl.ln_2_b,
                    NORM_TYPE_NORMAL, proj_eps, il);
            }
        }

        cur = lm_ggml_reshape_2d(ctx0, queries, n_embd, num_queries * nblocks_proj);
        cur = lm_ggml_add(ctx0, build_mm(model.qf_proj_linear_w, cur), model.qf_proj_linear_b);
        cb(cur, "projector_out", -1);
    }

    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
