#include "models.h"

static constexpr int PARAKEET_LOCAL_ATTN_THRESHOLD = 8192;
static constexpr int PARAKEET_LOCAL_ATTN_WINDOW    = 128;

// conv subsampling + conformer encoder
lm_ggml_cgraph * clip_graph_parakeet::build() {

    // Conv subsampling
    lm_ggml_tensor * inp = build_inp_raw(1);
    inp = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inp));

    // [freq, time, channels, batch]
    lm_ggml_tensor * cur = lm_ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], inp, 2, 2, 1, 1, 1, 1);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
    cb(cur, "pre_conv_0", -1);

    cur = lm_ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_0_relu", -1);

    // [freq, time, channels, batch]
    cur = lm_ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
    cb(cur, "pre_conv_2", -1);

    // [freq, time, channels, batch]
    cur = lm_ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
    cb(cur, "pre_conv_3", -1);

    cur = lm_ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_3_relu", -1);

    // [freq, time, channels, batch]
    cur = lm_ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
    cb(cur, "pre_conv_5_direct", -1);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
    cb(cur, "pre_conv_5", -1);

    // [freq, time, channels, batch]
    cur = lm_ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
    cb(cur, "pre_conv_6", -1);

    cur = lm_ggml_relu(ctx0, cur);
    cb(cur, "pre_conv_6_relu", -1);

    // [freq, time, chan]
    cur = lm_ggml_permute(ctx0, cur, 0, 2, 1, 3);
    // [freq, chan, time]
    cur = lm_ggml_cont(ctx0, cur);

    const int n_freq   = cur->ne[0];
    const int n_chan   = cur->ne[1];
    const int n_frames = cur->ne[2];

    // [freq, time, chan, batch] -> [(freq * chan), time]
    cur = lm_ggml_reshape_2d(ctx0, cur, n_freq * n_chan, n_frames);

    cur = build_mm(model.pre_encode_out_w, cur);
    cur = lm_ggml_add(ctx0, cur, model.pre_encode_out_b);

    lm_ggml_set_name(cur, "pre_enc_out");

    // Encoder

    const auto & hparams  = model.hparams;
    const int n_layer     = hparams.n_layer;
    const int n_state     = hparams.n_embd;
    const float fc_factor = 0.5f;

    const int  n_time      = cur->ne[1];
    const bool local_attn  = n_time > PARAKEET_LOCAL_ATTN_THRESHOLD;
    const int  att_left    = local_attn ? PARAKEET_LOCAL_ATTN_WINDOW : n_time - 1;
    const int  att_right   = local_attn ? PARAKEET_LOCAL_ATTN_WINDOW : n_time - 1;
    const int  window_size = local_attn ? att_left + att_right + 1 : 2 * n_time - 1;
    const int  d_half      = n_state / 2;
    const int  mask_dim    = local_attn ? window_size : n_time;

    // mask [key, n_time]
    struct lm_ggml_tensor * attn_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, mask_dim, n_time);
    lm_ggml_set_name(attn_mask, "attn_mask");
    lm_ggml_set_input(attn_mask);

    struct lm_ggml_tensor * local_mask = nullptr;
    if (local_attn) {
        const int chunk = att_left + att_right;
        local_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, chunk + window_size - 1, chunk);
        lm_ggml_set_name(local_mask, "local_mask");
        lm_ggml_set_input(local_mask);
    }

    struct lm_ggml_tensor * pos_freqs = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_F32, d_half);
    lm_ggml_set_name(pos_freqs, "pos_freqs");
    lm_ggml_set_input(pos_freqs);

    struct lm_ggml_tensor * rel_positions = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, 1, window_size);
    lm_ggml_set_name(rel_positions, "rel_positions");
    lm_ggml_set_input(rel_positions);

    struct lm_ggml_tensor * freqs = lm_ggml_repeat_4d(ctx0, pos_freqs, d_half, window_size, 1, 1);
    struct lm_ggml_tensor * theta = lm_ggml_mul(ctx0, freqs, rel_positions);

    struct lm_ggml_tensor * sin = lm_ggml_reshape_3d(ctx0, lm_ggml_sin(ctx0, theta), 1, d_half, window_size);
    struct lm_ggml_tensor * cos = lm_ggml_reshape_3d(ctx0, lm_ggml_cos(ctx0, theta), 1, d_half, window_size);
    struct lm_ggml_tensor * pos_emb = lm_ggml_reshape_2d(ctx0, lm_ggml_cont(ctx0, lm_ggml_concat(ctx0, sin, cos, 0)), n_state, window_size);
    lm_ggml_set_name(pos_emb, "pos_emb");

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        // FFN1
        {
            struct lm_ggml_tensor * residual = cur;
            lm_ggml_format_name(cur, "enc_%d_res", il);

            // norm
            cur = lm_ggml_norm(ctx0, cur, hparams.eps);
            cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.ff_norm_w), layer.ff_norm_b);
            lm_ggml_format_name(cur, "enc_%d_ffn_norm_1", il);

            cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_SILU, il);
            lm_ggml_format_name(cur, "enc_%d_ffn_1", il);

            cur = lm_ggml_add(ctx0, residual, lm_ggml_scale(ctx0, cur, fc_factor));
            lm_ggml_format_name(cur, "enc_%d_res_ffn", il);
        }

        // self attention block using relative positional encoding from model.position_embedding.
        {
            // [feat, time_frames, 1, 1]
            struct lm_ggml_tensor * residual = cur;

            cur = lm_ggml_norm(ctx0, cur, hparams.eps);
            cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.ln_1_w), layer.ln_1_b);
            lm_ggml_format_name(cur, "enc_%d_attn_norm", il);

            const int n_head = hparams.n_head;
            const int d_head = n_state / n_head;

            // [feat, time_frames, 1, 1]
            struct lm_ggml_tensor * Q_cur = build_mm(layer.q_w, cur);
            struct lm_ggml_tensor * K_cur = build_mm(layer.k_w, cur);
            struct lm_ggml_tensor * V_cur = build_mm(layer.v_w, cur);

            // [d_head, n_heads, n_time, 1]
            Q_cur = lm_ggml_reshape_3d(ctx0, Q_cur, d_head, n_head, n_time);
            K_cur = lm_ggml_reshape_3d(ctx0, K_cur, d_head, n_head, n_time);
            V_cur = lm_ggml_reshape_3d(ctx0, V_cur, d_head, n_head, n_time);

            // [n_state, window_size]
            struct lm_ggml_tensor * pos = build_mm(layer.linear_pos_w, pos_emb);
            // [feat, head, window_size, 1]
            pos = lm_ggml_reshape_3d(ctx0, pos, d_head, n_head, pos_emb->ne[1]);
            // [feat, window_size, head, 1]
            pos = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, pos, 0, 2, 1, 3));
            lm_ggml_format_name(pos, "enc_%d_attn_pos", il);

            if (local_attn) {
                const int  chunk         = att_left + att_right;
                const int  n_group       = (n_time + chunk - 1) / chunk;
                const int  n_time_padded = n_group * chunk;
                const int  n_kv_chunk    = chunk + window_size - 1;
                const int  n_kv_dense    = n_kv_chunk * n_group;
                const bool need_padding  = n_time_padded > n_time;

                Q_cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Q_cur, 0, 2, 1, 3));
                K_cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, K_cur, 0, 2, 1, 3));
                V_cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, V_cur, 0, 2, 1, 3));

                // content bias
                struct lm_ggml_tensor * bias_u = lm_ggml_reshape_3d(ctx0, layer.pos_bias_u, d_head, 1, n_head);
                struct lm_ggml_tensor * Q_u = lm_ggml_add(ctx0, Q_cur, bias_u);

                // position bias
                struct lm_ggml_tensor * bias_v = lm_ggml_reshape_3d(ctx0, layer.pos_bias_v, d_head, 1, n_head);
                struct lm_ggml_tensor * Q_v = lm_ggml_add(ctx0, Q_cur, bias_v);

                // right pad the time dimension
                struct lm_ggml_tensor * Q_u_padded = need_padding ?
                    lm_ggml_pad_ext(ctx0, Q_u, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : Q_u;
                Q_u_padded = lm_ggml_reshape_4d(ctx0, Q_u_padded, d_head, chunk, n_group, n_head);

                // pad front and back for the first and last time frames
                struct lm_ggml_tensor * K_padded = lm_ggml_pad_ext(ctx0, K_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                if (n_kv_dense > K_padded->ne[1]) {
                    K_padded = lm_ggml_pad_ext(ctx0, K_padded, 0, 0, 0, n_kv_dense - K_padded->ne[1], 0, 0, 0, 0);
                }

                // sliding window view: each group spans n_kv_chunk keys but steps by chunk
                struct lm_ggml_tensor * K_chunk = lm_ggml_view_4d(ctx0, K_padded,
                        d_head, n_kv_chunk, n_group, n_head,
                        K_padded->nb[1],
                        (size_t) chunk * K_padded->nb[1],
                        K_padded->nb[2],
                        0);
                K_chunk = lm_ggml_cont(ctx0, K_chunk);

                struct lm_ggml_tensor * content_scores = lm_ggml_mul_mat(ctx0, K_chunk, Q_u_padded);

                // trim the dense output down to window_size scores per query
                content_scores = lm_ggml_view_4d(ctx0, content_scores,
                        window_size, chunk, n_group, n_head,
                        (size_t) (chunk + window_size) * content_scores->nb[0],
                        content_scores->nb[2],
                        content_scores->nb[3],
                        0);
                content_scores = lm_ggml_cont(ctx0, content_scores);

                // ungroup: [window_size, n_time_padded, n_head]
                content_scores = lm_ggml_reshape_3d(ctx0, content_scores, window_size, n_time_padded, n_head);
                if (need_padding) {
                    content_scores = lm_ggml_view_3d(ctx0, content_scores,
                            window_size, n_time, n_head,
                            content_scores->nb[1],
                            content_scores->nb[2],
                            0);
                }

                // Q_v: [d_head, time, head]
                Q_v = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Q_v, 0, 2, 1, 3));
                struct lm_ggml_tensor * rel_pos_scores = lm_ggml_mul_mat(ctx0, pos, Q_v);

                struct lm_ggml_tensor * attn_scores = lm_ggml_add(ctx0, content_scores, rel_pos_scores);
                attn_scores = lm_ggml_soft_max_ext(ctx0, attn_scores, attn_mask, 1.0f / std::sqrt(d_head), 0.0f);
                lm_ggml_format_name(attn_scores, "enc_%d_attn_probs", il);

                // expand probs back to n_kv_chunk width for the V matmul
                struct lm_ggml_tensor * probs_padded = need_padding ?
                    lm_ggml_pad_ext(ctx0, attn_scores, 0, 0, 0, n_time_padded - n_time, 0, 0, 0, 0) : attn_scores;

                probs_padded = lm_ggml_reshape_4d(ctx0, probs_padded, window_size, chunk, n_group, n_head);
                probs_padded = lm_ggml_pad_ext(ctx0, probs_padded, 0, chunk, 0, 0, 0, 0, 0, 0);
                probs_padded = lm_ggml_view_4d(ctx0, probs_padded,
                        n_kv_chunk, chunk, n_group, n_head,
                        (size_t) n_kv_chunk * probs_padded->nb[0],
                        probs_padded->nb[2],
                        probs_padded->nb[3],
                        0);
                probs_padded = lm_ggml_cont(ctx0, probs_padded);
                probs_padded = lm_ggml_mul(ctx0, probs_padded, local_mask);

                struct lm_ggml_tensor * V_padded = lm_ggml_pad_ext(ctx0, V_cur, 0, 0, att_left, att_right, 0, 0, 0, 0);
                if (n_kv_dense > V_padded->ne[1]) {
                    V_padded = lm_ggml_pad_ext(ctx0, V_padded, 0, 0, 0, n_kv_dense - V_padded->ne[1], 0, 0, 0, 0);
                }
                V_padded = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, V_padded));

                struct lm_ggml_tensor * V_chunk = lm_ggml_view_4d(ctx0, V_padded,
                        n_kv_chunk, d_head, n_group, n_head,
                        V_padded->nb[1],
                        (size_t) chunk * V_padded->nb[0],
                        V_padded->nb[2],
                        0);
                V_chunk = lm_ggml_cont(ctx0, V_chunk);

                cur = lm_ggml_mul_mat(ctx0, V_chunk, probs_padded);
                cur = lm_ggml_reshape_3d(ctx0, cur, d_head, n_time_padded, n_head);
                if (need_padding) {
                    cur = lm_ggml_view_3d(ctx0, cur, d_head, n_time, n_head, cur->nb[1], cur->nb[2], 0);
                }
                cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, cur, 0, 2, 1, 3));
                cur = lm_ggml_reshape_2d(ctx0, cur, n_state, n_time);
                cur = build_mm(layer.o_w, cur);
            } else {
                // full attention
                struct lm_ggml_tensor * Q_u = lm_ggml_add(ctx0, Q_cur, layer.pos_bias_u);
                lm_ggml_format_name(Q_u, "enc_%d_attn_q_u", il);

                struct lm_ggml_tensor * K_prep = lm_ggml_permute(ctx0, K_cur, 0, 2, 1, 3);
                struct lm_ggml_tensor * Q_prep = lm_ggml_permute(ctx0, Q_u,   0, 2, 1, 3);
                struct lm_ggml_tensor * content_scores = lm_ggml_mul_mat(ctx0, K_prep, Q_prep);
                lm_ggml_format_name(content_scores, "enc_%d_attn_content_scores", il);

                struct lm_ggml_tensor * Q_v = lm_ggml_add(ctx0, Q_cur, layer.pos_bias_v);
                lm_ggml_format_name(Q_v, "enc_%d_attn_q_v", il);

                Q_v = lm_ggml_permute(ctx0, Q_v, 0, 2, 1, 3);
                Q_v = lm_ggml_cont(ctx0, Q_v);
                lm_ggml_format_name(Q_v, "enc_%d_attn_q_v_perm", il);

                struct lm_ggml_tensor * rel_pos_scores = lm_ggml_mul_mat(ctx0, pos, Q_v);
                lm_ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos", il);

                // Relative positional shift
                {
                    const auto pos_window = rel_pos_scores->ne[0];
                    const auto n_frame    = rel_pos_scores->ne[1];
                    const auto n_head     = rel_pos_scores->ne[2];

                    rel_pos_scores = lm_ggml_pad(ctx0, rel_pos_scores, 1, 0, 0, 0);
                    rel_pos_scores = lm_ggml_roll(ctx0, rel_pos_scores, 1, 0, 0, 0);

                    rel_pos_scores = lm_ggml_reshape_3d(ctx0, rel_pos_scores, n_frame, pos_window + 1, n_head);
                    rel_pos_scores = lm_ggml_cont(ctx0, rel_pos_scores);
                    lm_ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_reshaped", il);

                    int center = pos_window / 2;
                    size_t offset = rel_pos_scores->nb[0] * (center+1);

                    rel_pos_scores = lm_ggml_view_3d(ctx0, rel_pos_scores,
                                                  n_frame, pos_window, n_head,
                                                  (pos_window) * 4,
                                                  rel_pos_scores->nb[2],
                                                  offset);
                    rel_pos_scores = lm_ggml_cont(ctx0, rel_pos_scores);
                    lm_ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted", il);

                    rel_pos_scores = lm_ggml_view_3d(ctx0, rel_pos_scores,
                                                  content_scores->ne[0],
                                                  content_scores->ne[1],
                                                  rel_pos_scores->ne[2],
                                                  rel_pos_scores->nb[1],
                                                  rel_pos_scores->nb[2],
                                                  0);
                    rel_pos_scores = lm_ggml_cont(ctx0, rel_pos_scores);
                    lm_ggml_format_name(rel_pos_scores, "enc_%d_attn_rel_pos_shifted_view", il);
                }

                struct lm_ggml_tensor * attn_scores = lm_ggml_add(ctx0, content_scores, rel_pos_scores);
                lm_ggml_format_name(attn_scores, "enc_%d_attn_scores", il);
                attn_scores = lm_ggml_scale(ctx0, attn_scores, 1.0f / std::sqrt(d_head));
                attn_scores = lm_ggml_add(ctx0, attn_scores, attn_mask);
                lm_ggml_format_name(attn_scores, "enc_%d_attn_scores_scaled", il);

                struct lm_ggml_tensor * probs = lm_ggml_soft_max(ctx0, attn_scores);
                lm_ggml_format_name(probs, "enc_%d_attn_probs", il);

                V_cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, V_cur, 1, 2, 0, 3));
                lm_ggml_format_name(V_cur, "enc_%d_attn_v_cur", il);
                cur = lm_ggml_mul_mat(ctx0, probs, V_cur);
                lm_ggml_format_name(cur, "enc_%d_attn_inp", il);

                cur = lm_ggml_permute(ctx0, cur, 2, 0, 1, 3);
                cur = lm_ggml_cont_2d(ctx0, cur, n_state, n_time);
                cur = build_mm(layer.o_w, cur);
            }
            lm_ggml_format_name(cur, "enc_%d_attn_out", il);

            cur = lm_ggml_add(ctx0, residual, cur);
            lm_ggml_format_name(cur, "enc_%d_attn_res", il);
        }

        // Convolution
        {
            struct lm_ggml_tensor * residual = cur;
            lm_ggml_format_name(cur, "enc_%d_residual_conv", il);

            cur = lm_ggml_norm(ctx0, cur, hparams.eps);
            cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.norm_conv_w), layer.norm_conv_b);
            lm_ggml_format_name(cur, "enc_%d_norm_conv", il);

            // pointwise 1d convolution:
            cur = build_mm(layer.conv_pw1_w, cur);
            lm_ggml_format_name(cur, "enc_%d_conv_pw1", il);

            {
                int64_t d = cur->ne[0] / 2;
                struct lm_ggml_tensor * signal = lm_ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], 0);
                struct lm_ggml_tensor * gate   = lm_ggml_view_2d(ctx0, cur, d, cur->ne[1], cur->nb[1], d * cur->nb[0]);

                cur = lm_ggml_mul(ctx0, signal, lm_ggml_sigmoid(ctx0, gate));
                lm_ggml_format_name(cur, "enc_%d_conv_glu", il);
            }

            cur = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, cur));

            // use lm_ggml_ssm_conv for f32 precision
            const int dw_pad = (hparams.audio_conv_kernel_size - 1) / 2;
            cur = lm_ggml_pad(ctx0, cur, dw_pad, 0, 0, 0);
            cur = lm_ggml_roll(ctx0, cur, dw_pad, 0, 0, 0);
            cur = lm_ggml_pad(ctx0, cur, dw_pad, 0, 0, 0);
            lm_ggml_format_name(cur, "enc_%d_conv_dw_pad", il);

            cur = lm_ggml_ssm_conv(ctx0, cur, layer.conv_dw_w);
            lm_ggml_format_name(cur, "enc_%d_conv_1d_dw", il);

            cur = lm_ggml_sub(ctx0, cur, layer.conv_norm_mean);
            struct lm_ggml_tensor * std = lm_ggml_sqrt(ctx0, layer.conv_norm_var);
            cur = lm_ggml_div(ctx0, cur, std);
            cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.conv_norm_w), layer.conv_norm_b);
            lm_ggml_format_name(cur, "enc_%d_conv_bn", il);

            cur = lm_ggml_silu(ctx0, cur);
            lm_ggml_format_name(cur, "enc_%d_conv_silu", il);

            cur = build_mm(layer.conv_pw2_w, cur);
            lm_ggml_format_name(cur, "enc_%d_conv_pw2", il);

            cur = lm_ggml_add(ctx0, residual, cur);
            lm_ggml_format_name(cur, "enc_%d_conv_res", il);
        }

        // FFN2
        {
            struct lm_ggml_tensor * residual = cur;
            cur = lm_ggml_norm(ctx0, cur, hparams.eps);
            cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.ff_norm_1_w), layer.ff_norm_1_b);
            lm_ggml_format_name(cur, "enc_%d_ffn_norm_2", il);

            cur = build_ffn(cur, layer.ff_up_1_w, nullptr, nullptr, nullptr, layer.ff_down_1_w, nullptr, FFN_SILU, il);
            cur = lm_ggml_add(ctx0, residual, lm_ggml_scale(ctx0, cur, 0.5));
            lm_ggml_format_name(cur, "enc_%d_ffn_res", il);
        }

        cur = lm_ggml_norm(ctx0, cur, hparams.eps);
        cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, cur, layer.ln_2_w), layer.ln_2_b);
    }

    cb(cur, "encoder_out", -1);

    cur = lm_ggml_rms_norm(ctx0, cur, 1e-6);
    cur = lm_ggml_mul(ctx0, cur, model.mm_norm_pre_w);
    cb(cur, "sound_projection.norm", -1);

    cur = build_ffn(cur, model.mm_0_w, model.mm_0_b, nullptr, nullptr, model.mm_1_w, model.mm_1_b, FFN_RELU_SQR, -1);
    cb(cur, "projected", -1);

    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
