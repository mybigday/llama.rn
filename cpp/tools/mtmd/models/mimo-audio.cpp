#include "models.h"

lm_ggml_cgraph * clip_graph_mimo_audio::build() {
    lm_ggml_tensor * inp = build_inp_raw(1); // [n_frames, n_mel, 1]

    lm_ggml_tensor * cur = lm_ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
    cur = lm_ggml_add(ctx0, cur, model.conv1d_1_b);
    cur = lm_ggml_gelu_erf(ctx0, cur);

    cur = lm_ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
    cur = lm_ggml_add(ctx0, cur, model.conv1d_2_b);
    cur = lm_ggml_gelu_erf(ctx0, cur);

    lm_ggml_tensor * inpL = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, cur)); // [n_embd, n_pos]
    const int64_t n_pos = inpL->ne[1];
    cb(inpL, "after_conv1d", -1);

    LM_GGML_ASSERT((int) hparams.wa_pattern_mode.size() == n_layer);

    lm_ggml_tensor * inp_pos = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos);
    lm_ggml_set_name(inp_pos, "mimo_audio_positions");
    lm_ggml_set_input(inp_pos);

    lm_ggml_tensor * full_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_pos, n_pos);
    lm_ggml_set_name(full_mask, "mimo_audio_full_mask");
    lm_ggml_set_input(full_mask);

    lm_ggml_tensor * window_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_pos, n_pos);
    lm_ggml_set_name(window_mask, "mimo_audio_window_mask");
    lm_ggml_set_input(window_mask);

    build_vit_opts opts;
    opts.attn_mask_layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        opts.attn_mask_layers[il] = hparams.wa_pattern_mode[il] == -1 ? full_mask : window_mask;
    }
    // the skip connection below must be added before the post-transformer  norm,
    // so build_vit must not apply that norm itself
    opts.skip_post_ln = true;

    // encoder_skip_layer_id=3 (1-indexed) -> capture output of layer index 2
    const int skip_capture_il = 2;
    LM_GGML_ASSERT(n_layer > skip_capture_il);
    lm_ggml_tensor * skip_hidden = nullptr;
    opts.callback_layer_out = [&](lm_ggml_tensor * layer_cur, int il) {
        if (il == skip_capture_il) {
            skip_hidden = layer_cur;
        }
    };

    auto add_pos = [&](lm_ggml_tensor * x, const clip_layer &) {
        return lm_ggml_rope_ext(ctx0, x, inp_pos, nullptr, d_head,
                             LM_GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };

    inpL = build_vit(inpL, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, add_pos, opts);
    inpL = lm_ggml_reshape_2d(ctx0, inpL, n_embd, n_pos); // build_vit restores a (size-1) batch dim

    LM_GGML_ASSERT(skip_hidden != nullptr);
    inpL = lm_ggml_add(ctx0, inpL, skip_hidden);

    inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    cb(inpL, "after_transformer", -1);

    // downsample: strided conv (no bias) + gelu + layernorm
    {
        lm_ggml_tensor * ds = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, inpL)); // [n_pos, n_embd]
        ds = lm_ggml_conv_1d(ctx0, model.downsample_conv_w, ds, 2, 0, 1);
        ds = lm_ggml_gelu_erf(ctx0, ds);
        ds = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, ds)); // [n_embd, n_pos/2]
        ds = build_norm(ds, model.downsample_norm_w, model.downsample_norm_b, NORM_TYPE_NORMAL, eps, -1);
        inpL = ds;
    }
    cb(inpL, "after_downsample", -1);

    // RVQ quantize: codebook ne=[dim, max_bins, n_q]
    // quantize input vector to codes (type=I32)
    std::vector<lm_ggml_tensor *> codes;
    {
        LM_GGML_ASSERT(model.rvq_codebook != nullptr);
        const int64_t dim = model.rvq_codebook->ne[0];
        LM_GGML_ASSERT(dim == inpL->ne[0]);
        LM_GGML_ASSERT((int64_t) hparams.rvq_codebook_size.size() == model.rvq_codebook->ne[2]);

        lm_ggml_tensor * residual = inpL; // [dim, n_pos_ds]

        for (size_t q = 0; q < hparams.rvq_codebook_size.size(); q++) {
            const int64_t bins = hparams.rvq_codebook_size[q];
            lm_ggml_tensor * codebook_q = lm_ggml_view_2d(ctx0, model.rvq_codebook, dim, bins,
                model.rvq_codebook->nb[1], q * model.rvq_codebook->nb[2]);
            codebook_q = lm_ggml_cont(ctx0, codebook_q);

            lm_ggml_tensor * codebook_norm = lm_ggml_sum_rows(ctx0, lm_ggml_sqr(ctx0, codebook_q)); // [1, bins]
            codebook_norm = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, codebook_norm));          // [bins, 1]

            lm_ggml_tensor * dot    = lm_ggml_mul_mat(ctx0, codebook_q, residual); // [bins, n_pos_ds]
            lm_ggml_tensor * scores = lm_ggml_sub(ctx0, lm_ggml_scale(ctx0, dot, 2.0f), codebook_norm);

            lm_ggml_tensor * idx = lm_ggml_argmax(ctx0, scores); // [n_pos_ds]
            codes.push_back(idx);

            lm_ggml_tensor * quant = lm_ggml_get_rows(ctx0, codebook_q, idx); // [dim, n_pos_ds]
            residual = lm_ggml_sub(ctx0, residual, quant);
            cb(idx, "rvq_code", (int) q);
        }
    }

    // convert codes to LLM embeddings
    lm_ggml_tensor * code_embd_sum = nullptr;
    {
        LM_GGML_ASSERT(model.mm_a_code_embd != nullptr);
        const int64_t dim   = model.mm_a_code_embd->ne[0];
        const int64_t vocab = model.mm_a_code_embd->ne[1];
        LM_GGML_ASSERT((int64_t) codes.size() == model.mm_a_code_embd->ne[2]);
        LM_GGML_ASSERT(dim == inpL->ne[0]);

        for (size_t i = 0; i < codes.size(); i++) {
            lm_ggml_tensor * table_i = lm_ggml_view_2d(ctx0, model.mm_a_code_embd, dim, vocab,
                model.mm_a_code_embd->nb[1], i * model.mm_a_code_embd->nb[2]);
            table_i = lm_ggml_cont(ctx0, table_i);

            lm_ggml_tensor * embd_i = lm_ggml_get_rows(ctx0, table_i, codes[i]); // [dim, n_pos_ds]
            code_embd_sum = code_embd_sum ? lm_ggml_add(ctx0, code_embd_sum, embd_i) : embd_i;
        }
        cb(code_embd_sum, "code_embd_sum", -1);
    }

    // input_local_transformer
    // groups of `group_size` consecutive downsampled frames are processed together, attending only within their own group.
    // Implemented as a block-diagonal mask + in-group-repeating positions
    // (rather than a real batch dim) - same technique as the encoder's masks above, and as gemma4a's / deepseekocr2's chunked attention.

    // note: hand-rolled here instead of build_vit() because this is a second, independent layer stack
    // (own layer array/count, RMSNorm instead of LN, SiLU FFN, own RoPE theta)

    lm_ggml_tensor * projected;
    {
        const int group_size = hparams.audio_local_group_size;
        LM_GGML_ASSERT(group_size > 0);
        const int64_t n_pos_ds = code_embd_sum->ne[1];
        const int64_t n_groups = (n_pos_ds + group_size - 1) / group_size;
        const int64_t n_padded = n_groups * group_size;

        lm_ggml_tensor * cur_local = code_embd_sum;
        if (n_padded != n_pos_ds) {
            cur_local = lm_ggml_pad(ctx0, cur_local, 0, (int) (n_padded - n_pos_ds), 0, 0);
        }

        lm_ggml_tensor * local_pos = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_padded);
        lm_ggml_set_name(local_pos, "mimo_audio_local_positions");
        lm_ggml_set_input(local_pos);

        lm_ggml_tensor * local_mask = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_padded, n_padded);
        lm_ggml_set_name(local_mask, "mimo_audio_local_mask");
        lm_ggml_set_input(local_mask);

        const float local_rope_theta = 640000.0f; // audio_config.rope_theta (differs from the encoder's)
        auto apply_local_rope = [&](lm_ggml_tensor * x) {
            return lm_ggml_rope_ext(ctx0, x, local_pos, nullptr, d_head,
                                 LM_GGML_ROPE_TYPE_NEOX, 0, local_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        };

        for (int il = 0; il < hparams.audio_local_n_layer; il++) {
            auto & layer = model.mm_a_local_layers[il];

            lm_ggml_tensor * attn_in = build_norm(cur_local, layer.ln_1_w, nullptr, NORM_TYPE_RMS, eps, il);

            lm_ggml_tensor * Qcur = build_mm(layer.q_w, attn_in);
            if (layer.q_b) {
                Qcur = lm_ggml_add(ctx0, Qcur, layer.q_b);
            }
            lm_ggml_tensor * Kcur = build_mm(layer.k_w, attn_in);
            if (layer.k_b) {
                Kcur = lm_ggml_add(ctx0, Kcur, layer.k_b);
            }
            lm_ggml_tensor * Vcur = build_mm(layer.v_w, attn_in);
            if (layer.v_b) {
                Vcur = lm_ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = lm_ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_padded);
            Kcur = lm_ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_padded);
            Vcur = lm_ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_padded);

            Qcur = apply_local_rope(Qcur);
            Kcur = apply_local_rope(Kcur);

            lm_ggml_tensor * attn_out = build_attn(layer.o_w, nullptr, Qcur, Kcur, Vcur, local_mask, kq_scale, il);
            cur_local = lm_ggml_add(ctx0, cur_local, attn_out);

            lm_ggml_tensor * ffn_in = build_norm(cur_local, layer.ln_2_w, nullptr, NORM_TYPE_RMS, eps, il);
            lm_ggml_tensor * ffn_out = build_ffn(ffn_in,
                layer.ff_up_w, nullptr,
                layer.ff_gate_w, nullptr,
                layer.ff_down_w, nullptr,
                FFN_SILU, il);
            cur_local = lm_ggml_add(ctx0, cur_local, ffn_out);
        }

        cur_local = build_norm(cur_local, model.mm_a_local_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);
        cb(cur_local, "after_local_transformer", -1);

        // flatten each group of `group_size` frames into one (group_size*n_embd)-dim vector
        // (matching AudioProjection's flattened input)
        lm_ggml_tensor * grouped = lm_ggml_reshape_2d(ctx0, cur_local, n_embd * group_size, n_groups);

        // AudioProjection: Linear (no bias) -> GELU -> Linear (no bias)
        projected = build_ffn(grouped,
            model.mm_1_w, nullptr,
            nullptr, nullptr,
            model.mm_2_w, nullptr,
            FFN_GELU_ERF, -1);
        cb(projected, "after_projection", -1);
    }

    lm_ggml_build_forward_expand(gf, projected);
    return gf;
}
