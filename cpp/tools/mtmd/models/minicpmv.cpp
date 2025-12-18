#include "models.h"

lm_ggml_cgraph * clip_graph_minicpmv::build() {
    LM_GGML_ASSERT(model.class_embedding == nullptr);
    const int n_pos       = n_patches;
    const int n_embd_proj = n_mmproj_embd;

    // position embeddings for the projector (not for ViT)
    // see: https://huggingface.co/openbmb/MiniCPM-o-2_6/blob/main/resampler.py#L70
    // base frequency omega
    lm_ggml_tensor * omega = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_F32, n_embd_proj / 4);
    lm_ggml_set_name(omega, "omega");
    lm_ggml_set_input(omega);

    // 2D input positions (using float for sinusoidal embeddings)
    lm_ggml_tensor * pos_h = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, 1, n_pos);
    lm_ggml_set_name(pos_h, "pos_h");
    lm_ggml_set_input(pos_h);
    lm_ggml_tensor * pos_w = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, 1, n_pos);
    lm_ggml_set_name(pos_w, "pos_w");
    lm_ggml_set_input(pos_w);

    // for selecting learned pos embd, used by ViT
    struct lm_ggml_tensor * positions = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, n_pos);
    lm_ggml_set_name(positions, "positions");
    lm_ggml_set_input(positions);

    lm_ggml_tensor * learned_pos_embd = lm_ggml_get_rows(ctx0, model.position_embeddings, positions);

    lm_ggml_tensor * inp = build_inp();
    lm_ggml_tensor * embeddings = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    // resampler projector (it is just another transformer)

    lm_ggml_tensor * q = model.mm_model_query;
    lm_ggml_tensor * v = lm_ggml_mul_mat(ctx0, model.mm_model_kv_proj, embeddings);

    // norm
    q = build_norm(q, model.mm_model_ln_q_w,  model.mm_model_ln_q_b,  NORM_TYPE_NORMAL, eps, -1);
    v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

    // calculate sinusoidal pos embd
    lm_ggml_tensor * pos_embed = nullptr;
    {
        // outer product
        lm_ggml_tensor * omega_b = lm_ggml_repeat_4d(ctx0, omega, omega->ne[0], n_pos, 1, 1); // n_pos rows
        lm_ggml_tensor * theta_x = lm_ggml_mul(ctx0, omega_b, pos_w);
        lm_ggml_tensor * theta_y = lm_ggml_mul(ctx0, omega_b, pos_h);
        // sin and cos
        lm_ggml_tensor * pos_embd_x = lm_ggml_concat(
            ctx0,
            lm_ggml_sin(ctx0, theta_x),
            lm_ggml_cos(ctx0, theta_x),
            0 // concat on first dim
        );
        lm_ggml_tensor * pos_embd_y = lm_ggml_concat(
            ctx0,
            lm_ggml_sin(ctx0, theta_y),
            lm_ggml_cos(ctx0, theta_y),
            0 // concat on first dim
        );
        pos_embed = lm_ggml_concat(ctx0, pos_embd_x, pos_embd_y, 0);
    }

    // k = v + pos_embed
    lm_ggml_tensor * k = lm_ggml_add(ctx0, v, pos_embed);

    // attention
    {
        const int d_head = 128;
        int n_head = n_embd_proj/d_head;
        // Use actual config value if available, otherwise fall back to hardcoded values
        int num_query = hparams.minicpmv_query_num;
        lm_ggml_tensor * Q = lm_ggml_add(ctx0,
            lm_ggml_mul_mat(ctx0, model.mm_model_attn_q_w, q),
            model.mm_model_attn_q_b);
        lm_ggml_tensor * K = lm_ggml_add(ctx0,
            lm_ggml_mul_mat(ctx0, model.mm_model_attn_k_w, k),
            model.mm_model_attn_k_b);
        lm_ggml_tensor * V = lm_ggml_add(ctx0,
            lm_ggml_mul_mat(ctx0, model.mm_model_attn_v_w, v),
            model.mm_model_attn_v_b);

        Q = lm_ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
        K = lm_ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
        V = lm_ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

        cb(Q, "resampler_Q", -1);
        cb(K, "resampler_K", -1);
        cb(V, "resampler_V", -1);

        float resampler_kq_scale = 1.0f/ sqrtf(float(d_head));
        embeddings = build_attn(
            model.mm_model_attn_o_w,
            model.mm_model_attn_o_b,
            Q, K, V, nullptr, resampler_kq_scale, -1);
        cb(embeddings, "resampler_attn_out", -1);
    }
    // layernorm
    embeddings = build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);

    // projection
    embeddings = lm_ggml_mul_mat(ctx0, model.mm_model_proj, embeddings);

    // build the graph
    lm_ggml_build_forward_expand(gf, embeddings);

    return gf;
}
