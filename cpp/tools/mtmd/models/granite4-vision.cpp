#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

/*
 * Granite Vision 4.1 clip graph
 *
 *   Stage 1a: SigLIP vision tower (N layers, post-norm)
 *   Stage 1b: WindowQFormer blocks (deepstack + spatial)
 *   Stage 1c: Concatenate and pack outputs
 *   Stage 1d: Append newline tokens if add_newline is set
 */

// ---------------------------------------------------------------------------
// Member method implementations
// ---------------------------------------------------------------------------

lm_ggml_tensor * clip_graph_granite4_vision::gather(
        lm_ggml_tensor * src,
        const std::string & name,
        int idx_len) {
    lm_ggml_tensor * idx = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_I32, idx_len);
    lm_ggml_set_name(idx, name.c_str());
    lm_ggml_set_input(idx);
    return lm_ggml_get_rows(ctx0, src, idx);
}

lm_ggml_tensor * clip_graph_granite4_vision::interp_down(
        lm_ggml_tensor * src,
        int side,
        int new_side) {
    const int n_embd = src->ne[0];
    lm_ggml_tensor * t = lm_ggml_reshape_4d(ctx0, src, n_embd, side, side, 1);
    t = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, t, 2, 0, 1, 3));
    const int kernel = side / new_side;
    t = lm_ggml_pool_2d(ctx0, t, LM_GGML_OP_POOL_AVG, kernel, kernel, kernel, kernel, 0, 0);
    t = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, t, 1, 2, 0, 3));
    return lm_ggml_reshape_2d(ctx0, t, n_embd, new_side * new_side);
}

// ---------------------------------------------------------------------------
// build_block - WindowQFormer block implementation
// ---------------------------------------------------------------------------

lm_ggml_tensor * clip_graph_granite4_vision::build_block(
        const qf_block & blk,
        lm_ggml_tensor * h,
        int bid,
        int spatial_offset,
        int image_side,
        int window_side,
        int query_side,
        float qformer_eps) {

    const int n_embd = h->ne[0];
    LM_GGML_ASSERT(h->ne[1] == image_side * image_side);
    const int n = image_side / window_side;
    const int new_side = n * query_side;
    const int n_windows = n * n;
    const int enc_len = window_side * window_side;
    const int query_len = query_side * query_side;

    auto cbx = [&](lm_ggml_tensor * & t, const char * step) {
        const std::string name = "g4v_blk" + std::to_string(bid) + "_" + step;
        lm_ggml_set_name(t, name.c_str());
    };

    // 1. Top-level LN
    cbx(h, "inp");
    lm_ggml_tensor * x = build_norm(h, blk.qf_proj_norm_w, blk.qf_proj_norm_b, NORM_TYPE_NORMAL, eps, bid);
    cbx(x, "norm");

    // 2. enc = _win(x, image_side, window_side)
    lm_ggml_tensor * enc;
    {
        lm_ggml_tensor * enc_flat = gather(x,
            "g4v_blk" + std::to_string(bid) + "_win_idx",
            image_side * image_side);
        enc = lm_ggml_reshape_3d(ctx0, enc_flat, n_embd, enc_len, n_windows);
    }
    cbx(enc, "enc");

    // 3. downsampled = downsampler(x)
    lm_ggml_tensor * d;
    (void) spatial_offset;
    if (spatial_offset >= 0) {
        d = gather(x,
            "g4v_blk" + std::to_string(bid) + "_spatial_idx",
            new_side * new_side);
    } else {
        d = interp_down(x, image_side, new_side);
    }
    cbx(d, "downsampled");

    // 4. query_embeds = query + _win(d, new_side, query_side)
    lm_ggml_tensor * q_in;
    {
        lm_ggml_tensor * dw_flat = gather(d,
            "g4v_blk" + std::to_string(bid) + "_qwin_idx",
            new_side * new_side);
        lm_ggml_tensor * dw = lm_ggml_reshape_3d(ctx0, dw_flat, n_embd, query_len, n_windows);
        q_in = lm_ggml_add(ctx0, dw, blk.qf_proj_query);
    }
    cbx(q_in, "query_embeds");

    // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
    lm_ggml_tensor * e_in = lm_ggml_add(ctx0, enc, blk.qf_proj_img_pos);
    cbx(e_in, "encoder_embeds");

    // 6. Qformer forward.
    lm_ggml_tensor * q = build_norm(q_in, blk.qf_proj_post_norm_w, blk.qf_proj_post_norm_b, NORM_TYPE_NORMAL, qformer_eps, bid);

    // Helper for linear projections with window batching
    auto linear = [&](lm_ggml_tensor * x, lm_ggml_tensor * w, lm_ggml_tensor * b) -> lm_ggml_tensor * {
        lm_ggml_tensor * t = lm_ggml_reshape_2d(ctx0, x, x->ne[0], x->ne[1] * x->ne[2]);
        t = build_mm(w, t);
        if (b) t = lm_ggml_add(ctx0, t, b);
        return t;
    };

    // Get the single QFormer layer
    LM_GGML_ASSERT(blk.qf_proj_layers.size() == 1);
    const auto & pl = blk.qf_proj_layers[0];

    // 6a. Self-attention
    lm_ggml_tensor * sa_out;
    {
        const int d_h = 64;
        const int n_head = n_embd / d_h;
        const int nq = q->ne[1];
        const float scale = 1.0f / std::sqrt((float) d_h);

        lm_ggml_tensor * Q = linear(q, pl.q_w, pl.q_b);
        lm_ggml_tensor * K = linear(q, pl.k_w, pl.k_b);
        lm_ggml_tensor * V = linear(q, pl.v_w, pl.v_b);

        Q = lm_ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_windows);
        K = lm_ggml_reshape_4d(ctx0, K, d_h, n_head, nq, n_windows);
        V = lm_ggml_reshape_4d(ctx0, V, d_h, n_head, nq, n_windows);

        sa_out = build_attn(pl.o_w, pl.o_b, Q, K, V, nullptr, scale, bid);
        sa_out = lm_ggml_reshape_3d(ctx0, sa_out, n_embd, nq, n_windows);

        sa_out = lm_ggml_add(ctx0, sa_out, q);
        sa_out = build_norm(sa_out, pl.ln_1_w, pl.ln_1_b,
                            NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(sa_out, "sa_out");

    // 6b. Cross-attention
    lm_ggml_tensor * ca_out;
    {
        const int d_h = 64;
        const int n_head = n_embd / d_h;
        const int nq = sa_out->ne[1];
        const int nkv = e_in->ne[1];
        const float scale = 1.0f / std::sqrt((float) d_h);

        lm_ggml_tensor * Q = linear(sa_out, pl.cross_attn_q_w, pl.cross_attn_q_b);
        lm_ggml_tensor * K = linear(e_in, pl.cross_attn_k_w, pl.cross_attn_k_b);
        lm_ggml_tensor * V = linear(e_in, pl.cross_attn_v_w, pl.cross_attn_v_b);

        Q = lm_ggml_reshape_4d(ctx0, Q, d_h, n_head, nq, n_windows);
        K = lm_ggml_reshape_4d(ctx0, K, d_h, n_head, nkv, n_windows);
        V = lm_ggml_reshape_4d(ctx0, V, d_h, n_head, nkv, n_windows);

        ca_out = build_attn(pl.cross_attn_o_w, pl.cross_attn_o_b,
                            Q, K, V, nullptr, scale, bid);
        ca_out = lm_ggml_reshape_3d(ctx0, ca_out, n_embd, nq, n_windows);

        ca_out = lm_ggml_add(ctx0, ca_out, sa_out);
        ca_out = build_norm(ca_out, pl.cross_attn_norm_w, pl.cross_attn_norm_b,
                            NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ca_out, "ca_out");

    // 6c. FFN
    lm_ggml_tensor * ffn;
    {
        lm_ggml_tensor * t = lm_ggml_reshape_2d(ctx0, ca_out, n_embd, query_len * n_windows);
        t = build_mm(pl.ff_up_w, t);
        if (pl.ff_up_b) t = lm_ggml_add(ctx0, t, pl.ff_up_b);
        t = lm_ggml_gelu_erf(ctx0, t);
        t = build_mm(pl.ff_down_w, t);
        if (pl.ff_down_b) t = lm_ggml_add(ctx0, t, pl.ff_down_b);
        t = lm_ggml_reshape_3d(ctx0, t, n_embd, query_len, n_windows);
        ffn = lm_ggml_add(ctx0, t, ca_out);
        ffn = build_norm(ffn, pl.ln_2_w, pl.ln_2_b, NORM_TYPE_NORMAL, qformer_eps, bid);
    }
    cbx(ffn, "qformer_out");

    // 7. _unwin back to raster
    lm_ggml_tensor * unwinned;
    {
        lm_ggml_tensor * flat = lm_ggml_reshape_2d(ctx0, ffn, n_embd, query_len * n_windows);
        unwinned = gather(flat,
            "g4v_blk" + std::to_string(bid) + "_unwin_idx",
            new_side * new_side);
    }
    cbx(unwinned, "unwin");

    // 8. out_linear
    lm_ggml_tensor * out = build_mm(blk.qf_proj_linear_w, unwinned);
    if (blk.qf_proj_linear_b) out = lm_ggml_add(ctx0, out, blk.qf_proj_linear_b);
    cbx(out, "out");

    return out;
}

// ---------------------------------------------------------------------------
// build() - top-level graph
// ---------------------------------------------------------------------------

// Build the K-tiled, base-scaled newline row tensor.
// Shape: (n_mmproj_embd, 1)
lm_ggml_tensor * clip_graph_granite4_vision::build_newline_row(lm_ggml_context * ctx0) {
    const int K = (int) model.qf_proj_blocks.size();
    LM_GGML_ASSERT(K > 0);
    LM_GGML_ASSERT(n_mmproj_embd % K == 0);
    const int projection_dim = n_mmproj_embd / K;
    LM_GGML_ASSERT(model.image_newline != nullptr);
    LM_GGML_ASSERT(lm_ggml_nelements(model.image_newline) == projection_dim);

    // Build newline_row[k*projection_dim + d] = nl[d] * (k == 0 ? base : 1.0)
    lm_ggml_tensor * nl = model.image_newline; // (projection_dim,)
    lm_ggml_tensor * nl_first_2d = lm_ggml_reshape_2d(ctx0, nl, projection_dim, 1);
    lm_ggml_tensor * nl_row_2d;
    if (K == 1) {
        nl_row_2d = nl_first_2d;
    } else {
        lm_ggml_tensor * nl_2d = lm_ggml_reshape_2d(ctx0, nl, projection_dim, 1);
        lm_ggml_tensor * rest_template = lm_ggml_new_tensor_2d(
            ctx0, LM_GGML_TYPE_F32, projection_dim, K - 1);
        lm_ggml_tensor * nl_rest = lm_ggml_repeat(ctx0, nl_2d, rest_template);
        nl_row_2d = lm_ggml_concat(ctx0, nl_first_2d, nl_rest, 1); // (projection_dim, K)
    }
    nl_row_2d = lm_ggml_cont(ctx0, nl_row_2d);
    return lm_ggml_reshape_2d(ctx0, nl_row_2d, n_mmproj_embd, 1);
}

// Append a single newline row at the end of the tile output.
lm_ggml_tensor * clip_graph_granite4_vision::append_rowwise_newlines(lm_ggml_context * ctx0, lm_ggml_tensor * tile_output) {
    // For the single-tile case, append one newline row at the end.
    // For the multi-tile rowwise case, this will be called per-tile
    // (though currently only the single-tile path uses it).
    lm_ggml_tensor * nl_row = build_newline_row(ctx0);
    return lm_ggml_concat(ctx0, tile_output, nl_row, 1);
}

lm_ggml_cgraph * clip_graph_granite4_vision::build() {
    LM_GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    LM_GGML_ASSERT(model.position_embeddings != nullptr);
    LM_GGML_ASSERT(model.class_embedding == nullptr);
    LM_GGML_ASSERT(!model.qf_proj_blocks.empty());

    // --- Stage 1a: SigLIP encoder producing intermediate hidden states ---
    lm_ggml_tensor * inp = build_inp();
    inp = lm_ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "pos_embed", -1);

    lm_ggml_tensor * inpL = inp;
    std::vector<lm_ggml_tensor *> layer_outs(n_layer, nullptr);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        lm_ggml_tensor * cur = inpL;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        // Self-attention
        lm_ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        if (layer.q_b) Qcur = lm_ggml_add(ctx0, Qcur, layer.q_b);
        lm_ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        if (layer.k_b) Kcur = lm_ggml_add(ctx0, Kcur, layer.k_b);
        lm_ggml_tensor * Vcur = build_mm(layer.v_w, cur);
        if (layer.v_b) Vcur = lm_ggml_add(ctx0, Vcur, layer.v_b);

        Qcur = lm_ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
        Kcur = lm_ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
        Vcur = lm_ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

        cur = build_attn(layer.o_w, layer.o_b,
                         Qcur, Kcur, Vcur, nullptr, kq_scale, il);

        cur = lm_ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur,
                        layer.ff_up_w, layer.ff_up_b,
                        layer.ff_gate_w, layer.ff_gate_b,
                        layer.ff_down_w, layer.ff_down_b,
                        hparams.ffn_op, il);
        cur = lm_ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);
        layer_outs[il] = cur;
        inpL = cur;
    }

    // --- Stage 1b/1c: WindowQFormer blocks ---
    const int projector_count = hparams.vision_feature_layer.size();
    const float qformer_eps = 1e-12f;

    lm_ggml_tensor * mmproj = nullptr;
    for (int bid = 0; bid < projector_count; ++bid) {
        const auto & blk = model.qf_proj_blocks[bid];

        int vlayer = hparams.vision_feature_layer[bid];
        LM_GGML_ASSERT(vlayer >= 0 && vlayer < n_layer);
        lm_ggml_tensor * h = layer_outs[vlayer];

        lm_ggml_tensor * stream = build_block(
            blk, h, bid,
            hparams.proj_spatial_offsets[bid],
            n_patches_x,
            hparams.downsample_window_side,
            hparams.downsample_query_side,
            qformer_eps);
        cb(stream, (std::string("proj_") + std::to_string(bid) + std::string("_v_out")).c_str(), vlayer);
        mmproj = mmproj ? lm_ggml_concat(ctx0, mmproj, stream, 0) : stream;
    }

    // --- Stage 1d: Append newline tokens if add_newline is set ---
    if (add_newline) {
        mmproj = append_rowwise_newlines(ctx0, mmproj);
        lm_ggml_set_name(mmproj, "g4v_mmproj_out_nl");
    } else {
        lm_ggml_set_name(mmproj, "g4v_mmproj_out");
    }
    lm_ggml_build_forward_expand(gf, mmproj);

    return gf;
}
