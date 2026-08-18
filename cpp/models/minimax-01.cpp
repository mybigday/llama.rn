#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_minimax_01::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_RESIDUAL_SCALE,              hparams.f_residual_scale);

    // we use n_embd_head_la to set recurrent memory n_embd_s
    hparams.n_embd_head_la = hparams.n_embd_head_k_full;

    // Mark recurrent layers (lightning attention layers).
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 8;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 80: type = LLM_TYPE_456B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_minimax_01::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_recr(i)) {
            create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        } else {
            layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd_head_k * n_head}, 0);
            layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd_head_k * n_head}, 0);
            layer.wg = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_01::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

class llm_graph_input_la : public llm_graph_input_i {
public:
    llm_graph_input_la(const llama_hparams & hparams) : hparams(hparams) {}

    void set_input(const llama_ubatch * ubatch) override {
        // this operates on assumption that we have an equal ubatch split

        const int64_t n_head = hparams.n_head();
        const int32_t n_seqs = ubatch->n_seqs;
        const int32_t n_seqs_unq = ubatch->n_seqs_unq;
        const int32_t n_tokens = ubatch->n_tokens;
        const int32_t n_seq_tokens = ubatch->n_seq_tokens;

        std::vector<llama_pos> p0(n_seqs_unq);
        std::fill(p0.begin(), p0.end(), std::numeric_limits<llama_pos>::max());

        // get lowest token position in a ubatch for each stream
        for (int i = 0; i < n_tokens; ++i) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            int32_t seq_idx = ubatch->seq_idx[seq_id];
            llama_pos pos = ubatch->pos[i];
            if (p0[seq_idx] > pos) {
                p0[seq_idx] = pos;
            }
        }

        if (inp_slopes) {
            LM_GGML_ASSERT(lm_ggml_backend_buffer_is_host(inp_slopes->buffer));

            float * data = (float *) inp_slopes->data;

            float start = powf(2, -powf(2, -(log2f(n_head) - 3)));
            float ratio = start;

            for (int h = 0; h < n_head; ++h) {
                data[h] = start * powf(ratio, h);
            }
        }

        if (inp_q_decay) {
            LM_GGML_ASSERT(lm_ggml_backend_buffer_is_host(inp_q_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_q_decay->data;

            for (int s = 0; s < n_seqs; ++s) {
                for (int i = 0; i < n_seq_tokens; ++i) {
                    llama_seq_id seq_id = ubatch->seq_id[s * n_seq_tokens + i][0];
                    int32_t seq_idx = ubatch->seq_idx[seq_id];
                    llama_pos pos = ubatch->pos[s * n_seq_tokens + i];
                    int pos_rel = pos - p0[seq_idx];

                    for (int h = 0; h < n_head; ++h) {
                        data[seq_idx * n_head * n_seq_tokens + i * n_head + h] = -slopes[h] * (pos_rel + 1);
                    }
                }
            }
        }

        if (inp_k_decay) {
            LM_GGML_ASSERT(lm_ggml_backend_buffer_is_host(inp_k_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_k_decay->data;

            for (int s = 0; s < n_seqs; ++s) {
                for (int i = 0; i < n_seq_tokens; ++i) {
                    llama_seq_id seq_id = ubatch->seq_id[s * n_seq_tokens + i][0];
                    int32_t seq_idx = ubatch->seq_idx[seq_id];
                    llama_pos pos = ubatch->pos[s * n_seq_tokens + i];
                    int pos_rel = pos - p0[seq_idx];

                    for (int h = 0; h < n_head; ++h) {
                        data[seq_idx * n_head * n_seq_tokens + i * n_head + h] = -slopes[h] * (n_seq_tokens - pos_rel - 1);
                    }
                }
            }
        }

        if (inp_diag_decay) {
            LM_GGML_ASSERT(lm_ggml_backend_buffer_is_host(inp_diag_decay->buffer));

            float * slopes = (float *) inp_slopes->data;
            float * data = (float *) inp_diag_decay->data;

            for (int s = 0; s < n_seqs; ++s) {
                for (int h = 0; h < n_head; ++h) {
                    for (int j = 0; j < n_seq_tokens; ++j) {
                        llama_seq_id seq_id = ubatch->seq_id[s * n_seq_tokens + j][0];
                        int32_t seq_idx = ubatch->seq_idx[seq_id];
                        llama_pos pos_j = ubatch->pos[s * n_seq_tokens + j];
                        int pos_rel_j = pos_j - p0[seq_idx];

                        for (int i = 0; i < n_seq_tokens; ++i) {
                            llama_pos pos_i = ubatch->pos[s * n_seq_tokens + i];
                            int pos_rel_i = pos_i - p0[seq_idx];

                            int index = pos_rel_j - pos_rel_i;
                            float s_index = index >= 0 ? -slopes[h] * index : -INFINITY;
                            data[seq_idx * n_head * n_seq_tokens * n_seq_tokens + h * n_seq_tokens * n_seq_tokens + j * n_seq_tokens + i] = s_index;
                        }
                    }
                }
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        bool res = true;

        if (params.ubatch.n_seq_tokens > 1) {
            res &= (   inp_q_decay &&    inp_q_decay->ne[2] == params.ubatch.n_seq_tokens);
            res &= (   inp_k_decay &&    inp_k_decay->ne[2] == params.ubatch.n_seq_tokens);
            res &= (inp_diag_decay && inp_diag_decay->ne[1] == params.ubatch.n_seq_tokens);
        }

        return res;
    }

    const llama_hparams & hparams;

    lm_ggml_tensor * inp_slopes     = nullptr; // F32 [n_head]
    lm_ggml_tensor * inp_q_decay    = nullptr; // F32 [1, n_head, n_batch]
    lm_ggml_tensor * inp_k_decay    = nullptr; // F32 [1, n_head, n_batch]
    lm_ggml_tensor * inp_diag_decay = nullptr; // F32 [n_batch, n_batch, n_head]
};

llama_model_minimax_01::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    LM_GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // LM_GGML_ASSERT(n_embd_head == n_rot); this is wrong in case of minimax, head_dim = 128, n_rot = 64

    const int64_t n_seqs  = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    LM_GGML_ASSERT(n_seqs != 0);
    LM_GGML_ASSERT(ubatch.equal_seqs());
    LM_GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    lm_ggml_tensor * cur;
    lm_ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * inp_hybrid = build_inp_mem_hybrid();
    auto * inp_rs = inp_hybrid->get_recr();

    lm_ggml_tensor * inp_pos = build_inp_pos();
    lm_ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_la * la = nullptr;

    auto inp = std::make_unique<llm_graph_input_la>(hparams);

    inp->inp_slopes = lm_ggml_new_tensor_1d(ctx0, LM_GGML_TYPE_F32, n_head);
    lm_ggml_set_input(inp->inp_slopes);
    cb(inp->inp_slopes, "slopes", -1);

    if (n_seq_tokens != 1) {
        inp->inp_q_decay = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, 1, n_head, n_seq_tokens, n_seqs);
        lm_ggml_set_input(inp->inp_q_decay);
        cb(inp->inp_q_decay, "q_decay_exp", -1);

        inp->inp_k_decay = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, 1, n_head, n_seq_tokens, n_seqs);
        lm_ggml_set_input(inp->inp_k_decay);
        cb(inp->inp_k_decay, "k_decay_exp", -1);

        inp->inp_diag_decay = lm_ggml_new_tensor_4d(ctx0, LM_GGML_TYPE_F32, n_seq_tokens, n_seq_tokens, n_head, n_seqs);
        lm_ggml_set_input(inp->inp_diag_decay);
        cb(inp->inp_diag_decay, "diag_decay_exp", -1);
    }

    la = (llm_graph_input_la *) res->add_input(std::move(inp));

    lm_ggml_tensor * slopes = la->inp_slopes;

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        lm_ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        lm_ggml_tensor * residual = cur;

        // self_attention
        if (!hparams.is_recr(il)) {
            // softmax attention layer

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = lm_ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            Kcur = lm_ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
                );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_hybrid->get_attn(),
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        } else {
            // lightning attention layer

            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            // TODO unneeded - any way to make conv states optional in recurrent memory?
            lm_ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            lm_ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            lm_ggml_build_forward_expand(gf, conv_state_all);

            float slope_scale = 1.0 - 1.0 * il / (n_layer - 1) + 1e-5;
            lm_ggml_tensor * slope_rate = lm_ggml_scale(ctx0, slopes, slope_scale);
            cb(slope_rate, "slope_rate", il);

            cur = lm_ggml_reshape_4d(ctx0, cur, cur->ne[0], n_seq_tokens, 1, n_seqs);

            lm_ggml_tensor * QKVcur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(QKVcur, "QKVcur", il);

            QKVcur = lm_ggml_silu(ctx0, QKVcur);
            cb(QKVcur, "QKVcur_silu", il);

            QKVcur = lm_ggml_reshape_4d(ctx0, QKVcur, n_embd_head * 3, n_head, n_seq_tokens, n_seqs);

            lm_ggml_tensor * Qcur = lm_ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 0*lm_ggml_element_size(QKVcur)*n_embd_head);
            lm_ggml_tensor * Kcur = lm_ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 1*lm_ggml_element_size(QKVcur)*n_embd_head);
            lm_ggml_tensor * Vcur = lm_ggml_view_4d(ctx0, QKVcur, n_embd_head, n_head, n_seq_tokens, n_seqs, QKVcur->nb[1], QKVcur->nb[2], QKVcur->nb[3], 2*lm_ggml_element_size(QKVcur)*n_embd_head);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // get previous KV
            lm_ggml_tensor * la_states_all = mctx_cur->get_s_l(il);
            lm_ggml_tensor * state = build_rs(inp_rs, la_states_all, hparams.n_embd_s(), n_seqs);

            lm_ggml_tensor * kv_old = lm_ggml_reshape_4d(ctx0, state, n_embd_head, n_embd_head, n_head, n_seqs);
            cb(kv_old, "kv_old", il);

            lm_ggml_tensor * qkv = nullptr;
            lm_ggml_tensor * kv_new = nullptr;

            if (n_seq_tokens == 1) {
                // lightning attention - optimized single token case for TG

                lm_ggml_tensor * slopes_neg = lm_ggml_scale(ctx0, slope_rate, -1.0);
                cb(slopes_neg, "slopes_neg", il);

                lm_ggml_tensor * ratio = lm_ggml_exp(ctx0, slopes_neg);
                cb(ratio, "ratio", il);

                lm_ggml_tensor * ratio_3d = lm_ggml_reshape_3d(ctx0, ratio, 1, 1, n_head);
                cb(ratio_3d, "ratio3d", il);

                lm_ggml_tensor * v_trans = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Vcur, 1, 2, 0, 3));
                cb(v_trans, "v_trans", il);

                lm_ggml_tensor * k_trans = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Kcur, 1, 2, 0, 3));
                cb(k_trans, "k_trans", il);

                lm_ggml_tensor * kv_cur = lm_ggml_mul_mat(ctx0, k_trans, v_trans);
                cb(kv_cur, "kv_cur", il);

                lm_ggml_tensor * kv_old_s = lm_ggml_mul(ctx0, kv_old, ratio_3d);
                cb(kv_old_s, "kv_old_s", il);

                kv_new = lm_ggml_add(ctx0, kv_old_s, kv_cur);
                cb(kv_new, "kv_new", il);

                lm_ggml_tensor * q_trans = lm_ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q_trans, "q_trans", il);

                qkv = lm_ggml_mul_mat(ctx0, kv_new, q_trans);
                cb(qkv, "qkv", il);
            } else if(n_seq_tokens > 1) {
                // lightning attention - general multi token case for PP

                lm_ggml_tensor *    q_decay_exp = la->inp_q_decay;
                lm_ggml_tensor *    k_decay_exp = la->inp_k_decay;
                lm_ggml_tensor * diag_decay_exp = la->inp_diag_decay;

                lm_ggml_tensor *    q_decay = lm_ggml_exp(ctx0, lm_ggml_scale(ctx0, q_decay_exp, slope_scale));
                cb(q_decay, "q_decay", il);
                lm_ggml_tensor *    k_decay = lm_ggml_exp(ctx0, lm_ggml_scale(ctx0, k_decay_exp, slope_scale));
                cb(k_decay, "k_decay", il);
                lm_ggml_tensor * diag_decay = lm_ggml_exp(ctx0, lm_ggml_scale(ctx0, diag_decay_exp, slope_scale));
                cb(diag_decay, "diag_decay", il);

                lm_ggml_tensor * q_s = lm_ggml_mul(ctx0, Qcur, q_decay);
                cb(q_s, "q_s", il);

                lm_ggml_tensor * q_s_trans = lm_ggml_permute(ctx0, q_s, 0, 2, 1, 3);
                cb(q_s_trans, "q_s_trans", il);

                lm_ggml_tensor * qkv_none_diag = lm_ggml_mul_mat(ctx0, kv_old, q_s_trans);
                cb(qkv_none_diag, "qkv_none_diag", il);

                lm_ggml_tensor * q_trans = lm_ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                cb(q_trans, "q_trans", il);

                lm_ggml_tensor * k_trans = lm_ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
                cb(k_trans, "k_trans", il);

                lm_ggml_tensor * qk = lm_ggml_mul_mat(ctx0, k_trans, q_trans);
                cb(qk, "qk", il);

                qk = lm_ggml_mul(ctx0, qk, diag_decay);
                cb(qk, "qk_s", il);

                lm_ggml_tensor * v_trans = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, Vcur, 1, 2, 0, 3));
                cb(v_trans, "v_trans", il);

                lm_ggml_tensor * qkv_diag = lm_ggml_mul_mat(ctx0, v_trans, qk);
                cb(qkv_diag, "qkv_diag", il);

                qkv = lm_ggml_add(ctx0, qkv_none_diag, qkv_diag);
                cb(qkv, "qkv", il);

                lm_ggml_build_forward_expand(gf, qkv);

                lm_ggml_tensor * slopes_neg = lm_ggml_scale(ctx0, slope_rate, -1.0*n_seq_tokens);
                cb(slopes_neg, "slopes_neg", il);

                lm_ggml_tensor * block_decay = lm_ggml_exp(ctx0, slopes_neg);
                cb(block_decay, "block_decay", il);

                lm_ggml_tensor * block_decay_3d = lm_ggml_reshape_3d(ctx0, block_decay, 1, 1, n_head);
                cb(block_decay_3d, "block_decay_3d", il);

                lm_ggml_tensor * kv_old_s = lm_ggml_mul(ctx0, kv_old, block_decay_3d);
                cb(kv_old_s, "kv_old_s", il);

                lm_ggml_tensor * k_after_decay = lm_ggml_mul(ctx0, Kcur, k_decay);
                cb(k_after_decay, "k_after_decay", il);

                lm_ggml_tensor * k_after_decay_trans = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, k_after_decay, 1, 2, 0, 3));
                cb(k_after_decay_trans, "k_after_decay_trans", il);

                lm_ggml_tensor * kv_cur = lm_ggml_mul_mat(ctx0, k_after_decay_trans, v_trans);
                cb(kv_cur, "kv_cur", il);

                kv_new = lm_ggml_add(ctx0, kv_old_s, kv_cur);
                cb(kv_new, "kv_new", il);
            }

            // store new KV
            lm_ggml_build_forward_expand(gf,
                                     lm_ggml_cpy(ctx0, kv_new,
                                              lm_ggml_view_1d(ctx0, la_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * lm_ggml_element_size(la_states_all))));

            qkv = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, qkv, 0, 2, 1, 3));
            cb(qkv, "qkv_permuted", il);

            qkv = lm_ggml_reshape_4d(ctx0, qkv, qkv->ne[0]*qkv->ne[1], qkv->ne[2], 1, qkv->ne[3]);

            // norm
            lm_ggml_tensor * qkv_norm = build_norm(qkv,
                    model.layers[il].attn_norm_2, NULL,
                    LLM_NORM_RMS, il);
            cb(qkv_norm, "qkv_norm", il);

            lm_ggml_tensor * g = build_lora_mm(model.layers[il].wg, cur);
            cb(g, "g", il);

            g = lm_ggml_sigmoid(ctx0, g);
            cb(g, "g_sigm", il);

            cur = lm_ggml_mul(ctx0, g, qkv_norm);

            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out", il);

            cur = lm_ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens*n_seqs);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = lm_ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = lm_ggml_get_rows(ctx0, inpSA, inp_out_ids);
            residual = lm_ggml_get_rows(ctx0, residual, inp_out_ids);
        }

        residual = lm_ggml_scale(ctx0, residual, hparams.f_residual_scale);
        cb(residual, "residual_scaled_attn", il);

        lm_ggml_tensor * ffn_inp = lm_ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        residual = cur;

        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                il);
        cb(cur, "ffn_moe_out", il);

        residual = lm_ggml_scale(ctx0, residual, hparams.f_residual_scale);
        cb(residual, "residual_scaled_ffn", il);

        cur = lm_ggml_add(ctx0, cur, residual);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    lm_ggml_build_forward_expand(gf, cur);
}
