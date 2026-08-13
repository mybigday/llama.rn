#include "models.h"

#include <cmath>

// pocket-tts generation stages
//
// GEN_CODE: backbone hidden state -> next 32-d latent (flow matching) + end-of-speech score
// GEN_WAV : a window of latents -> PCM, through the mimi decoder
//
// there is no codebook anywhere, "codes" in the mtmd API are continuous features here

lm_ggml_tensor * clip_graph_pockettts_gen::modulate(lm_ggml_tensor * x, lm_ggml_tensor * shift, lm_ggml_tensor * scale) const {
    lm_ggml_tensor * cur = lm_ggml_mul(ctx0, x, lm_ggml_scale_bias(ctx0, scale, 1.0f, 1.0f));
    return lm_ggml_add(ctx0, cur, shift);
}

// see TimestepEmbedder in the reference
lm_ggml_tensor * clip_graph_pockettts_gen::time_embed(const clip_flow_net::time_embd & te, float t) const {
    // t is a graph-build constant, so the cos/sin table can be folded into a scaled copy
    lm_ggml_tensor * args = lm_ggml_scale(ctx0, te.freqs, t);
    lm_ggml_tensor * emb  = lm_ggml_concat(ctx0, lm_ggml_cos(ctx0, args), lm_ggml_sin(ctx0, args), 0);

    lm_ggml_tensor * cur = build_mm(te.up_w, emb);
    cur = lm_ggml_add(ctx0, cur, te.up_b);
    cur = lm_ggml_silu(ctx0, cur);
    cur = build_mm(te.down_w, cur);
    cur = lm_ggml_add(ctx0, cur, te.down_b);

    // this "RMSNorm" divides by the unbiased variance, not the mean square
    // it also rescales the input, not the centered value, see _rms_norm() in mlp.py
    {
        const int64_t n = cur->ne[0];
        lm_ggml_tensor * mean = lm_ggml_mean(ctx0, cur);
        lm_ggml_tensor * dev  = lm_ggml_sub(ctx0, cur, mean);
        lm_ggml_tensor * var  = lm_ggml_mean(ctx0, lm_ggml_sqr(ctx0, dev));
        var = lm_ggml_scale_bias(ctx0, var, (float) n / (float) (n - 1), 1e-5f);
        cur = lm_ggml_div(ctx0, cur, lm_ggml_sqrt(ctx0, var));
        cur = lm_ggml_mul(ctx0, cur, te.norm);
    }

    return cur;
}

// one velocity evaluation: v(cond, s, t, x)
lm_ggml_tensor * clip_graph_pockettts_gen::flow_forward(lm_ggml_tensor * cond, lm_ggml_tensor * x, float s, float t) const {
    const auto & flow = model.flow;

    lm_ggml_tensor * cur = build_mm(flow.input_proj_w, x);
    cur = lm_ggml_add(ctx0, cur, flow.input_proj_b);

    // the two time conditions are averaged, then added to the projected backbone state
    lm_ggml_tensor * ts = lm_ggml_add(ctx0, time_embed(flow.time[0], s), time_embed(flow.time[1], t));
    ts = lm_ggml_scale(ctx0, ts, 1.0f / (float) flow.time.size());

    lm_ggml_tensor * c = build_mm(flow.cond_embd_w, cond);
    c = lm_ggml_add(ctx0, c, flow.cond_embd_b);

    lm_ggml_tensor * y = lm_ggml_add(ctx0, ts, c);
    cb(y, "flow_cond", -1);

    const int64_t n_ch = flow.blocks.empty() ? 0 : flow.blocks[0].norm_w->ne[0];

    for (size_t il = 0; il < flow.blocks.size(); il++) {
        const auto & blk = flow.blocks[il];

        lm_ggml_tensor * mod = build_mm(blk.ada_w, lm_ggml_silu(ctx0, y));
        mod = lm_ggml_add(ctx0, mod, blk.ada_b);

        lm_ggml_tensor * shift = lm_ggml_view_1d(ctx0, mod, n_ch, 0);
        lm_ggml_tensor * scale = lm_ggml_view_1d(ctx0, mod, n_ch, (size_t) n_ch * mod->nb[0]);
        lm_ggml_tensor * gate  = lm_ggml_view_1d(ctx0, mod, n_ch, (size_t) 2 * n_ch * mod->nb[0]);

        lm_ggml_tensor * h = build_norm(cur, blk.norm_w, blk.norm_b, NORM_TYPE_NORMAL, 1e-6f, (int) il);
        h = modulate(h, shift, scale);
        h = build_mm(blk.up_w, h);
        h = lm_ggml_add(ctx0, h, blk.up_b);
        h = lm_ggml_silu(ctx0, h);
        h = build_mm(blk.down_w, h);
        h = lm_ggml_add(ctx0, h, blk.down_b);

        cur = lm_ggml_add(ctx0, cur, lm_ggml_mul(ctx0, gate, h));
        cb(cur, "flow_blk", (int) il);
    }

    // final layer: the norm has no weights, only the AdaLN modulation
    lm_ggml_tensor * mod = build_mm(flow.final_ada_w, lm_ggml_silu(ctx0, y));
    mod = lm_ggml_add(ctx0, mod, flow.final_ada_b);

    lm_ggml_tensor * shift = lm_ggml_view_1d(ctx0, mod, n_ch, 0);
    lm_ggml_tensor * scale = lm_ggml_view_1d(ctx0, mod, n_ch, (size_t) n_ch * mod->nb[0]);

    cur = build_norm(cur, nullptr, nullptr, NORM_TYPE_NORMAL, 1e-6f, -1);
    cur = modulate(cur, shift, scale);
    cur = build_mm(flow.final_proj_w, cur);
    cur = lm_ggml_add(ctx0, cur, flow.final_proj_b);

    return cur;
}

// state carried between GEN_WAV calls: rope offset, per-layer KV window, conv left context
// and the transposed-conv overlap tails
std::vector<c2w_state_slot> list_pockettts_state_slots(const clip_hparams & hparams, const clip_model & model) {
    std::vector<c2w_state_slot> slots;
    if (model.gen_upsample_w == nullptr) {
        return slots; // not a pocket-tts decoder
    }
    const auto & seanet = model.seanet;

    // the slots below are sized from these
    LM_GGML_ASSERT(!model.gen_tfm_layers.empty());
    LM_GGML_ASSERT((int) seanet.stages.size() >= hparams.seanet_n_stage);
    LM_GGML_ASSERT((int) hparams.seanet_ratios.size() >= hparams.seanet_n_stage);
    LM_GGML_ASSERT(hparams.mimi_tfm_context > 1 && hparams.mimi_downsample > 0);

    slots.push_back({"tfm_pos", 1, 1});

    const int64_t n_embd_a = model.gen_tfm_layers[0].q_w->ne[1];
    const int64_t prefix   = hparams.mimi_tfm_context - 1;
    for (size_t il = 0; il < model.gen_tfm_layers.size(); il++) {
        slots.push_back({"tfm_k_" + std::to_string(il), n_embd_a, prefix});
        slots.push_back({"tfm_v_" + std::to_string(il), n_embd_a, prefix});
    }

    // upsample is depthwise, its output channel count is the input one
    slots.push_back({"up", model.gen_upsample_w->ne[0] - hparams.mimi_downsample, model.gen_upsample_w->ne[2]});

    slots.push_back({"dec_in", seanet.conv_in_w->ne[0] - 1, seanet.conv_in_w->ne[1]});
    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage  = seanet.stages[i];
        const int    stride = hparams.seanet_ratios[hparams.seanet_n_stage - 1 - i];
        slots.push_back({"dec_up_" + std::to_string(i), stage.scale_conv_w->ne[0] - stride, stage.scale_conv_w->ne[1]});
        slots.push_back({"dec_res_" + std::to_string(i), stage.res_conv1_w->ne[0] - 1, stage.res_conv1_w->ne[1]});
    }
    slots.push_back({"dec_out", seanet.conv_out_w->ne[0] - 1, seanet.conv_out_w->ne[1]});

    return slots;
}

lm_ggml_cgraph * clip_graph_pockettts_gen::build() {
    if (gen_process == CLIP_GEN_PROCESS_GEN_CODE) {
        // the backbone hidden state arrives as the single batch entry
        lm_ggml_tensor * h_state = build_inp_raw(1);
        h_state = lm_ggml_reshape_2d(ctx0, h_state, n_mmproj_embd, 1);

        // end-of-speech probe, thresholded on the host side
        lm_ggml_tensor * eos = build_mm(model.gen_out_eos_w, h_state);
        eos = lm_ggml_add(ctx0, eos, model.gen_out_eos_b);
        lm_ggml_set_name(eos, "out_eos_score");
        lm_ggml_set_output(eos);
        lm_ggml_build_forward_expand(gf, eos);

        const int64_t n_latent = model.gen_input_lin_w->ne[0];

        lm_ggml_tensor * noise = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, n_latent, 1);
        lm_ggml_set_name(noise, "inp_noise");
        lm_ggml_set_input(noise);

        // lsd_decode: integrate the velocity field from the noise sample
        lm_ggml_tensor * cur = noise;
        for (int i = 0; i < n_step; i++) {
            const float s = (float) i / (float) n_step;
            const float t = (float) (i + 1) / (float) n_step;
            lm_ggml_tensor * v = flow_forward(h_state, cur, s, t);
            cur = lm_ggml_add(ctx0, cur, lm_ggml_scale(ctx0, v, 1.0f / (float) n_step));
        }
        cb(cur, "flow_latent", -1);

        lm_ggml_set_name(cur, "out_feats");
        lm_ggml_set_output(cur);
        lm_ggml_build_forward_expand(gf, cur);

        // the same latent, projected into the backbone's input space for the next step
        lm_ggml_tensor * embd = build_mm(model.gen_input_lin_w, cur);
        cb(embd, "gen_embd", -1);
        lm_ggml_build_forward_expand(gf, embd);

        return gf;
    }

    // GEN_WAV: [32, n_frames] latents -> PCM
    lm_ggml_tensor * feats = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32,
                                             model.gen_input_lin_w->ne[0], n_frames);
    lm_ggml_set_name(feats, "inp_feats");
    lm_ggml_set_input(feats);

    // denormalize, then the DummyQuantizer up-projection
    lm_ggml_tensor * cur = lm_ggml_add(ctx0, lm_ggml_mul(ctx0, feats, model.gen_emb_std), model.gen_emb_mean);
    cur = build_mm(model.gen_quant_out_w, cur);
    cb(cur, "quant_out", -1);

    clip_graph_pockettts_seanet seanet(*this);
    for (const auto & slot : list_pockettts_state_slots(hparams, model)) {
        lm_ggml_tensor * t = lm_ggml_new_tensor_2d(ctx0, LM_GGML_TYPE_F32, slot.ne0, slot.ne1);
        lm_ggml_set_name(t, ("state_in_" + slot.name).c_str());
        lm_ggml_set_input(t);
        seanet.state_in[slot.name] = t;
    }

    // model frame rate -> encoder frame rate, depthwise transposed conv
    cur = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, cur));
    cur = seanet.conv_transpose1d(cur, model.gen_upsample_w, nullptr, hparams.mimi_downsample, "up");
    cb(cur, "mimi_upsample", -1);

    cur = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, cur));

    // positions continue across calls, the counter lives in the state
    const int64_t n_pos  = cur->ne[1];
    const int64_t prefix = hparams.mimi_tfm_context - 1;
    const int64_t n_kv   = prefix + n_pos;

    lm_ggml_tensor * base    = lm_ggml_reshape_1d(ctx0, seanet.state_in.at("tfm_pos"), 1);
    lm_ggml_tensor * inp_pos = lm_ggml_cast(ctx0, lm_ggml_add(ctx0, lm_ggml_arange(ctx0, 0.0f, (float) n_pos, 1.0f), base),
                                      LM_GGML_TYPE_I32);
    seanet.state_out.push_back({"tfm_pos", lm_ggml_scale_bias(ctx0, seanet.state_in.at("tfm_pos"), 1.0f, (float) n_pos)});

    // banded causal mask over [cached prefix | this chunk]
    // the last factor masks out cache rows that hold no real frame yet
    lm_ggml_tensor * pos_k = lm_ggml_reshape_2d(ctx0, lm_ggml_arange(ctx0, 0.0f, (float) n_kv, 1.0f), n_kv, 1);
    lm_ggml_tensor * pos_q = lm_ggml_reshape_2d(ctx0, lm_ggml_arange(ctx0, (float) prefix, (float) (prefix + n_pos), 1.0f), 1, n_pos);
    lm_ggml_tensor * diff  = lm_ggml_sub(ctx0, lm_ggml_repeat_4d(ctx0, pos_q, n_kv, n_pos, 1, 1), pos_k);

    lm_ggml_tensor * keep = lm_ggml_mul(ctx0,
        lm_ggml_step(ctx0, lm_ggml_scale_bias(ctx0, diff, 1.0f, 0.5f)),                                     // delta >= 0
        lm_ggml_step(ctx0, lm_ggml_scale_bias(ctx0, diff, -1.0f, (float) hparams.mimi_tfm_context - 0.5f))); // delta < context
    keep = lm_ggml_mul(ctx0, keep,
        lm_ggml_step(ctx0, lm_ggml_scale_bias(ctx0, lm_ggml_add(ctx0, pos_k, base), 1.0f, 0.5f - (float) prefix)));
    lm_ggml_tensor * kq_mask = lm_ggml_reshape_4d(ctx0, lm_ggml_log(ctx0, keep), n_kv, n_pos, 1, 1);

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.gen_tfm_layers[il];
        lm_ggml_tensor * inp = cur;

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

        lm_ggml_tensor * Qcur = build_mm(layer.q_w, cur);
        lm_ggml_tensor * Kcur = build_mm(layer.k_w, cur);
        lm_ggml_tensor * Vcur = build_mm(layer.v_w, cur);

        Qcur = lm_ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
        Kcur = lm_ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);

        Qcur = lm_ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, d_head, LM_GGML_ROPE_TYPE_NORMAL, 0,
                             hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = lm_ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, d_head, LM_GGML_ROPE_TYPE_NORMAL, 0,
                             hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // prepend the cached window, then keep this chunk's tail for the next call
        const std::string k_name = "tfm_k_" + std::to_string(il);
        const std::string v_name = "tfm_v_" + std::to_string(il);
        lm_ggml_tensor * k_full = lm_ggml_concat(ctx0, seanet.state_in.at(k_name),
                                           lm_ggml_reshape_2d(ctx0, Kcur, d_head * n_head, n_pos), 1);
        lm_ggml_tensor * v_full = lm_ggml_concat(ctx0, seanet.state_in.at(v_name), Vcur, 1);
        seanet.state_out.push_back({k_name, lm_ggml_cont(ctx0, lm_ggml_view_2d(ctx0, k_full, k_full->ne[0], prefix,
                                                                         k_full->nb[1], (size_t) n_pos * k_full->nb[1]))});
        seanet.state_out.push_back({v_name, lm_ggml_cont(ctx0, lm_ggml_view_2d(ctx0, v_full, v_full->ne[0], prefix,
                                                                         v_full->nb[1], (size_t) n_pos * v_full->nb[1]))});

        lm_ggml_tensor * q_cur = lm_ggml_reshape_4d(ctx0, Qcur,   d_head, n_head, n_pos, 1);
        lm_ggml_tensor * k_cur = lm_ggml_reshape_4d(ctx0, k_full, d_head, n_head, n_kv,  1);
        lm_ggml_tensor * v_cur = lm_ggml_reshape_4d(ctx0, v_full, d_head, n_head, n_kv,  1);

        cur = build_attn(layer.o_w, nullptr, q_cur, k_cur, v_cur, kq_mask, kq_scale, il);
        cur = lm_ggml_mul(ctx0, cur, layer.ls_1_w);
        cur = lm_ggml_add(ctx0, cur, inp);

        inp = cur;
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cur = build_ffn(cur, layer.ff_up_w, nullptr, nullptr, nullptr, layer.ff_down_w, nullptr, FFN_GELU, il);
        cur = lm_ggml_mul(ctx0, cur, layer.ls_2_w);
        cur = lm_ggml_add(ctx0, cur, inp);
    }
    cb(cur, "mimi_dec_tfm", -1);

    cur = lm_ggml_cont(ctx0, lm_ggml_transpose(ctx0, cur));
    cur = seanet.decode(cur);

    for (const auto & s : seanet.state_out) {
        lm_ggml_set_name(s.second, ("state_out_" + s.first).c_str());
        lm_ggml_set_output(s.second);
        lm_ggml_build_forward_expand(gf, s.second);
    }

    // [n_samples, 1] -> [n_samples], clamped like the reference output
    cur = lm_ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    cur = lm_ggml_clamp(ctx0, cur, -1.0f, 1.0f);
    lm_ggml_set_name(cur, "out_audio");
    lm_ggml_set_output(cur);
    lm_ggml_build_forward_expand(gf, cur);

    return gf;
}
