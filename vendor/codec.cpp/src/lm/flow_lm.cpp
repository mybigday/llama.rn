#include "lm_internal.h"
#include "spm_unigram.h"

#include "../ops/ggml_ops.h"
#include "../runtime/graph.h"
#include "../runtime/gguf_kv.h"
#include "../runtime/tensor_utils.h"

#include <ggml.h>
#include <gguf.h>

#include <cmath>
#include <cstring>
#include <new>
#include <random>
#include <string>
#include <vector>

// =====================================================================
// codec_lm kind: flow_lm  (Kyutai Pocket-TTS)
//
// Continuous-latent AR adaptor.  The FlowLM transformer + out_norm run in
// llama.cpp as a stock `starcoder2` backbone (scripts/convert-backbone-to-
// gguf.py; its token_embd is the text LUT).  This side owns what surrounds it:
//   prefix rows  = [ bos_before_voice | speaker_proj(voice latents) ]
//   step input   = input_linear( bos_emb (first frame) | prev latent )
//   per step     = eos_logit = out_eos(h);  latent = LSD_decode(flow_net(h, ·), noise)
// where h is the backbone's (out_norm'ed) hidden.  One cached graph per step
// emits the latent, the EOS logit and the next step's input embedding.
//
// The flow head uses eps-1e-6 LayerNorm (affine in_ln, affine-free final) and
// an UNBIASED (ddof=1) RMSNorm inside the timestep embedders — both
// replicated exactly here.
// =====================================================================

namespace {

// POD config copied into graph build_user_data (must be trivially copyable —
// the graph cache key hashes it byte-wise).  Keeps the non-copyable tokenizer
// out of the build structs.
struct flow_impl {
    int32_t d_model;       // 1024 (backbone hidden)
    int32_t ldim;          // 32
    int32_t flow_dim;      // 512
    int32_t flow_depth;    // 6
    int32_t n_txt_bins;    // 4000
    int32_t insert_bos_before_voice;
    int32_t lsd_decode_steps;
    int32_t frames_after_eos;
    float   temperature;   // 0.7
    float   eos_threshold; // -4.0
    float   flow_ln_eps;   // 1e-6 (flow head LN)
    float   flow_rms_eps;  // 1e-5 (time-embed RMSNorm)
};

// Per-model state owned by codec_lm (adds the non-copyable tokenizer).
struct flow_model {
    flow_impl  cfg;
    SpmUnigram spm;
    bool       has_spm = false;
};

struct flow_state {
    int32_t            frame = 0;      // AR frames generated so far
    std::mt19937       rng;
    std::vector<float> next_embd;      // [d_model] input embedding for the next backbone step
    std::vector<float> teacher;        // [ldim] armed teacher latent (denormalized), empty = off
};

ggml_tensor * lin(ggml_context * c, ggml_tensor * w, ggml_tensor * x, ggml_tensor * b) {
    ggml_tensor * y = codec_mul_mat(c, w, x);
    return b ? ggml_add(c, y, codec_graph_cast_f32(c, b)) : y;
}

// modulate(x, shift, scale) = x*(1+scale) + shift.  x is (dim, N); shift/scale
// are (dim, 1) broadcast over N.
ggml_tensor * modulate(ggml_context * c, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) {
    ggml_tensor * one_plus = ggml_scale_bias(c, scale, 1.0f, 1.0f);   // scale + 1
    return ggml_add(c, ggml_mul(c, x, one_plus), shift);
}

// Unbiased (ddof=1) RMSNorm from pocket_tts.modules.mlp._rms_norm:
//   var = eps + x.var(dim=-1, keepdim=True)   [unbiased over `dim` elements]
//   y   = x * alpha * rsqrt(var)
// x is (dim, N).  alpha is (dim,).
ggml_tensor * rms_norm_unbiased(ggml_context * c, ggml_tensor * x, ggml_tensor * alpha, float eps) {
    const int64_t D = x->ne[0];
    ggml_tensor * mean = ggml_scale(c, ggml_sum_rows(c, x), 1.0f / (float) D);   // (1, N)
    ggml_tensor * xc   = ggml_sub(c, x, mean);
    ggml_tensor * sq   = ggml_mul(c, xc, xc);
    // unbiased variance: sum(sq)/(D-1)
    ggml_tensor * var  = ggml_scale(c, ggml_sum_rows(c, sq), 1.0f / (float) (D - 1));  // (1, N)
    ggml_tensor * denom = ggml_sqrt(c, ggml_scale_bias(c, var, 1.0f, eps));            // sqrt(var+eps), (1,N)
    ggml_tensor * y    = ggml_div(c, x, denom);                     // x / sqrt(var+eps)
    return ggml_mul(c, y, ggml_reshape_2d(c, codec_graph_cast_f32(c, alpha), D, 1));
}

// TimestepEmbedder(scalar t) -> (flow_dim,): args = t*freqs; cat[cos,sin]; mlp;
// then unbiased RMSNorm.  `sval` is the scalar time; freqs is (flow_dim/2,).
ggml_tensor * flow_time_embed(ggml_context * ctx, const codec_model * model, const flow_impl & I,
    int32_t idx, float sval) {
    const std::string p = "lm.flow.time_embed." + std::to_string(idx);
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, p + s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, p + s); };
    ggml_tensor * freqs = codec_graph_cast_f32(ctx, W(".freqs"));   // (half,) = freq_embed/2
    const int64_t half = freqs->ne[0];
    ggml_tensor * args  = ggml_scale(ctx, freqs, sval);             // t*freqs
    // frequency embedding = cat([cos(args), sin(args)]) -> (2*half,) = freq_embed_size
    ggml_tensor * emb   = ggml_concat(ctx, codec_op_cos(ctx, args), codec_op_sin(ctx, args), 0);
    emb = ggml_reshape_2d(ctx, emb, 2 * half, 1);
    ggml_tensor * h = ggml_silu(ctx, lin(ctx, WM(".l1.w"), emb, W(".l1.b")));   // (flow_dim,1)
    h = lin(ctx, WM(".l2.w"), h, W(".l2.b"));
    return rms_norm_unbiased(ctx, h, W(".rms.alpha"), I.flow_rms_eps);   // (flow_dim,1)
}

// SimpleMLPAdaLN flow_net(c, s, t, x): c is (d_model,1) cond, x is (ldim,1)
// input, s/t are scalars.  Returns (ldim,1) flow direction.
ggml_tensor * flow_net(ggml_context * ctx, const codec_model * model, const flow_impl & I,
    ggml_tensor * cond, float sval, float tval, ggml_tensor * x) {
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };

    ggml_tensor * xh = lin(ctx, WM("lm.flow.input_proj.w"), x, W("lm.flow.input_proj.b"));   // (flow_dim,1)
    ggml_tensor * ts = flow_time_embed(ctx, model, I, 0, sval);
    ggml_tensor * tt = flow_time_embed(ctx, model, I, 1, tval);
    ggml_tensor * t_comb = ggml_scale(ctx, ggml_add(ctx, ts, tt), 0.5f);                     // (flow_dim,1)
    ggml_tensor * ce = lin(ctx, WM("lm.flow.cond_embed.w"), cond, W("lm.flow.cond_embed.b"));// (flow_dim,1)
    ggml_tensor * y  = ggml_add(ctx, t_comb, ce);                                            // (flow_dim,1)
    ggml_tensor * sy = ggml_silu(ctx, y);   // SiLU applied inside adaLN Sequential

    for (int32_t b = 0; b < I.flow_depth; ++b) {
        const std::string rp = "lm.flow.res." + std::to_string(b);
        auto RW  = [&](const char * s) { return codec_graph_weight(ctx, model, rp + s); };
        auto RWM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, rp + s); };
        // adaLN(silu(y)) -> chunk(3): shift, scale, gate  (each flow_dim).
        ggml_tensor * mod = lin(ctx, RWM(".adaln.w"), sy, RW(".adaln.b"));   // (3*flow_dim,1)
        ggml_tensor * shift = ggml_cont(ctx, ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], 0));
        ggml_tensor * scale = ggml_cont(ctx, ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], (size_t) I.flow_dim * mod->nb[0]));
        ggml_tensor * gate  = ggml_cont(ctx, ggml_view_2d(ctx, mod, I.flow_dim, 1, mod->nb[1], (size_t) 2 * I.flow_dim * mod->nb[0]));
        ggml_tensor * hn = codec_op_layer_norm_ct(ctx, xh, I.flow_ln_eps, RW(".in_ln.w"), RW(".in_ln.b"));
        hn = modulate(ctx, hn, shift, scale);
        hn = lin(ctx, RWM(".mlp.l2.w"), ggml_silu(ctx, lin(ctx, RWM(".mlp.l1.w"), hn, RW(".mlp.l1.b"))), RW(".mlp.l2.b"));
        xh = ggml_add(ctx, xh, ggml_mul(ctx, gate, hn));
    }
    // FinalLayer: adaLN(silu(y)) -> chunk(2): shift, scale; norm_final (no
    // affine); modulate; linear.
    ggml_tensor * fmod = lin(ctx, WM("lm.flow.final.adaln.w"), sy, W("lm.flow.final.adaln.b"));  // (2*flow_dim,1)
    ggml_tensor * fshift = ggml_cont(ctx, ggml_view_2d(ctx, fmod, I.flow_dim, 1, fmod->nb[1], 0));
    ggml_tensor * fscale = ggml_cont(ctx, ggml_view_2d(ctx, fmod, I.flow_dim, 1, fmod->nb[1], (size_t) I.flow_dim * fmod->nb[0]));
    ggml_tensor * xf = ggml_norm(ctx, xh, I.flow_ln_eps);   // affine-free LayerNorm
    xf = modulate(ctx, xf, fshift, fscale);
    return lin(ctx, WM("lm.flow.final.linear.w"), xf, W("lm.flow.final.linear.b"));   // (ldim,1)
}

// emb_std / emb_mean as (ldim, 1) F32: the FlowLM predicts normalized latents,
// Mimi consumes  latent * emb_std + emb_mean.
ggml_tensor * denorm(ggml_context * ctx, const codec_model * model, const flow_impl & I, ggml_tensor * x) {
    ggml_tensor * sd = ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, codec_graph_weight(ctx, model, "lm.emb_std")), I.ldim, 1);
    ggml_tensor * mu = ggml_reshape_2d(ctx, codec_graph_cast_f32(ctx, codec_graph_weight(ctx, model, "lm.emb_mean")), I.ldim, 1);
    return ggml_add(ctx, ggml_mul(ctx, x, sd), mu);
}

// ---------------------------------------------------------------------
// Per-step graph: backbone hidden h -> EOS logit + LSD latent + the next
// step's input embedding input_linear(latent).
// ---------------------------------------------------------------------
struct flow_step_build {
    flow_impl imp;
    int32_t   n_steps;     // LSD steps (unrolled)
    const codec_model * model;
};

bool build_step(ggml_context * ctx, void * ud, ggml_tensor ** out) {
    flow_step_build * p = static_cast<flow_step_build *>(ud);
    const flow_impl & I = p->imp;
    const codec_model * model = p->model;
    auto W  = [&](const char * s) { return codec_graph_weight(ctx, model, s); };
    auto WM = [&](const char * s) { return codec_graph_weight_mat(ctx, model, s); };

    ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I.d_model, 1);
    ggml_set_name(c, "flow.st.h"); ggml_set_input(c);

    ggml_tensor * eos = lin(ctx, WM("lm.out_eos.w"), c, W("lm.out_eos.b"));   // (1,1)
    ggml_set_name(eos, "flow.st.eos"); ggml_set_output(eos);

    // ---- LSD flow (unrolled Euler) ----
    //   current = noise;  for i in [0,n): s=i/n, t=(i+1)/n;
    //     current += flow_net(c, s, t, current) / n
    ggml_tensor * noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I.ldim, 1);
    ggml_set_name(noise, "flow.st.noise"); ggml_set_input(noise);
    ggml_tensor * cur = noise;
    const int32_t n = p->n_steps;
    for (int32_t i = 0; i < n; ++i) {
        const float sval = (float) i / (float) n;
        const float tval = (float) (i + 1) / (float) n;
        ggml_tensor * dir = flow_net(ctx, model, I, c, sval, tval, cur);
        cur = ggml_add(ctx, cur, ggml_scale(ctx, dir, 1.0f / (float) n));
    }
    ggml_tensor * embd = codec_mul_mat(ctx, WM("lm.input_linear.w"), cur);   // (d_model,1)
    ggml_set_name(embd, "flow.st.embd"); ggml_set_output(embd);
    ggml_tensor * lat = denorm(ctx, model, I, cur);
    ggml_set_name(lat, "flow.st.latent"); ggml_set_output(lat);
    *out = lat;
    return true;
}

// ---------------------------------------------------------------------
// Input-embedding graph: input_linear(bos_emb) (is_bos) or
// input_linear(normalize(latent)) for a teacher-forced latent.
// ---------------------------------------------------------------------
struct flow_inproj_build { flow_impl imp; int32_t is_bos; const codec_model * model; };

bool build_inproj(ggml_context * ctx, void * ud, ggml_tensor ** out) {
    flow_inproj_build * p = static_cast<flow_inproj_build *>(ud);
    const flow_impl & I = p->imp;
    const codec_model * model = p->model;
    auto W = [&](const char * s) { return codec_graph_cast_f32(ctx, codec_graph_weight(ctx, model, s)); };
    ggml_tensor * x;
    if (p->is_bos) {
        x = ggml_reshape_2d(ctx, W("lm.bos_emb"), I.ldim, 1);
    } else {
        ggml_tensor * lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I.ldim, 1);
        ggml_set_name(lat, "flow.ip.latent"); ggml_set_input(lat);
        x = ggml_div(ctx, ggml_sub(ctx, lat, ggml_reshape_2d(ctx, W("lm.emb_mean"), I.ldim, 1)),
                     ggml_reshape_2d(ctx, W("lm.emb_std"), I.ldim, 1));
    }
    ggml_tensor * embd = codec_mul_mat(ctx, codec_graph_weight_mat(ctx, model, "lm.input_linear.w"), x);
    ggml_set_name(embd, "flow.ip.embd"); ggml_set_output(embd);
    *out = embd;
    return true;
}

// ---------------------------------------------------------------------
// speaker_proj graph: mu [ldim, T] -> [d_model, T] rows (F.linear, no bias).
// ---------------------------------------------------------------------
struct flow_spk_build { flow_impl imp; int32_t T; const codec_model * model; };

bool build_spkproj(ggml_context * ctx, void * ud, ggml_tensor ** out) {
    flow_spk_build * p = static_cast<flow_spk_build *>(ud);
    const flow_impl & I = p->imp;
    ggml_tensor * mu = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I.ldim, p->T);
    ggml_set_name(mu, "flow.sp.mu"); ggml_set_input(mu);
    ggml_tensor * w = codec_graph_weight_mat(ctx, p->model, "lm.speaker_proj.w");   // (ldim, d_model)
    ggml_tensor * rows = codec_mul_mat(ctx, w, mu);   // (d_model, T)
    ggml_set_name(rows, "flow.sp.rows"); ggml_set_output(rows);
    *out = rows;
    return true;
}

// =====================================================================
// vtable impl
// =====================================================================

bool init(codec_lm * lm) {
    if (lm == nullptr || lm->codec == nullptr) return false;
    gguf_context * gf = lm->codec->gguf;
    flow_model * M = new (std::nothrow) flow_model();
    if (!M) { lm->last_error = "oom"; return false; }
    flow_impl * I = &M->cfg;
    I->d_model    = codec_read_i32_kv(gf, "codec.lm.d_model", 1024);
    I->ldim       = codec_read_i32_kv(gf, "codec.lm.ldim", 32);
    I->flow_dim   = codec_read_i32_kv(gf, "codec.lm.flow_dim", 512);
    I->flow_depth = codec_read_i32_kv(gf, "codec.lm.flow_depth", 6);
    I->n_txt_bins = codec_read_i32_kv(gf, "codec.lm.lut_n_bins", 4000);
    I->insert_bos_before_voice = codec_read_bool_kv(gf, "codec.lm.insert_bos_before_voice", false) ? 1 : 0;
    I->lsd_decode_steps = codec_read_i32_kv(gf, "codec.lm.lsd_decode_steps", 1);
    I->frames_after_eos = codec_read_i32_kv(gf, "codec.lm.frames_after_eos", -1);
    I->temperature  = codec_read_f32_kv(gf, "codec.lm.temperature", 0.7f);
    I->eos_threshold= codec_read_f32_kv(gf, "codec.lm.eos_threshold", -4.0f);
    I->flow_ln_eps  = 1e-6f;
    I->flow_rms_eps = 1e-5f;

    // Load SentencePiece tokenizer from the base64 KV.
    const std::string b64 = codec_lm_read_string_kv(lm->codec, "codec.lm.tokenizer.spm_b64");
    if (!b64.empty()) {
        // base64 decode.
        static const auto dec = [](char c) -> int {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62; if (c == '/') return 63; return -1; };
        std::vector<uint8_t> raw; raw.reserve(b64.size() * 3 / 4);
        int val = 0, bits = -8;
        for (char c : b64) {
            if (c == '=') break;
            int d = dec(c); if (d < 0) continue;
            val = (val << 6) | d; bits += 6;
            if (bits >= 0) { raw.push_back((uint8_t) ((val >> bits) & 0xff)); bits -= 8; }
        }
        if (M->spm.load(raw.data(), raw.size())) M->has_spm = true;
    }

    lm->impl = M;
    return true;
}

void free_lm(codec_lm * lm) {
    if (lm && lm->impl) { delete static_cast<flow_model *>(lm->impl); lm->impl = nullptr; }
}

bool state_init(codec_lm_state * st) {
    flow_state * s = new (std::nothrow) flow_state();
    if (!s) return false;
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
    st->impl = s;
    return true;
}
void state_free(codec_lm_state * st) {
    if (st && st->impl) {
        delete static_cast<flow_state *>(st->impl);
        st->impl = nullptr;
    }
}
void state_reset(codec_lm_state * st) {
    flow_state * s = static_cast<flow_state *>(st->impl);
    s->frame = 0;
    s->next_embd.clear();
    s->teacher.clear();
    s->rng.seed((uint32_t) (st->ctx && st->ctx->params.seed >= 0 ? st->ctx->params.seed : 0));
}

// input_linear(bos_emb) (latent == nullptr) or input_linear(normalize(latent)).
bool run_inproj(codec_lm_state * st, const float * latent, float * out_embd) {
    const flow_impl & I = static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_inproj_build b = {}; b.imp = I; b.is_bos = latent == nullptr ? 1 : 0; b.model = st->lm->codec;
    codec_graph_eval_guard guard(st->ctx, false);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_INPROJ; key.hop = b.is_bos;
    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    const bool ok = codec_graph_cache_get_or_build(st->ctx, key, build_inproj, &b, sizeof(b), &entry, &err) &&
        codec_graph_prepare_io(st->ctx, entry, &err) &&
        (latent == nullptr || codec_runtime_write_tensor(codec_graph_get_tensor(st->ctx, entry, "flow.ip.latent"),
                                                         latent, (size_t) I.ldim * sizeof(float), &err)) &&
        codec_graph_compute(st->ctx, entry, nth, &err) &&
        codec_runtime_read_tensor(codec_graph_get_tensor(st->ctx, entry, "flow.ip.embd"),
                                  out_embd, (size_t) I.d_model * sizeof(float), &err);
    if (!ok) st->last_error = err;
    return ok;
}

// One AR frame: h [d_model] -> out_latent [ldim] (denormalized, what the codec
// decodes) + out_stop (raw EOS head decision; the frames-after-EOS tail is the
// host's policy).  `noise` [ldim] ~ N(0, temperature), or NULL to sample here.
enum codec_status step_generate(codec_lm_state * st, const float * h_in, float /*cfg_value*/,
        int32_t n_timesteps, const float * noise, float * out_latent, int32_t * out_stop) {
    const flow_impl & I = static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_state * s = static_cast<flow_state *>(st->impl);
    const int32_t n_steps = n_timesteps > 0 ? n_timesteps : (I.lsd_decode_steps > 0 ? I.lsd_decode_steps : 1);

    std::vector<float> zbuf;
    const float * z = noise;
    if (z == nullptr) {
        zbuf.resize((size_t) I.ldim);
        std::normal_distribution<float> nd(0.0f, std::sqrt(I.temperature));
        for (int32_t d = 0; d < I.ldim; ++d) zbuf[(size_t) d] = nd(s->rng);
        z = zbuf.data();
    }

    flow_step_build b = {}; b.imp = I; b.n_steps = n_steps; b.model = st->lm->codec;
    codec_graph_eval_guard guard(st->ctx, /*persist=*/true);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_STEP; key.n_q = n_steps;
    if (!codec_graph_cache_get_or_build(st->ctx, key, build_step, &b, sizeof(b), &entry, &err)) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    auto G = [&](const char * nm) { return codec_graph_get_tensor(st->ctx, entry, nm); };
    const int32_t nth = st->lm->codec->n_threads > 0 ? st->lm->codec->n_threads : 1;
    float eos = 0.0f;
    s->next_embd.resize((size_t) I.d_model);
    const bool ok = codec_graph_prepare_io(st->ctx, entry, &err) &&
        codec_runtime_write_tensor(G("flow.st.h"), h_in, (size_t) I.d_model * sizeof(float), &err) &&
        codec_runtime_write_tensor(G("flow.st.noise"), z, (size_t) I.ldim * sizeof(float), &err) &&
        codec_graph_compute(st->ctx, entry, nth, &err) &&
        codec_runtime_read_tensor(G("flow.st.latent"), out_latent, (size_t) I.ldim * sizeof(float), &err) &&
        codec_runtime_read_tensor(G("flow.st.eos"), &eos, sizeof(float), &err) &&
        codec_runtime_read_tensor(G("flow.st.embd"), s->next_embd.data(), (size_t) I.d_model * sizeof(float), &err);
    if (!ok) { st->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    if (!s->teacher.empty()) {   // teacher-forced: the reference latent feeds the next step
        const std::vector<float> t = std::move(s->teacher);
        s->teacher.clear();
        if (!run_inproj(st, t.data(), s->next_embd.data())) return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (out_stop) *out_stop = eos > I.eos_threshold ? 1 : 0;
    s->frame += 1;
    return CODEC_STATUS_SUCCESS;
}

// Input embedding for the next backbone step: input_linear(bos_emb) before the
// first frame, then input_linear(last latent).
enum codec_status step_feedback_embd(codec_lm_state * st, float * out_embd) {
    const flow_impl & I = static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_state * s = static_cast<flow_state *>(st->impl);
    if (s->frame == 0) {
        return run_inproj(st, nullptr, out_embd) ? CODEC_STATUS_SUCCESS : CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(out_embd, s->next_embd.data(), (size_t) I.d_model * sizeof(float));
    return CODEC_STATUS_SUCCESS;
}

enum codec_status set_teacher_patch(codec_lm_state * st, const float * patch, int32_t n) {
    const flow_impl & I = static_cast<flow_model *>(st->lm->impl)->cfg;
    flow_state * s = static_cast<flow_state *>(st->impl);
    if (patch == nullptr) { s->teacher.clear(); return CODEC_STATUS_SUCCESS; }
    if (n != I.ldim) { st->last_error = "teacher latent must be [ldim]"; return CODEC_STATUS_INVALID_ARG; }
    s->teacher.assign(patch, patch + n);
    return CODEC_STATUS_SUCCESS;
}

}  // namespace

// =====================================================================
// Public flow_lm entry points (declared in codec_lm.h).
// =====================================================================

// Cache the info struct in the impl so we can return a stable pointer.
static codec_lm_flow_info g_flow_info;   // single-model process assumption is fine here

const struct codec_lm_flow_info * codec_lm_flow_get_info(struct codec_lm * lm) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || lm->impl == nullptr) return nullptr;
    flow_model * M = static_cast<flow_model *>(lm->impl);
    flow_impl * I = &M->cfg;
    g_flow_info.d_model          = I->d_model;
    g_flow_info.ldim             = I->ldim;
    g_flow_info.n_txt_bins       = I->n_txt_bins;
    g_flow_info.insert_bos_before_voice = I->insert_bos_before_voice;
    g_flow_info.frames_after_eos = I->frames_after_eos;
    g_flow_info.temperature      = I->temperature;
    g_flow_info.eos_threshold    = I->eos_threshold;
    g_flow_info.lsd_decode_steps = I->lsd_decode_steps;
    g_flow_info.has_tokenizer    = M->has_spm ? 1 : 0;
    return &g_flow_info;
}

enum codec_status codec_lm_flow_tokenize(struct codec_lm * lm, const char * text,
        int32_t * out_ids, int32_t cap, int32_t * n_out) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || text == nullptr ||
        out_ids == nullptr || n_out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_model * M = static_cast<flow_model *>(lm->impl);
    if (!M->has_spm) { lm->last_error = "no SentencePiece tokenizer baked in"; return CODEC_STATUS_NOT_SUPPORTED; }
    std::vector<int32_t> ids = M->spm.encode(text);
    if ((int32_t) ids.size() > cap) { lm->last_error = "token buffer too small"; return CODEC_STATUS_INVALID_ARG; }
    std::memcpy(out_ids, ids.data(), ids.size() * sizeof(int32_t));
    *n_out = (int32_t) ids.size();
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_lm_flow_speaker_rows(struct codec_lm * lm, const float * mu,
        int32_t n_voice, float * out, int32_t out_cap_rows, int32_t * n_out) {
    if (lm == nullptr || lm->kind != CODEC_LM_KIND_FLOW_LM || mu == nullptr || out == nullptr) {
        return CODEC_STATUS_INVALID_ARG;
    }
    flow_impl * I = &static_cast<flow_model *>(lm->impl)->cfg;
    if (ggml_get_tensor(lm->codec->weights, "lm.speaker_proj.w") == nullptr) {
        lm->last_error = "model has no speaker_proj (no voice cloning)"; return CODEC_STATUS_NOT_SUPPORTED;
    }
    const int32_t n_bos = I->insert_bos_before_voice ? 1 : 0;
    if (n_voice <= 0 || out_cap_rows < n_bos + n_voice) return CODEC_STATUS_INVALID_ARG;
    if (n_bos) {
        ggml_tensor * bos = ggml_get_tensor(lm->codec->weights, "lm.bos_before_voice");
        std::vector<float> row;
        if (bos == nullptr || !codec_tensor_as_vec_f32(bos, &row) || (int32_t) row.size() != I->d_model) {
            lm->last_error = "missing lm.bos_before_voice"; return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::memcpy(out, row.data(), row.size() * sizeof(float));
        out += I->d_model;
    }
    // Uses a codec_context borrowed from the model for a one-off graph compute.
    // We build a transient context (same pattern as codec_lm_state's ctx).
    codec_context * ctx = new (std::nothrow) codec_context();
    if (!ctx) return CODEC_STATUS_INTERNAL_ERROR;
    ctx->model = lm->codec; ctx->backend = lm->codec->backend; ctx->params = codec_context_default_params();
    std::string err;
    enum codec_status rc = CODEC_STATUS_SUCCESS;
    if (!codec_runtime_init(ctx, &err)) { delete ctx; lm->last_error = err; return CODEC_STATUS_INTERNAL_ERROR; }
    {
        flow_spk_build b = {}; b.imp = *I; b.T = n_voice; b.model = lm->codec;
        codec_graph_eval_guard guard(ctx, false);
        codec_graph_cache_entry * entry = nullptr;
        codec_graph_cache_key key = {}; key.kind = CODEC_GRAPH_FLOW_LM_SPKPROJ; key.n_frames = n_voice;
        if (!codec_graph_cache_get_or_build(ctx, key, build_spkproj, &b, sizeof(b), &entry, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        if (!codec_graph_prepare_io(ctx, entry, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        if (!codec_runtime_write_tensor(codec_graph_get_tensor(ctx, entry, "flow.sp.mu"), mu, (size_t) I->ldim * n_voice * sizeof(float), &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        {
            const int32_t nth = lm->codec->n_threads > 0 ? lm->codec->n_threads : 1;
            if (!codec_graph_compute(ctx, entry, nth, &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
        }
        if (!codec_runtime_read_tensor(codec_graph_get_tensor(ctx, entry, "flow.sp.rows"), out, (size_t) n_voice * I->d_model * sizeof(float), &err)) { rc = CODEC_STATUS_INTERNAL_ERROR; goto done; }
    }
    if (n_out) *n_out = n_bos + n_voice;
done:
    if (rc != CODEC_STATUS_SUCCESS) lm->last_error = err;
    codec_runtime_free(ctx); delete ctx;
    return rc;
}

// ---------------------------------------------------------------------

const codec_lm_kind_vtable codec_lm_vtable_flow_lm = {
    /*.kind               =*/ CODEC_LM_KIND_FLOW_LM,
    /*.name               =*/ "flow_lm",
    /*.init               =*/ init,
    /*.free               =*/ free_lm,
    /*.state_init         =*/ state_init,
    /*.state_free         =*/ state_free,
    /*.state_reset        =*/ state_reset,
    /*.step_begin         =*/ nullptr,
    /*.step_pending       =*/ nullptr,
    /*.step_logits        =*/ nullptr,
    /*.step_push_code     =*/ nullptr,
    /*.step_finish        =*/ nullptr,
    /*.audio_embd         =*/ nullptr,
    /*.compose_audio_embd =*/ nullptr,
    /*.compose_next_embd  =*/ nullptr,
    /*.speaker_encode     =*/ nullptr,
    /*.step_generate      =*/ step_generate,
    /*.step_feedback_embd =*/ step_feedback_embd,
    /*.text_prefill       =*/ nullptr,
    /*.set_min_len        =*/ nullptr,
    /*.set_teacher_patch  =*/ set_teacher_patch,
};
