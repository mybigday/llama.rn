// XY-Tokenizer (OpenMOSS-Team/XY_Tokenizer_TTSD_V0_hf) runtime.
//
// 16 kHz mono PCM in → 24 kHz mono PCM out, via:
//   mel-fbank (CPU, Whisper-style)
//   → parallel semantic + acoustic OmniAudioEncoder (Whisper-style transformer)
//   → semantic_encoder_adapter (4-layer transformer)
//   → concat(semantic_adapter_out, acoustic_out) along channel
//   → pre_rvq_adapter (proj 1536→768 + 4-layer transformer)
//   → ResidualDownConv (avg_pool 4×, gate/up Conv1d + Linear + LN)
//   → quantizer.input_proj (1×1 conv 3072→512)  → 8-level cosine-NN RVQ
//   → quantizer.output_proj (1×1 conv 512→3072) → post_rvq_adapter (3072→768→…→3072)
//   → UpConv (deconv stride=4)                  → OmniAudioDecoder (deconv 768→80 mel)
//   → Vocos backbone (embed conv 80→512 + 30 ConvNeXt blocks + final LN)
//   → iSTFT head (Linear 512→962, n_fft=960, hop=240) → CPU iSTFT → 24 kHz PCM
//
// Both encode and decode are single ggml graphs (one cached graph per public
// call).  The mel-fbank is computed CPU-side because it is a one-shot
// preprocessing pass that doesn't fit cleanly into a forward-only graph
// (per-frame DC removal + power-spec + log10 normalisation).  The iSTFT after
// the Vocos head is CPU-side via `codec_runtime_istft_from_head` to mirror
// what wavtokenizer/snac already do.

#include "xy_tokenizer.h"

#include "../codec_internal.h"
#include "../runtime/audio_dsp.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/lm_attn.h"
#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// (Whisper-style transformer layer + sliced-pos-emb-add helpers live in
// `src/ops/lm_ggml_ops.{cpp,h}` as `codec_op_whisper_encoder_layer_tc` and
// `codec_op_add_sliced_pos_emb_tc`.)

// (Whisper-style mel-fbank lives in `src/runtime/audio_dsp.{cpp,h}` as
// `codec_runtime_whisper_mel_features` — it bundles Slaney triangular
// filterbank + reflection-pad center-True STFT + log10/normalise.)

// =====================================================================
// Init from GGUF
// =====================================================================

enum codec_status codec_xy_tokenizer_init(struct codec_model * model) {
    if (model == nullptr) return CODEC_STATUS_INVALID_ARG;
    if (model->impl == nullptr) {
        codec_context_set_error(nullptr, "xy_tokenizer init: missing impl");
        return CODEC_STATUS_INVALID_STATE;
    }

    codec_xy_tokenizer & xy = *static_cast<codec_xy_tokenizer *>(model->impl);
    auto * gf = model->gguf;

    xy.encode_sample_rate = codec_read_i32_kv(gf, "codec.encode_sample_rate", 16000);
    xy.sample_rate        = codec_read_i32_kv(gf, "codec.sample_rate",        24000);
    xy.encoder_downsample_rate = codec_read_i32_kv(gf, "xy.encoder_downsample_rate", 1280);
    xy.decoder_upsample_rate   = codec_read_i32_kv(gf, "xy.decoder_upsample_rate",   1920);
    xy.latent_dim         = codec_read_i32_kv(gf, "codec.latent_dim",         3072);
    xy.codebook_dim       = codec_read_i32_kv(gf, "codec.codebook_dim",       512);
    xy.codebook_size      = codec_read_i32_kv(gf, "codec.codebook_size",      1024);
    xy.n_q                = codec_read_i32_kv(gf, "codec.n_q",                8);
    xy.rvq_dim            = codec_read_i32_kv(gf, "xy.rvq_dim",               512);

    xy.mel_n_mels         = codec_read_i32_kv(gf, "xy.mel.n_mels", 80);
    xy.mel_n_fft          = codec_read_i32_kv(gf, "xy.mel.n_fft",  400);
    xy.mel_hop_length     = codec_read_i32_kv(gf, "xy.mel.hop_length", 160);
    xy.mel_chunk_length_s = codec_read_i32_kv(gf, "xy.mel.chunk_length_seconds", 30);

    xy.sem_enc_n_layers          = codec_read_i32_kv(gf, "xy.sem_enc.n_layers", 12);
    xy.sem_enc_adapter_n_layers  = codec_read_i32_kv(gf, "xy.sem_enc_adapter.n_layers", 4);
    xy.pre_rvq_adapter_n_layers  = codec_read_i32_kv(gf, "xy.pre_rvq_adapter.n_layers", 4);
    xy.post_rvq_adapter_n_layers = codec_read_i32_kv(gf, "xy.post_rvq_adapter.n_layers", 4);

    xy.downsample_avg_pooler  = codec_read_i32_kv(gf, "xy.downsample.avg_pooler", 4);
    xy.upsample_stride        = codec_read_i32_kv(gf, "xy.upsample.stride", 4);

    xy.vocos_n_blocks  = codec_read_i32_kv(gf, "xy.vocos.n_blocks", 30);
    xy.vocos_n_fft     = codec_read_i32_kv(gf, "xy.vocos.head.n_fft", 960);
    xy.vocos_hop       = codec_read_i32_kv(gf, "xy.vocos.head.hop_size", 240);
    xy.vocos_head_out_dim = xy.vocos_n_fft + 2;

    // Infer per-encoder d_model and n_heads from the q_proj weight shape.
    lm_ggml_tensor * sem_q = nullptr;
    for (struct lm_ggml_tensor * t = lm_ggml_get_first_tensor(model->weights);
         t != nullptr; t = lm_ggml_get_next_tensor(model->weights, t)) {
        if (std::string(lm_ggml_get_name(t)) == "xy.sem_enc.l0.attn.q.w") { sem_q = t; break; }
    }
    if (sem_q != nullptr) {
        // ggml ne[0] = in_dim = d_model, ne[1] = out_dim = d_model.  Vocos
        // d_model is loaded from the shipped sinusoid PE shape.
        xy.sem_enc_d_model = (int32_t) sem_q->ne[1];
    }
    // n_heads is encoded by config; use 12 as the documented default (Whisper-base).
    xy.sem_enc_n_heads = 12;
    xy.sem_enc_ffn_dim = xy.sem_enc_d_model * 4;

    model->sample_rate        = xy.sample_rate;
    model->encode_sample_rate = xy.encode_sample_rate;
    model->has_encoder        = xy.has_encoder;
    model->has_decoder        = xy.has_decoder;
    model->hop_size           = xy.decoder_upsample_rate;  // sample-domain hop
    model->n_q                = xy.n_q;
    model->codebook_size      = xy.codebook_size;
    model->latent_dim         = xy.latent_dim;
    model->n_fft              = xy.vocos_n_fft;
    model->n_mels             = xy.mel_n_mels;
    return CODEC_STATUS_SUCCESS;
}

// =====================================================================
// Helpers — Whisper-style transformer layer
// =====================================================================
namespace {

// Whisper attention: q_proj has bias, k_proj has NO bias, v_proj/out_proj have
// bias.  Standard non-causal SDPA.  Pre-LayerNorm, GELU MLP, post-LayerNorm.
//
// `x_tc` is `[t, c=hidden]`.  Returns `[t, c=hidden]`.  When `n_valid > 0`
// and `< t`, attention scores for keys at positions `>= n_valid` are masked
// to -inf and rows for queries `>= n_valid` are zeroed (mirrors HF's
// `valid_k`/`valid_q` SDPA bias path).
// Whisper-style encoder block: pos_emb add + N transformer layers + final LN.
// Inputs come in as `x_tc = [t, hidden]`; pos_emb is `[max_pos, hidden]`.
lm_ggml_tensor * xy_op_whisper_module_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model,
    const std::string & base,
    int32_t n_layers,
    int32_t head_dim,
    int32_t n_heads,
    int32_t n_valid) {

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };

    x_tc = codec_op_add_sliced_pos_emb_tc(ctx, x_tc, W(base + ".pos_emb"));

    for (int32_t li = 0; li < n_layers; ++li) {
        const std::string lp = base + ".l" + std::to_string(li);
        x_tc = codec_op_whisper_encoder_layer_tc(ctx, x_tc,
            W(lp + ".norm1.w"), W(lp + ".norm1.b"),
            W(lp + ".attn.q.w"), W(lp + ".attn.q.b"),
            W(lp + ".attn.k.w"),
            W(lp + ".attn.v.w"), W(lp + ".attn.v.b"),
            W(lp + ".attn.out.w"), W(lp + ".attn.out.b"),
            W(lp + ".norm2.w"), W(lp + ".norm2.b"),
            W(lp + ".mlp.fc1.w"), W(lp + ".mlp.fc1.b"),
            W(lp + ".mlp.fc2.w"), W(lp + ".mlp.fc2.b"),
            head_dim, n_heads, n_valid);
        if (x_tc == nullptr) return nullptr;
    }
    return codec_op_layer_norm_tc(ctx, x_tc, 1e-5f,
                                  W(base + ".layer_norm.w"),
                                  W(base + ".layer_norm.b"));
}

}  // namespace

// =====================================================================
// Encode graph
// =====================================================================
//
//   mel features (n_mels, n_frames)
//   ├── conv1 (n_mels → 768, k=3, p=1, GELU)
//   ├── conv2 (768 → 768, k=3, s=2, p=1, GELU) ──────────► encoded mel (T_mel = n_frames/2, 768)
//   ├── + sinusoidal pos_emb (rows 0..T_mel-1)
//   │
//   ├── (run twice — semantic & acoustic — sharing input)
//   │   ├── 12-layer Whisper transformer
//   │   └── final LayerNorm
//   │
//   ├── semantic_encoder_adapter: pos_emb + 4 layers + final LN (in/out 768)
//   │
//   ├── concat(semantic_adapter, acoustic_enc) along channel ─► (T_mel, 1536)
//   │
//   ├── pre_rvq_adapter: Linear 1536→768 + pos_emb + 4 layers + final LN ─► (T_mel, 768)
//   │
//   ├── ResidualDownConv (avg_pooler=4):
//   │     gate = Conv1d(768, 3072, k=4, s=4)(x)
//   │     up   = Conv1d(768, 3072, k=4, s=4)(x)
//   │     fold = reshape(x, (T_mel/4, 4*768=3072))
//   │     out  = LN(down_proj(silu(gate) * up) + fold)            ─► (T_codes, 3072)
//   │
//   └── quantizer.input_proj (Conv1d 1×1 3072→512), then
//       8 Euclidean-NN VQ levels (codebook 1024×512), summing reconstructions.
//       Output: int32 codes shape (n_q, T_codes).

namespace {

// PyTorch ConvTranspose1d-style deconv with kernel k, stride s, no padding.
// We don't use lm_ggml_conv_transpose_1d directly here because XY's deconv uses
// kernel_size==stride==4 (UpConv) and kernel_size==stride_size==2 (deconv1)
// which cleanly decomposes into a per-step matmul: each input frame writes a
// `k`-long contiguous block to the output, so the output length is `T_in * k`
// (equivalent to lm_ggml_conv_transpose_1d(stride=k, p=0, d=1)).  That's exactly
// what `lm_ggml_conv_transpose_1d` returns when `s0 == kernel`.

// Run an OmniAudioEncoder body: input (mel, n_frames) → conv1+conv2 stride=2
// → pos_emb add → 12 transformer layers → final LN → output (T_mel, d_model).
lm_ggml_tensor * xy_omni_encoder_module_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * mel_ct,      // (n_mels, n_frames) — channels-first
    const codec_model * model,
    const std::string & base,
    int32_t n_layers,
    int32_t d_model,
    int32_t n_heads,
    int32_t n_valid) {

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };

    // PyTorch Conv1d expects (B, C_in, T).  ggml conv1d wants tc input (t, c).
    // mel_ct is already in TC layout (ne[0]=t inner, ne[1]=mel outer) —
    // matches PyTorch's (B=1, C=mel, T) row-major memory.  No transpose.
    lm_ggml_tensor * x_tc_in = mel_ct;
    lm_ggml_tensor * x = codec_conv1d(ctx, x_tc_in,
                                   W(base + ".conv1.w"), W(base + ".conv1.b"),
                                   /*stride=*/1, /*dilation=*/1, /*padding=*/1);
    if (x == nullptr) return nullptr;
// codec_conv1d returns ne=(t, c_out, 1) (im2col path keeps a batch dim).
    // Squeeze to 2D so pos_emb adds element-wise.
x = lm_ggml_gelu_erf(ctx, x);
    x = codec_conv1d(ctx, x,
                     W(base + ".conv2.w"), W(base + ".conv2.b"),
                     /*stride=*/2, /*dilation=*/1, /*padding=*/1);
    if (x == nullptr) return nullptr;
    x = lm_ggml_gelu_erf(ctx, x);                                   // (T_mel, d_model)
// The transformer module helper adds pos_emb internally — don't add again.
    x = xy_op_whisper_module_tc(ctx, x, model, base, n_layers,
                                d_model / n_heads, n_heads, n_valid);
    return x;
}

// Adapter Transformer: optional `proj` Linear in, pos_emb add, N layers, final
// LN, optional `out_proj` Linear out.  All in/out shapes are (t, c).
lm_ggml_tensor * xy_adapter_module_tc(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,
    const codec_model * model,
    const std::string & base,
    int32_t n_layers,
    int32_t d_model,
    int32_t n_heads,
    int32_t n_valid) {

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };

    lm_ggml_tensor * proj_w = W(base + ".proj.w");
    if (proj_w != nullptr) {
        x_tc = codec_op_linear_tc(ctx, x_tc, proj_w, W(base + ".proj.b"));
        if (x_tc == nullptr) return nullptr;
    }
    // pos_emb is added inside xy_op_whisper_module_tc; don't add again here.
    x_tc = xy_op_whisper_module_tc(ctx, x_tc, model, base, n_layers,
                                    d_model / n_heads, n_heads, n_valid);
    if (x_tc == nullptr) return nullptr;
    lm_ggml_tensor * out_w = W(base + ".out_proj.w");
    if (out_w != nullptr) {
        x_tc = codec_op_linear_tc(ctx, x_tc, out_w, W(base + ".out_proj.b"));
    }
    return x_tc;
}

// ResidualDownConv with avg_pooler=4.
//   gate = gate_proj(x) ∈ (T/4, intermediate)        [Conv1d k=4 s=4 no bias]
//   up   = up_proj(x)   ∈ (T/4, intermediate)
//   x_fold = x.reshape(T/4, intermediate)            [no learnable params]
//   y = down_proj(silu(gate) * up) + x_fold          [down_proj = Linear no bias]
//   y = LayerNorm(y)
lm_ggml_tensor * xy_residual_down_conv(
    lm_ggml_context * ctx,
    lm_ggml_tensor * x_tc,                        // (T, d_model=768)
    const codec_model * model,
    int32_t avg_pooler) {

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };

    const int64_t t  = x_tc->ne[0];
    const int64_t d  = x_tc->ne[1];
    const int64_t t_out = t / avg_pooler;
    const int64_t inter = d * avg_pooler;

    lm_ggml_tensor * gate_w = W("xy.downsample.gate.w");
lm_ggml_tensor * gate = codec_conv1d(ctx, x_tc, gate_w, nullptr,
                                      /*stride=*/avg_pooler, /*dilation=*/1, /*padding=*/0);
    lm_ggml_tensor * up   = codec_conv1d(ctx, x_tc, W("xy.downsample.up.w"),   nullptr,
                                      /*stride=*/avg_pooler, /*dilation=*/1, /*padding=*/0);
    if (gate == nullptr || up == nullptr) return nullptr;
    // gate / up have ggml ne=(T/4, intermediate) which is *channel-first* in
    // PyTorch terms (the inner stride is the time index).  Move to CT layout
    // (ne=(intermediate, T/4)) so the upcoming reshape of x_tc to
    // (intermediate, T/4) and the layernorm-along-channel both line up.
    lm_ggml_tensor * gate_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, gate));     // (intermediate, T/4)
    lm_ggml_tensor * up_ct   = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, up));

    // PyTorch fold: x.reshape(T/4, 4*D).  In ggml the input x_tc has ne=(T, D)
    // which is PyTorch's channel-first (D, T) row-major; after a contiguous
    // transpose to (D, T) ggml-ne we get PyTorch's (T, D) row-major and a
    // straight reshape produces ne=(intermediate=4*D, T/4) which in PyTorch
    // terms is (T/4, intermediate) row-major.
    lm_ggml_tensor * x_pt = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, x_tc));        // ne=(D, T)
    lm_ggml_tensor * x_fold_ct = lm_ggml_reshape_2d(ctx, x_pt, inter, t_out);    // (intermediate, T/4)

    lm_ggml_tensor * gate_silu = lm_ggml_silu(ctx, gate_ct);
    lm_ggml_tensor * mul = lm_ggml_mul(ctx, gate_silu, up_ct);                   // (intermediate, T/4)

    // down_proj = nn.Linear(intermediate, intermediate, bias=False).  In
    // ggml ne=(in=intermediate, out=intermediate); codec_op_linear contracts
    // on ne[0].  Apply on the CT input directly.
    lm_ggml_tensor * down_ct = codec_op_linear(ctx, mul, W("xy.downsample.down.w"), nullptr);
    if (down_ct == nullptr) return nullptr;

    lm_ggml_tensor * sum_ct = lm_ggml_add(ctx, down_ct, x_fold_ct);
    lm_ggml_tensor * ln_ct = codec_op_layer_norm_ct(ctx, sum_ct, 1e-5f,
                                                  W("xy.downsample.layer_norm.w"),
                                                  W("xy.downsample.layer_norm.b"));
    if (ln_ct == nullptr) return nullptr;
    return lm_ggml_cont(ctx, lm_ggml_transpose(ctx, ln_ct));                     // back to (T/4, intermediate) TC
}

// One residual VQ level: argmin_i ||z[t] - codebook[i]||^2.
// Returns the codebook indices (i32, length t) and the residual (z - z_q).
struct xy_vq_step {
    lm_ggml_tensor * indices;   // i32 (t,)
    lm_ggml_tensor * residual;  // f32 (t, codebook_dim)
};
xy_vq_step xy_rvq_one_level(
    lm_ggml_context * ctx,
    lm_ggml_tensor * z_tc,                 // (t, codebook_dim)
    const codec_model * model,
    int32_t qi) {
    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };
    const std::string base = "xy.q." + std::to_string(qi);

    lm_ggml_tensor * cb     = W(base + ".codebook");          // ggml ne=(d, V)
    lm_ggml_tensor * cb_sq  = W(base + ".codebook_sq_norm");  // ggml ne=(V,)

    // Compute scores(t, V) = 2 * (z @ cb.T) - sq_norm.
    // mul_mat contracts on ne[0]: z_tc has ne=(d, t)  (after transpose), cb is
    // ne=(d, V) — output (V, t).  Then we add bias (-sq_norm) along V.
    lm_ggml_tensor * z_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, z_tc));   // (d, t)
    lm_ggml_tensor * dots = lm_ggml_mul_mat(ctx, cb, z_ct);                 // (V, t)
    dots = lm_ggml_scale(ctx, dots, 2.0f);
    // Subtract sq_norm[V] from every column of (V, t).  Broadcasting: reshape
    // sq_norm as (V, 1) then lm_ggml_repeat to (V, t).
    lm_ggml_tensor * sq2 = lm_ggml_reshape_2d(ctx, cb_sq, cb_sq->ne[0], 1);
    lm_ggml_tensor * sq_b = lm_ggml_repeat(ctx, sq2, dots);
    dots = lm_ggml_sub(ctx, dots, sq_b);

    lm_ggml_tensor * idx = lm_ggml_argmax(ctx, dots);   // (t,) i32

    // Reconstruct z_q[t] = codebook[idx[t]].
    // lm_ggml_get_rows expects rows of a 2D tensor along ne[1]; codebook ne=(d,V)
    // fits — get_rows(cb, idx) → ne=(d, t).
    lm_ggml_tensor * z_q_ct = lm_ggml_get_rows(ctx, cb, idx);
    lm_ggml_tensor * z_q_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, z_q_ct));  // (t, d)

    lm_ggml_tensor * residual = lm_ggml_sub(ctx, z_tc, z_q_tc);
    xy_vq_step out{idx, residual};
    return out;
}

// Sum 8 RVQ reconstructions (codebook lookup, no projections) for the decode
// path.  Codes are int32 ggml ne=(n_codes, n_q), so each "row" along ne[1]
// is one quantiser's per-time indices and we can pull it via a strided
// view_1d.
lm_ggml_tensor * xy_rvq_decode_sum(
    lm_ggml_context * ctx,
    lm_ggml_tensor * codes_tq,                  // i32 ne=(n_codes, n_q)
    const codec_model * model,
    int32_t n_q) {

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, model, nm);
    };

    lm_ggml_tensor * acc_tc = nullptr;
    const int64_t n_codes = codes_tq->ne[0];
    for (int32_t qi = 0; qi < n_q; ++qi) {
        const std::string base = "xy.q." + std::to_string(qi);
        lm_ggml_tensor * cb = W(base + ".codebook");
        lm_ggml_tensor * idx = lm_ggml_view_1d(ctx, codes_tq, n_codes,
                                         qi * codes_tq->nb[1]);
        lm_ggml_tensor * z_ct = lm_ggml_get_rows(ctx, cb, idx);                // (d, t)
        lm_ggml_tensor * z_tc = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, z_ct));  // (t, d)
        acc_tc = (acc_tc == nullptr) ? z_tc : lm_ggml_add(ctx, acc_tc, z_tc);
    }
    return acc_tc;
}

}  // namespace

// -----------------------------------------------------------------
// Encode graph builder
// -----------------------------------------------------------------
struct xy_encode_build {
    int32_t n_mel_frames = 0;          // number of mel frames after Whisper feature extractor
    int32_t n_mel_valid  = 0;          // valid (non-padded) mel frames; rest are masked
    const codec_xy_tokenizer * cfg = nullptr;
    const codec_model * model = nullptr;
};

static const char * xy_name_mel()    { return "xy.encode.mel_in"; }
static const char * xy_name_codes()  { return "xy.encode.codes"; }

static bool xy_build_encode(lm_ggml_context * ctx, void * user_data, lm_ggml_tensor ** out) {
    auto * p = static_cast<xy_encode_build *>(user_data);
    if (ctx == nullptr || p == nullptr || out == nullptr) return false;
    const codec_xy_tokenizer & cfg = *p->cfg;

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, p->model, nm);
    };

    // Mel input: ggml ne=(n_frames, n_mels) so the inner stride is time
    // (matching PyTorch's (B=1, C=80, T) row-major memory layout that the
    // HF feature extractor produces).  This is also already the TC layout
    // that codec_conv1d expects.
    lm_ggml_tensor * t_mel = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32,
                                             p->n_mel_frames, cfg.mel_n_mels);
    lm_ggml_set_name(t_mel, xy_name_mel());

    // The HF encoder pipeline runs over a chunk-padded mel and masks
    // attention to `mel_valid` valid frames; the valid count drops by
    // each downsample stride.
    const int32_t n_valid_mel  = (p->n_mel_valid > 0) ? p->n_mel_valid : p->n_mel_frames;
    const int32_t n_valid_conv = n_valid_mel / 2;          // OmniAudioEncoder.conv2 stride=2

    // Two parallel encoders, then semantic_encoder_adapter, then concat.
    lm_ggml_tensor * sem = xy_omni_encoder_module_tc(
        ctx, t_mel, p->model, "xy.sem_enc",
        cfg.sem_enc_n_layers, cfg.sem_enc_d_model, cfg.sem_enc_n_heads,
        /*n_valid=*/n_valid_conv);
    lm_ggml_tensor * acoust = xy_omni_encoder_module_tc(
        ctx, t_mel, p->model, "xy.acoust_enc",
        cfg.sem_enc_n_layers, cfg.sem_enc_d_model, cfg.sem_enc_n_heads,
        /*n_valid=*/n_valid_conv);
    if (sem == nullptr || acoust == nullptr) return false;

    sem = xy_adapter_module_tc(ctx, sem, p->model, "xy.sem_enc_adapter",
                               cfg.sem_enc_adapter_n_layers,
                               cfg.sem_enc_d_model, cfg.sem_enc_n_heads,
                               /*n_valid=*/n_valid_conv);
    if (sem == nullptr) return false;

    // Concat along channel: (t, 2*d_model).  lm_ggml_concat along ne[1] (since
    // both have ne=(t, d) and we want ne=(t, 2d) which is ne[1] axis).
    lm_ggml_tensor * cat = lm_ggml_concat(ctx, sem, acoust, /*dim=*/1);
    cat = xy_adapter_module_tc(ctx, cat, p->model, "xy.pre_rvq_adapter",
                               cfg.pre_rvq_adapter_n_layers,
                               cfg.sem_enc_d_model, cfg.sem_enc_n_heads,
                               /*n_valid=*/n_valid_conv);
    if (cat == nullptr) return false;
    lm_ggml_tensor * down = xy_residual_down_conv(ctx, cat, p->model, cfg.downsample_avg_pooler);
    if (down == nullptr) return false;
    // input_proj is a 1×1 conv (3072→512).  Use codec_conv1d with k=1
    // (pointwise path: returns 2D).
    lm_ggml_tensor * z = codec_conv1d(ctx, down,
                                   W("xy.q.in_proj.w"),
                                   W("xy.q.in_proj.b"),
                                   /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (z == nullptr) return false;

    // 8-level RVQ.  Build per-level argmax + accumulate residuals.
    const int32_t n_q = cfg.n_q;
    std::vector<lm_ggml_tensor *> codes_per_level(n_q, nullptr);
    lm_ggml_tensor * residual = z;
    for (int32_t qi = 0; qi < n_q; ++qi) {
        xy_vq_step step = xy_rvq_one_level(ctx, residual, p->model, qi);
        if (step.indices == nullptr) return false;
        codes_per_level[qi] = step.indices;
        residual = step.residual;
    }

    // Pack into (t, n_q) interleaved.  Each codes_per_level[qi] has ne=(t,) i32.
    // Reshape each to (1, t) and concat along ne[0] → (n_q, t).
    const int64_t t_codes = down->ne[0];
    std::vector<lm_ggml_tensor *> rows;
    rows.reserve((size_t) n_q);
    for (int32_t qi = 0; qi < n_q; ++qi) {
        lm_ggml_tensor * r = lm_ggml_reshape_2d(ctx, codes_per_level[qi], 1, t_codes);
        rows.push_back(r);
    }
    lm_ggml_tensor * codes_packed = rows[0];
    for (int32_t qi = 1; qi < n_q; ++qi) {
        codes_packed = lm_ggml_concat(ctx, codes_packed, rows[(size_t) qi], /*dim=*/0);
    }
    // codes_packed has ne=(n_q, t_codes) i32.  Match encoder convention used by
    // the public API: `codec_token_buffer.data[t * n_q + q]`.  We transpose to
    // (t, n_q) at marshalling time on CPU side.
    lm_ggml_set_name(codes_packed, xy_name_codes());
    *out = codes_packed;
    return true;
}

// -----------------------------------------------------------------
// Decode graph builder
// -----------------------------------------------------------------
struct xy_decode_build {
    int32_t n_codes = 0;
    const codec_xy_tokenizer * cfg = nullptr;
    const codec_model * model = nullptr;
};

static const char * xy_name_dec_codes() { return "xy.decode.codes"; }
static const char * xy_name_dec_head()  { return "xy.decode.head_out"; }

static bool xy_build_decode(lm_ggml_context * ctx, void * user_data, lm_ggml_tensor ** out) {
    auto * p = static_cast<xy_decode_build *>(user_data);
    if (ctx == nullptr || p == nullptr || out == nullptr) return false;
    const codec_xy_tokenizer & cfg = *p->cfg;

    auto W = [&](const std::string & nm) -> lm_ggml_tensor * {
        return codec_graph_weight(ctx, p->model, nm);
    };

    // Codes input: ggml ne=(n_codes, n_q) so each "row" along ne[1] is one
    // quantiser's per-time indices, viewable via a contiguous view_1d.
    lm_ggml_tensor * t_codes = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_I32, p->n_codes, cfg.n_q);
    lm_ggml_set_name(t_codes, xy_name_dec_codes());

    // Sum codebook lookups across the 8 levels.
    lm_ggml_tensor * z = xy_rvq_decode_sum(ctx, t_codes, p->model, cfg.n_q);  // (t, codebook_dim=512)
// output_proj 1×1 conv (512 → 3072).
    lm_ggml_tensor * x = codec_conv1d(ctx, z,
                                   W("xy.q.out_proj.w"),
                                   W("xy.q.out_proj.b"),
                                   /*stride=*/1, /*dilation=*/1, /*padding=*/0);
    if (x == nullptr) return false;
// post_rvq_adapter: Linear 3072→768 + pos_emb + 4 layers + final LN +
    // Linear 768→3072.
    x = xy_adapter_module_tc(ctx, x, p->model, "xy.post_rvq_adapter",
                              cfg.post_rvq_adapter_n_layers,
                              cfg.sem_enc_d_model, cfg.sem_enc_n_heads,
                              /*n_valid=*/0);
    if (x == nullptr) return false;
// UpConv: ConvTranspose1d 3072→768, k=4, s=4, no bias.  PyTorch weight
    // shape (in=3072, out=768, k=4) lands in ggml as ne=(k, out, in).
    // lm_ggml_conv_transpose_1d expects input ne=(t, in_c, batch) — exactly what
    // x already has (ne=(t, 3072)) — so no transpose.  Output ne=(t*s0, out=768, 1).
    {
        lm_ggml_tensor * up = lm_ggml_conv_transpose_1d(
            ctx, W("xy.upsample.up_conv.w"), x,
            /*s0=*/cfg.upsample_stride, /*p0=*/0, /*d0=*/1);
        if (up == nullptr) return false;
        x = up;   // (t*4, 768) tc
}

    // OmniAudioDecoder: 12 transformer layers + final LN, then deconv1+deconv2.
    // pos_emb is added inside xy_op_whisper_module_tc.
    x = xy_op_whisper_module_tc(ctx, x, p->model, "xy.acoust_dec",
                                 cfg.sem_enc_n_layers,
                                 cfg.sem_enc_d_model / cfg.sem_enc_n_heads,
                                 cfg.sem_enc_n_heads,
                                 /*n_valid=*/0);
    if (x == nullptr) return false;
{
        // deconv1: 768→768, k=3, stride=2, padding=0, output_padding=0.
        // x already has ggml ne=(t, 768) which is what conv_transpose_1d
        // expects ((t, in_c, batch=1)).
        lm_ggml_tensor * d1 = lm_ggml_conv_transpose_1d(
            ctx, W("xy.acoust_dec.deconv1.w"), x,
            /*s0=*/2, /*p0=*/0, /*d0=*/1);
        if (d1 == nullptr) return false;
        // lm_ggml_conv_transpose_1d doesn't add bias automatically.
        lm_ggml_tensor * d1_b = W("xy.acoust_dec.deconv1.b");
        if (d1_b != nullptr) {
            lm_ggml_tensor * d1_b_2d = lm_ggml_reshape_2d(ctx, d1_b, 1, d1_b->ne[0]);
            lm_ggml_tensor * d1_b_rep = lm_ggml_repeat(ctx, d1_b_2d, d1);
            d1 = lm_ggml_add(ctx, d1, d1_b_rep);
        }
        d1 = lm_ggml_gelu_erf(ctx, d1);
        lm_ggml_tensor * d2 = lm_ggml_conv_transpose_1d(
            ctx, W("xy.acoust_dec.deconv2.w"), d1,
            /*s0=*/1, /*p0=*/0, /*d0=*/1);
        if (d2 == nullptr) return false;
        lm_ggml_tensor * d2_b = W("xy.acoust_dec.deconv2.b");
        if (d2_b != nullptr) {
            lm_ggml_tensor * d2_b_2d = lm_ggml_reshape_2d(ctx, d2_b, 1, d2_b->ne[0]);
            lm_ggml_tensor * d2_b_rep = lm_ggml_repeat(ctx, d2_b_2d, d2);
            d2 = lm_ggml_add(ctx, d2, d2_b_rep);
        }
        d2 = lm_ggml_gelu_erf(ctx, d2);
        // d2 has ggml ne=(t_audio, 80) — already TC.
        x = d2;
}

    // Vocos backbone:
    //   1. embed: Conv1d 80→512, k=7, padding=3 (length-preserving).
    //   2. initial LayerNorm (over channel dim).
    //   3. 30 ConvNeXt blocks (depthwise k=7 pad=3, GELU, channel-scale gamma).
    //   4. final LayerNorm.
    //   5. head: Linear 512→962 (mag-and-phase head, n_fft+2 = 962).
    {
        lm_ggml_tensor * embed = codec_conv1d(ctx, x,
                                           W("xy.vocos.embed.w"),
                                           W("xy.vocos.embed.b"),
                                           /*stride=*/1, /*dilation=*/1, /*padding=*/3);
        if (embed == nullptr) return false;
// Initial LayerNorm in CT layout to match ConvNeXt block convention.
        lm_ggml_tensor * h_ct = lm_ggml_cont(ctx, lm_ggml_transpose(ctx, embed));
        h_ct = codec_op_layer_norm_ct(ctx, h_ct, 1e-6f,
                                      W("xy.vocos.norm.w"),
                                      W("xy.vocos.norm.b"));
for (int32_t bi = 0; bi < cfg.vocos_n_blocks; ++bi) {
            const std::string bp = "xy.vocos.b" + std::to_string(bi);
            h_ct = codec_op_convnext_block_ct(
                ctx, h_ct,
                W(bp + ".dwconv.w"), W(bp + ".dwconv.b"),
                W(bp + ".norm.w"),   W(bp + ".norm.b"),
                W(bp + ".pwconv1.w"), W(bp + ".pwconv1.b"),
                W(bp + ".pwconv2.w"), W(bp + ".pwconv2.b"),
                W(bp + ".gamma"),
                /*dw_padding=*/3);
            if (h_ct == nullptr) return false;
        }
        h_ct = codec_op_layer_norm_ct(ctx, h_ct, 1e-6f,
                                      W("xy.vocos.final_layer_norm.w"),
                                      W("xy.vocos.final_layer_norm.b"));
        // Head: Linear 512→962 producing a (962, t_audio) tensor.  CPU iSTFT.
        lm_ggml_tensor * head_out = codec_op_linear(ctx, h_ct,
                                                 W("xy.vocos.head.out.w"),
                                                 W("xy.vocos.head.out.b"));
        if (head_out == nullptr) return false;
        lm_ggml_set_name(head_out, xy_name_dec_head());
        *out = head_out;
    }
    return true;
}

// =====================================================================
// CPU side — public encode / decode
// =====================================================================

static enum codec_status codec_xy_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * /*out_latent*/,
    struct codec_encode_params /*params*/) {
    if (ctx == nullptr || ctx->model == nullptr || ctx->model->impl == nullptr) {
        return CODEC_STATUS_INVALID_STATE;
    }
    codec_xy_tokenizer & cfg = *static_cast<codec_xy_tokenizer *>(ctx->model->impl);

    // Compute mel features on CPU (shared helper in audio_dsp).
    std::vector<float> mel;
    int32_t n_frames = 0;
    std::string mel_err;
    if (!codec_runtime_whisper_mel_features(
            pcm,
            cfg.encode_sample_rate,
            cfg.mel_n_fft,
            cfg.mel_hop_length,
            cfg.mel_n_mels,
            /*pad_to_samples=*/cfg.encoder_downsample_rate,
            &mel, &n_frames, &mel_err)) {
        codec_context_set_error(ctx, mel_err.empty() ? "XY-Tokenizer: mel-fbank failed" : mel_err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (n_frames <= 0) {
        codec_context_set_error(ctx, "XY-Tokenizer: empty mel features");
        return CODEC_STATUS_INVALID_ARG;
    }
    // Number of valid (non-padded) mel frames.  HF reports this as
    // `attention_mask.sum() == pcm_size_orig / hop`; we reproduce it from the
    // original PCM length here so the encoder masks attention correctly and
    // we can trim the output to HF's effective code count.
    const int32_t n_in_pcm    = (int32_t) pcm.size();
    const int32_t n_mel_valid = std::min(n_frames, n_in_pcm / cfg.mel_hop_length);

    xy_encode_build build = {};
    build.n_mel_frames = n_frames;
    build.n_mel_valid  = n_mel_valid;
    build.cfg = &cfg;
    build.model = ctx->model;
    // Number of valid output codes = floor(n_mel_valid / 2 / avg_pooler)
    // — matches HF's `output_lengths = input_lengths // self.avg_pooler`.
    const int32_t n_codes_valid = (n_mel_valid / 2) / std::max(1, cfg.downsample_avg_pooler);

    const int32_t avg = std::max(1, cfg.downsample_avg_pooler);
    const int32_t t_after_conv2 = n_frames / 2;          // mel conv stride 2
    const int32_t n_codes = t_after_conv2 / avg;
    if (n_codes <= 0) {
        codec_context_set_error(ctx, "XY-Tokenizer: input too short");
        return CODEC_STATUS_INVALID_ARG;
    }

    codec_graph_eval_guard guard(ctx);
    std::string err;
    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_XY_TOKENIZER_ENCODE, /*n_frames=*/n_codes,
              /*n_q=*/cfg.n_q, /*hop=*/cfg.encoder_downsample_rate,
              /*n_in=*/n_frames, /*latent_dim=*/cfg.latent_dim }, xy_build_encode, &build, sizeof(build), &entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_mel = codec_graph_get_tensor(ctx, entry, xy_name_mel());
    lm_ggml_tensor * t_codes = codec_graph_get_tensor(ctx, entry, xy_name_codes());
    if (t_mel == nullptr || t_codes == nullptr) {
        codec_context_set_error(ctx, "XY-Tokenizer encode graph invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_runtime_write_tensor(t_mel, mel.data(), mel.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_compute(ctx, entry, ctx->model->n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Read codes (n_q, n_codes) and re-pack into the (T, Q) layout.
    std::vector<int32_t> qt((size_t) cfg.n_q * (size_t) n_codes, 0);
    if (!codec_runtime_read_tensor(t_codes, qt.data(),
                                    qt.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Trim to the valid code count derived from the original PCM length —
    // matches HF's `output_lengths = input_lengths // self.avg_pooler` so
    // downstream consumers see exactly the same number of codes per second
    // as the HF reference.
    const int32_t n_codes_out = std::min(n_codes, n_codes_valid);
    int32_t * tokens = static_cast<int32_t *>(std::malloc((size_t) n_codes_out * (size_t) cfg.n_q * sizeof(int32_t)));
    if (tokens == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
    // codes tensor has ggml ne=(n_q, n_codes) so memory layout is
    // `data[q + t * n_q]`; transpose into the (T, Q) layout that
    // codec_token_buffer expects (`data[t * n_q + q]`).  The two layouts
    // happen to coincide here, so the loop is a straight copy of the
    // valid range.
    for (int32_t t = 0; t < n_codes_out; ++t) {
        for (int32_t q = 0; q < cfg.n_q; ++q) {
            tokens[(size_t) t * (size_t) cfg.n_q + (size_t) q] =
                qt[(size_t) q + (size_t) t * (size_t) cfg.n_q];
        }
    }
    codec_token_buffer_reset(out_tokens);
    out_tokens->data          = tokens;
    out_tokens->n_tokens      = (int32_t) ((size_t) n_codes_out * (size_t) cfg.n_q);
    out_tokens->n_frames      = n_codes_out;
    out_tokens->n_q           = cfg.n_q;
    out_tokens->codebook_size = cfg.codebook_size;
    out_tokens->sample_rate   = cfg.sample_rate;
    out_tokens->hop_size      = cfg.decoder_upsample_rate;
    return CODEC_STATUS_SUCCESS;
}

// Decode a single ≤chunk_code_length window of codes through the decode graph
// and return the raw iSTFT PCM for that window.  `chunk_codes_tq` is a (T, Q)
// int32 slice (codec_token_buffer layout) of length `n_codes` frames.
//
// The decode pipeline's transformer pos_emb tables are sized for a fixed
// maximum window (post_rvq_adapter.pos_emb has 375 rows == chunk_code_length),
// so a window longer than that slices past the table and asserts in
// lm_ggml_view_2d.  Chunking (mirroring HF `XYTokenizerModel.decode`) is therefore
// required, not merely an optimisation.
static enum codec_status codec_xy_decode_chunk(
    struct codec_context * ctx,
    const codec_xy_tokenizer & cfg,
    const int32_t * chunk_codes_tq,
    int32_t n_codes,
    std::vector<float> * out_pcm,
    std::string * err) {

    xy_decode_build build = {};
    build.n_codes = n_codes;
    build.cfg = &cfg;
    build.model = ctx->model;

    codec_graph_cache_entry * entry = nullptr;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_XY_TOKENIZER_DECODE, /*n_frames=*/n_codes,
              /*n_q=*/cfg.n_q, /*hop=*/cfg.decoder_upsample_rate,
              /*n_in=*/0, /*latent_dim=*/cfg.latent_dim }, xy_build_decode, &build, sizeof(build), &entry, err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    lm_ggml_tensor * t_codes = codec_graph_get_tensor(ctx, entry, xy_name_dec_codes());
    lm_ggml_tensor * t_head  = codec_graph_get_tensor(ctx, entry, xy_name_dec_head());
    if (t_codes == nullptr || t_head == nullptr) {
        if (err != nullptr) *err = "XY-Tokenizer decode graph invalid";
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_prepare_io(ctx, entry, err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Re-pack tokens (T, Q) → ggml-ne (n_codes, n_q) memory `data[t + q*n_codes]`.
    std::vector<int32_t> qt((size_t) cfg.n_q * (size_t) n_codes, 0);
    for (int32_t t = 0; t < n_codes; ++t) {
        for (int32_t q = 0; q < cfg.n_q; ++q) {
            qt[(size_t) q * (size_t) n_codes + (size_t) t] =
                chunk_codes_tq[(size_t) t * (size_t) cfg.n_q + (size_t) q];
        }
    }
    if (!codec_runtime_write_tensor(t_codes, qt.data(),
                                     qt.size() * sizeof(int32_t), err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    if (!codec_graph_compute(ctx, entry, ctx->model->n_threads, err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Head is laid out as ne=(out_dim=962, t_audio).  For each frame we have
    // the first `n_fft/2 + 1` channels = mag (post-exp) and the next
    // `n_fft/2 + 1` channels = phase.  `codec_runtime_istft_from_head`
    // expects the same `[out_dim, n_frames]` (column-major) layout we
    // produce here.
    const int32_t out_dim  = (int32_t) t_head->ne[0];
    const int32_t t_audio  = (int32_t) t_head->ne[1];
    std::vector<float> head((size_t) out_dim * (size_t) t_audio, 0.0f);
    if (!codec_runtime_read_tensor(t_head, head.data(),
                                    head.size() * sizeof(float), err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    // Read the iSTFT window so iSTFT matches exactly (Vocos uses
    // torch.hann_window which is *symmetric* — `/(N-1)` — matching the
    // default `codec_runtime_istft_from_head` window if window=nullptr).
    out_pcm->clear();
    if (!codec_runtime_istft_from_head(head, out_dim, t_audio,
                                        cfg.vocos_hop, /*window=*/nullptr,
                                        /*skip_dc_nyquist=*/false,
                                        /*trim_pad_override=*/-1,
                                        out_pcm, err)) {
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    return CODEC_STATUS_SUCCESS;
}

static enum codec_status codec_xy_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params /*params*/) {
    if (ctx == nullptr || tokens == nullptr || out_pcm == nullptr ||
        ctx->model == nullptr || ctx->model->impl == nullptr) {
        return CODEC_STATUS_INVALID_STATE;
    }
    codec_xy_tokenizer & cfg = *static_cast<codec_xy_tokenizer *>(ctx->model->impl);
    if (tokens->n_q != cfg.n_q || tokens->n_frames <= 0) {
        codec_context_set_error(ctx, "XY-Tokenizer: tokens shape mismatch");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t total_codes = tokens->n_frames;

    // Chunked decode, mirroring HF `XYTokenizerModel.decode(overlap_seconds=10)`.
    // The decode transformer pos_emb tables are sized for a fixed window
    // (chunk_code_length = chunk_length_s * enc_sr / enc_downsample = 375), so
    // any window longer than that overruns the table.  We decode overlapping
    // windows of up to `chunk_code_length` codes, advancing by
    // `duration_code_length` codes each step, and keep only the leading
    // `duration_wav_length` samples of each window (the trailing overlap codes
    // are look-ahead context that gets discarded).  A final trim clamps the
    // stitched waveform to `total_codes * decoder_upsample_rate` samples.
    const int32_t chunk_len_s  = std::max(1, cfg.mel_chunk_length_s);
    const int32_t overlap_s    = 10;   // HF default
    const int32_t duration_s   = std::max(1, chunk_len_s - overlap_s);
    const int32_t chunk_code_length = std::max<int32_t>(1,
        (int32_t) (((int64_t) chunk_len_s * cfg.encode_sample_rate) / std::max(1, cfg.encoder_downsample_rate)));
    // Advance step; clamp to [1, chunk_code_length] so the loop always makes
    // progress and never advances past the window it just decoded.
    const int32_t duration_code_length = std::min(chunk_code_length, std::max<int32_t>(1,
        (int32_t) (((int64_t) duration_s * cfg.encode_sample_rate) / std::max(1, cfg.encoder_downsample_rate))));
    const int64_t duration_wav_length =
        (int64_t) duration_code_length * (int64_t) cfg.decoder_upsample_rate;
    const int64_t total_wav_length =
        (int64_t) total_codes * (int64_t) cfg.decoder_upsample_rate;

    codec_graph_eval_guard guard(ctx);
    std::string err;

    std::vector<float> stitched;
    stitched.reserve((size_t) total_wav_length + (size_t) cfg.decoder_upsample_rate);

    std::vector<float> chunk_pcm;
    for (int32_t start = 0; start < total_codes; start += duration_code_length) {
        const int32_t end = std::min(start + chunk_code_length, total_codes);
        const int32_t n_codes = end - start;
        if (n_codes <= 0) break;

        const int32_t * chunk_ptr =
            tokens->data + (size_t) start * (size_t) cfg.n_q;
        enum codec_status st =
            codec_xy_decode_chunk(ctx, cfg, chunk_ptr, n_codes, &chunk_pcm, &err);
        if (st != CODEC_STATUS_SUCCESS) {
            codec_context_set_error(ctx, err);
            return st;
        }

        // Keep only the leading `duration_wav_length` samples of this window,
        // exactly as HF does for *every* chunk (`valid_wav_lengths =
        // clamp(chunk_wav_lengths, 0, duration_wav_length)`).  The trailing
        // overlap codes are look-ahead context; the next window re-decodes
        // them starting at its own position 0.  The final trim below clamps
        // the stitched result to the exact total sample count.
        int64_t keep = std::min<int64_t>((int64_t) chunk_pcm.size(), duration_wav_length);
        stitched.insert(stitched.end(), chunk_pcm.begin(), chunk_pcm.begin() + (size_t) keep);
    }

    // Trim to exactly total_codes * upsample_rate samples, mirroring HF's
    // `wav_tensor[:, :code_lengths * decoder_upsample_rate]`.  Each window
    // produces at least its share of samples (the per-chunk iSTFT overshoots
    // n_codes*hop by an edge tail), so the stitched stream is never short.
    if ((int64_t) stitched.size() > total_wav_length) {
        stitched.resize((size_t) total_wav_length);
    }

    float * pcm_out = static_cast<float *>(std::malloc(std::max<size_t>(1, stitched.size()) * sizeof(float)));
    if (pcm_out == nullptr) return CODEC_STATUS_INTERNAL_ERROR;
    std::memcpy(pcm_out, stitched.data(), stitched.size() * sizeof(float));
    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data        = pcm_out;
    out_pcm->n_samples   = (int32_t) stitched.size();
    out_pcm->sample_rate = cfg.sample_rate;
    out_pcm->n_channels  = 1;
    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_xy_tokenizer_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_xy_encode(ctx, pcm, out_tokens, out_latent, params);
}

enum codec_status codec_xy_tokenizer_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_xy_decode(ctx, tokens, out_pcm, params);
}

// =====================================================================
// vtable
// =====================================================================

static void * codec_xy_create_impl()  { return new (std::nothrow) codec_xy_tokenizer(); }
static void   codec_xy_destroy_impl(void * p) { delete static_cast<codec_xy_tokenizer *>(p); }

const struct codec_model_vtable * codec_xy_tokenizer_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_XY_TOKENIZER,
        "xy_tokenizer",
        codec_xy_create_impl,
        codec_xy_destroy_impl,
        codec_xy_tokenizer_init,
        codec_graph_size_exact,
        codec_xy_tokenizer_encode,
        codec_xy_tokenizer_decode,
    };
    return &vtable;
}
