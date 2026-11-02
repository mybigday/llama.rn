#ifndef CODEC_BLOCKS_CONFORMER_H
#define CODEC_BLOCKS_CONFORMER_H

#include "../codec_internal.h"

// WeNet/CosyVoice ConformerEncoder pieces (rel_pos_espnet, rel_selfattn,
// normalize_before, no macaron / cnn module), shared by the CosyVoice v1
// flow + text encoders and the CosyVoice2 / Chatterbox UpsampleConformerEncoder.
struct codec_conformer_params {
    int32_t hidden = 512;
    int32_t n_heads = 8;
    float block_eps = 1e-12f;    // norm_mha / norm_ff
    float embed_eps = 1e-5f;     // embed LayerNorm and after_norm
    // Chunk-of-one attention mask (static_chunk_size = 1 with all left
    // chunks): CosyVoice v1's text encoder is effectively causal.
    bool causal = false;
};

// LinearNoSubsampling: Linear → LayerNorm → * sqrt(hidden). x [in, T] → [T, hidden].
// Weights `<prefix>.lin`, `<prefix>.ln`.
ggml_tensor * codec_block_conformer_embed(
    ggml_context * ctx, const codec_model * model, const std::string & prefix,
    const codec_conformer_params & p, ggml_tensor * x_ct);

// One block: x += attn(norm_mha(x)); x += ffn_swish(norm_ff(x)). x [T, hidden];
// pos_emb is codec_op_espnet_rel_pos_emb(T, hidden). Weights `<prefix>.{norm_mha,
// norm_ff, attn.{q,k,v,o,pos,pbu,pbv}, ff.{w1,w2}}`.
ggml_tensor * codec_block_conformer_layer(
    ggml_context * ctx, const codec_model * model, const std::string & prefix,
    const codec_conformer_params & p, ggml_tensor * x_tc, ggml_tensor * pos_emb);

// `n_layers` blocks `<prefix>.<i>` over x [T, hidden] (builds the rel-pos PE).
ggml_tensor * codec_block_conformer_stack(
    ggml_context * ctx, const codec_model * model, const std::string & prefix,
    const codec_conformer_params & p, ggml_tensor * x_tc, int32_t n_layers);

// PreLookaheadLayer: right-pad L → Conv(k=L+1) → leaky_relu(0.01) → left-pad
// (k2-1) → Conv(k2) → + x. x [T, c]. Weights `<prefix>.cv1`, `<prefix>.cv2`.
ggml_tensor * codec_block_pre_lookahead(
    ggml_context * ctx, const codec_model * model, const std::string & prefix, ggml_tensor * x_tc);

// Upsample1D of UpsampleConformerEncoder: nearest x stride → left-pad
// 2*stride → Conv1d(k = 2*stride+1). x [T, c] → [stride*T, c].
ggml_tensor * codec_block_upsample_conv(
    ggml_context * ctx, const codec_model * model, const std::string & prefix,
    ggml_tensor * x_tc, int32_t stride);

#endif
