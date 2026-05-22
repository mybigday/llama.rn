#include "dac.h"

#include "../ops/conv1d.h"
#include "../ops/convtr1d.h"
#include "../ops/lm_ggml_ops.h"
#include "../ops/rvq.h"
#include "../runtime/graph.h"
#include "../runtime/tensor_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

enum codec_status codec_dac_init(struct codec_model * model) {
    codec_dac & dac = *static_cast<codec_dac *>(model->impl);

    dac.sample_rate = codec_read_i32_kv(model->gguf, "codec.sample_rate", 24000);
    dac.hop_size = codec_read_i32_kv(model->gguf, "codec.hop_size", 512);
    dac.n_q = codec_read_i32_kv(model->gguf, "codec.n_q", 4);
    dac.codebook_size = codec_read_i32_kv(model->gguf, "codec.codebook_size", 1024);
    dac.latent_dim = codec_read_i32_kv(model->gguf, "codec.latent_dim", 1024);
    dac.codebook_dim = codec_read_i32_kv(model->gguf, "codec.codebook_dim", 8);
    dac.has_encoder = codec_read_bool_kv(model->gguf, "codec.has_encoder", true);
    dac.has_decoder = codec_read_bool_kv(model->gguf, "codec.has_decoder", true);

    model->sample_rate = dac.sample_rate;
    model->has_encoder = dac.has_encoder;
    model->has_decoder = dac.has_decoder;
    model->hop_size = dac.hop_size;
    model->n_q = dac.n_q;
    model->codebook_size = dac.codebook_size;
    model->n_fft = -1;
    model->win_length = -1;
    model->n_mels = -1;
    model->latent_dim = dac.latent_dim;

    return CODEC_STATUS_SUCCESS;
}

static constexpr int32_t CODEC_DAC_MAX_BLOCKS = 8;
static constexpr int32_t CODEC_DAC_RES_UNITS = 3;
static constexpr int32_t CODEC_DAC_RES_DILATIONS[CODEC_DAC_RES_UNITS] = { 1, 3, 9 };
static constexpr int32_t CODEC_DAC_NAMING_MODEL = 1;
static constexpr int32_t CODEC_DAC_NAMING_LEGACY = 0;

struct dac_decode_build {
    int32_t t;
    int32_t q;
    int32_t hop;
    int32_t codebook_dim;
    int32_t hidden_dim;
    int32_t codebook_size;
    int32_t naming_mode;
    int32_t n_blocks;
    int32_t conv1_kernel;
    int32_t conv1_out;
    int32_t conv2_kernel;
    int32_t conv2_in;
    int32_t block_stride[CODEC_DAC_MAX_BLOCKS];
    int32_t block_convtr_kernel[CODEC_DAC_MAX_BLOCKS];
    int32_t block_convtr_in[CODEC_DAC_MAX_BLOCKS];
    int32_t block_convtr_out[CODEC_DAC_MAX_BLOCKS];
};

struct dac_decoder_shapes {
    int32_t stride[CODEC_DAC_MAX_BLOCKS] = {0};
    int32_t convtr_kernel[CODEC_DAC_MAX_BLOCKS] = {0};
    int32_t convtr_in[CODEC_DAC_MAX_BLOCKS] = {0};
    int32_t convtr_out[CODEC_DAC_MAX_BLOCKS] = {0};
};

struct dac_encode_build {
    int32_t n_in;
    int32_t hop;
    int32_t n_q;
    int32_t codebook_dim;
    int32_t codebook_size;
    int32_t hidden_dim;
    int32_t n_blocks;
    int32_t conv1_kernel;
    int32_t conv1_out;
    int32_t conv2_kernel;
    int32_t conv2_in;
    int32_t block_stride[CODEC_DAC_MAX_BLOCKS];
    int32_t block_conv_kernel[CODEC_DAC_MAX_BLOCKS];
    int32_t block_conv_in[CODEC_DAC_MAX_BLOCKS];
    int32_t block_conv_out[CODEC_DAC_MAX_BLOCKS];
};

static lm_ggml_tensor * codec_dac_get_tensor(codec_model * model, const std::string & name) {
    if (model == nullptr || model->weights == nullptr) {
        return nullptr;
    }
    return lm_ggml_get_tensor(model->weights, name.c_str());
}

static std::string codec_dac_decode_codebook_tensor_name(int32_t qi) {
    return "dac.decode.vq.q" + std::to_string(qi) + ".codebook";
}

static std::string codec_dac_decode_out_proj_tensor_name(int32_t qi) {
    return "dac.decode.vq.q" + std::to_string(qi) + ".out_proj";
}
static std::string codec_dac_decode_out_proj_bias_tensor_name(int32_t qi) {
    return "dac.decode.vq.q" + std::to_string(qi) + ".out_proj.b";
}

static std::string codec_dac_decode_conv1_w_tensor_name() { return "dac.decode.dec.conv1.w"; }
static std::string codec_dac_decode_conv1_b_tensor_name() { return "dac.decode.dec.conv1.b"; }
static std::string codec_dac_decode_conv2_w_tensor_name() { return "dac.decode.dec.conv2.w"; }
static std::string codec_dac_decode_conv2_b_tensor_name() { return "dac.decode.dec.conv2.b"; }

static std::string codec_dac_decode_block_snake_tensor_name(int32_t bi) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".snake.a";
}

static std::string codec_dac_decode_block_convtr_w_tensor_name(int32_t bi) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".convtr.w";
}

static std::string codec_dac_decode_block_convtr_b_tensor_name(int32_t bi) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".convtr.b";
}

static std::string codec_dac_decode_resunit_snake1_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".s1.a";
}

static std::string codec_dac_decode_resunit_snake2_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".s2.a";
}

static std::string codec_dac_decode_resunit_conv1_w_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c1.w";
}

static std::string codec_dac_decode_resunit_conv1_b_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c1.b";
}

static std::string codec_dac_decode_resunit_conv2_w_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c2.w";
}

static std::string codec_dac_decode_resunit_conv2_b_tensor_name(int32_t bi, int32_t ri) {
    return "dac.decode.dec.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c2.b";
}

static std::string codec_dac_decode_final_snake_tensor_name() { return "dac.decode.dec.final_snake.a"; }

static std::string codec_dac_model_block_prefix(int32_t naming_mode, int32_t bi) {
    if (naming_mode == CODEC_DAC_NAMING_MODEL) {
        return "dec.model." + std::to_string(bi + 1) + ".block.";
    }
    return "dec.block." + std::to_string(bi) + ".";
}

static std::string codec_dac_model_conv1_w_name(int32_t naming_mode) {
    return naming_mode == CODEC_DAC_NAMING_MODEL ? "dec.model.0.weight" : "dec.conv1.weight";
}

static std::string codec_dac_model_conv1_b_name(int32_t naming_mode) {
    return naming_mode == CODEC_DAC_NAMING_MODEL ? "dec.model.0.bias" : "dec.conv1.bias";
}

static std::string codec_dac_model_conv2_w_name(int32_t naming_mode, int32_t n_blocks) {
    if (naming_mode == CODEC_DAC_NAMING_MODEL) {
        return "dec.model." + std::to_string(n_blocks + 2) + ".weight";
    }
    return "dec.conv2.weight";
}

static std::string codec_dac_model_conv2_b_name(int32_t naming_mode, int32_t n_blocks) {
    if (naming_mode == CODEC_DAC_NAMING_MODEL) {
        return "dec.model." + std::to_string(n_blocks + 2) + ".bias";
    }
    return "dec.conv2.bias";
}

static std::string codec_dac_model_final_snake_name(int32_t naming_mode, int32_t n_blocks) {
    if (naming_mode == CODEC_DAC_NAMING_MODEL) {
        return "dec.model." + std::to_string(n_blocks + 1) + ".alpha";
    }
    return "dec.snake1.alpha";
}

static std::string codec_dac_model_enc_conv1_w_name() { return "enc.block.0.weight"; }
static std::string codec_dac_model_enc_conv1_b_name() { return "enc.block.0.bias"; }
static std::string codec_dac_model_enc_block_prefix(int32_t bi) {
    return "enc.block." + std::to_string(bi + 1) + ".block.";
}
static std::string codec_dac_model_enc_block_conv1_w_name(int32_t bi) {
    return codec_dac_model_enc_block_prefix(bi) + "conv1.weight";
}
static std::string codec_dac_model_enc_block_conv1_b_name(int32_t bi) {
    return codec_dac_model_enc_block_prefix(bi) + "conv1.bias";
}
static std::string codec_dac_model_enc_block_snake_name(int32_t bi) {
    return codec_dac_model_enc_block_prefix(bi) + "snake1.alpha";
}
static std::string codec_dac_model_enc_resunit_snake1_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".snake1.alpha";
}
static std::string codec_dac_model_enc_resunit_snake2_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".snake2.alpha";
}
static std::string codec_dac_model_enc_resunit_conv1_w_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".conv1.weight";
}
static std::string codec_dac_model_enc_resunit_conv1_b_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".conv1.bias";
}
static std::string codec_dac_model_enc_resunit_conv2_w_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".conv2.weight";
}
static std::string codec_dac_model_enc_resunit_conv2_b_name(int32_t bi, int32_t ri) {
    return codec_dac_model_enc_block_prefix(bi) + "res_unit" + std::to_string(ri + 1) + ".conv2.bias";
}
static std::string codec_dac_model_enc_final_snake_name() { return "enc.block.5.alpha"; }
static std::string codec_dac_model_enc_conv2_w_name() { return "enc.block.6.weight"; }
static std::string codec_dac_model_enc_conv2_b_name() { return "enc.block.6.bias"; }

static std::string codec_dac_encode_conv1_w_tensor_name() { return "dac.encode.enc.conv1.w"; }
static std::string codec_dac_encode_conv1_b_tensor_name() { return "dac.encode.enc.conv1.b"; }
static std::string codec_dac_encode_conv2_w_tensor_name() { return "dac.encode.enc.conv2.w"; }
static std::string codec_dac_encode_conv2_b_tensor_name() { return "dac.encode.enc.conv2.b"; }
static std::string codec_dac_encode_block_snake_tensor_name(int32_t bi) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".snake.a";
}
static std::string codec_dac_encode_block_conv_w_tensor_name(int32_t bi) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".conv.w";
}
static std::string codec_dac_encode_block_conv_b_tensor_name(int32_t bi) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".conv.b";
}
static std::string codec_dac_encode_resunit_snake1_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".s1.a";
}
static std::string codec_dac_encode_resunit_snake2_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".s2.a";
}
static std::string codec_dac_encode_resunit_conv1_w_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c1.w";
}
static std::string codec_dac_encode_resunit_conv1_b_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c1.b";
}
static std::string codec_dac_encode_resunit_conv2_w_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c2.w";
}
static std::string codec_dac_encode_resunit_conv2_b_tensor_name(int32_t bi, int32_t ri) {
    return "dac.encode.enc.b" + std::to_string(bi) + ".r" + std::to_string(ri) + ".c2.b";
}
static std::string codec_dac_encode_final_snake_tensor_name() { return "dac.encode.enc.final_snake.a"; }
static std::string codec_dac_encode_codebook_tensor_name(int32_t qi) {
    return "dac.encode.vq.q" + std::to_string(qi) + ".codebook";
}
static std::string codec_dac_encode_in_proj_w_tensor_name(int32_t qi) {
    return "dac.encode.vq.q" + std::to_string(qi) + ".in_proj.w";
}
static std::string codec_dac_encode_in_proj_b_tensor_name(int32_t qi) {
    return "dac.encode.vq.q" + std::to_string(qi) + ".in_proj.b";
}
static std::string codec_dac_encode_out_proj_w_tensor_name(int32_t qi) {
    return "dac.encode.vq.q" + std::to_string(qi) + ".out_proj.w";
}
static std::string codec_dac_encode_out_proj_b_tensor_name(int32_t qi) {
    return "dac.encode.vq.q" + std::to_string(qi) + ".out_proj.b";
}

static std::string codec_dac_model_block_snake_name(int32_t naming_mode, int32_t bi) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "snake1.alpha";
}

static std::string codec_dac_model_block_convtr_w_name(int32_t naming_mode, int32_t bi) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "conv_t1.weight";
}

static std::string codec_dac_model_block_convtr_b_name(int32_t naming_mode, int32_t bi) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "conv_t1.bias";
}

static std::string codec_dac_model_resunit_snake1_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".snake1.alpha";
}

static std::string codec_dac_model_resunit_snake2_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".snake2.alpha";
}

static std::string codec_dac_model_resunit_conv1_w_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".conv1.weight";
}

static std::string codec_dac_model_resunit_conv1_b_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".conv1.bias";
}

static std::string codec_dac_model_resunit_conv2_w_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".conv2.weight";
}

static std::string codec_dac_model_resunit_conv2_b_name(int32_t naming_mode, int32_t bi, int32_t ri) {
    return codec_dac_model_block_prefix(naming_mode, bi) + "res_unit" + std::to_string(ri + 1) + ".conv2.bias";
}

static bool codec_dac_infer_conv1d_shape(
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t * kernel,
    int32_t * in_c,
    int32_t * out_c) {

    if (w == nullptr || b == nullptr || kernel == nullptr || in_c == nullptr || out_c == nullptr) {
        return false;
    }
    const int32_t n0 = (int32_t) codec_ne(w, 0);
    const int32_t n1 = (int32_t) codec_ne(w, 1);
    const int32_t n2 = (int32_t) codec_ne(w, 2);
    const int32_t bo = (int32_t) codec_ne(b, 0);
    if (n0 == bo) {
        *out_c = n0;
        *in_c = n1;
        *kernel = n2;
        return true;
    }
    if (n2 == bo) {
        *out_c = n2;
        *in_c = n1;
        *kernel = n0;
        return true;
    }
    return false;
}

static bool codec_dac_infer_convtr_shape(
    lm_ggml_tensor * w,
    lm_ggml_tensor * b,
    int32_t * kernel,
    int32_t * in_c,
    int32_t * out_c) {

    if (w == nullptr || b == nullptr || kernel == nullptr || in_c == nullptr || out_c == nullptr) {
        return false;
    }
    const int32_t n0 = (int32_t) codec_ne(w, 0);
    const int32_t n1 = (int32_t) codec_ne(w, 1);
    const int32_t n2 = (int32_t) codec_ne(w, 2);
    const int32_t bo = (int32_t) codec_ne(b, 0);
    if (n1 != bo) {
        return false;
    }
    *out_c = bo;
    if (n0 <= 64) {
        *kernel = n0;
        *in_c = n2;
        return true;
    }
    *in_c = n0;
    *kernel = n2;
    return true;
}

static bool codec_dac_copy_conv1d_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err) {

    lm_ggml_tensor * src = codec_dac_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading DAC tensor: " + src_name;
        }
        return false;
    }

    const int32_t dk = (int32_t) codec_ne(dst, 0);
    const int32_t din = (int32_t) codec_ne(dst, 1);
    const int32_t dout = (int32_t) codec_ne(dst, 2);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    const int32_t n2 = (int32_t) codec_ne(src, 2);
    std::vector<float> dst_v((size_t) dk * (size_t) din * (size_t) dout, 0.0f);

    if (n0 == dk && n1 == din && n2 == dout) {
        dst_v = src_v;
    } else if (n0 == dout && n1 == din && n2 == dk) {
        for (int32_t k = 0; k < dk; ++k) {
            for (int32_t i = 0; i < din; ++i) {
                for (int32_t o = 0; o < dout; ++o) {
                    const size_t src_idx = (size_t) o + (size_t) dout * ((size_t) i + (size_t) din * (size_t) k);
                    const size_t dst_idx = (size_t) k + (size_t) dk * ((size_t) i + (size_t) din * (size_t) o);
                    dst_v[dst_idx] = src_v[src_idx];
                }
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected DAC conv1d shape: " + src_name;
        }
        return false;
    }

    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_dac_copy_convtr_weight_to_3d(
    codec_context * ctx,
    const std::string & src_name,
    lm_ggml_tensor * dst,
    std::string * err) {

    lm_ggml_tensor * src = codec_dac_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading DAC tensor: " + src_name;
        }
        return false;
    }

    const int32_t dk = (int32_t) codec_ne(dst, 0);
    const int32_t dout = (int32_t) codec_ne(dst, 1);
    const int32_t din = (int32_t) codec_ne(dst, 2);
    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    const int32_t n2 = (int32_t) codec_ne(src, 2);
    std::vector<float> dst_v((size_t) dk * (size_t) dout * (size_t) din, 0.0f);

    if (n0 == dk && n1 == dout && n2 == din) {
        dst_v = src_v;
    } else if (n0 == din && n1 == dout && n2 == dk) {
        for (int32_t k = 0; k < dk; ++k) {
            for (int32_t o = 0; o < dout; ++o) {
                for (int32_t i = 0; i < din; ++i) {
                    const size_t src_idx = (size_t) i + (size_t) din * ((size_t) o + (size_t) dout * (size_t) k);
                    const size_t dst_idx = (size_t) k + (size_t) dk * ((size_t) o + (size_t) dout * (size_t) i);
                    dst_v[dst_idx] = src_v[src_idx];
                }
            }
        }
    } else {
        if (err != nullptr) {
            *err = "unexpected DAC convtr shape: " + src_name;
        }
        return false;
    }

    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_dac_copy_bias_1d(codec_context * ctx, const std::string & src_name, lm_ggml_tensor * dst, std::string * err) {
    lm_ggml_tensor * src = codec_dac_get_tensor(ctx->model, src_name);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC tensor: " + src_name;
        }
        return false;
    }
    std::vector<float> v;
    if (!codec_tensor_as_vec_f32(src, &v) || (int32_t) v.size() != (int32_t) codec_ne(dst, 0)) {
        if (err != nullptr) {
            *err = "invalid DAC bias tensor: " + src_name;
        }
        return false;
    }
    return codec_runtime_write_tensor(dst, v.data(), v.size() * sizeof(float), err);
}

static bool codec_dac_copy_snake_alpha(codec_context * ctx, const std::string & src_name, lm_ggml_tensor * dst, std::string * err) {
    return codec_dac_copy_bias_1d(ctx, src_name, dst, err);
}

static lm_ggml_tensor * codec_dac_sum_quantized_latent(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * t_tok,
    int32_t t,
    int32_t q,
    int32_t codebook_dim,
    int32_t hidden_dim,
    int32_t codebook_size) {

    lm_ggml_tensor * acc = nullptr;
    for (int32_t qi = 0; qi < q; ++qi) {
        lm_ggml_tensor * t_codebook = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, codebook_dim, codebook_size);
        lm_ggml_set_name(t_codebook, codec_dac_decode_codebook_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_proj = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, codebook_dim, hidden_dim);
        lm_ggml_set_name(t_out_proj, codec_dac_decode_out_proj_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_bias = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, hidden_dim);
        lm_ggml_set_name(t_out_bias, codec_dac_decode_out_proj_bias_tensor_name(qi).c_str());

        lm_ggml_tensor * t_idx = lm_ggml_view_1d(ctx_eval, t_tok, t, (size_t) qi * t_tok->nb[1]);
        lm_ggml_tensor * t_embed = lm_ggml_get_rows(ctx_eval, t_codebook, t_idx);     // [codebook_dim, t]
        lm_ggml_tensor * t_lat_q = lm_ggml_mul_mat(ctx_eval, t_out_proj, t_embed);     // [hidden_dim, t]
        lm_ggml_tensor * t_lat_qt = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_lat_q));
        lm_ggml_tensor * t_bias_2d = lm_ggml_reshape_2d(ctx_eval, t_out_bias, 1, hidden_dim);
        t_lat_qt = lm_ggml_add(ctx_eval, t_lat_qt, lm_ggml_repeat(ctx_eval, t_bias_2d, t_lat_qt));
        t_lat_q = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_lat_qt));
        acc = (acc == nullptr) ? t_lat_q : lm_ggml_add(ctx_eval, acc, t_lat_q);
    }
    return acc;
}

static lm_ggml_tensor * codec_dac_resunit_ggml(
    lm_ggml_context * ctx_eval,
    lm_ggml_tensor * x,
    lm_ggml_tensor * s1,
    lm_ggml_tensor * c1_w,
    lm_ggml_tensor * c1_b,
    lm_ggml_tensor * s2,
    lm_ggml_tensor * c2_w,
    lm_ggml_tensor * c2_b,
    int32_t dilation) {

    if (ctx_eval == nullptr || x == nullptr || s1 == nullptr || c1_w == nullptr || c1_b == nullptr ||
        s2 == nullptr || c2_w == nullptr || c2_b == nullptr || dilation <= 0) {
        return nullptr;
    }

    lm_ggml_tensor * h = codec_op_snake(ctx_eval, x, s1, 1e-9f);
    h = codec_conv1d(ctx_eval, h, c1_w, c1_b, 1, dilation, 3 * dilation);
    if (h == nullptr) {
        return nullptr;
    }
    h = codec_op_snake(ctx_eval, h, s2, 1e-9f);
    h = codec_conv1d(ctx_eval, h, c2_w, c2_b, 1, 1, 0);
    if (h == nullptr) {
        return nullptr;
    }

    lm_ggml_tensor * skip = x;
    if (skip->ne[0] != h->ne[0]) {
        const int32_t diff = (int32_t) (skip->ne[0] - h->ne[0]);
        if (diff < 0) {
            return nullptr;
        }
        const int32_t crop_l = diff / 2;
        const int32_t crop_r = diff - crop_l;
        skip = codec_op_crop_1d(ctx_eval, skip, crop_l, crop_r);
        if (skip == nullptr) {
            return nullptr;
        }
    }
    return lm_ggml_cont(ctx_eval, lm_ggml_add(ctx_eval, skip, h));
}

static bool codec_dac_build_decode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    dac_decode_build * p = static_cast<dac_decode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->t <= 0 || p->q <= 0 ||
        p->codebook_dim <= 0 || p->hidden_dim <= 0 || p->codebook_size <= 1 || p->n_blocks <= 0 || p->n_blocks > CODEC_DAC_MAX_BLOCKS ||
        p->conv1_kernel <= 0 || p->conv1_out <= 0 || p->conv2_kernel <= 0 || p->conv2_in <= 0) {
        return false;
    }

    lm_ggml_tensor * t_tok = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_I32, p->t, p->q);
    lm_ggml_set_name(t_tok, "dac.decode.tok");

    lm_ggml_tensor * t_latent_ct = codec_dac_sum_quantized_latent(
        ctx_eval, t_tok, p->t, p->q, p->codebook_dim, p->hidden_dim, p->codebook_size);
    if (t_latent_ct == nullptr) {
        return false;
    }
    lm_ggml_tensor * x = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, t_latent_ct)); // [t, hidden]

    lm_ggml_tensor * t_c1_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, p->conv1_kernel, p->hidden_dim, p->conv1_out);
    lm_ggml_set_name(t_c1_w, codec_dac_decode_conv1_w_tensor_name().c_str());
    lm_ggml_tensor * t_c1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->conv1_out);
    lm_ggml_set_name(t_c1_b, codec_dac_decode_conv1_b_tensor_name().c_str());
    x = codec_conv1d(ctx_eval, x, t_c1_w, t_c1_b, 1, 1, p->conv1_kernel / 2);
    if (x == nullptr) {
        return false;
    }

    for (int32_t bi = 0; bi < p->n_blocks; ++bi) {
        lm_ggml_tensor * t_blk_snake = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
        lm_ggml_set_name(t_blk_snake, codec_dac_decode_block_snake_tensor_name(bi).c_str());
        x = codec_op_snake(ctx_eval, x, t_blk_snake, 1e-9f);

        const int32_t blk_k = p->block_convtr_kernel[bi];
        const int32_t blk_in = p->block_convtr_in[bi];
        const int32_t blk_out = p->block_convtr_out[bi];
        const int32_t blk_stride = p->block_stride[bi];
        if (blk_k <= 0 || blk_in <= 0 || blk_out <= 0 || blk_stride <= 0) {
            return false;
        }
        lm_ggml_tensor * t_ctr_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, blk_k, blk_out, blk_in);
        lm_ggml_set_name(t_ctr_w, codec_dac_decode_block_convtr_w_tensor_name(bi).c_str());
        lm_ggml_tensor * t_ctr_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, blk_out);
        lm_ggml_set_name(t_ctr_b, codec_dac_decode_block_convtr_b_tensor_name(bi).c_str());
        const int32_t blk_padding = std::max(0, (blk_stride + 1) / 2);
        x = codec_convtr1d(ctx_eval, x, t_ctr_w, t_ctr_b, blk_stride, blk_padding, 1);
        if (x == nullptr) {
            return false;
        }

        for (int32_t ri = 0; ri < CODEC_DAC_RES_UNITS; ++ri) {
            lm_ggml_tensor * t_s1 = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_s1, codec_dac_decode_resunit_snake1_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c1rw = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 7, (int64_t) x->ne[1], (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c1rw, codec_dac_decode_resunit_conv1_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c1rb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c1rb, codec_dac_decode_resunit_conv1_b_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_s2 = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_s2, codec_dac_decode_resunit_snake2_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c2rw = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, (int64_t) x->ne[1], (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c2rw, codec_dac_decode_resunit_conv2_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c2rb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c2rb, codec_dac_decode_resunit_conv2_b_tensor_name(bi, ri).c_str());
            x = codec_dac_resunit_ggml(
                ctx_eval,
                x,
                t_s1,
                t_c1rw,
                t_c1rb,
                t_s2,
                t_c2rw,
                t_c2rb,
                CODEC_DAC_RES_DILATIONS[ri]);
            if (x == nullptr) {
                return false;
            }
        }
    }

    lm_ggml_tensor * t_final_snake = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->conv2_in);
    lm_ggml_set_name(t_final_snake, codec_dac_decode_final_snake_tensor_name().c_str());
    x = codec_op_snake(ctx_eval, x, t_final_snake, 1e-9f);

    lm_ggml_tensor * t_c2_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, p->conv2_kernel, p->conv2_in, 1);
    lm_ggml_set_name(t_c2_w, codec_dac_decode_conv2_w_tensor_name().c_str());
    lm_ggml_tensor * t_c2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, 1);
    lm_ggml_set_name(t_c2_b, codec_dac_decode_conv2_b_tensor_name().c_str());
    lm_ggml_tensor * t_pcm = codec_conv1d(ctx_eval, x, t_c2_w, t_c2_b, 1, 1, p->conv2_kernel / 2);
    if (t_pcm == nullptr) {
        return false;
    }

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, lm_ggml_tanh(ctx_eval, t_pcm));
    lm_ggml_set_name(t_out, "dac.decode.out");

    *out = t_out;
    return true;
}

static bool codec_dac_copy_out_proj_to_2d(
    codec_context * ctx,
    const std::string & model_name,
    lm_ggml_tensor * dst,
    int32_t codebook_dim,
    int32_t hidden_dim,
    std::string * err) {

    lm_ggml_tensor * src = lm_ggml_get_tensor(ctx->model->weights, model_name.c_str());
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC tensor: " + model_name;
        }
        return false;
    }
    std::vector<float> src_v;
    if (!codec_tensor_as_vec_f32(src, &src_v)) {
        if (err != nullptr) {
            *err = "failed reading DAC tensor: " + model_name;
        }
        return false;
    }

    const int32_t n0 = (int32_t) codec_ne(src, 0);
    const int32_t n1 = (int32_t) codec_ne(src, 1);
    const int32_t n2 = (int32_t) std::max<int64_t>(1, codec_ne(src, 2));
    std::vector<float> dst_v((size_t) codebook_dim * (size_t) hidden_dim, 0.0f);

    if (n0 == codebook_dim && n1 == hidden_dim && n2 == 1) {
        dst_v = src_v;
    } else if (n0 == 1 && n1 == codebook_dim && n2 == hidden_dim) {
        for (int32_t i = 0; i < codebook_dim; ++i) {
            for (int32_t o = 0; o < hidden_dim; ++o) {
                const size_t src_idx = (size_t) 0 + (size_t) n0 * ((size_t) i + (size_t) n1 * (size_t) o);
                dst_v[(size_t) i + (size_t) codebook_dim * (size_t) o] = src_v[src_idx];
            }
        }
    } else if (n0 == 1 && n1 == hidden_dim && n2 == codebook_dim) {
        for (int32_t i = 0; i < codebook_dim; ++i) {
            for (int32_t o = 0; o < hidden_dim; ++o) {
                const size_t src_idx = (size_t) 0 + (size_t) n0 * ((size_t) o + (size_t) n1 * (size_t) i);
                dst_v[(size_t) i + (size_t) codebook_dim * (size_t) o] = src_v[src_idx];
            }
        }
    } else if (n0 == codebook_dim && n1 == hidden_dim && codec_ne(src, 2) == 0) {
        dst_v = src_v;
    } else {
        if (err != nullptr) {
            *err = "unexpected DAC out_proj shape: " + model_name;
        }
        return false;
    }

    return codec_runtime_write_tensor(dst, dst_v.data(), dst_v.size() * sizeof(float), err);
}

static bool codec_dac_init_decode_build(
    codec_context * ctx,
    int32_t t,
    int32_t q,
    dac_decode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || t <= 0 || q <= 0) {
        if (err != nullptr) {
            *err = "invalid DAC decode build arguments";
        }
        return false;
    }

    const codec_dac & dac = *static_cast<const codec_dac *>(ctx->model->impl);
    build->t = t;
    build->q = q;
    build->hop = std::max(1, dac.hop_size);
    build->codebook_dim = std::max(1, dac.codebook_dim);
    build->hidden_dim = std::max(1, dac.latent_dim);
    build->codebook_size = std::max(2, dac.codebook_size);
    build->n_blocks = 0;
    std::fill_n(build->block_stride, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_convtr_kernel, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_convtr_in, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_convtr_out, CODEC_DAC_MAX_BLOCKS, 0);

    const int32_t naming_mode = codec_dac_get_tensor(ctx->model, codec_dac_model_conv1_w_name(CODEC_DAC_NAMING_MODEL)) != nullptr ?
        CODEC_DAC_NAMING_MODEL : CODEC_DAC_NAMING_LEGACY;
    build->naming_mode = naming_mode;

    lm_ggml_tensor * conv1_w = codec_dac_get_tensor(ctx->model, codec_dac_model_conv1_w_name(naming_mode));
    lm_ggml_tensor * conv1_b = codec_dac_get_tensor(ctx->model, codec_dac_model_conv1_b_name(naming_mode));
    if (conv1_w == nullptr || conv1_b == nullptr ||
        !codec_dac_infer_conv1d_shape(conv1_w, conv1_b, &build->conv1_kernel, &build->hidden_dim, &build->conv1_out)) {
        if (err != nullptr) {
            *err = "failed to infer DAC decoder conv1 shape";
        }
        return false;
    }

    int32_t prev_c = build->conv1_out;
    for (int32_t bi = 0; bi < CODEC_DAC_MAX_BLOCKS; ++bi) {
        const std::string ctr_w_name = codec_dac_model_block_convtr_w_name(naming_mode, bi);
        const std::string ctr_b_name = codec_dac_model_block_convtr_b_name(naming_mode, bi);
        lm_ggml_tensor * ctr_w = codec_dac_get_tensor(ctx->model, ctr_w_name);
        lm_ggml_tensor * ctr_b = codec_dac_get_tensor(ctx->model, ctr_b_name);
        if (ctr_w == nullptr || ctr_b == nullptr) {
            break;
        }
        int32_t k = 0, in_c = 0, out_c = 0;
        if (!codec_dac_infer_convtr_shape(ctr_w, ctr_b, &k, &in_c, &out_c) || in_c != prev_c) {
            if (err != nullptr) {
                *err = "invalid DAC decoder conv_t1 shape at block " + std::to_string(bi);
            }
            return false;
        }
        build->block_convtr_kernel[bi] = k;
        build->block_convtr_in[bi] = in_c;
        build->block_convtr_out[bi] = out_c;
        build->block_stride[bi] = std::max(1, k / 2);
        build->n_blocks = bi + 1;
        prev_c = out_c;
    }

    if (build->n_blocks <= 0) {
        if (err != nullptr) {
            *err = "no DAC decoder conv_t1 blocks found";
        }
        return false;
    }

    lm_ggml_tensor * conv2_w = codec_dac_get_tensor(ctx->model, codec_dac_model_conv2_w_name(naming_mode, build->n_blocks));
    lm_ggml_tensor * conv2_b = codec_dac_get_tensor(ctx->model, codec_dac_model_conv2_b_name(naming_mode, build->n_blocks));
    int32_t conv2_out = 0;
    if (conv2_w == nullptr || conv2_b == nullptr ||
        !codec_dac_infer_conv1d_shape(conv2_w, conv2_b, &build->conv2_kernel, &build->conv2_in, &conv2_out) ||
        build->conv2_in != prev_c || conv2_out != 1) {
        if (err != nullptr) {
            *err = "failed to infer DAC decoder conv2 shape";
        }
        return false;
    }

    if (codec_dac_get_tensor(ctx->model, codec_dac_model_final_snake_name(naming_mode, build->n_blocks)) == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC final snake tensor";
        }
        return false;
    }

    return true;
}

static bool codec_dac_init_encode_build(
    codec_context * ctx,
    int32_t n_in,
    int32_t n_q,
    dac_encode_build * build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || build == nullptr || n_in <= 0 || n_q <= 0) {
        if (err != nullptr) {
            *err = "invalid DAC encode build arguments";
        }
        return false;
    }

    const codec_dac & dac = *static_cast<const codec_dac *>(ctx->model->impl);
    build->n_in = n_in;
    build->hop = std::max(1, dac.hop_size);
    build->n_q = n_q;
    build->codebook_dim = std::max(1, dac.codebook_dim);
    build->codebook_size = std::max(2, dac.codebook_size);
    build->hidden_dim = std::max(1, dac.latent_dim);
    build->n_blocks = 0;
    std::fill_n(build->block_stride, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_conv_kernel, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_conv_in, CODEC_DAC_MAX_BLOCKS, 0);
    std::fill_n(build->block_conv_out, CODEC_DAC_MAX_BLOCKS, 0);

    lm_ggml_tensor * conv1_w = codec_dac_get_tensor(ctx->model, codec_dac_model_enc_conv1_w_name());
    lm_ggml_tensor * conv1_b = codec_dac_get_tensor(ctx->model, codec_dac_model_enc_conv1_b_name());
    int32_t conv1_in = 0;
    if (conv1_w == nullptr || conv1_b == nullptr ||
        !codec_dac_infer_conv1d_shape(conv1_w, conv1_b, &build->conv1_kernel, &conv1_in, &build->conv1_out)) {
        if (err != nullptr) {
            *err = "failed to infer DAC encoder conv1 shape";
        }
        return false;
    }
    (void) conv1_in;

    int32_t prev_c = build->conv1_out;
    for (int32_t bi = 0; bi < CODEC_DAC_MAX_BLOCKS; ++bi) {
        const std::string cw_name = codec_dac_model_enc_block_conv1_w_name(bi);
        const std::string cb_name = codec_dac_model_enc_block_conv1_b_name(bi);
        lm_ggml_tensor * cw = codec_dac_get_tensor(ctx->model, cw_name);
        lm_ggml_tensor * cb = codec_dac_get_tensor(ctx->model, cb_name);
        if (cw == nullptr || cb == nullptr) {
            break;
        }
        int32_t k = 0, in_c = 0, out_c = 0;
        if (!codec_dac_infer_conv1d_shape(cw, cb, &k, &in_c, &out_c) || in_c != prev_c) {
            if (err != nullptr) {
                *err = "invalid DAC encoder conv1 shape at block " + std::to_string(bi);
            }
            return false;
        }
        build->block_conv_kernel[bi] = k;
        build->block_conv_in[bi] = in_c;
        build->block_conv_out[bi] = out_c;
        build->block_stride[bi] = std::max(1, k / 2);
        build->n_blocks = bi + 1;
        prev_c = out_c;
    }

    if (build->n_blocks <= 0) {
        if (err != nullptr) {
            *err = "no DAC encoder blocks found";
        }
        return false;
    }

    lm_ggml_tensor * conv2_w = codec_dac_get_tensor(ctx->model, codec_dac_model_enc_conv2_w_name());
    lm_ggml_tensor * conv2_b = codec_dac_get_tensor(ctx->model, codec_dac_model_enc_conv2_b_name());
    int32_t conv2_out = 0;
    if (conv2_w == nullptr || conv2_b == nullptr ||
        !codec_dac_infer_conv1d_shape(conv2_w, conv2_b, &build->conv2_kernel, &build->conv2_in, &conv2_out) ||
        build->conv2_in != prev_c || conv2_out != build->hidden_dim) {
        if (err != nullptr) {
            *err = "failed to infer DAC encoder conv2 shape";
        }
        return false;
    }

    if (codec_dac_get_tensor(ctx->model, codec_dac_model_enc_final_snake_name()) == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC encoder final snake tensor";
        }
        return false;
    }

    return true;
}

static bool codec_dac_write_encode_weights(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    const dac_encode_build & build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || entry == nullptr) {
        if (err != nullptr) {
            *err = "invalid DAC encoder write arguments";
        }
        return false;
    }

    lm_ggml_tensor * t_c1_w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_conv1_w_tensor_name().c_str());
    lm_ggml_tensor * t_c1_b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_conv1_b_tensor_name().c_str());
    lm_ggml_tensor * t_c2_w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_conv2_w_tensor_name().c_str());
    lm_ggml_tensor * t_c2_b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_conv2_b_tensor_name().c_str());
    lm_ggml_tensor * t_fs_a = codec_graph_get_tensor(ctx, entry, codec_dac_encode_final_snake_tensor_name().c_str());
    if (t_c1_w == nullptr || t_c1_b == nullptr || t_c2_w == nullptr || t_c2_b == nullptr || t_fs_a == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC encoder core graph tensors";
        }
        return false;
    }

    if (!codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_enc_conv1_w_name(), t_c1_w, err) ||
        !codec_dac_copy_bias_1d(ctx, codec_dac_model_enc_conv1_b_name(), t_c1_b, err) ||
        !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_enc_conv2_w_name(), t_c2_w, err) ||
        !codec_dac_copy_bias_1d(ctx, codec_dac_model_enc_conv2_b_name(), t_c2_b, err) ||
        !codec_dac_copy_snake_alpha(ctx, codec_dac_model_enc_final_snake_name(), t_fs_a, err)) {
        return false;
    }

    for (int32_t bi = 0; bi < build.n_blocks; ++bi) {
        lm_ggml_tensor * t_bs = codec_graph_get_tensor(ctx, entry, codec_dac_encode_block_snake_tensor_name(bi).c_str());
        lm_ggml_tensor * t_cw = codec_graph_get_tensor(ctx, entry, codec_dac_encode_block_conv_w_tensor_name(bi).c_str());
        lm_ggml_tensor * t_cb = codec_graph_get_tensor(ctx, entry, codec_dac_encode_block_conv_b_tensor_name(bi).c_str());
        if (t_bs == nullptr || t_cw == nullptr || t_cb == nullptr) {
            if (err != nullptr) {
                *err = "missing DAC encoder block graph tensors";
            }
            return false;
        }
        if (!codec_dac_copy_snake_alpha(ctx, codec_dac_model_enc_block_snake_name(bi), t_bs, err) ||
            !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_enc_block_conv1_w_name(bi), t_cw, err) ||
            !codec_dac_copy_bias_1d(ctx, codec_dac_model_enc_block_conv1_b_name(bi), t_cb, err)) {
            return false;
        }

        for (int32_t ri = 0; ri < CODEC_DAC_RES_UNITS; ++ri) {
            lm_ggml_tensor * t_s1 = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_snake1_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_s2 = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_snake2_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r1w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_conv1_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r1b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_conv1_b_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r2w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_conv2_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r2b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_resunit_conv2_b_tensor_name(bi, ri).c_str());
            if (t_s1 == nullptr || t_s2 == nullptr || t_r1w == nullptr || t_r1b == nullptr || t_r2w == nullptr || t_r2b == nullptr) {
                if (err != nullptr) {
                    *err = "missing DAC encoder residual graph tensors";
                }
                return false;
            }
            if (!codec_dac_copy_snake_alpha(ctx, codec_dac_model_enc_resunit_snake1_name(bi, ri), t_s1, err) ||
                !codec_dac_copy_snake_alpha(ctx, codec_dac_model_enc_resunit_snake2_name(bi, ri), t_s2, err) ||
                !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_enc_resunit_conv1_w_name(bi, ri), t_r1w, err) ||
                !codec_dac_copy_bias_1d(ctx, codec_dac_model_enc_resunit_conv1_b_name(bi, ri), t_r1b, err) ||
                !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_enc_resunit_conv2_w_name(bi, ri), t_r2w, err) ||
                !codec_dac_copy_bias_1d(ctx, codec_dac_model_enc_resunit_conv2_b_name(bi, ri), t_r2b, err)) {
                return false;
            }
        }
    }

    for (int32_t qi = 0; qi < build.n_q; ++qi) {
        lm_ggml_tensor * t_codebook = codec_graph_get_tensor(ctx, entry, codec_dac_encode_codebook_tensor_name(qi).c_str());
        lm_ggml_tensor * t_in_w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_in_proj_w_tensor_name(qi).c_str());
        lm_ggml_tensor * t_in_b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_in_proj_b_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_w = codec_graph_get_tensor(ctx, entry, codec_dac_encode_out_proj_w_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_b = codec_graph_get_tensor(ctx, entry, codec_dac_encode_out_proj_b_tensor_name(qi).c_str());
        if (t_codebook == nullptr || t_in_w == nullptr || t_in_b == nullptr || t_out_w == nullptr || t_out_b == nullptr) {
            if (err != nullptr) {
                *err = "missing DAC encoder VQ tensors";
            }
            return false;
        }

        const std::string cb_name = "vq.q" + std::to_string(qi) + ".codebook.weight";
        lm_ggml_tensor * cb_src = lm_ggml_get_tensor(ctx->model->weights, cb_name.c_str());
        if (cb_src == nullptr) {
            if (err != nullptr) {
                *err = "missing DAC codebook tensor: " + cb_name;
            }
            return false;
        }
        std::vector<float> cb;
        if (!codec_tensor_as_vec_f32(cb_src, &cb)) {
            if (err != nullptr) {
                *err = "failed reading DAC codebook tensor";
            }
            return false;
        }
        const int32_t cb0 = (int32_t) codec_ne(cb_src, 0);
        const int32_t cb1 = (int32_t) codec_ne(cb_src, 1);
        std::vector<float> cb_dst((size_t) build.codebook_dim * (size_t) build.codebook_size, 0.0f);
        if (cb0 == build.codebook_size && cb1 == build.codebook_dim) {
            for (int32_t i = 0; i < build.codebook_dim; ++i) {
                for (int32_t j = 0; j < build.codebook_size; ++j) {
                    cb_dst[(size_t) i + (size_t) build.codebook_dim * (size_t) j] =
                        cb[(size_t) j + (size_t) build.codebook_size * (size_t) i];
                }
            }
        } else if (cb0 == build.codebook_dim && cb1 == build.codebook_size) {
            cb_dst = cb;
        } else {
            if (err != nullptr) {
                *err = "unexpected DAC codebook tensor shape";
            }
            return false;
        }
        if (!codec_runtime_write_tensor(t_codebook, cb_dst.data(), cb_dst.size() * sizeof(float), err)) {
            return false;
        }

        const std::string in_w_name = "vq.q" + std::to_string(qi) + ".in_proj.weight";
        const std::string in_b_name = "vq.q" + std::to_string(qi) + ".in_proj.bias";
        const std::string out_w_name = "vq.q" + std::to_string(qi) + ".out_proj.weight";
        const std::string out_b_name = "vq.q" + std::to_string(qi) + ".out_proj.bias";
        if (!codec_dac_copy_conv1d_weight_to_3d(ctx, in_w_name, t_in_w, err) ||
            !codec_dac_copy_bias_1d(ctx, in_b_name, t_in_b, err) ||
            !codec_dac_copy_conv1d_weight_to_3d(ctx, out_w_name, t_out_w, err) ||
            !codec_dac_copy_bias_1d(ctx, out_b_name, t_out_b, err)) {
            return false;
        }
    }

    return true;
}

static bool codec_dac_write_decode_weights(
    codec_context * ctx,
    codec_graph_cache_entry * entry,
    const dac_decode_build & build,
    std::string * err) {

    if (ctx == nullptr || ctx->model == nullptr || entry == nullptr) {
        if (err != nullptr) {
            *err = "invalid DAC decoder write arguments";
        }
        return false;
    }

    lm_ggml_tensor * t_c1_w = codec_graph_get_tensor(ctx, entry, codec_dac_decode_conv1_w_tensor_name().c_str());
    lm_ggml_tensor * t_c1_b = codec_graph_get_tensor(ctx, entry, codec_dac_decode_conv1_b_tensor_name().c_str());
    lm_ggml_tensor * t_c2_w = codec_graph_get_tensor(ctx, entry, codec_dac_decode_conv2_w_tensor_name().c_str());
    lm_ggml_tensor * t_c2_b = codec_graph_get_tensor(ctx, entry, codec_dac_decode_conv2_b_tensor_name().c_str());
    lm_ggml_tensor * t_fs_a = codec_graph_get_tensor(ctx, entry, codec_dac_decode_final_snake_tensor_name().c_str());
    if (t_c1_w == nullptr || t_c1_b == nullptr || t_c2_w == nullptr || t_c2_b == nullptr || t_fs_a == nullptr) {
        if (err != nullptr) {
            *err = "missing DAC decoder core graph tensors";
        }
        return false;
    }

    if (!codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_conv1_w_name(build.naming_mode), t_c1_w, err) ||
        !codec_dac_copy_bias_1d(ctx, codec_dac_model_conv1_b_name(build.naming_mode), t_c1_b, err) ||
        !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_conv2_w_name(build.naming_mode, build.n_blocks), t_c2_w, err) ||
        !codec_dac_copy_bias_1d(ctx, codec_dac_model_conv2_b_name(build.naming_mode, build.n_blocks), t_c2_b, err) ||
        !codec_dac_copy_snake_alpha(ctx, codec_dac_model_final_snake_name(build.naming_mode, build.n_blocks), t_fs_a, err)) {
        return false;
    }

    for (int32_t bi = 0; bi < build.n_blocks; ++bi) {
        lm_ggml_tensor * t_bs = codec_graph_get_tensor(ctx, entry, codec_dac_decode_block_snake_tensor_name(bi).c_str());
        lm_ggml_tensor * t_cw = codec_graph_get_tensor(ctx, entry, codec_dac_decode_block_convtr_w_tensor_name(bi).c_str());
        lm_ggml_tensor * t_cb = codec_graph_get_tensor(ctx, entry, codec_dac_decode_block_convtr_b_tensor_name(bi).c_str());
        if (t_bs == nullptr || t_cw == nullptr || t_cb == nullptr) {
            if (err != nullptr) {
                *err = "missing DAC decoder block graph tensors";
            }
            return false;
        }
        if (!codec_dac_copy_snake_alpha(ctx, codec_dac_model_block_snake_name(build.naming_mode, bi), t_bs, err) ||
            !codec_dac_copy_convtr_weight_to_3d(ctx, codec_dac_model_block_convtr_w_name(build.naming_mode, bi), t_cw, err) ||
            !codec_dac_copy_bias_1d(ctx, codec_dac_model_block_convtr_b_name(build.naming_mode, bi), t_cb, err)) {
            return false;
        }

        for (int32_t ri = 0; ri < CODEC_DAC_RES_UNITS; ++ri) {
            lm_ggml_tensor * t_s1 = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_snake1_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_s2 = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_snake2_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r1w = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_conv1_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r1b = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_conv1_b_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r2w = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_conv2_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_r2b = codec_graph_get_tensor(ctx, entry, codec_dac_decode_resunit_conv2_b_tensor_name(bi, ri).c_str());
            if (t_s1 == nullptr || t_s2 == nullptr || t_r1w == nullptr || t_r1b == nullptr || t_r2w == nullptr || t_r2b == nullptr) {
                if (err != nullptr) {
                    *err = "missing DAC decoder residual graph tensors";
                }
                return false;
            }
            if (!codec_dac_copy_snake_alpha(ctx, codec_dac_model_resunit_snake1_name(build.naming_mode, bi, ri), t_s1, err) ||
                !codec_dac_copy_snake_alpha(ctx, codec_dac_model_resunit_snake2_name(build.naming_mode, bi, ri), t_s2, err) ||
                !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_resunit_conv1_w_name(build.naming_mode, bi, ri), t_r1w, err) ||
                !codec_dac_copy_bias_1d(ctx, codec_dac_model_resunit_conv1_b_name(build.naming_mode, bi, ri), t_r1b, err) ||
                !codec_dac_copy_conv1d_weight_to_3d(ctx, codec_dac_model_resunit_conv2_w_name(build.naming_mode, bi, ri), t_r2w, err) ||
                !codec_dac_copy_bias_1d(ctx, codec_dac_model_resunit_conv2_b_name(build.naming_mode, bi, ri), t_r2b, err)) {
                return false;
            }
        }
    }

    return true;
}

struct dac_decode_latent_build {
    int32_t n_frames;
    int32_t latent_dim;
    int32_t hop;
};

static bool codec_dac_build_decode_latent(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    dac_decode_latent_build * p = static_cast<dac_decode_latent_build *>(user_data);
    lm_ggml_tensor * t_lat = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_frames, p->latent_dim);
    lm_ggml_set_name(t_lat, "dac.decode_latent.lat");

    lm_ggml_tensor * t_ch0 = lm_ggml_view_2d(ctx_eval, t_lat, p->n_frames, 1, t_lat->nb[1], 0);
    lm_ggml_tensor * t_kernel = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, p->hop, 1, 1);
    lm_ggml_set_name(t_kernel, "dac.decode_latent.kernel");

    lm_ggml_tensor * t_pcm = codec_convtr1d(ctx_eval, t_ch0, t_kernel, nullptr, p->hop, 0, 1);
    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, lm_ggml_tanh(ctx_eval, t_pcm));
    lm_ggml_set_name(t_out, "dac.decode_latent.out");

    *out = t_out;
    return true;
}

static bool codec_dac_build_encode(lm_ggml_context * ctx_eval, void * user_data, lm_ggml_tensor ** out) {
    dac_encode_build * p = static_cast<dac_encode_build *>(user_data);
    if (ctx_eval == nullptr || p == nullptr || out == nullptr || p->n_in <= 0 || p->n_q <= 0 ||
        p->codebook_dim <= 0 || p->codebook_size <= 1 || p->hidden_dim <= 0 || p->n_blocks <= 0) {
        return false;
    }

    lm_ggml_tensor * t_pcm = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->n_in, 1);
    lm_ggml_set_name(t_pcm, "dac.encode.pcm");

    lm_ggml_tensor * t_c1_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, p->conv1_kernel, 1, p->conv1_out);
    lm_ggml_set_name(t_c1_w, codec_dac_encode_conv1_w_tensor_name().c_str());
    lm_ggml_tensor * t_c1_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->conv1_out);
    lm_ggml_set_name(t_c1_b, codec_dac_encode_conv1_b_tensor_name().c_str());

    lm_ggml_tensor * x = codec_conv1d(ctx_eval, t_pcm, t_c1_w, t_c1_b, 1, 1, p->conv1_kernel / 2);
    if (x == nullptr) {
        return false;
    }

    for (int32_t bi = 0; bi < p->n_blocks; ++bi) {
        for (int32_t ri = 0; ri < CODEC_DAC_RES_UNITS; ++ri) {
            lm_ggml_tensor * t_s1 = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_s1, codec_dac_encode_resunit_snake1_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c1rw = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 7, (int64_t) x->ne[1], (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c1rw, codec_dac_encode_resunit_conv1_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c1rb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c1rb, codec_dac_encode_resunit_conv1_b_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_s2 = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_s2, codec_dac_encode_resunit_snake2_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c2rw = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, (int64_t) x->ne[1], (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c2rw, codec_dac_encode_resunit_conv2_w_tensor_name(bi, ri).c_str());
            lm_ggml_tensor * t_c2rb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
            lm_ggml_set_name(t_c2rb, codec_dac_encode_resunit_conv2_b_tensor_name(bi, ri).c_str());
            x = codec_dac_resunit_ggml(
                ctx_eval,
                x,
                t_s1,
                t_c1rw,
                t_c1rb,
                t_s2,
                t_c2rw,
                t_c2rb,
                CODEC_DAC_RES_DILATIONS[ri]);
            if (x == nullptr) {
                return false;
            }
        }

        lm_ggml_tensor * t_blk_snake = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
        lm_ggml_set_name(t_blk_snake, codec_dac_encode_block_snake_tensor_name(bi).c_str());
        x = codec_op_snake(ctx_eval, x, t_blk_snake, 1e-9f);

        const int32_t blk_k = p->block_conv_kernel[bi];
        const int32_t blk_in = p->block_conv_in[bi];
        const int32_t blk_out = p->block_conv_out[bi];
        const int32_t blk_stride = p->block_stride[bi];
        if (blk_k <= 0 || blk_in <= 0 || blk_out <= 0 || blk_stride <= 0) {
            return false;
        }
        lm_ggml_tensor * t_cw = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, blk_k, blk_in, blk_out);
        lm_ggml_set_name(t_cw, codec_dac_encode_block_conv_w_tensor_name(bi).c_str());
        lm_ggml_tensor * t_cb = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, blk_out);
        lm_ggml_set_name(t_cb, codec_dac_encode_block_conv_b_tensor_name(bi).c_str());
        const int32_t blk_pad = std::max(0, (blk_stride + 1) / 2);
        x = codec_conv1d(ctx_eval, x, t_cw, t_cb, blk_stride, 1, blk_pad);
        if (x == nullptr) {
            return false;
        }
    }

    lm_ggml_tensor * t_fs = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, (int64_t) x->ne[1]);
    lm_ggml_set_name(t_fs, codec_dac_encode_final_snake_tensor_name().c_str());
    x = codec_op_snake(ctx_eval, x, t_fs, 1e-9f);

    lm_ggml_tensor * t_c2_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, p->conv2_kernel, p->conv2_in, p->hidden_dim);
    lm_ggml_set_name(t_c2_w, codec_dac_encode_conv2_w_tensor_name().c_str());
    lm_ggml_tensor * t_c2_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->hidden_dim);
    lm_ggml_set_name(t_c2_b, codec_dac_encode_conv2_b_tensor_name().c_str());
    x = codec_conv1d(ctx_eval, x, t_c2_w, t_c2_b, 1, 1, p->conv2_kernel / 2);
    if (x == nullptr) {
        return false;
    }

    lm_ggml_tensor * residual_tc = x;
    lm_ggml_tensor * tokens = nullptr;

    for (int32_t qi = 0; qi < p->n_q; ++qi) {
        lm_ggml_tensor * t_codebook = lm_ggml_new_tensor_2d(ctx_eval, LM_GGML_TYPE_F32, p->codebook_dim, p->codebook_size);
        lm_ggml_set_name(t_codebook, codec_dac_encode_codebook_tensor_name(qi).c_str());

        lm_ggml_tensor * t_in_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->hidden_dim, p->codebook_dim);
        lm_ggml_set_name(t_in_w, codec_dac_encode_in_proj_w_tensor_name(qi).c_str());
        lm_ggml_tensor * t_in_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->codebook_dim);
        lm_ggml_set_name(t_in_b, codec_dac_encode_in_proj_b_tensor_name(qi).c_str());
        lm_ggml_tensor * z_tc = codec_conv1d(ctx_eval, residual_tc, t_in_w, t_in_b, 1, 1, 0); // [t, codebook_dim]
        if (z_tc == nullptr) {
            return false;
        }
        lm_ggml_tensor * z_ct = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, z_tc)); // [codebook_dim, t]

        codec_rvq_layer_result_ggml layer = {};
        if (!codec_rvq_build_layer_ggml(ctx_eval, z_ct, t_codebook, &layer)) {
            return false;
        }
        lm_ggml_tensor * quantized = lm_ggml_sub(ctx_eval, z_ct, layer.residual); // [codebook_dim, t]
        lm_ggml_tensor * quantized_tc = lm_ggml_cont(ctx_eval, lm_ggml_transpose(ctx_eval, quantized));

        lm_ggml_tensor * t_out_w = lm_ggml_new_tensor_3d(ctx_eval, LM_GGML_TYPE_F32, 1, p->codebook_dim, p->hidden_dim);
        lm_ggml_set_name(t_out_w, codec_dac_encode_out_proj_w_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_b = lm_ggml_new_tensor_1d(ctx_eval, LM_GGML_TYPE_F32, p->hidden_dim);
        lm_ggml_set_name(t_out_b, codec_dac_encode_out_proj_b_tensor_name(qi).c_str());
        lm_ggml_tensor * zq_tc = codec_conv1d(ctx_eval, quantized_tc, t_out_w, t_out_b, 1, 1, 0); // [t, hidden_dim]
        if (zq_tc == nullptr) {
            return false;
        }

        residual_tc = lm_ggml_sub(ctx_eval, residual_tc, zq_tc);

        lm_ggml_tensor * idx2d = lm_ggml_reshape_2d(ctx_eval, layer.indices, layer.indices->ne[0], 1);
        tokens = (tokens == nullptr) ? idx2d : lm_ggml_concat(ctx_eval, tokens, idx2d, 1);
    }

    lm_ggml_tensor * t_out = lm_ggml_cont(ctx_eval, tokens);
    lm_ggml_set_name(t_out, "dac.encode.out");
    *out = t_out;
    return true;
}

static enum codec_status codec_dac_decode_tokens_graph(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    int32_t use_n_q,
    struct codec_pcm_buffer * out_pcm,
    int32_t hop_size,
    int32_t sample_rate) {

    if (tokens == nullptr || tokens->data == nullptr || tokens->n_frames <= 0 || tokens->n_q < use_n_q) {
        codec_context_set_error(ctx, "invalid DAC token buffer");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t t = tokens->n_frames;
    const int32_t q = use_n_q;
    const size_t mem = 32 * 1024 * 1024 + (size_t) t * (size_t) q * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;

    dac_decode_build build = {};
    if (!codec_dac_init_decode_build(ctx, t, q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_DECODE, /*n_frames=*/t, /*n_q=*/q, /*hop=*/hop_size, /*n_in=*/0, /*latent_dim=*/0 },
            mem,
            codec_dac_build_decode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_tok = codec_graph_get_tensor(ctx, entry, "dac.decode.tok");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.decode.out");
    if (t_tok == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<int32_t> tok_i32((size_t) t * (size_t) q, 0);
    for (int32_t ti = 0; ti < t; ++ti) {
        for (int32_t qi = 0; qi < q; ++qi) {
            int32_t tok = tokens->data[(size_t) ti * (size_t) tokens->n_q + (size_t) qi];
            tok = std::max(0, std::min(build.codebook_size - 1, tok));
            tok_i32[(size_t) qi * (size_t) t + (size_t) ti] = tok;
        }
    }
    for (int32_t qi = 0; qi < q; ++qi) {
        lm_ggml_tensor * t_codebook = codec_graph_get_tensor(ctx, entry, codec_dac_decode_codebook_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_proj = codec_graph_get_tensor(ctx, entry, codec_dac_decode_out_proj_tensor_name(qi).c_str());
        lm_ggml_tensor * t_out_bias = codec_graph_get_tensor(ctx, entry, codec_dac_decode_out_proj_bias_tensor_name(qi).c_str());
        if (t_codebook == nullptr || t_out_proj == nullptr || t_out_bias == nullptr) {
            codec_context_set_error(ctx, "cached DAC decode graph is missing VQ tensors");
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        const std::string cb_name = "vq.q" + std::to_string(qi) + ".codebook.weight";
        lm_ggml_tensor * cb_src = lm_ggml_get_tensor(ctx->model->weights, cb_name.c_str());
        if (cb_src == nullptr) {
            codec_context_set_error(ctx, "missing DAC codebook tensor: " + cb_name);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        std::vector<float> cb;
        if (!codec_tensor_as_vec_f32(cb_src, &cb)) {
            codec_context_set_error(ctx, "failed reading DAC codebook tensor");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        const int32_t cb0 = (int32_t) codec_ne(cb_src, 0);
        const int32_t cb1 = (int32_t) codec_ne(cb_src, 1);
        std::vector<float> cb_dst((size_t) build.codebook_dim * (size_t) build.codebook_size, 0.0f);
        if (cb0 == build.codebook_dim && cb1 == build.codebook_size) {
            cb_dst = cb;
        } else if (cb0 == build.codebook_size && cb1 == build.codebook_dim) {
            for (int32_t i = 0; i < build.codebook_dim; ++i) {
                for (int32_t j = 0; j < build.codebook_size; ++j) {
                    cb_dst[(size_t) i + (size_t) build.codebook_dim * (size_t) j] =
                        cb[(size_t) j + (size_t) build.codebook_size * (size_t) i];
                }
            }
        } else {
            codec_context_set_error(ctx, "unexpected DAC codebook tensor shape");
            return CODEC_STATUS_INTERNAL_ERROR;
        }
        if (!codec_runtime_write_tensor(t_codebook, cb_dst.data(), cb_dst.size() * sizeof(float), &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }

        const std::string op_name = "vq.q" + std::to_string(qi) + ".out_proj.weight";
        const std::string ob_name = "vq.q" + std::to_string(qi) + ".out_proj.bias";
        if (!codec_dac_copy_out_proj_to_2d(ctx, op_name, t_out_proj, build.codebook_dim, build.hidden_dim, &err) ||
            !codec_dac_copy_bias_1d(ctx, ob_name, t_out_bias, &err)) {
            codec_context_set_error(ctx, err);
            return CODEC_STATUS_INTERNAL_ERROR;
        }
    }

    if (!codec_dac_write_decode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_write_tensor(t_tok, tok_i32.data(), tok_i32.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_samples = (int32_t) t_out->ne[0];
    float * pcm = static_cast<float *>(std::malloc((size_t) n_samples * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_read_tensor(t_out, pcm, (size_t) n_samples * sizeof(float), &err)) {
        std::free(pcm);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

enum codec_status codec_dac_decode_latent(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    (void) params;

    codec_dac & dac = *static_cast<codec_dac *>(ctx->model->impl);
    if (!dac.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (quantized_representation == nullptr || latent_dim <= 0 || n_frames <= 0) {
        codec_context_set_error(ctx, "invalid DAC latent input");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, dac.hop_size);
    const size_t mem = 32 * 1024 * 1024 + (size_t) n_frames * (size_t) latent_dim * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);

    dac_decode_latent_build build = { n_frames, latent_dim, hop };
    codec_graph_cache_entry * entry = nullptr;
    std::string err;
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_DECODE_LATENT, /*n_frames=*/n_frames, /*n_q=*/0, /*hop=*/hop, /*n_in=*/0, /*latent_dim=*/latent_dim },
            mem,
            codec_dac_build_decode_latent,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_lat = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.lat");
    lm_ggml_tensor * t_kernel = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.kernel");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.decode_latent.out");
    if (t_lat == nullptr || t_kernel == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC latent decode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    std::vector<float> kernel((size_t) hop, 1.0f / (float) hop);
    if (!codec_runtime_write_tensor(t_lat, quantized_representation, (size_t) n_frames * (size_t) latent_dim * sizeof(float), &err) ||
        !codec_runtime_write_tensor(t_kernel, kernel.data(), kernel.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_samples = (int32_t) t_out->ne[0];
    float * pcm = static_cast<float *>(std::malloc((size_t) n_samples * sizeof(float)));
    if (pcm == nullptr) {
        codec_context_set_error(ctx, "failed to allocate pcm output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_read_tensor(t_out, pcm, (size_t) n_samples * sizeof(float), &err)) {
        std::free(pcm);
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    codec_pcm_buffer_reset(out_pcm);
    out_pcm->data = pcm;
    out_pcm->n_samples = n_samples;
    out_pcm->sample_rate = dac.sample_rate;
    out_pcm->n_channels = 1;

    return CODEC_STATUS_SUCCESS;
}

static void * codec_dac_create_impl() {
    return new (std::nothrow) codec_dac();
}

static void codec_dac_destroy_impl(void * ptr) {
    delete static_cast<codec_dac *>(ptr);
}

static enum codec_status codec_dac_encode_wrap(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {
    return codec_dac_encode(ctx, pcm, out_tokens, out_latent, params);
}

static enum codec_status codec_dac_decode_wrap(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_dac_decode(ctx, tokens, out_pcm, params);
}

static enum codec_status codec_dac_decode_latent_wrap(
    struct codec_context * ctx,
    const float * quantized_representation,
    int32_t latent_dim,
    int32_t n_frames,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {
    return codec_dac_decode_latent(ctx, quantized_representation, latent_dim, n_frames, out_pcm, params);
}

const struct codec_model_vtable * codec_dac_vtable() {
    static const codec_model_vtable vtable = {
        CODEC_ARCH_DAC,
        "DAC",
        codec_dac_create_impl,
        codec_dac_destroy_impl,
        codec_dac_init,
        codec_dac_encode_wrap,
        codec_dac_decode_wrap,
        codec_dac_decode_latent_wrap,
    };
    return &vtable;
}

enum codec_status codec_dac_decode(
    struct codec_context * ctx,
    const struct codec_token_buffer * tokens,
    struct codec_pcm_buffer * out_pcm,
    struct codec_decode_params params) {

    codec_dac & dac = *static_cast<codec_dac *>(ctx->model->impl);
    if (!dac.has_decoder) {
        codec_context_set_error(ctx, "model metadata indicates no decoder");
        return CODEC_STATUS_INVALID_STATE;
    }

    const int32_t model_n_q = std::max(1, dac.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "DAC decode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    return codec_dac_decode_tokens_graph(
        ctx,
        tokens,
        use_n_q,
        out_pcm,
        std::max(1, dac.hop_size),
        dac.sample_rate);
}

enum codec_status codec_dac_encode(
    struct codec_context * ctx,
    const std::vector<float> & pcm,
    struct codec_token_buffer * out_tokens,
    struct codec_latent_buffer * out_latent,
    struct codec_encode_params params) {

    codec_dac & dac = *static_cast<codec_dac *>(ctx->model->impl);
    if (!dac.has_encoder) {
        codec_context_set_error(ctx, "model metadata indicates no encoder");
        return CODEC_STATUS_INVALID_STATE;
    }
    if (pcm.empty()) {
        codec_context_set_error(ctx, "empty pcm");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t model_n_q = std::max(1, dac.n_q);
    const int32_t use_n_q = params.n_q == 0 ? model_n_q : params.n_q;
    if (params.n_q < 0 || use_n_q < 1 || use_n_q > model_n_q) {
        codec_context_set_error(ctx, "DAC encode n_q must be 0 or in [1, model_n_q]");
        return CODEC_STATUS_INVALID_ARG;
    }

    const int32_t hop = std::max(1, params.hop_size > 0 ? params.hop_size : dac.hop_size);
    const int32_t n_in = (int32_t) pcm.size();

    const size_t mem = 32 * 1024 * 1024 + (size_t) n_in * sizeof(float) * 16;
    codec_graph_eval_guard eval_guard(ctx);
    std::string err;
    dac_encode_build build = {};
    if (!codec_dac_init_encode_build(ctx, n_in, use_n_q, &build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    codec_graph_cache_entry * entry = nullptr;
    err.clear();
    if (!codec_graph_cache_get_or_build(
            ctx,
            { CODEC_GRAPH_DAC_ENCODE, /*n_frames=*/0, /*n_q=*/use_n_q, /*hop=*/hop, /*n_in=*/n_in, /*latent_dim=*/build.hidden_dim },
            mem,
            codec_dac_build_encode,
            &build,
            sizeof(build),
            &entry,
            &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    lm_ggml_tensor * t_pcm = codec_graph_get_tensor(ctx, entry, "dac.encode.pcm");
    lm_ggml_tensor * t_out = codec_graph_get_tensor(ctx, entry, "dac.encode.out");
    if (t_pcm == nullptr || t_out == nullptr) {
        codec_context_set_error(ctx, "cached DAC encode graph is invalid");
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_graph_prepare_io(ctx, entry, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_runtime_write_tensor(t_pcm, pcm.data(), pcm.size() * sizeof(float), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    if (!codec_dac_write_encode_weights(ctx, entry, build, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_threads = ctx->model->n_threads > 0 ? ctx->model->n_threads : 1;
    if (!codec_graph_compute(ctx, entry, n_threads, &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    const int32_t n_frames = (int32_t) t_out->ne[0];
    const int32_t n_q = (int32_t) t_out->ne[1];
    std::vector<int32_t> tok((size_t) n_frames * (size_t) n_q, 0);
    if (!codec_runtime_read_tensor(t_out, tok.data(), tok.size() * sizeof(int32_t), &err)) {
        codec_context_set_error(ctx, err);
        return CODEC_STATUS_INTERNAL_ERROR;
    }

    int32_t * data = static_cast<int32_t *>(std::malloc((size_t) n_frames * (size_t) n_q * sizeof(int32_t)));
    if (data == nullptr) {
        codec_context_set_error(ctx, "failed to allocate token output");
        return CODEC_STATUS_INTERNAL_ERROR;
    }
    std::memcpy(data, tok.data(), tok.size() * sizeof(int32_t));

    codec_token_buffer_reset(out_tokens);
    out_tokens->data = data;
    out_tokens->n_tokens = n_frames * n_q;
    out_tokens->n_frames = n_frames;
    out_tokens->n_q = n_q;
    out_tokens->codebook_size = build.codebook_size;
    out_tokens->sample_rate = dac.sample_rate;
    out_tokens->hop_size = hop;

    if (out_latent != nullptr) {
        codec_latent_buffer_reset(out_latent);
    }

    return CODEC_STATUS_SUCCESS;
}
