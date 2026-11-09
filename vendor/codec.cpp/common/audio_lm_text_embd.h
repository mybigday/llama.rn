#ifndef CODEC_AUDIO_LM_TEXT_EMBD_H
#define CODEC_AUDIO_LM_TEXT_EMBD_H

// Internal: the backbone's text embedding table (`token_embd.weight`), read
// from the backbone GGUF for models whose per-step input adds a text lane the
// host cannot provide (llama.cpp exposes no raw embedding rows).  Rows are
// dequantized on demand through ggml's type traits (F32 / F16 / BF16 / quants).

#include <ggml.h>
#include <gguf.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace codec_common {

struct alm_text_embd {
    gguf_context * gg   = nullptr;
    ggml_context * meta = nullptr;
    const uint8_t * base = nullptr;
    std::vector<uint8_t> blob;
    int64_t hidden = 0;
    int64_t vocab  = 0;
    size_t  row_bytes = 0;
    ggml_to_float_t to_float = nullptr;

    alm_text_embd() = default;
    alm_text_embd(const alm_text_embd &) = delete;
    alm_text_embd & operator=(const alm_text_embd &) = delete;
    ~alm_text_embd() {
        if (gg)   gguf_free(gg);
        if (meta) ggml_free(meta);
    }

    bool loaded() const { return base != nullptr; }

    bool load(const char * path, int32_t want_hidden, std::string & err) {
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
        gg = gguf_init_from_file(path, gp);
        if (!gg) { err = "cannot read the backbone GGUF"; return false; }
        const int64_t tid = gguf_find_tensor(gg, "token_embd.weight");
        ggml_tensor * t = tid >= 0 ? ggml_get_tensor(meta, "token_embd.weight") : nullptr;
        if (!t) { err = "token_embd.weight not found in the backbone"; return false; }
        hidden = t->ne[0];
        vocab  = t->ne[1];
        if ((int32_t) hidden != want_hidden) { err = "backbone token_embd width does not match the codec"; return false; }
        const ggml_type_traits * tr = ggml_get_type_traits(t->type);
        to_float = tr ? tr->to_float : nullptr;
        if (!to_float) { err = "no to_float for the token_embd type"; return false; }
        row_bytes = ggml_row_size(t->type, hidden);
        FILE * f = std::fopen(path, "rb");
        if (!f) { err = "cannot open the backbone GGUF"; return false; }
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        blob.resize((size_t) sz);
        const size_t rd = std::fread(blob.data(), 1, (size_t) sz, f);
        std::fclose(f);
        if (rd != (size_t) sz) { err = "short read of the backbone GGUF"; return false; }
        base = blob.data() + gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, tid);
        return true;
    }

    // Row `token` dequantized into `out` (hidden floats).
    bool row(int32_t token, float * out) const {
        if (token < 0 || token >= (int32_t) vocab || !base) return false;
        to_float(base + (size_t) token * row_bytes, out, hidden);
        return true;
    }
};

}  // namespace codec_common

#endif  // CODEC_AUDIO_LM_TEXT_EMBD_H
