// tts_runner.cpp — full reference TTS host loop (backbone-driven).
//
// This is the OPTIONAL reference loop layer described in tts_runner.h: it
// LINKS the isolated llama.cpp backbone (libttsbackbone) and owns the whole
// synthesize flow — backbone load, tokenize/prefill, every per-model flow,
// sampling, CFG pair handling, streaming interleave, embed injection, EOS
// handling, and codes→PCM decode.  codec_common's per-step hooks (audio_lm_*)
// are unchanged and composed here; the loop-owning contract for hosts like
// llama.rn (docs/codec_common_api.md §Boundary) is untouched.
//
// Ported from examples/tts-cli.cpp's cmd_synthesize + the run_* flow
// helpers; the flow bodies are verbatim, only the entry points fill a
// tts_runner_result (PCM + stats) instead of writing WAV files.  Built only
// when CODEC_TTS_BACKBONE=ON.

#include "llama.h"   // must precede tts_runner.h (defines LLAMA_H guard)

#include "tts_runner.h"

#include <type_traits>

// The inline llama_logit_bias fallback in tts_runner.h (for no-llama TUs) is
// only ABI-safe while this holds.  Assert it in the TU that sees the real type.
static_assert(sizeof(llama_logit_bias) == 8,
              "llama_logit_bias layout drift breaks tts_runner.h's fallback typedef");
static_assert(std::is_same<llama_token, int32_t>::value,
              "llama_logit_bias::token must be int32_t for the fallback typedef");

#include "codec_common.h"
#include "utils/wav_io.h"
#include "common.h"     // common_params_sampling, common_grammar
#include "sampling.h"   // common_sampler_*
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace codec_common {

// ─────────────────────────────────────────────────────────────────────────────
// Flat audio-token logit-bias mask (Type A / direct-audio backbone models).
// ─────────────────────────────────────────────────────────────────────────────

std::vector<llama_logit_bias> build_flat_audio_mask(int32_t offset, int32_t count,
                                                    int32_t eos_id, int32_t n_vocab) {
    std::vector<llama_logit_bias> out;
    if (offset < 0 || count <= 0 || n_vocab <= 0) return out;
    const int32_t lo = offset;
    const int32_t hi = offset + count;                 // exclusive
    out.reserve((size_t) n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        const bool allowed = (id >= lo && id < hi) || (eos_id >= 0 && id == eos_id);
        if (!allowed) out.push_back(llama_logit_bias{ id, -INFINITY });
    }
    return out;
}

namespace {

// ── Backbone text-embedding table reader ──────────────────────────────
// MOSS-TTS-Realtime composes each backbone-step input as
//   text_embd[text_token] + compose_audio_embd(prev_frame_codes)
// where the audio part lives in the codec_lm but the TEXT embedding table
// (`token_embd.weight`, [hidden, V_text]) lives in the backbone GGUF.
// llama.cpp exposes no raw-embedding API, so we mmap the backbone GGUF a
// second time and dequant embedding rows on demand via ggml type traits
// (handles bf16 / f16 / quantised transparently).
struct TextEmbdTable {
    gguf_context * gg   = nullptr;
    ggml_context * meta = nullptr;   // holds tensor metadata (no_alloc)
    const uint8_t * base = nullptr;  // mmapped tensor-data region
    std::vector<uint8_t> blob;       // owns the file bytes
    int64_t hidden = 0;
    int64_t vocab  = 0;
    ggml_type type = GGML_TYPE_F32;
    size_t row_bytes = 0;
    ggml_to_float_t to_float = nullptr;

    bool load(const char * path, int32_t want_hidden, std::string & err) {
        gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
        gg = gguf_init_from_file(path, gp);
        if (!gg) { err = "gguf_init_from_file failed"; return false; }
        const int64_t tid = gguf_find_tensor(gg, "token_embd.weight");
        if (tid < 0) { err = "token_embd.weight not found in backbone"; return false; }
        ggml_tensor * t = ggml_get_tensor(meta, "token_embd.weight");
        if (!t) { err = "token_embd metadata lookup failed"; return false; }
        hidden = t->ne[0];
        vocab  = t->ne[1];
        type   = t->type;
        if ((int32_t) hidden != want_hidden) {
            err = "token_embd hidden mismatch"; return false;
        }
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        to_float = tr ? tr->to_float : nullptr;
        // For a quantised type to_float works on a whole row (k = hidden,
        // which must be a multiple of block size for legal types).
        if (!to_float) { err = "no to_float for token_embd type"; return false; }
        row_bytes = ggml_row_size(type, hidden);

        // Read the whole file into `blob`, then point `base` at the tensor
        // data region (data_offset within the file).
        FILE * f = std::fopen(path, "rb");
        if (!f) { err = "fopen backbone failed"; return false; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        blob.resize((size_t) sz);
        size_t rd = std::fread(blob.data(), 1, (size_t) sz, f);
        std::fclose(f);
        if (rd != (size_t) sz) { err = "backbone read short"; return false; }
        const size_t data_off = gguf_get_data_offset(gg);
        const size_t t_off     = gguf_get_tensor_offset(gg, tid);
        base = blob.data() + data_off + t_off;
        return true;
    }

    // Dequant embedding row `token` into `out` (hidden floats).
    bool row(int32_t token, float * out) const {
        if (token < 0 || token >= (int32_t) vocab || !base || !to_float) return false;
        const void * src = base + (size_t) token * row_bytes;
        to_float(src, out, hidden);
        return true;
    }

    ~TextEmbdTable() {
        if (gg)   gguf_free(gg);
        if (meta) ggml_free(meta);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Two sampler families, split by logits SOURCE:
//
//  * BackboneSampler (common_sampler) — for logits produced by llama_decode
//    and read via llama_get_logits_ith(lctx, idx).  This is the cb0-from-
//    backbone token (MOSS-TTSD merged text+speech vocab).  We use llama.cpp's
//    `common` sampling layer so we don't reimplement chain assembly, grammar
//    apply/resample, or penalty bookkeeping.  Grammar (GBNF) only makes sense
//    here — it constrains real backbone-vocab tokens.
//
//  * SamplerChain (raw llama_sampler over float*) — for logits that are NOT
//    tied to a llama_context: codec_lm audio-codebook heads (residual depth
//    decoder, Chatterbox speech_head) and the LFM2 recomputed text logits
//    (llama.cpp omits the output head when embeddings=true).  common_sampler
//    can't read those (it calls llama_get_logits_ith internally), so they keep
//    the raw chain.  Grammar never applies to audio codebooks.
// ═══════════════════════════════════════════════════════════════════════

// ── common_sampler wrapper for BACKBONE logits ────────────────────────
// Owns a common_sampler built from a common_params_sampling that mirrors the
// old raw-chain knobs (temp / top-k / top-p / min-p / rep-penalty), plus an
// optional GBNF grammar.  Samples from a llama_context position via
// common_sampler_sample (which reads llama_get_logits_ith + applies the
// chain + grammar apply/resample) and accepts the token (penalty + grammar
// state).  build() returns false + err on a grammar parse failure (clean
// error, not a crash).
struct BackboneSampler {
    common_sampler * smpl = nullptr;

    BackboneSampler() = default;
    BackboneSampler(const BackboneSampler &) = delete;
    BackboneSampler & operator=(const BackboneSampler &) = delete;
    ~BackboneSampler() { if (smpl) common_sampler_free(smpl); }

    bool build(const llama_model * model, uint32_t seed, float temp,
               int32_t top_k, float top_p, float min_p, float rep_penalty,
               int32_t rep_last_n, const std::string & grammar,
               const std::vector<llama_logit_bias> & logit_bias,
               std::string * err) {
        common_params_sampling sp;
        sp.seed          = seed;
        sp.no_perf       = true;
        sp.temp          = temp;                         // <=0 → greedy
        sp.top_k         = top_k > 0 ? top_k : 0;        // 0 = disabled (vocab)
        sp.top_p         = (top_p > 0.0f && top_p < 1.0f) ? top_p : 1.0f;
        sp.min_p         = min_p > 0.0f ? min_p : 0.0f;
        sp.penalty_repeat = rep_penalty;                 // 1.0 = disabled
        sp.penalty_last_n = rep_penalty != 1.0f ? (rep_last_n > 0 ? rep_last_n : -1) : 0;
        sp.penalty_freq   = 0.0f;
        sp.penalty_present = 0.0f;
        // Reduce the chain to exactly the reference warpers (no DRY / XTC /
        // typical / top-n-sigma), matching the old SamplerChain order:
        //   penalties → temp → top_k → min_p → top_p → dist.
        sp.samplers = {
            COMMON_SAMPLER_TYPE_PENALTIES,
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_MIN_P,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_TEMPERATURE,
        };
        if (!grammar.empty()) {
            sp.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
        }
        if (!logit_bias.empty()) {
            sp.logit_bias = logit_bias;
        }
        try {
            smpl = common_sampler_init(model, sp);
        } catch (const std::exception & e) {
            if (err) *err = std::string("grammar/sampler init failed: ") + e.what();
            smpl = nullptr;
            return false;
        }
        if (!smpl) {
            if (err) *err = "common_sampler_init returned null (bad grammar?)";
            return false;
        }
        return true;
    }

    // Sample the backbone token at context position `idx` (usually -1),
    // apply the grammar (if any), and accept it (penalty + grammar state).
    llama_token sample(llama_context * lctx, int32_t idx) {
        const llama_token id = common_sampler_sample(smpl, lctx, idx, /*grammar_first=*/false);
        common_sampler_accept(smpl, id, /*is_generated=*/true);
        return id;
    }
};

// ── llama.cpp sampler-chain wrapper (codec_lm / recomputed logits) ─────
// A SamplerChain owns one llama_sampler chain and drives it over RAW
// float* logits (from a codec_lm head or the LFM2 recomputed text logits)
// via llama_token_data_array — the sampler API operates on arbitrary
// logits and never needs a vocab handle for temp/top-k/top-p/min-p/
// penalties/dist/greedy.
//
// We wrap each logit as {id=index, logit, p=0}; after llama_sampler_apply
// the chosen token is data[cur_p.selected].id (== the code index, since
// samplers reorder/shrink the array but preserve ids).  For penalty-based
// chains we call llama_sampler_accept(sampled) to maintain the ring-buffer
// window (llama.cpp's penalties sampler needs accept per emitted token).
//
// Semantics verified against the pinned llama.cpp llama-sampler.cpp:
//  * penalties: logit<=0 ? *repeat : /repeat  (freq=present=0) — matches
//    the HF RepetitionPenaltyLogitsProcessor / apply_repetition_penalty
//    convention used by chatterbox-T3 and MOSS-realtime.
//  * temp: divides logits by t (no softmax); top-k/top-p/min-p each
//    recompute softmax internally, mirroring HF's per-warper softmax.
//  * greedy: strict-`>` argmax from index 0 — byte-identical to the old
//    greedy path (needed for the CSM greedy WAV parity).
struct SamplerChain {
    llama_sampler * chain = nullptr;
    bool has_penalties = false;
    std::vector<llama_token_data> buf;

    SamplerChain() = default;
    SamplerChain(const SamplerChain &) = delete;
    SamplerChain & operator=(const SamplerChain &) = delete;
    ~SamplerChain() { if (chain) llama_sampler_free(chain); }

    // Greedy chain (temp<=0): argmax only, no RNG.
    void init_greedy() {
        reset();
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        chain = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        has_penalties = false;
    }

    // Sampled chain in the reference order.  Any argument left at its
    // "disabled" sentinel (penalty==1, top_k<=0, top_p>=1, min_p<=0)
    // becomes a no-op in llama.cpp, so a single builder serves every flow.
    //   penalties → temp → top_k → min_p → top_p → dist(seed)
    // The min_p-before-top_p order matches chatterbox-T3's reference
    // (min_p_warper then top_p_warper); realtime leaves min_p=0 (noop) so it
    // reduces to temp → top_k → top_p, mirroring the old Sampler which
    // truncated to top_k on the sorted list then applied top_p within it.
    void init_sampled(uint32_t seed, float temp, int32_t top_k, float top_p,
                      float min_p, float rep_penalty, int32_t rep_last_n) {
        reset();
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        chain = llama_sampler_chain_init(sp);
        if (rep_penalty != 1.0f) {
            const int32_t last_n = rep_last_n > 0 ? rep_last_n : -1;  // -1 = full history
            llama_sampler_chain_add(chain,
                llama_sampler_init_penalties(last_n, rep_penalty, 0.0f, 0.0f));
            has_penalties = true;
        }
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        if (top_k > 0)               llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        if (min_p > 0.0f)            llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
        if (top_p > 0.0f && top_p < 1.0f)
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    void reset() {
        if (chain) { llama_sampler_free(chain); chain = nullptr; }
        has_penalties = false;
    }

    // Sample one token from raw logits[0..n).  Returns the chosen index.
    int32_t sample(const float * logits, int32_t n) {
        if (n <= 0 || !chain) return 0;
        buf.resize((size_t) n);
        for (int32_t i = 0; i < n; ++i) buf[(size_t) i] = { (llama_token) i, logits[i], 0.0f };
        llama_token_data_array cur = { buf.data(), (size_t) n, -1, false };
        llama_sampler_apply(chain, &cur);
        const int64_t sel = cur.selected >= 0 ? cur.selected : 0;
        const llama_token id = cur.data[sel].id;
        llama_sampler_accept(chain, id);  // maintain penalty window (no-op otherwise)
        return (int32_t) id;
    }
};

// Linear-interpolation resample of mono F32 PCM from `in_sr` to `out_sr`.
// Shared by every speaker-encoder path in the runner (Chatterbox VE @ 16 kHz
// and, via audio_lm_build_prompt, the ECAPA-TDNN @ 24 kHz path).  Each
// encoder declares its working rate; the runner feeds PCM at that rate.
std::vector<float> resample_mono_f32(const std::vector<float> & in,
                                     int32_t in_sr, int32_t out_sr) {
    if (in.empty() || in_sr <= 0 || out_sr <= 0 || in_sr == out_sr) return in;
    const int64_t n_in  = (int64_t) in.size();
    const int64_t n_out = n_in * out_sr / in_sr;
    std::vector<float> out((size_t) std::max<int64_t>(n_out, 1));
    for (int64_t i = 0; i < (int64_t) out.size(); ++i) {
        const double src = (double) i * in_sr / out_sr;
        int64_t i0 = (int64_t) src;
        const double f = src - (double) i0;
        const float a0 = in[(size_t) std::min<int64_t>(i0,     n_in - 1)];
        const float a1 = in[(size_t) std::min<int64_t>(i0 + 1, n_in - 1)];
        out[(size_t) i] = (float) ((double) a0 * (1.0 - f) + (double) a1 * f);
    }
    return out;
}

// Load ref audio (mono F32) from `path` into `ref_pcm`; fills the geometry
// out-params.  Returns true (and leaves ref_pcm empty) when path is empty.
bool load_ref_audio(const std::string & path, std::vector<float> & ref_pcm,
                    int32_t * out_n, int32_t * out_sr, std::string * err) {
    if (path.empty()) return true;
    codec_example_wav_data w;
    std::string werr;
    if (!codec_example_load_wav_pcm16(path.c_str(), &w, &werr)) {
        *err = "failed to load " + path + ": " + werr;
        return false;
    }
    const int32_t nch = w.n_channels > 0 ? w.n_channels : 1;
    const int32_t nframes = (int32_t) (w.pcm_i16.size() / (size_t) nch);
    ref_pcm.assign((size_t) nframes, 0.0f);
    for (int32_t i = 0; i < nframes; ++i) {
        float acc = 0.0f;
        for (int32_t c = 0; c < nch; ++c) acc += w.pcm_i16[(size_t) i * nch + c] / 32768.0f;
        ref_pcm[(size_t) i] = acc / (float) nch;
    }
    if (out_n)  *out_n  = (int32_t) ref_pcm.size();
    if (out_sr) *out_sr = w.sample_rate;
    return true;
}

// Replace all occurrences of `from` with `to` in `s`.
std::string replace_all_str(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Tokenize a raw string with explicit bos/special controls.
std::vector<llama_token> tokenize_str(const llama_vocab * vocab,
                                      const std::string & s,
                                      bool add_bos, bool parse_special) {
    int32_t cap = (int32_t) s.size() + 8;
    std::vector<llama_token> toks(cap);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), cap, add_bos, parse_special);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), (int32_t) toks.size(), add_bos, parse_special); }
    toks.resize(std::max(0, n));
    return toks;
}

std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab,
                                         const audio_lm_prompt_info & pi,
                                         const std::string & text_in) {
    std::string text = text_in;
    // MOSS-TTSD dialogue tags: the processor maps [S1]/[S2] → <speaker1>/
    // <speaker2> before tokenizing (see processing_moss_ttsd prepare_sample).
    if (pi.model_kind == audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY) {
        text = replace_all_str(text, "[S1]", "<speaker1>");
        text = replace_all_str(text, "[S2]", "<speaker2>");
    }
    const std::string full = pi.prompt_prefix + text + pi.prompt_suffix;
    int32_t cap = (int32_t) full.size() + 8;
    std::vector<llama_token> toks(cap);
    int32_t n = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                               toks.data(), cap, pi.add_bos, pi.parse_special);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                               toks.data(), (int32_t) toks.size(), pi.add_bos, pi.parse_special); }
    toks.resize(std::max(0, n));
    return toks;
}

// Decode a token batch, requesting per-position embeddings (logits at
// every position when `all_pos` else only last).
bool decode_tokens(llama_context * lctx, const std::vector<llama_token> & toks,
                   int32_t n_past, bool all_pos) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = n_past + (int32_t) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (all_pos || i == toks.size() - 1) ? 1 : 0;
    }
    b.n_tokens = (int32_t) toks.size();
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Decode a single embedding vector (inputs_embeds path).
bool decode_embed(llama_context * lctx, const float * embd, int32_t dim, int32_t n_past) {
    llama_batch b = llama_batch_init(1, dim, 1);
    std::memcpy(b.embd, embd, (size_t) dim * sizeof(float));
    b.token = nullptr;
    b.pos[0] = n_past;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    b.n_tokens = 1;
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Decode a contiguous block of `n` inputs_embeds rows (single sequence),
// flagging only the last for logits.  Rows are `n * dim` floats.
bool decode_embed_block(llama_context * lctx, const float * embds, int32_t dim,
                        int32_t n, int32_t n_past) {
    if (n <= 0) return true;
    llama_batch b = llama_batch_init(n, dim, 1);
    std::memcpy(b.embd, embds, (size_t) n * dim * sizeof(float));
    b.token = nullptr;
    for (int32_t i = 0; i < n; ++i) {
        b.pos[i] = n_past + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (i == n - 1) ? 1 : 0;
    }
    b.n_tokens = n;
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Decode a batch of `n_seq` embed rows (one per CFG lane) at a single
// position, requesting logits at the last (only) row of each lane.
bool decode_embed_batch(llama_context * lctx, const float * embds, int32_t dim,
                        int32_t n_seq, int32_t pos) {
    llama_batch b = llama_batch_init(n_seq, dim, 1);
    std::memcpy(b.embd, embds, (size_t) n_seq * dim * sizeof(float));
    b.token = nullptr;
    b.n_tokens = n_seq;
    for (int32_t s = 0; s < n_seq; ++s) {
        b.pos[s] = pos;
        b.n_seq_id[s] = 1;
        b.seq_id[s][0] = s;
        b.logits[s] = 1;
    }
    int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Flow 1 — continuous CFM (BlueMagpie).
bool run_continuous(audio_lm_context * ctx, llama_context * lctx,
                    const std::vector<llama_token> & toks, int32_t hidden,
                    int32_t max_frames, const tts_runner_params & a,
                    int32_t * out_frames, const char ** out_stop) {
    audio_lm_set_continuous_params(ctx, a.cfg, a.timesteps, a.min_len);
    if (!decode_tokens(lctx, toks, 0, /*all_pos=*/true)) {
        std::fprintf(stderr, "prefill decode failed\n"); return false;
    }
    const int32_t np = (int32_t) toks.size();
    std::vector<float> hid((size_t) np * hidden);
    for (int32_t i = 0; i < np; ++i) {
        const float * h = llama_get_embeddings_ith(lctx, i);
        if (!h) { std::fprintf(stderr, "no embeddings at pos %d\n", i); return false; }
        std::memcpy(hid.data() + (size_t) i * hidden, h, (size_t) hidden * sizeof(float));
    }
    int32_t n_past = np;
    if (!audio_lm_text_prefill(ctx, hid.data(), np, hidden)) return false;

    std::vector<float> cur(hid.end() - hidden, hid.end());
    for (int32_t step = 0; step < max_frames; ++step) {
        auto act = audio_lm_observe_hidden(ctx, cur.data(), hidden, nullptr);
        if (act == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "stop_head"; break;
        }
        (*out_frames)++;
        int32_t dim = 0;
        const float * fb = audio_lm_get_next_embed(ctx, &dim);
        if (!fb || dim != hidden) return false;
        std::vector<float> fbc(fb, fb + dim);
        if (!decode_embed(lctx, fbc.data(), dim, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Flow 3-streaming — MOSS-TTS-Realtime.
bool run_realtime_streaming(audio_lm_context * ctx,
                            llama_context * lctx, const llama_vocab * vocab,
                            const audio_lm_prompt_info & pi,
                            const TextEmbdTable & tetab,
                            const std::string & payload_text,
                            int32_t hidden, int32_t n_cb, int32_t max_frames,
                            uint32_t seed, float temp, float top_p,
                            int32_t top_k, float rep_penalty, int32_t rep_window,
                            int32_t * out_frames, const char ** out_stop) {
    audio_lm_set_uses_embed_override(ctx, true, 1);

    std::vector<llama_token> ctx_toks =
        tokenize_str(vocab, pi.prompt_prefix + pi.prompt_suffix,
                     pi.add_bos, pi.parse_special);
    std::vector<llama_token> text_toks =
        tokenize_str(vocab, payload_text, /*add_bos=*/false, /*parse_special=*/false);
    if (ctx_toks.empty() || text_toks.empty()) {
        std::fprintf(stderr, "realtime: empty context or text tokens\n");
        return false;
    }

    const int32_t audio_pad = pi.audio_pad_code;
    const int32_t bos_c0    = pi.bos_code_c0;
    const int32_t text_pad  = pi.text_pad_id;
    const int32_t prefill_n = std::min<int32_t>(pi.prefill_text_len,
                                                (int32_t) text_toks.size());

    std::vector<int32_t> pad_codes((size_t) n_cb, audio_pad);
    auto compose_row = [&](int32_t text_tok, const int32_t * codes,
                           float * dst) -> bool {
        if (!tetab.row(text_tok, dst)) {
            std::fprintf(stderr, "realtime: text_embd row %d failed\n", text_tok);
            return false;
        }
        std::vector<float> aud((size_t) hidden, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(
                ctx, codes, n_cb, aud.data(), hidden)) {
            std::fprintf(stderr, "realtime: compose_audio failed: %s\n",
                         audio_lm_last_error(ctx));
            return false;
        }
        for (int32_t i = 0; i < hidden; ++i) dst[i] += aud[i];
        return true;
    };

    const int32_t n_rows = (int32_t) ctx_toks.size() + prefill_n;
    std::vector<float> block((size_t) n_rows * hidden);
    int32_t r = 0;
    for (size_t i = 0; i < ctx_toks.size(); ++i, ++r) {
        if (!compose_row(ctx_toks[i], pad_codes.data(),
                         block.data() + (size_t) r * hidden)) return false;
    }
    for (int32_t i = 0; i < prefill_n; ++i, ++r) {
        std::vector<int32_t> codes = pad_codes;
        if (i == prefill_n - 1) codes[0] = bos_c0;
        if (!compose_row(text_toks[(size_t) i], codes.data(),
                         block.data() + (size_t) r * hidden)) return false;
    }
    if (!decode_embed_block(lctx, block.data(), hidden, n_rows, 0)) {
        std::fprintf(stderr, "realtime: prefill decode failed\n");
        return false;
    }
    int32_t n_past = n_rows;

    std::vector<float> cur(hidden);
    {
        const float * h0 = llama_get_embeddings_ith(lctx, -1);
        if (!h0) return false;
        std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));
    }

    // Per-codebook sampler chains: each carries its own penalty ring-buffer
    // (window = rep_window) so the CTRL-style repetition penalty is applied
    // per codebook exactly like the old windowed apply_rep_penalty.
    std::vector<std::unique_ptr<SamplerChain>> cb_smpl((size_t) n_cb);
    for (int32_t cb = 0; cb < n_cb; ++cb) {
        cb_smpl[(size_t) cb] = std::make_unique<SamplerChain>();
        if (temp <= 0.0f) cb_smpl[(size_t) cb]->init_greedy();
        else cb_smpl[(size_t) cb]->init_sampled(seed, temp, top_k, top_p,
                                                /*min_p=*/0.0f, rep_penalty, rep_window);
    }

    int32_t text_idx = prefill_n;
    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        if (!audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code = cb_smpl[(size_t) cb]->sample(lg, nlog);
            if (!audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        auto act = audio_lm_observe_codes(ctx, codes.data(), n_cb,
                                          cur.data(), hidden);
        if (act == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0";
            break;
        }
        (*out_frames)++;

        int32_t text_tok = (text_idx < (int32_t) text_toks.size())
                         ? text_toks[(size_t) text_idx] : text_pad;
        ++text_idx;
        std::vector<float> row(hidden);
        if (!compose_row(text_tok, codes.data(), row.data())) return false;
        if (!decode_embed(lctx, row.data(), hidden, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Flow 5 — LFM2-Audio sequential text→audio TTS.
bool run_lfm2_sequential(audio_lm_context * ctx, llama_context * lctx,
                         const llama_vocab * vocab,
                         const audio_lm_prompt_info & pi,
                         const TextEmbdTable & tetab,
                         const std::vector<llama_token> & toks, int32_t hidden,
                         int32_t n_cb, int32_t max_frames, uint32_t seed,
                         float temp, float top_p, int32_t top_k,
                         int32_t * out_frames, const char ** out_stop) {
    audio_lm_set_uses_embed_override(ctx, true, 1);

    if (!decode_tokens(lctx, toks, 0, /*all_pos=*/false)) {
        std::fprintf(stderr, "lfm2: prefill decode failed\n");
        return false;
    }
    int32_t n_past = (int32_t) toks.size();
    const int32_t n_vocab = (int32_t) tetab.vocab;

    std::vector<float> tlog((size_t) n_vocab);
    std::vector<float> erow((size_t) hidden);
    auto text_logits = [&](const float * h) -> const float * {
        for (int32_t v = 0; v < n_vocab; ++v) {
            if (!tetab.row(v, erow.data())) { tlog[v] = -1e30f; continue; }
            double acc = 0.0;
            for (int32_t i = 0; i < hidden; ++i) acc += (double) h[i] * (double) erow[i];
            tlog[v] = (float) acc;
        }
        return tlog.data();
    };

    (void) vocab;
    // One chain (no rep penalty) drives both the text warm-up and the audio
    // codebooks, matching the old single-RNG-stream Sampler.
    SamplerChain smpl;
    if (temp <= 0.0f) smpl.init_greedy();
    else smpl.init_sampled(seed, temp, top_k, top_p, /*min_p=*/0.0f,
                           /*rep_penalty=*/1.0f, /*rep_last_n=*/0);

    for (int32_t t = 0; t < pi.max_text_tokens; ++t) {
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) { std::fprintf(stderr, "lfm2: no hidden for text logits\n"); return false; }
        const float * bl = text_logits(h);
        int32_t tok = smpl.sample(bl, n_vocab);
        if (tok == pi.audio_start_id) break;
        if (tok == pi.text_end_id)    { *out_stop = "text_end"; return true; }
        std::vector<llama_token> one(1, (llama_token) tok);
        if (!decode_tokens(lctx, one, n_past++, /*all_pos=*/false)) {
            std::fprintf(stderr, "lfm2: text step decode failed\n");
            return false;
        }
    }
    {
        std::vector<llama_token> as(1, (llama_token) pi.audio_start_id);
        if (!decode_tokens(lctx, as, n_past++, /*all_pos=*/false)) return false;
    }

    std::vector<float> cur(hidden);
    const float * h0 = llama_get_embeddings_ith(lctx, -1);
    if (!h0) return false;
    std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));

    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        if (!audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code = smpl.sample(lg, nlog);
            if (!audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        auto act = audio_lm_observe_codes(ctx, codes.data(), n_cb,
                                          cur.data(), hidden);
        if (act == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0";
            break;
        }
        (*out_frames)++;

        std::vector<float> row(hidden, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(
                ctx, codes.data(), n_cb, row.data(), hidden)) {
            std::fprintf(stderr, "lfm2: compose_audio failed: %s\n",
                         audio_lm_last_error(ctx));
            return false;
        }
        if (!decode_embed(lctx, row.data(), hidden, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// Flow 2/3 — codebook AR.
bool run_codebook_ar(audio_lm_context * ctx, llama_context * lctx,
                     const llama_model * lmodel, const llama_vocab * vocab,
                     const audio_lm_prompt_info & pi,
                     const std::vector<llama_token> & toks, int32_t hidden, int32_t n_cb,
                     int32_t max_frames, uint32_t seed,
                     float temp, float top_p, int32_t top_k,
                     const tts_constraint & constraint,
                     const std::vector<float> & speaker_prefix,
                     const std::string & payload_text,
                     int32_t * out_frames, const char ** out_stop) {
    // A flat Type A logit-bias mask is only consumed by the cb0-from-backbone
    // sampler path.  Type A single-cb decode is not wired up yet (no such model
    // ships), so a non-empty mask here would be silently dropped → unconstrained
    // audio.  Make that loud rather than silent.
    if (!constraint.logit_bias.empty() && !pi.cb0_from_backbone) {
        std::fprintf(stderr,
            "WARN: flat Type A logit-bias mask (%zu masked tokens) computed but the "
            "token-single-cb decode path is not wired; audio will be UNCONSTRAINED.\n",
            constraint.logit_bias.size());
    }
    audio_lm_set_uses_embed_override(ctx, true, 1);
    int32_t n_past = 0;

    std::vector<llama_token> talker_text;
    int32_t talker_trailing = 0;
    const bool talker = audio_lm_talker_has_projection(ctx);
    if (talker) {
        std::vector<llama_token> role =
            tokenize_str(vocab, "<|im_start|>assistant\n", /*add_bos=*/false,
                         /*parse_special=*/true);
        talker_text = tokenize_str(vocab, payload_text,
                                   /*add_bos=*/false, /*parse_special=*/false);
        if (talker_text.empty()) { std::fprintf(stderr, "talker: empty text\n"); return false; }

        const int32_t cap_rows = (int32_t) role.size() + 6 + 4;
        std::vector<float> prefix((size_t) cap_rows * hidden);
        int32_t n_rows = 0, consumed = 0;
        const float * xv = (!speaker_prefix.empty() &&
                            (int32_t) speaker_prefix.size() == hidden)
                         ? speaker_prefix.data() : nullptr;
        if (!audio_lm_build_talker_prefix(
                ctx, role.data(), (int32_t) role.size(),
                talker_text.data(), (int32_t) talker_text.size(),
                xv, xv ? hidden : 0,
                prefix.data(), cap_rows, &n_rows, &consumed)) {
            std::fprintf(stderr, "build_talker_prefix failed: %s\n",
                         audio_lm_last_error(ctx));
            return false;
        }
        talker_trailing = 0;
        llama_batch b = llama_batch_init(n_rows, hidden, 1);
        std::memcpy(b.embd, prefix.data(), (size_t) n_rows * hidden * sizeof(float));
        b.token = nullptr; b.n_tokens = n_rows;
        for (int32_t i = 0; i < n_rows; ++i) {
            b.pos[i] = n_past + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_rows - 1) ? 1 : 0;
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "talker prefill decode failed\n"); return false; }
        n_past += n_rows;
    } else
    if (!speaker_prefix.empty() && (int32_t) speaker_prefix.size() == hidden) {
        std::vector<float> pfx(speaker_prefix);
        if (!decode_embed(lctx, pfx.data(), hidden, n_past)) {
            std::fprintf(stderr, "speaker prefix decode failed\n"); return false;
        }
        n_past += 1;
    }
    if (!talker) {
    if (audio_lm_prompt_needs_composed_embd(ctx)) {
        std::vector<float> prompt_embd((size_t) toks.size() * hidden);
        for (size_t i = 0; i < toks.size(); ++i) {
            if (!audio_lm_compose_prompt_embd(
                    ctx, toks[i], prompt_embd.data() + i * hidden, hidden)) {
                std::fprintf(stderr, "compose_prompt_embd failed: %s\n",
                             audio_lm_last_error(ctx));
                return false;
            }
        }
        llama_batch b = llama_batch_init((int32_t) toks.size(), hidden, 1);
        std::memcpy(b.embd, prompt_embd.data(), prompt_embd.size() * sizeof(float));
        b.token = nullptr;
        b.n_tokens = (int32_t) toks.size();
        for (size_t i = 0; i < toks.size(); ++i) {
            b.pos[i] = n_past + (int32_t) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == toks.size() - 1) ? 1 : 0;
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "prefill (composed) decode failed\n"); return false; }
    } else if (!decode_tokens(lctx, toks, n_past, /*all_pos=*/false)) {
        std::fprintf(stderr, "prefill decode failed\n"); return false;
    }
    n_past += (int32_t) toks.size();
    }  // end if (!talker)

    std::vector<float> cur(hidden);
    const float * h0 = llama_get_embeddings_ith(lctx, -1);
    if (!h0) return false;
    std::memcpy(cur.data(), h0, (size_t) hidden * sizeof(float));

    // Raw chain (no rep penalty) for the codec_lm audio codebook heads —
    // arbitrary float arrays, no llama_context / grammar.  Greedy when
    // temp<=0 so the CSM greedy path stays byte-identical.
    SamplerChain smpl;
    if (temp <= 0.0f) smpl.init_greedy();
    else smpl.init_sampled(seed, temp, top_k, top_p, /*min_p=*/0.0f,
                           /*rep_penalty=*/1.0f, /*rep_last_n=*/0);

    // cb0-from-backbone (MOSS-TTSD): sampled from the backbone's own logits
    // via llama.cpp's common_sampler, with the optional GBNF grammar attached
    // (constrains cb0 to the speech range ∪ eos).  Only built when needed.
    BackboneSampler bbsmpl;
    if (pi.cb0_from_backbone) {
        std::string berr;
        if (!bbsmpl.build(lmodel, seed, temp, top_k, top_p, /*min_p=*/0.0f,
                          /*rep_penalty=*/1.0f, /*rep_last_n=*/0,
                          constraint.grammar, constraint.logit_bias, &berr)) {
            std::fprintf(stderr, "backbone sampler init failed: %s\n", berr.c_str());
            *out_stop = "grammar_error";
            return false;
        }
    }

    std::vector<int32_t> codes(n_cb);
    for (int32_t step = 0; step < max_frames; ++step) {
        if (pi.cb0_from_backbone) {
            const float * bl = llama_get_logits_ith(lctx, -1);
            if (!bl) return false;
            int32_t c0 = bbsmpl.sample(lctx, -1);
            if (!audio_lm_step_set_text_context(ctx, c0)) return false;
            codes[0] = c0;
        }
        if (!audio_lm_step_begin(ctx, cur.data(), hidden)) return false;
        for (int32_t cb = 0; cb < n_cb; ++cb) {
            int32_t idx = 0, nlog = 0;
            const float * lg = audio_lm_step_logits(ctx, &idx, &nlog);
            if (!lg) return false;
            int32_t code = (pi.cb0_from_backbone && cb == 0)
                         ? codes[0]
                         : smpl.sample(lg, nlog);
            if (!audio_lm_step_push_code(ctx, code)) return false;
        }
        if (!audio_lm_step_finish(ctx, codes.data(), n_cb)) return false;

        auto act = audio_lm_observe_codes(ctx, codes.data(), n_cb, cur.data(), hidden);
        if (act == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(ctx);
            if (e && *e) return false;
            *out_stop = "eos_code_c0"; break;
        }
        (*out_frames)++;
        int32_t dim = 0;
        const float * nb = audio_lm_get_next_embed(ctx, &dim);
        if (!nb || dim != hidden) return false;
        std::vector<float> nbc(nb, nb + dim);
        if (talker) {
            std::vector<float> tt(hidden);
            if (!audio_lm_talker_trailing_text_embd(
                    ctx, talker_text.data(), (int32_t) talker_text.size(),
                    talker_trailing, tt.data(), hidden)) {
                std::fprintf(stderr, "talker trailing text failed: %s\n",
                             audio_lm_last_error(ctx));
                return false;
            }
            for (int32_t i = 0; i < hidden; ++i) nbc[i] += tt[i];
            ++talker_trailing;
        }
        if (!decode_embed(lctx, nbc.data(), dim, n_past++)) return false;
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (!h) return false;
        std::memcpy(cur.data(), h, (size_t) hidden * sizeof(float));
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Type A single-codebook token AR decode (KIND_TOKEN_SINGLE_CB).
//
// NeuTTS-family backbones emit `<|speech_N|>` tokens DIRECTLY from the backbone
// lm_head — there is no codec_lm depth decoder, no codebook heads, and no embed
// feedback.  The loop is a plain token-in/token-out AR over the backbone vocab,
// constrained to the speech range ∪ eos via a flat logit-bias mask.  Each
// sampled speech token is observed (which internally accumulates the FSQ code
// into `ctx->codes`), then fed straight back as a 1-token batch.  The model
// stops when it emits `<|SPEECH_GENERATION_END|>` (OBSERVE_STOP).
//
// Contract (verified in audio_lm.cpp):
//   * `audio_lm_observe_token` appends the code to `ctx->codes` internally and
//     returns OBSERVE_CONSUMED (Type A; uses_embed_override left false), so the
//     caller runs `audio_lm_decode_audio` directly with NO manual push.
//   * `build_flat_audio_mask` leaves the eos id UN-biased (allowed set =
//     [off, off+count) ∪ {eos}), so the model can always reach eos and stop.
//
// Two modes, selected by `constrain_audio_mask`:
//   * true  (NeuTTS, n_q=1): flat audio-only mask + min-length eos-forbidden
//     guard.  The stream is pure audio, so forbidding everything but the audio
//     range ∪ eos is correct.  BYTE-IDENTICAL to the original single-mode fn.
//   * false (OuteTTS, multi-cb Type A): FREE generation.  The output stream
//     INTERLEAVES structural tokens (<|word_start|>, word text, <|features|>,
//     <|t_..|>, <|energy_N|>, <|code|>, <|word_end|>, ...) between the c1/c2
//     codes, so a flat audio-only mask would forbid exactly those and break
//     generation.  Instead: empty logit_bias (no constraint) + rep-penalty
//     (OuteTTS mandates a 64-token repetition window) + NO min-length guard
//     (min_frames ignored → eos never suppressed).  The multi-cb range is
//     configured by the CALLER via audio_lm_set_audio_token_ranges BEFORE this
//     call, so free-gen mode MUST NOT touch audio_lm_set_audio_token_range
//     (singular) — doing so would clobber the multi-cb range.
bool run_token_single_cb(audio_lm_context * ctx, llama_context * lctx,
                         const llama_model * lmodel, const llama_vocab * vocab,
                         const audio_lm_prompt_info & /*pi*/,
                         const std::vector<llama_token> & prompt_toks,
                         int32_t off, int32_t count, int32_t eos_id,
                         int32_t max_frames, int32_t min_frames, uint32_t seed,
                         float temp, float top_p, int32_t top_k,
                         bool constrain_audio_mask, float rep_penalty,
                         int32_t rep_last_n,
                         int32_t * out_frames, const char ** out_stop) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // 1) Constrained mode only: teach observe_token the single-cb speech range +
    //    eos BEFORE any observe call so <|speech_N|> classifies as CONSUMED and
    //    eos as STOP.  In free-gen (multi-cb) mode the caller has ALREADY set the
    //    range via audio_lm_set_audio_token_ranges; re-setting the singular range
    //    here would clobber it, so we skip it entirely.
    if (constrain_audio_mask) {
        audio_lm_set_audio_token_range(ctx, off, count, eos_id);
    }

    // 2) Build the primary sampler's logit_bias.
    //    * constrained: flat mask (speech ∪ eos).  Empty ⇒ bad range → error.
    //    * free-gen:    empty mask (no constraint) — structural tokens must pass.
    std::vector<llama_logit_bias> mask;
    if (constrain_audio_mask) {
        mask = build_flat_audio_mask(off, count, eos_id, n_vocab);
        if (mask.empty()) {
            std::fprintf(stderr,
                "run_token_single_cb: flat audio mask empty (off=%d count=%d eos=%d n_vocab=%d)\n",
                off, count, eos_id, n_vocab);
            return false;
        }
    }

    // 3) Backbone sampler with the mask attached (no grammar for Type A).
    BackboneSampler bbsmpl;
    std::string berr;
    if (!bbsmpl.build(lmodel, seed, temp, top_k, top_p, /*min_p=*/0.0f,
                      rep_penalty, rep_last_n,
                      /*grammar=*/std::string(), mask, &berr)) {
        std::fprintf(stderr, "run_token_single_cb: sampler init failed: %s\n", berr.c_str());
        *out_stop = "grammar_error";
        return false;
    }

    // 3b) Min-length guard sampler (phase 1).  At temp 1.0 / top-k 50 the
    //     backbone can sample <|SPEECH_GENERATION_END|> from the very first
    //     step for unlucky seeds, yielding 0-few frames of near-silent audio
    //     (measured: some seeds emit eos as the FIRST token).  A pinned seed
    //     hides this, but users don't control the seed, so it's a real defect.
    //     Fix: forbid eos until at least `min_frames` audio frames exist.
    //
    //     We must GUARANTEE a non-eos token when we suppress eos (rejection-
    //     resampling can loop when the eos logit dominates), so the guard
    //     draws from a SECOND sampler whose flat mask FORBIDS eos: passing
    //     eos_id=-1 to build_flat_audio_mask leaves the allowed set to just
    //     [off, off+count) (eos gets a -inf bias, since -1 is not a valid id).
    //
    //     RNG-DESYNC PITFALL: the two samplers own independent mt19937 streams
    //     seeded identically.  If we sampled *exclusively* from the guard while
    //     under min_frames, the primary stream would stay frozen and, at the
    //     first eos-allowed step, replay its very first variate — the same draw
    //     that picked eos at step 0 pre-fix — deterministically re-hitting the
    //     early-eos it was meant to avoid (observed: a seed stopping at EXACTLY
    //     frame min_frames).  So we ALWAYS draw from the primary sampler first
    //     (advancing its stream every step); only if it yields eos while under
    //     min_frames do we substitute a single guard draw for the actual token.
    //     The primary's accept-of-eos is harmless here (no grammar; penalties
    //     disabled), and its stream stays decorrelated from the guard's.
    //
    //     Default min_frames = 25 (~0.33 s at the NeuCodec 75-frame/s rate) is
    //     enough to escape the early-eos basin while staying short enough to
    //     never truncate a genuinely brief utterance; `--min-len` overrides.
    //     Free-gen mode (constrain_audio_mask=false) DISABLES the guard: the
    //     eos-forbidden mask [off,off+count) would forbid the interleaved
    //     structural tokens, and OuteTTS has no early-eos basin to escape (its
    //     eos is <|audio_end|>, reached only after full word blocks).
    BackboneSampler bbsmpl_min;
    const bool use_min_guard = constrain_audio_mask && min_frames > 0;
    if (use_min_guard) {
        std::vector<llama_logit_bias> mask_noeos =
            build_flat_audio_mask(off, count, /*eos_id=*/-1, n_vocab);
        std::string berr_min;
        if (!bbsmpl_min.build(lmodel, seed, temp, top_k, top_p, /*min_p=*/0.0f,
                              /*rep_penalty=*/1.0f, /*rep_last_n=*/0,
                              /*grammar=*/std::string(), mask_noeos, &berr_min)) {
            std::fprintf(stderr,
                "run_token_single_cb: min-guard sampler init failed: %s\n", berr_min.c_str());
            *out_stop = "grammar_error";
            return false;
        }
    }

    // 4) Prefill the ICL prompt.
    int32_t n_past = 0;
    if (!decode_tokens(lctx, prompt_toks, n_past, /*all_pos=*/false)) {
        std::fprintf(stderr, "run_token_single_cb: prefill decode failed\n");
        return false;
    }
    n_past += (int32_t) prompt_toks.size();

    // 5) AR loop — sample → observe → feed back.
    for (int32_t step = 0; step < max_frames; ++step) {
        // Always draw from the primary sampler first so its RNG stream advances
        // every step (see the RNG-desync note above).  Under min_frames, if the
        // primary picked eos, substitute a single eos-forbidden guard draw — the
        // primary's frozen-first-variate replay is thereby broken, and the guard
        // GUARANTEES a non-eos token this step.  At/after min_frames, keep the
        // primary token as-is (eos allowed → OBSERVE_STOP breaks).
        llama_token tok = bbsmpl.sample(lctx, -1);
        if (use_min_guard && (*out_frames) < min_frames && tok == eos_id) {
            tok = bbsmpl_min.sample(lctx, -1);
        }
        observe_action act =
            audio_lm_observe_token(ctx, tok, /*last_hidden=*/nullptr, /*hidden_dim=*/0);
        if (act == OBSERVE_STOP) {
            // For Type A (uses_embed_override off, no codec_lm) observe_token
            // only returns STOP on the clean eos sentinel — the error-bearing
            // STOP path (compose_next_embd failure) is unreachable here.  Do
            // NOT gate on audio_lm_last_error: it can hold a stale message from
            // an earlier deferred get_prompt_info and would misflag a clean eos.
            *out_stop = "eos";
            break;
        }
        if (act == OBSERVE_CONSUMED_EMBED) {
            // Type A must never compose an embed (uses_embed_override is off).
            std::fprintf(stderr,
                "run_token_single_cb: unexpected OBSERVE_CONSUMED_EMBED (Type A "
                "contract violation)\n");
            return false;
        }
        // Feed the just-sampled token back (standard 1-token path).
        if (!decode_tokens(lctx, std::vector<llama_token>{tok}, n_past, /*all_pos=*/false)) {
            std::fprintf(stderr, "run_token_single_cb: feedback decode failed\n");
            return false;
        }
        n_past += 1;
        // Count COMPLETE n_q-frames, not individual consumed tokens.  For n_q=1
        // (NeuTTS) every CONSUMED token completes a frame so this is identical
        // to the old ++(*out_frames).  For n_q>1 (OuteTTS, n_q=2) the
        // accumulator only advances its frame count when the full n_q-tuple is
        // stored, so a single CONSUMED mid-frame leaves the count unchanged
        // until the last token of the frame arrives — avoiding over-counting.
        *out_frames = audio_lm_codes_n_frames(ctx);
        // OBSERVE_PASSTHROUGH mid-audio is unexpected under the mask (only
        // speech ∪ eos are reachable); fed back but not counted as a frame.
    }
    return true;
}

// Flow 4 — Chatterbox T3.
bool run_chatterbox(audio_lm_context * ctx, llama_context * lctx,
                    codec_lm * lm, const codec_lm_chatterbox_info * ci,
                    const tts_runner_params & a, int32_t hidden, int32_t max_frames,
                    std::vector<int32_t> * out_codes,
                    int32_t * out_frames, const char ** out_stop) {
    const float cfg_weight = a.has_cfg_weight ? a.cfg_weight : 0.5f;
    const float temperature = a.has_temp ? a.temp : 0.8f;
    const float top_p = a.has_top_p ? a.top_p : 1.0f;
    const float min_p = a.has_min_p ? a.min_p : 0.05f;
    const float rep_pen = a.has_rep_penalty ? a.repetition_penalty : 1.2f;

    std::vector<int32_t> text_ids(a.text.size() + 64);
    int32_t n_text = 0;
    if (codec_lm_chatterbox_tokenize(lm, a.text.c_str(), text_ids.data(),
                                     (int32_t) text_ids.size(), &n_text) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "chatterbox tokenize failed: %s\n", codec_lm_get_last_error(lm));
        return false;
    }
    text_ids.resize(n_text);
    std::printf("chatterbox: %d text tokens, cfg_weight=%.2f temp=%.2f min_p=%.2f top_p=%.2f rep=%.2f\n",
                n_text, cfg_weight, temperature, min_p, top_p, rep_pen);

    std::vector<float> ref_pcm;
    const float * ref_pcm_ptr = nullptr;
    int32_t ref_n = 0, ref_sr = 0;
    if (!a.ref_audio_path.empty()) {
        std::vector<float> loaded;
        int32_t ln = 0, lsr = 0;
        std::string lerr;
        if (!load_ref_audio(a.ref_audio_path, loaded, &ln, &lsr, &lerr)) {
            std::fprintf(stderr, "%s\n", lerr.c_str());
            return false;
        }
        if (!loaded.empty()) {
            ref_pcm.swap(loaded);
            // The Chatterbox VE expects 16 kHz mono; linearly resample.
            const int32_t target_sr = 16000;
            if (lsr != target_sr && lsr > 0) {
                ref_pcm = resample_mono_f32(ref_pcm, lsr, target_sr);
            }
            ref_pcm_ptr = ref_pcm.data();
            ref_n = (int32_t) ref_pcm.size();
            ref_sr = 16000;
            std::printf("chatterbox: using ref audio %s (%d samples @ 16000 Hz after resample)\n",
                        a.ref_audio_path.c_str(), ref_n);
        }
    }

    const int32_t cond_rows = ci->cond_rows;
    const int32_t seq_len_cap = cond_rows + (n_text + 2) + 2;
    const int32_t n_seq_cap = (cfg_weight > 0.0f) ? 2 : 1;
    std::vector<float> prompt((size_t) seq_len_cap * n_seq_cap * hidden);
    int32_t seq_len = 0, n_seq = 0;
    if (codec_lm_chatterbox_build_prompt(
            lm, text_ids.data(), n_text, cfg_weight,
            nullptr, 0, nullptr, 0, nullptr,
            ref_pcm_ptr, ref_n, ref_sr,
            prompt.data(), seq_len_cap * n_seq_cap, &seq_len, &n_seq) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "chatterbox build_prompt failed: %s\n", codec_lm_get_last_error(lm));
        return false;
    }
    std::printf("chatterbox: prompt seq_len=%d n_seq=%d (%d rows total)\n",
                seq_len, n_seq, seq_len * n_seq);

    {
        const int32_t total = seq_len * n_seq;
        llama_batch b = llama_batch_init(total, hidden, 1);
        b.token = nullptr;
        b.n_tokens = total;
        int32_t bi = 0;
        for (int32_t s = 0; s < n_seq; ++s) {
            for (int32_t rr = 0; rr < seq_len; ++rr) {
                std::memcpy(b.embd + (size_t) bi * hidden,
                            prompt.data() + ((size_t) s * seq_len + rr) * hidden,
                            (size_t) hidden * sizeof(float));
                b.pos[bi] = rr;
                b.n_seq_id[bi] = 1;
                b.seq_id[bi][0] = s;
                b.logits[bi] = (rr == seq_len - 1) ? 1 : 0;
                ++bi;
            }
        }
        int rc = llama_decode(lctx, b);
        llama_batch_free(b);
        if (rc != 0) { std::fprintf(stderr, "chatterbox prefill decode failed\n"); return false; }
    }

    const int32_t V = ci->speech_vocab_size;
    // T3-faithful chain: penalties(full history) → temp → min_p → top_p →
    // dist(seed).  Seed the penalty ring buffer with start_speech_token, as
    // the old sample_t3 pre-loaded `generated` with it before step 0.
    SamplerChain smpl;
    if (temperature <= 0.0f) smpl.init_greedy();
    else {
        smpl.init_sampled(a.seed ? a.seed : 0xC0DEC1ABu, temperature, /*top_k=*/0,
                          top_p, min_p, rep_pen, /*rep_last_n=*/-1);
        if (smpl.has_penalties) llama_sampler_accept(smpl.chain, ci->start_speech_token);
    }
    int32_t n_past = seq_len;

    auto lane_hidden = [&](int32_t lane) -> const float * {
        return llama_get_embeddings_ith(lctx, -(n_seq - lane));
    };
    auto speech_logits = [&](const float * h, std::vector<float> * out) -> bool {
        if (!audio_lm_step_begin(ctx, h, hidden)) return false;
        int32_t cb = 0, nlog = 0;
        const float * lg = audio_lm_step_logits(ctx, &cb, &nlog);
        if (!lg || nlog <= 0) return false;
        out->assign(lg, lg + nlog);
        audio_lm_step_push_code(ctx, 0);
        int32_t dummy = 0;
        audio_lm_step_finish(ctx, &dummy, 1);
        return true;
    };

    for (int32_t step = 0; step < max_frames; ++step) {
        const float * hc = lane_hidden(0);
        const float * hu = (n_seq == 2) ? lane_hidden(1) : nullptr;
        if (!hc) { std::fprintf(stderr, "no hidden at step %d\n", step); return false; }
        std::vector<float> cond, uncond;
        if (!speech_logits(hc, &cond)) {
            std::fprintf(stderr, "speech_head (cond) failed: %s\n", audio_lm_last_error(ctx));
            return false;
        }
        if (hu && !speech_logits(hu, &uncond)) {
            std::fprintf(stderr, "speech_head (uncond) failed\n"); return false;
        }
        const int32_t VV = (int32_t) cond.size();
        std::vector<float> logits(VV);
        for (int32_t i = 0; i < VV; ++i)
            logits[i] = hu ? (cond[i] + cfg_weight * (cond[i] - uncond[i])) : cond[i];
        int32_t code = smpl.sample(logits.data(), VV);
        (void) V;
        if (code == ci->stop_speech_token) { *out_stop = "eos_code_c0"; break; }
        if (code < ci->start_speech_token) out_codes->push_back(code);
        (*out_frames)++;

        std::vector<float> nb(hidden);
        if (codec_lm_chatterbox_compose_speech_embd(lm, code, step + 1, nb.data(), hidden)
                != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "compose_speech_embd failed: %s\n", codec_lm_get_last_error(lm));
            return false;
        }
        std::vector<float> row((size_t) n_seq * hidden);
        for (int32_t s = 0; s < n_seq; ++s)
            std::memcpy(row.data() + (size_t) s * hidden, nb.data(), (size_t) hidden * sizeof(float));
        if (!decode_embed_batch(lctx, row.data(), hidden, n_seq, n_past)) {
            std::fprintf(stderr, "chatterbox step decode failed\n"); return false;
        }
        ++n_past;
    }
    return true;
}

// Marshal a codec_common audio_lm_audio_output into the runner result.
void fill_result_from_output(const audio_lm_audio_output & pcm,
                             int32_t n_frames, const char * stop,
                             tts_runner_result * out) {
    out->pcm          = pcm.pcm;
    out->sample_rate  = pcm.sample_rate;
    out->n_channels   = pcm.n_channels;
    out->n_frames     = n_frames;
    out->stop_reason  = stop;
}

// ── OuteTTS V2/V3 detection ───────────────────────────────────────────────────
//
// Scans the backbone vocab for signature special-token pieces that uniquely
// identify an OuteTTS family.  Uses O(vocab) linear scan per probe — called
// once at init so the cost is negligible.
//
// Fail-closed: anything not clearly OuteTTS returns false so non-OuteTTS
// models (Qwen3-TTS, MOSS-TTSD, Chatterbox, …) are never mis-constrained.

static bool vocab_has_piece(const llama_vocab * vocab, const char * piece) {
    if (!vocab || !piece) return false;
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (int32_t id = 0; id < n; ++id) {
        const char * t = llama_vocab_get_text(vocab, id);
        if (t && std::strcmp(t, piece) == 0) return true;
    }
    return false;
}

// ── NeuTTS detection ─────────────────────────────────────────────────────────
//
// Find the vocab id for a given piece by linear scan; returns -1 if absent.
// (vocab_has_piece does the same but returns bool — we need the id here.)
static int32_t vocab_piece_id(const llama_vocab * vocab, const char * piece) {
    if (!vocab || !piece) return -1;
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (int32_t id = 0; id < n; ++id) {
        const char * t = llama_vocab_get_text(vocab, id);
        if (t && std::strcmp(t, piece) == 0) return id;
    }
    return -1;
}

// Identify an OuteTTS backbone from vocab signature tokens.  V3 carries
// per-word feature/paired-code markers; V2 carries <|space|> + bare code
// tokens and lacks the V3 markers.  Return false on anything ambiguous so
// non-OuteTTS models are never mis-constrained.
static bool detect_outetts_version(const llama_model * lmodel,
                                   const llama_vocab * vocab,
                                   codec_common::outetts_version * out) {
    if (!vocab || !out) return false;

    auto has = [&](const char * piece) -> bool {
        return vocab_has_piece(vocab, piece);
    };

    // V3 signature: any of these per-word markers present.
    const bool v3 = has("<|word_start|>") || has("<|features|>") || has("<|c1_0|>");
    // V2 signature: space + audio-end tokens, but NOT the V3 markers.
    const bool v2 = !v3 && has("<|space|>") && has("<|audio_end|>");

    // Corroborate with general.name when available (non-fatal; skip on error).
    char name[256] = {0};
    if (lmodel) {
        llama_model_meta_val_str(lmodel, "general.name", name, sizeof(name));
    }

    if (v3) {
        *out = codec_common::outetts_version::V3;
        std::printf("[tts] OuteTTS detection: V3 (general.name=\"%s\")\n", name);
        return true;
    }
    if (v2) {
        *out = codec_common::outetts_version::V2;
        std::printf("[tts] OuteTTS detection: V2 (general.name=\"%s\")\n", name);
        return true;
    }
    std::printf("[tts] OuteTTS detection: negative — not OuteTTS (general.name=\"%s\")\n", name);
    return false;
}

// Resolve the full constraint for a backbone synthesis request.
// Resolution order (first match wins):
//   1. Explicit user GBNF grammar.
//   2. OuteTTS V2/V3 structured backbone → prompt-dependent grammar.
//   3. Codec-metadata GBNF grammar (MOSS-TTSD tts_auto_grammar).
//   4. Flat audio-token logit-bias mask for Type A direct-audio models.
// Returns an empty constraint for models with no constraint (CSM / Chatterbox /
// Qwen3-TTS / LFM2 / Realtime) — identical behaviour to before this change.
static tts_constraint resolve_constraint(const llama_model * lmodel,
                                         const llama_vocab * vocab,
                                         const audio_lm_prompt_info & pi,
                                         const std::string & user_grammar,
                                         const std::string & text,
                                         int32_t n_vocab) {
    tts_constraint c;
    // 1) explicit user grammar wins.
    if (!user_grammar.empty()) { c.grammar = user_grammar; return c; }
    // 2) structured backbone family (OuteTTS) → prompt-dependent grammar.
    codec_common::outetts_version ov;
    if (detect_outetts_version(lmodel, vocab, &ov)) {
        c.grammar = codec_common::outetts_build_grammar(ov, text);
        return c;
    }
    // 3a) codec-metadata grammar (MOSS-TTSD; unchanged).
    c.grammar = tts_auto_grammar(pi, text);
    if (!c.grammar.empty()) return c;
    // 3b) flat Type A → logit-bias mask.
    if (pi.audio_tok_offset >= 0) {
        c.logit_bias = build_flat_audio_mask(pi.audio_tok_offset, pi.audio_tok_count,
                                             pi.audio_tok_eos, n_vocab);
    }
    return c;
}

}  // namespace

// ── NeuTTS detection (public API, declared in tts_runner.h) ──────────────────
//
// Identify a NeuTTS backbone from its vocab signature tokens.  Both nano
// (Llama 229M) and air (Qwen2 0.5B) carry these three special tokens; no
// other supported backbone does.  Fail-closed: returns false on anything
// ambiguous.
bool detect_neutts(const llama_vocab * vocab) {
    return vocab_has_piece(vocab, "<|SPEECH_GENERATION_END|>")
        && vocab_has_piece(vocab, "<|TEXT_PROMPT_START|>")
        && vocab_has_piece(vocab, "<|speech_0|>");
}

// Derive the audio-token range from a NeuTTS backbone vocab.
// *offset = id of <|speech_0|>; *count = contiguous block length (65536 for
// both backbones); *eos_id = id of <|SPEECH_GENERATION_END|>.
// Returns false if any marker token is absent.
bool neutts_audio_token_range(const llama_vocab * vocab,
                              int32_t * offset,
                              int32_t * count,
                              int32_t * eos_id) {
    const int32_t off = vocab_piece_id(vocab, "<|speech_0|>");
    const int32_t eos = vocab_piece_id(vocab, "<|SPEECH_GENERATION_END|>");
    if (off < 0 || eos < 0) return false;
    // Count contiguous <|speech_N|> tokens from N=0 upward.
    // Explicit upper bound = (vocab size − off) prevents O(V²) scan on a
    // pathological vocab where the series never breaks.
    const int32_t nv = llama_vocab_n_tokens(vocab);
    int32_t n = 0;
    char buf[32];
    for (; off + n < nv; ++n) {
        std::snprintf(buf, sizeof(buf), "<|speech_%d|>", n);
        if (vocab_piece_id(vocab, buf) != off + n) break;
    }
    *offset = off;
    *count  = n;
    *eos_id = eos;
    return true;
}

// Derive the multi-codebook audio-token ranges from an OuteTTS V3 backbone
// vocab.  Resolves everything from vocab pieces so it is robust across the
// 0.6B (Qwen3) and 1B (Llama) backbones, whose ids differ (facts §Step 2):
//   offsets[0] = id of <|c1_0|>   (0.6B → 151669, 1B → 128256)
//   offsets[1] = id of <|c2_0|>   (0.6B → 152694, 1B → 129281)
//   counts     = {1024, 1024}     (valid DAC code range 0..1023; index 1024 is
//                                  a padding token never emitted by the codec)
//   *sentinel  = id of <|code|>   (0.6B → 156730)
//   *eos       = id of <|audio_end|> (0.6B → 156729)
// Returns false if any marker token is absent.
bool outetts_audio_token_ranges(const llama_vocab * vocab,
                                int32_t offsets[2], int32_t counts[2],
                                int32_t * sentinel, int32_t * eos) {
    if (!vocab || !offsets || !counts || !sentinel || !eos) return false;
    const int32_t c1_base   = vocab_piece_id(vocab, "<|c1_0|>");
    const int32_t c2_base   = vocab_piece_id(vocab, "<|c2_0|>");
    const int32_t code_id   = vocab_piece_id(vocab, "<|code|>");
    const int32_t audio_end = vocab_piece_id(vocab, "<|audio_end|>");
    if (c1_base < 0 || c2_base < 0 || code_id < 0 || audio_end < 0) return false;
    offsets[0] = c1_base;
    offsets[1] = c2_base;
    counts[0]  = 1024;
    counts[1]  = 1024;
    *sentinel  = code_id;
    *eos       = audio_end;
    return true;
}

// Build the full NeuTTS ICL prefill token sequence.
//
// 1. Assembles the verbatim prompt string (phonemes mode, per neutts-facts §Step 5):
//      "user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref_text_phonemes} {input_phonemes}"
//      "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>"
// 2. Tokenizes it with parse_special=true so the <|...|> markers resolve.
// 3. Appends one vocab id (offset + c) per ref code.
// Guard: returns empty vector on any error (null vocab, tokenize failure, out-of-range code).
std::vector<llama_token> build_neutts_prompt(
        const llama_vocab          * vocab,
        const std::string          & ref_text_phonemes,
        const std::string          & input_phonemes,
        const std::vector<int32_t> & ref_codes) {
    if (!vocab) return {};

    // 1. Verbatim template (neutts-facts §Step 5).
    const std::string prompt_str =
        "user: Convert the text to speech:<|TEXT_PROMPT_START|>"
        + ref_text_phonemes + " " + input_phonemes
        + "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>";

    // 2. Tokenize with parse_special=true (mirrors tokenize_str idiom).
    std::vector<llama_token> toks = tokenize_str(vocab, prompt_str, /*add_bos=*/false, /*parse_special=*/true);
    if (toks.empty()) {
        std::fprintf(stderr, "[neutts] build_neutts_prompt: tokenize returned empty\n");
        return {};
    }

    // 3. Resolve speech-token range and append ref code ids.
    int32_t offset = -1, count = -1, eos_id = -1;
    if (!neutts_audio_token_range(vocab, &offset, &count, &eos_id)) {
        std::fprintf(stderr, "[neutts] build_neutts_prompt: neutts_audio_token_range failed\n");
        return {};
    }

    toks.reserve(toks.size() + ref_codes.size());
    for (int32_t c : ref_codes) {
        if (c < 0 || c >= count) {
            std::fprintf(stderr,
                "[neutts] build_neutts_prompt: ref code %d out of range [0, %d) — aborting\n",
                c, count);
            return {};
        }
        toks.push_back(offset + c);
    }
    return toks;
}

// ── OuteTTS V3 speaker loader + prompt builder (public API) ──────────────────
//
// Speaker JSON schema (outetts 0.4.4, outetts10-facts §Step 3):
//
//   {
//     "text":  "<reference sentence>",
//     "words": [
//       { "word": "The", "duration": 0.20,
//         "c1": [720,...], "c2": [658,...],
//         "features": {"energy":10, "spectral_centroid":15, "pitch":45} },
//       ...
//     ],
//     "global_features": {"energy":13, "spectral_centroid":20, "pitch":28},
//     "interface_version": 3
//   }
//
// We parse this with a minimal hand-parser (no external JSON library dependency
// needed for this fixed schema).

namespace {

// ── Tiny JSON helpers ─────────────────────────────────────────────────────────
// These operate on the raw file content (one std::string) and implement only
// what the speaker JSON schema requires: string extraction, key lookup, integer
// array parsing, and float parsing.  They do NOT handle Unicode escapes, nested
// objects deeper than 2 levels, or arrays of objects except words[].

// Advance pos past whitespace.
static void json_skip_ws(const std::string & s, size_t & pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\r' || s[pos] == '\n'))
        ++pos;
}

// Parse a JSON string starting at s[pos] (which must be '"').
// Advances pos to after the closing '"'.
static std::string json_parse_string(const std::string & s, size_t & pos) {
    if (pos >= s.size() || s[pos] != '"') return "";
    ++pos;  // skip opening "
    std::string out;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            ++pos;
            switch (s[pos]) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                default:   out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        ++pos;
    }
    if (pos < s.size()) ++pos;  // skip closing "
    return out;
}

// Return the position of the value after key "key" in the JSON object rooted
// at [obj_start, obj_end).  Returns std::string::npos if not found.
static size_t json_find_key(const std::string & s, size_t obj_start, size_t obj_end,
                             const std::string & key) {
    size_t pos = obj_start;
    while (pos < obj_end) {
        json_skip_ws(s, pos);
        if (pos >= obj_end) break;
        if (s[pos] != '"') { ++pos; continue; }  // skip non-key chars (commas, etc.)
        std::string k = json_parse_string(s, pos);
        // Skip the colon separator after the key.
        json_skip_ws(s, pos);
        if (pos < obj_end && s[pos] == ':') ++pos;
        json_skip_ws(s, pos);
        if (k == key) {
            // pos now points at the start of the value — return it.
            return pos;
        }
        // Key doesn't match: skip the value so we advance past it.
        if (pos >= obj_end) break;
        char c = s[pos];
        if (c == '"') {
            json_parse_string(s, pos);
        } else if (c == '{' || c == '[') {
            // Skip matched bracket (handles nested objects/arrays).
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            while (pos < obj_end) {
                char ch = s[pos++];
                if (in_str) { if (ch == '\\') { if (pos < obj_end) ++pos; }
                              else if (ch == '"') in_str = false; }
                else { if (ch == '"') in_str = true;
                       else if (ch == open)  ++depth;
                       else if (ch == close) { --depth; if (depth == 0) break; } }
            }
        } else {
            // number / bool / null — advance to next comma or closing bracket.
            while (pos < obj_end && s[pos] != ',' && s[pos] != '}' && s[pos] != ']') ++pos;
        }
    }
    return std::string::npos;
}

// Parse an integer array [N, N, ...] starting at s[pos] (which must be '[').
// Advances pos past the closing ']'.
static std::vector<int32_t> json_parse_int_array(const std::string & s, size_t & pos) {
    std::vector<int32_t> out;
    if (pos >= s.size() || s[pos] != '[') return out;
    ++pos;  // skip '['
    while (pos < s.size()) {
        json_skip_ws(s, pos);
        if (pos >= s.size()) break;
        if (s[pos] == ']') { ++pos; break; }
        if (s[pos] == ',') { ++pos; continue; }
        // parse integer (may be negative)
        bool neg = (s[pos] == '-');
        if (neg) ++pos;
        int32_t v = 0;
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
            v = v * 10 + (s[pos++] - '0');
        }
        out.push_back(neg ? -v : v);
    }
    return out;
}

// Parse a float at s[pos]; advances pos past the number.
static float json_parse_float(const std::string & s, size_t & pos) {
    size_t start = pos;
    if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) ++pos;
    while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        ++pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
    }
    return (float) std::stod(s.substr(start, pos - start));
}

// Parse an integer at s[pos]; advances pos past the number.
static int32_t json_parse_int(const std::string & s, size_t & pos) {
    json_skip_ws(s, pos);
    bool neg = (pos < s.size() && s[pos] == '-');
    if (neg) ++pos;
    int32_t v = 0;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') v = v * 10 + (s[pos++] - '0');
    return neg ? -v : v;
}

// Find the matching close bracket for the open bracket at s[open_pos].
// Returns std::string::npos if not found.
static size_t json_match_bracket(const std::string & s, size_t open_pos) {
    if (open_pos >= s.size()) return std::string::npos;
    char open = s[open_pos], close = (open == '{') ? '}' : ']';
    int depth = 0;
    bool in_str = false;
    size_t pos = open_pos;
    while (pos < s.size()) {
        char c = s[pos++];
        if (in_str) { if (c == '\\') { if (pos < s.size()) ++pos; } else if (c == '"') in_str = false; }
        else { if (c == '"') in_str = true; else if (c == open) ++depth; else if (c == close) { --depth; if (depth == 0) return pos - 1; } }
    }
    return std::string::npos;
}

}  // namespace

// ── load_outetts_speaker ──────────────────────────────────────────────────────

bool load_outetts_speaker(const std::string & json_path, OutettsSpeaker * out) {
    if (!out) return false;

    // Read entire file.
    std::ifstream f(json_path);
    if (!f.is_open()) {
        std::fprintf(stderr, "[outetts] load_outetts_speaker: cannot open %s\n", json_path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();

    // Find top-level object boundaries.
    size_t root = src.find('{');
    if (root == std::string::npos) {
        std::fprintf(stderr, "[outetts] load_outetts_speaker: no JSON object in %s\n", json_path.c_str());
        return false;
    }
    size_t root_end = json_match_bracket(src, root);
    if (root_end == std::string::npos) root_end = src.size();

    // Parse "text" field.
    {
        size_t vp = json_find_key(src, root + 1, root_end, "text");
        if (vp != std::string::npos && vp < src.size() && src[vp] == '"') {
            out->text = json_parse_string(src, vp);
        }
    }

    // Parse "words" array.
    {
        size_t vp = json_find_key(src, root + 1, root_end, "words");
        if (vp == std::string::npos || vp >= src.size() || src[vp] != '[') {
            std::fprintf(stderr, "[outetts] load_outetts_speaker: 'words' array not found in %s\n",
                         json_path.c_str());
            return false;
        }
        size_t words_end = json_match_bracket(src, vp);
        if (words_end == std::string::npos) words_end = src.size();
        size_t pos = vp + 1;  // skip '['

        while (pos < words_end) {
            json_skip_ws(src, pos);
            if (pos >= words_end) break;
            if (src[pos] == ']') break;
            if (src[pos] != '{') { ++pos; continue; }

            size_t wobj_end = json_match_bracket(src, pos);
            if (wobj_end == std::string::npos) wobj_end = words_end;
            size_t wstart = pos + 1;

            OutettsSpeakerWord w;

            // "word"
            {
                size_t kp = json_find_key(src, wstart, wobj_end, "word");
                if (kp != std::string::npos && src[kp] == '"') {
                    w.word = json_parse_string(src, kp);
                }
            }

            // "duration"
            {
                size_t kp = json_find_key(src, wstart, wobj_end, "duration");
                if (kp != std::string::npos) {
                    w.duration = json_parse_float(src, kp);
                }
            }

            // "c1"
            {
                size_t kp = json_find_key(src, wstart, wobj_end, "c1");
                if (kp != std::string::npos && src[kp] == '[') {
                    w.c1 = json_parse_int_array(src, kp);
                }
            }

            // "c2"
            {
                size_t kp = json_find_key(src, wstart, wobj_end, "c2");
                if (kp != std::string::npos && src[kp] == '[') {
                    w.c2 = json_parse_int_array(src, kp);
                }
            }

            // "features": {"energy": N, "spectral_centroid": N, "pitch": N}
            {
                size_t fp = json_find_key(src, wstart, wobj_end, "features");
                if (fp != std::string::npos && src[fp] == '{') {
                    size_t fobj_end = json_match_bracket(src, fp);
                    if (fobj_end == std::string::npos) fobj_end = wobj_end;
                    size_t fstart = fp + 1;

                    size_t ep = json_find_key(src, fstart, fobj_end, "energy");
                    if (ep != std::string::npos) w.energy = json_parse_int(src, ep);

                    size_t sp = json_find_key(src, fstart, fobj_end, "spectral_centroid");
                    if (sp != std::string::npos) w.spectral_centroid = json_parse_int(src, sp);

                    size_t pp = json_find_key(src, fstart, fobj_end, "pitch");
                    if (pp != std::string::npos) w.pitch = json_parse_int(src, pp);
                }
            }

            out->words.push_back(std::move(w));
            pos = wobj_end + 1;  // advance past '}' of word object
        }
    }

    return !out->words.empty();
}

// ── build_outetts_v3_prompt ───────────────────────────────────────────────────
//
// Exact V3 prefill (outetts10-facts §Step 1):
//
//   <|im_start|>\n
//   <|text_start|>SPEAKER_TEXT. TARGET_TEXT<|text_end|>\n
//   <|audio_start|>\n
//   {speaker word blocks, each ending with \n}
//   <|word_start|>
//
// Each word block (from create_codes in prompt_processor.py):
//   <|word_start|>WORD<|features|><|t_D.DD|><|energy_N|><|spectral_centroid_N|><|pitch_N|>
//   <|code|><|c1_X|><|c2_X|>...<|word_end|>
//
// c1_base (0.6B) = 151669, c2_base (0.6B) = 152694 (facts §Step 2).
// The format specials (<|...|>) are resolved via llama_tokenize(parse_special=true)
// together with the surrounding text; code tokens are appended as raw ids.
//
// Guard: returns empty on null vocab, empty text, or any c1/c2 code out of [0,1024).

std::vector<llama_token> build_outetts_v3_prompt(const llama_vocab    * vocab,
                                                  const std::string   & text,
                                                  const OutettsSpeaker& speaker) {
    if (!vocab || text.empty()) return {};

    // Validate all codes up front before we allocate anything.
    for (const auto & w : speaker.words) {
        for (int32_t c : w.c1) {
            if (c < 0 || c >= 1024) {
                std::fprintf(stderr,
                    "[outetts] build_outetts_v3_prompt: c1 code %d out of range [0,1024)\n", c);
                return {};
            }
        }
        for (int32_t c : w.c2) {
            if (c < 0 || c >= 1024) {
                std::fprintf(stderr,
                    "[outetts] build_outetts_v3_prompt: c2 code %d out of range [0,1024)\n", c);
                return {};
            }
        }
    }

    // Resolve c1/c2 base ids from the vocab (facts §Step 2: look up <|c1_0|> / <|c2_0|>).
    const int32_t c1_base = vocab_piece_id(vocab, "<|c1_0|>");
    const int32_t c2_base = vocab_piece_id(vocab, "<|c2_0|>");
    if (c1_base < 0 || c2_base < 0) {
        std::fprintf(stderr,
            "[outetts] build_outetts_v3_prompt: <|c1_0|> or <|c2_0|> not found in vocab\n");
        return {};
    }

    // ── Build the prompt string.
    // The string contains all special tokens in <|...|> form; llama_tokenize
    // with parse_special=true will resolve them to their ids.  Code tokens are
    // NOT embedded in the string — they are appended as raw ids after tokenization
    // to avoid any piece-merging ambiguity.
    //
    // To interleave text and code tokens correctly (the code tokens must appear
    // at the right position within the word block), we build the prompt in
    // segments.  Each word produces:
    //   text_segment: "<|word_start|>WORD<|features|><|t_D.DD|>...<|code|>"
    //   code_ids:     [c1_base+c1[0], c2_base+c2[0], c1_base+c1[1], ...]
    //   end_segment:  "<|word_end|>\n"
    // After all words, we append the trailing "<|word_start|>" primer.

    // Merged text: "SPEAKER_TEXT. TARGET_TEXT"
    const std::string merged_text = speaker.text + ". " + text;

    // Header segment (fully textual with specials)
    const std::string header =
        "<|im_start|>\n"
        "<|text_start|>" + merged_text + "<|text_end|>\n"
        "<|audio_start|>\n";

    // We'll accumulate (text_segment, code_tokens) pairs then do one tokenize
    // per text segment and concatenate everything.

    struct Segment {
        std::string          text_piece;   // tokenized with parse_special=true
        std::vector<int32_t> code_ids;     // raw token ids appended after text_piece
    };

    std::vector<Segment> segments;
    segments.push_back({header, {}});

    for (const auto & w : speaker.words) {
        // Per-word text piece (from create_codes in prompt_processor.py):
        //   "<|word_start|>WORD<|features|><|t_D.DD|><|energy_N|><|spectral_centroid_N|><|pitch_N|><|code|>"
        char dur_buf[16];
        std::snprintf(dur_buf, sizeof(dur_buf), "%.2f", (double) w.duration);

        std::string wp = "<|word_start|>" + w.word
            + "<|features|>"
            + "<|t_" + dur_buf + "|>"
            + "<|energy_"            + std::to_string(w.energy)            + "|>"
            + "<|spectral_centroid_" + std::to_string(w.spectral_centroid) + "|>"
            + "<|pitch_"             + std::to_string(w.pitch)             + "|>"
            + "<|code|>";

        // Code token ids: interleaved c1/c2 pairs per frame.
        std::vector<int32_t> code_ids;
        const size_t n_frames = std::min(w.c1.size(), w.c2.size());
        code_ids.reserve(n_frames * 2);
        for (size_t i = 0; i < n_frames; ++i) {
            code_ids.push_back(c1_base + w.c1[i]);
            code_ids.push_back(c2_base + w.c2[i]);
        }

        // End piece for this word (appended after the code ids).
        segments.push_back({wp,            std::move(code_ids)});
        segments.push_back({"<|word_end|>\n", {}});
    }

    // Trailing <|word_start|> primer (primes generation of first target word).
    segments.push_back({"<|word_start|>", {}});

    // ── Assemble the full token sequence.
    std::vector<llama_token> result;
    for (const auto & seg : segments) {
        if (!seg.text_piece.empty()) {
            auto toks = tokenize_str(vocab, seg.text_piece, /*add_bos=*/false, /*parse_special=*/true);
            if (toks.empty()) {
                std::fprintf(stderr,
                    "[outetts] build_outetts_v3_prompt: tokenize failed for segment: \"%s\"\n",
                    seg.text_piece.substr(0, 80).c_str());
                return {};
            }
            result.insert(result.end(), toks.begin(), toks.end());
        }
        for (int32_t id : seg.code_ids) {
            result.push_back(id);
        }
    }

    return result;
}

bool tts_runner_synthesize(const tts_runner_params & a, tts_runner_result * out) {
    // Pocket-TTS FlowLM is self-contained (no backbone) — try it first; a
    // non-zero return means it handled the request.
    {
        tts_runner_result fr;
        int handled = tts_runner_synthesize_selfcontained(a, &fr);
        if (handled) {
            if (!fr.error.empty()) { out->error = fr.error; return false; }
            *out = std::move(fr);
            return true;
        }
    }
    if (a.backbone_path.empty()) {
        out->error = "synthesize requires backbone_path (llama.cpp model)";
        return false;
    }

    audio_lm_params p;
    p.codec_path = a.codec_path;
    p.use_gpu    = a.use_gpu;
    p.n_threads  = a.n_threads;
    std::string err;
    auto * ctx = audio_lm_init(p, &err);
    if (!ctx) { out->error = "audio_lm_init failed: " + err; return false; }

    // audio_lm_get_prompt_info requires a codec_lm adaptor.  NeuTTS uses a
    // codec-only NeuCodec gguf (no codec_lm — the backbone lm_head emits the
    // audio tokens directly), so get_prompt_info fails there.  DEFER the abort:
    // load the backbone, detect NeuTTS from its vocab, and if it IS NeuTTS run
    // the self-contained Type A flow (which never touches `pi`).  If it is NOT
    // NeuTTS, abort below with the original error.  Gate the deferral on the
    // exact codec-only signal (no codec_lm handle), not on the error string.
    audio_lm_prompt_info pi;
    const bool pi_ok = audio_lm_get_prompt_info(ctx, &pi);
    std::string deferred_pi_err;
    if (!pi_ok) {
        // Capture the error now — later ctx calls overwrite audio_lm_last_error.
        deferred_pi_err = std::string("get_prompt_info failed: ") + audio_lm_last_error(ctx);
        if (audio_lm_get_lm(ctx) != nullptr) {
            // A real codec_lm is present but prompt-info still failed — that is
            // a genuine error, not the codec-only NeuTTS case.  Abort now.
            out->error = deferred_pi_err;
            audio_lm_free(ctx);
            return false;
        }
    }
    const int32_t hidden = audio_lm_hidden_dim(ctx);
    const int32_t n_cb   = audio_lm_n_codebook(ctx);
    if (pi_ok) {
        std::printf("model: arch=%s kind=%d n_cb=%d hidden=%d cb0_backbone=%d audio_offset=%d eos_c0=%d\n",
                    pi.host_arch.c_str(), (int) pi.model_kind, n_cb, hidden,
                    (int) pi.cb0_from_backbone, pi.audio_codebook_offset, pi.eos_code_c0);
    } else {
        std::printf("model: no codec_lm adaptor (codec-only gguf) — deferring to NeuTTS detection\n");
    }

    // ── Moshi: formally out of scope for one-shot synthesize ────────────
    if (pi_ok &&
        pi.host_arch == "llama" &&
        pi.model_kind == audio_lm_prompt_info::KIND_RESIDUAL_DEPTH_AR &&
        pi.eos_code_c0 < 0 && !pi.cb0_from_backbone) {
        out->error =
            "synthesize: this looks like a Moshi codec_lm (full-duplex dialogue, "
            "no audio EOS, Helium backbone).  Moshi is not supported by "
            "synthesize: the pinned llama.cpp has no Helium arch, moshiko.gguf "
            "ships no backbone, and the duplex protocol has no one-shot stop "
            "condition.  Its one-shot-TTS sibling is kyutai/dsm.  See "
            "docs/codec_common_api.md.";
        audio_lm_free(ctx);
        return false;
    }

    // ── Backbone init ──────────────────────────────────────────────
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * lmodel = llama_model_load_from_file(a.backbone_path.c_str(), mp);
    if (!lmodel) {
        out->error = "llama_model_load_from_file failed: " + a.backbone_path;
        audio_lm_free(ctx); llama_backend_free();
        return false;
    }
    const int32_t n_embd = llama_model_n_embd(lmodel);
    // The n_embd==codec-hidden invariant only applies to codec_lm-backed flows
    // (the backbone hidden feeds the codec_lm adaptor).  NeuTTS is codec-only
    // (hidden==0, pi_ok==false); its backbone drives the codec purely via token
    // ids, so skip this check there.
    if (pi_ok && n_embd != hidden) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "backbone n_embd=%d != codec hidden=%d — wrong backbone?", n_embd, hidden);
        out->error = buf;
        llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
        return false;
    }
    codec_lm * lm_handle = audio_lm_get_lm(ctx);
    const codec_lm_chatterbox_info * cbx =
        lm_handle ? codec_lm_chatterbox_get_info(lm_handle) : nullptr;
    const bool is_chatterbox = (cbx != nullptr);
    const float cbx_cfg = a.has_cfg_weight ? a.cfg_weight : 0.5f;
    const int32_t cbx_n_seq = (is_chatterbox && cbx_cfg > 0.0f) ? 2 : 1;

    const int32_t max_frames = a.max_frames > 0 ? a.max_frames : 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) std::max(4096, max_frames + 512);
    cp.n_batch      = cp.n_ctx;
    cp.n_ubatch     = cp.n_ctx;
    cp.n_seq_max    = (uint32_t) cbx_n_seq;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (a.n_threads > 0) { cp.n_threads = a.n_threads; cp.n_threads_batch = a.n_threads; }
    llama_context * lctx = llama_init_from_model(lmodel, cp);
    if (!lctx) {
        out->error = "llama_init_from_model failed";
        llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(lmodel);

    // ── Type A sanity: surface audio-token range bounds and edge pieces ──
    // A mis-counted codec.audio_token.offset (added-token order drift in the
    // converter) produces in-range-but-wrong ids → silent garbage.  Log the
    // range edges and warn on overflow so converter bugs surface early.
    {
        int32_t a_off = -1, a_cnt = 0, a_eos = -1;
        audio_lm_get_audio_token_range(ctx, &a_off, &a_cnt, &a_eos);
        if (a_off >= 0) {
            const int32_t nv = llama_vocab_n_tokens(vocab);
            if (a_off + a_cnt > nv) {
                std::fprintf(stderr, "WARN: audio-token range [%d,%d) exceeds n_vocab=%d "
                             "(bad codec.audio_token.offset/count?)\n",
                             a_off, a_off + a_cnt, nv);
            }
            char pf[128] = {0}, pl[128] = {0};
            llama_token_to_piece(vocab, a_off,             pf, sizeof(pf), 0, true);
            llama_token_to_piece(vocab, a_off + a_cnt - 1, pl, sizeof(pl), 0, true);
            std::printf("audio-token range [%d,%d) eos=%d; first=<%s> last=<%s>\n",
                        a_off, a_off + a_cnt, a_eos, pf, pl);
        }
    }

    // ── Chatterbox T3 flow (Flow 4) ───────────────────────────────────
    if (is_chatterbox) {
        std::vector<int32_t> speech_codes;
        int32_t n_frames = 0;
        const char * stop_reason = "max_frames";
        bool ok = run_chatterbox(ctx, lctx, lm_handle, cbx, a, hidden, max_frames,
                                 &speech_codes, &n_frames, &stop_reason);
        llama_free(lctx);
        llama_model_free(lmodel);
        llama_backend_free();
        if (!ok) {
            out->error = std::string("chatterbox AR failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        std::printf("chatterbox AR done: %d frames, %zu speech codes, stop=%s\n",
                    n_frames, speech_codes.size(), stop_reason);
        if (speech_codes.empty()) {
            out->error = "no speech codes generated";
            audio_lm_free(ctx);
            return false;
        }
        audio_lm_reset(ctx);
        if (!audio_lm_push_codes(ctx, speech_codes.data(),
                                 (int32_t) speech_codes.size(), 1)) {
            out->error = std::string("push_codes failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        audio_lm_audio_output pcm;
        if (!audio_lm_decode_audio(ctx, &pcm)) {
            out->error = std::string("decode_audio failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        fill_result_from_output(pcm, n_frames, stop_reason, out);
        audio_lm_free(ctx);
        return true;
    }

    // ── NeuTTS ICL ref-encode + prompt assembly + self-contained decode ─────
    // When the backbone is a NeuTTS model, the prefill is built from:
    //   1. The verbatim ICL template (ref phonemes + input phonemes)
    //   2. The NeuCodec-encoded reference audio codes appended as speech tokens
    // then run_token_single_cb drives the Type A decode and returns directly
    // (self-contained; never touches `pi` or the codec_lm switch below).
    std::vector<llama_token> neutts_prompt_toks;
    const bool is_neutts = detect_neutts(vocab);

    // OuteTTS is ALSO a codec-only backbone (get_prompt_info fails, pi_ok=false)
    // that drives a 24kHz 2-codebook DAC purely via token ids.  Detect it here
    // (V3 only for this path) so the deferred-pi abort below skips it too.
    codec_common::outetts_version ov{};
    const bool is_outetts =
        detect_outetts_version(lmodel, vocab, &ov) &&
        ov == codec_common::outetts_version::V3;

    // If get_prompt_info was deferred (codec-only gguf) but this is NEITHER a
    // NeuTTS nor an OuteTTS backbone, the model genuinely lacks a usable
    // codec_lm — abort now with the original deferred error rather than fall
    // through to the codec_lm-backed flows (which would dereference an
    // unpopulated `pi`).
    if (!pi_ok && !is_neutts && !is_outetts) {
        out->error = deferred_pi_err;
        llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
        return false;
    }

    // ── OuteTTS V3 free-gen multi-cb Type A decode + DAC ────────────────────
    // OuteTTS (like NeuTTS) is codec-only: the backbone lm_head emits an
    // INTERLEAVED stream of structural tokens (<|word_start|>, word text,
    // <|features|>, <|t_..|>, <|energy_N|>, <|code|>, <|c1_N|>, <|c2_N|>,
    // <|word_end|>, ...) that ends with <|audio_end|>.  We drive it with FREE
    // generation (no flat audio mask — that would forbid the structural tokens)
    // + a 64-token repetition window (mandated by the OuteTTS README), then let
    // the multi-cb observe accumulator strip the non-audio tokens and pack the
    // c1/c2 pairs into the (T, n_q=2) frame layout for DAC decode.  Grammar OFF.
    // Self-contained: frees all resources on every early-return and returns.
    if (is_outetts) {
        // 1) Load the bundled default V3 speaker.  Resolve the asset relative to
        //    a few candidate roots (cwd / repo layout), mirroring the tests.
        static const char * kSpeakerCandidates[] = {
            "assets/speakers/en-female-1-neutral.json",
            "../assets/speakers/en-female-1-neutral.json",
            "docs/superpowers/notes/outetts10-en-female-1-neutral.json",
        };
        std::string speaker_path;
        for (const char * cand : kSpeakerCandidates) {
            std::ifstream probe(cand);
            if (probe.good()) { speaker_path = cand; break; }
        }
        if (speaker_path.empty()) {
            out->error =
                "OuteTTS: bundled default speaker not found "
                "(assets/speakers/en-female-1-neutral.json); run tts-cli from the repo root.";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        codec_common::OutettsSpeaker speaker;
        if (!load_outetts_speaker(speaker_path, &speaker)) {
            out->error = std::string("OuteTTS: failed to parse speaker JSON: ") + speaker_path;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // 2) Build the V3 prefill (header + speaker word-blocks + <|word_start|>
        //    primer + interleaved raw c1/c2 code ids).
        std::vector<llama_token> outetts_prompt_toks =
            build_outetts_v3_prompt(vocab, a.text, speaker);
        if (outetts_prompt_toks.empty()) {
            out->error =
                "OuteTTS: build_outetts_v3_prompt returned empty "
                "(empty text, bad vocab, or out-of-range speaker code)";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // 3) Configure the multi-cb Type A range (c1/c2 offsets + counts,
        //    <|code|> frame sentinel, <|audio_end|> eos) BEFORE any observe.
        int32_t off2[2] = {-1, -1}, cnt2[2] = {0, 0}, sentinel = -1, eos_id = -1;
        if (!outetts_audio_token_ranges(vocab, off2, cnt2, &sentinel, &eos_id)) {
            out->error =
                "OuteTTS: outetts_audio_token_ranges failed "
                "(<|c1_0|>/<|c2_0|>/<|code|>/<|audio_end|> missing from vocab)";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        audio_lm_set_audio_token_ranges(ctx, off2, cnt2, /*n_q=*/2, sentinel, eos_id);
        std::printf("[outetts] V3 multi-cb range: c1=[%d,%d) c2=[%d,%d) sentinel=%d eos=%d; "
                    "%zu prompt tokens\n",
                    off2[0], off2[0] + cnt2[0], off2[1], off2[1] + cnt2[1],
                    sentinel, eos_id, outetts_prompt_toks.size());

        // 4) Type A single-cb-driver kind (the decode drives the DAC via codes).
        pi.model_kind = audio_lm_prompt_info::KIND_TOKEN_SINGLE_CB;

        // Validate prefill fits the backbone context.
        const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
        if ((int32_t) outetts_prompt_toks.size() + 1 >= n_ctx) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "OuteTTS: V3 prefill (%zu tokens) does not fit backbone n_ctx=%d; "
                "raise --max-frames.",
                outetts_prompt_toks.size(), n_ctx);
            out->error = buf;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // OuteTTS sampling defaults (README): temp 1.0, top-k 50, 64-token
        // repetition window with penalty 1.1 (overridable via --rep-penalty).
        const float    ot_temp    = a.has_temp  ? a.temp  : 1.0f;
        const float    ot_top_p   = a.has_top_p ? a.top_p : 1.0f;
        const int32_t  ot_top_k   = a.has_top_k ? a.top_k : 50;
        const uint32_t ot_seed    = a.seed ? a.seed : 0xC0DEC1ABu;
        const float    ot_rep_pen = a.has_rep_penalty ? a.repetition_penalty : 1.1f;
        const int32_t  ot_rep_n   = 64;

        // 5) Free-gen multi-cb decode.  constrain_audio_mask=false ⇒ empty mask
        //    (structural tokens pass), no min-length guard, and NO singular
        //    range re-set (the multi-cb range above is preserved).  off/count/
        //    eos_id are unused in free-gen mode but passed for signature parity.
        int32_t n_frames = 0;
        const char * stop_reason = "max_frames";
        bool ok = run_token_single_cb(ctx, lctx, lmodel, vocab, pi, outetts_prompt_toks,
                                      off2[0], cnt2[0], eos_id, max_frames,
                                      /*min_frames=*/0, ot_seed,
                                      ot_temp, ot_top_p, ot_top_k,
                                      /*constrain_audio_mask=*/false,
                                      ot_rep_pen, ot_rep_n,
                                      &n_frames, &stop_reason);
        llama_free(lctx);
        llama_model_free(lmodel);
        llama_backend_free();
        if (!ok) {
            out->error = std::string("OuteTTS AR failed (") + stop_reason
                       + "); see stderr for the concrete cause";
            audio_lm_free(ctx);
            return false;
        }
        std::printf("OuteTTS AR done: %d frames, stop=%s\n", n_frames, stop_reason);
        if (n_frames == 0) {
            out->error = "OuteTTS: no speech frames generated";
            audio_lm_free(ctx);
            return false;
        }
        // observe_token accumulated the c1/c2 codes into ctx->codes already —
        // decode directly (DAC n_q=2, codes [1,2,n_frames]).
        audio_lm_audio_output pcm;
        if (!audio_lm_decode_audio(ctx, &pcm)) {
            out->error = std::string("OuteTTS: decode_audio failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        fill_result_from_output(pcm, n_frames, stop_reason, out);
        audio_lm_free(ctx);
        return true;
    }

    if (is_neutts) {
        if (a.ref_audio_path.empty() || a.ref_text.empty()) {
            out->error =
                "NeuTTS requires both --ref-audio and --ref-text for in-context learning; "
                "re-run with a reference audio clip and its transcript.";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        // Load reference audio (mono F32).
        std::vector<float> ref_pcm;
        int32_t ref_n = 0, ref_sr = 0;
        std::string ref_err;
        if (!load_ref_audio(a.ref_audio_path, ref_pcm, &ref_n, &ref_sr, &ref_err)) {
            out->error = ref_err;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        // Resample to 16000 Hz (NeuCodec encode_sample_rate; neutts-facts §Step 4).
        const int32_t enc_sr = 16000;
        if (ref_sr > 0 && ref_sr != enc_sr) {
            ref_pcm = resample_mono_f32(ref_pcm, ref_sr, enc_sr);
        }
        ref_n = (int32_t) ref_pcm.size();

        // Load the NeuCodec encoder (base variant, has_encoder=True).
        // The encoder-capable gguf is models/neucodec/neucodec.gguf; the
        // repo-root neucodec.gguf is decode-only and will fail here.
        codec_model_params cmp = codec_model_default_params();
        cmp.use_gpu   = a.use_gpu;
        cmp.n_threads = a.n_threads > 0 ? a.n_threads : 1;
        codec_model * cmodel = codec_model_load_from_file(a.codec_path.c_str(), cmp);
        if (!cmodel) {
            out->error = std::string("NeuTTS: failed to load NeuCodec model from ") + a.codec_path;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        codec_context * cctx = codec_init_from_model(cmodel, codec_context_default_params());
        if (!cctx) {
            out->error = "NeuTTS: codec_init_from_model failed";
            codec_model_free(cmodel);
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        codec_audio au;
        au.data        = ref_pcm.data();
        au.n_samples   = ref_n;
        au.sample_rate = enc_sr;
        au.n_channels  = 1;
        au.pcm_type    = CODEC_PCM_TYPE_F32;

        codec_token_buffer tb;
        std::memset(&tb, 0, sizeof(tb));
        const codec_status enc_status = codec_encode(cctx, &au, &tb, codec_encode_default_params());
        if (enc_status != CODEC_STATUS_SUCCESS) {
            out->error =
                std::string("NeuTTS: codec_encode failed — ensure the encoder-capable gguf is "
                            "used (models/neucodec/neucodec.gguf, has_encoder=true); "
                            "the repo-root neucodec.gguf is decode-only.  Error: ")
                + codec_get_last_error(cctx);
            codec_free(cctx); codec_model_free(cmodel);
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // Extract 1-D code vector (n_q=1; layout data[t*n_q + q]).
        std::vector<int32_t> ref_codes;
        ref_codes.reserve((size_t) tb.n_frames);
        for (int32_t t = 0; t < tb.n_frames; ++t) {
            ref_codes.push_back(tb.data[t]);   // n_q=1 so q=0 only
        }
        codec_token_buffer_free(&tb);
        codec_free(cctx);
        codec_model_free(cmodel);

        // Assemble the ICL prefill token sequence.
        // (ref_text and text pass through as-is here; espeak phonemization is Task 5.)
        neutts_prompt_toks = build_neutts_prompt(vocab, a.ref_text, a.text, ref_codes);
        if (neutts_prompt_toks.empty()) {
            out->error = "NeuTTS: build_neutts_prompt returned empty (bad vocab or out-of-range codes)";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        std::printf("[neutts] ICL prompt: %d ref codes, %zu prompt tokens\n",
                    (int) ref_codes.size(), neutts_prompt_toks.size());

        // ── Type A single-cb decode (self-contained, mirrors chatterbox) ──
        // NeuTTS is backbone-only: the lm_head emits <|speech_N|> directly, so
        // it does NOT route through the pi.model_kind switch / run_codebook_ar.
        int32_t off = -1, count = -1, eos_id = -1;
        if (!neutts_audio_token_range(vocab, &off, &count, &eos_id)) {
            out->error = "NeuTTS: neutts_audio_token_range failed at decode dispatch";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // Validate prefill fits the backbone context.  n_ctx was sized as
        // max(4096, max_frames+512); a long ref clip yields many ref codes, so
        // fail cleanly if prefill + generation budget overruns it.
        const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
        if ((int32_t) neutts_prompt_toks.size() + 1 >= n_ctx) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "NeuTTS: ICL prefill (%zu tokens) does not fit backbone n_ctx=%d; "
                "use a shorter reference clip or raise --max-frames.",
                neutts_prompt_toks.size(), n_ctx);
            out->error = buf;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // NeuTTS sampling defaults.  `pi` is unpopulated for the codec-only
        // gguf (pi.default_* == 0 ⇒ greedy), which tends to loop under a
        // direct-token AR; fall back to the NeuTTS reference knobs
        // (temp 1.0 / top_k 50) unless the user overrode them.
        const float   nt_temp  = a.has_temp  ? a.temp  : (pi_ok ? pi.default_temperature : 1.0f);
        const float   nt_top_p = a.has_top_p ? a.top_p : (pi_ok ? pi.default_top_p       : 1.0f);
        const int32_t nt_top_k = a.has_top_k ? a.top_k : (pi_ok ? pi.default_top_k       : 50);
        const uint32_t nt_seed = a.seed ? a.seed : 0xC0DEC1ABu;

        int32_t n_frames = 0;
        const char * stop_reason = "max_frames";
        // Min-length guard: forbid eos until at least `nt_min_frames` audio
        // frames exist, suppressing the early-eos near-silence failure mode
        // (some seeds sample eos on the FIRST token at temp 1.0 / top-k 50).
        // NeuTTS default is 25 frames (~0.35 s at the NeuCodec frame rate);
        // `--min-len` (a.min_len) overrides.
        const int32_t nt_min_frames = a.min_len > 0 ? a.min_len : 25;
        bool ok = run_token_single_cb(ctx, lctx, lmodel, vocab, pi, neutts_prompt_toks,
                                      off, count, eos_id, max_frames, nt_min_frames, nt_seed,
                                      nt_temp, nt_top_p, nt_top_k,
                                      /*constrain_audio_mask=*/true,
                                      /*rep_penalty=*/1.0f, /*rep_last_n=*/0,
                                      &n_frames, &stop_reason);
        llama_free(lctx);
        llama_model_free(lmodel);
        llama_backend_free();
        if (!ok) {
            // run_token_single_cb prints the concrete cause (mask / prefill /
            // feedback / contract-violation) to stderr; the ctx last_error may
            // be stale from the deferred get_prompt_info, so keep this generic.
            out->error = std::string("NeuTTS AR failed (") + stop_reason
                       + "); see stderr for the concrete cause";
            audio_lm_free(ctx);
            return false;
        }
        std::printf("NeuTTS AR done: %d frames, stop=%s\n", n_frames, stop_reason);
        if (n_frames == 0) {
            out->error = "NeuTTS: no speech frames generated";
            audio_lm_free(ctx);
            return false;
        }
        // observe_token accumulated the codes into ctx->codes already — decode
        // directly (no reset/push; reset would clear the accumulator).
        audio_lm_audio_output pcm;
        if (!audio_lm_decode_audio(ctx, &pcm)) {
            out->error = std::string("NeuTTS: decode_audio failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        fill_result_from_output(pcm, n_frames, stop_reason, out);
        audio_lm_free(ctx);
        return true;
    }

    // ── Prompt tokenize + prefill ─────────────────────────────────
    std::vector<llama_token> toks = tokenize_prompt(vocab, pi, a.text);
    if (toks.empty()) {
        out->error = "empty prompt after tokenize";
        llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
        return false;
    }
    std::printf("prompt: \"%s%s%s\" → %zu tokens\n",
                pi.prompt_prefix.c_str(), a.text.c_str(), pi.prompt_suffix.c_str(), toks.size());

    // ── Speaker conditioning (voice clone) ────────────────────────────
    std::vector<float> speaker_prefix;
    {
        std::vector<float> ref_pcm;
        int32_t ref_n = 0, ref_sr = 0;
        std::string lerr;
        if (!load_ref_audio(a.ref_audio_path, ref_pcm, &ref_n, &ref_sr, &lerr)) {
            out->error = lerr;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        const bool have_ref = !ref_pcm.empty();
        if (audio_lm_has_speaker_enc(ctx) && have_ref) {
            audio_lm_input in;
            in.text = a.text;
            in.ref_pcm = ref_pcm.data();
            in.ref_n_samples = ref_n;
            in.ref_sample_rate = ref_sr;
            audio_lm_prompt sp;
            if (!audio_lm_build_prompt(ctx, in, &sp)) {
                out->error = std::string("build_prompt (speaker) failed: ") + audio_lm_last_error(ctx);
                llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
                return false;
            }
            if (!sp.embeds_prefix.empty() && sp.embeds_prefix_hidden == hidden) {
                speaker_prefix.assign(sp.embeds_prefix.begin(),
                                      sp.embeds_prefix.begin() + hidden);
                std::printf("speaker: x-vector prefix rows=%d hidden=%d (from %s)\n",
                            sp.embeds_prefix_rows, sp.embeds_prefix_hidden, a.ref_audio_path.c_str());
            }
        } else if (!a.ref_audio_path.empty() && !audio_lm_has_speaker_enc(ctx)) {
            std::printf("note: --ref-audio given but model has no speaker encoder; ignoring\n");
        }

        // Faithful no-speaker behavior.  The Qwen3-TTS Base model is a voice-
        // CLONE model: its reference generate_voice_clone() raises
        //   ValueError("Either voice_clone_prompt or ref_audio must be provided")
        // when ref_audio is None — the talker prompt embeds an x-vector row
        // between the think-tags and pad/bos, and running speaker-free is off-
        // spec (unreliable / early-truncating output).  So when the model is a
        // talker (needs the x-vector) and no usable speaker prefix was built,
        // refuse with a clear message rather than emit garbage.
        if (audio_lm_talker_has_projection(ctx) &&
            (speaker_prefix.empty() ||
             (int32_t) speaker_prefix.size() != hidden)) {
            out->error =
                "qwen3-tts is a voice-clone model and requires --ref-audio "
                "(a reference speaker clip); none was provided or the speaker "
                "encode produced no x-vector.  Pass --ref-audio <wav> (any "
                "sample rate / channel count; it is resampled to the encoder's "
                "rate automatically).";
            llama_free(lctx); llama_model_free(lmodel);
            audio_lm_free(ctx); llama_backend_free();
            return false;
        }
    }

    const float temp  = a.has_temp  ? a.temp  : pi.default_temperature;
    const float top_p = a.has_top_p ? a.top_p : pi.default_top_p;
    const int32_t top_k = a.has_top_k ? a.top_k : pi.default_top_k;
    const uint32_t seed = a.seed ? a.seed : 0xC0DEC1ABu;

    // Resolve the backbone constraint (grammar or logit-bias mask).
    const int32_t n_vocab_bb = llama_vocab_n_tokens(vocab);
    const tts_constraint constraint =
        resolve_constraint(lmodel, vocab, pi, a.grammar, a.text, n_vocab_bb);
    if (!constraint.grammar.empty()) {
        std::printf("grammar: %s (%zu bytes)\n",
                    !a.grammar.empty() ? "user-supplied" : "auto (model-derived)",
                    constraint.grammar.size());
    } else if (!constraint.logit_bias.empty()) {
        std::printf("logit-bias: flat audio mask (%zu tokens masked)\n",
                    constraint.logit_bias.size());
    }

    const char * stop_reason = "max_frames";
    int32_t n_frames = 0;
    bool    ar_ok    = false;

    if (pi.is_continuous) {
        ar_ok = run_continuous(ctx, lctx, toks, hidden, max_frames, a,
                               &n_frames, &stop_reason);
    } else if (pi.sequential_text_audio) {
        TextEmbdTable tetab;
        std::string terr;
        if (!tetab.load(a.backbone_path.c_str(), hidden, terr)) {
            out->error = "lfm2: text_embd load failed: " + terr;
            llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
            audio_lm_free(ctx);
            return false;
        }
        ar_ok = run_lfm2_sequential(ctx, lctx, vocab, pi, tetab, toks, hidden,
                                    n_cb, max_frames, seed, temp, top_p,
                                    top_k, &n_frames, &stop_reason);
    } else if (pi.streaming_interleave) {
        TextEmbdTable tetab;
        std::string terr;
        if (!tetab.load(a.backbone_path.c_str(), hidden, terr)) {
            out->error = "realtime: text_embd load failed: " + terr;
            llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
            audio_lm_free(ctx);
            return false;
        }
        const float rep_pen = a.has_rep_penalty ? a.repetition_penalty
                                                : pi.default_repetition_penalty;
        ar_ok = run_realtime_streaming(ctx, lctx, vocab, pi, tetab, a.text,
                                       hidden, n_cb, max_frames, seed,
                                       temp, top_p, top_k, rep_pen,
                                       pi.repetition_window,
                                       &n_frames, &stop_reason);
    } else {
        ar_ok = run_codebook_ar(ctx, lctx, lmodel, vocab, pi, toks, hidden, n_cb,
                                max_frames, seed, temp, top_p, top_k, constraint,
                                speaker_prefix, a.text, &n_frames, &stop_reason);
    }

    llama_free(lctx);
    llama_model_free(lmodel);
    llama_backend_free();

    if (!ar_ok) {
        out->error = std::string("AR loop failed: ") + audio_lm_last_error(ctx);
        audio_lm_free(ctx);
        return false;
    }
    std::printf("AR loop done: %d frames, stop=%s\n", n_frames, stop_reason);
    if (n_frames == 0) {
        out->error = "no audio frames generated";
        audio_lm_free(ctx);
        return false;
    }

    audio_lm_audio_output pcm;
    if (!audio_lm_decode_audio(ctx, &pcm)) {
        out->error = std::string("decode_audio failed: ") + audio_lm_last_error(ctx);
        audio_lm_free(ctx);
        return false;
    }
    fill_result_from_output(pcm, n_frames, stop_reason, out);
    audio_lm_free(ctx);
    return true;
}

}  // namespace codec_common
