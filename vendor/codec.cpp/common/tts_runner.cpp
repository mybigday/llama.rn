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
#include "kitten_text.h"
#include "utils/wav_io.h"
#include "common.h"     // common_params_sampling, common_grammar
#include "sampling.h"   // common_sampler_*
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
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
// llama.cpp's penalties sampler clamps penalty_last_n to >= 0 and treats 0 as
// "disabled" (common's old -1 -> n_ctx mapping is commented out), so "-1 =
// whole history" silently turned repetition penalties off.  Use a window
// larger than any generation instead.
static constexpr int32_t kRepPenaltyFullHistory = 1 << 16;

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
        sp.penalty_last_n = rep_penalty != 1.0f ? (rep_last_n > 0 ? rep_last_n : kRepPenaltyFullHistory) : 0;
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
            const int32_t last_n = rep_last_n > 0 ? rep_last_n : kRepPenaltyFullHistory;
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

    // HF LogitsProcessorList order used by Chatterbox Turbo
    // (t3.py::inference_turbo): temp → top_k → top_p → repetition penalty
    // → dist(seed).  Penalties run on the already-truncated candidates, so
    // the kept set is chosen before the penalty, as in the reference.
    void init_sampled_penalty_last(uint32_t seed, float temp, int32_t top_k, float top_p,
                                   float rep_penalty) {
        reset();
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        chain = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        if (top_p > 0.0f && top_p < 1.0f)
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        if (rep_penalty != 1.0f) {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(kRepPenaltyFullHistory, rep_penalty, 0.0f, 0.0f));
            has_penalties = true;
        }
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    }

    // HF generate() warper order used by kittenml (logits.py): temp → top_k
    // → top_p → min_p → dist(seed).  Penalties are applied by the caller.
    void init_sampled_hf_order(uint32_t seed, float temp, int32_t top_k, float top_p, float min_p) {
        reset();
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        sp.no_perf = true;
        chain = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temp));
        if (top_k > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
        if (top_p > 0.0f && top_p < 1.0f) llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
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
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static bool timed_decode_audio(audio_lm_context * ctx, audio_lm_audio_output * pcm, tts_runner_result * out) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = audio_lm_decode_audio(ctx, pcm);
    out->decode_ms += ms_since(t0);
    return ok;
}

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
//     `grammar` (GBNF, may be empty) constrains that stream instead: OuteTTS
//     passes the prompt-derived word-block grammar (outetts_build_grammar).
bool run_token_single_cb(audio_lm_context * ctx, llama_context * lctx,
                         const llama_model * lmodel, const llama_vocab * vocab,
                         const audio_lm_prompt_info & /*pi*/,
                         const std::vector<llama_token> & prompt_toks,
                         int32_t off, int32_t count, int32_t eos_id,
                         int32_t max_frames, int32_t min_frames, uint32_t seed,
                         float temp, float top_p, int32_t top_k,
                         bool constrain_audio_mask, float rep_penalty,
                         int32_t rep_last_n, const std::string & grammar,
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

    // 3) Backbone sampler with the mask (or the grammar) attached.
    BackboneSampler bbsmpl;
    std::string berr;
    if (!bbsmpl.build(lmodel, seed, temp, top_k, top_p, /*min_p=*/0.0f,
                      rep_penalty, rep_last_n, grammar, mask, &berr)) {
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

    // 5) AR loop — sample → observe → feed back, until eos, `max_frames`
    //    complete frames, or a full context (OuteTTS spends several tokens
    //    per frame, so the step count is not the frame count).
    const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
    while (*out_frames < max_frames) {
        if (n_past >= n_ctx) { *out_stop = "n_ctx"; break; }
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
        if (act != OBSERVE_STOP && llama_vocab_is_eog(vocab, tok)) {
            // End of generation without the audio-end token (e.g. a model that
            // spelled it out of text pieces): nothing more will follow.
            act = OBSERVE_STOP;
        }
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
    // CLI flags win, then the GGUF's reference sampling defaults
    // (`codec.lm.chatterbox.sampling.*`, < 0 = unset), then the Llama-T3
    // regime.  Turbo has no CFG lane at all.
    const bool turbo = ci->is_turbo != 0;
    auto pick = [](bool has, float cli, float meta, float fallback) {
        return has ? cli : (meta >= 0.0f ? meta : fallback);
    };
    const float cfg_weight = turbo ? 0.0f : pick(a.has_cfg_weight, a.cfg_weight, ci->cfg_weight, 0.5f);
    const float temperature = pick(a.has_temp, a.temp, ci->temperature, 0.8f);
    const float top_p = pick(a.has_top_p, a.top_p, ci->top_p, 1.0f);
    const float min_p = pick(a.has_min_p, a.min_p, ci->min_p, 0.05f);
    const float rep_pen = pick(a.has_rep_penalty, a.repetition_penalty, ci->repetition_penalty, 1.2f);
    const int32_t top_k = a.has_top_k ? a.top_k : (ci->top_k >= 0 ? ci->top_k : 0);

    std::vector<int32_t> text_ids;
    if (ci->backbone_tokenizer) {
        // Turbo: punc_norm on the codec side, then the backbone's GPT-2 BPE
        // (paralinguistic tags such as "[laugh]" are added tokens).
        std::vector<char> normed(a.text.size() * 2 + 64);
        if (codec_lm_chatterbox_normalize_text(lm, a.text.c_str(), normed.data(),
                                               (int32_t) normed.size()) != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "chatterbox normalize failed: %s\n", codec_lm_get_last_error(lm));
            return false;
        }
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
        const int32_t n_chars = (int32_t) std::strlen(normed.data());
        text_ids.resize((size_t) n_chars + 16);
        int32_t n = llama_tokenize(vocab, normed.data(), n_chars, text_ids.data(),
                                   (int32_t) text_ids.size(), /*add_special=*/false,
                                   /*parse_special=*/true);
        if (n < 0) {
            text_ids.resize((size_t) -n);
            n = llama_tokenize(vocab, normed.data(), n_chars, text_ids.data(),
                               (int32_t) text_ids.size(), false, true);
        }
        if (n < 0) { std::fprintf(stderr, "chatterbox backbone tokenize failed\n"); return false; }
        text_ids.resize((size_t) n);
    } else {
        text_ids.resize(a.text.size() + 64);
        int32_t n_text = 0;
        if (codec_lm_chatterbox_tokenize(lm, a.text.c_str(), text_ids.data(),
                                         (int32_t) text_ids.size(), &n_text) != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "chatterbox tokenize failed: %s\n", codec_lm_get_last_error(lm));
            return false;
        }
        text_ids.resize(n_text);
    }
    const int32_t n_text = (int32_t) text_ids.size();
    std::printf("chatterbox%s: %d text tokens, cfg_weight=%.2f temp=%.2f top_k=%d min_p=%.2f top_p=%.2f rep=%.2f\n",
                turbo ? " turbo" : "", n_text, cfg_weight, temperature, top_k, min_p, top_p, rep_pen);

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
    const int32_t seq_len_cap = cond_rows + (n_text + 2) + 2;  // also bounds Turbo's single-BOS layout
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
    // Turbo uses the HF warper order instead and its history is only the
    // generated tokens (inference_turbo's input_ids exclude the BOS after
    // step 0).
    SamplerChain smpl;
    if (temperature <= 0.0f) smpl.init_greedy();
    else if (turbo) {
        smpl.init_sampled_penalty_last(a.seed ? a.seed : 0xC0DEC1ABu, temperature, top_k,
                                       top_p, rep_pen);
    } else {
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
    // Turbo appends S3GEN_SIL × 3 before S3Gen (tts_turbo.py::generate).
    if (ci->silence_token >= 0) {
        for (int32_t i = 0; i < ci->n_trailing_silence; ++i) out_codes->push_back(ci->silence_token);
    }
    return true;
}

// ── CosyVoice (v1 300M / BreezyVoice, CosyVoice2, CosyVoice3) ──────────
//
// Prompt (cosyvoice/llm/llm.py inference):
//   v1:    [sos] [spk row] [text_encoder(prompt_text ++ text)] [task_id] [speech_emb(prompt tokens)]
//   v2/v3: [sos] [backbone embed(prompt_text ++ text)]          [task_id] [speech_emb(prompt tokens)]
// Modes follow CosyVoiceFrontEnd: zero-shot when the speaker carries prompt
// tokens and a transcript (--ref-text), cross-lingual when it has prompt
// tokens but no transcript (the LLM sees neither), instruct when --instruct
// is given (instruct text replaces the prompt text, no prompt tokens), SFT
// for x-vector-only speakers.  Sampling is the reference's ras_sampling
// (nucleus top_p/top_k, then a full-distribution redraw when the pick
// repeats inside the last win_size tokens) or BreezyVoice's plain top-k with
// eos redraw; no temperature.

namespace {

struct CosyVoiceSampler {
    std::mt19937 rng;
    explicit CosyVoiceSampler(uint32_t seed) : rng(seed) {}

    static void softmax(const std::vector<float> & x, std::vector<double> * p) {
        double mx = -INFINITY;
        for (float v : x) mx = std::max(mx, (double) v);
        p->resize(x.size());
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) { (*p)[i] = std::isinf(x[i]) && x[i] < 0 ? 0.0 : std::exp(x[i] - mx); s += (*p)[i]; }
        for (double & v : *p) v /= s;
    }
    int32_t draw(const std::vector<double> & w, const std::vector<int32_t> & ids) {
        std::discrete_distribution<int32_t> d(w.begin(), w.end());
        return ids[(size_t) d(rng)];
    }
    // nucleus_sampling: sorted probs, keep while cum < top_p and count < top_k.
    int32_t nucleus(const std::vector<double> & p, float top_p, int32_t top_k) {
        std::vector<int32_t> order(p.size());
        for (size_t i = 0; i < p.size(); ++i) order[i] = (int32_t) i;
        const size_t k = std::min<size_t>((size_t) std::max(top_k, 1), order.size());
        std::partial_sort(order.begin(), order.begin() + (long) k, order.end(),
                          [&](int32_t a, int32_t b) { return p[(size_t) a] > p[(size_t) b] || (p[(size_t) a] == p[(size_t) b] && a < b); });
        std::vector<double> w;
        std::vector<int32_t> ids;
        double cum = 0.0;
        for (size_t i = 0; i < k && cum < top_p; ++i) {
            cum += p[(size_t) order[i]];
            w.push_back(p[(size_t) order[i]]);
            ids.push_back(order[i]);
        }
        return draw(w, ids);
    }
    int32_t full(const std::vector<double> & p) {
        std::vector<int32_t> ids(p.size());
        for (size_t i = 0; i < p.size(); ++i) ids[i] = (int32_t) i;
        return draw(p, ids);
    }
};

}  // namespace

bool run_cosyvoice(audio_lm_context * ctx, llama_context * lctx, codec_lm * lm,
                   const codec_lm_cosyvoice_info * ci, const tts_runner_params & a,
                   int32_t hidden, int32_t max_frames, std::vector<int32_t> * out_codes,
                   int32_t * out_frames, const char ** out_stop) {
    const codec_speaker * spk = audio_lm_decode_speaker(ctx);
    if (spk == nullptr) {
        std::fprintf(stderr, "cosyvoice: no speaker (pass --ref-audio, --voice or --speaker-file)\n");
        return false;
    }
    std::vector<int32_t> prompt_tokens;
    std::vector<float> xvec;
    {
        enum codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * d = nullptr;
        if (codec_speaker_get_slot(spk, "prompt_token", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_I32) {
            const auto * v = static_cast<const int32_t *>(d);
            prompt_tokens.assign(v, v + ne[0]);
        }
        if (codec_speaker_get_slot(spk, "embedding", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_F32) {
            const auto * v = static_cast<const float *>(d);
            xvec.assign(v, v + ne[0] * ne[1]);
        }
    }
    std::string ref_text = a.ref_text;
    if (ref_text.empty() && codec_speaker_ref_text(spk) != nullptr) ref_text = codec_speaker_ref_text(spk);
    const std::string prefix = ci->prompt_prefix != nullptr ? ci->prompt_prefix : "";

    // Mode → (LLM prompt text, LLM prompt speech tokens, target text).
    std::string prompt_text, tts_text = a.text;
    std::vector<int32_t> llm_prompt_tokens;
    bool llm_xvec = true;   // v1: x-vector row in the LLM prompt
    const char * mode = "sft";
    if (!a.instruct.empty()) {
        // v1 frontend_instruct: instruct text as prompt text, no x-vector in
        // the LLM ("information leakage"); v2/v3 frontend_instruct2: instruct
        // text as prompt text, no prompt speech tokens.  The instruct text
        // ends with <|endofprompt|> (v3 also starts with the system prompt).
        mode = "instruct";
        prompt_text = a.instruct;
        llm_xvec = false;
        if (ci->version != 1 && prompt_text.find("<|endofprompt|>") == std::string::npos) {
            if (!prefix.empty()) prompt_text = prefix.substr(0, prefix.find("<|endofprompt|>")) + " " + prompt_text;
            prompt_text += "<|endofprompt|>";
        }
    } else if (!prompt_tokens.empty() && !ref_text.empty()) {
        mode = "zero_shot";
        prompt_text = prefix + ref_text;
        llm_prompt_tokens = prompt_tokens;
    } else if (!prompt_tokens.empty()) {
        mode = "cross_lingual";
        tts_text = prefix + tts_text;
    } else if (!prefix.empty()) {
        tts_text = prefix + tts_text;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
    auto tokenize = [&](const std::string & s) {
        std::vector<llama_token> t(s.size() + 16);
        int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), t.data(), (int32_t) t.size(), false, true);
        if (n < 0) { t.resize((size_t) -n); n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), t.data(), (int32_t) t.size(), false, true); }
        t.resize((size_t) std::max(n, 0));
        return t;
    };
    const std::vector<llama_token> ptxt = prompt_text.empty() ? std::vector<llama_token>() : tokenize(prompt_text);
    const std::vector<llama_token> ttxt = tokenize(tts_text);
    if (ttxt.empty()) { std::fprintf(stderr, "cosyvoice: empty text\n"); return false; }
    std::vector<llama_token> all_txt(ptxt);
    all_txt.insert(all_txt.end(), ttxt.begin(), ttxt.end());

    const int32_t text_len = (int32_t) ttxt.size();
    const int32_t min_len = (int32_t) (text_len * ci->min_token_text_ratio);
    int32_t max_len = (int32_t) (text_len * ci->max_token_text_ratio);
    if (max_frames > 0) max_len = std::min(max_len, max_frames);
    std::printf("cosyvoice v%d (%s): %zu prompt-text + %d text tokens, %zu prompt speech tokens, min_len=%d max_len=%d, %s\n",
                ci->version, mode, ptxt.size(), text_len, llm_prompt_tokens.size(), min_len, max_len,
                ci->ras_sampling ? "ras" : "top_k");

    // ---- prefill ----
    const int32_t H = hidden;
    std::vector<float> row((size_t) H);
    int32_t n_past = 0;
    if (codec_lm_cosyvoice_special_embd(lm, 0, row.data(), H) != CODEC_STATUS_SUCCESS) return false;
    std::vector<float> head(row);   // sos [+ v1 spk row + text rows]
    int32_t n_head = 1;
    if (ci->text_in_backbone) {
        if (!decode_embed_block(lctx, head.data(), H, 1, n_past)) return false;
        n_past += 1;
        if (!decode_tokens(lctx, all_txt, n_past, false)) return false;
        n_past += (int32_t) all_txt.size();
    } else {
        std::vector<int32_t> ids(all_txt.begin(), all_txt.end());
        const int32_t cap = (int32_t) ids.size() + 1;
        std::vector<float> rows((size_t) cap * H);
        int32_t n_rows = 0;
        const bool use_xvec = llm_xvec && !xvec.empty();
        if (codec_lm_cosyvoice_text_rows(lm, ids.data(), (int32_t) ids.size(),
                                         use_xvec ? xvec.data() : nullptr, use_xvec ? (int32_t) xvec.size() : 0,
                                         rows.data(), cap, &n_rows) != CODEC_STATUS_SUCCESS) {
            std::fprintf(stderr, "cosyvoice text rows failed: %s\n", codec_lm_get_last_error(lm));
            return false;
        }
        head.insert(head.end(), rows.begin(), rows.begin() + (long) n_rows * H);
        n_head += n_rows;
        if (!decode_embed_block(lctx, head.data(), H, n_head, n_past)) return false;
        n_past += n_head;
    }
    std::vector<float> tail((size_t) (1 + llm_prompt_tokens.size()) * H);
    if (codec_lm_cosyvoice_special_embd(lm, 1, tail.data(), H) != CODEC_STATUS_SUCCESS ||
        codec_lm_cosyvoice_speech_embd(lm, llm_prompt_tokens.data(), (int32_t) llm_prompt_tokens.size(),
                                       tail.data() + H, (int32_t) llm_prompt_tokens.size()) != CODEC_STATUS_SUCCESS) {
        std::fprintf(stderr, "cosyvoice prompt rows failed: %s\n", codec_lm_get_last_error(lm));
        return false;
    }
    if (!decode_embed_block(lctx, tail.data(), H, 1 + (int32_t) llm_prompt_tokens.size(), n_past)) return false;
    n_past += 1 + (int32_t) llm_prompt_tokens.size();

    // ---- AR loop ---- (bounded by the backbone context: a 30 s v1 prompt is 1500 rows)
    max_len = std::min(max_len, (int32_t) llama_n_ctx(lctx) - n_past - 1);
    const bool greedy = a.has_temp && a.temp <= 0.0f;
    const int32_t top_k = a.has_top_k ? a.top_k : ci->top_k;
    const float top_p = a.has_top_p ? a.top_p : ci->top_p;
    CosyVoiceSampler smp(a.seed);
    std::vector<int32_t> decoded;
    std::vector<double> prob;
    *out_stop = "max_len";
    for (int32_t step = 0; step < max_len; ++step) {
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (h == nullptr || !audio_lm_step_begin(ctx, h, H)) return false;
        int32_t cb = 0, nlog = 0;
        const float * lg = audio_lm_step_logits(ctx, &cb, &nlog);
        if (lg == nullptr || nlog <= 0) return false;
        std::vector<float> logits(lg, lg + nlog);
        audio_lm_step_push_code(ctx, 0);
        int32_t dummy = 0;
        audio_lm_step_finish(ctx, &dummy, 1);

        const bool ignore_eos = step < min_len;
        int32_t id = 0;
        if (greedy) {
            if (ignore_eos) logits[(size_t) ci->ignore_eos_id] = -INFINITY;
            id = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else if (ci->ras_sampling) {
            if (ignore_eos) logits[(size_t) ci->ignore_eos_id] = -INFINITY;
            CosyVoiceSampler::softmax(logits, &prob);
            id = smp.nucleus(prob, top_p, top_k);
            const int32_t win = ci->win_size;
            int32_t rep = 0;
            for (size_t i = decoded.size() > (size_t) win ? decoded.size() - (size_t) win : 0; i < decoded.size(); ++i) {
                rep += decoded[i] == id;
            }
            if ((float) rep >= (float) win * ci->tau_r) {
                logits[(size_t) id] = -INFINITY;
                CosyVoiceSampler::softmax(logits, &prob);
                id = smp.full(prob);
            }
        } else {
            // TransformerLM.sampling_ids: top-k multinomial, redraw while eos is ignored.
            CosyVoiceSampler::softmax(logits, &prob);
            std::vector<int32_t> order(prob.size());
            for (size_t i = 0; i < prob.size(); ++i) order[i] = (int32_t) i;
            const size_t k = std::min<size_t>((size_t) std::max(top_k, 1), order.size());
            std::partial_sort(order.begin(), order.begin() + (long) k, order.end(),
                              [&](int32_t x, int32_t y) { return prob[(size_t) x] > prob[(size_t) y]; });
            std::vector<int32_t> ids(order.begin(), order.begin() + (long) k);
            std::vector<double> w;
            for (int32_t i : ids) w.push_back(prob[(size_t) i]);
            for (int32_t tries = 0; tries < 100; ++tries) {
                id = smp.draw(w, ids);
                if (!ignore_eos || id != ci->ignore_eos_id) break;
            }
        }
        if (id >= ci->stop_begin && id < ci->stop_end) { *out_stop = "eos"; break; }
        decoded.push_back(id);
        (*out_frames)++;
        if (codec_lm_cosyvoice_speech_embd(lm, &id, 1, row.data(), 1) != CODEC_STATUS_SUCCESS ||
            !decode_embed(lctx, row.data(), H, n_past)) {
            std::fprintf(stderr, "cosyvoice step %d decode failed\n", step);
            return false;
        }
        ++n_past;
    }

    // v3 llm_job: after more than max_silent_run consecutive silent/breath
    // tokens, further ones are dropped from the flow input (they were fed back).
    int32_t run = 0;
    for (int32_t id : decoded) {
        bool silent = false;
        for (int32_t i = 0; i < ci->n_silent_tokens; ++i) silent = silent || ci->silent_tokens[i] == id;
        run = silent ? run + 1 : 0;
        if (silent && ci->max_silent_run > 0 && run > ci->max_silent_run) continue;
        if (id >= 0 && id < ci->speech_token_size) out_codes->push_back(id);
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

// ── OuteTTS V1/V2/V3 detection ───────────────────────────────────────────────────
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
    // V2 (0.3) signature: space + audio-end tokens, but NOT the V3 markers.
    const bool v2 = !v3 && has("<|space|>") && has("<|audio_end|>");
    // V1 (0.2): text_sep + code_start/code_end, no <|space|>.
    const bool v1 = !v3 && !v2 && has("<|text_sep|>") && has("<|code_start|>") && has("<|audio_end|>");

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
    if (v1) {
        *out = codec_common::outetts_version::V1;
        std::printf("[tts] OuteTTS detection: V1 (general.name=\"%s\")\n", name);
        return true;
    }
    std::printf("[tts] OuteTTS detection: negative — not OuteTTS (general.name=\"%s\")\n", name);
    return false;
}

// The paths that sample outside BackboneSampler (codec_lm heads, KittenTTS's
// HF-order chain) cannot apply a user GBNF; say so instead of dropping it.
static void warn_grammar_unused(const std::string & grammar, const char * model) {
    if (!grammar.empty()) {
        std::fprintf(stderr, "WARN: --grammar is not supported for %s and is ignored\n", model);
    }
}

// Resolve the full constraint for a backbone synthesis request.
// Resolution order (first match wins):
//   1. Explicit user GBNF grammar.
//   2. Codec-metadata GBNF grammar (MOSS-TTSD tts_auto_grammar).
//   3. Flat audio-token logit-bias mask for Type A direct-audio models.
// (OuteTTS and NeuTTS return before this through run_token_single_cb.)
// Returns an empty constraint for models with no constraint (CSM / Chatterbox /
// Qwen3-TTS / LFM2 / Realtime) — identical behaviour to before this change.
static tts_constraint resolve_constraint(const audio_lm_prompt_info & pi,
                                         const std::string & user_grammar,
                                         const std::string & text,
                                         int32_t n_vocab) {
    tts_constraint c;
    // 1) explicit user grammar wins.
    if (!user_grammar.empty()) { c.grammar = user_grammar; return c; }
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
    // Count contiguous <|speech_N|> tokens from N=0 upward, reading the
    // piece at off + N (a name lookup per N scans the vocab 65536 times).
    const int32_t nv = llama_vocab_n_tokens(vocab);
    int32_t n = 0;
    char buf[32];
    for (; off + n < nv; ++n) {
        std::snprintf(buf, sizeof(buf), "<|speech_%d|>", n);
        const char * t = llama_vocab_get_text(vocab, off + n);
        if (t == nullptr || std::strcmp(t, buf) != 0) break;
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

// ── OuteTTS prompt (speaker = codec_speaker ref_text + audio_prompt) ─────────
//
// The speaker's audio section is baked into the codec GGUF as the backbone's
// token ids (`audio_prompt`, scripts/bake-speakers.py); `ref_text` is its text
// section.  Prefill per upstream outetts/version/vN/prompt_processor.py:
//
//   V1 (0.2)  <|im_start|>\n<|text_start|>REF<|text_sep|>IN<|text_end|>\n<|audio_start|>\n AUDIO
//   V2 (0.3)  <|im_start|>\n<|text_start|>REF<|space|>IN<|text_end|>\n<|audio_start|>\n AUDIO
//   V3 (1.0)  <|im_start|>\n<|text_start|>REF␣IN<|text_end|>\n<|audio_start|>\n AUDIO <|word_start|>
//
// V1/V2 words are separated by the version's separator token (REF and IN are
// space-separated words here); V3 joins the texts like merge_speaker_text.

namespace {

// UTF-8 codepoints of `s` (malformed bytes pass through as themselves).
std::vector<uint32_t> utf8_codepoints(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char b = (unsigned char) s[i];
        size_t len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 1;
        if (i + len > s.size()) len = 1;
        uint32_t cp = len == 1 ? b : (b & (0xFF >> (len + 1)));
        for (size_t k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string trim_ws(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

// v3 PromptProcessor.merge_speaker_text.
std::string outetts_v3_merge(const std::string & speaker_text, const std::string & input) {
    const std::string spk = trim_ws(speaker_text);
    bool cjk = false, hangul = false;
    for (uint32_t c : utf8_codepoints(spk)) {
        if ((c >= 0x3040 && c <= 0x30FF) || (c >= 0x4E00 && c <= 0x9FFF)) cjk = true;
        if (c >= 0xAC00 && c <= 0xD7AF) hangul = true;
    }
    (void) hangul;  // Hangul uses ". " like Latin text
    std::string rs;
    if (!spk.empty()) {
        const std::vector<uint32_t> cps = utf8_codepoints(spk);
        const uint32_t last = cps.back();
        const bool ends = cjk ? (last == 0x3002 || last == 0xFF1F || last == 0xFF01 || last == '?' || last == '!')
                              : (last == '.' || last == '?' || last == '!');
        rs = !ends ? (cjk ? "\xE3\x80\x82" : ". ") : (cjk ? "" : " ");
    }
    return spk + rs + trim_ws(input);
}

}  // namespace

static std::string outetts_input_words(codec_common::outetts_version v, const std::string & text) {
    using codec_common::outetts_version;
    if (v == outetts_version::V3) return trim_ws(text);
    // V1: llama.cpp tts process_text (numbers to words is the caller's job).
    // V2: also keeps . ! ? , " as punctuation tokens glued to their word
    //     (TextProcessor.join_punctuation, ASCII subset).
    std::string lower;
    for (char c : text) {
        const char l = (char) std::tolower((unsigned char) c);
        lower += (l == '-' || l == '_' || l == '/' || l == '\\' || (v == outetts_version::V1 && (l == ',' || l == '.')))
                     ? ' ' : l;
    }
    auto punct = [](char c) -> const char * {
        switch (c) {
            case '.': return "<|period|>";
            case '!': return "<|exclamation_mark|>";
            case '?': return "<|question_mark|>";
            case ',': return "<|comma|>";
            case '"': return "<|double_quote|>";
            default:  return nullptr;
        }
    };
    std::vector<std::string> words;
    std::string cur, before, after;
    auto flush = [&]() {
        if (!cur.empty()) {
            words.push_back(before + cur + after);
        } else if (!words.empty()) {
            words.back() += before + after;   // standalone punctuation joins the previous word
        }
        cur.clear(); before.clear(); after.clear();
    };
    for (size_t i = 0; i <= lower.size(); ++i) {
        const char c = i < lower.size() ? lower[i] : ' ';
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { flush(); continue; }
        if (c >= 'a' && c <= 'z') {
            cur += c;
        } else if (v == outetts_version::V2 && punct(c) != nullptr) {
            if (i + 2 < lower.size() && c == '.' && lower[i + 1] == '.' && lower[i + 2] == '.') {
                (cur.empty() ? before : after) += "<|ellipsis|>";
                i += 2;
            } else {
                (cur.empty() ? before : after) += punct(c);
            }
        }
    }
    std::string out;
    for (const auto & w : words) out += (out.empty() ? "" : " ") + w;
    return out;
}

static std::vector<llama_token> build_outetts_prompt(const llama_vocab * vocab, codec_common::outetts_version v,
                                                     const std::string & input_words, const codec_speaker * spk,
                                                     std::string * err) {
    using codec_common::outetts_version;
    codec_speaker_slot_type ty;
    int64_t ne[4];
    const void * data = nullptr;
    const char * ref_text = spk != nullptr ? codec_speaker_ref_text(spk) : nullptr;
    if (spk == nullptr || ref_text == nullptr || !codec_speaker_get_slot(spk, "audio_prompt", &ty, ne, &data) ||
        ty != CODEC_SPEAKER_SLOT_I32) {
        *err = "OuteTTS needs a speaker with ref_text + audio_prompt (a codec GGUF with baked voices, "
               "scripts/bake-speakers.py)";
        return {};
    }
    if (input_words.empty()) { *err = "OuteTTS: no words in the input text"; return {}; }
    auto sep_join = [](const std::string & words, const char * sep) {
        std::string o;
        for (char c : words) o += c == ' ' ? std::string(sep) : std::string(1, c);
        return o;
    };
    std::string text;
    if (v == outetts_version::V1) text = sep_join(std::string(ref_text) + " " + input_words, "<|text_sep|>");
    else if (v == outetts_version::V2) text = sep_join(std::string(ref_text) + " " + input_words, "<|space|>");
    else text = outetts_v3_merge(ref_text, input_words);

    std::vector<llama_token> toks = tokenize_str(
        vocab, "<|im_start|>\n<|text_start|>" + text + "<|text_end|>\n<|audio_start|>\n",
        /*add_bos=*/false, /*parse_special=*/true);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const auto * ids = static_cast<const int32_t *>(data);
    for (int64_t i = 0; i < ne[0]; ++i) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            *err = "OuteTTS: speaker audio_prompt token out of the backbone vocab (codec GGUF baked for another model?)";
            return {};
        }
        toks.push_back(ids[i]);
    }
    if (v == outetts_version::V3) {
        const std::vector<llama_token> ws = tokenize_str(vocab, "<|word_start|>", false, true);
        toks.insert(toks.end(), ws.begin(), ws.end());
    }
    return toks;
}

// ── KittenTTS 2 (Flow: Qwen3 speech LM → S3 tokens → MeanFlow S3Gen) ─────
// Port of kittenml/kittentts2 (model.py, prompt.py, logits.py).  The codec
// GGUF is a chatterbox_s3g model with the LM contract under codec.lm.kitten.*
// and the voice in its speakers: lm_cond (position-0 embedding row),
// ref_tokens + ref_text (in-context reference) and the S3G decoder slots.

bool detect_kitten_tts2(const llama_vocab * vocab) {
    return vocab_has_piece(vocab, "<|a_0|>")
        && vocab_has_piece(vocab, "<final_seg>")
        && vocab_has_piece(vocab, "[START]");
}

struct kitten_config {
    int32_t audio_base = 0, n_audio = 0;
    int32_t speech_start = 0, speech_end = 0, text_start = 0, start = 0, stop = 0, final_seg = 0;
    int32_t ref_speech_start = 0, ref_speech_end = 0, ref_text_start = 0, ref_text_end = 0;
    float temp = 0.8f, top_p = 0.8f, min_p = 0.0f;
    int32_t top_k = 50;
    float rep_penalty = 1.1f;
    int32_t rep_window = 50;
    float run_penalty = 1.3f;
    int32_t run_grace = 10;
    int32_t silence = 4299, n_trailing_silence = 3;
    bool use_reference_prompt = true;
    std::string emotion_control = "{emo: 1}";
    int32_t chunk_chars = 380, chunk_min_chars = 130, max_new_tokens = 1000;
    float chunk_gap_s = 0.16f;
};

static bool kitten_read_config(const audio_lm_context * ctx, const std::string & preset_req,
                               kitten_config * c, std::string * err) {
    auto str = [&](const std::string & k) { return audio_lm_meta_str(ctx, ("codec.lm.kitten." + k).c_str()); };
    auto i32 = [&](const std::string & k, int32_t * dst) {
        if (const char * v = str(k)) *dst = std::atoi(v);
    };
    auto f32 = [&](const std::string & k, float * dst) {
        if (const char * v = str(k)) *dst = (float) std::atof(v);
    };
    if (!str("token.audio_id_base")) {
        *err = "codec GGUF lacks codec.lm.kitten.* (convert with --model-type kitten_tts2)";
        return false;
    }
    i32("token.audio_id_base", &c->audio_base);           i32("token.num_audio_tokens", &c->n_audio);
    i32("token.speech_start_id", &c->speech_start);       i32("token.speech_end_id", &c->speech_end);
    i32("token.text_start_id", &c->text_start);           i32("token.start_id", &c->start);
    i32("token.stop_id", &c->stop);                       i32("token.final_seg_id", &c->final_seg);
    i32("token.reference_speech_start_id", &c->ref_speech_start);
    i32("token.reference_speech_end_id", &c->ref_speech_end);
    i32("token.reference_text_start_id", &c->ref_text_start);
    i32("token.reference_text_end_id", &c->ref_text_end);
    const char * dp = str("default_preset");
    const std::string preset = !preset_req.empty() ? preset_req : (dp ? dp : "stable");
    if (!str("preset." + preset + ".temperature")) {
        *err = "unknown KittenTTS preset '" + preset + "'";
        return false;
    }
    f32("preset." + preset + ".temperature", &c->temp);
    f32("preset." + preset + ".top_p", &c->top_p);
    f32("preset." + preset + ".min_p", &c->min_p);
    float top_k = (float) c->top_k;
    f32("preset." + preset + ".top_k", &top_k);
    c->top_k = (int32_t) top_k;
    f32("repetition_penalty", &c->rep_penalty);           i32("repetition_window", &c->rep_window);
    f32("token_run_penalty", &c->run_penalty);            i32("token_run_grace", &c->run_grace);
    i32("silence_token", &c->silence);                    i32("n_trailing_silence", &c->n_trailing_silence);
    if (const char * v = str("use_reference_prompt")) c->use_reference_prompt = std::string(v) == "true";
    if (const char * v = str("emotion_control")) c->emotion_control = v;
    i32("chunk_chars", &c->chunk_chars);                  i32("chunk_min_chars", &c->chunk_min_chars);
    f32("chunk_gap_s", &c->chunk_gap_s);                  i32("max_new_tokens", &c->max_new_tokens);
    return true;
}

static std::string strip_copy(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

static std::vector<llama_token> kitten_tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> toks(text.size() + 16);
    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(),
                               (int32_t) toks.size(), /*add_special=*/false, /*parse_special=*/false);
    if (n < 0) {
        toks.resize((size_t) -n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(),
                           (int32_t) toks.size(), false, false);
    }
    toks.resize((size_t) std::max(0, n));
    return toks;
}

// logits.py: TokenRunPenalty then RepetitionPenaltyExempt, both on raw logits.
static void kitten_apply_penalties(const kitten_config & c, const std::vector<llama_token> & seq,
                                   size_t prompt_len, float * logits) {
    const llama_token last = seq.back();
    if (last >= c.audio_base && last < c.audio_base + c.n_audio) {
        int32_t run = 0;
        for (size_t i = seq.size(); i > 0 && seq[i - 1] == last; --i) ++run;
        if (run >= c.run_grace) logits[last] -= (float) (run - c.run_grace + 1) * c.run_penalty;
    }
    if (c.rep_penalty != 1.0f) {
        const size_t begin = std::max(prompt_len, c.rep_window > 0 && seq.size() > (size_t) c.rep_window
                                                      ? seq.size() - (size_t) c.rep_window : (size_t) 0);
        std::vector<llama_token> seen(seq.begin() + (long) begin, seq.end());
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        const llama_token silence = c.audio_base + c.silence;
        for (llama_token t : seen) {
            if (t == c.speech_end || t == c.stop || t == silence) continue;
            float & l = logits[t];
            l = l < 0.0f ? l * c.rep_penalty : l / c.rep_penalty;
        }
    }
}

bool run_kitten_tts2(audio_lm_context * ctx, llama_context * lctx, const llama_vocab * vocab,
                     const tts_runner_params & a, tts_runner_result * out) {
    kitten_config c;
    std::string err;
    if (!kitten_read_config(ctx, a.preset, &c, &err)) { out->error = err; return false; }
    if (a.has_temp)  c.temp  = a.temp;
    if (a.has_top_p) c.top_p = a.top_p;
    if (a.has_top_k) c.top_k = a.top_k;
    if (a.has_min_p) c.min_p = a.min_p;
    if (a.has_rep_penalty) c.rep_penalty = a.repetition_penalty;

    const codec_speaker * spk = audio_lm_decode_speaker(ctx);
    const float * lm_cond = nullptr;
    int64_t ne[4] = {0};
    codec_speaker_slot_type ty;
    const void * data = nullptr;
    const int32_t n_embd = llama_model_n_embd(llama_get_model(lctx));
    if (spk == nullptr || !codec_speaker_get_slot(spk, "lm_cond", &ty, ne, &data) ||
        ty != CODEC_SPEAKER_SLOT_F32 || ne[0] != n_embd) {
        out->error = "KittenTTS speaker has no lm_cond row matching the backbone width";
        return false;
    }
    lm_cond = static_cast<const float *>(data);

    // In-context reference: [START] <ref_text_start> transcript <ref_text_end>
    // <ref_speech_start> audio <ref_speech_end>.  Needs the transcript; without
    // it only the speaker embedding identifies the voice (prompt.py).
    std::vector<llama_token> prefix = {c.start};
    const char * ref_text = codec_speaker_ref_text(spk);
    if (c.use_reference_prompt && ref_text != nullptr &&
        codec_speaker_get_slot(spk, "ref_tokens", &ty, ne, &data) && ty == CODEC_SPEAKER_SLOT_I32) {
        const auto * rt = static_cast<const int32_t *>(data);
        const auto text_ids = kitten_tokenize(vocab, strip_copy(ref_text));
        prefix.push_back(c.ref_text_start);
        prefix.insert(prefix.end(), text_ids.begin(), text_ids.end());
        prefix.push_back(c.ref_text_end);
        prefix.push_back(c.ref_speech_start);
        for (int64_t i = 0; i < ne[0]; ++i) prefix.push_back(c.audio_base + rt[i]);
        prefix.push_back(c.ref_speech_end);
    } else {
        std::printf("kitten: no reference transcript/tokens — speaker embedding only\n");
    }

    const std::string spoken = kitten_prepare_text(a.text);
    std::vector<llama_token> emotion;
    if (kitten_has_expression_tag(a.text)) emotion = kitten_tokenize(vocab, c.emotion_control);
    const auto chunks = kitten_split_for_synthesis(spoken, c.chunk_chars, c.chunk_min_chars);
    if (chunks.empty()) { out->error = "no text to speak"; return false; }
    const int32_t max_new = a.max_frames > 0 ? a.max_frames : c.max_new_tokens;
    const uint32_t n_ctx = llama_n_ctx(lctx);

    std::printf("kitten: voice=%s, %zu chunk(s), temp=%.2f top_k=%d top_p=%.2f min_p=%.2f rep=%.2f/%d run=%.2f/%d%s\n",
                codec_speaker_name(spk), chunks.size(), c.temp, c.top_k, c.top_p, c.min_p,
                c.rep_penalty, c.rep_window, c.run_penalty, c.run_grace, emotion.empty() ? "" : " +emo");

    std::vector<std::vector<float>> wavs;
    int32_t total_frames = 0;
    const char * stop_reason = "max_frames";
    int32_t sample_rate = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        std::vector<llama_token> prompt = prefix;
        prompt.insert(prompt.end(), emotion.begin(), emotion.end());
        prompt.push_back(c.text_start);
        const auto text_ids = kitten_tokenize(vocab, chunks[ci]);
        prompt.insert(prompt.end(), text_ids.begin(), text_ids.end());
        if (c.final_seg) prompt.push_back(c.final_seg);
        prompt.push_back(c.speech_start);

        // Per-chunk budget sized from the chunk's length (model.py::_chunk_budget).
        int32_t budget = max_new;
        if (chunks.size() > 1) {
            const int32_t expected = (int32_t) ((double) chunks[ci].size() / 20.0 * 25.0 * 1.8);
            budget = std::max(200, std::min(max_new, expected));
        }
        if (prompt.size() + (size_t) budget > n_ctx) {
            budget = (int32_t) n_ctx - (int32_t) prompt.size();
            if (budget <= 0) { out->error = "KittenTTS prompt exceeds the context"; return false; }
        }

        // Prefill: position 0 ([START]) carries the speaker row instead of its token embedding.
        llama_memory_clear(llama_get_memory(lctx), true);
        if (!decode_embed(lctx, lm_cond, n_embd, 0) ||
            !decode_tokens(lctx, std::vector<llama_token>(prompt.begin() + 1, prompt.end()), 1, false)) {
            out->error = "KittenTTS prefill decode failed";
            return false;
        }

        SamplerChain chain;
        if (c.temp <= 0.0f) chain.init_greedy();
        else chain.init_sampled_hf_order(a.seed + (uint32_t) ci, c.temp, c.top_k, c.top_p, c.min_p);

        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> seq = prompt;
        std::vector<int32_t> codes;
        std::vector<float> logits((size_t) n_vocab);
        const char * chunk_stop = "max_frames";
        for (int32_t step = 0; step < budget; ++step) {
            std::memcpy(logits.data(), llama_get_logits_ith(lctx, -1), (size_t) n_vocab * sizeof(float));
            kitten_apply_penalties(c, seq, prompt.size(), logits.data());
            // Speech is audio tokens until <speech_end> / [STOP]: mask the rest of
            // the Qwen3 vocabulary so a quantized backbone cannot drift into text.
            for (int32_t i = 0; i < n_vocab; ++i) {
                const bool audio = i >= c.audio_base && i < c.audio_base + c.n_audio;
                if (!audio && i != c.speech_end && i != c.stop) logits[(size_t) i] = -INFINITY;
            }
            const llama_token t = (llama_token) chain.sample(logits.data(), n_vocab);
            seq.push_back(t);
            if (t == c.speech_end || t == c.stop) { chunk_stop = "eos"; break; }
            codes.push_back(t - c.audio_base);
            if (!decode_tokens(lctx, {t}, (int32_t) seq.size() - 1, false)) {
                out->error = "KittenTTS decode step failed";
                return false;
            }
        }
        std::printf("kitten chunk %zu/%zu: %zu prompt tokens, %zu audio tokens, stop=%s\n",
                    ci + 1, chunks.size(), prompt.size(), codes.size(), chunk_stop);
        total_frames += (int32_t) codes.size();
        stop_reason = chunk_stop;
        if (codes.empty()) continue;

        // S3Gen drops its last tokens as lookahead: pad with silence (vocoder.py).
        codes.insert(codes.end(), (size_t) c.n_trailing_silence, c.silence);
        audio_lm_reset(ctx);
        audio_lm_audio_output pcm;
        if (!audio_lm_push_codes(ctx, codes.data(), (int32_t) codes.size(), 1) ||
            !timed_decode_audio(ctx, &pcm, out)) {
            out->error = std::string("KittenTTS decode failed: ") + audio_lm_last_error(ctx);
            return false;
        }
        sample_rate = pcm.sample_rate;
        wavs.push_back(std::move(pcm.pcm));
    }
    if (wavs.empty()) { out->error = "the model produced no audio tokens"; return false; }
    std::printf("KittenTTS AR done: %d frames, stop=%s\n", total_frames, stop_reason);
    out->pcm = kitten_join_chunks(wavs, sample_rate, c.chunk_gap_s);
    out->sample_rate = sample_rate;
    out->n_channels = 1;
    out->n_frames = total_frames;
    out->stop_reason = stop_reason;
    return true;
}

// ── Soprano (codec-only Vocos decoder + Qwen3 backbone) ─────────────────────
// Reference: soprano/tts.py + backends/transformers.py.  The prompt is
// `[STOP][TEXT]{text}[START]`; the backbone generates until [STOP] (its eos)
// and the decoder input is the final-norm hidden state that PREDICTS each
// non-eos token (the first comes from the prompt's last position).  Sampling
// defaults: temperature 0 (greedy), top-p 0.95, repetition penalty 1.2 over
// prompt + generated ids, at most 512 new tokens.  The decoder output is
// (n-1)*2048 samples, which is what the reference keeps.
bool detect_soprano(audio_lm_context * ctx) {
    const char * arch = audio_lm_meta_str(ctx, "general.architecture");
    return arch != nullptr && std::strcmp(arch, "soprano") == 0;
}

bool run_soprano(audio_lm_context * ctx, llama_context * lctx, const llama_vocab * vocab,
                 const tts_runner_params & a, tts_runner_result * out) {
    const llama_model * lmodel = llama_get_model(lctx);
    const int32_t n_embd = llama_model_n_embd(lmodel);
    const llama_token eos = llama_vocab_eos(vocab);
    const std::vector<llama_token> prompt =
        tokenize_str(vocab, "[STOP][TEXT]" + a.text + "[START]", /*add_bos=*/false, /*parse_special=*/true);
    if (prompt.size() < 3) { out->error = "Soprano: empty prompt"; return false; }

    // The speech tokens are the contiguous pieces "[0]".."[N-1]"; the rest of
    // the vocab is the text BPE.  Mask the stream to speech tokens + [STOP].
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    int32_t off = -1, count = 0;
    char piece[32];
    for (int32_t id = 0; id < n_vocab; ++id) {
        const int32_t n = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, true);
        if (n <= 0) { if (off >= 0) break; continue; }
        piece[n] = '\0';
        char want[32];
        std::snprintf(want, sizeof(want), "[%d]", count);
        if (std::strcmp(piece, want) == 0) { if (off < 0) off = id; ++count; }
        else if (off >= 0) break;
    }
    if (off < 0 || count == 0) { out->error = "Soprano: no [N] speech tokens in the backbone vocab"; return false; }

    const float temp  = a.has_temp ? a.temp : 0.0f;
    const float top_p = a.has_top_p ? a.top_p : 0.95f;
    const float rep   = a.has_rep_penalty ? a.repetition_penalty : 1.2f;
    const int32_t max_new = a.max_frames > 0 ? a.max_frames : 512;
    BackboneSampler smpl;
    std::string err;
    if (!smpl.build(lmodel, a.seed, temp, a.has_top_k ? a.top_k : 0, top_p, /*min_p=*/0.0f, rep,
                    /*rep_last_n=*/0, a.grammar,
                    a.grammar.empty() ? build_flat_audio_mask(off, count, eos, n_vocab) : std::vector<llama_logit_bias>{},
                    &err)) {
        out->error = "Soprano: " + err;
        return false;
    }
    for (llama_token t : prompt) common_sampler_accept(smpl.smpl, t, /*is_generated=*/false);
    std::printf("soprano: %zu prompt tokens, temp=%.2f top_p=%.2f rep=%.2f\n", prompt.size(), temp, top_p, rep);

    if (!decode_tokens(lctx, prompt, 0, /*all_pos=*/false)) { out->error = "Soprano: prompt decode failed"; return false; }
    int32_t n_past = (int32_t) prompt.size();
    std::vector<float> hiddens;
    const char * stop_reason = "max_frames";
    for (int32_t step = 0; step < max_new; ++step) {
        const float * h = llama_get_embeddings_ith(lctx, -1);
        if (h == nullptr) { out->error = "Soprano: backbone returned no hidden state"; return false; }
        const llama_token tok = smpl.sample(lctx, -1);
        if (tok == eos) { stop_reason = "eos"; break; }
        hiddens.insert(hiddens.end(), h, h + n_embd);
        if (!decode_tokens(lctx, {tok}, n_past++, false)) { out->error = "Soprano: decode failed"; return false; }
    }
    const int32_t n_frames = (int32_t) (hiddens.size() / (size_t) n_embd);
    std::printf("soprano AR done: %d frames, stop=%s\n", n_frames, stop_reason);
    if (n_frames < 2) { out->error = "Soprano: the model produced no speech"; return false; }

    audio_lm_audio_output pcm;
    const auto t_dec = std::chrono::steady_clock::now();
    if (!audio_lm_decode_latents(ctx, hiddens.data(), n_frames, n_embd, &pcm)) {
        out->error = audio_lm_last_error(ctx);
        return false;
    }
    out->decode_ms = ms_since(t_dec);
    fill_result_from_output(pcm, n_frames, stop_reason, out);
    return true;
}

// Selects the codec decoder speaker for every flow: --speaker-file, else a
// speaker derived from --ref-audio when the codec bundles the encoders, else
// --voice, else the model default.  Optionally saves it (--save-speaker).
static bool apply_decode_speaker(audio_lm_context * ctx, const tts_runner_params & a, std::string * err) {
    if (!a.speaker_file.empty()) {
        // --voice picks a speaker inside the file (default: the file's default).
        if (!audio_lm_load_decode_speaker(ctx, a.speaker_file.c_str(),
                                          a.voice.empty() ? nullptr : a.voice.c_str())) {
            *err = audio_lm_last_error(ctx);
            return false;
        }
        std::printf("speaker: loaded %s%s%s\n", a.speaker_file.c_str(),
                    a.voice.empty() ? "" : " : ", a.voice.c_str());
    } else if (!a.ref_audio_path.empty() && audio_lm_codec_can_speaker_from_audio(ctx)) {
        std::vector<float> pcm;
        int32_t n = 0, sr = 0;
        if (!load_ref_audio(a.ref_audio_path, pcm, &n, &sr, err)) return false;
        if (!audio_lm_decode_speaker_from_audio(ctx, pcm.data(), n, sr,
                                                a.ref_text.empty() ? nullptr : a.ref_text.c_str())) {
            *err = audio_lm_last_error(ctx);
            return false;
        }
        std::printf("speaker: derived from %s (%d samples @ %d Hz)\n", a.ref_audio_path.c_str(), n, sr);
    } else if (!a.voice.empty()) {
        if (!audio_lm_set_decode_speaker(ctx, a.voice.c_str())) {
            *err = audio_lm_last_error(ctx);
            return false;
        }
        std::printf("speaker: voice %s\n", a.voice.c_str());
    }
    if (a.speaker_file.empty() && !a.ref_audio_path.empty() &&
        audio_lm_codec_has_speakers(ctx) && !audio_lm_codec_can_speaker_from_audio(ctx)) {
        // No speaker encoders: GGUFs converted before codec_speaker, or
        // codecs whose speakers cannot come from audio (OuteTTS).
        std::fprintf(stderr, "speaker: this codec GGUF cannot derive a speaker from --ref-audio; keeping the %s "
                             "voice (Chatterbox / CosyVoice GGUFs converted before codec_speaker: re-convert)\n",
                     a.voice.empty() ? "default" : a.voice.c_str());
    }
    if (!a.save_speaker.empty()) {
        if (!audio_lm_save_decode_speaker(ctx, a.save_speaker.c_str(), "saved")) {
            *err = audio_lm_last_error(ctx);
            return false;
        }
        std::printf("speaker: saved to %s\n", a.save_speaker.c_str());
    }
    return true;
}

static bool tts_runner_synthesize_impl(const tts_runner_params & a, tts_runner_result * out,
                                       std::chrono::steady_clock::time_point t0);

bool tts_runner_synthesize(const tts_runner_params & a, tts_runner_result * out) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = tts_runner_synthesize_impl(a, out, t0);
    out->total_ms = ms_since(t0);
    return ok;
}

static bool tts_runner_synthesize_impl(const tts_runner_params & a, tts_runner_result * out,
                                       std::chrono::steady_clock::time_point t0) {
    // Pocket-TTS FlowLM is self-contained (no backbone) — try it first; a
    // non-zero return means it handled the request.
    {
        tts_runner_result fr;
        int handled = tts_runner_synthesize_selfcontained(a, &fr);
        if (handled) {
            if (!fr.error.empty()) { out->error = fr.error; return false; }
            fr.load_ms = 0.0;  // not split for the self-contained path
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
    p.kv_overrides = a.kv_overrides;
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
    {
        std::string spk_err;
        const auto t_spk = std::chrono::steady_clock::now();
        const bool spk_ok = apply_decode_speaker(ctx, a, &spk_err);
        out->speaker_ms = ms_since(t_spk);
        if (!spk_ok) {
            out->error = spk_err;
            audio_lm_free(ctx);
            return false;
        }
    }

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
    // Without --gpu the backbone stays on the CPU: an empty device list, else llama.cpp
    // would still offload batched ops to any GPU it finds.
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mp.n_gpu_layers = a.use_gpu ? 999 : 0;
    if (!a.use_gpu) mp.devices = no_devices;
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
    const float cbx_cfg = !is_chatterbox || cbx->is_turbo ? 0.0f
        : a.has_cfg_weight ? a.cfg_weight : (cbx->cfg_weight >= 0.0f ? cbx->cfg_weight : 0.5f);
    const int32_t cbx_n_seq = (is_chatterbox && cbx_cfg > 0.0f) ? 2 : 1;

    // OuteTTS is ALSO a codec-only backbone (get_prompt_info fails, pi_ok=false)
    // that drives a 24kHz 2-codebook DAC purely via token ids.  Detect it here
    // (V3 only for this path) so the deferred-pi abort below skips it too.
    // (Chatterbox T3's backbone has no vocab at all.)
    codec_common::outetts_version ov{};
    const llama_vocab * bb_vocab = llama_model_get_vocab(lmodel);
    const bool is_outetts =
        llama_vocab_type(bb_vocab) != LLAMA_VOCAB_TYPE_NONE &&
        detect_outetts_version(lmodel, bb_vocab, &ov);
    const bool outetts_legacy = is_outetts && ov != codec_common::outetts_version::V3;

    const int32_t max_frames = a.max_frames > 0 ? a.max_frames : 512;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) std::max(4096, max_frames + 512);
    cp.n_batch      = cp.n_ctx;
    cp.n_ubatch     = cp.n_ctx;
    if (is_outetts) {
        // An OuteTTS 1.0 frame is a c1/c2 pair plus its share of the word-block
        // tokens (~2.4 tokens), after a ~1.8k-token speaker prompt; 0.2/0.3
        // spend ~1.1 tokens per frame.  Without --max-frames the budget is the
        // reference's max_length (8192 tokens for 1.0, 4096 for 0.2/0.3).
        const int32_t tok_per_frame = outetts_legacy ? 2 : 3;
        const uint32_t max_len = outetts_legacy ? 4096u : 8192u;
        cp.n_ctx    = a.max_frames > 0 ? (uint32_t) std::max(4096, 2048 + tok_per_frame * a.max_frames) : max_len;
        cp.n_batch  = cp.n_ctx;
        cp.n_ubatch = 4096;
    }
    cp.n_seq_max    = (uint32_t) cbx_n_seq;
    cp.embeddings   = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    // CosyVoice v1's relative-position term is a KQ bias, which the
    // flash-attention path does not take.
    if (pi_ok && pi.host_arch == "cosyvoice") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (a.n_threads > 0) { cp.n_threads = a.n_threads; cp.n_threads_batch = a.n_threads; }
    cp.op_offload = a.use_gpu;
    llama_context * lctx = llama_init_from_model(lmodel, cp);
    out->load_ms = ms_since(t0);
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
        warn_grammar_unused(a.grammar, "Chatterbox");
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
        if (!timed_decode_audio(ctx, &pcm, out)) {
            out->error = std::string("decode_audio failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        fill_result_from_output(pcm, n_frames, stop_reason, out);
        audio_lm_free(ctx);
        return true;
    }

    // ── CosyVoice ────────────────────────────────────────────────────────
    if (const codec_lm_cosyvoice_info * cvi = lm_handle ? codec_lm_cosyvoice_get_info(lm_handle) : nullptr) {
        warn_grammar_unused(a.grammar, "CosyVoice");
        std::vector<int32_t> speech_codes;
        int32_t n_frames = 0;
        const char * stop_reason = "max_len";
        bool ok = run_cosyvoice(ctx, lctx, lm_handle, cvi, a, hidden, a.max_frames, &speech_codes,
                                &n_frames, &stop_reason);
        llama_free(lctx);
        llama_model_free(lmodel);
        llama_backend_free();
        if (!ok || speech_codes.empty()) {
            out->error = ok ? "no speech codes generated" : std::string("cosyvoice AR failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        std::printf("cosyvoice AR done: %d tokens (%zu to the flow), stop=%s\n",
                    n_frames, speech_codes.size(), stop_reason);
        audio_lm_reset(ctx);
        audio_lm_audio_output pcm;
        if (!audio_lm_push_codes(ctx, speech_codes.data(), (int32_t) speech_codes.size(), 1) ||
            !timed_decode_audio(ctx, &pcm, out)) {
            out->error = std::string("decode failed: ") + audio_lm_last_error(ctx);
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

    // If get_prompt_info was deferred (codec-only gguf) but this is NEITHER a
    // NeuTTS nor an OuteTTS backbone, the model genuinely lacks a usable
    // codec_lm — abort now with the original deferred error rather than fall
    // through to the codec_lm-backed flows (which would dereference an
    // unpopulated `pi`).
    // Soprano: codec-only Vocos decoder fed the backbone's hidden states.
    if (!pi_ok && detect_soprano(ctx)) {
        const bool ok = run_soprano(ctx, lctx, vocab, a, out);
        llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
        audio_lm_free(ctx);
        return ok;
    }

    // KittenTTS 2: codec-only S3G GGUF + Qwen3 speech-LM backbone.
    if (!pi_ok && detect_kitten_tts2(vocab)) {
        warn_grammar_unused(a.grammar, "KittenTTS");
        const bool ok = run_kitten_tts2(ctx, lctx, vocab, a, out);
        llama_free(lctx); llama_model_free(lmodel); llama_backend_free();
        audio_lm_free(ctx);
        return ok;
    }

    if (!pi_ok && !is_neutts && !is_outetts) {
        out->error = deferred_pi_err;
        llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
        return false;
    }

    // ── OuteTTS free-gen Type A decode (1.0: DAC n_q=2, 0.2/0.3: WavTokenizer) ─
    // OuteTTS (like NeuTTS) is codec-only: the backbone lm_head emits an
    // INTERLEAVED stream of structural tokens (1.0: <|word_start|>, word text,
    // <|features|>, <|t_..|>, <|energy_N|>, <|code|>, <|c1_N|>, <|c2_N|>,
    // <|word_end|>; 0.2/0.3: word, <|t_..|>, <|N|> codes, separators) that
    // ends with <|audio_end|>.  We drive it with FREE generation (no flat audio
    // mask — that would forbid the structural tokens), let the observe
    // accumulator strip the non-audio tokens and pack the codes into frames.
    // The prompt-derived word-block grammar (as llama.rn's rn-tts) keeps the
    // stream well formed: only the input's words, complete headers, in-range
    // codes.  The voice is the codec speaker (ref_text + audio_prompt).
    // Self-contained: frees all resources on every early-return and returns.
    if (is_outetts) {
        using codec_common::outetts_version;
        const char * ver_name = ov == outetts_version::V1 ? "0.2" : ov == outetts_version::V2 ? "0.3" : "1.0";
        const std::string in_words = outetts_input_words(ov, a.text);
        std::string perr;
        std::vector<llama_token> outetts_prompt_toks =
            build_outetts_prompt(vocab, ov, in_words, audio_lm_decode_speaker(ctx), &perr);
        if (outetts_prompt_toks.empty()) {
            out->error = perr;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // Audio-token ranges BEFORE any observe: 1.0 = c1/c2 offsets, <|code|>
        // frame sentinel; 0.2/0.3 = <|0|>.. (WavTokenizer codebook 4096).
        int32_t off2[2] = {-1, -1}, cnt2[2] = {0, 0}, sentinel = -1, eos_id = -1;
        int32_t n_q_oute = 2;
        if (ov == outetts_version::V3) {
            if (!outetts_audio_token_ranges(vocab, off2, cnt2, &sentinel, &eos_id)) {
                out->error =
                    "OuteTTS: outetts_audio_token_ranges failed "
                    "(<|c1_0|>/<|c2_0|>/<|code|>/<|audio_end|> missing from vocab)";
                llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
                return false;
            }
        } else {
            n_q_oute = 1;
            off2[0] = vocab_piece_id(vocab, "<|0|>");
            cnt2[0] = 4096;
            eos_id  = vocab_piece_id(vocab, "<|audio_end|>");
            if (off2[0] < 0 || eos_id < 0 || vocab_piece_id(vocab, "<|4095|>") != off2[0] + 4095) {
                out->error = "OuteTTS: <|0|>..<|4095|> / <|audio_end|> not found as a contiguous vocab range";
                llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
                return false;
            }
        }
        audio_lm_set_audio_token_ranges(ctx, off2, cnt2, n_q_oute, sentinel, eos_id);
        std::printf("[outetts] %s: codes c1=[%d,%d)", ver_name, off2[0], off2[0] + cnt2[0]);
        if (n_q_oute == 2) std::printf(" c2=[%d,%d)", off2[1], off2[1] + cnt2[1]);
        std::printf(" sentinel=%d eos=%d; %zu prompt tokens\n", sentinel, eos_id, outetts_prompt_toks.size());

        pi.model_kind = audio_lm_prompt_info::KIND_TOKEN_SINGLE_CB;

        // Validate prefill fits the backbone context.
        const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
        if ((int32_t) outetts_prompt_toks.size() + 1 >= n_ctx) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "OuteTTS: prefill (%zu tokens) does not fit backbone n_ctx=%d; "
                "raise --max-frames.",
                outetts_prompt_toks.size(), n_ctx);
            out->error = buf;
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        // Sampling defaults: 1.0 (README) temp 1.0, top-k 50, 64-token window
        // with penalty 1.1; 0.2/0.3 (GenerationConfig, HF generate) temp 0.1,
        // top-k 50, penalty 1.1 over the whole sequence.
        const bool     legacy     = ov != outetts_version::V3;
        const float    ot_temp    = a.has_temp  ? a.temp  : (legacy ? 0.1f : 1.0f);
        const float    ot_top_p   = a.has_top_p ? a.top_p : 1.0f;
        const int32_t  ot_top_k   = a.has_top_k ? a.top_k : 50;
        const uint32_t ot_seed    = a.seed ? a.seed : 0xC0DEC1ABu;
        const float    ot_rep_pen = a.has_rep_penalty ? a.repetition_penalty : 1.1f;
        const int32_t  ot_rep_n   = legacy ? 0 : 64;   // 0 = full history
        const std::string ot_grammar = !a.grammar.empty()
            ? a.grammar
            : codec_common::outetts_build_grammar(ov, in_words);
        std::printf("[outetts] grammar: %s (%zu bytes)\n",
                    !a.grammar.empty() ? "user-supplied" : "auto (prompt-derived)", ot_grammar.size());

        // 5) Free-gen multi-cb decode.  constrain_audio_mask=false ⇒ empty mask
        //    (structural tokens pass), no min-length guard, and NO singular
        //    range re-set (the multi-cb range above is preserved).  off/count/
        //    eos_id are unused in free-gen mode but passed for signature parity.
        int32_t n_frames = 0;
        const char * stop_reason = "max_frames";
        bool ok = run_token_single_cb(ctx, lctx, lmodel, vocab, pi, outetts_prompt_toks,
                                      off2[0], cnt2[0], eos_id,
                                      a.max_frames > 0 ? a.max_frames : INT32_MAX,
                                      /*min_frames=*/0, ot_seed,
                                      ot_temp, ot_top_p, ot_top_k,
                                      /*constrain_audio_mask=*/false,
                                      ot_rep_pen, ot_rep_n, ot_grammar,
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
        // observe_token accumulated the codes into ctx->codes already —
        // decode directly (DAC n_q=2 / WavTokenizer n_q=1).
        audio_lm_audio_output pcm;
        if (!timed_decode_audio(ctx, &pcm, out)) {
            out->error = std::string("OuteTTS: decode_audio failed: ") + audio_lm_last_error(ctx);
            audio_lm_free(ctx);
            return false;
        }
        fill_result_from_output(pcm, n_frames, stop_reason, out);
        audio_lm_free(ctx);
        return true;
    }

    if (is_neutts) {
        // The reference voice is the codec speaker apply_decode_speaker chose:
        // --speaker-file, --ref-audio (codecs with the distill encoder derive
        // `ref_codes` in-library), --voice, or the codec GGUF's default.
        // `ref_text` holds the reference transcript as phonemes; --ref-text
        // overrides it.
        if (!a.ref_audio_path.empty() && a.speaker_file.empty() && !audio_lm_codec_can_speaker_from_audio(ctx)) {
            out->error = "NeuTTS: --ref-audio needs a codec GGUF with the NeuCodec distill encoder "
                         "(this one cannot encode); use --voice / --speaker-file or a distill_neucodec codec";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        const codec_speaker * spk = audio_lm_decode_speaker(ctx);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        if (spk == nullptr || !codec_speaker_get_slot(spk, "ref_codes", &ty, ne, &data) ||
            ty != CODEC_SPEAKER_SLOT_I32) {
            out->error = "NeuTTS needs a reference voice: a codec GGUF with builtin voices (--voice), "
                         "--speaker-file, or --ref-audio + --ref-text with a distill_neucodec codec";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }
        const std::vector<int32_t> ref_codes(static_cast<const int32_t *>(data),
                                             static_cast<const int32_t *>(data) + ne[0]);
        const char * spk_text = codec_speaker_ref_text(spk);
        const std::string ref_text = !a.ref_text.empty() ? a.ref_text : (spk_text != nullptr ? spk_text : "");
        if (ref_text.empty()) {
            out->error = "NeuTTS: the reference voice has no transcript; pass --ref-text (phonemes)";
            llama_free(lctx); llama_model_free(lmodel); audio_lm_free(ctx); llama_backend_free();
            return false;
        }

        neutts_prompt_toks = build_neutts_prompt(vocab, ref_text, a.text, ref_codes);
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
                                      /*rep_penalty=*/1.0f, /*rep_last_n=*/0, a.grammar,
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
        if (!timed_decode_audio(ctx, &pcm, out)) {
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
        resolve_constraint(pi, a.grammar, a.text, n_vocab_bb);
    if (!constraint.grammar.empty()) {
        std::printf("grammar: %s (%zu bytes)\n",
                    !a.grammar.empty() ? "user-supplied" : "auto (model-derived)",
                    constraint.grammar.size());
    } else if (!constraint.logit_bias.empty()) {
        std::printf("logit-bias: flat audio mask (%zu tokens masked)\n",
                    constraint.logit_bias.size());
    }

    // Only the cb0-from-backbone codebook AR (MOSS-TTSD) samples the backbone
    // vocabulary; every other flow here samples a codec_lm head.
    if (pi.is_continuous || pi.sequential_text_audio || pi.streaming_interleave ||
        !pi.cb0_from_backbone) {
        warn_grammar_unused(a.grammar, "this model (it samples a codec_lm head)");
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
    if (!timed_decode_audio(ctx, &pcm, out)) {
        out->error = std::string("decode_audio failed: ") + audio_lm_last_error(ctx);
        audio_lm_free(ctx);
        return false;
    }
    fill_result_from_output(pcm, n_frames, stop_reason, out);
    audio_lm_free(ctx);
    return true;
}

}  // namespace codec_common
