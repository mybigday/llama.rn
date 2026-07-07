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

#include "tts_runner.h"

#include "codec_common.h"
#include "utils/wav_io.h"

#include "llama.h"
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
#include <memory>
#include <string>
#include <vector>

namespace codec_common {

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
    lm_gguf_context * gg   = nullptr;
    lm_ggml_context * meta = nullptr;   // holds tensor metadata (no_alloc)
    const uint8_t * base = nullptr;  // mmapped tensor-data region
    std::vector<uint8_t> blob;       // owns the file bytes
    int64_t hidden = 0;
    int64_t vocab  = 0;
    lm_ggml_type type = LM_GGML_TYPE_F32;
    size_t row_bytes = 0;
    lm_ggml_to_float_t to_float = nullptr;

    bool load(const char * path, int32_t want_hidden, std::string & err) {
        lm_gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &meta };
        gg = lm_gguf_init_from_file(path, gp);
        if (!gg) { err = "lm_gguf_init_from_file failed"; return false; }
        const int64_t tid = lm_gguf_find_tensor(gg, "token_embd.weight");
        if (tid < 0) { err = "token_embd.weight not found in backbone"; return false; }
        lm_ggml_tensor * t = lm_ggml_get_tensor(meta, "token_embd.weight");
        if (!t) { err = "token_embd metadata lookup failed"; return false; }
        hidden = t->ne[0];
        vocab  = t->ne[1];
        type   = t->type;
        if ((int32_t) hidden != want_hidden) {
            err = "token_embd hidden mismatch"; return false;
        }
        const lm_ggml_type_traits * tr = lm_ggml_get_type_traits(type);
        to_float = tr ? tr->to_float : nullptr;
        // For a quantised type to_float works on a whole row (k = hidden,
        // which must be a multiple of block size for legal types).
        if (!to_float) { err = "no to_float for token_embd type"; return false; }
        row_bytes = lm_ggml_row_size(type, hidden);

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
        const size_t data_off = lm_gguf_get_data_offset(gg);
        const size_t t_off     = lm_gguf_get_tensor_offset(gg, tid);
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
        if (gg)   lm_gguf_free(gg);
        if (meta) lm_ggml_free(meta);
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
                     const std::string & grammar,
                     const std::vector<float> & speaker_prefix,
                     const std::string & payload_text,
                     int32_t * out_frames, const char ** out_stop) {
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
                          /*rep_penalty=*/1.0f, /*rep_last_n=*/0, grammar, &berr)) {
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

}  // namespace

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

    audio_lm_prompt_info pi;
    if (!audio_lm_get_prompt_info(ctx, &pi)) {
        out->error = std::string("get_prompt_info failed: ") + audio_lm_last_error(ctx);
        audio_lm_free(ctx);
        return false;
    }
    const int32_t hidden = audio_lm_hidden_dim(ctx);
    const int32_t n_cb   = audio_lm_n_codebook(ctx);
    std::printf("model: arch=%s kind=%d n_cb=%d hidden=%d cb0_backbone=%d audio_offset=%d eos_c0=%d\n",
                pi.host_arch.c_str(), (int) pi.model_kind, n_cb, hidden,
                (int) pi.cb0_from_backbone, pi.audio_codebook_offset, pi.eos_code_c0);

    // ── Moshi: formally out of scope for one-shot synthesize ────────────
    if (pi.host_arch == "llama" &&
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
    if (n_embd != hidden) {
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

    // Grammar for the backbone sampler: an explicit user GBNF wins; else the
    // model's metadata-derived auto-grammar (empty for models with none).
    std::string grammar = !a.grammar.empty() ? a.grammar
                                             : tts_auto_grammar(pi, a.text);
    if (!grammar.empty()) {
        std::printf("grammar: %s (%zu bytes)\n",
                    !a.grammar.empty() ? "user-supplied" : "auto (model-derived)",
                    grammar.size());
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
                                max_frames, seed, temp, top_p, top_k, grammar,
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
