// tts_runner.cpp — the reference TTS host loop over the public
// codec_audio_lm API (include/codec_audio_lm.h), with llama.cpp as the
// backbone: llama.cpp tokenizes, decodes and samples; codec_audio_lm builds
// the prompt, says how to sample and what to feed each step, and decodes the
// audio.  It is the same loop a host such as llama.rn writes against the API.
// Built only when CODEC_TTS_BACKBONE=ON (it links libttsbackbone);
// tts_runner_base.cpp holds the backbone-free parts.

#include "llama.h"

#include "tts_runner.h"

#include "codec_audio_lm.h"
#include "codec_common.h"
#include "common.h"     // common_params_sampling, common_grammar
#include "sampling.h"   // common_sampler_*

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace codec_common {

namespace {

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

// Sampler settings beyond the classic knobs (codec_audio_lm_sampler).
struct BackboneSamplerExtra {
    float   penalty_freq    = 0.0f;
    float   penalty_present = 0.0f;
    std::vector<llama_token> penalty_exempt;
    float   dry_multiplier     = 0.0f;
    float   dry_base           = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = -1;
    std::string samplers;   // llama.cpp --samplers order; "" = the default below
};

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
               std::string * err, const BackboneSamplerExtra * extra = nullptr) {
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
        if (extra != nullptr) {
            sp.penalty_freq    = extra->penalty_freq;
            sp.penalty_present = extra->penalty_present;
            sp.penalty_exempt  = extra->penalty_exempt;
            if (sp.penalty_last_n == 0 && (sp.penalty_freq != 0.0f || sp.penalty_present != 0.0f)) {
                sp.penalty_last_n = rep_last_n > 0 ? rep_last_n : kRepPenaltyFullHistory;
            }
            sp.dry_multiplier     = extra->dry_multiplier;
            sp.dry_base           = extra->dry_base;
            sp.dry_allowed_length = extra->dry_allowed_length;
            sp.dry_penalty_last_n = extra->dry_penalty_last_n;
            sp.dry_sequence_breakers.clear();
            if (!extra->samplers.empty()) {
                std::vector<std::string> names;
                std::stringstream ss(extra->samplers);
                for (std::string n; std::getline(ss, n, ';');) if (!n.empty()) names.push_back(n);
                sp.samplers = common_sampler_types_from_names(names);
            } else if (sp.dry_multiplier > 0.0f) {
                sp.samplers.insert(sp.samplers.begin() + 1, COMMON_SAMPLER_TYPE_DRY);
            }
        }
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

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<llama_token> tokenize_str(const llama_vocab * vocab, const std::string & s, bool add_special,
                                      bool parse_special) {
    std::vector<llama_token> toks(s.size() + 8);
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), toks.data(), (int32_t) toks.size(), add_special,
                               parse_special);
    if (n < 0) {
        toks.resize((size_t) -n);
        n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), toks.data(), (int32_t) toks.size(), add_special,
                           parse_special);
    }
    toks.resize((size_t) std::max(0, n));
    return toks;
}

// Tokens at n_past.. on sequence 0; outputs on every position when all_pos,
// else the last.
bool decode_tokens(llama_context * lctx, const std::vector<llama_token> & toks, int32_t n_past, bool all_pos) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = n_past + (int32_t) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = (all_pos || i == toks.size() - 1) ? 1 : 0;
    }
    b.n_tokens = (int32_t) toks.size();
    const int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// `n` embedding rows per sequence, sequences side by side (row r of sequence q
// at rows[(q * n + r) * dim]), positions n_past..; outputs on each
// sequence's last row.
bool decode_embd(llama_context * lctx, const float * rows, int32_t dim, int32_t n, int32_t n_seq, int32_t n_past) {
    if (n <= 0) return true;
    const int32_t total = n * n_seq;
    llama_batch b = llama_batch_init(total, dim, 1);
    std::memcpy(b.embd, rows, (size_t) total * dim * sizeof(float));
    b.token = nullptr;
    b.n_tokens = total;
    for (int32_t q = 0, i = 0; q < n_seq; ++q) {
        for (int32_t r = 0; r < n; ++r, ++i) {
            b.pos[i] = n_past + r;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = q;
            b.logits[i] = r == n - 1 ? 1 : 0;
        }
    }
    const int rc = llama_decode(lctx, b);
    llama_batch_free(b);
    return rc == 0;
}

// Frame streaming through codec_audio_lm_stream_poll / _finish: each final
// piece goes to on_audio as it is ready.
class stream_out {
public:
    stream_out(codec_audio_lm * alm, const tts_runner_params & a, tts_runner_result * out) : alm_(alm), a_(a), out_(out) {}

    bool poll() { return step(false); }
    bool finish() { return step(true); }
    bool used() const { return used_; }   // this segment's audio went out here

private:
    bool step(bool final) {
        used_ = true;
        codec_pcm_buffer b{};
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = final ? codec_audio_lm_stream_finish(alm_, &b) : codec_audio_lm_stream_poll(alm_, a_.stream_chunk_ms, &b);
        out_->decode_ms += ms_since(t0);
        if (!ok) {
            out_->error = std::string("streaming: ") + codec_audio_lm_last_error(alm_);
            return false;
        }
        out_->sample_rate = b.sample_rate;
        out_->n_channels  = std::max(1, b.n_channels);
        const bool go = b.n_samples == 0 || a_.on_audio(b.data, b.n_samples, b.sample_rate, out_->n_channels);
        codec_pcm_buffer_free(&b);
        if (!go) out_->error = "stopped by the caller";
        return go;
    }

    codec_audio_lm * alm_;
    const tts_runner_params & a_;
    tts_runner_result * out_;
    bool used_ = false;
};

int32_t vocab_tokenize(void * ud, const char * text, int32_t len, int32_t * out, int32_t cap, bool add_special,
                       bool parse_special) {
    return llama_tokenize((const llama_vocab *) ud, text, len, out, cap, add_special, parse_special);
}
int32_t vocab_piece(void * ud, int32_t token, char * buf, int32_t cap) {
    return llama_token_to_piece((const llama_vocab *) ud, token, buf, cap, 0, true);
}
bool vocab_is_eog(void * ud, int32_t token) { return llama_vocab_is_eog((const llama_vocab *) ud, token); }

codec_audio_lm_vocab host_vocab(const llama_vocab * v) {
    codec_audio_lm_vocab cv{};
    cv.user_data      = (void *) v;
    cv.tokenize       = vocab_tokenize;
    cv.token_to_piece = vocab_piece;
    cv.n_tokens       = llama_vocab_n_tokens(v);
    cv.eos            = llama_vocab_eos(v);
    cv.is_eog         = vocab_is_eog;
    return cv;
}

}  // namespace

// Session: the codec_audio_lm context and (unless the model is
// self-contained) the backbone, loaded once for any number of runs.
struct tts_runner_session {
    codec_audio_lm * alm    = nullptr;
    bool             llama_inited = false;
    llama_model    * lmodel = nullptr;
    llama_context  * lctx   = nullptr;
    int32_t          max_frames = 0;   // load-time cap the context is sized for

    ~tts_runner_session() {
        if (lctx)   llama_free(lctx);
        if (lmodel) llama_model_free(lmodel);
        if (llama_inited) llama_backend_free();
        if (alm)    codec_audio_lm_free(alm);
    }
};

// One generation (one segment): prompt, the backbone loop, audio.
static bool run_one(tts_runner_session * s, const tts_runner_params & a, tts_runner_result * out, stream_out * so) {
    codec_audio_lm * alm  = s->alm;
    if (s->lctx == nullptr) return tts_runner_generate_self_contained(alm, a, out);
    llama_context  * lctx = s->lctx;
    const llama_vocab * vocab = llama_model_get_vocab(s->lmodel);
    const codec_audio_lm_vocab cv = host_vocab(vocab);
    auto fail = [&](const std::string & what) {
        out->error = what + ": " + codec_audio_lm_last_error(alm);
        return false;
    };
    const codec_audio_lm_tts_request req = tts_runner_request(a);
    if (!codec_audio_lm_tts_prompt(alm, &req, &cv)) return fail("prompt");
    codec_audio_lm_completion c;
    if (!codec_audio_lm_get_completion(alm, &c)) return fail("completion");
    if (c.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) return tts_runner_generate_self_contained(alm, a, out);
    const int32_t n_seq = std::max(1, c.n_seq);
    if (n_seq > 1 && c.sample_first) {
        out->error = "backbone sampling over several sequences is not supported";
        return false;
    }

    // Backbone sampler: the completion's settings; a user grammar replaces the model's.
    const codec_audio_lm_sampler & sp = c.sampler;
    const std::string grammar = !a.grammar.empty() ? a.grammar : (c.grammar ? c.grammar : "");
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    // Logit bias forbidding every token outside `allowed` (minus `except`).
    auto allowed_bias = [&](int32_t except) {
        std::vector<llama_logit_bias> bias;
        if (c.n_allowed == 0) return bias;
        std::vector<char> ok((size_t) n_vocab, 0);
        for (int32_t i = 0; i < c.n_allowed; ++i) {
            for (int32_t t = std::max(0, c.allowed[i].begin); t < std::min(n_vocab, c.allowed[i].end); ++t) ok[(size_t) t] = 1;
        }
        if (except >= 0 && except < n_vocab) ok[(size_t) except] = 0;
        for (int32_t t = 0; t < n_vocab; ++t) if (!ok[(size_t) t]) bias.push_back(llama_logit_bias{ t, -INFINITY });
        return bias;
    };
    BackboneSampler smpl;
    std::string serr;
    BackboneSamplerExtra extra;
    extra.penalty_freq    = sp.penalty_freq;
    extra.penalty_present = sp.penalty_present;
    extra.penalty_exempt.assign(sp.penalty_exempt, sp.penalty_exempt + sp.n_penalty_exempt);
    extra.dry_multiplier     = sp.dry_multiplier;
    extra.dry_base           = sp.dry_base;
    extra.dry_allowed_length = sp.dry_allowed_length;
    extra.dry_penalty_last_n = sp.dry_penalty_last_n;
    extra.samplers = sp.samplers ? sp.samplers : "";
    if (!smpl.build(s->lmodel, sp.seed, sp.temp, sp.top_k, sp.top_p, sp.min_p, sp.penalty_repeat, sp.penalty_last_n,
                    grammar, allowed_bias(-1), &serr, &extra)) {
        out->error = serr;
        return false;
    }
    // Minimum length: a second sampler without the stop token (and without
    // penalties / grammar) replaces a stop drawn too early.
    BackboneSampler guard;
    const bool use_guard = c.min_frames > 0 && c.stop_token >= 0;
    if (use_guard && !guard.build(s->lmodel, sp.seed, sp.temp, sp.top_k, sp.top_p, 0.0f, 1.0f, 0, std::string(),
                                  allowed_bias(c.stop_token), &serr)) {
        out->error = serr;
        return false;
    }

    // Prompt: chunk i of every sequence in one batch (sequence q on seq_id q),
    // outputs on each sequence's last row; every position's hidden state when
    // the model asks for them.
    llama_memory_clear(llama_get_memory(lctx), true);   // every generation starts from an empty context
    const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
    int32_t n_past = 0;
    std::vector<float> prompt_hidden;
    for (int32_t i = 0; i < c.n_chunks; ++i) {
        const codec_audio_lm_chunk & ch = c.chunks[i];
        if (ch.type == CODEC_AUDIO_LM_CHUNK_EMBD) {
            if (ch.n <= 0) continue;
            std::vector<float> rows;
            const float * src = ch.embd;
            if (n_seq > 1) {
                for (int32_t q = 0; q < n_seq; ++q) {
                    const codec_audio_lm_chunk & cq = c.chunks[q * c.n_chunks + i];
                    if (cq.type != CODEC_AUDIO_LM_CHUNK_EMBD || cq.n != ch.n) {
                        out->error = "the sequences' prompt chunks differ";
                        return false;
                    }
                    rows.insert(rows.end(), cq.embd, cq.embd + (size_t) cq.n * c.n_embd);
                }
                src = rows.data();
            }
            if (c.needs_prompt_hidden) { out->error = "prompt hidden states over embedding chunks are not supported"; return false; }
            if (n_past + ch.n + 1 >= n_ctx) { out->error = "the prompt does not fit the backbone context"; return false; }
            if (!decode_embd(lctx, src, c.n_embd, ch.n, n_seq, n_past)) { out->error = "prompt decode failed"; return false; }
            n_past += ch.n;
            continue;
        }
        if (n_seq > 1) { out->error = "several sequences take embedding prompts only"; return false; }
        const std::vector<llama_token> toks = ch.type == CODEC_AUDIO_LM_CHUNK_TEXT
            ? tokenize_str(vocab, ch.text, ch.add_special, ch.parse_special)
            : std::vector<llama_token>(ch.tokens, ch.tokens + ch.n);
        if (toks.empty()) continue;
        if (n_past + (int32_t) toks.size() + 1 >= n_ctx) {
            out->error = "the prompt (" + std::to_string(n_past + toks.size()) + " tokens) does not fit the backbone "
                         "context (" + std::to_string(n_ctx) + "); shorten the reference or raise --max-frames";
            return false;
        }
        if (sp.penalize_prompt) for (llama_token t : toks) common_sampler_accept(smpl.smpl, t, /*is_generated=*/false);
        if (!decode_tokens(lctx, toks, n_past, c.needs_prompt_hidden)) { out->error = "prompt decode failed"; return false; }
        for (int32_t k = 0; c.needs_prompt_hidden && k < (int32_t) toks.size(); ++k) {
            const float * hk = llama_get_embeddings_ith(lctx, k);
            if (hk == nullptr) { out->error = "backbone returned no prompt hidden state"; return false; }
            prompt_hidden.insert(prompt_hidden.end(), hk, hk + c.n_embd);
        }
        n_past += (int32_t) toks.size();
    }
    if (c.needs_prompt_hidden &&
        !codec_audio_lm_prefill(alm, prompt_hidden.data(), (int32_t) (prompt_hidden.size() / (size_t) c.n_embd))) {
        return fail("prefill");
    }

    // Steps: sample (when asked), observe, feed what the verdict says.
    const int32_t max_frames = a.max_frames > 0 ? a.max_frames : c.max_frames;
    const bool streaming = so != nullptr && c.streamable;
    bool sample = c.sample_first;
    std::vector<float> hiddens((size_t) n_seq * c.n_embd), rows((size_t) n_seq * c.n_embd);
    int32_t frames = 0;
    const char * stop = nullptr;
    while (frames < max_frames) {
        if (n_past >= n_ctx) { stop = "n_ctx"; break; }
        const float * h = nullptr;
        if (c.needs_embeddings && n_seq == 1) {
            h = llama_get_embeddings_ith(lctx, -1);
            if (h == nullptr) { out->error = "backbone returned no hidden state"; return false; }
        } else if (c.needs_embeddings) {
            // Sequence q's last output is -(n_seq - q) from the end.
            for (int32_t q = 0; q < n_seq; ++q) {
                const float * hq = llama_get_embeddings_ith(lctx, -(n_seq - q));
                if (hq == nullptr) { out->error = "backbone returned no hidden state"; return false; }
                std::memcpy(hiddens.data() + (size_t) q * c.n_embd, hq, (size_t) c.n_embd * sizeof(float));
            }
            h = hiddens.data();
        }
        llama_token tok = -1;
        if (sample) {
            tok = smpl.sample(lctx, -1);   // always first: its random stream advances every step
            if (use_guard && frames < c.min_frames && tok == c.stop_token) tok = guard.sample(lctx, -1);
        }
        codec_audio_lm_verdict v{};
        if (!codec_audio_lm_observe(alm, sample ? &tok : nullptr, h, &v)) return fail("observe");
        if (v.is_audio) ++frames;
        if (v.action == CODEC_AUDIO_LM_STOP) break;
        bool ok = true;
        if (v.action == CODEC_AUDIO_LM_FEED_TOKEN) {
            ok = decode_tokens(lctx, {v.token >= 0 ? v.token : tok}, n_past, false);
        } else {
            for (int32_t q = 0; q < n_seq; ++q) {
                std::memcpy(rows.data() + (size_t) q * c.n_embd, codec_audio_lm_next_embd(alm, q), (size_t) c.n_embd * sizeof(float));
            }
            ok = decode_embd(lctx, rows.data(), c.n_embd, 1, n_seq, n_past);
        }
        if (!ok) { out->error = "decode failed"; return false; }
        ++n_past;
        if (v.is_audio && streaming && !so->poll()) return false;
        sample = v.sample_backbone;
    }
    if (stop == nullptr) stop = codec_audio_lm_stop_reason(alm);
    out->n_frames    = frames;
    out->stop_reason = stop;
    if (streaming) return so->finish();

    codec_pcm_buffer buf{};
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = codec_audio_lm_decode(alm, &buf);
    out->decode_ms += ms_since(t0);
    if (!ok) return fail("decode");
    return tts_runner_take_pcm(&buf, out);
}

tts_runner_session * tts_runner_load(const tts_runner_params & a, std::string * err) {
    auto * s = new tts_runner_session;
    s->max_frames = a.max_frames;
    auto fail = [&](const std::string & e) -> tts_runner_session * {
        *err = e;
        delete s;
        return nullptr;
    };
    s->alm = tts_runner_open_codec(a, err);
    if (s->alm == nullptr) return fail(*err);
    codec_audio_lm_caps caps{};
    codec_audio_lm_get_caps(s->alm, &caps);
    if (!caps.needs_backbone) return s;   // self-contained (Pocket-TTS FlowLM)
    if (a.backbone_path.empty()) return fail("synthesize requires backbone_path (llama.cpp model)");

    llama_backend_init();
    s->llama_inited = true;
    llama_model_params mp = llama_model_default_params();
    // Without --gpu the backbone stays on the CPU: an empty device list, else
    // llama.cpp would still offload batched ops to any GPU it finds.
    static ggml_backend_dev_t no_devices[] = { nullptr };
    mp.n_gpu_layers = a.use_gpu ? 999 : 0;
    if (!a.use_gpu) mp.devices = no_devices;
    s->lmodel = llama_model_load_from_file(a.backbone_path.c_str(), mp);
    if (!s->lmodel) return fail("llama_model_load_from_file failed: " + a.backbone_path);

    // The context the model asks for, sized once for the load-time max_frames.
    const codec_audio_lm_vocab cv = host_vocab(llama_model_get_vocab(s->lmodel));
    codec_audio_lm_backbone_req req{};
    if (!codec_audio_lm_backbone_requirements(s->alm, &cv, a.max_frames, &req)) {
        return fail(std::string("backbone: ") + codec_audio_lm_last_error(s->alm));
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx        = (uint32_t) req.n_ctx;
    cp.n_batch      = cp.n_ctx;
    cp.n_ubatch     = req.n_ubatch > 0 ? (uint32_t) req.n_ubatch : cp.n_ctx;
    cp.n_seq_max    = (uint32_t) std::max(1, req.n_seq_max);
    cp.embeddings   = req.embeddings;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    if (!req.flash_attn) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (a.n_threads > 0) { cp.n_threads = a.n_threads; cp.n_threads_batch = a.n_threads; }
    cp.op_offload = a.use_gpu;
    s->lctx = llama_init_from_model(s->lmodel, cp);
    if (!s->lctx) return fail("llama_init_from_model failed");
    return s;
}

void tts_runner_free(tts_runner_session * s) { delete s; }

tts_runner_info tts_runner_get_info(tts_runner_session * s, const std::string & speaker_file) {
    tts_runner_info info;
    if (s == nullptr) return info;
    const int32_t n = codec_audio_lm_list_voices(s->alm, speaker_file.empty() ? nullptr : speaker_file.c_str());
    if (n < 0) info.error = codec_audio_lm_last_error(s->alm);
    for (int32_t i = 0; i < n; ++i) {
        const char * name = nullptr;
        bool is_default = false;
        if (!codec_audio_lm_voice(s->alm, i, &name, nullptr, &is_default)) continue;
        info.voices.push_back(name);
        if (is_default) info.default_voice = name;
    }
    codec_audio_lm_caps caps{};
    codec_audio_lm_get_caps(s->alm, &caps);
    info.takes_instruct = caps.takes_instruct;
    return info;
}

// One request: the voice once, then each text segment as its own generation,
// joined the model's way.
static bool tts_runner_run_segments(tts_runner_session * s, const tts_runner_params & a, tts_runner_result * out) {
    tts_runner_params ra = a;
    ra.max_frames = s->max_frames;   // the context was sized for the load-time cap
    std::vector<std::string> segments;
    const int32_t n_seg = codec_audio_lm_tts_split(s->alm, a.text.c_str(), a.segment_bytes);
    for (int32_t i = 0; i < n_seg; ++i) segments.push_back(codec_audio_lm_tts_segment(s->alm, i));
    if (segments.empty()) {
        out->error = "empty text";
        return false;
    }
    if (!tts_runner_set_voice(s->alm, a, &out->speaker_ms, &out->error)) return false;
    out->pcm.clear();
    out->n_frames  = 0;
    out->decode_ms = 0.0;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments.size() > 1) {
            codec_log(CODEC_LOG_LEVEL_INFO, "segment %zu/%zu: %zu bytes\n", i + 1, segments.size(), segments[i].size());
        }
        ra.text = segments[i];
        ra.seed = (a.seed ? a.seed : 0xC0DEC1ABu) + (uint32_t) i;
        tts_runner_result r;
        std::unique_ptr<stream_out> so;
        if (a.on_audio && a.stream_chunk_ms > 0) so = std::make_unique<stream_out>(s->alm, ra, &r);
        const bool ok = run_one(s, ra, &r, so.get());
        out->decode_ms += r.decode_ms;
        if (!ok) {
            out->error = segments.size() > 1 ? "segment " + std::to_string(i + 1) + ": " + r.error : r.error;
            return false;
        }
        out->n_frames   += r.n_frames;
        out->stop_reason = r.stop_reason;
        out->sample_rate = r.sample_rate;
        out->n_channels  = r.n_channels;
        if (so && so->used()) continue;   // streamed as it was generated
        // The model's joining: edge handling in place, then a gap of silence.
        codec_pcm_buffer own{};
        own.data = static_cast<float *>(std::malloc(std::max<size_t>(1, r.pcm.size()) * sizeof(float)));
        if (!r.pcm.empty()) std::memcpy(own.data, r.pcm.data(), r.pcm.size() * sizeof(float));
        own.n_channels  = r.n_channels;
        own.n_samples   = (int32_t) (r.pcm.size() / (size_t) std::max(1, r.n_channels));
        own.sample_rate = r.sample_rate;
        const int32_t gap = std::max(0, codec_audio_lm_tts_join(s->alm, &own, (int32_t) i, (int32_t) segments.size()));
        r.pcm.assign(own.data, own.data + (size_t) own.n_samples * (size_t) std::max(1, own.n_channels));
        codec_pcm_buffer_free(&own);
        const std::vector<float> silence((size_t) gap * (size_t) std::max(1, r.n_channels), 0.0f);
        if (!a.on_audio) {
            out->pcm.insert(out->pcm.end(), r.pcm.begin(), r.pcm.end());
            out->pcm.insert(out->pcm.end(), silence.begin(), silence.end());
        } else if ((!r.pcm.empty() && !a.on_audio(r.pcm.data(), (int32_t) (r.pcm.size() / std::max(1, r.n_channels)),
                                                  r.sample_rate, r.n_channels)) ||
                   (!silence.empty() && !a.on_audio(silence.data(), gap, r.sample_rate, r.n_channels))) {
            out->error = "stopped by the caller";
            return false;
        }
    }
    return true;
}

bool tts_runner_run(tts_runner_session * s, const tts_runner_params & a, tts_runner_result * out) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = tts_runner_run_segments(s, a, out);
    out->total_ms = ms_since(t0);
    return ok;
}

bool tts_runner_synthesize(const tts_runner_params & a, tts_runner_result * out) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    tts_runner_session * s = tts_runner_load(a, &err);
    if (s == nullptr) {
        out->error = err;
        out->total_ms = ms_since(t0);
        return false;
    }
    const double load_ms = ms_since(t0);
    const bool ok = tts_runner_run(s, a, out);
    out->load_ms = load_ms + out->speaker_ms;
    tts_runner_free(s);
    out->total_ms = ms_since(t0);
    return ok;
}

}  // namespace codec_common
