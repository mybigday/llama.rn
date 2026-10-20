#include "rn-tts.h"
#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"
#include "codec.h"
#include "codec_audio_lm.h"
#include "codec_common.h"
#include "llama.h"
#include "sampling.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#if defined(__linux__)
#include <sched.h>
#endif

namespace rnllama {

// A strict-CPU ggml threadpool (Android's default, see set_best_cores) pins
// the thread that calls into it to a single core and leaves it pinned.
// codec.cpp computes without a threadpool, so the workers it spawns inherit
// that one-core mask and every codec thread ends up sharing one core (a ~10x
// slower decode on a Pixel 7 Pro). Widen the calling thread to the context's
// CPU set while codec work runs and restore the old mask afterwards. Scopes
// must not span llama_decode, or the backbone's caller thread floats onto
// cores its pinned workers occupy.
class codec_affinity_scope {
public:
    explicit codec_affinity_scope(const llama_rn_context * main_ctx) {
#if defined(__linux__)
        if (main_ctx == nullptr || !main_ctx->params.cpuparams.mask_valid) return;
        cpu_set_t wide;
        CPU_ZERO(&wide);
        for (int i = 0; i < GGML_MAX_N_THREADS && i < CPU_SETSIZE; ++i) {
            if (main_ctx->params.cpuparams.cpumask[i]) CPU_SET(i, &wide);
        }
        if (CPU_COUNT(&wide) == 0) return;
        if (sched_getaffinity(0, sizeof(saved), &saved) != 0) return;
        if (CPU_EQUAL(&saved, &wide)) return;
        active = sched_setaffinity(0, sizeof(wide), &wide) == 0;
#else
        (void) main_ctx;
#endif
    }
    ~codec_affinity_scope() {
#if defined(__linux__)
        if (active) sched_setaffinity(0, sizeof(saved), &saved);
#endif
    }
    codec_affinity_scope(const codec_affinity_scope &) = delete;
    codec_affinity_scope & operator=(const codec_affinity_scope &) = delete;

private:
#if defined(__linux__)
    cpu_set_t saved;
    bool active = false;
#endif
};

namespace {

std::string lowercase_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

bool contains_ci(const std::string & haystack, const char * needle) {
    return lowercase_copy(haystack).find(needle) != std::string::npos;
}

std::string alm_error(const codec_audio_lm * alm, const char * what) {
    const char * e = codec_audio_lm_last_error(alm);
    return std::string(what) + (e && *e ? std::string(": ") + e : std::string());
}

// ── The backbone vocabulary for codec_audio_lm ──────────────────────────

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

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s, bool add_special,
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

// ── Backbone sampling ──────────────────────────────────────────────────
// common_params_sampling for the completion's codec_audio_lm_sampler, as
// codec.cpp's tts_runner builds it.  penalty_exempt needs a llama.cpp patch
// llama.rn does not carry; no model in llama.rn's catalog uses it.

// llama.cpp's penalties sampler treats penalty_last_n = 0 as "off", so "whole
// history" is a window longer than any generation.
constexpr int32_t kPenaltyFullHistory = 1 << 16;

// `plain`: the minimum-length guard (temp / top_k / top_p only).
common_params_sampling sampling_params(const codec_audio_lm_sampler & sp, const std::string & grammar,
                                       const std::vector<llama_logit_bias> & bias, bool plain) {
    common_params_sampling p;
    p.seed    = sp.seed;
    p.no_perf = true;
    p.temp    = sp.temp;
    p.top_k   = sp.top_k > 0 ? sp.top_k : 0;
    p.top_p   = (sp.top_p > 0.0f && sp.top_p < 1.0f) ? sp.top_p : 1.0f;
    p.min_p   = plain ? 0.0f : std::max(0.0f, sp.min_p);
    p.penalty_repeat  = plain ? 1.0f : sp.penalty_repeat;
    p.penalty_freq    = plain ? 0.0f : sp.penalty_freq;
    p.penalty_present = plain ? 0.0f : sp.penalty_present;
    const bool penalties = p.penalty_repeat != 1.0f || p.penalty_freq != 0.0f || p.penalty_present != 0.0f;
    p.penalty_last_n = penalties ? (sp.penalty_last_n > 0 ? sp.penalty_last_n : kPenaltyFullHistory) : 0;
    p.dry_multiplier     = plain ? 0.0f : sp.dry_multiplier;
    p.dry_base           = sp.dry_base;
    p.dry_allowed_length = sp.dry_allowed_length;
    p.dry_penalty_last_n = sp.dry_penalty_last_n;
    p.dry_sequence_breakers.clear();
    if (!plain && sp.samplers != nullptr && *sp.samplers) {
        std::vector<std::string> names;
        std::stringstream ss(sp.samplers);
        for (std::string n; std::getline(ss, n, ';');) if (!n.empty()) names.push_back(n);
        p.samplers = common_sampler_types_from_names(names);
    } else {
        p.samplers = {
            COMMON_SAMPLER_TYPE_PENALTIES,
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_MIN_P,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_TEMPERATURE,
        };
        if (p.dry_multiplier > 0.0f) p.samplers.insert(p.samplers.begin() + 1, COMMON_SAMPLER_TYPE_DRY);
    }
    if (!plain && !grammar.empty()) p.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, grammar);
    p.logit_bias = bias;
    return p;
}

// ── Backbone decode ────────────────────────────────────────────────────
// Prompts are decoded in pieces of at most n_batch positions per call.

struct decoder {
    llama_context * lctx;
    int32_t n_batch;
    int32_t n_past = 0;

    // Tokens on sequence 0.  `hiddens` (may be null) receives every
    // position's hidden state; otherwise only the last position outputs.
    void tokens(const std::vector<llama_token> & toks, std::vector<float> * hiddens, int32_t n_embd) {
        for (size_t at = 0; at < toks.size();) {
            const int32_t n = (int32_t) std::min<size_t>((size_t) n_batch, toks.size() - at);
            llama_batch b = llama_batch_init(n, 0, 1);
            for (int32_t i = 0; i < n; ++i) {
                b.token[i] = toks[at + i];
                b.pos[i] = n_past + i;
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = (hiddens != nullptr || at + i + 1 == toks.size()) ? 1 : 0;
            }
            b.n_tokens = n;
            const int rc = llama_decode(lctx, b);
            llama_batch_free(b);
            if (rc != 0) throw std::runtime_error("TTS: backbone decode failed");
            for (int32_t i = 0; hiddens != nullptr && i < n; ++i) {
                const float * h = llama_get_embeddings_ith(lctx, i);
                if (h == nullptr) throw std::runtime_error("TTS: backbone returned no prompt hidden state");
                hiddens->insert(hiddens->end(), h, h + n_embd);
            }
            n_past += n;
            at += (size_t) n;
        }
    }

    // `n` rows per sequence, sequences side by side (row r of sequence q at
    // rows[(q * n + r) * dim]); outputs on each sequence's last row.
    void embd(const float * rows, int32_t dim, int32_t n, int32_t n_seq) {
        const int32_t step = std::max(1, n_batch / std::max(1, n_seq));
        for (int32_t at = 0; at < n;) {
            const int32_t m = std::min(step, n - at);
            llama_batch b = llama_batch_init(m * n_seq, dim, 1);
            b.token = nullptr;
            b.n_tokens = m * n_seq;
            for (int32_t q = 0, i = 0; q < n_seq; ++q) {
                std::memcpy(b.embd + (size_t) i * dim, rows + ((size_t) q * n + at) * dim,
                            (size_t) m * dim * sizeof(float));
                for (int32_t r = 0; r < m; ++r, ++i) {
                    b.pos[i] = n_past + r;
                    b.n_seq_id[i] = 1;
                    b.seq_id[i][0] = q;
                    b.logits[i] = (at + r + 1 == n) ? 1 : 0;
                }
            }
            const int rc = llama_decode(lctx, b);
            llama_batch_free(b);
            if (rc != 0) throw std::runtime_error("TTS: backbone decode failed");
            n_past += m;
            at += m;
        }
    }
};

// Logit bias forbidding every token outside `allowed` (and `except`).
std::vector<llama_logit_bias> allowed_bias(const codec_audio_lm_completion & c, int32_t n_vocab, int32_t except) {
    std::vector<llama_logit_bias> bias;
    if (c.n_allowed == 0) return bias;
    std::vector<char> ok((size_t) n_vocab, 0);
    for (int32_t i = 0; i < c.n_allowed; ++i) {
        for (int32_t t = std::max(0, c.allowed[i].begin); t < std::min(n_vocab, c.allowed[i].end); ++t) {
            ok[(size_t) t] = 1;
        }
    }
    if (except >= 0 && except < n_vocab) ok[(size_t) except] = 0;
    for (int32_t t = 0; t < n_vocab; ++t) {
        if (!ok[(size_t) t]) bias.push_back(llama_logit_bias{t, -INFINITY});
    }
    return bias;
}

bool meta_true(const codec_audio_lm * alm, const char * key) {
    const char * v = codec_audio_lm_meta_str(alm, key);
    return v != nullptr && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0);
}

std::string meta_string(const codec_audio_lm * alm, const char * key) {
    const char * v = codec_audio_lm_meta_str(alm, key);
    return v ? v : "";
}

}  // namespace

// Minimum length: a second sampler without the stop token replaces a stop
// drawn too early.
struct tts_guard_sampler {
    common_sampler * smpl = nullptr;
    ~tts_guard_sampler() { if (smpl) common_sampler_free(smpl); }
};

// ── Lifecycle ─────────────────────────────────────────────────────────

llama_rn_context_tts::llama_rn_context_tts(const std::string &vocoder_model_path, const std::string &backbone_path,
                                           bool use_gpu, const std::vector<std::string> &kv_overrides,
                                           int n_threads) {
    // "KEY=TYPE:VALUE", the llama.cpp --override-kv syntax.
    std::vector<codec_model_kv_override> overrides;
    for (const auto & spec : kv_overrides) {
        codec_model_kv_override o{};
        if (!codec_common::parse_kv_override(spec.c_str(), &o)) {
            throw std::runtime_error("invalid codec kv override: " + spec);
        }
        overrides.push_back(o);
    }
    codec_audio_lm_params p = codec_audio_lm_default_params();
    p.codec_path     = vocoder_model_path.c_str();
    p.backbone_path  = backbone_path.empty() ? nullptr : backbone_path.c_str();
    p.kv_overrides   = overrides.empty() ? nullptr : overrides.data();
    p.n_kv_overrides = (int32_t) overrides.size();
    p.n_threads      = n_threads;
    p.use_gpu        = use_gpu;
    char err[512] = {0};
    alm = codec_audio_lm_init(&p, err, sizeof err);
    if (alm == nullptr) {
        throw std::runtime_error(std::string("codec_audio_lm_init: ") + (*err ? err : "failed"));
    }
}

llama_rn_context_tts::~llama_rn_context_tts() {
    if (alm != nullptr) codec_audio_lm_free(alm);
}

void llama_rn_context_tts::reset() {
    pending_speaker_id = -1;
    // The completion result has taken the last generation's data.
    audio_tokens.clear();
    audio_latents.clear();
    audio_latent_dim = 0;
    n_frames = 0;
}

// ── Capabilities ──────────────────────────────────────────────────────

llama_rn_tts_capabilities llama_rn_context_tts::getTTSCapabilities(llama_rn_context* main_ctx) {
    llama_rn_tts_capabilities cap;
    codec_audio_lm_caps c{};
    codec_audio_lm_get_caps(alm, &c);
    cap.requires_phonemes      = c.text_needs_phonemes;
    cap.can_speaker_from_audio = c.takes_ref_audio;
    cap.can_take_instruct      = c.takes_instruct;
    cap.needs_voice            = c.needs_voice;
    cap.sample_rate            = c.sample_rate;

    const int32_t n = codec_audio_lm_list_voices(alm, nullptr);
    for (int32_t i = 0; i < n; ++i) {
        const char * name = nullptr;
        const char * ref_text = nullptr;
        bool is_default = false;
        if (codec_audio_lm_voice(alm, i, &name, &ref_text, &is_default) && name != nullptr) {
            cap.builtin_voices.push_back({name, ref_text ? ref_text : "", is_default});
        }
    }

    // The model name for the JS layer, from the codec architecture (and the
    // backbone where the codec is shared).
    const std::string arch = meta_string(alm, "general.architecture");
    const std::string backbone_name = main_ctx && main_ctx->model ? main_ctx->model->name : "";
    auto set = [&](tts_type type, const char * family, const char * prompt_kind) {
        cap.type = type;
        cap.family = family;
        cap.prompt_kind = prompt_kind;
    };
    if (arch == "wavtokenizer_large" || arch == "wavtokenizer-large") {
        if (contains_ci(backbone_name, "0.3")) set(OUTETTS_V0_3, "outetts", "outetts_v0_3");
        else if (contains_ci(backbone_name, "0.1")) set(OUTETTS_V0_1, "outetts", "outetts_legacy");
        else set(OUTETTS_V0_2, "outetts", "outetts_legacy");
    } else if (arch == "dac") {
        set(OUTETTS_V1_0, "outetts", "outetts_v1_0");
    } else if (arch == "soprano") {
        set(SOPRANO_1_1_80M, "soprano", "soprano");
    } else if (arch == "neucodec" || arch == "distill_neucodec") {
        const llama_vocab * vocab = main_ctx && main_ctx->model ? llama_model_get_vocab(main_ctx->model) : nullptr;
        const std::vector<llama_token> speech0 = vocab ? tokenize(vocab, "<|speech_0|>", false, true) : std::vector<llama_token>{};
        const bool air = speech0.size() == 1 && speech0[0] == 151671;
        set(air ? NEUTTS_AIR : NEUTTS_NANO, "neutts", "neutts");
    } else if (arch == "mimi") {
        set(CSM_1B, "csm", "csm");
    } else if (arch == "qwen3_tts_tokenizer" || arch == "qwen3-tts-tokenizer" || arch == "qwen3") {
        set(QWEN3_TTS_0_6B, "qwen3_tts", "qwen3_tts");
    } else if (arch == "moss_audio_tokenizer" || arch == "moss-audio-tokenizer" || arch == "moss_audio") {
        set(MOSS_TTS_REALTIME, "moss_tts", "moss_tts_realtime");
    } else if (arch == "xy_tokenizer" || arch == "xy-tokenizer") {
        set(MOSS_TTSD_V07, "moss_ttsd", "moss_ttsd");
    } else if (arch == "chatterbox_s3g" || arch == "chatterbox-s3g" || arch == "s3g") {
        if (meta_string(alm, "codec.lm.chatterbox.variant") == "turbo") {
            set(CHATTERBOX_TURBO, "chatterbox", "chatterbox_turbo");
        } else if (meta_true(alm, "codec.lm.chatterbox.is_multilingual")) {
            set(CHATTERBOX_T3_MULTILINGUAL, "chatterbox", "chatterbox_multilingual");
        } else {
            set(CHATTERBOX_T3, "chatterbox", "chatterbox");
        }
    } else if (arch == "bluemagpie_audiovae" || arch == "bluemagpie-audiovae") {
        set(BLUEMAGPIE_TTS, "bluemagpie", "bluemagpie");
        // Trained on Taiwanese-Mandarin data.
        cap.default_language = "zh-tw";
    } else if (arch == "cosyvoice") {
        set(COSYVOICE, "cosyvoice", "cosyvoice");
        // BreezyVoice reads Mandarin with 注音 annotations (g2pw upstream);
        // the phonemizer hook is where the app adds them.
        const bool bopomofo = meta_true(alm, "codec.lm.cosyvoice.bopomofo_frontend");
        cap.requires_phonemes = cap.requires_phonemes || bopomofo;
        cap.default_language = bopomofo ? "zh-tw" : "zh";
    }
    return cap;
}

std::string llama_rn_context_tts::checkBackbone(llama_rn_context* main_ctx) {
    if (main_ctx == nullptr || main_ctx->ctx == nullptr) return "no backbone context";
    const codec_audio_lm_vocab cv = host_vocab(llama_model_get_vocab(main_ctx->model));
    codec_audio_lm_backbone_req req{};
    if (!codec_audio_lm_backbone_requirements(alm, &cv, 0, &req)) return alm_error(alm, "backbone");
    // A guidance pair needs two sequences; with one, generate without guidance.
    no_guidance = (int32_t) llama_n_seq_max(main_ctx->ctx) < req.n_seq_max;
    if (no_guidance) {
        LOG_WARNING("TTS: the context decodes %u sequence(s), the model's guidance needs %d; generating "
                    "without guidance (create the context with n_parallel >= %d)",
                    llama_n_seq_max(main_ctx->ctx), req.n_seq_max, req.n_seq_max);
    }
    if (!req.flash_attn && main_ctx->params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        return "this model needs flash attention disabled (flash_attn_type 'off' or 'auto', F16 KV cache)";
    }
    if ((int32_t) llama_n_ctx(main_ctx->ctx) < req.n_ctx) {
        LOG_WARNING("TTS: the context has %u positions, the model asks for %d; long texts may stop early",
                    llama_n_ctx(main_ctx->ctx), req.n_ctx);
    }
    return "";
}

// ── Requests ──────────────────────────────────────────────────────────

void llama_rn_context_tts::applyVoice(const armed_voice &v) {
    if (voice_applied && applied_v == v) return;
    codec_audio_lm_tts_voice voice = codec_audio_lm_default_tts_voice();
    if (v.speaker_id >= 0) {
        const rn_speaker * spk = getSpeaker(v.speaker_id);
        if (spk == nullptr) throw std::runtime_error("TTS: speaker " + std::to_string(v.speaker_id) + " not found");
        if (!spk->speaker_file.empty()) {
            voice.speaker_file = spk->speaker_file.c_str();
            voice.voice = spk->speaker_name.empty() ? nullptr : spk->speaker_name.c_str();
        } else {
            voice.ref_pcm         = spk->pcm.data();
            voice.ref_n_samples   = (int32_t) spk->pcm.size();
            voice.ref_sample_rate = spk->sample_rate;
            voice.ref_text        = spk->ref_text.empty() ? nullptr : spk->ref_text.c_str();
        }
    } else if (!v.voice.empty()) {
        voice.voice = v.voice.c_str();
    }
    voice_applied = false;
    if (!codec_audio_lm_tts_set_voice(alm, &voice)) throw std::runtime_error(alm_error(alm, "TTS voice"));
    applied_v = v;
    voice_applied = true;
}

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(
        llama_rn_context* main_ctx, const json &speaker, const std::string &text_to_speak, int speakerId,
        const std::string &instruct) {
    const std::string why = checkBackbone(main_ctx);
    if (!why.empty()) throw std::runtime_error("TTS: " + why);
    if (!instruct.empty()) {
        codec_audio_lm_caps c{};
        codec_audio_lm_get_caps(alm, &c);
        if (!c.takes_instruct) throw std::runtime_error("TTS: this model takes no instruct");
    }

    armed_voice v;
    v.speaker_id = speakerId;
    if (speakerId < 0 && speaker.is_object() && speaker.contains("codec_voice") && speaker["codec_voice"].is_string()) {
        v.voice = speaker["codec_voice"].get<std::string>();
    } else if (speakerId < 0 && speaker.is_object() && !speaker.empty()) {
        LOG_WARNING("TTS: speaker payloads are not used; the codec's builtin voices and LlamaSpeaker are");
    }
    {
        codec_affinity_scope affinity(main_ctx);
        applyVoice(v);
    }
    armed = true;
    armed_text = text_to_speak;
    armed_instruct = instruct;
    armed_v = v;

    // The generation turns hidden states on itself where the model needs them.
    llama_rn_audio_completion_result res;
    res.prompt = text_to_speak;
    res.embedding = false;
    res.flow = "tokens";
    return res;
}

int llama_rn_context_tts::beginGeneration(llama_rn_context* main_ctx) {
    if (!armed) throw std::runtime_error("TTS: no request; call getFormattedAudioCompletion first");
    // One completion per request: the context runs text completions again after it.
    armed = false;
    audio_tokens.clear();
    audio_latents.clear();
    audio_latent_dim = 0;
    n_frames = 0;
    stop_reason.clear();

    armed_voice v = armed_v;
    if (pending_speaker_id >= 0) v = armed_voice{"", pending_speaker_id};
    segments.clear();
    {
        codec_affinity_scope affinity(main_ctx);
        applyVoice(v);
        const int32_t n = codec_audio_lm_tts_split(alm, armed_text.c_str(), 0);
        if (n < 0) throw std::runtime_error(alm_error(alm, "TTS split"));
        for (int32_t i = 0; i < n; ++i) segments.emplace_back(codec_audio_lm_tts_segment(alm, i));
    }
    if (segments.empty()) throw std::runtime_error("TTS: nothing to say");

    saved_sampling = std::make_unique<common_params_sampling>(main_ctx->params.sampling);
    saved_embedding = main_ctx->params.embedding;
    generating = true;
    seg = 0;
    return beginSegment(main_ctx);
}

int llama_rn_context_tts::beginSegment(llama_rn_context* main_ctx) {
    llama_context * lctx = main_ctx->ctx;
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    const codec_audio_lm_vocab cv = host_vocab(vocab);
    auto fail = [&](const char * what) { return std::runtime_error(alm_error(alm, what)); };

    codec_audio_lm_tts_request req = codec_audio_lm_default_tts_request();
    req.text = segments[seg].c_str();
    req.instruct = armed_instruct.empty() ? nullptr : armed_instruct.c_str();
    // The completion's seed, when it sets one; the rest of the sampling is the model's.
    if (saved_sampling && saved_sampling->seed != LLAMA_DEFAULT_SEED) req.seed = saved_sampling->seed;
    if (no_guidance) req.lm_guidance = 0.0f;
    codec_audio_lm_completion c{};
    {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_tts_prompt(alm, &req, &cv)) throw fail("TTS prompt");
        if (!codec_audio_lm_get_completion(alm, &c)) throw fail("TTS completion");
    }
    if (c.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) {
        throw std::runtime_error("TTS: this model runs without a backbone; llama.rn does not host it");
    }
    n_seq = std::max(1, c.n_seq);
    if (n_seq > 1 && c.sample_first) {
        throw std::runtime_error("TTS: backbone sampling over several sequences is not supported");
    }
    n_embd       = c.n_embd;
    needs_hidden = c.needs_embeddings;
    sample_next  = c.sample_first;
    min_frames   = c.min_frames;
    stop_token   = c.stop_token;
    seg_frames   = 0;
    seg_open     = true;
    const int32_t n_ctx = (int32_t) llama_n_ctx(lctx);
    seg_max_frames = c.max_frames > 0 ? c.max_frames : n_ctx;

    // The completion's sampler follows the model; the guard drops the stop
    // token while the generation is shorter than min_frames.
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    main_ctx->params.sampling = sampling_params(c.sampler, c.grammar ? c.grammar : "", allowed_bias(c, n_vocab, -1), false);
    guard.reset();
    if (min_frames > 0 && stop_token >= 0) {
        guard = std::make_unique<tts_guard_sampler>();
        common_params_sampling gp = sampling_params(c.sampler, "", allowed_bias(c, n_vocab, stop_token), true);
        guard->smpl = common_sampler_init(main_ctx->model, gp);
        if (guard->smpl == nullptr) throw std::runtime_error("TTS: guard sampler init failed");
    }

    // Every segment starts from an empty context.
    llama_memory_clear(llama_get_memory(lctx), true);
    llama_set_embeddings(lctx, c.needs_embeddings || c.needs_prompt_hidden);
    decoder dec{lctx, std::max(1, main_ctx->params.n_batch)};
    const int32_t hidden_dim = llama_model_n_embd(main_ctx->model);
    std::vector<float> prompt_hidden;
    penalty_prompt.clear();
    for (int32_t i = 0; i < c.n_chunks; ++i) {
        const codec_audio_lm_chunk & ch = c.chunks[i];
        if (ch.type == CODEC_AUDIO_LM_CHUNK_EMBD) {
            if (ch.n <= 0) continue;
            if (c.needs_prompt_hidden) throw std::runtime_error("TTS: prompt hidden states over embedding chunks are not supported");
            if (dec.n_past + ch.n + 1 >= n_ctx) throw std::runtime_error("TTS: the prompt does not fit the context");
            std::vector<float> rows;
            const float * src = ch.embd;
            if (n_seq > 1) {
                for (int32_t q = 0; q < n_seq; ++q) {
                    const codec_audio_lm_chunk & cq = c.chunks[q * c.n_chunks + i];
                    if (cq.type != CODEC_AUDIO_LM_CHUNK_EMBD || cq.n != ch.n) {
                        throw std::runtime_error("TTS: the sequences' prompt chunks differ");
                    }
                    rows.insert(rows.end(), cq.embd, cq.embd + (size_t) cq.n * c.n_embd);
                }
                src = rows.data();
            }
            dec.embd(src, c.n_embd, ch.n, n_seq);
            continue;
        }
        if (n_seq > 1) throw std::runtime_error("TTS: several sequences take embedding prompts only");
        const std::vector<llama_token> toks = ch.type == CODEC_AUDIO_LM_CHUNK_TEXT
            ? tokenize(vocab, ch.text ? ch.text : "", ch.add_special, ch.parse_special)
            : std::vector<llama_token>(ch.tokens, ch.tokens + ch.n);
        if (toks.empty()) continue;
        if (dec.n_past + (int32_t) toks.size() + 1 >= n_ctx) {
            throw std::runtime_error("TTS: the prompt (" + std::to_string(dec.n_past + toks.size()) +
                                     " tokens) does not fit the context (" + std::to_string(n_ctx) + ")");
        }
        if (c.sampler.penalize_prompt) penalty_prompt.insert(penalty_prompt.end(), toks.begin(), toks.end());
        dec.tokens(toks, c.needs_prompt_hidden ? &prompt_hidden : nullptr, hidden_dim);
    }
    if (c.needs_prompt_hidden) {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_prefill(alm, prompt_hidden.data(), (int32_t) (prompt_hidden.size() / (size_t) hidden_dim))) {
            throw fail("TTS prefill");
        }
    }
    return dec.n_past;
}

llama_rn_context_tts::step_result llama_rn_context_tts::step(llama_rn_context* main_ctx, common_sampler * smpl,
                                                             int n_past) {
    llama_context * lctx = main_ctx->ctx;
    step_result r;
    if (seg_frames >= seg_max_frames) {
        stop_reason = "max_frames";
        r.stop = true;
        return r;
    }
    if (!penalty_prompt.empty()) {
        for (llama_token t : penalty_prompt) common_sampler_accept(smpl, t, /*is_generated=*/false);
        penalty_prompt.clear();
    }

    std::vector<float> hiddens;
    if (needs_hidden) {
        // Sequence q's last output is -(n_seq - q) from the end.
        const int32_t dim = llama_model_n_embd(main_ctx->model);
        hiddens.resize((size_t) n_seq * dim);
        for (int32_t q = 0; q < n_seq; ++q) {
            const float * hq = llama_get_embeddings_ith(lctx, -(n_seq - q));
            if (hq == nullptr) throw std::runtime_error("TTS: backbone returned no hidden state");
            std::memcpy(hiddens.data() + (size_t) q * dim, hq, (size_t) dim * sizeof(float));
        }
    }
    llama_token tok = -1;
    if (sample_next) {
        // The primary sampler draws first, so its random stream advances every step.
        tok = common_sampler_sample(smpl, lctx, -1, /*grammar_first=*/false);
        common_sampler_accept(smpl, tok, /*is_generated=*/true);
        if (guard && seg_frames < min_frames && tok == stop_token) {
            tok = common_sampler_sample(guard->smpl, lctx, -1, false);
            common_sampler_accept(guard->smpl, tok, true);
        }
    }
    r.sampled = tok;
    codec_audio_lm_verdict v{};
    std::vector<float> rows;
    {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_observe(alm, sample_next ? &tok : nullptr, needs_hidden ? hiddens.data() : nullptr, &v)) {
            throw std::runtime_error(alm_error(alm, "TTS step"));
        }
        if (v.action == CODEC_AUDIO_LM_FEED_EMBD) {
            rows.resize((size_t) n_seq * n_embd);
            for (int32_t q = 0; q < n_seq; ++q) {
                std::memcpy(rows.data() + (size_t) q * n_embd, codec_audio_lm_next_embd(alm, q),
                            (size_t) n_embd * sizeof(float));
            }
        }
    }
    if (v.is_audio) {
        ++seg_frames;
        r.audio = true;
        r.tok = tok >= 0 ? tok : (llama_token) n_frames;
        ++n_frames;
    }
    if (v.action == CODEC_AUDIO_LM_STOP) {
        stop_reason = codec_audio_lm_stop_reason(alm);
        r.stop = true;
        return r;
    }
    decoder dec{lctx, std::max(1, main_ctx->params.n_batch), n_past};
    if (v.action == CODEC_AUDIO_LM_FEED_TOKEN) {
        dec.tokens({v.token >= 0 ? v.token : tok}, nullptr, n_embd);
    } else {
        dec.embd(rows.data(), n_embd, 1, n_seq);
    }
    r.fed = true;
    sample_next = v.sample_backbone;
    return r;
}

void llama_rn_context_tts::finishSegment() {
    seg_open = false;
    codec_audio_lm_generation gen{};
    if (!codec_audio_lm_get_generation(alm, &gen)) throw std::runtime_error(alm_error(alm, "TTS generation"));
    if (gen.n_frames <= 0) return;
    const size_t n = (size_t) gen.n_frames * (size_t) gen.width;
    if (gen.kind == CODEC_AUDIO_LM_DATA_LATENTS) {
        audio_latent_dim = gen.width;
        audio_latents.insert(audio_latents.end(), gen.latents, gen.latents + n);
    } else {
        code_width = gen.width;
        audio_tokens.insert(audio_tokens.end(), gen.codes, gen.codes + n);
    }
}

int llama_rn_context_tts::nextSegment(llama_rn_context* main_ctx) {
    finishSegment();
    if (seg + 1 >= segments.size()) return -1;
    ++seg;
    stop_reason.clear();
    return beginSegment(main_ctx);
}

void llama_rn_context_tts::endGeneration(llama_rn_context* main_ctx) {
    if (!generating) return;
    generating = false;
    guard.reset();
    penalty_prompt.clear();
    if (saved_sampling) main_ctx->params.sampling = *saved_sampling;
    saved_sampling.reset();
    main_ctx->params.embedding = saved_embedding;
    llama_set_embeddings(main_ctx->ctx, saved_embedding);
    // The KV cache holds TTS rows; text completions start over.
    llama_memory_clear(llama_get_memory(main_ctx->ctx), true);
    if (stop_reason.empty()) stop_reason = "interrupted";
    if (seg_open) finishSegment();
}

llama_rn_audio_codes_result llama_rn_context_tts::generateAudioCodes(llama_rn_context* main_ctx,
                                                                     const llama_rn_audio_codes_options &opts,
                                                                     const llama_rn_audio_codes_progress_cb &on_frame) {
    llama_rn_context_completion * completion = main_ctx->completion;
    completion->rewind();
    getFormattedAudioCompletion(main_ctx, json(nullptr), opts.prompt, pending_speaker_id);
    main_ctx->params.n_predict = opts.max_frames > 0 ? opts.max_frames : -1;
    if (!completion->initSampling()) throw std::runtime_error("generateAudioCodes: sampler init failed");
    completion->beginCompletion();
    completion->loadPrompt({});
    int step_i = 0;
    while (completion->has_next_token && !completion->is_interrupted) {
        const completion_token_output t = completion->doCompletion();
        if (t.tok >= 0 && on_frame && !on_frame(step_i++, {t.tok})) completion->is_interrupted = true;
    }
    completion->endCompletion();

    llama_rn_audio_codes_result res;
    res.codes.assign(audio_tokens.begin(), audio_tokens.end());
    res.n_codebook = code_width;
    res.n_frames = n_frames;
    res.aborted = stop_reason == "interrupted";
    res.stopped_on_eos = completion->stopped_eos;
    return res;
}

// ── Speakers ──────────────────────────────────────────────────────────

int llama_rn_context_tts::createSpeaker(llama_rn_context* main_ctx, const std::vector<float> &pcm, int sample_rate,
                                        const std::string &ref_text, float emotion, bool has_emotion, bool bake) {
    (void) main_ctx;
    (void) emotion;
    (void) has_emotion;
    (void) bake;
    if (pcm.empty() || sample_rate <= 0) throw std::runtime_error("createSpeaker: empty reference audio");
    rn_speaker spk;
    spk.pcm = pcm;
    spk.sample_rate = sample_rate;
    spk.ref_text = ref_text;
    const int id = next_speaker_id++;
    speakers[id] = std::move(spk);
    return id;
}

bool llama_rn_context_tts::bakeSpeaker(llama_rn_context* main_ctx, int id) {
    (void) main_ctx;
    return getSpeaker(id) != nullptr;
}

const rn_speaker* llama_rn_context_tts::getSpeaker(int id) const {
    const auto it = speakers.find(id);
    return it == speakers.end() ? nullptr : &it->second;
}

void llama_rn_context_tts::releaseSpeaker(int id) {
    speakers.erase(id);
    if (voice_applied && applied_v.speaker_id == id) voice_applied = false;
}

int llama_rn_context_tts::loadSpeaker(const std::string &path, const std::string &name) {
    const std::vector<llama_rn_speaker_info> in_file = listSpeakers(path);
    if (!name.empty() && std::none_of(in_file.begin(), in_file.end(),
                                      [&](const llama_rn_speaker_info & s) { return s.name == name; })) {
        throw std::runtime_error("loadSpeaker: no speaker '" + name + "' in " + path);
    }
    rn_speaker spk;
    spk.speaker_file = path;
    spk.speaker_name = name;
    const int id = next_speaker_id++;
    speakers[id] = std::move(spk);
    return id;
}

void llama_rn_context_tts::saveSpeaker(int id, const std::string &path, const std::string &name) {
    if (!name.empty()) LOG_WARNING("saveSpeaker: codec.cpp names the saved speaker itself; '%s' is not used", name.c_str());
    const rn_speaker * spk = getSpeaker(id);
    if (spk == nullptr) throw std::runtime_error("saveSpeaker: speaker " + std::to_string(id) + " not found");
    codec_audio_lm_tts_voice voice = codec_audio_lm_default_tts_voice();
    if (!spk->speaker_file.empty()) {
        voice.speaker_file = spk->speaker_file.c_str();
        voice.voice = spk->speaker_name.empty() ? nullptr : spk->speaker_name.c_str();
    } else {
        voice.ref_pcm         = spk->pcm.data();
        voice.ref_n_samples   = (int32_t) spk->pcm.size();
        voice.ref_sample_rate = spk->sample_rate;
        voice.ref_text        = spk->ref_text.empty() ? nullptr : spk->ref_text.c_str();
    }
    voice.save_speaker = path.c_str();
    voice_applied = false;
    if (!codec_audio_lm_tts_set_voice(alm, &voice)) throw std::runtime_error(alm_error(alm, "saveSpeaker"));
    applied_v = armed_voice{"", id};
    voice_applied = true;
}

std::vector<llama_rn_speaker_info> llama_rn_context_tts::listSpeakers(const std::string &path) {
    const int32_t n = codec_audio_lm_list_voices(alm, path.c_str());
    if (n < 0) throw std::runtime_error(alm_error(alm, "listSpeakers"));
    std::vector<llama_rn_speaker_info> out;
    for (int32_t i = 0; i < n; ++i) {
        const char * name = nullptr;
        const char * ref_text = nullptr;
        bool is_default = false;
        if (codec_audio_lm_voice(alm, i, &name, &ref_text, &is_default) && name != nullptr) {
            out.push_back({name, ref_text ? ref_text : "", is_default});
        }
    }
    return out;
}

// ── Audio ─────────────────────────────────────────────────────────────

std::vector<float> llama_rn_context_tts::decodeGeneration(const codec_audio_lm_generation &gen, const char * what) {
    codec_pcm_buffer buf{};
    if (!codec_audio_lm_decode_generation(alm, &gen, &buf)) throw std::runtime_error(alm_error(alm, what));
    const int32_t n_ch = std::max(1, buf.n_channels);
    std::vector<float> pcm((size_t) buf.n_samples);
    for (int32_t i = 0; i < buf.n_samples; ++i) {
        float acc = 0.0f;
        for (int32_t ch = 0; ch < n_ch; ++ch) acc += buf.data[(size_t) i * n_ch + ch];
        pcm[(size_t) i] = acc / (float) n_ch;
    }
    codec_pcm_buffer_free(&buf);
    return pcm;
}

std::vector<float> llama_rn_context_tts::decodeAudioTokens(llama_rn_context* main_ctx,
                                                           const std::vector<llama_token> &tokens) {
    if (tokens.empty()) throw std::runtime_error("decodeAudioTokens: no audio tokens");
    if (code_width <= 0) throw std::runtime_error("decodeAudioTokens: this model's codes are unknown until it generates");
    if (tokens.size() % (size_t) code_width != 0) {
        throw std::runtime_error("decodeAudioTokens: " + std::to_string(tokens.size()) + " codes are not whole frames of " +
                                 std::to_string(code_width));
    }
    codec_audio_lm_generation gen{};
    gen.kind     = CODEC_AUDIO_LM_DATA_CODES;
    gen.n_frames = (int32_t) (tokens.size() / (size_t) code_width);
    gen.width    = code_width;
    gen.codes    = tokens.data();
    codec_affinity_scope affinity(main_ctx);
    return decodeGeneration(gen, "decodeAudioTokens");
}

std::vector<float> llama_rn_context_tts::decodeAudioEmbeddings(llama_rn_context* main_ctx,
                                                               const std::vector<float> &embeddings,
                                                               int embedding_dim) {
    if (embeddings.empty() || embedding_dim <= 0 || embeddings.size() % (size_t) embedding_dim != 0) {
        throw std::runtime_error("decodeAudioEmbeddings: expected whole frames of embedding_dim");
    }
    codec_audio_lm_generation gen{};
    gen.kind     = CODEC_AUDIO_LM_DATA_LATENTS;
    gen.n_frames = (int32_t) (embeddings.size() / (size_t) embedding_dim);
    gen.width    = embedding_dim;
    gen.latents  = embeddings.data();
    codec_affinity_scope affinity(main_ctx);
    return decodeGeneration(gen, "decodeAudioEmbeddings");
}

int llama_rn_context_tts::getAudioSampleRate() const {
    codec_audio_lm_caps c{};
    codec_audio_lm_get_caps(alm, &c);
    return c.sample_rate;
}

}
