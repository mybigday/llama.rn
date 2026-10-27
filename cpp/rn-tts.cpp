#include "rn-tts.h"
#include "rn-llama.h"
#include "rn-completion.h"
#include "common.h"
#include "codec.h"
#include "codec_audio_lm.h"
#include "codec_speaker.h"
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
    // A backbone without a vocabulary (old Chatterbox GGUFs) tokenizes
    // nothing: codec.cpp falls back to its own tokenizer.  llama_tokenize
    // would assert.
    const llama_vocab * vocab = (const llama_vocab *) ud;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) return 0;
    return llama_tokenize(vocab, text, len, out, cap, add_special, parse_special);
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


bool meta_true(const codec_audio_lm * alm, const char * key) {
    const char * v = codec_audio_lm_meta_str(alm, key);
    return v != nullptr && (std::strcmp(v, "true") == 0 || std::strcmp(v, "1") == 0);
}

std::string meta_string(const codec_audio_lm * alm, const char * key) {
    const char * v = codec_audio_lm_meta_str(alm, key);
    return v ? v : "";
}

}  // namespace

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

void llama_rn_context_tts::reset() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The completion result has taken the last generation's data.
    audio_tokens.clear();
    audio_latents.clear();
    audio_latent_dim = 0;
    n_frames = 0;
    stop_reason.clear();
    has_generation = false;
}

// ── Capabilities ──────────────────────────────────────────────────────

llama_rn_tts_capabilities llama_rn_context_tts::getTTSCapabilities(llama_rn_context* main_ctx) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    llama_rn_tts_capabilities cap;
    codec_audio_lm_caps c{};
    codec_audio_lm_get_caps(alm, &c);
    cap.requires_phonemes      = c.text_needs_phonemes;
    cap.can_speaker_from_audio = c.takes_ref_audio;
    cap.can_take_instruct      = c.takes_instruct;
    cap.needs_voice            = c.needs_voice;
    cap.uses_ref_text          = c.takes_ref_text;
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
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (main_ctx == nullptr || main_ctx->ctx == nullptr) return "no backbone context";
    const codec_audio_lm_vocab cv = host_vocab(llama_model_get_vocab(main_ctx->model));
    codec_audio_lm_backbone_req req{};
    if (!codec_audio_lm_backbone_requirements(alm, &cv, 0, &req)) return alm_error(alm, "backbone");
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

rn_voice llama_rn_context_tts::resolveVoice(int speaker_id, const std::string &voice_name) {
    if (speaker_id >= 0) {
        const rn_speaker * spk = getSpeaker(speaker_id);
        if (spk == nullptr) throw std::runtime_error("TTS: speaker " + std::to_string(speaker_id) + " not found");
        return spk->voice;
    }
    // Builtin voices belong to the model: borrowed, never freed.
    const codec_speaker * builtin = codec_audio_lm_get_voice(alm, voice_name.empty() ? nullptr : voice_name.c_str());
    if (builtin == nullptr && !voice_name.empty()) throw std::runtime_error("TTS: no voice '" + voice_name + "'");
    return rn_voice(builtin, [](const codec_speaker *) {});
}

namespace {

// llama.rn's llama.cpp has no penalty exemption; no hosted model asks for it.
void check_exempt(int32_t n_exempt) {
    if (n_exempt > 0) throw std::runtime_error("TTS: this model needs penalty-exempt tokens, which llama.rn does not support");
}

}  // namespace

llama_rn_audio_completion_result llama_rn_context_tts::getFormattedAudioCompletion(
        llama_rn_context* main_ctx, const json &speaker, const std::string &text_to_speak, int speakerId,
        const std::string &instruct) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const std::string why = checkBackbone(main_ctx);
    if (!why.empty()) throw std::runtime_error("TTS: " + why);
    if (!instruct.empty()) {
        codec_audio_lm_caps caps{};
        codec_audio_lm_get_caps(alm, &caps);
        if (!caps.takes_instruct) throw std::runtime_error("TTS: this model takes no instruct");
    }
    std::string voice_name;
    if (speakerId < 0 && speaker.is_object() && speaker.contains("codec_voice") && speaker["codec_voice"].is_string()) {
        voice_name = speaker["codec_voice"].get<std::string>();
    } else if (speakerId < 0 && speaker.is_object() && !speaker.empty()) {
        throw std::runtime_error("TTS: speaker payloads are not used; pass a builtin voice name or a LlamaSpeaker");
    }
    // A voice written as text (OuteTTS / NeuTTS references) goes into the
    // prompt; an embedding voice becomes a <__voice__> marker that the
    // completion's speaker fills.
    const rn_voice voice = resolveVoice(speakerId, voice_name);

    codec_audio_lm_tts_request req = codec_audio_lm_default_tts_request();
    req.text       = text_to_speak.c_str();
    req.voice      = voice.get();
    req.instruct   = instruct.empty() ? nullptr : instruct.c_str();
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    const codec_audio_lm_vocab cv = host_vocab(vocab);
    codec_audio_lm_format f{};
    {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_tts_format(alm, &req, &cv, &f)) throw std::runtime_error(alm_error(alm, "TTS format"));
    }
    if (f.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) {
        throw std::runtime_error("TTS: this model runs without a backbone; llama.rn does not host it");
    }
    if ((int32_t) llama_n_seq_max(main_ctx->ctx) < std::max(1, f.n_seq)) {
        throw std::runtime_error("TTS: the model's guidance decodes " + std::to_string(f.n_seq) +
                                 " sequences; create the context with n_parallel >= " + std::to_string(f.n_seq) +
                                 " or turn the guidance off with the codec's kv_overrides");
    }
    check_exempt(f.n_penalty_exempt);

    // What the request's format needs, as completion() params: the prompt,
    // the grammar or token mask, and hidden states.  Sampling and length are
    // the caller's.
    json p = json::object();
    p["prompt"]    = f.prompt ? f.prompt : "";
    p["embedding"] = f.needs_embeddings || f.needs_prompt_hidden;
    // The grammar and the allowed ranges say the same; the grammar wins.
    // The ranges become a logit bias table: -Infinity outside them.
    const std::string grammar = f.grammar ? f.grammar : "";
    std::vector<float> bias;
    if (!grammar.empty()) {
        p["grammar"] = grammar;
    } else if (f.n_allowed > 0) {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        bias.assign((size_t) n_vocab, -INFINITY);
        for (int32_t i = 0; i < f.n_allowed; ++i) {
            for (int32_t t = std::max(0, f.allowed[i].begin); t < std::min(n_vocab, f.allowed[i].end); ++t) {
                bias[(size_t) t] = 0.0f;
            }
        }
    }

    llama_rn_audio_completion_result res;
    res.prompt = p["prompt"].get<std::string>();
    res.grammar = grammar;
    res.embedding = p["embedding"].get<bool>();
    res.flow = "tokens";
    res.completion_params = std::move(p);
    res.logit_bias = std::move(bias);
    return res;
}

// ── Completion hooks ──────────────────────────────────────────────────

llama_rn_tts_prompt llama_rn_context_tts::buildPrompt(llama_rn_context* main_ctx) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The completion's voice fills the prompt's <__voice__> markers (the
    // model's default without one).
    gen_voice = resolveVoice(main_ctx->tts_speaker_id, main_ctx->tts_voice);
    codec_audio_lm_tts_request req = codec_audio_lm_default_tts_request();
    req.voice      = gen_voice.get();
    req.min_frames = main_ctx->n_min_predict;   // < 0: the model's own minimum
    const llama_vocab * vocab = llama_model_get_vocab(main_ctx->model);
    const codec_audio_lm_vocab cv = host_vocab(vocab);
    codec_audio_lm_completion c{};
    {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_tts_build(alm, &req, main_ctx->params.prompt.c_str(), &cv, &c)) {
            throw std::runtime_error(alm_error(alm, "TTS prompt"));
        }
        // codec.cpp's own noise follows the completion's seed (the one its sampler drew).
        if (!codec_audio_lm_set_seed(alm, common_sampler_get_seed(main_ctx->completion->ctx_sampling))) {
            throw std::runtime_error(alm_error(alm, "TTS seed"));
        }
    }
    if (c.flow == CODEC_AUDIO_LM_FLOW_SELF_CONTAINED) {
        throw std::runtime_error("TTS: this model runs without a backbone; llama.rn does not host it");
    }
    if ((c.needs_embeddings || c.needs_prompt_hidden) && !main_ctx->params.embedding) {
        throw std::runtime_error("TTS: this model reads hidden states; pass embedding: true "
                                 "(getFormattedAudioCompletion's completionParams)");
    }
    llama_rn_tts_prompt out;
    out.n_seq = std::max(1, c.n_seq);
    if (out.n_seq > 1 && c.sample_first) {
        throw std::runtime_error("TTS: backbone sampling over several sequences is not supported");
    }
    bool exempt = c.n_penalty_exempt > 0;
    for (int32_t g = 0; g < c.n_head_groups; ++g) exempt = exempt || c.head_groups[g].n_penalty_exempt > 0;
    check_exempt(exempt ? 1 : 0);

    // Chunks: sequence q of chunk i is c.chunks[q * n_chunks + i].
    for (int32_t i = 0; i < c.n_chunks; ++i) {
        const codec_audio_lm_chunk & ch = c.chunks[i];
        llama_rn_tts_prompt::chunk o;
        if (ch.type == CODEC_AUDIO_LM_CHUNK_EMBD) {
            if (ch.n <= 0) continue;
            if (c.needs_prompt_hidden) throw std::runtime_error("TTS: prompt hidden states over embedding chunks are not supported");
            o.n = ch.n;
            o.n_embd = c.n_embd;
            for (int32_t q = 0; q < out.n_seq; ++q) {
                const codec_audio_lm_chunk & cq = c.chunks[q * c.n_chunks + i];
                if (cq.type != CODEC_AUDIO_LM_CHUNK_EMBD || cq.n != ch.n) {
                    throw std::runtime_error("TTS: the sequences' prompt chunks differ");
                }
                o.embd.insert(o.embd.end(), cq.embd, cq.embd + (size_t) cq.n * c.n_embd);
            }
        } else {
            if (out.n_seq > 1) throw std::runtime_error("TTS: several sequences take embedding prompts only");
            if (ch.type == CODEC_AUDIO_LM_CHUNK_TEXT) {
                if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
                    throw std::runtime_error("TTS: the backbone has no vocabulary to tokenize the prompt");
                }
                o.tokens = tokenize(vocab, ch.text ? ch.text : "", ch.add_special, ch.parse_special);
            } else {
                o.tokens.assign(ch.tokens, ch.tokens + ch.n);
            }
            o.n = (int32_t) o.tokens.size();
            if (o.n == 0) continue;
        }
        out.chunks.push_back(std::move(o));
    }
    out.needs_prompt_hidden = c.needs_prompt_hidden;
    for (int32_t g = 0; g < c.n_head_groups; ++g) {
        const codec_audio_lm_head_group & hg = c.head_groups[g];
        out.head_groups.push_back({hg.penalties, std::vector<llama_token>(hg.history, hg.history + std::max(0, hg.n_history))});
    }

    n_seq = out.n_seq;
    n_embd = c.n_embd;
    needs_hidden = c.needs_embeddings;
    sample_backbone = c.sample_first;
    pending_next_embd.clear();
    stop_reason.clear();
    n_frames = 0;
    has_generation = true;
    return out;
}

void llama_rn_context_tts::prefill(llama_rn_context* main_ctx, const float * hiddens, int n_pos) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    codec_affinity_scope affinity(main_ctx);
    if (!codec_audio_lm_prefill(alm, hiddens, n_pos)) throw std::runtime_error(alm_error(alm, "TTS prefill"));
}

llama_rn_tts_step llama_rn_context_tts::step(llama_rn_context* main_ctx, llama_token sampled,
                                             const std::function<llama_token(int, const float *, int)> &sample_head) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<float> hiddens;
    if (needs_hidden) {
        // Sequence q's last output is -(n_seq - q) from the end.
        const int32_t dim = llama_model_n_embd(main_ctx->model);
        hiddens.resize((size_t) n_seq * dim);
        for (int32_t q = 0; q < n_seq; ++q) {
            const float * hq = llama_get_embeddings_ith(main_ctx->ctx, -(n_seq - q));
            if (hq == nullptr) throw std::runtime_error("TTS: backbone returned no hidden state");
            std::memcpy(hiddens.data() + (size_t) q * dim, hq, (size_t) dim * sizeof(float));
        }
    }
    codec_audio_lm_verdict v{};
    llama_rn_tts_step r;
    {
        codec_affinity_scope affinity(main_ctx);
        if (!codec_audio_lm_observe(alm, sampled >= 0 ? &sampled : nullptr, needs_hidden ? hiddens.data() : nullptr, &v)) {
            throw std::runtime_error(alm_error(alm, "TTS step"));
        }
        // codec.cpp's codebook heads: the completion draws each token.
        while (v.action == CODEC_AUDIO_LM_SAMPLE_HEAD) {
            if (v.frame_done) ++n_frames;
            const llama_token id = sample_head(v.head_group, codec_audio_lm_head_logits(alm), v.n_vocab);
            if (!codec_audio_lm_head_token(alm, id, &v)) throw std::runtime_error(alm_error(alm, "TTS head"));
        }
        if (v.action == CODEC_AUDIO_LM_FEED_EMBD) {
            pending_next_embd.resize((size_t) n_seq * n_embd);
            for (int32_t q = 0; q < n_seq; ++q) {
                std::memcpy(pending_next_embd.data() + (size_t) q * n_embd, codec_audio_lm_next_embd(alm, q),
                            (size_t) n_embd * sizeof(float));
            }
        }
    }
    if (v.frame_done) ++n_frames;
    r.audio = v.is_audio;
    if (v.action == CODEC_AUDIO_LM_STOP) {
        stop_reason = codec_audio_lm_stop_reason(alm);
        r.stop = true;
        return r;
    }
    if (v.action == CODEC_AUDIO_LM_FEED_TOKEN) {
        if (n_seq > 1) throw std::runtime_error("TTS: token feeds over several sequences are not supported");
        r.feed = v.token >= 0 ? v.token : sampled;
        if (r.feed < 0) throw std::runtime_error("TTS: the model asked to feed a token none was sampled for");
    }
    // codec.cpp would read n_seq sampled tokens; nextToken samples one.
    if (v.sample_backbone && n_seq > 1) {
        throw std::runtime_error("TTS: backbone sampling over several sequences is not supported");
    }
    sample_backbone = v.sample_backbone;
    return r;
}

void llama_rn_context_tts::collectGeneration() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    audio_tokens.clear();
    audio_latents.clear();
    audio_latent_dim = 0;
    if (!has_generation) return;
    codec_audio_lm_generation gen{};
    if (!codec_audio_lm_get_generation(alm, &gen)) throw std::runtime_error(alm_error(alm, "TTS generation"));
    if (gen.n_frames <= 0) return;
    const size_t n = (size_t) gen.n_frames * (size_t) gen.width;
    if (gen.kind == CODEC_AUDIO_LM_DATA_LATENTS) {
        audio_latent_dim = gen.width;
        audio_latents.assign(gen.latents, gen.latents + n);
    } else {
        code_width = gen.width;
        audio_tokens.assign(gen.codes, gen.codes + n);
    }
}

llama_rn_context_tts::~llama_rn_context_tts() {
    speakers.clear();
    gen_voice.reset();
    if (alm != nullptr) codec_audio_lm_free(alm);
}

// ── Speakers ──────────────────────────────────────────────────────────

int llama_rn_context_tts::createSpeaker(llama_rn_context* main_ctx, const std::vector<float> &pcm, int sample_rate,
                                        const std::string &ref_text, float emotion, bool has_emotion, bool bake) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    (void) emotion;
    (void) has_emotion;
    (void) bake;
    if (pcm.empty() || sample_rate <= 0) throw std::runtime_error("createSpeaker: empty reference audio");
    codec_speaker * voice = nullptr;
    {
        codec_affinity_scope affinity(main_ctx);
        voice = codec_audio_lm_create_voice(alm, pcm.data(), (int32_t) pcm.size(), sample_rate,
                                            ref_text.empty() ? nullptr : ref_text.c_str());
    }
    if (voice == nullptr) throw std::runtime_error(alm_error(alm, "createSpeaker"));
    rn_speaker spk;
    spk.voice = rn_voice(voice, [](const codec_speaker * p) { codec_speaker_free(const_cast<codec_speaker *>(p)); });
    const int id = next_speaker_id++;
    speakers[id] = std::move(spk);
    return id;
}

bool llama_rn_context_tts::bakeSpeaker(llama_rn_context* main_ctx, int id) {
    (void) main_ctx;
    return getSpeaker(id) != nullptr;
}

const rn_speaker* llama_rn_context_tts::getSpeaker(int id) const {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const auto it = speakers.find(id);
    return it == speakers.end() ? nullptr : &it->second;
}

void llama_rn_context_tts::releaseSpeaker(int id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    speakers.erase(id);
}

int llama_rn_context_tts::loadSpeaker(const std::string &path, const std::string &name) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    codec_speaker * voice = codec_audio_lm_load_voice(alm, path.c_str(), name.empty() ? nullptr : name.c_str());
    if (voice == nullptr) throw std::runtime_error(alm_error(alm, "loadSpeaker"));
    rn_speaker spk;
    spk.voice = rn_voice(voice, [](const codec_speaker * p) { codec_speaker_free(const_cast<codec_speaker *>(p)); });
    const int id = next_speaker_id++;
    speakers[id] = std::move(spk);
    return id;
}

void llama_rn_context_tts::saveSpeaker(int id, const std::string &path, const std::string &name) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const rn_speaker * spk = getSpeaker(id);
    if (spk == nullptr) throw std::runtime_error("saveSpeaker: speaker " + std::to_string(id) + " not found");
    if (!codec_audio_lm_save_voice(alm, spk->voice.get(), path.c_str(), name.empty() ? nullptr : name.c_str())) {
        throw std::runtime_error(alm_error(alm, "saveSpeaker"));
    }
}

std::vector<llama_rn_speaker_info> llama_rn_context_tts::listSpeakers(const std::string &path) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
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
    if (!codec_audio_lm_decode_generation(alm, &gen, gen_voice.get(), &buf)) throw std::runtime_error(alm_error(alm, what));
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
    std::lock_guard<std::recursive_mutex> lock(mutex);
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
    std::lock_guard<std::recursive_mutex> lock(mutex);
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
    std::lock_guard<std::recursive_mutex> lock(mutex);
    codec_audio_lm_caps c{};
    codec_audio_lm_get_caps(alm, &c);
    return c.sample_rate;
}

}
