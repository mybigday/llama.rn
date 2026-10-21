#include "audio_lm_family.h"
#include "lm/lm_models.h"
#include "kitten_text.h"
#include "tts_segment.h"

#include <algorithm>
#include <cstring>
#include <random>

namespace codec_common {

// ── alm_vocab ────────────────────────────────────────────────────────

std::vector<int32_t> alm_vocab::tokenize(const std::string & text, bool add_special, bool parse_special) const {
    if (!valid()) return {};
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = v->tokenize(v->user_data, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                            add_special, parse_special);
    if (n < 0) {
        out.resize((size_t) -n);
        n = v->tokenize(v->user_data, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                        add_special, parse_special);
    }
    out.resize((size_t) std::max(0, n));
    return out;
}

std::string alm_vocab::piece(int32_t token) const {
    if (!valid()) return {};
    char buf[128];
    int32_t n = v->token_to_piece(v->user_data, token, buf, (int32_t) sizeof(buf));
    if (n >= 0) return std::string(buf, (size_t) std::min<int32_t>(n, (int32_t) sizeof(buf)));
    std::string big((size_t) -n, '\0');
    n = v->token_to_piece(v->user_data, token, big.data(), (int32_t) big.size());
    big.resize((size_t) std::max(0, n));
    return big;
}

int32_t alm_vocab::find_piece(const std::string & text) const {
    for (int32_t id = 0; id < n_tokens(); ++id) {
        if (piece(id) == text) return id;
    }
    return -1;
}

// ── chunks / completion ─────────────────────────────────────────────

alm_chunk alm_chunk::of_text(std::string t, bool add_special, bool parse_special) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_TEXT;
    c.text = std::move(t);
    c.add_special = add_special;
    c.parse_special = parse_special;
    return c;
}

alm_chunk alm_chunk::of_tokens(std::vector<int32_t> t) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_TOKENS;
    c.tokens = std::move(t);
    return c;
}

alm_chunk alm_chunk::of_embd(std::vector<float> e, int32_t n_rows) {
    alm_chunk c;
    c.type = CODEC_AUDIO_LM_CHUNK_EMBD;
    c.embd = std::move(e);
    c.n_rows = n_rows;
    return c;
}

std::string alm_codec_marker(const std::string & name) { return "<__codec:" + name + "__>"; }

bool alm_prompt::parse(const std::string & prompt, std::string * err) {
    pieces_.clear();
    pos_ = 0;
    static const std::string voice = CODEC_AUDIO_LM_MARKER_VOICE, codec = "<__codec:";
    size_t at = 0;
    while (at < prompt.size()) {
        const size_t v = prompt.find(voice, at), c = prompt.find(codec, at);
        const size_t m = std::min(v, c);
        if (m > at) pieces_.push_back({false, prompt.substr(at, m == std::string::npos ? std::string::npos : m - at)});
        if (m == std::string::npos) break;
        if (m == v) {
            pieces_.push_back({true, "voice"});
            at = m + voice.size();
        } else {
            const size_t e = prompt.find("__>", m + codec.size());
            if (e == std::string::npos || e == m + codec.size()) {
                if (err) *err = "malformed <__codec:NAME__> marker in the prompt";
                return false;
            }
            pieces_.push_back({true, "codec:" + prompt.substr(m + codec.size(), e - m - codec.size())});
            at = e + 3;
        }
    }
    return true;
}

std::string alm_prompt::text() {
    if (done() || pieces_[pos_].marker) return "";
    return pieces_[pos_++].s;
}

bool alm_prompt::voice() {
    if (next_marker() != "voice") return false;
    ++pos_;
    return true;
}

bool alm_prompt::codec(const char * name) {
    if (next_marker() != std::string("codec:") + name) return false;
    ++pos_;
    return true;
}

std::string alm_prompt::next_marker() const {
    return !done() && pieces_[pos_].marker ? pieces_[pos_].s : "";
}

bool alm_prompt::rest_text(std::string * out) {
    out->clear();
    for (; !done(); ++pos_) {
        if (pieces_[pos_].marker) return false;
        *out += pieces_[pos_].s;
    }
    return true;
}

uint32_t alm_resolve_seed(uint32_t seed) {
    return seed == CODEC_AUDIO_LM_DEFAULT_SEED ? std::random_device{}() : seed;
}

std::string alm_min_length_grammar(int32_t end, int32_t min_tokens, const std::string & body) {
    if (min_tokens <= 0 || end < 0) return "";
    const std::string e = "<[" + std::to_string(end) + "]>";
    const std::string tok = body.empty() ? "!" + e : body;
    // llama.cpp's grammar never matches an end-of-generation token as a
    // terminal: it allows one once the grammar can end.  The optional end
    // covers both an EOG end token and a plain one.
    return "root ::= tok{" + std::to_string(min_tokens) + "} tok* " + e + "?\ntok ::= " + tok + "\n";
}

// ── alm_family defaults ─────────────────────────────────────────────

bool alm_family::fail_alm(const char * what) {
    const char * e = audio_lm_last_error(alm_);
    return fail(std::string(what) + (e && *e ? std::string(": ") + e : std::string()));
}

void alm_family::caps(codec_audio_lm_caps * out) const {
    *out = codec_audio_lm_caps{};
    out->modality       = audio_lm_modality(alm_);
    out->needs_backbone = true;
    out->takes_ref_audio = audio_lm_codec_can_speaker_from_audio(alm_);
    out->segment_bytes   = segment_split().max_bytes;
}

bool alm_family::backbone_requirements(const alm_vocab &, int32_t max_frames, codec_audio_lm_backbone_req * out) {
    out->n_ctx      = std::max(4096, (max_frames > 0 ? max_frames : 512) + 512);
    out->n_ubatch   = 0;
    out->n_seq_max  = 1;
    out->embeddings = true;
    out->flash_attn = true;
    return true;
}

codec_speaker * alm_family::create_voice(const float * pcm, int32_t n_samples, int32_t sample_rate,
                                        const char * ref_text) {
    if (!audio_lm_codec_can_speaker_from_audio(alm_)) {
        fail("this model cannot clone a voice from reference audio");
        return nullptr;
    }
    codec_speaker * spk = audio_lm_speaker_from_audio(alm_, pcm, n_samples, sample_rate, ref_text);
    if (spk == nullptr) fail_alm("voice");
    return spk;
}

std::vector<std::string> alm_family::split(const std::string & text, int32_t max_bytes) {
    split_params p = segment_split();
    if (max_bytes > 0) p.max_bytes = max_bytes;
    switch (p.mode) {
        case split_mode::sentence_pack:
            return kitten_split_for_synthesis(text, p.max_bytes, p.min_bytes);
        case split_mode::clause:
        default:
            return tts_split_text(text, p.max_bytes);
    }
}

int32_t alm_family::join(std::vector<float> & pcm, int32_t sample_rate, int32_t i, int32_t n) {
    if (n <= 1) return 0;
    const join_params p = segment_join();
    pcm = tts_shape_segment(pcm, sample_rate, p.trim, p.fade_ms);
    return i + 1 < n ? (int32_t) (sample_rate * (p.gap_ms / 1000.0)) : 0;
}

bool alm_family::prefill(const float *, int32_t) { return fail("this model takes no prompt hidden states"); }

const float * alm_family::next_embd(int32_t) const { return nullptr; }

bool alm_family::generate_self_contained() { return fail("this model needs a backbone"); }

bool alm_family::generation(alm_generation * out) {
    out->latent = audio_lm_is_continuous(alm_);
    return audio_lm_generation_input(alm_, &out->codes, &out->latents, &out->n_frames, &out->width) ||
           fail_alm("decode");
}

bool alm_family::decode_generation(const alm_generation & g, audio_lm_audio_output * out) {
    if (g.n_frames <= 0 || g.width <= 0) return fail("empty generation");
    const bool ok = g.latent
        ? audio_lm_decode_latents_frame_major(alm_, g.latents.data(), g.n_frames, g.width, out)
        : audio_lm_decode_codec_codes(alm_, g.codes.data(), g.n_frames, g.width, out);
    return ok || fail_alm("decode");
}

// Codes of a family that keeps its own (single-codebook) token list.
bool alm_family::generation_from_codes(const std::vector<int32_t> & codes, alm_generation * out) {
    out->latent = false;
    if (codes.empty()) return fail("no speech codes generated");
    if (!audio_lm_codec_codes(alm_, codes.data(), (int32_t) codes.size(), 1, 0, -1, &out->codes, &out->width)) {
        return fail_alm("decode");
    }
    out->n_frames = (int32_t) (out->codes.size() / (size_t) out->width);
    return true;
}

// ── detection ────────────────────────────────────────────────────────

static std::unique_ptr<alm_family> make_family(audio_lm_context * alm, std::string * err);

std::unique_ptr<alm_family> alm_make_family(audio_lm_context * alm, const std::string & backbone_path,
                                            std::string * err) {
    auto f = make_family(alm, err);
    if (f) f->backbone_path = backbone_path;
    return f;
}

static std::unique_ptr<alm_family> make_family(audio_lm_context * alm, std::string * err) {
    const char * arch = audio_lm_meta_str(alm, "general.architecture");
    const std::string a = arch ? arch : "";
    if (a == "soprano") return alm_family_soprano(alm);
    if (alm_is_kitten(alm)) return alm_family_kitten(alm);
    if (codec_lm * lm = audio_lm_get_lm(alm)) {
        if (codec_lm_flow_get_info(lm) != nullptr) return alm_family_flow(alm, lm);
        if (codec_lm_chatterbox_get_info(lm) != nullptr) return alm_family_chatterbox(alm, lm);
        if (codec_lm_cosyvoice_get_info(lm) != nullptr) return alm_family_cosyvoice(alm, lm);
        audio_lm_prompt_info pi;
        if (codec_lm_chatterbox_get_info(lm) == nullptr && codec_lm_cosyvoice_get_info(lm) == nullptr &&
            audio_lm_get_prompt_info(alm, &pi)) {
            if (pi.is_continuous)         return alm_family_continuous(alm);
            if (pi.streaming_interleave)  return alm_family_realtime(alm);
            if (pi.sequential_text_audio) return alm_family_lfm2(alm);
            // Moshi's codebook 0 is its text stream; CSM's is audio.
            const char * c0mod = audio_lm_meta_str(alm, "codec.lm.residual.c0_input_modality");
            if (pi.model_kind == audio_lm_prompt_info::KIND_RESIDUAL_DEPTH_AR && c0mod &&
                std::strcmp(c0mod, "text") == 0) {
                if (err) *err = "Moshi (full-duplex dialogue, no audio end) is not a one-shot TTS model";
                return nullptr;
            }
            if (pi.model_kind == audio_lm_prompt_info::KIND_RESIDUAL_DEPTH_AR ||
                pi.model_kind == audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY) {
                return alm_family_codebook(alm);
            }
        }
    }
    if (audio_lm_get_lm(alm) == nullptr) {
        // Codec-only GGUFs whose model writes codec tokens with its own vocab.
        if (a == "neucodec" || a == "distill_neucodec") return alm_family_neutts(alm);
        const codec_speaker * spk = audio_lm_decode_speaker(alm);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        if (spk != nullptr && codec_speaker_get_slot(spk, "audio_prompt", &ty, ne, &data)) return alm_family_outetts(alm);
    }
    if (err) *err = std::string("no audio-LM recipe for this codec GGUF (architecture ") + (arch ? arch : "?") + ")";
    return nullptr;
}

}  // namespace codec_common
