// Token families: the backbone's own vocabulary carries the audio codes
// (codec-only GGUFs, no codec_lm).  NeuTTS writes <|speech_N|> NeuCodec
// tokens after an in-context reference; OuteTTS interleaves word blocks with
// its codec tokens under a prompt-derived grammar.  observe_token packs the
// codes into frames; decode is the codec's.

#include "audio_lm_family.h"

#include <climits>
#include <cstring>

namespace codec_common {
namespace {

class token_family : public alm_family {
public:
    using alm_family::alm_family;

    bool observe(const int32_t * tokens, const float *, codec_audio_lm_verdict * v) override {
        if (tokens == nullptr) return fail("observe needs the sampled token");
        const int32_t before = audio_lm_codes_n_frames(alm_);
        observe_action act = audio_lm_observe_token(alm_, tokens[0], nullptr, 0);
        // End of generation without the audio-end token (e.g. a model that
        // spelled it out of text pieces): nothing more will follow.
        if (act != OBSERVE_STOP && is_eog_(tokens[0])) act = OBSERVE_STOP;
        if (act == OBSERVE_STOP) {
            stop_reason_ = "eos";
            codec_log(CODEC_LOG_LEVEL_INFO, "%s AR done: %d frames, stop=eos\n", log_name(), audio_lm_codes_n_frames(alm_));
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        if (act == OBSERVE_CONSUMED_EMBED) return fail("unexpected embedding feedback for a token model");
        *v = {CODEC_AUDIO_LM_FEED_TOKEN, true, audio_lm_codes_n_frames(alm_) > before, -1};
        return true;
    }

protected:
    virtual const char * log_name() const = 0;

    void begin(const alm_vocab & vocab) {
        stop_reason_ = "max_frames";
        eog_vocab_ = *vocab.v;
    }
    bool is_eog_(int32_t t) const { return alm_vocab{&eog_vocab_}.is_eog(t); }

    codec_audio_lm_vocab eog_vocab_{};
};

// ── NeuTTS ───────────────────────────────────────────────────────────

bool neutts_range(const alm_vocab & vocab, int32_t * off, int32_t * count, int32_t * eos) {
    *off = vocab.find_piece("<|speech_0|>");
    *eos = vocab.find_piece("<|SPEECH_GENERATION_END|>");
    if (*off < 0 || *eos < 0) return false;
    // Count contiguous <|speech_N|> tokens from N = 0 upward.
    int32_t n = 0;
    while (*off + n < vocab.n_tokens() && vocab.piece(*off + n) == "<|speech_" + std::to_string(n) + "|>") ++n;
    *count = n;
    return true;
}

class neutts_family : public token_family {
public:
    using token_family::token_family;
    const char * name() const override { return "neutts"; }

    void caps(codec_audio_lm_caps * out) const override {
        token_family::caps(out);
        out->text_needs_phonemes = true;
        out->takes_ref_text      = true;
        out->needs_voice         = true;
        out->sample_rate         = 24000;
    }

    codec_speaker * create_voice(const float * pcm, int32_t n_samples, int32_t sample_rate,
                                 const char * ref_text) override {
        if (!audio_lm_codec_can_speaker_from_audio(alm_)) {
            fail("NeuTTS: reference audio needs a codec GGUF with the NeuCodec distill encoder "
                 "(this one cannot encode); use a voice / speaker file or a distill_neucodec codec");
            return nullptr;
        }
        return token_family::create_voice(pcm, n_samples, sample_rate, ref_text);
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        const std::string & text = req.text;
        if (!vocab.valid()) return fail("NeuTTS needs the backbone vocabulary");
        if (vocab.find_piece("<|TEXT_PROMPT_START|>") < 0) return fail("the backbone is not a NeuTTS model");
        begin(vocab);
        const codec_speaker * spk = audio_lm_decode_speaker(alm_);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        if (spk == nullptr || !codec_speaker_get_slot(spk, "ref_codes", &ty, ne, &data) || ty != CODEC_SPEAKER_SLOT_I32) {
            return fail("NeuTTS needs a reference voice: a codec GGUF with builtin voices, a speaker file, "
                        "or reference audio + its transcript with a distill_neucodec codec");
        }
        const char * spk_text = codec_speaker_ref_text(spk);
        const std::string ref_text = spk_text ? spk_text : "";
        if (ref_text.empty()) return fail("NeuTTS: the reference voice has no transcript; pass its phonemes as ref_text");

        int32_t off = -1, count = 0, eos = -1;
        if (!neutts_range(vocab, &off, &count, &eos)) return fail("NeuTTS: <|speech_N|> / end tokens not in the vocab");
        // In-context reference, phoneme mode (neuphonic/neutts).
        std::vector<int32_t> toks = vocab.tokenize(
            "user: Convert the text to speech:<|TEXT_PROMPT_START|>" + ref_text + " " + text +
            "<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>", false, true);
        if (toks.empty()) return fail("NeuTTS: the prompt tokenized to nothing");
        const auto * codes = static_cast<const int32_t *>(data);
        for (int64_t i = 0; i < ne[0]; ++i) {
            if (codes[i] < 0 || codes[i] >= count) return fail("NeuTTS: reference code out of range");
            toks.push_back(off + codes[i]);
        }
        audio_lm_set_audio_token_range(alm_, off, count, eos);
        codec_log(CODEC_LOG_LEVEL_INFO, "[neutts] ICL prompt: %d ref codes, %zu prompt tokens\n", (int) ne[0], toks.size());

        *out = alm_completion();
        out->flow = CODEC_AUDIO_LM_FLOW_TOKENS;
        out->seqs = {{alm_chunk::of_tokens(std::move(toks))}};
        out->sampler.temp  = 1.0f;
        out->sampler.top_k = 50;
        out->allowed = {{off, off + count}, {eos, eos + 1}};
        // Some seeds sample the end token first at temp 1 / top-k 50; keep it
        // out for the first 25 frames (~0.3 s).
        out->min_frames = 25;
        out->stop_token = eos;
        out->max_frames = 512;
        out->streamable = true;
        if (req.min_frames > 0) out->min_frames = req.min_frames;
        req.apply(out->sampler);
        return true;
    }

private:
    const char * log_name() const override { return "NeuTTS"; }
};

// ── OuteTTS ──────────────────────────────────────────────────────────

// V1 = OuteTTS 0.2, V2 = 0.3, V3 = 1.0, recognized from their marker tokens.
bool outetts_detect(const alm_vocab & vocab, outetts_version * out) {
    auto has = [&](const char * p) { return vocab.find_piece(p) >= 0; };
    if (has("<|word_start|>") || has("<|features|>") || has("<|c1_0|>")) { *out = outetts_version::V3; return true; }
    if (has("<|space|>") && has("<|audio_end|>")) { *out = outetts_version::V2; return true; }
    if (has("<|text_sep|>") && has("<|code_start|>") && has("<|audio_end|>")) { *out = outetts_version::V1; return true; }
    return false;
}

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
    bool cjk = false;
    for (uint32_t c : utf8_codepoints(spk)) {
        if ((c >= 0x3040 && c <= 0x30FF) || (c >= 0x4E00 && c <= 0x9FFF)) cjk = true;
    }
    std::string rs;
    if (!spk.empty()) {
        const uint32_t last = utf8_codepoints(spk).back();
        const bool ends = cjk ? (last == 0x3002 || last == 0xFF1F || last == 0xFF01 || last == '?' || last == '!')
                              : (last == '.' || last == '?' || last == '!');
        rs = !ends ? (cjk ? "\xE3\x80\x82" : ". ") : (cjk ? "" : " ");
    }
    return spk + rs + trim_ws(input);
}

// The input as the version's word sequence: 1.0 as is; 0.2 / 0.3 lowercased
// a-z words (0.3 keeps . ! ? , " as punctuation tokens glued to their word).
std::string outetts_input_words(outetts_version v, const std::string & text) {
    if (v == outetts_version::V3) return trim_ws(text);
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
        if (!cur.empty()) words.push_back(before + cur + after);
        else if (!words.empty()) words.back() += before + after;   // standalone punctuation joins the previous word
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

class outetts_family : public token_family {
public:
    using token_family::token_family;
    const char * name() const override { return "outetts"; }

    void caps(codec_audio_lm_caps * out) const override {
        token_family::caps(out);
        out->takes_ref_audio = false;
        out->sample_rate     = codec_model_sample_rate_of();
    }

    bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames, codec_audio_lm_backbone_req * out) override {
        token_family::backbone_requirements(vocab, max_frames, out);
        outetts_version v{};
        if (!vocab.valid() || !outetts_detect(vocab, &v)) return fail("the backbone is not an OuteTTS model");
        // A 1.0 frame is a c1/c2 pair plus its share of the word-block tokens
        // (~2.4 tokens) after a ~1.8k-token speaker prompt; 0.2/0.3 spend ~1.1
        // tokens per frame.  Without a cap: the reference max_length.
        const bool legacy = v != outetts_version::V3;
        out->n_ctx = max_frames > 0 ? std::max(4096, 2048 + (legacy ? 2 : 3) * max_frames) : (legacy ? 4096 : 8192);
        out->n_ubatch = 4096;
        return true;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        const std::string & text = req.text;
        if (!vocab.valid()) return fail("OuteTTS needs the backbone vocabulary");
        outetts_version v{};
        if (!outetts_detect(vocab, &v)) return fail("the backbone is not an OuteTTS model");
        begin(vocab);
        const char * ver = v == outetts_version::V1 ? "0.2" : v == outetts_version::V2 ? "0.3" : "1.0";
        const std::string in_words = outetts_input_words(v, text);
        if (in_words.empty()) return fail("OuteTTS: no words in the input text");

        // Speaker: ref_text (text section) + audio_prompt (this model's token
        // ids for the audio section), baked into the codec GGUF.
        const codec_speaker * spk = audio_lm_decode_speaker(alm_);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        const char * ref_text = spk ? codec_speaker_ref_text(spk) : nullptr;
        if (spk == nullptr || ref_text == nullptr || !codec_speaker_get_slot(spk, "audio_prompt", &ty, ne, &data) ||
            ty != CODEC_SPEAKER_SLOT_I32) {
            return fail("OuteTTS needs a speaker with ref_text + audio_prompt (a codec GGUF with baked voices)");
        }
        auto sep_join = [](const std::string & words, const char * sep) {
            std::string o;
            for (char c : words) o += c == ' ' ? std::string(sep) : std::string(1, c);
            return o;
        };
        std::string body;
        if (v == outetts_version::V1)      body = sep_join(std::string(ref_text) + " " + in_words, "<|text_sep|>");
        else if (v == outetts_version::V2) body = sep_join(std::string(ref_text) + " " + in_words, "<|space|>");
        else                               body = outetts_v3_merge(ref_text, in_words);
        std::vector<int32_t> toks = vocab.tokenize("<|im_start|>\n<|text_start|>" + body + "<|text_end|>\n<|audio_start|>\n",
                                                   false, true);
        const auto * ids = static_cast<const int32_t *>(data);
        for (int64_t i = 0; i < ne[0]; ++i) {
            if (ids[i] < 0 || ids[i] >= vocab.n_tokens()) {
                return fail("OuteTTS: speaker audio_prompt token out of the backbone vocab (codec GGUF baked for another model?)");
            }
            toks.push_back(ids[i]);
        }
        if (v == outetts_version::V3) {
            const std::vector<int32_t> ws = vocab.tokenize("<|word_start|>", false, true);
            toks.insert(toks.end(), ws.begin(), ws.end());
        }

        // Codec token ranges: 1.0 = c1 / c2 with a <|code|> frame sentinel;
        // 0.2 / 0.3 = <|0|>.. (WavTokenizer codebook 4096).
        int32_t offs[2] = {-1, -1}, cnts[2] = {0, 0}, sentinel = -1, eos = -1, n_q = 2;
        if (v == outetts_version::V3) {
            offs[0] = vocab.find_piece("<|c1_0|>");
            offs[1] = vocab.find_piece("<|c2_0|>");
            sentinel = vocab.find_piece("<|code|>");
            eos = vocab.find_piece("<|audio_end|>");
            cnts[0] = cnts[1] = 1024;   // 1024 is a padding code the codec never emits
            if (offs[0] < 0 || offs[1] < 0 || sentinel < 0 || eos < 0) {
                return fail("OuteTTS: <|c1_0|> / <|c2_0|> / <|code|> / <|audio_end|> missing from the vocab");
            }
        } else {
            n_q = 1;
            offs[0] = vocab.find_piece("<|0|>");
            cnts[0] = 4096;
            eos = vocab.find_piece("<|audio_end|>");
            if (offs[0] < 0 || eos < 0 || vocab.piece(offs[0] + 4095) != "<|4095|>") {
                return fail("OuteTTS: <|0|>..<|4095|> / <|audio_end|> not found as a contiguous vocab range");
            }
        }
        audio_lm_set_audio_token_ranges(alm_, offs, cnts, n_q, sentinel, eos);
        codec_log(CODEC_LOG_LEVEL_INFO, "[outetts] %s: %zu prompt tokens\n", ver, toks.size());

        *out = alm_completion();
        out->flow = CODEC_AUDIO_LM_FLOW_TOKENS;
        out->seqs = {{alm_chunk::of_tokens(std::move(toks))}};
        // 1.0 (README): temp 1.0, top-k 50, penalty 1.1 over a 64-token window;
        // 0.2 / 0.3 (GenerationConfig): temp 0.1, top-k 50, penalty 1.1 over everything.
        const bool legacy = v != outetts_version::V3;
        out->sampler.temp           = legacy ? 0.1f : 1.0f;
        out->sampler.top_k          = 50;
        out->sampler.penalty_repeat = 1.1f;
        out->sampler.penalty_last_n = legacy ? 0 : 64;
        // Free generation (word blocks interleave with the codes) kept well
        // formed by the prompt-derived grammar: only the input's words,
        // complete headers, in-range codes.
        out->grammar    = outetts_build_grammar(v, in_words);
        out->max_frames = INT32_MAX;   // the context bounds it
        out->streamable = true;
        req.apply(out->sampler);
        return true;
    }

private:
    const char * log_name() const override { return "OuteTTS"; }
    int32_t codec_model_sample_rate_of() const;
};

int32_t outetts_family::codec_model_sample_rate_of() const {
    const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
    return sr ? std::atoi(sr) : 24000;
}

}  // namespace

std::unique_ptr<alm_family> alm_family_neutts(audio_lm_context * alm) { return std::make_unique<neutts_family>(alm); }
std::unique_ptr<alm_family> alm_family_outetts(audio_lm_context * alm) { return std::make_unique<outetts_family>(alm); }

}  // namespace codec_common
