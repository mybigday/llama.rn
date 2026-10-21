// KittenTTS 2: a Qwen3 speech LM writes S3 tokens as ordinary vocab rows,
// decoded by Chatterbox-Turbo's MeanFlow S3Gen.  Port of kittenml/kittentts2
// (model.py, prompt.py, logits.py, text.py).  The codec GGUF is a
// chatterbox_s3g model with the LM contract under codec.lm.kitten.* and the
// voice in its speakers: lm_cond (position-0 embedding row), ref_tokens +
// ref_text (in-context reference) and the S3G decoder slots.

#include "audio_lm_family.h"
#include "kitten_text.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace codec_common {
namespace {

struct kitten_config {
    int32_t audio_base = 0, n_audio = 0;
    int32_t speech_start = 0, speech_end = 0, text_start = 0, start = 0, stop = 0, final_seg = 0;
    int32_t ref_speech_start = 0, ref_speech_end = 0, ref_text_start = 0, ref_text_end = 0;
    int32_t silence = 4299, n_trailing_silence = 3;
    bool use_reference_prompt = true;
    std::string emotion_control = "{emo: 1}";
    int32_t chunk_chars = 380, chunk_min_chars = 130, max_new_tokens = 1000;
    float chunk_gap_s = 0.16f;
};

void kitten_read_config(const audio_lm_context * ctx, kitten_config * c) {
    auto str = [&](const std::string & k) { return audio_lm_meta_str(ctx, ("codec.lm.kitten." + k).c_str()); };
    auto i32 = [&](const std::string & k, int32_t * dst) { if (const char * v = str(k)) *dst = std::atoi(v); };
    auto f32 = [&](const std::string & k, float * dst) { if (const char * v = str(k)) *dst = (float) std::atof(v); };
    i32("token.audio_id_base", &c->audio_base);           i32("token.num_audio_tokens", &c->n_audio);
    i32("token.speech_start_id", &c->speech_start);       i32("token.speech_end_id", &c->speech_end);
    i32("token.text_start_id", &c->text_start);           i32("token.start_id", &c->start);
    i32("token.stop_id", &c->stop);                       i32("token.final_seg_id", &c->final_seg);
    i32("token.reference_speech_start_id", &c->ref_speech_start);
    i32("token.reference_speech_end_id", &c->ref_speech_end);
    i32("token.reference_text_start_id", &c->ref_text_start);
    i32("token.reference_text_end_id", &c->ref_text_end);
    i32("silence_token", &c->silence);                    i32("n_trailing_silence", &c->n_trailing_silence);
    if (const char * v = str("use_reference_prompt")) c->use_reference_prompt = std::string(v) == "true";
    if (const char * v = str("emotion_control")) c->emotion_control = v;
    i32("chunk_chars", &c->chunk_chars);                  i32("chunk_min_chars", &c->chunk_min_chars);
    f32("chunk_gap_s", &c->chunk_gap_s);                  i32("max_new_tokens", &c->max_new_tokens);
}

std::string strip_copy(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

class kitten_family : public alm_family {
public:
    explicit kitten_family(audio_lm_context * alm) : alm_family(alm) { kitten_read_config(alm, &cfg_); }
    const char * name() const override { return "kitten"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_text = true;
        out->sample_rate    = 24000;
    }

    // Text > chunk_chars is split on sentences (min chunk_min_chars), each
    // chunk generated with a fresh context (text.py split_for_synthesis);
    // joined the general way (trimmed edges, 8 ms fades, chunk_gap_s apart).
    split_params segment_split() const override {
        return {split_mode::sentence_pack, cfg_.chunk_chars, cfg_.chunk_min_chars};
    }
    join_params segment_join() const override {
        return {true, 8, (int32_t) std::lround(cfg_.chunk_gap_s * 1000.0f)};
    }

    std::vector<std::string> split(const std::string & text, int32_t max_bytes) override {
        full_has_tag_ = kitten_has_expression_tag(text);
        segments_ = alm_family::split(kitten_prepare_text(text), max_bytes);
        n_segments_ = (int32_t) segments_.size();
        return segments_;
    }

    // Prompt (prompt.py; the four reference delimiters have no vocab piece):
    //   <__voice__> (position 0: the speaker's lm_cond row)
    //   [<__codec:ref_text_start__>{transcript}<__codec:ref_text_end__>
    //    <__codec:ref_speech_start__><|a_N|>...<__codec:ref_speech_end__>]
    //   [{emo: 1}]<|text_start|>{text}[<final_seg>]<|speech_start|>
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("KittenTTS needs the backbone vocabulary");
        const kitten_config & c = cfg_;
        const codec_speaker * spk = req.voice;
        int64_t ne[4] = {0};
        codec_speaker_slot_type ty;
        const void * data = nullptr;
        if (spk == nullptr || !codec_speaker_get_slot(spk, "lm_cond", &ty, ne, &data) || ty != CODEC_SPEAKER_SLOT_F32) {
            return fail("KittenTTS speaker has no lm_cond row");
        }
        const int32_t n_embd = (int32_t) ne[0];
        std::string prompt = CODEC_AUDIO_LM_MARKER_VOICE;
        // In-context reference: needs the transcript; without it only the
        // speaker embedding identifies the voice (prompt.py).
        const char * ref_text = codec_speaker_ref_text(spk);
        if (c.use_reference_prompt && ref_text != nullptr &&
            codec_speaker_get_slot(spk, "ref_tokens", &ty, ne, &data) && ty == CODEC_SPEAKER_SLOT_I32) {
            const auto * rt = static_cast<const int32_t *>(data);
            prompt += alm_codec_marker("ref_text_start") + strip_copy(ref_text) + alm_codec_marker("ref_text_end") +
                      alm_codec_marker("ref_speech_start");
            for (int64_t i = 0; i < ne[0]; ++i) prompt += vocab.piece(c.audio_base + rt[i]);
            prompt += alm_codec_marker("ref_speech_end");
        }
        const std::string spoken = strip_copy(kitten_prepare_text(req.text));
        if (spoken.empty() && req.text_known) return fail("no text to speak");
        // A segment of the last split takes the whole text's settings; any
        // other text stands alone.
        const bool of_split = std::find(segments_.begin(), segments_.end(), req.text) != segments_.end();
        if (of_split ? full_has_tag_ : kitten_has_expression_tag(req.text)) prompt += c.emotion_control;
        prompt += vocab.piece(c.text_start) + spoken;
        if (c.final_seg) prompt += vocab.piece(c.final_seg);
        prompt += vocab.piece(c.speech_start);
        out->prompt = std::move(prompt);
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_TOKENS;
        d.n_embd = n_embd;
        // The penalties should never touch the end and silence tokens (logits.py).
        d.penalty_exempt = {c.speech_end, c.stop, c.audio_base + c.silence};
        // Speech is audio tokens until <speech_end> / [STOP]: keep a quantized
        // backbone from drifting into the Qwen3 text vocabulary.
        d.allowed = {{c.audio_base, c.audio_base + c.n_audio}, {c.speech_end, c.speech_end + 1}, {c.stop, c.stop + 1}};
        d.max_frames = budget(spoken);
        return true;
    }

    // Per-chunk budget sized from the chunk's length (model.py::_chunk_budget)
    // when the text is one segment of the last split.
    int32_t budget(const std::string & spoken) const {
        bool of_split = false;
        for (const std::string & s : segments_) of_split = of_split || strip_copy(kitten_prepare_text(s)) == spoken;
        if (!of_split || n_segments_ <= 1) return cfg_.max_new_tokens;
        const int32_t expected = (int32_t) ((double) spoken.size() / 20.0 * 25.0 * 1.8);
        return std::max(200, std::min(cfg_.max_new_tokens, expected));
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        const kitten_config & c = cfg_;
        codes_.clear();
        stop_reason_ = "max_frames";
        cfg_run_ = c;
        const codec_speaker * spk = req.voice;
        int64_t ne[4] = {0};
        codec_speaker_slot_type ty;
        const void * data = nullptr;
        if (!prompt.voice()) return fail("KittenTTS's prompt starts with <__voice__>");
        if (spk == nullptr || !codec_speaker_get_slot(spk, "lm_cond", &ty, ne, &data) || ty != CODEC_SPEAKER_SLOT_F32) {
            return fail("KittenTTS speaker has no lm_cond row");
        }
        const auto * lm_cond = static_cast<const float *>(data);
        const std::pair<const char *, int32_t> rows[] = {{"ref_text_start", c.ref_text_start}, {"ref_text_end", c.ref_text_end},
                                                         {"ref_speech_start", c.ref_speech_start}, {"ref_speech_end", c.ref_speech_end}};
        std::vector<int32_t> toks;   // after position 0
        std::string last_text;
        while (!prompt.done()) {
            const std::string t = prompt.text();
            if (!t.empty()) {
                last_text = t;
                const auto ids = vocab.tokenize(t, false, true);
                toks.insert(toks.end(), ids.begin(), ids.end());
                continue;
            }
            bool known = false;
            for (const auto & r : rows) {
                if (prompt.codec(r.first)) { toks.push_back(r.second); known = true; break; }
            }
            if (!known) break;   // the caller reports the unexpected marker
        }
        // The spoken text: between <|text_start|> and <final_seg> / <|speech_start|>.
        const std::string ts = vocab.piece(c.text_start), te = vocab.piece(c.final_seg ? c.final_seg : c.speech_start);
        const size_t a = last_text.rfind(ts);
        const size_t e = a == std::string::npos ? std::string::npos : last_text.find(te, a + ts.size());
        out->max_frames = budget(e == std::string::npos ? "" : last_text.substr(a + ts.size(), e - a - ts.size()));
        codec_log(CODEC_LOG_LEVEL_INFO, "kitten: voice=%s\n", codec_speaker_name(spk));
        out->seqs = {{alm_chunk::of_embd(std::vector<float>(lm_cond, lm_cond + out->n_embd), 1),
                      alm_chunk::of_tokens(std::move(toks))}};
        return true;
    }

    bool observe(const int32_t * tokens, const float *, codec_audio_lm_verdict * v) override {
        if (tokens == nullptr) return fail("observe needs the sampled token");
        const int32_t t = tokens[0];
        if (t == cfg_run_.speech_end || t == cfg_run_.stop) {
            stop_reason_ = "eos";
            codec_log(CODEC_LOG_LEVEL_INFO, "KittenTTS AR done: %zu frames, stop=eos\n", codes_.size());
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1};
            return true;
        }
        codes_.push_back(t - cfg_run_.audio_base);
        *v = {CODEC_AUDIO_LM_FEED_TOKEN, true, true, true, -1};
        return true;
    }

    bool generation(alm_generation * out) override {
        if (codes_.empty()) return fail("the model produced no audio tokens");
        // S3Gen drops its last tokens as lookahead: pad with silence (vocoder.py).
        std::vector<int32_t> codes = codes_;
        codes.insert(codes.end(), (size_t) cfg_run_.n_trailing_silence, cfg_run_.silence);
        return generation_from_codes(codes, out);
    }

private:
    kitten_config            cfg_;       // the model's contract (codec.lm.kitten.*)
    kitten_config            cfg_run_;   // the current request's
    std::vector<int32_t>     codes_;
    std::vector<std::string> segments_;
    int32_t                  n_segments_   = 0;
    bool                     full_has_tag_ = false;
};

}  // namespace

bool alm_is_kitten(audio_lm_context * alm) { return audio_lm_meta_str(alm, "codec.lm.kitten.token.audio_id_base") != nullptr; }

std::unique_ptr<alm_family> alm_family_kitten(audio_lm_context * alm) { return std::make_unique<kitten_family>(alm); }

}  // namespace codec_common
