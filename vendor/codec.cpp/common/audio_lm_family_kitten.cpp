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
    std::string presets;   // ';'-separated
};

bool kitten_read_config(const audio_lm_context * ctx, const std::string & preset_req, kitten_config * c,
                        std::string * err) {
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
    for (const char * p : {"stable", "expressive"}) {
        if (str(std::string("preset.") + p + ".temperature")) c->presets += (c->presets.empty() ? "" : ";") + std::string(p);
    }
    const char * dp = str("default_preset");
    const std::string preset = !preset_req.empty() ? preset_req : (dp ? dp : "stable");
    if (!str("preset." + preset + ".temperature")) {
        if (err) *err = "unknown KittenTTS preset '" + preset + "'";
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

std::string strip_copy(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

class kitten_family : public alm_family {
public:
    explicit kitten_family(audio_lm_context * alm) : alm_family(alm) { kitten_read_config(alm, "", &cfg_, nullptr); }
    const char * name() const override { return "kitten"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_text = true;
        out->sample_rate    = 24000;
        out->presets        = cfg_.presets.c_str();
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

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        if (!vocab.valid()) return fail("KittenTTS needs the backbone vocabulary");
        kitten_config c;
        std::string err;
        if (!kitten_read_config(alm_, req.preset, &c, &err)) return fail(err);
        codes_.clear();
        stop_reason_ = "max_frames";
        cfg_run_ = c;

        const codec_speaker * spk = audio_lm_decode_speaker(alm_);
        int64_t ne[4] = {0};
        codec_speaker_slot_type ty;
        const void * data = nullptr;
        if (spk == nullptr || !codec_speaker_get_slot(spk, "lm_cond", &ty, ne, &data) || ty != CODEC_SPEAKER_SLOT_F32) {
            return fail("KittenTTS speaker has no lm_cond row");
        }
        const int32_t n_embd = (int32_t) ne[0];
        const auto * lm_cond = static_cast<const float *>(data);

        // In-context reference: [START] <ref_text_start> transcript <ref_text_end>
        // <ref_speech_start> audio <ref_speech_end>.  Needs the transcript;
        // without it only the speaker embedding identifies the voice (prompt.py).
        std::vector<int32_t> toks;   // after position 0 ([START] -> lm_cond row)
        const char * ref_text = codec_speaker_ref_text(spk);
        if (c.use_reference_prompt && ref_text != nullptr &&
            codec_speaker_get_slot(spk, "ref_tokens", &ty, ne, &data) && ty == CODEC_SPEAKER_SLOT_I32) {
            const auto * rt = static_cast<const int32_t *>(data);
            const auto text_ids = vocab.tokenize(strip_copy(ref_text), false, false);
            toks.push_back(c.ref_text_start);
            toks.insert(toks.end(), text_ids.begin(), text_ids.end());
            toks.push_back(c.ref_text_end);
            toks.push_back(c.ref_speech_start);
            for (int64_t i = 0; i < ne[0]; ++i) toks.push_back(c.audio_base + rt[i]);
            toks.push_back(c.ref_speech_end);
        } else {
            codec_log(CODEC_LOG_LEVEL_INFO, "kitten: no reference transcript/tokens — speaker embedding only\n");
        }
        const std::string spoken = kitten_prepare_text(req.text);
        if (strip_copy(spoken).empty()) return fail("no text to speak");
        // A segment of the last split takes the whole text's settings; any
        // other text stands alone.
        const bool of_split = std::find(segments_.begin(), segments_.end(), req.text) != segments_.end();
        const int32_t n_segments = of_split ? n_segments_ : 1;
        if (of_split ? full_has_tag_ : kitten_has_expression_tag(req.text)) {
            const auto emo = vocab.tokenize(c.emotion_control, false, false);
            toks.insert(toks.end(), emo.begin(), emo.end());
        }
        toks.push_back(c.text_start);
        const auto text_ids = vocab.tokenize(strip_copy(spoken), false, false);
        toks.insert(toks.end(), text_ids.begin(), text_ids.end());
        if (c.final_seg) toks.push_back(c.final_seg);
        toks.push_back(c.speech_start);

        // Per-chunk budget sized from the chunk's length (model.py::_chunk_budget).
        int32_t budget = c.max_new_tokens;
        if (n_segments > 1) {
            const int32_t expected = (int32_t) ((double) strip_copy(spoken).size() / 20.0 * 25.0 * 1.8);
            budget = std::max(200, std::min(c.max_new_tokens, expected));
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "kitten: voice=%s, temp=%.2f top_k=%d top_p=%.2f min_p=%.2f rep=%.2f/%d run=%.2f/%d\n",
                  codec_speaker_name(spk), c.temp, c.top_k, c.top_p, c.min_p, c.rep_penalty, c.rep_window,
                  c.run_penalty, c.run_grace);

        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_TOKENS;
        out->n_embd = n_embd;
        out->seqs   = {{alm_chunk::of_embd(std::vector<float>(lm_cond, lm_cond + n_embd), 1),
                        alm_chunk::of_tokens(std::move(toks))}};
        // logits.py: token-run penalty (DRY with allowed length = the grace),
        // repetition penalty over the last generated tokens exempting the end
        // and silence tokens, then temperature -> top-k -> top-p -> min-p.
        out->sampler.temp               = c.temp;
        out->sampler.top_k              = c.top_k;
        out->sampler.top_p              = c.top_p;
        out->sampler.min_p              = c.min_p;
        out->sampler.penalty_repeat     = c.rep_penalty;
        out->sampler.penalty_last_n     = c.rep_window;
        out->sampler.dry_multiplier     = c.run_penalty;
        out->sampler.dry_base           = 1.5f;
        out->sampler.dry_allowed_length = c.run_grace;
        out->sampler.dry_penalty_last_n = -1;
        out->penalty_exempt = {c.speech_end, c.stop, c.audio_base + c.silence};
        out->samplers = "dry;penalties;temperature;top_k;top_p;min_p";
        // Speech is audio tokens until <speech_end> / [STOP]: keep a quantized
        // backbone from drifting into the Qwen3 text vocabulary.
        out->allowed = {{c.audio_base, c.audio_base + c.n_audio}, {c.speech_end, c.speech_end + 1}, {c.stop, c.stop + 1}};
        out->max_frames = budget;
        req.apply(out->sampler);
        return true;
    }

    bool observe(const int32_t * tokens, const float *, codec_audio_lm_verdict * v) override {
        if (tokens == nullptr) return fail("observe needs the sampled token");
        const int32_t t = tokens[0];
        if (t == cfg_run_.speech_end || t == cfg_run_.stop) {
            stop_reason_ = "eos";
            codec_log(CODEC_LOG_LEVEL_INFO, "KittenTTS AR done: %zu frames, stop=eos\n", codes_.size());
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        codes_.push_back(t - cfg_run_.audio_base);
        *v = {CODEC_AUDIO_LM_FEED_TOKEN, true, true, -1};
        return true;
    }

    bool decode(audio_lm_audio_output * out) override {
        if (codes_.empty()) return fail("the model produced no audio tokens");
        // S3Gen drops its last tokens as lookahead: pad with silence (vocoder.py).
        std::vector<int32_t> codes = codes_;
        codes.insert(codes.end(), (size_t) cfg_run_.n_trailing_silence, cfg_run_.silence);
        audio_lm_reset(alm_);
        if (!audio_lm_push_codes(alm_, codes.data(), (int32_t) codes.size(), 1)) return fail_alm("push codes");
        return audio_lm_decode_audio(alm_, out) || fail_alm("decode");
    }

private:
    kitten_config            cfg_;       // defaults (default preset) for caps / split
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
