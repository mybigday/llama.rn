// Text-lane families: residual depth-AR models whose backbone input mixes the
// backbone's own text embedding with the codec-side audio embedding, read
// from the backbone GGUF (alm_text_embd).
//
//  * MOSS-TTS-Realtime streams the text into the assistant turn one token per
//    audio frame: prefill = context + the first prefill_text_len text tokens
//    (audio lanes = pad, the last row's cb0 = audio BOS); each step adds the
//    next text token (text pad once the text ends) to the previous frame's
//    audio embedding, until cb0 is the audio end.
//  * LFM2-Audio is sequential: after the prompt it writes text tokens from its
//    tied-embedding head (recomputed here: llama.cpp returns no logits with
//    embeddings on) until <|audio_start|>, then one audio frame per step.

#include "audio_lm_family.h"
#include "audio_lm_text_embd.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>

namespace codec_common {
namespace {

class text_lane_family : public alm_family {
public:
    using alm_family::alm_family;

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->needs_backbone_path = true;
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

protected:
    // The end code (eos_code_c0) waits for the request's min_frames audio
    // frames, masked in the codebook-0 head; the model's eos_min_step is a
    // floor (before it the codec takes the end code as an ordinary code).
    int32_t min_frames_of(const alm_tts_request & req) const {
        audio_lm_prompt_info pi;
        if (!audio_lm_get_prompt_info(alm_, &pi)) return std::max(0, req.min_frames);
        return std::max(pi.eos_code_c0 >= 0 ? pi.eos_min_step : 0, std::max(0, req.min_frames));
    }

    bool begin(alm_tts_request const & req, const alm_vocab & vocab) {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        n_cb_   = audio_lm_n_codebook(alm_);
        frames_ = 0;
        mask_end_until_ = std::max(0, req.min_frames);
        stop_reason_ = "max_frames";
        {
            std::lock_guard<std::mutex> lock(table_->m);
            if (!table_->t.loaded()) {
                if (backbone_path.empty()) return fail("this model needs the backbone GGUF path (its text embedding table)");
                std::string err;
                if (!table_->t.load(backbone_path.c_str(), hidden_, err)) return fail(err);
            }
        }
        audio_lm_set_uses_embed_override(alm_, true, 1);
        return true;
    }

    // One audio frame from the depth decoder: start_frame(), one SAMPLE_HEAD per
    // codebook (head_token() resumes), then after_frame() with the codes.
    bool start_frame(const float * hidden, codec_audio_lm_verdict * v) {
        if (!audio_lm_step_begin(alm_, hidden, hidden_)) return fail_alm("step");
        hidden_frame_.assign(hidden, hidden + hidden_);
        cb_ = 0;
        return next_head(v);
    }

public:
    void share_from(alm_family & base) override {
        if (auto * b = dynamic_cast<text_lane_family *>(&base)) table_ = b->table_;
    }

    const float * head_logits() const override { return logits_.data(); }

    bool head_token(int32_t id, codec_audio_lm_verdict * v) override {
        if (id < 0 || id >= (int32_t) logits_.size()) return fail("head token out of range");
        if (in_text_) { in_text_ = false; return text_token(id, v); }
        if (cb_ >= n_cb_) return fail("no codebook is waiting for a token");
        if (!audio_lm_step_push_code(alm_, id)) return fail_alm("step");
        ++cb_;
        return next_head(v);
    }

protected:
    // The head group of codebook `cb`, and what follows a whole frame.
    virtual int32_t group_of(int32_t cb) const = 0;
    virtual bool after_frame(const std::vector<int32_t> & codes, codec_audio_lm_verdict * v) = 0;
    // A text-head token (LFM2's warm-up), when in_text_ was set.
    virtual bool text_token(int32_t, codec_audio_lm_verdict *) { return fail("no text head"); }

    bool next_head(codec_audio_lm_verdict * v) {
        if (cb_ < n_cb_) {
            int32_t idx = 0, n = 0;
            const float * lg = audio_lm_step_logits(alm_, &idx, &n);
            if (lg == nullptr) return fail_alm("step logits");
            logits_.assign(lg, lg + n);
            if (cb_ == 0 && frames_ < mask_end_until_ && pi_.eos_code_c0 >= 0 && pi_.eos_code_c0 < n) {
                logits_[(size_t) pi_.eos_code_c0] = -INFINITY;   // not the end yet (min_frames)
            }
            *v = {CODEC_AUDIO_LM_SAMPLE_HEAD, false, true, false, -1, group_of(cb_), n};
            return true;
        }
        std::vector<int32_t> codes((size_t) n_cb_, 0);
        if (!audio_lm_step_finish(alm_, codes.data(), n_cb_)) return fail_alm("step");
        if (audio_lm_observe_codes(alm_, codes.data(), n_cb_, hidden_frame_.data(), hidden_) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "eos_code_c0";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=eos_code_c0\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
            return true;
        }
        ++frames_;
        return after_frame(codes, v);
    }

    std::vector<float> hidden_frame_, logits_;
    int32_t cb_ = 0;
    bool in_text_ = false;
    audio_lm_prompt_info pi_;
    int32_t hidden_ = 0, n_cb_ = 0, frames_ = 0, mask_end_until_ = 0;
    // The backbone's table, loaded on first use and shared by the contexts of
    // one model (codec_audio_lm_init_shared).
    struct shared_table {
        std::mutex    m;
        alm_text_embd t;
    };
    std::shared_ptr<shared_table> table_ = std::make_shared<shared_table>();
    std::vector<float> next_;
};

class realtime_family : public text_lane_family {
public:
    using text_lane_family::text_lane_family;
    const char * name() const override { return "realtime"; }

    // Prompt: {the GGUF's context}<__codec:text__>{text} (BOS added; the text
    // streams into the assistant turn, one token per frame).
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        audio_lm_prompt_info pi;
        if (!audio_lm_get_prompt_info(alm_, &pi)) return fail_alm("prompt info");
        out->prompt = pi.prompt_prefix + pi.prompt_suffix + alm_codec_marker("text") + req.text;
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        d.n_embd = audio_lm_hidden_dim(alm_);
        // One head group per codebook, each with its own penalty history.
        d.head_groups.assign((size_t) audio_lm_n_codebook(alm_), alm_head_group{true, {}, {}});
        d.needs_embeddings = true;
        d.sample_first     = false;
        d.streamable       = true;
        d.max_frames       = 512;
        d.min_frames       = min_frames_of(req);
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        if (!begin(req, vocab)) return false;
        const std::vector<int32_t> ctx_toks = vocab.tokenize(prompt.text(), pi_.add_bos, pi_.parse_special);
        if (!prompt.codec("text")) return fail("this model's prompt has <__codec:text__> before the text");
        text_ = vocab.tokenize(prompt.text(), false, false);
        if (ctx_toks.empty() || text_.empty()) return fail("empty context or text tokens");
        const int32_t prefill_n = std::min<int32_t>(pi_.prefill_text_len, (int32_t) text_.size());
        const std::vector<int32_t> pad((size_t) n_cb_, pi_.audio_pad_code);
        std::vector<float> rows((ctx_toks.size() + (size_t) prefill_n) * hidden_);
        size_t r = 0;
        for (int32_t t : ctx_toks) {
            if (!compose(t, pad.data(), rows.data() + r++ * hidden_)) return false;
        }
        for (int32_t i = 0; i < prefill_n; ++i) {
            std::vector<int32_t> codes = pad;
            if (i == prefill_n - 1) codes[0] = pi_.bos_code_c0;
            if (!compose(text_[(size_t) i], codes.data(), rows.data() + r++ * hidden_)) return false;
        }
        text_idx_ = prefill_n;
        out->seqs = {{alm_chunk::of_embd(std::move(rows), (int32_t) r)}};
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        return start_frame(hiddens, v);
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

protected:
    int32_t group_of(int32_t cb) const override { return cb; }

    bool after_frame(const std::vector<int32_t> & codes, codec_audio_lm_verdict * v) override {
        const int32_t t = text_idx_ < (int32_t) text_.size() ? text_[(size_t) text_idx_] : pi_.text_pad_id;
        ++text_idx_;
        next_.assign((size_t) hidden_, 0.0f);
        if (!compose(t, codes.data(), next_.data())) return false;
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, true, -1, 0, 0};
        return true;
    }

private:
    // text_embd[text_token] + the codes' audio embedding.
    bool compose(int32_t text_token, const int32_t * codes, float * dst) {
        if (!table_->t.row(text_token, dst)) return fail("text embedding row " + std::to_string(text_token));
        std::vector<float> aud((size_t) hidden_, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(alm_, codes, n_cb_, aud.data(), hidden_)) return fail_alm("audio embedding");
        for (int32_t i = 0; i < hidden_; ++i) dst[i] += aud[(size_t) i];
        return true;
    }

    std::vector<int32_t> text_;
    int32_t text_idx_ = 0;
};

class lfm2_family : public text_lane_family {
public:
    using text_lane_family::text_lane_family;
    const char * name() const override { return "lfm2"; }

    // Prompt: {the GGUF's prefix}{text}{suffix} (BOS added).
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        audio_lm_prompt_info pi;
        if (!audio_lm_get_prompt_info(alm_, &pi)) return fail_alm("prompt info");
        out->prompt = pi.prompt_prefix + req.text + pi.prompt_suffix;
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        d.n_embd = audio_lm_hidden_dim(alm_);
        // One head group (one random stream) for the text warm-up and the audio
        // codebooks, without penalties (their histories would mix).
        d.head_groups = {alm_head_group{false, {}, {}}};
        d.needs_embeddings = true;
        d.sample_first     = false;
        d.streamable       = true;
        d.max_frames       = 512;
        d.min_frames       = min_frames_of(req);
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        if (!begin(req, vocab)) return false;
        std::string text;
        if (!prompt.rest_text(&text)) return fail("this model's prompt takes no markers");
        const std::vector<int32_t> toks = vocab.tokenize(text, pi_.add_bos, pi_.parse_special);
        if (toks.empty()) return fail("empty prompt after tokenize");
        audio_ = false;
        text_steps_ = 0;
        out->seqs = {{alm_chunk::of_tokens(toks)}};
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        if (!audio_) {
            // Text warm-up: at most max_text_tokens, then <|audio_start|> opens the audio.
            if (text_steps_ < pi_.max_text_tokens) {
                text_logits(hiddens);
                in_text_ = true;
                *v = {CODEC_AUDIO_LM_SAMPLE_HEAD, false, false, false, -1, 0, (int32_t) logits_.size()};   // text, not audio
                return true;
            }
            return text_token(pi_.audio_start_id, v);
        }
        return start_frame(hiddens, v);
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

protected:
    int32_t group_of(int32_t) const override { return 0; }

    bool after_frame(const std::vector<int32_t> & codes, codec_audio_lm_verdict * v) override {
        next_.assign((size_t) hidden_, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(alm_, codes.data(), n_cb_, next_.data(), hidden_)) return fail_alm("audio embedding");
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, true, -1, 0, 0};
        return true;
    }

    bool text_token(int32_t tok, codec_audio_lm_verdict * v) override {
        ++text_steps_;
        if (tok == pi_.text_end_id) {
            stop_reason_ = "text_end";
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
            return true;
        }
        if (tok == pi_.audio_start_id) audio_ = true;
        *v = {CODEC_AUDIO_LM_FEED_TOKEN, false, audio_, false, tok, 0, 0};   // <|audio_start|> opens the audio
        return true;
    }

private:
    // The tied-embedding head: logits[v] = <hidden, token_embd[v]>.
    void text_logits(const float * h) {
        const int32_t n_vocab = (int32_t) table_->t.vocab;
        logits_.resize((size_t) n_vocab);
        row_.resize((size_t) hidden_);
        for (int32_t t = 0; t < n_vocab; ++t) {
            if (!table_->t.row(t, row_.data())) { logits_[(size_t) t] = -1e30f; continue; }
            double acc = 0.0;
            for (int32_t i = 0; i < hidden_; ++i) acc += (double) h[i] * (double) row_[(size_t) i];
            logits_[(size_t) t] = (float) acc;
        }
    }

    bool audio_ = false;
    int32_t text_steps_ = 0;
    std::vector<float> row_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_realtime(audio_lm_context * alm) { return std::make_unique<realtime_family>(alm); }
std::unique_ptr<alm_family> alm_family_lfm2(audio_lm_context * alm) { return std::make_unique<lfm2_family>(alm); }

}  // namespace codec_common
