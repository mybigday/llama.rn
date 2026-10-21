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
#include "audio_lm_sampler.h"
#include "audio_lm_text_embd.h"

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
    bool begin(alm_tts_request const &, const alm_vocab & vocab) {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        n_cb_   = audio_lm_n_codebook(alm_);
        frames_ = 0;
        stop_reason_ = "max_frames";
        if (!table_.loaded()) {
            if (backbone_path.empty()) return fail("this model needs the backbone GGUF path (its text embedding table)");
            std::string err;
            if (!table_.load(backbone_path.c_str(), hidden_, err)) return fail(err);
        }
        audio_lm_set_uses_embed_override(alm_, true, 1);
        return true;
    }

    // One audio frame from the depth decoder; false + *stop on the audio end.
    bool frame(const float * hidden, std::vector<alm_head_sampler> & samplers, std::vector<int32_t> & codes, bool * stop) {
        *stop = false;
        codes.assign((size_t) n_cb_, 0);
        if (!audio_lm_step_begin(alm_, hidden, hidden_)) return fail_alm("step");
        for (int32_t cb = 0; cb < n_cb_; ++cb) {
            int32_t idx = 0, n = 0;
            const float * lg = audio_lm_step_logits(alm_, &idx, &n);
            if (lg == nullptr) return fail_alm("step logits");
            alm_head_sampler & smpl = samplers.size() == 1 ? samplers[0] : samplers[(size_t) cb];
            if (!audio_lm_step_push_code(alm_, smpl.sample(lg, n))) return fail_alm("step");
        }
        if (!audio_lm_step_finish(alm_, codes.data(), n_cb_)) return fail_alm("step");
        if (audio_lm_observe_codes(alm_, codes.data(), n_cb_, hidden, hidden_) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "eos_code_c0";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=eos_code_c0\n", frames_);
            *stop = true;
            return true;
        }
        ++frames_;
        return true;
    }

    // The request's llama.cpp settings, with one resolved seed for the heads.
    codec_audio_lm_sampler head_settings(const alm_tts_request & req, alm_completion * out) const {
        req.sampling_to(out);
        codec_audio_lm_sampler hs = req.head_sampler();
        hs.seed = req.seed();
        return hs;
    }

    audio_lm_prompt_info pi_;
    int32_t hidden_ = 0, n_cb_ = 0, frames_ = 0;
    alm_text_embd table_;
    std::vector<float> next_;
};

class realtime_family : public text_lane_family {
public:
    using text_lane_family::text_lane_family;
    const char * name() const override { return "realtime"; }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        if (!begin(req, vocab)) return false;
        const std::vector<int32_t> ctx_toks = vocab.tokenize(pi_.prompt_prefix + pi_.prompt_suffix, pi_.add_bos, pi_.parse_special);
        text_ = vocab.tokenize(req.text, false, false);
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

        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden_;
        out->seqs   = {{alm_chunk::of_embd(std::move(rows), (int32_t) r)}};
        // One sampler per codebook, each with its own penalty history.
        const codec_audio_lm_sampler hs = head_settings(req, out);
        samplers_.assign((size_t) n_cb_, alm_head_sampler());
        std::string serr;
        for (auto & s : samplers_) if (!s.init(hs, true, &serr)) return fail(serr);
        out->needs_embeddings = true;
        out->sample_first     = false;
        out->streamable       = true;
        out->max_frames       = 512;
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        std::vector<int32_t> codes;
        bool stop = false;
        if (!frame(hiddens, samplers_, codes, &stop)) return false;
        if (stop) { *v = {CODEC_AUDIO_LM_STOP, false, false, -1}; return true; }
        const int32_t t = text_idx_ < (int32_t) text_.size() ? text_[(size_t) text_idx_] : pi_.text_pad_id;
        ++text_idx_;
        next_.assign((size_t) hidden_, 0.0f);
        if (!compose(t, codes.data(), next_.data())) return false;
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

private:
    // text_embd[text_token] + the codes' audio embedding.
    bool compose(int32_t text_token, const int32_t * codes, float * dst) {
        if (!table_.row(text_token, dst)) return fail("text embedding row " + std::to_string(text_token));
        std::vector<float> aud((size_t) hidden_, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(alm_, codes, n_cb_, aud.data(), hidden_)) return fail_alm("audio embedding");
        for (int32_t i = 0; i < hidden_; ++i) dst[i] += aud[(size_t) i];
        return true;
    }

    std::vector<int32_t> text_;
    int32_t text_idx_ = 0;
    std::vector<alm_head_sampler> samplers_;
};

class lfm2_family : public text_lane_family {
public:
    using text_lane_family::text_lane_family;
    const char * name() const override { return "lfm2"; }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        if (!begin(req, vocab)) return false;
        const std::vector<int32_t> toks = vocab.tokenize(pi_.prompt_prefix + req.text + pi_.prompt_suffix,
                                                         pi_.add_bos, pi_.parse_special);
        if (toks.empty()) return fail("empty prompt after tokenize");
        audio_ = false;
        text_steps_ = 0;
        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden_;
        out->seqs   = {{alm_chunk::of_tokens(toks)}};
        // One sampler (one random stream) for the text warm-up and the audio
        // codebooks; it takes no penalties (their histories would mix).
        const codec_audio_lm_sampler hs = head_settings(req, out);
        samplers_.assign(1, alm_head_sampler());
        std::string serr;
        if (!samplers_[0].init(hs, false, &serr)) return fail(serr);
        out->needs_embeddings = true;
        out->sample_first     = false;
        out->streamable       = true;
        out->max_frames       = 512;
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        if (!audio_) {
            // Text warm-up: at most max_text_tokens, then <|audio_start|> opens the audio.
            int32_t tok = pi_.audio_start_id;
            if (text_steps_ < pi_.max_text_tokens) {
                tok = samplers_[0].sample(text_logits(hiddens), (int32_t) table_.vocab);
                ++text_steps_;
                if (tok == pi_.text_end_id) {
                    stop_reason_ = "text_end";
                    *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
                    return true;
                }
            }
            if (tok == pi_.audio_start_id) audio_ = true;
            *v = {CODEC_AUDIO_LM_FEED_TOKEN, false, false, tok};
            return true;
        }
        std::vector<int32_t> codes;
        bool stop = false;
        if (!frame(hiddens, samplers_, codes, &stop)) return false;
        if (stop) { *v = {CODEC_AUDIO_LM_STOP, false, false, -1}; return true; }
        next_.assign((size_t) hidden_, 0.0f);
        if (!audio_lm_compose_audio_codes_embd(alm_, codes.data(), n_cb_, next_.data(), hidden_)) return fail_alm("audio embedding");
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

private:
    // The tied-embedding head: logits[v] = <hidden, token_embd[v]>.
    const float * text_logits(const float * h) {
        const int32_t n_vocab = (int32_t) table_.vocab;
        logits_.resize((size_t) n_vocab);
        row_.resize((size_t) hidden_);
        for (int32_t t = 0; t < n_vocab; ++t) {
            if (!table_.row(t, row_.data())) { logits_[(size_t) t] = -1e30f; continue; }
            double acc = 0.0;
            for (int32_t i = 0; i < hidden_; ++i) acc += (double) h[i] * (double) row_[(size_t) i];
            logits_[(size_t) t] = (float) acc;
        }
        return logits_.data();
    }

    bool audio_ = false;
    int32_t text_steps_ = 0;
    std::vector<alm_head_sampler> samplers_;
    std::vector<float> logits_, row_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_realtime(audio_lm_context * alm) { return std::make_unique<realtime_family>(alm); }
std::unique_ptr<alm_family> alm_family_lfm2(audio_lm_context * alm) { return std::make_unique<lfm2_family>(alm); }

}  // namespace codec_common
