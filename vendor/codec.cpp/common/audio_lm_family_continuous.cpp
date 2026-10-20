// Continuous-latent families (BlueMagpie): the backbone hidden state drives a
// codec-side CFM head that writes one latent patch per step; the patch's
// feedback embedding is the next backbone input, and a stop head ends the
// generation.  The whole prompt's hidden states prime the head first.

#include "audio_lm_family.h"

namespace codec_common {
namespace {

class continuous_family : public alm_family {
public:
    using alm_family::alm_family;
    const char * name() const override { return "continuous"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        frames_ = 0;
        stop_reason_ = "max_frames";
        audio_lm_set_continuous_params(alm_, req.flow_guidance >= 0.0f ? req.flow_guidance : 2.0f,
                                       req.flow_steps > 0 ? req.flow_steps : 10, req.min_frames);
        const std::vector<int32_t> toks = vocab.tokenize(pi_.prompt_prefix + req.text + pi_.prompt_suffix,
                                                         pi_.add_bos, pi_.parse_special);
        if (toks.empty()) return fail("empty prompt after tokenize");
        codec_log(CODEC_LOG_LEVEL_INFO, "prompt: \"%s%s%s\" -> %zu tokens\n", pi_.prompt_prefix.c_str(), req.text.c_str(),
                  pi_.prompt_suffix.c_str(), toks.size());
        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_HIDDEN;
        out->n_embd = hidden_;
        out->seqs   = {{alm_chunk::of_tokens(toks)}};
        out->needs_embeddings    = true;
        out->needs_prompt_hidden = true;
        out->sample_first        = false;
        out->streamable          = true;
        out->max_frames          = 512;
        return true;
    }

    bool prefill(const float * hiddens, int32_t n_pos) override {
        return audio_lm_text_prefill(alm_, hiddens, n_pos, hidden_) || fail_alm("prefill");
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        if (audio_lm_observe_hidden(alm_, hiddens, hidden_, nullptr) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "stop_head";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=stop_head\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        ++frames_;
        int32_t dim = 0;
        const float * fb = audio_lm_get_next_embed(alm_, &dim);
        if (fb == nullptr || dim != hidden_) return fail("no feedback embedding");
        next_.assign(fb, fb + dim);
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

private:
    audio_lm_prompt_info pi_;
    int32_t hidden_ = 0, frames_ = 0;
    std::vector<float> next_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_continuous(audio_lm_context * alm) { return std::make_unique<continuous_family>(alm); }

}  // namespace codec_common
