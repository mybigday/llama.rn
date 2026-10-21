// Continuous-latent families (BlueMagpie): the backbone hidden state drives a
// codec-side CFM head that writes one latent patch per step; the patch's
// feedback embedding is the next backbone input, and a stop head ends the
// generation.  The whole prompt's hidden states prime the head first.

#include "audio_lm_family.h"

#include <random>

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

    // Prompt: {the GGUF's prefix}{text}{suffix} (the model's BOS added).
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        audio_lm_prompt_info pi;
        if (!audio_lm_get_prompt_info(alm_, &pi)) return fail_alm("prompt info");
        out->prompt = pi.prompt_prefix + req.text + pi.prompt_suffix;
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_HIDDEN;
        d.n_embd = audio_lm_hidden_dim(alm_);
        d.needs_embeddings    = true;
        d.needs_prompt_hidden = true;
        d.sample_first        = false;
        d.streamable          = true;
        d.max_frames          = 512;
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        std::string text;
        if (!prompt.rest_text(&text)) return fail("this model's prompt takes no markers");
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        frames_ = 0;
        set_seed(CODEC_AUDIO_LM_DEFAULT_SEED);   // random unless the host sets one
        stop_reason_ = "max_frames";
        // CFM steps / guidance: the request's, else the GGUF's (VoxCPM.generate's 10 / 2.0).
        const char * m_steps = audio_lm_meta_str(alm_, "codec.lm.cfm_n_timesteps");
        const char * m_cfg   = audio_lm_meta_str(alm_, "codec.lm.cfm_cfg_value");
        audio_lm_set_continuous_params(alm_, req.flow_guidance >= 0.0f ? req.flow_guidance : (m_cfg ? (float) std::atof(m_cfg) : 2.0f),
                                       req.flow_steps > 0 ? req.flow_steps : (m_steps ? std::atoi(m_steps) : 10), req.min_frames);
        const std::vector<int32_t> toks = vocab.tokenize(text, pi_.add_bos, pi_.parse_special);
        if (toks.empty()) return fail("empty prompt after tokenize");
        codec_log(CODEC_LOG_LEVEL_INFO, "prompt: \"%s\" -> %zu tokens\n", text.c_str(), toks.size());
        out->seqs = {{alm_chunk::of_tokens(toks)}};
        return true;
    }

    bool prefill(const float * hiddens, int32_t n_pos) override {
        return audio_lm_text_prefill(alm_, hiddens, n_pos, hidden_) || fail_alm("prefill");
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        // CFM init noise ~ N(0, 1) per patch (torch.randn), from the host's seed.
        const codec_lm_info * li = codec_lm_get_info(audio_lm_get_lm(alm_));
        noise_.resize((size_t) li->patch_size * li->latent_dim);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float & z : noise_) z = nd(rng_);
        if (audio_lm_observe_hidden(alm_, hiddens, hidden_, noise_.data()) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "stop_head";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=stop_head\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1};
            return true;
        }
        ++frames_;
        int32_t dim = 0;
        const float * fb = audio_lm_get_next_embed(alm_, &dim);
        if (fb == nullptr || dim != hidden_) return fail("no feedback embedding");
        next_.assign(fb, fb + dim);
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

    void set_seed(uint32_t seed) override { rng_.seed(alm_resolve_seed(seed)); }

private:
    audio_lm_prompt_info pi_;
    std::mt19937 rng_;
    std::vector<float> noise_;
    int32_t hidden_ = 0, frames_ = 0;
    std::vector<float> next_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_continuous(audio_lm_context * alm) { return std::make_unique<continuous_family>(alm); }

}  // namespace codec_common
