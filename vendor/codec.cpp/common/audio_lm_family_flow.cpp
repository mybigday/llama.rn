// Pocket-TTS (codec.lm.kind = "flow_lm"): the FlowLM transformer is the
// backbone (llama.cpp starcoder2, its token_embd the text LUT); the codec_lm
// adaptor owns the voice prefix rows, the per-frame input embedding
// (input_linear of BOS / the previous latent), the EOS head and the LSD flow
// head.  Reference: pocket_tts/models/tts_model.py.
//
// Backbone sequence: [ text tokens | voice prefix rows | frame inputs ].  The
// first frame input (from BOS) ends the prompt; each step's hidden goes
// through codec_lm_step_generate (one denormalized 32-d latent), whose
// feedback embedding is the next input.  Once the EOS head fires the model
// runs frames_after_eos more frames and drops the one it stops on
// (_autoregressive_generation).  The codec decodes the latents at the end.

#include "audio_lm_family.h"
#include "lm/lm_models.h"

#include <cmath>
#include <cstring>
#include <random>

namespace codec_common {
namespace {

// prepare_text_prompt: strip, collapse double spaces, uppercase the first
// letter, ensure trailing punctuation.  Returns the reference's
// frames_after_eos guess (3 if <= 4 words else 1).
int32_t flow_prepare_text(std::string & text) {
    const size_t b = text.find_first_not_of(" \t\r\n");
    const size_t e = text.find_last_not_of(" \t\r\n");
    text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
    for (char & c : text) if (c == '\n' || c == '\r') c = ' ';
    for (size_t i = text.find("  "); i != std::string::npos; i = text.find("  ")) text.replace(i, 2, " ");
    if (text.empty()) return 3;
    int32_t words = 0;
    bool in_w = false;
    for (char c : text) { if (c == ' ') in_w = false; else if (!in_w) { in_w = true; ++words; } }
    if (text[0] >= 'a' && text[0] <= 'z') text[0] = (char) (text[0] - 'a' + 'A');
    const char last = text.back();
    if ((last >= 'a' && last <= 'z') || (last >= 'A' && last <= 'Z') || (last >= '0' && last <= '9')) text += '.';
    return words <= 4 ? 3 : 1;
}

class flow_family : public alm_family {
public:
    flow_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_flow_info * fi) : alm_family(alm), lm_(lm), fi_(fi) {}
    ~flow_family() override { if (st_) codec_lm_state_free(st_); }
    const char * name() const override { return "flow"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = true;
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

    // The voice conditions the LM (reference latents), not a decoder speaker.
    bool set_voice(const alm_tts_voice & v) override {
        voice_ = v;
        return true;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab &, alm_completion * out) override {
        const int32_t ldim = fi_->ldim, hidden = fi_->d_model;
        std::string text = req.text;
        const int32_t fae_guess = flow_prepare_text(text) + 2;   // the reference adds 2
        fae_ = fi_->frames_after_eos >= 0 ? fi_->frames_after_eos : fae_guess;
        std::vector<int32_t> ids(text.size() + 16);
        int32_t n_tok = 0;
        if (codec_lm_flow_tokenize(lm_, text.c_str(), ids.data(), (int32_t) ids.size(), &n_tok) != CODEC_STATUS_SUCCESS ||
            n_tok <= 0) {
            return fail(std::string("tokenize: ") + codec_lm_get_last_error(lm_));
        }
        ids.resize((size_t) n_tok);
        codec_log(CODEC_LOG_LEVEL_INFO, "pocket: text=\"%s\" -> %d tokens\n", text.c_str(), n_tok);

        // Voice conditioning: the reference's latents -> [bos_before_voice | speaker_proj] rows.
        std::vector<float> rows;
        int32_t n_rows = 0;
        if (!voice_.ref_pcm.empty()) {
            std::vector<float> mu;
            int32_t n_voice = 0, dim = 0;
            if (!audio_lm_encode_latent(alm_, voice_.ref_pcm.data(), (int32_t) voice_.ref_pcm.size(), voice_.ref_sample_rate,
                                        &mu, &n_voice, &dim) || dim != ldim) {
                return fail_alm("reference audio encode");
            }
            rows.resize((size_t) (n_voice + 1) * hidden);
            if (codec_lm_flow_speaker_rows(lm_, mu.data(), n_voice, rows.data(), n_voice + 1, &n_rows) != CODEC_STATUS_SUCCESS) {
                return fail(std::string("speaker rows: ") + codec_lm_get_last_error(lm_));
            }
            rows.resize((size_t) n_rows * hidden);
            codec_log(CODEC_LOG_LEVEL_INFO, "pocket: voice conditioning -> %d rows\n", n_rows);
        }

        if (st_) codec_lm_state_free(st_);
        st_ = codec_lm_state_new(lm_);
        if (!st_) return fail("state allocation failed");
        std::vector<float> first((size_t) hidden);   // the first frame input: input_linear(BOS)
        if (codec_lm_step_feedback_embd(st_, first.data()) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("frame input: ") + codec_lm_state_get_last_error(st_));
        }

        // CFM init noise ~ N(0, temperature): Box-Muller on a uniform mt19937
        // stream (std::normal_distribution's polar method gives poor early
        // draws; the FlowLM is very sensitive to the first frames' noise).
        std::seed_seq seq{ req.seed_or_default(), 0x9E3779B9u, 0x243F6A88u, 0xB7E15162u };
        rng_.seed(seq);
        min_len_ = req.min_frames >= 0 ? req.min_frames : 0;   // ignore the EOS head before this frame
        eos_step_ = -1;
        step_ = 0;
        latents_.clear();
        n_frames_ = 0;
        stop_reason_ = "max_frames";

        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden;
        out->seqs   = {{alm_chunk::of_tokens(ids), alm_chunk::of_embd(std::move(rows), n_rows),
                        alm_chunk::of_embd(std::move(first), 1)}};
        out->needs_embeddings = true;
        out->sample_first     = false;
        // _estimate_max_gen_len: 12.5 frames/s for ~3 tokens/s + 2 s.
        out->max_frames = std::max(8, (int32_t) std::ceil(((double) n_tok / 3.0 + 2.0) * 12.5));
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        const int32_t ldim = fi_->ldim;
        const float noise_std = std::sqrt(fi_->temperature);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<float> noise((size_t) ldim), lat((size_t) ldim);
        for (float & z : noise) {
            float u1 = u(rng_), u2 = u(rng_);
            if (u1 < 1e-12f) u1 = 1e-12f;
            z = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2) * noise_std;
        }
        int32_t is_eos = 0;
        if (codec_lm_step_generate(st_, hiddens, 0.0f, 0, noise.data(), lat.data(), &is_eos) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("step: ") + codec_lm_state_get_last_error(st_));
        }
        const int32_t step = step_++;
        if (is_eos && eos_step_ < 0 && step >= min_len_) eos_step_ = step;
        if (eos_step_ >= 0 && step >= eos_step_ + fae_) {
            stop_reason_ = "eos_head";
            codec_log(CODEC_LOG_LEVEL_INFO, "pocket AR done: %d frames, eos_step=%d, stop=eos_head\n", n_frames_, eos_step_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        latents_.insert(latents_.end(), lat.begin(), lat.end());
        ++n_frames_;
        next_.assign((size_t) fi_->d_model, 0.0f);
        if (codec_lm_step_feedback_embd(st_, next_.data()) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("frame input: ") + codec_lm_state_get_last_error(st_));
        }
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }
    int32_t decodable_frames() const override { return n_frames_; }

    bool decode(audio_lm_audio_output * out) override {
        const int32_t ldim = fi_->ldim, n = n_frames_;
        if (n == 0) return fail("no frames generated");
        // Frame-major [t * ldim + d] -> the codec's channel-major [d * n + t].
        std::vector<float> cm((size_t) ldim * n);
        for (int32_t t = 0; t < n; ++t) {
            for (int32_t d = 0; d < ldim; ++d) cm[(size_t) d * n + t] = latents_[(size_t) t * ldim + d];
        }
        return audio_lm_decode_latents(alm_, cm.data(), n, ldim, out) || fail_alm("decode");
    }

private:
    codec_lm * lm_;
    const codec_lm_flow_info * fi_;
    codec_lm_state * st_ = nullptr;
    std::mt19937 rng_;
    int32_t fae_ = 3, min_len_ = 0, eos_step_ = -1, step_ = 0, n_frames_ = 0;
    std::vector<float> latents_, next_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_flow(audio_lm_context * alm, codec_lm * lm) {
    const codec_lm_flow_info * fi = codec_lm_flow_get_info(lm);
    return fi ? std::make_unique<flow_family>(alm, lm, fi) : nullptr;
}

}  // namespace codec_common
