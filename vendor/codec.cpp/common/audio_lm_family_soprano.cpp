// Soprano: a codec-only Vocos decoder fed the backbone's hidden states.
// Reference: soprano/tts.py + backends/transformers.py.  The prompt is
// `[STOP][TEXT]{text}[START]`; the backbone generates until [STOP] (its eos)
// and the decoder input is the final-norm hidden state that PREDICTS each
// non-eos token (the first comes from the prompt's last position).  Sampling:
// temperature 0 (greedy), top-p 0.95, repetition penalty 1.2 over prompt +
// generated ids, at most 512 new tokens, restricted to the "[N]" speech
// tokens + [STOP].  The decoder output, (n-1)*2048 samples, is the whole result.

#include "audio_lm_family.h"

namespace codec_common {
namespace {

class soprano_family : public alm_family {
public:
    using alm_family::alm_family;
    const char * name() const override { return "soprano"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->sample_rate = 32000;
    }

    // Prompt: [STOP][TEXT]{text}[START]
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("Soprano needs the backbone vocabulary");
        // The speech tokens are the contiguous pieces "[0]".."[N-1]"; the rest
        // of the vocab is the text BPE.
        int32_t off = -1, count = 0;
        for (int32_t id = 0; id < vocab.n_tokens(); ++id) {
            const std::string p = vocab.piece(id);
            if (p == "[" + std::to_string(count) + "]") { if (off < 0) off = id; ++count; }
            else if (off >= 0) break;
        }
        if (off < 0) return fail("Soprano: no [N] speech tokens in the backbone vocab");
        const int32_t eos = vocab.eos();
        out->prompt = "[STOP][TEXT]" + req.text + "[START]";
        alm_completion & d = out->desc;
        d.flow = CODEC_AUDIO_LM_FLOW_HIDDEN;
        d.allowed = {{off, off + count}};
        if (eos >= 0) d.allowed.push_back({eos, eos + 1});
        d.needs_embeddings = true;
        d.sample_first = true;
        d.max_frames = 512;
        d.min_frames = std::max(0, req.min_frames);   // the model has no minimum of its own
        d.grammar = alm_min_length_grammar(eos, d.min_frames);
        return true;
    }

    bool tts_build(const alm_tts_request &, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        std::string text;
        if (!prompt.rest_text(&text)) return fail("Soprano's prompt takes no markers");
        hiddens_.clear();
        n_embd_ = 0;
        stop_reason_ = "max_frames";
        eos_ = vocab.eos();
        out->seqs = {{alm_chunk::of_tokens(vocab.tokenize(text, false, true))}};
        return true;
    }

    bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (tokens == nullptr || hiddens == nullptr) return fail("Soprano: observe needs the token and the hidden state");
        if (n_embd_ == 0) n_embd_ = audio_lm_codec_latent_dim(alm_);
        if (n_embd_ <= 0) return fail("Soprano: the codec reports no latent width");
        if (tokens[0] == eos_) {
            stop_reason_ = "eos";
            codec_log(CODEC_LOG_LEVEL_INFO, "soprano AR done: %zu frames, stop=eos\n",
                      hiddens_.size() / (size_t) n_embd_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1};
            return true;
        }
        hiddens_.insert(hiddens_.end(), hiddens, hiddens + n_embd_);
        *v = {CODEC_AUDIO_LM_FEED_TOKEN, true, true, true, -1};
        return true;
    }

    bool generation(alm_generation * out) override {
        const int32_t n = n_embd_ > 0 ? (int32_t) (hiddens_.size() / (size_t) n_embd_) : 0;
        if (n < 2) return fail("Soprano: the model produced no speech");
        out->latent   = true;
        out->n_frames = n;
        out->width    = n_embd_;
        out->latents  = hiddens_;
        return true;
    }
    // The Soprano decoder takes the backbone hiddens frame-major as they are.
    bool decode_generation(const alm_generation & g, audio_lm_audio_output * out) override {
        if (!g.latent || g.n_frames < 2 || g.width != n_embd_) return fail("not a Soprano generation");
        return audio_lm_decode_latents(alm_, g.latents.data(), g.n_frames, g.width, out) || fail_alm("decode");
    }

private:
    int32_t            eos_    = -1;
    int32_t            n_embd_ = 0;
    std::vector<float> hiddens_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_soprano(audio_lm_context * alm) {
    return std::make_unique<soprano_family>(alm);
}

}  // namespace codec_common
