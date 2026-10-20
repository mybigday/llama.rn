// Codebook families: the backbone hidden state drives codec_lm heads that
// emit one frame of n codebook codes per step, fed back as the next input
// embedding.  Residual depth-AR (CSM, Qwen3-TTS: a depth decoder fills all
// codebooks) and parallel heads with delay (MOSS-TTSD: codebook 0 is sampled
// from the backbone's own vocabulary under a grammar, the rest from heads).

#include "audio_lm_family.h"
#include "audio_lm_sampler.h"

#include <cstring>

namespace codec_common {
namespace {

class codebook_family : public alm_family {
public:
    using alm_family::alm_family;
    const char * name() const override { return "codebook"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = audio_lm_has_speaker_enc(alm_) || audio_lm_codec_can_speaker_from_audio(alm_);
        out->takes_instruct  = meta_true("codec.lm.qwen3tts.takes_instruct");
        out->needs_voice     = audio_lm_talker_has_projection(alm_) && !builtin_voice();
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        n_cb_   = audio_lm_n_codebook(alm_);
        frames_ = 0;
        stop_reason_ = "max_frames";
        talker_text_.clear();
        talker_step_ = 0;

        // Speaker conditioning from reference audio (x-vector prefix row).
        std::vector<float> speaker;
        if (audio_lm_has_speaker_enc(alm_) && !voice_.ref_pcm.empty()) {
            audio_lm_input in;
            in.text = req.text;
            in.ref_pcm = voice_.ref_pcm.data();
            in.ref_n_samples = (int32_t) voice_.ref_pcm.size();
            in.ref_sample_rate = voice_.ref_sample_rate;
            audio_lm_prompt sp;
            if (!audio_lm_build_prompt(alm_, in, &sp)) return fail_alm("speaker from reference audio");
            if (!sp.embeds_prefix.empty() && sp.embeds_prefix_hidden == hidden_) {
                speaker.assign(sp.embeds_prefix.begin(), sp.embeds_prefix.begin() + hidden_);
                codec_log(CODEC_LOG_LEVEL_INFO, "speaker: x-vector prefix rows=%d hidden=%d\n",
                          sp.embeds_prefix_rows, sp.embeds_prefix_hidden);
            }
        } else if (!voice_.ref_pcm.empty() && !audio_lm_has_speaker_enc(alm_)) {
            codec_log(CODEC_LOG_LEVEL_INFO, "note: reference audio given but the model has no speaker encoder; ignoring\n");
        }
        const bool talker = audio_lm_talker_has_projection(alm_);
        // The Qwen3-TTS Base model clones a voice: its reference refuses to
        // run without reference audio (CustomVoice codecs carry builtin voices).
        if (talker && !builtin_voice() && (int32_t) speaker.size() != hidden_) {
            return fail("this is a voice-clone model and needs reference audio; none was given or the "
                        "speaker encoder produced no x-vector");
        }

        audio_lm_set_uses_embed_override(alm_, true, 1);
        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden_;
        std::vector<alm_chunk> chunks;
        if (talker) {
            // Instruct-capable talkers (CustomVoice 1.7B) take the instruction
            // as a text-only user turn ahead of the assistant header.
            std::string header = "<|im_start|>assistant\n";
            if (!req.instruct.empty()) {
                if (meta_true("codec.lm.qwen3tts.takes_instruct")) {
                    header = "<|im_start|>user\n" + req.instruct + "<|im_end|>\n" + header;
                } else {
                    codec_log(CODEC_LOG_LEVEL_WARN, "note: the instruction is ignored (this model takes none)\n");
                }
            }
            const std::vector<int32_t> role = vocab.tokenize(header, false, true);
            talker_text_ = vocab.tokenize(req.text, false, false);
            if (talker_text_.empty()) return fail("empty text");
            const int32_t cap_rows = (int32_t) role.size() + 6 + 4;
            std::vector<float> prefix((size_t) cap_rows * hidden_);
            int32_t n_rows = 0, consumed = 0;
            const bool xv = (int32_t) speaker.size() == hidden_;
            if (!audio_lm_build_talker_prefix(alm_, role.data(), (int32_t) role.size(), talker_text_.data(),
                                              (int32_t) talker_text_.size(), xv ? speaker.data() : nullptr,
                                              xv ? hidden_ : 0, prefix.data(), cap_rows, &n_rows, &consumed)) {
                return fail_alm("prompt prefix");
            }
            prefix.resize((size_t) n_rows * hidden_);
            chunks.push_back(alm_chunk::of_embd(std::move(prefix), n_rows));
        } else {
            if ((int32_t) speaker.size() == hidden_) chunks.push_back(alm_chunk::of_embd(speaker, 1));
            std::string text = req.text;
            // MOSS-TTSD dialogue tags: the processor maps [S1]/[S2] to <speaker1>/<speaker2>.
            if (pi_.model_kind == audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY) {
                text = replace_all(text, "[S1]", "<speaker1>");
                text = replace_all(text, "[S2]", "<speaker2>");
            }
            const std::vector<int32_t> toks = vocab.tokenize(pi_.prompt_prefix + text + pi_.prompt_suffix,
                                                             pi_.add_bos, pi_.parse_special);
            if (toks.empty()) return fail("empty prompt after tokenize");
            codec_log(CODEC_LOG_LEVEL_INFO, "prompt: \"%s%s%s\" -> %zu tokens\n", pi_.prompt_prefix.c_str(),
                      req.text.c_str(), pi_.prompt_suffix.c_str(), toks.size());
            if (audio_lm_prompt_needs_composed_embd(alm_)) {
                // Merged text + speech cb0: each prompt row sums all codebook tables.
                std::vector<float> rows(toks.size() * (size_t) hidden_);
                for (size_t i = 0; i < toks.size(); ++i) {
                    if (!audio_lm_compose_prompt_embd(alm_, toks[i], rows.data() + i * hidden_, hidden_)) {
                        return fail_alm("prompt embedding");
                    }
                }
                chunks.push_back(alm_chunk::of_embd(std::move(rows), (int32_t) toks.size()));
            } else {
                chunks.push_back(alm_chunk::of_tokens(toks));
            }
        }
        out->seqs = {std::move(chunks)};

        // Defaults from the codec_lm metadata, request overrides on top; the
        // same settings drive the codec heads.
        out->sampler.temp  = pi_.default_temperature;
        out->sampler.top_p = pi_.default_top_p;
        out->sampler.top_k = pi_.default_top_k;
        req.apply(out->sampler);
        out->sampler.min_p = 0.0f;
        out->sampler.penalty_repeat = 1.0f;
        if (pi_.cb0_from_backbone) {
            // Codebook 0 is a backbone token: keep it to the speech range + end.
            out->grammar = tts_auto_grammar(pi_, req.text);
            if (out->grammar.empty() && pi_.audio_tok_offset >= 0) {
                out->allowed = {{pi_.audio_tok_offset, pi_.audio_tok_offset + pi_.audio_tok_count}};
                if (pi_.audio_tok_eos >= 0) out->allowed.push_back({pi_.audio_tok_eos, pi_.audio_tok_eos + 1});
            }
        }
        const codec_audio_lm_sampler & sp = out->sampler;
        head_.init(sp.seed, sp.temp, sp.top_k, sp.top_p, 0.0f, 1.0f, 0);
        out->needs_embeddings = true;
        out->sample_first     = pi_.cb0_from_backbone;
        out->streamable       = true;
        out->max_frames       = 512;
        return true;
    }

    bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        std::vector<int32_t> codes((size_t) n_cb_);
        if (pi_.cb0_from_backbone) {
            if (tokens == nullptr) return fail("observe needs the sampled codebook-0 token");
            if (!audio_lm_step_set_text_context(alm_, tokens[0])) return fail_alm("step");
            codes[0] = tokens[0];
        }
        if (!audio_lm_step_begin(alm_, hiddens, hidden_)) return fail_alm("step");
        for (int32_t cb = 0; cb < n_cb_; ++cb) {
            int32_t idx = 0, n = 0;
            const float * lg = audio_lm_step_logits(alm_, &idx, &n);
            if (lg == nullptr) return fail_alm("step logits");
            const int32_t code = (pi_.cb0_from_backbone && cb == 0) ? codes[0] : head_.sample(lg, n);
            if (!audio_lm_step_push_code(alm_, code)) return fail_alm("step");
        }
        if (!audio_lm_step_finish(alm_, codes.data(), n_cb_)) return fail_alm("step");
        if (audio_lm_observe_codes(alm_, codes.data(), n_cb_, hiddens, hidden_) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "eos_code_c0";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=eos_code_c0\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        ++frames_;
        int32_t dim = 0;
        const float * nb = audio_lm_get_next_embed(alm_, &dim);
        if (nb == nullptr || dim != hidden_) return fail("no next embedding");
        next_.assign(nb, nb + dim);
        if (audio_lm_talker_has_projection(alm_)) {
            // The text not in the prefix rides along, one token per step.
            std::vector<float> tt((size_t) hidden_);
            if (!audio_lm_talker_trailing_text_embd(alm_, talker_text_.data(), (int32_t) talker_text_.size(),
                                                    talker_step_, tt.data(), hidden_)) {
                return fail_alm("text lane");
            }
            for (int32_t i = 0; i < hidden_; ++i) next_[(size_t) i] += tt[(size_t) i];
            ++talker_step_;
        }
        *v = {CODEC_AUDIO_LM_FEED_EMBD, pi_.cb0_from_backbone, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

private:
    bool meta_true(const char * key) const {
        const char * v = audio_lm_meta_str(alm_, key);
        return v != nullptr && std::strcmp(v, "true") == 0;
    }
    bool builtin_voice() const {
        const codec_speaker * spk = audio_lm_decode_speaker(alm_);
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        return spk != nullptr && codec_speaker_get_slot(spk, "spk_id", &ty, ne, &data);
    }
    static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
        for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) s.replace(pos, from.size(), to);
        return s;
    }

    audio_lm_prompt_info pi_;
    int32_t hidden_ = 0, n_cb_ = 0, frames_ = 0;
    std::vector<int32_t> talker_text_;
    int32_t talker_step_ = 0;
    std::vector<float> next_;
    alm_head_sampler head_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_codebook(audio_lm_context * alm) { return std::make_unique<codebook_family>(alm); }

}  // namespace codec_common
