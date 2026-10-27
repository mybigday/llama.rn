// Codebook families: the backbone hidden state drives codec_lm heads that
// emit one frame of n codebook codes per step, fed back as the next input
// embedding.  Residual depth-AR (CSM, Qwen3-TTS: a depth decoder fills all
// codebooks) and parallel heads with delay (MOSS-TTSD: codebook 0 is sampled
// from the backbone's own vocabulary under a grammar, the rest from heads).

#include "audio_lm_family.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace codec_common {
namespace {

class codebook_family : public alm_family {
public:
    using alm_family::alm_family;
    const char * name() const override { return "codebook"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = audio_lm_codec_can_speaker_from_audio(alm_);
        out->takes_ref_text  = out->takes_ref_audio && audio_lm_talker_has_projection(alm_);
        out->takes_instruct  = meta_true("codec.lm.qwen3tts.takes_instruct");
        out->needs_voice     = audio_lm_talker_has_projection(alm_) && !talker_voice(audio_lm_decode_speaker(alm_));
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : 24000;
    }

    // Prompts:
    //   Qwen3-TTS talker: [<|im_start|>user\n{instruct}<|im_end|>\n]<|im_start|>assistant\n<__voice__>
    //                     [{reference transcript}]<__codec:text__>{text}
    //     (<__voice__>: the codec control rows with the voice's speaker / language
    //     tags and x-vector; with a transcript, the reference frames follow the
    //     prefix and the transcript leads the text lane: in-context voice)
    //   others: [<__voice__> (an x-vector row)]{the GGUF's prefix}{text}{suffix} (BOS added)
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        if (!vocab.valid()) return fail("this model needs the backbone vocabulary");
        audio_lm_prompt_info pi;
        if (!audio_lm_get_prompt_info(alm_, &pi)) return fail_alm("prompt info");
        const int32_t hidden = audio_lm_hidden_dim(alm_);
        voice_slots v;
        if (!read_voice(req.voice, &v)) return false;
        const bool talker = audio_lm_talker_has_projection(alm_);
        if (talker) {
            // The Qwen3-TTS Base model clones a voice: its reference refuses to
            // run without one (CustomVoice codecs carry builtin voices).
            if (!talker_voice(req.voice)) {
                return fail("this is a voice-clone model and needs a voice (reference audio or a speaker file)");
            }
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
            out->prompt = header + CODEC_AUDIO_LM_MARKER_VOICE + (v.icl_codes.empty() ? "" : v.ref_text) +
                          alm_codec_marker("text") + req.text;
        } else {
            std::string text = req.text;
            // MOSS-TTSD dialogue tags: the processor maps [S1]/[S2] to <speaker1>/<speaker2>.
            if (pi.model_kind == audio_lm_prompt_info::KIND_PARALLEL_HEADS_DELAY) {
                text = replace_all(text, "[S1]", "<speaker1>");
                text = replace_all(text, "[S2]", "<speaker2>");
            }
            out->prompt = ((int32_t) v.xvector.size() == hidden ? CODEC_AUDIO_LM_MARKER_VOICE : "") + pi.prompt_prefix +
                          text + pi.prompt_suffix;
        }
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        d.n_embd = hidden;
        // The end code (eos_code_c0) waits for the request's min_frames frames:
        // masked in the codebook-0 head, or by the grammar when codebook 0 is a
        // backbone token.  The model's eos_min_step is a floor: before it the
        // codec takes the end code as an ordinary code (CSM's 0 is one).
        d.min_frames = std::max(pi.eos_code_c0 >= 0 ? pi.eos_min_step : 0, std::max(0, req.min_frames));
        // The host samples every codebook.  Codebook 0, when a codec head, is
        // its own group with the penalties (Qwen3-TTS: the talker's
        // repetition_penalty over the frames' first codes); the depth codebooks
        // share one group without them (the subtalker).
        if (pi.cb0_from_backbone) d.head_groups = {alm_head_group{false, {}, {}}};
        else                      d.head_groups = {alm_head_group{true, {}, {}}, alm_head_group{false, {}, {}}};
        if (pi.cb0_from_backbone) {
            // Codebook 0 is a backbone token: keep it to the speech range + end.
            if (req.text_known) d.grammar = tts_auto_grammar(pi, req.text, std::max(0, req.min_frames));
            if (d.grammar.empty() && pi.audio_tok_offset >= 0) {
                d.allowed = {{pi.audio_tok_offset, pi.audio_tok_offset + pi.audio_tok_count}};
                if (pi.audio_tok_eos >= 0) d.allowed.push_back({pi.audio_tok_eos, pi.audio_tok_eos + 1});
            }
        }
        d.needs_embeddings = true;
        d.sample_first     = pi.cb0_from_backbone;
        d.streamable       = true;
        d.max_frames       = 512;
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        if (!audio_lm_get_prompt_info(alm_, &pi_)) return fail_alm("prompt info");
        hidden_ = audio_lm_hidden_dim(alm_);
        n_cb_   = audio_lm_n_codebook(alm_);
        frames_ = 0;
        mask_end_until_ = std::max(0, req.min_frames);
        stop_reason_ = "max_frames";
        talker_text_.clear();
        talker_step_ = 0;
        voice_slots v;
        if (!read_voice(req.voice, &v)) return false;
        const std::vector<float> & speaker = v.xvector;
        audio_lm_set_uses_embed_override(alm_, true, 1);
        std::vector<alm_chunk> chunks;
        if (audio_lm_talker_has_projection(alm_)) {
            const std::string header = prompt.text();
            if (!prompt.voice()) return fail("the talker prompt has <__voice__> after the assistant header");
            const std::string ref_text = prompt.text();
            if (!prompt.codec("text")) return fail("the talker prompt has <__codec:text__> before the text");
            const std::vector<int32_t> role = vocab.tokenize(header, false, true);
            talker_text_ = vocab.tokenize(prompt.text(), false, false);
            if (talker_text_.empty()) return fail("empty text");
            // ICL (generate_icl_prompt, streaming text): the text lane is the
            // reference transcript followed by the text, and the codec lane
            // after codec_bos is the reference's frames; the text rides on.
            icl_codes_.clear();
            if (!ref_text.empty()) {
                if (v.icl_codes.empty()) return fail("a reference transcript in the prompt, but the voice has no reference frames");
                icl_codes_ = v.icl_codes;
            }
            const int32_t n_icl = (int32_t) icl_codes_.size() / std::max(1, n_cb_);
            if (n_icl > 0) {
                std::vector<int32_t> ids = vocab.tokenize(ref_text, false, false);
                ids.insert(ids.end(), talker_text_.begin(), talker_text_.end());
                talker_text_ = std::move(ids);
                codec_log(CODEC_LOG_LEVEL_INFO, "talker: in-context voice prompt, %d reference frames\n", n_icl);
            }
            const int32_t cap_rows = (int32_t) role.size() + 6 + 4 + n_icl;
            std::vector<float> prefix((size_t) cap_rows * hidden_);
            int32_t n_rows = 0, consumed = 0;
            const bool xv = (int32_t) speaker.size() == hidden_;
            if (!audio_lm_build_talker_prefix(alm_, role.data(), (int32_t) role.size(), talker_text_.data(),
                                              (int32_t) talker_text_.size(), xv ? speaker.data() : nullptr,
                                              xv ? hidden_ : 0, prefix.data(), cap_rows, &n_rows, &consumed)) {
                return fail_alm("prompt prefix");
            }
            std::vector<float> lane((size_t) hidden_);
            for (int32_t t = 0; t < n_icl; ++t, ++n_rows) {
                float * row = prefix.data() + (size_t) n_rows * hidden_;
                if (codec_lm_compose_audio_embd(audio_lm_get_lm(alm_), icl_codes_.data() + (size_t) t * n_cb_, row) !=
                        CODEC_STATUS_SUCCESS ||
                    !audio_lm_talker_trailing_text_embd(alm_, talker_text_.data(), (int32_t) talker_text_.size(), t,
                                                        lane.data(), hidden_)) {
                    return fail_alm("in-context prompt");
                }
                for (int32_t i = 0; i < hidden_; ++i) row[i] += lane[(size_t) i];
            }
            talker_step_ = n_icl;
            prefix.resize((size_t) n_rows * hidden_);
            chunks.push_back(alm_chunk::of_embd(std::move(prefix), n_rows));
        } else {
            if (prompt.voice()) {
                if ((int32_t) speaker.size() != hidden_) return fail("the prompt has <__voice__> but the voice has no x-vector row");
                chunks.push_back(alm_chunk::of_embd(speaker, 1));
            }
            std::string text;
            if (!prompt.rest_text(&text)) return fail("this model's prompt takes <__voice__> only at its start");
            const std::vector<int32_t> toks = vocab.tokenize(text, pi_.add_bos, pi_.parse_special);
            if (toks.empty()) return fail("empty prompt after tokenize");
            codec_log(CODEC_LOG_LEVEL_INFO, "prompt: \"%s\" -> %zu tokens\n", text.c_str(), toks.size());
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
        return true;
    }

    // One frame: observe() starts it, then one SAMPLE_HEAD per codec-side
    // codebook (head_token() resumes), then the frame's verdict.
    bool observe(const int32_t * tokens, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        if (pi_.cb0_from_backbone) {
            if (tokens == nullptr) return fail("observe needs the sampled codebook-0 token");
            if (!audio_lm_step_set_text_context(alm_, tokens[0])) return fail_alm("step");
            cb0_token_ = tokens[0];
        }
        if (!audio_lm_step_begin(alm_, hiddens, hidden_)) return fail_alm("step");
        hiddens_.assign(hiddens, hiddens + hidden_);
        cb_ = 0;
        return next_head(v);
    }

    const float * head_logits() const override { return logits_.data(); }

    bool head_token(int32_t id, codec_audio_lm_verdict * v) override {
        if (cb_ >= n_cb_) return fail("no codebook is waiting for a token");
        if (id < 0 || id >= (int32_t) logits_.size()) return fail("head token out of range");
        if (!audio_lm_step_push_code(alm_, id)) return fail_alm("step");
        ++cb_;
        return next_head(v);
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

private:
    // The next codebook to sample, or the frame's end.
    bool next_head(codec_audio_lm_verdict * v) {
        for (; cb_ < n_cb_; ++cb_) {
            int32_t idx = 0, n = 0;
            const float * lg = audio_lm_step_logits(alm_, &idx, &n);
            if (lg == nullptr) return fail_alm("step logits");
            if (pi_.cb0_from_backbone && cb_ == 0) {   // the backbone's token
                if (!audio_lm_step_push_code(alm_, cb0_token_)) return fail_alm("step");
                continue;
            }
            logits_.assign(lg, lg + n);
            if (cb_ == 0 && frames_ < mask_end_until_ && pi_.eos_code_c0 >= 0 && pi_.eos_code_c0 < n) {
                logits_[(size_t) pi_.eos_code_c0] = -INFINITY;   // not the end yet (min_frames)
            }
            const int32_t group = (!pi_.cb0_from_backbone && cb_ > 0) ? 1 : 0;   // head_groups above
            *v = {CODEC_AUDIO_LM_SAMPLE_HEAD, false, true, false, -1, group, n};
            return true;
        }
        return finish_frame(v);
    }

    bool finish_frame(codec_audio_lm_verdict * v) {
        std::vector<int32_t> codes((size_t) n_cb_);
        if (!audio_lm_step_finish(alm_, codes.data(), n_cb_)) return fail_alm("step");
        if (audio_lm_observe_codes(alm_, codes.data(), n_cb_, hiddens_.data(), hidden_) == OBSERVE_STOP) {
            const char * e = audio_lm_last_error(alm_);
            if (e && *e) return fail_alm("frame");
            stop_reason_ = "eos_code_c0";
            codec_log(CODEC_LOG_LEVEL_INFO, "AR loop done: %d frames, stop=eos_code_c0\n", frames_);
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
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
        *v = {CODEC_AUDIO_LM_FEED_EMBD, pi_.cb0_from_backbone, true, true, -1, 0, 0};
        return true;
    }


private:
    bool meta_true(const char * key) const {
        const char * v = audio_lm_meta_str(alm_, key);
        return v != nullptr && std::strcmp(v, "true") == 0;
    }
    // A talker voice: a CustomVoice builtin (spk_id) or a cloned one (xvector).
    bool talker_voice(const codec_speaker * spk) const {
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * data = nullptr;
        return spk != nullptr &&
               (codec_speaker_get_slot(spk, "spk_id", &ty, ne, &data) || codec_speaker_get_slot(spk, "xvector", &ty, ne, &data));
    }

public:
    // ICL voices decode the reference's codes ahead of the generated ones and
    // cut the reference's share of the audio (generate_voice_clone).
    bool decode_generation(const alm_generation & g, audio_lm_audio_output * out) override {
        std::vector<int32_t> ref;
        if (const codec_speaker * spk = audio_lm_decode_speaker(alm_)) {
            codec_speaker_slot_type ty;
            int64_t ne[4];
            const void * d = nullptr;
            const char * text = codec_speaker_ref_text(spk);
            if (!g.latent && audio_lm_talker_has_projection(alm_) && text != nullptr && *text &&
                codec_speaker_get_slot(spk, "ref_codes", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_I32 && ne[0] == g.width) {
                ref.assign(static_cast<const int32_t *>(d), static_cast<const int32_t *>(d) + ne[0] * ne[1]);
            }
        }
        if (ref.empty()) return alm_family::decode_generation(g, out);
        const int32_t n_ref = (int32_t) ref.size() / g.width;
        alm_generation all = g;
        all.codes.insert(all.codes.begin(), ref.begin(), ref.end());
        all.n_frames += n_ref;
        if (!alm_family::decode_generation(all, out)) return false;
        const size_t frame_len = out->pcm.size() / (size_t) std::max(1, out->n_channels);
        const size_t cut = (size_t) ((double) n_ref / all.n_frames * frame_len);
        out->pcm.erase(out->pcm.begin(), out->pcm.begin() + (ptrdiff_t) (cut * out->n_channels));
        return true;
    }

private:
    // A cloned Qwen3-TTS voice: the x-vector speaker row, and with the
    // reference's codes + transcript the in-context prompt (ICL).
    struct voice_slots {
        std::vector<float>   xvector;
        std::vector<int32_t> icl_codes;   // empty without a transcript (x_vector_only_mode)
        std::string          ref_text;
    };
    bool read_voice(const codec_speaker * spk, voice_slots * v) {
        *v = voice_slots();
        const int32_t n_cb = audio_lm_n_codebook(alm_);
        if (spk != nullptr) {
            codec_speaker_slot_type ty;
            int64_t ne[4];
            const void * d = nullptr;
            if (codec_speaker_get_slot(spk, "xvector", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_F32) {
                v->xvector.assign(static_cast<const float *>(d), static_cast<const float *>(d) + ne[0]);
            }
            if (codec_speaker_get_slot(spk, "ref_codes", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_I32 && ne[0] == n_cb) {
                v->icl_codes.assign(static_cast<const int32_t *>(d), static_cast<const int32_t *>(d) + ne[0] * ne[1]);
            }
            if (const char * t = codec_speaker_ref_text(spk)) v->ref_text = t;
        }
        if (!v->xvector.empty() && (int32_t) v->xvector.size() != audio_lm_hidden_dim(alm_)) {
            return fail("the voice's x-vector does not fit this model");
        }
        if (v->ref_text.empty()) v->icl_codes.clear();
        return true;
    }

    static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
        for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) s.replace(pos, from.size(), to);
        return s;
    }

    audio_lm_prompt_info pi_;
    int32_t hidden_ = 0, n_cb_ = 0, frames_ = 0, mask_end_until_ = 0;
    std::vector<int32_t> talker_text_;
    int32_t talker_step_ = 0;
    std::vector<int32_t> icl_codes_;   // reference frames [n, n_cb] of an ICL voice
    std::vector<float> next_;
    std::vector<float> hiddens_;   // the frame's backbone hidden (observe -> finish_frame)
    std::vector<float> logits_;    // the head waiting for the host's token
    int32_t cb_ = 0, cb0_token_ = -1;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_codebook(audio_lm_context * alm) { return std::make_unique<codebook_family>(alm); }

}  // namespace codec_common
