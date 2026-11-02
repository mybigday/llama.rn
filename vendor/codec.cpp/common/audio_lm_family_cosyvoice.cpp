// CosyVoice (v1 300M / BreezyVoice, CosyVoice2, CosyVoice3).
//
// Prompt (cosyvoice/llm/llm.py inference):
//   v1:    [sos] [spk row] [text_encoder(prompt_text ++ text)] [task_id] [speech_emb(prompt tokens)]
//   v2/v3: [sos] [backbone embed(prompt_text ++ text)]          [task_id] [speech_emb(prompt tokens)]
// Modes follow CosyVoiceFrontEnd: zero-shot when the speaker carries prompt
// tokens and a transcript, cross-lingual when it has prompt tokens but no
// transcript (the LLM sees neither), instruct when an instruction is given
// (it replaces the prompt text, no prompt tokens), SFT for x-vector-only
// speakers.  The host samples the speech head (the model card approximates
// the reference's ras_sampling with top-k / top-p and a repetition penalty
// over its window); the end token is held back for the first min_len tokens.  The flow decodes the
// whole token sequence at the end.

#include "audio_lm_family.h"
#include "lm/lm_models.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace codec_common {
namespace {

class cosyvoice_family : public alm_family {
public:
    cosyvoice_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_cosyvoice_info * ci)
        : alm_family(alm), lm_(lm), ci_(ci) {}
    const char * name() const override { return "cosyvoice"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = audio_lm_codec_can_speaker_from_audio(alm_);
        out->takes_ref_text  = true;
        out->takes_instruct  = true;
        out->needs_voice     = true;
        const char * sr = audio_lm_meta_str(alm_, "codec.sample_rate");
        out->sample_rate = sr ? std::atoi(sr) : (ci_->version == 1 ? 22050 : 24000);
    }

    bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames, codec_audio_lm_backbone_req * out) override {
        alm_family::backbone_requirements(vocab, max_frames, out);
        // CosyVoice v1's relative-position term is a KQ bias, which the
        // flash-attention path does not take.
        audio_lm_prompt_info pi;
        if (audio_lm_get_prompt_info(alm_, &pi) && pi.host_arch == "cosyvoice") out->flash_attn = false;
        return true;
    }

    // Prompt (the mode chosen here, CosyVoiceFrontEnd):
    //   <__codec:sos__>[<__voice__> (v1: the x-vector row)]{prompt text}<__codec:text__>{text}
    //   <__codec:task__>[<__voice__> (zero-shot: the reference's speech tokens)]
    bool tts_format(const alm_tts_request & req, const alm_vocab & vocab, alm_format * out) override {
        const codec_speaker * spk = req.voice;
        if (spk == nullptr) return fail("CosyVoice needs a speaker (reference audio, a voice or a speaker file)");
        if (!vocab.valid()) return fail("CosyVoice needs the backbone vocabulary");
        const voice_slots v = read_voice(spk);
        const std::string ref_text = codec_speaker_ref_text(spk) != nullptr ? codec_speaker_ref_text(spk) : "";
        const std::string prefix = ci_->prompt_prefix != nullptr ? ci_->prompt_prefix : "";

        // Mode -> (LLM prompt text, LLM prompt speech tokens, target text).
        std::string prompt_text, tts_text = req.text;
        bool llm_tokens = false;
        bool llm_xvec = true;   // v1: x-vector row in the LLM prompt
        if (!req.instruct.empty()) {
            // v1 frontend_instruct: instruct text as prompt text, no x-vector in
            // the LLM ("information leakage"); v2/v3 frontend_instruct2: instruct
            // text as prompt text, no prompt speech tokens.  The instruct text
            // ends with <|endofprompt|> (v3 also starts with the system prompt).
            prompt_text = req.instruct;
            llm_xvec = false;
            if (ci_->version != 1 && prompt_text.find("<|endofprompt|>") == std::string::npos) {
                if (!prefix.empty()) prompt_text = prefix.substr(0, prefix.find("<|endofprompt|>")) + " " + prompt_text;
                prompt_text += "<|endofprompt|>";
            }
        } else if (!v.prompt_tokens.empty() && !ref_text.empty()) {   // zero-shot
            prompt_text = prefix + ref_text;
            llm_tokens = true;
        } else if (!v.prompt_tokens.empty()) {                        // cross-lingual
            tts_text = prefix + tts_text;
        } else if (!prefix.empty()) {                                 // SFT
            tts_text = prefix + tts_text;
        }
        const bool xvec_row = !ci_->text_in_backbone && llm_xvec && !v.xvec.empty();
        out->prompt = alm_codec_marker("sos") + (xvec_row ? CODEC_AUDIO_LM_MARKER_VOICE : "") + prompt_text +
                      alm_codec_marker("text") + tts_text + alm_codec_marker("task") +
                      (llm_tokens ? CODEC_AUDIO_LM_MARKER_VOICE : "");
        const int32_t text_len = (int32_t) vocab.tokenize(tts_text, false, true).size();
        if (text_len == 0 && req.text_known) return fail("empty text");
        alm_completion & d = out->desc;
        d.flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        d.n_embd = audio_lm_hidden_dim(alm_);
        d.head_groups = {alm_head_group{true, {}, {}}};   // the speech head, sampled by the host
        d.needs_embeddings = true;
        d.sample_first     = false;
        d.max_frames       = (int32_t) (text_len * ci_->max_token_text_ratio) + 1;   // the text-length bound stops first
        d.min_frames       = req.min_frames >= 0 ? req.min_frames : (int32_t) (text_len * ci_->min_token_text_ratio);
        return true;
    }

    bool tts_build(const alm_tts_request & req, alm_prompt & prompt, const alm_vocab & vocab, alm_completion * out) override {
        const codec_speaker * spk = req.voice;
        if (spk == nullptr) return fail("CosyVoice needs a speaker (reference audio, a voice or a speaker file)");
        hidden_ = audio_lm_hidden_dim(alm_);
        decoded_.clear();
        stop_reason_ = "max_len";
        const voice_slots v = read_voice(spk);
        if (!prompt.codec("sos")) return fail("CosyVoice's prompt starts with <__codec:sos__>");
        const bool use_xvec = prompt.voice();
        if (use_xvec && (ci_->text_in_backbone || v.xvec.empty())) return fail("this CosyVoice prompt takes no x-vector row");
        const std::string prompt_text = prompt.text();
        if (!prompt.codec("text")) return fail("CosyVoice's prompt has <__codec:text__> before the text");
        const std::string tts_text = prompt.text();
        if (!prompt.codec("task")) return fail("CosyVoice's prompt has <__codec:task__> after the text");
        std::vector<int32_t> llm_prompt_tokens;
        if (prompt.voice()) {
            if (v.prompt_tokens.empty()) return fail("the prompt has the reference's <__voice__> but the voice has no prompt tokens");
            llm_prompt_tokens = v.prompt_tokens;
        }

        const std::vector<int32_t> ptxt = prompt_text.empty() ? std::vector<int32_t>() : vocab.tokenize(prompt_text, false, true);
        const std::vector<int32_t> ttxt = vocab.tokenize(tts_text, false, true);
        if (ttxt.empty()) return fail("empty text");
        std::vector<int32_t> all_txt(ptxt);
        all_txt.insert(all_txt.end(), ttxt.begin(), ttxt.end());

        const int32_t text_len = (int32_t) ttxt.size();
        min_len_ = req.min_frames >= 0 ? req.min_frames : (int32_t) (text_len * ci_->min_token_text_ratio);
        max_len_ = (int32_t) (text_len * ci_->max_token_text_ratio);
        codec_log(CODEC_LOG_LEVEL_INFO, "cosyvoice v%d: %zu prompt-text + %d text tokens, %zu prompt speech tokens, min_len=%d max_len=%d\n",
                  ci_->version, ptxt.size(), text_len, llm_prompt_tokens.size(), min_len_, max_len_);

        std::vector<alm_chunk> chunks;
        std::vector<float> head((size_t) hidden_);   // sos [+ v1 spk row + text rows]
        if (codec_lm_cosyvoice_special_embd(lm_, 0, head.data(), hidden_) != CODEC_STATUS_SUCCESS) return fail_lm("sos row");
        if (ci_->text_in_backbone) {
            chunks.push_back(alm_chunk::of_embd(std::move(head), 1));
            chunks.push_back(alm_chunk::of_tokens(all_txt));
        } else {
            const int32_t cap = (int32_t) all_txt.size() + 1;
            std::vector<float> rows((size_t) cap * hidden_);
            int32_t n_rows = 0;
            if (codec_lm_cosyvoice_text_rows(lm_, all_txt.data(), (int32_t) all_txt.size(), use_xvec ? v.xvec.data() : nullptr,
                                             use_xvec ? (int32_t) v.xvec.size() : 0, rows.data(), cap, &n_rows) != CODEC_STATUS_SUCCESS) {
                return fail_lm("text rows");
            }
            head.insert(head.end(), rows.begin(), rows.begin() + (long) n_rows * hidden_);
            chunks.push_back(alm_chunk::of_embd(std::move(head), 1 + n_rows));
        }
        const int32_t n_tail = 1 + (int32_t) llm_prompt_tokens.size();
        std::vector<float> tail((size_t) n_tail * hidden_);
        if (codec_lm_cosyvoice_special_embd(lm_, 1, tail.data(), hidden_) != CODEC_STATUS_SUCCESS ||
            codec_lm_cosyvoice_speech_embd(lm_, llm_prompt_tokens.data(), (int32_t) llm_prompt_tokens.size(),
                                           tail.data() + hidden_, (int32_t) llm_prompt_tokens.size()) != CODEC_STATUS_SUCCESS) {
            return fail_lm("prompt rows");
        }
        chunks.push_back(alm_chunk::of_embd(std::move(tail), n_tail));
        out->seqs = {std::move(chunks)};
        out->max_frames = max_len_ + 1;   // the text-length bound stops first
        out->min_frames = min_len_;
        return true;
    }

    struct voice_slots {
        std::vector<int32_t> prompt_tokens;
        std::vector<float>   xvec;
    };
    static voice_slots read_voice(const codec_speaker * spk) {
        voice_slots v;
        codec_speaker_slot_type ty;
        int64_t ne[4];
        const void * d = nullptr;
        if (codec_speaker_get_slot(spk, "prompt_token", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_I32) {
            const auto * p = static_cast<const int32_t *>(d);
            v.prompt_tokens.assign(p, p + ne[0]);
        }
        if (codec_speaker_get_slot(spk, "embedding", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_F32) {
            const auto * p = static_cast<const float *>(d);
            v.xvec.assign(p, p + ne[0] * ne[1]);
        }
        return v;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        const int32_t step = (int32_t) decoded_.size();
        if (step >= max_len_) {
            stop_reason_ = "max_len";
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
            return true;
        }
        if (!audio_lm_step_begin(alm_, hiddens, hidden_)) return fail_alm("speech head");
        int32_t cb = 0, n = 0;
        const float * lg = audio_lm_step_logits(alm_, &cb, &n);
        if (lg == nullptr || n <= 0) return fail_alm("speech head");
        logits_.assign(lg, lg + n);
        audio_lm_step_push_code(alm_, 0);
        int32_t dummy = 0;
        audio_lm_step_finish(alm_, &dummy, 1);

        // The reference holds the end token back for the first min_len tokens.
        if (step < min_len_) logits_[(size_t) ci_->ignore_eos_id] = -INFINITY;
        *v = {CODEC_AUDIO_LM_SAMPLE_HEAD, false, true, false, -1, 0, n};
        return true;
    }

    const float * head_logits() const override { return logits_.data(); }

    bool head_token(int32_t id, codec_audio_lm_verdict * v) override {
        if (id < 0 || id >= (int32_t) logits_.size()) return fail("head token out of range");
        if (id >= ci_->stop_begin && id < ci_->stop_end) {
            stop_reason_ = "eos";
            *v = {CODEC_AUDIO_LM_STOP, false, false, false, -1, 0, 0};
            return true;
        }
        decoded_.push_back(id);
        next_.assign((size_t) hidden_, 0.0f);
        if (codec_lm_cosyvoice_speech_embd(lm_, &id, 1, next_.data(), 1) != CODEC_STATUS_SUCCESS) return fail_lm("speech embedding");
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, true, -1, 0, 0};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

    bool generation(alm_generation * out) override {
        // v3 llm_job: after more than max_silent_run consecutive silent/breath
        // tokens, further ones are dropped from the flow input (they were fed back).
        std::vector<int32_t> codes;
        int32_t run = 0;
        for (int32_t id : decoded_) {
            bool silent = false;
            for (int32_t i = 0; i < ci_->n_silent_tokens; ++i) silent = silent || ci_->silent_tokens[i] == id;
            run = silent ? run + 1 : 0;
            if (silent && ci_->max_silent_run > 0 && run > ci_->max_silent_run) continue;
            if (id >= 0 && id < ci_->speech_token_size) codes.push_back(id);
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "cosyvoice AR done: %zu tokens (%zu to the flow), stop=%s\n", decoded_.size(), codes.size(),
                  stop_reason_.c_str());
        return generation_from_codes(codes, out);
    }

private:
    bool fail_lm(const char * what) { return fail(std::string(what) + ": " + codec_lm_get_last_error(lm_)); }

    codec_lm * lm_;
    const codec_lm_cosyvoice_info * ci_;
    int32_t hidden_ = 0, min_len_ = 0, max_len_ = 0;
    std::vector<int32_t> decoded_;
    std::vector<float> next_, logits_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_cosyvoice(audio_lm_context * alm, codec_lm * lm) {
    const codec_lm_cosyvoice_info * ci = codec_lm_cosyvoice_get_info(lm);
    return ci ? std::make_unique<cosyvoice_family>(alm, lm, ci) : nullptr;
}

}  // namespace codec_common
