// CosyVoice (v1 300M / BreezyVoice, CosyVoice2, CosyVoice3).
//
// Prompt (cosyvoice/llm/llm.py inference):
//   v1:    [sos] [spk row] [text_encoder(prompt_text ++ text)] [task_id] [speech_emb(prompt tokens)]
//   v2/v3: [sos] [backbone embed(prompt_text ++ text)]          [task_id] [speech_emb(prompt tokens)]
// Modes follow CosyVoiceFrontEnd: zero-shot when the speaker carries prompt
// tokens and a transcript, cross-lingual when it has prompt tokens but no
// transcript (the LLM sees neither), instruct when an instruction is given
// (it replaces the prompt text, no prompt tokens), SFT for x-vector-only
// speakers.  The speech head is sampled codec-side with the reference's
// ras_sampling (nucleus top_p / top_k, then a full-distribution redraw when
// the pick repeats inside the last win_size tokens) or BreezyVoice's plain
// top-k with eos redraw; no temperature.  The flow decodes the whole token
// sequence at the end.

#include "audio_lm_family.h"
#include "lm/lm_models.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace codec_common {
namespace {

struct cosyvoice_sampler {
    std::mt19937 rng;

    static void softmax(const std::vector<float> & x, std::vector<double> * p) {
        double mx = -INFINITY;
        for (float v : x) mx = std::max(mx, (double) v);
        p->resize(x.size());
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) { (*p)[i] = std::isinf(x[i]) && x[i] < 0 ? 0.0 : std::exp(x[i] - mx); s += (*p)[i]; }
        for (double & v : *p) v /= s;
    }
    int32_t draw(const std::vector<double> & w, const std::vector<int32_t> & ids) {
        std::discrete_distribution<int32_t> d(w.begin(), w.end());
        return ids[(size_t) d(rng)];
    }
    // nucleus_sampling: sorted probs, keep while cum < top_p and count < top_k.
    int32_t nucleus(const std::vector<double> & p, float top_p, int32_t top_k) {
        std::vector<int32_t> order(p.size());
        for (size_t i = 0; i < p.size(); ++i) order[i] = (int32_t) i;
        const size_t k = std::min<size_t>((size_t) std::max(top_k, 1), order.size());
        std::partial_sort(order.begin(), order.begin() + (long) k, order.end(),
                          [&](int32_t a, int32_t b) { return p[(size_t) a] > p[(size_t) b] || (p[(size_t) a] == p[(size_t) b] && a < b); });
        std::vector<double> w;
        std::vector<int32_t> ids;
        double cum = 0.0;
        for (size_t i = 0; i < k && cum < top_p; ++i) {
            cum += p[(size_t) order[i]];
            w.push_back(p[(size_t) order[i]]);
            ids.push_back(order[i]);
        }
        return draw(w, ids);
    }
    int32_t full(const std::vector<double> & p) {
        std::vector<int32_t> ids(p.size());
        for (size_t i = 0; i < p.size(); ++i) ids[i] = (int32_t) i;
        return draw(p, ids);
    }
};

class cosyvoice_family : public alm_family {
public:
    cosyvoice_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_cosyvoice_info * ci)
        : alm_family(alm), lm_(lm), ci_(ci) {}
    const char * name() const override { return "cosyvoice"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = true;
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

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        const codec_speaker * spk = audio_lm_decode_speaker(alm_);
        if (spk == nullptr) return fail("CosyVoice needs a speaker (reference audio, a voice or a speaker file)");
        hidden_ = audio_lm_hidden_dim(alm_);
        decoded_.clear();
        stop_reason_ = "max_len";
        std::vector<int32_t> prompt_tokens;
        std::vector<float> xvec;
        {
            codec_speaker_slot_type ty;
            int64_t ne[4];
            const void * d = nullptr;
            if (codec_speaker_get_slot(spk, "prompt_token", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_I32) {
                const auto * v = static_cast<const int32_t *>(d);
                prompt_tokens.assign(v, v + ne[0]);
            }
            if (codec_speaker_get_slot(spk, "embedding", &ty, ne, &d) && ty == CODEC_SPEAKER_SLOT_F32) {
                const auto * v = static_cast<const float *>(d);
                xvec.assign(v, v + ne[0] * ne[1]);
            }
        }
        std::string ref_text = voice_.ref_text;
        if (ref_text.empty() && codec_speaker_ref_text(spk) != nullptr) ref_text = codec_speaker_ref_text(spk);
        const std::string prefix = ci_->prompt_prefix != nullptr ? ci_->prompt_prefix : "";

        // Mode -> (LLM prompt text, LLM prompt speech tokens, target text).
        std::string prompt_text, tts_text = req.text;
        std::vector<int32_t> llm_prompt_tokens;
        bool llm_xvec = true;   // v1: x-vector row in the LLM prompt
        const char * mode = "sft";
        if (!req.instruct.empty()) {
            // v1 frontend_instruct: instruct text as prompt text, no x-vector in
            // the LLM ("information leakage"); v2/v3 frontend_instruct2: instruct
            // text as prompt text, no prompt speech tokens.  The instruct text
            // ends with <|endofprompt|> (v3 also starts with the system prompt).
            mode = "instruct";
            prompt_text = req.instruct;
            llm_xvec = false;
            if (ci_->version != 1 && prompt_text.find("<|endofprompt|>") == std::string::npos) {
                if (!prefix.empty()) prompt_text = prefix.substr(0, prefix.find("<|endofprompt|>")) + " " + prompt_text;
                prompt_text += "<|endofprompt|>";
            }
        } else if (!prompt_tokens.empty() && !ref_text.empty()) {
            mode = "zero_shot";
            prompt_text = prefix + ref_text;
            llm_prompt_tokens = prompt_tokens;
        } else if (!prompt_tokens.empty()) {
            mode = "cross_lingual";
            tts_text = prefix + tts_text;
        } else if (!prefix.empty()) {
            tts_text = prefix + tts_text;
        }

        if (!vocab.valid()) return fail("CosyVoice needs the backbone vocabulary");
        const std::vector<int32_t> ptxt = prompt_text.empty() ? std::vector<int32_t>() : vocab.tokenize(prompt_text, false, true);
        const std::vector<int32_t> ttxt = vocab.tokenize(tts_text, false, true);
        if (ttxt.empty()) return fail("empty text");
        std::vector<int32_t> all_txt(ptxt);
        all_txt.insert(all_txt.end(), ttxt.begin(), ttxt.end());

        const int32_t text_len = (int32_t) ttxt.size();
        min_len_ = (int32_t) (text_len * ci_->min_token_text_ratio);
        max_len_ = (int32_t) (text_len * ci_->max_token_text_ratio);
        codec_log(CODEC_LOG_LEVEL_INFO, "cosyvoice v%d (%s): %zu prompt-text + %d text tokens, %zu prompt speech tokens, min_len=%d max_len=%d, %s\n",
                  ci_->version, mode, ptxt.size(), text_len, llm_prompt_tokens.size(), min_len_, max_len_,
                  ci_->ras_sampling ? "ras" : "top_k");

        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden_;
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
            const bool use_xvec = llm_xvec && !xvec.empty();
            if (codec_lm_cosyvoice_text_rows(lm_, all_txt.data(), (int32_t) all_txt.size(), use_xvec ? xvec.data() : nullptr,
                                             use_xvec ? (int32_t) xvec.size() : 0, rows.data(), cap, &n_rows) != CODEC_STATUS_SUCCESS) {
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

        greedy_ = req.temperature == 0.0f;
        top_k_  = req.top_k >= 0 ? req.top_k : ci_->top_k;
        top_p_  = req.top_p >= 0.0f ? req.top_p : ci_->top_p;
        smp_.rng.seed(req.seed);
        out->needs_embeddings = true;
        out->sample_first     = false;
        out->max_frames       = max_len_ + 1;   // the text-length bound below stops first
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden state");
        const int32_t step = (int32_t) decoded_.size();
        if (step >= max_len_) {
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        if (!audio_lm_step_begin(alm_, hiddens, hidden_)) return fail_alm("speech head");
        int32_t cb = 0, n = 0;
        const float * lg = audio_lm_step_logits(alm_, &cb, &n);
        if (lg == nullptr || n <= 0) return fail_alm("speech head");
        std::vector<float> logits(lg, lg + n);
        audio_lm_step_push_code(alm_, 0);
        int32_t dummy = 0;
        audio_lm_step_finish(alm_, &dummy, 1);

        const bool ignore_eos = step < min_len_;
        int32_t id = 0;
        if (greedy_) {
            if (ignore_eos) logits[(size_t) ci_->ignore_eos_id] = -INFINITY;
            id = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        } else if (ci_->ras_sampling) {
            if (ignore_eos) logits[(size_t) ci_->ignore_eos_id] = -INFINITY;
            cosyvoice_sampler::softmax(logits, &prob_);
            id = smp_.nucleus(prob_, top_p_, top_k_);
            const int32_t win = ci_->win_size;
            int32_t rep = 0;
            for (size_t i = decoded_.size() > (size_t) win ? decoded_.size() - (size_t) win : 0; i < decoded_.size(); ++i) {
                rep += decoded_[i] == id;
            }
            if ((float) rep >= (float) win * ci_->tau_r) {
                logits[(size_t) id] = -INFINITY;
                cosyvoice_sampler::softmax(logits, &prob_);
                id = smp_.full(prob_);
            }
        } else {
            // TransformerLM.sampling_ids: top-k multinomial, redraw while eos is ignored.
            cosyvoice_sampler::softmax(logits, &prob_);
            std::vector<int32_t> order(prob_.size());
            for (size_t i = 0; i < prob_.size(); ++i) order[i] = (int32_t) i;
            const size_t k = std::min<size_t>((size_t) std::max(top_k_, 1), order.size());
            std::partial_sort(order.begin(), order.begin() + (long) k, order.end(),
                              [&](int32_t x, int32_t y) { return prob_[(size_t) x] > prob_[(size_t) y]; });
            std::vector<int32_t> ids(order.begin(), order.begin() + (long) k);
            std::vector<double> w;
            for (int32_t i : ids) w.push_back(prob_[(size_t) i]);
            for (int32_t tries = 0; tries < 100; ++tries) {
                id = smp_.draw(w, ids);
                if (!ignore_eos || id != ci_->ignore_eos_id) break;
            }
        }
        if (id >= ci_->stop_begin && id < ci_->stop_end) {
            stop_reason_ = "eos";
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        decoded_.push_back(id);
        next_.assign((size_t) hidden_, 0.0f);
        if (codec_lm_cosyvoice_speech_embd(lm_, &id, 1, next_.data(), 1) != CODEC_STATUS_SUCCESS) return fail_lm("speech embedding");
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }

    bool decode(audio_lm_audio_output * out) override {
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
        if (codes.empty()) return fail("no speech codes generated");
        audio_lm_reset(alm_);
        if (!audio_lm_push_codes(alm_, codes.data(), (int32_t) codes.size(), 1)) return fail_alm("push codes");
        return audio_lm_decode_audio(alm_, out) || fail_alm("decode");
    }

private:
    bool fail_lm(const char * what) { return fail(std::string(what) + ": " + codec_lm_get_last_error(lm_)); }

    codec_lm * lm_;
    const codec_lm_cosyvoice_info * ci_;
    int32_t hidden_ = 0, min_len_ = 0, max_len_ = 0, top_k_ = 0;
    float top_p_ = 1.0f;
    bool greedy_ = false;
    cosyvoice_sampler smp_;
    std::vector<int32_t> decoded_;
    std::vector<double> prob_;
    std::vector<float> next_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_cosyvoice(audio_lm_context * alm, codec_lm * lm) {
    const codec_lm_cosyvoice_info * ci = codec_lm_cosyvoice_get_info(lm);
    return ci ? std::make_unique<cosyvoice_family>(alm, lm, ci) : nullptr;
}

}  // namespace codec_common
