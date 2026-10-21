// Chatterbox (T3 + S3Gen): the T3 backbone is embeddings-in / hidden-out; the
// codec_lm speech head turns its hidden state into S3 speech-token logits.
// The Llama T3 runs classifier-free guidance as a second sequence whose text
// rows are zeroed (logits = cond + w * (cond - uncond)); Turbo (GPT-2 T3) has
// one sequence and tokenizes with the backbone's vocabulary.  Every step feeds
// speech_emb[code] + speech_pos_emb[step] to all sequences.  S3Gen decodes the
// speech tokens (Turbo appends its trailing silence tokens).

#include "audio_lm_family.h"
#include "lm/lm_models.h"
#include "audio_lm_sampler.h"
#include "tts_segment.h"

#include <cstring>

namespace codec_common {
namespace {

class chatterbox_family : public alm_family {
public:
    chatterbox_family(audio_lm_context * alm, codec_lm * lm, const codec_lm_chatterbox_info * ci)
        : alm_family(alm), lm_(lm), ci_(ci) {}
    const char * name() const override { return "chatterbox"; }

    void caps(codec_audio_lm_caps * out) const override {
        alm_family::caps(out);
        out->takes_ref_audio = audio_lm_codec_can_speaker_from_audio(alm_);
        out->sample_rate     = 24000;
    }

    bool backbone_requirements(const alm_vocab & vocab, int32_t max_frames, codec_audio_lm_backbone_req * out) override {
        alm_family::backbone_requirements(vocab, max_frames, out);
        out->n_seq_max = ci_->is_turbo ? 1 : 2;   // the guidance pair
        return true;
    }

    bool tts_prompt(const alm_tts_request & req, const alm_vocab & vocab, alm_completion * out) override {
        const bool turbo = ci_->is_turbo != 0;
        hidden_ = audio_lm_hidden_dim(alm_);
        codes_.clear();
        frames_ = 0;
        stop_reason_ = "max_frames";
        // Request overrides, then the GGUF's reference sampling (< 0 = unset),
        // then the Llama-T3 regime.  Turbo has no guidance pair.
        auto pick = [](float req_v, float meta, float fallback) { return req_v >= 0.0f ? req_v : (meta >= 0.0f ? meta : fallback); };
        cfg_ = turbo ? 0.0f : pick(req.lm_guidance, ci_->cfg_weight, 0.5f);
        const float temp  = pick(req.temperature, ci_->temperature, 0.8f);
        const float top_p = pick(req.top_p, ci_->top_p, 1.0f);
        const float min_p = pick(req.min_p, ci_->min_p, 0.05f);
        const float rep   = pick(req.repetition_penalty, ci_->repetition_penalty, 1.2f);
        const int32_t top_k = req.top_k >= 0 ? req.top_k : (ci_->top_k >= 0 ? ci_->top_k : 0);

        std::vector<int32_t> text_ids;
        if (ci_->backbone_tokenizer) {
            // Turbo: punc_norm on the codec side, then the backbone's GPT-2 BPE
            // (paralinguistic tags such as "[laugh]" are added tokens).
            if (!vocab.valid()) return fail("Chatterbox Turbo needs the backbone vocabulary");
            std::vector<char> normed(req.text.size() * 2 + 64);
            if (codec_lm_chatterbox_normalize_text(lm_, req.text.c_str(), normed.data(), (int32_t) normed.size()) !=
                CODEC_STATUS_SUCCESS) {
                return fail(std::string("text normalization: ") + codec_lm_get_last_error(lm_));
            }
            text_ids = vocab.tokenize(normed.data(), false, true);
        } else {
            text_ids.resize(req.text.size() + 64);
            int32_t n = 0;
            if (codec_lm_chatterbox_tokenize(lm_, req.text.c_str(), text_ids.data(), (int32_t) text_ids.size(), &n) !=
                CODEC_STATUS_SUCCESS) {
                return fail(std::string("tokenize: ") + codec_lm_get_last_error(lm_));
            }
            text_ids.resize((size_t) n);
        }
        const int32_t n_text = (int32_t) text_ids.size();
        codec_log(CODEC_LOG_LEVEL_INFO, "chatterbox%s: %d text tokens, cfg_weight=%.2f temp=%.2f top_k=%d min_p=%.2f top_p=%.2f rep=%.2f\n",
                  turbo ? " turbo" : "", n_text, cfg_, temp, top_k, min_p, top_p, rep);

        // A cloned voice carries T3's conditioning (VoiceEncoder embedding + cond
        // prompt tokens) in its speaker; builtin voices use the GGUF's builtin conds.
        const float * t3_emb = nullptr;
        const int32_t * t3_tok = nullptr;
        int32_t n_t3_emb = 0, n_t3_tok = 0;
        if (const codec_speaker * spk = audio_lm_decode_speaker(alm_)) {
            codec_speaker_slot_type ty_e, ty_t;
            int64_t ne_e[4], ne_t[4];
            const void * d_e = nullptr, * d_t = nullptr;
            if (codec_speaker_get_slot(spk, "t3_speaker_emb", &ty_e, ne_e, &d_e) && ty_e == CODEC_SPEAKER_SLOT_F32 &&
                codec_speaker_get_slot(spk, "t3_cond_tokens", &ty_t, ne_t, &d_t) && ty_t == CODEC_SPEAKER_SLOT_I32) {
                t3_emb = static_cast<const float *>(d_e);
                n_t3_emb = (int32_t) ne_e[0];
                t3_tok = static_cast<const int32_t *>(d_t);
                n_t3_tok = (int32_t) ne_t[0];
                codec_log(CODEC_LOG_LEVEL_INFO, "chatterbox: T3 conditioning from speaker %s (%d cond tokens)\n",
                          codec_speaker_name(spk), n_t3_tok);
            }
        }
        const int32_t seq_cap = std::max(ci_->cond_rows, 1 + n_t3_tok) + (n_text + 2) + 2;   // also bounds Turbo's single-BOS layout
        const int32_t n_seq_cap = cfg_ > 0.0f ? 2 : 1;
        std::vector<float> rows((size_t) seq_cap * n_seq_cap * hidden_);
        int32_t seq_len = 0, n_seq = 0;
        if (codec_lm_chatterbox_build_prompt(lm_, text_ids.data(), n_text, cfg_, t3_emb, n_t3_emb, t3_tok, n_t3_tok, nullptr,
                                             nullptr, 0, 0, rows.data(), seq_cap * n_seq_cap, &seq_len, &n_seq) !=
            CODEC_STATUS_SUCCESS) {
            return fail(std::string("prompt: ") + codec_lm_get_last_error(lm_));
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "chatterbox: prompt seq_len=%d n_seq=%d (%d rows total)\n", seq_len, n_seq, seq_len * n_seq);
        n_seq_ = n_seq;

        *out = alm_completion();
        out->flow   = CODEC_AUDIO_LM_FLOW_CODEC_HEAD;
        out->n_embd = hidden_;
        for (int32_t s = 0; s < n_seq; ++s) {
            const float * b = rows.data() + (size_t) s * seq_len * hidden_;
            out->seqs.push_back({alm_chunk::of_embd(std::vector<float>(b, b + (size_t) seq_len * hidden_), seq_len)});
        }
        // T3: penalties (whole history, seeded with the start-of-speech token as
        // sample_t3 does) -> temperature -> min-p -> top-p.  Turbo: the HF
        // warper order, temperature -> top-k -> top-p -> repetition penalty,
        // over the generated tokens only.
        const uint32_t seed = req.seed_or_default();
        if (turbo) {
            head_.init(seed, temp, top_k, top_p, 0.0f, rep, 0,
                       {alm_head_sampler::TEMPERATURE, alm_head_sampler::TOP_K, alm_head_sampler::TOP_P,
                        alm_head_sampler::PENALTIES});
        } else {
            head_.init(seed, temp, 0, top_p, min_p, rep, 0);
            head_.accept(ci_->start_speech_token);
        }
        out->needs_embeddings = true;
        out->sample_first     = false;
        out->max_frames       = 512;
        return true;
    }

    bool observe(const int32_t *, const float * hiddens, codec_audio_lm_verdict * v) override {
        if (hiddens == nullptr) return fail("observe needs the backbone hidden states");
        std::vector<float> cond, uncond;
        if (!speech_logits(hiddens, &cond)) return false;
        if (n_seq_ == 2 && !speech_logits(hiddens + hidden_, &uncond)) return false;
        const int32_t n = (int32_t) cond.size();
        std::vector<float> logits((size_t) n);
        for (int32_t i = 0; i < n; ++i) {
            logits[(size_t) i] = n_seq_ == 2 ? cond[(size_t) i] + cfg_ * (cond[(size_t) i] - uncond[(size_t) i]) : cond[(size_t) i];
        }
        const int32_t code = head_.sample(logits.data(), n);
        if (code == ci_->stop_speech_token) {
            stop_reason_ = "eos_code_c0";
            codec_log(CODEC_LOG_LEVEL_INFO, "chatterbox AR done: %d frames, %zu speech codes, stop=eos_code_c0\n", frames_,
                      codes_.size());
            *v = {CODEC_AUDIO_LM_STOP, false, false, -1};
            return true;
        }
        if (code < ci_->start_speech_token) codes_.push_back(code);
        ++frames_;
        next_.assign((size_t) hidden_, 0.0f);
        if (codec_lm_chatterbox_compose_speech_embd(lm_, code, frames_, next_.data(), hidden_) != CODEC_STATUS_SUCCESS) {
            return fail(std::string("speech embedding: ") + codec_lm_get_last_error(lm_));
        }
        *v = {CODEC_AUDIO_LM_FEED_EMBD, false, true, -1};
        return true;
    }

    const float * next_embd(int32_t) const override { return next_.data(); }   // the same row on every sequence

    bool generation(alm_generation * out) override {
        std::vector<int32_t> codes = codes_;
        // Turbo appends S3GEN_SIL x 3 before S3Gen (tts_turbo.py::generate).
        if (ci_->silence_token >= 0) codes.insert(codes.end(), (size_t) ci_->n_trailing_silence, ci_->silence_token);
        return generation_from_codes(codes, out);
    }

private:
    bool speech_logits(const float * h, std::vector<float> * out) {
        if (!audio_lm_step_begin(alm_, h, hidden_)) return fail_alm("speech head");
        int32_t cb = 0, n = 0;
        const float * lg = audio_lm_step_logits(alm_, &cb, &n);
        if (lg == nullptr || n <= 0) return fail_alm("speech head");
        out->assign(lg, lg + n);
        audio_lm_step_push_code(alm_, 0);
        int32_t dummy = 0;
        audio_lm_step_finish(alm_, &dummy, 1);
        return true;
    }

    codec_lm * lm_;
    const codec_lm_chatterbox_info * ci_;
    int32_t hidden_ = 0, n_seq_ = 1, frames_ = 0;
    float cfg_ = 0.0f;
    std::vector<int32_t> codes_;
    std::vector<float> next_;
    alm_head_sampler head_;
};

}  // namespace

std::unique_ptr<alm_family> alm_family_chatterbox(audio_lm_context * alm, codec_lm * lm) {
    const codec_lm_chatterbox_info * ci = codec_lm_chatterbox_get_info(lm);
    return ci ? std::make_unique<chatterbox_family>(alm, lm, ci) : nullptr;
}

}  // namespace codec_common
