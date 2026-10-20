// include/codec_audio_lm.h: the public C API.  Forwards to the model family
// (audio_lm_family.h) and owns what the API hands out.

#include "audio_lm_family.h"
#include "codec_audio_lm.h"
#include "codec_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

using namespace codec_common;

namespace {

// Frame-level streaming over the current generation: re-decodes [0, avail)
// once enough new frames exist (causal decoders need the whole left
// context), holds back the right edge the decoder may still revise, and
// cross-fades the joins.
class alm_streamer {
public:
    explicit alm_streamer(audio_lm_context * alm) : alm_(alm) {}

    // New final audio since the last call (possibly none).
    bool poll(int32_t chunk_ms, std::vector<float> * piece, int32_t * sr, int32_t * nch) {
        chunk_ms_ = chunk_ms;
        const int32_t avail = audio_lm_decodable_frames(alm_);
        if (spf_ == 0 ? avail < k_probe_frames : avail < committed_ + chunk_ + right_) return out(piece, sr, nch);
        return window(avail, false, piece) && out(piece, sr, nch);
    }

    bool finish(std::vector<float> * piece, int32_t * sr, int32_t * nch) {
        const int32_t avail = audio_lm_decodable_frames(alm_);
        if (avail > committed_ && !window(avail, true, piece)) return false;
        return out(piece, sr, nch);
    }

    const std::string & error() const { return error_; }

private:
    static constexpr int32_t k_probe_frames = 8;
    static constexpr double  k_right_s = 0.32;
    static constexpr double  k_xfade_s = 0.02;

    bool out(std::vector<float> *, int32_t * sr, int32_t * nch) const {
        *sr = sr_;
        *nch = nch_;
        return true;
    }

    bool window(int32_t avail, bool final, std::vector<float> * piece) {
        audio_lm_audio_output w;
        if (!audio_lm_decode_audio_range(alm_, 0, avail, &w)) {
            return fail(std::string("streaming decode failed: ") + audio_lm_last_error(alm_));
        }
        const int32_t nch = std::max(1, w.n_channels);
        const int64_t len = (int64_t) (w.pcm.size() / (size_t) nch);
        if (spf_ == 0) {
            spf_ = audio_lm_codec_hop(alm_);
            if (spf_ <= 0) return fail("streaming: the codec reports no hop size");
            sr_  = w.sample_rate;
            nch_ = nch;
            const double frames_per_s = (double) sr_ / spf_;
            chunk_ = std::max(1, (int32_t) std::lround(chunk_ms_ * frames_per_s / 1000.0));
            right_ = std::max(1, (int32_t) std::lround(k_right_s * frames_per_s));
            xfade_ = std::min((int32_t) std::lround(k_xfade_s * sr_), right_ * spf_);
        }
        if (nch != nch_ || std::llabs(len - (int64_t) avail * spf_) > spf_) {
            return fail("streaming: a window of " + std::to_string(avail) + " frames decoded to " +
                        std::to_string(len) + " samples, not about " + std::to_string((int64_t) avail * spf_));
        }
        const int32_t commit = final ? avail : std::max(committed_, avail - right_);
        const size_t  base   = (size_t) committed_ * spf_;
        if ((int64_t) base > len) return fail("streaming: decoded window too short");
        const size_t n_new = final ? (size_t) len - base : (size_t) (commit - committed_) * spf_;
        piece->assign(w.pcm.begin() + (ptrdiff_t) (base * nch_), w.pcm.begin() + (ptrdiff_t) ((base + n_new) * nch_));
        const size_t nx = std::min(pending_.size() / nch_, n_new);
        for (size_t i = 0; i < nx; ++i) {
            const float g = (float) (i + 1) / (float) (nx + 1);
            for (int32_t c = 0; c < nch_; ++c) {
                float & v = (*piece)[i * nch_ + c];
                v = g * v + (1.0f - g) * pending_[i * nch_ + c];
            }
        }
        const size_t tail = base + n_new;
        const size_t np = final ? 0 : std::min((size_t) xfade_, (size_t) len - tail);
        pending_.assign(w.pcm.begin() + (ptrdiff_t) (tail * nch_), w.pcm.begin() + (ptrdiff_t) ((tail + np) * nch_));
        committed_ = commit;
        return true;
    }

    bool fail(const std::string & e) { error_ = e; return false; }

    audio_lm_context * alm_;
    int32_t chunk_ms_ = 0;
    int32_t spf_ = 0, sr_ = 0, nch_ = 1;
    int32_t chunk_ = 0, right_ = 0, xfade_ = 0;
    int32_t committed_ = 0;
    std::vector<float> pending_;
    std::string error_;
};

bool fill_pcm(const float * data, size_t n_values, int32_t sr, int32_t nch, codec_pcm_buffer * out) {
    out->data = static_cast<float *>(std::malloc(std::max<size_t>(1, n_values) * sizeof(float)));
    if (out->data == nullptr) return false;
    if (n_values > 0) std::memcpy(out->data, data, n_values * sizeof(float));
    out->n_channels  = std::max(1, nch);
    out->n_samples   = (int32_t) (n_values / (size_t) out->n_channels);
    out->sample_rate = sr;
    return true;
}

}  // namespace

struct codec_audio_lm {
    audio_lm_context *          alm = nullptr;
    bool                        owns_alm = true;
    std::unique_ptr<alm_family> family;
    std::string                 error;

    alm_completion                    completion;
    std::vector<codec_audio_lm_chunk> chunks;   // views into `completion`
    std::unique_ptr<alm_streamer>     streamer;

    std::vector<audio_lm_speaker_entry> voices;
    std::vector<std::string>            segments;

    ~codec_audio_lm() {
        family.reset();
        if (alm && owns_alm) audio_lm_free(alm);
    }

    bool fail(const std::string & e) { error = e; return false; }
    bool fail_family() { return fail(family->error()); }
};

extern "C" {

struct codec_audio_lm_params codec_audio_lm_default_params(void) {
    codec_audio_lm_params p{};
    return p;
}

struct codec_audio_lm * codec_audio_lm_init(const struct codec_audio_lm_params * params, char * err, size_t err_size) {
    std::string e;
    auto ctx = std::unique_ptr<codec_audio_lm>(new (std::nothrow) codec_audio_lm());
    if (!ctx) {
        e = "out of memory";
    } else if (params == nullptr || params->codec_path == nullptr) {
        e = "codec_path is required";
    } else {
        audio_lm_params p;
        p.codec_path = params->codec_path;
        p.use_gpu    = params->use_gpu;
        p.n_threads  = params->n_threads;
        if (params->kv_overrides != nullptr && params->n_kv_overrides > 0) {
            p.kv_overrides.assign(params->kv_overrides, params->kv_overrides + params->n_kv_overrides);
        }
        ctx->alm = audio_lm_init(p, &e);
        if (ctx->alm != nullptr) {
            ctx->family = alm_make_family(ctx->alm, params->backbone_path ? params->backbone_path : "", &e);
        }
        if (ctx->family) return ctx.release();
    }
    if (err != nullptr && err_size > 0) std::snprintf(err, err_size, "%s", e.c_str());
    return nullptr;
}

void codec_audio_lm_free(struct codec_audio_lm * ctx) { delete ctx; }

const char * codec_audio_lm_last_error(const struct codec_audio_lm * ctx) {
    return ctx != nullptr ? ctx->error.c_str() : "null codec_audio_lm";
}

const char * codec_audio_lm_meta_str(const struct codec_audio_lm * ctx, const char * key) {
    return ctx != nullptr ? audio_lm_meta_str(ctx->alm, key) : nullptr;
}

void codec_audio_lm_get_caps(const struct codec_audio_lm * ctx, struct codec_audio_lm_caps * out) {
    if (ctx == nullptr || out == nullptr) return;
    ctx->family->caps(out);
}

int32_t codec_audio_lm_list_voices(struct codec_audio_lm * ctx, const char * speaker_file) {
    if (ctx == nullptr) return -1;
    ctx->voices.clear();
    if (speaker_file == nullptr || *speaker_file == '\0') {
        ctx->voices = audio_lm_list_codec_speakers(ctx->alm);
    } else if (!audio_lm_list_speaker_file(ctx->alm, speaker_file, &ctx->voices)) {
        ctx->fail(audio_lm_last_error(ctx->alm));
        return -1;
    }
    return (int32_t) ctx->voices.size();
}

bool codec_audio_lm_voice(const struct codec_audio_lm * ctx, int32_t i, const char ** name, const char ** ref_text,
                          bool * is_default) {
    if (ctx == nullptr || i < 0 || i >= (int32_t) ctx->voices.size()) return false;
    const auto & v = ctx->voices[(size_t) i];
    if (name)       *name       = v.name.c_str();
    if (ref_text)   *ref_text   = v.ref_text.c_str();
    if (is_default) *is_default = v.is_default;
    return true;
}

bool codec_audio_lm_backbone_requirements(struct codec_audio_lm * ctx, const struct codec_audio_lm_vocab * vocab,
                                          int32_t max_frames, struct codec_audio_lm_backbone_req * out) {
    if (ctx == nullptr || out == nullptr) return false;
    alm_vocab v{vocab};
    return ctx->family->backbone_requirements(v, max_frames, out) || ctx->fail_family();
}

struct codec_audio_lm_tts_voice codec_audio_lm_default_tts_voice(void) {
    codec_audio_lm_tts_voice v{};
    return v;
}

bool codec_audio_lm_tts_set_voice(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_voice * voice) {
    if (ctx == nullptr) return false;
    alm_tts_voice v;
    if (voice != nullptr) {
        if (voice->voice)        v.voice = voice->voice;
        if (voice->speaker_file) v.speaker_file = voice->speaker_file;
        if (voice->ref_pcm && voice->ref_n_samples > 0) {
            v.ref_pcm.assign(voice->ref_pcm, voice->ref_pcm + voice->ref_n_samples);
            v.ref_sample_rate = voice->ref_sample_rate;
        }
        if (voice->ref_text)     v.ref_text = voice->ref_text;
        if (voice->save_speaker) v.save_speaker = voice->save_speaker;
    }
    return ctx->family->set_voice(v) || ctx->fail_family();
}

struct codec_audio_lm_tts_request codec_audio_lm_default_tts_request(void) {
    codec_audio_lm_tts_request r{};
    r.temperature = r.top_p = r.min_p = r.repetition_penalty = -1.0f;
    r.lm_guidance = r.flow_guidance = -1.0f;
    r.top_k = r.min_frames = r.flow_steps = -1;
    r.max_frames = 0;
    return r;
}

bool codec_audio_lm_tts_prompt(struct codec_audio_lm * ctx, const struct codec_audio_lm_tts_request * req,
                               const struct codec_audio_lm_vocab * vocab) {
    if (ctx == nullptr) return false;
    if (req == nullptr || req->text == nullptr) return ctx->fail("text is required");
    audio_lm_reset(ctx->alm);
    ctx->streamer.reset();
    ctx->chunks.clear();
    alm_tts_request r;
    r.text = req->text;
    if (req->instruct) r.instruct = req->instruct;
    if (req->preset)   r.preset = req->preset;
    r.seed = req->seed;
    r.temperature = req->temperature;
    r.top_p = req->top_p;
    r.top_k = req->top_k;
    r.min_p = req->min_p;
    r.repetition_penalty = req->repetition_penalty;
    r.min_frames = req->min_frames;
    r.lm_guidance = req->lm_guidance;
    r.flow_guidance = req->flow_guidance;
    r.flow_steps = req->flow_steps;
    r.max_frames = req->max_frames;
    alm_vocab v{vocab};
    if (!ctx->family->tts_prompt(r, v, &ctx->completion)) return ctx->fail_family();
    if (r.max_frames > 0) ctx->completion.max_frames = r.max_frames;
    return true;
}

int32_t codec_audio_lm_tts_split(struct codec_audio_lm * ctx, const char * text, int32_t max_bytes) {
    if (ctx == nullptr || text == nullptr) return -1;
    ctx->segments = ctx->family->split(text, max_bytes);
    return (int32_t) ctx->segments.size();
}

const char * codec_audio_lm_tts_segment(const struct codec_audio_lm * ctx, int32_t i) {
    if (ctx == nullptr || i < 0 || i >= (int32_t) ctx->segments.size()) return nullptr;
    return ctx->segments[(size_t) i].c_str();
}

int32_t codec_audio_lm_tts_join(struct codec_audio_lm * ctx, struct codec_pcm_buffer * pcm, int32_t i, int32_t n) {
    if (ctx == nullptr || pcm == nullptr) return -1;
    const int32_t nch = std::max(1, pcm->n_channels);
    std::vector<float> v(pcm->data, pcm->data + (size_t) pcm->n_samples * (size_t) nch);
    const int32_t gap = ctx->family->join(v, pcm->sample_rate, i, n);
    if (v.size() != (size_t) pcm->n_samples * (size_t) nch) {
        float * d = static_cast<float *>(std::realloc(pcm->data, std::max<size_t>(1, v.size()) * sizeof(float)));
        if (d == nullptr) { ctx->fail("out of memory"); return -1; }
        pcm->data = d;
        pcm->n_samples = (int32_t) (v.size() / (size_t) nch);
    }
    if (!v.empty()) std::memcpy(pcm->data, v.data(), v.size() * sizeof(float));
    return gap;
}

bool codec_audio_lm_get_completion(struct codec_audio_lm * ctx, struct codec_audio_lm_completion * out) {
    if (ctx == nullptr || out == nullptr) return false;
    const alm_completion & c = ctx->completion;
    size_t n_chunks = 0;
    for (const auto & s : c.seqs) n_chunks = std::max(n_chunks, s.size());
    ctx->chunks.assign(c.seqs.size() * n_chunks, codec_audio_lm_chunk{});
    for (size_t s = 0; s < c.seqs.size(); ++s) {
        for (size_t i = 0; i < c.seqs[s].size(); ++i) {
            const alm_chunk & a = c.seqs[s][i];
            codec_audio_lm_chunk & d = ctx->chunks[s * n_chunks + i];
            d.type          = a.type;
            d.text          = a.type == CODEC_AUDIO_LM_CHUNK_TEXT ? a.text.c_str() : nullptr;
            d.add_special   = a.add_special;
            d.parse_special = a.parse_special;
            d.tokens        = a.tokens.empty() ? nullptr : a.tokens.data();
            d.embd          = a.embd.empty() ? nullptr : a.embd.data();
            d.n             = a.type == CODEC_AUDIO_LM_CHUNK_TOKENS ? (int32_t) a.tokens.size()
                            : a.type == CODEC_AUDIO_LM_CHUNK_EMBD   ? a.n_rows : 1;
        }
    }
    out->flow     = c.flow;
    out->n_seq    = (int32_t) c.seqs.size();
    out->n_embd   = c.n_embd;
    out->chunks   = ctx->chunks.empty() ? nullptr : ctx->chunks.data();
    out->n_chunks = (int32_t) n_chunks;
    out->sampler  = c.sampler;
    out->sampler.penalty_exempt   = c.penalty_exempt.empty() ? nullptr : c.penalty_exempt.data();
    out->sampler.n_penalty_exempt = (int32_t) c.penalty_exempt.size();
    out->sampler.samplers         = c.samplers.c_str();
    out->grammar  = c.grammar.c_str();
    out->allowed  = c.allowed.empty() ? nullptr : c.allowed.data();
    out->n_allowed = (int32_t) c.allowed.size();
    out->needs_embeddings    = c.needs_embeddings;
    out->needs_prompt_hidden = c.needs_prompt_hidden;
    out->sample_first        = c.sample_first;
    out->streamable          = c.streamable;
    out->max_frames          = c.max_frames;
    out->min_frames          = c.min_frames;
    out->stop_token          = c.stop_token;
    return true;
}

bool codec_audio_lm_prefill(struct codec_audio_lm * ctx, const float * hiddens, int32_t n_pos) {
    if (ctx == nullptr) return false;
    return ctx->family->prefill(hiddens, n_pos) || ctx->fail_family();
}

bool codec_audio_lm_observe(struct codec_audio_lm * ctx, const int32_t * tokens, const float * hiddens,
                            struct codec_audio_lm_verdict * out) {
    if (ctx == nullptr || out == nullptr) return false;
    return ctx->family->observe(tokens, hiddens, out) || ctx->fail_family();
}

const float * codec_audio_lm_next_embd(const struct codec_audio_lm * ctx, int32_t seq) {
    return ctx != nullptr ? ctx->family->next_embd(seq) : nullptr;
}

const char * codec_audio_lm_stop_reason(const struct codec_audio_lm * ctx) {
    return ctx != nullptr ? ctx->family->stop_reason().c_str() : "";
}

bool codec_audio_lm_generate_self_contained(struct codec_audio_lm * ctx) {
    if (ctx == nullptr) return false;
    return ctx->family->generate_self_contained() || ctx->fail_family();
}

bool codec_audio_lm_decode(struct codec_audio_lm * ctx, struct codec_pcm_buffer * out) {
    if (ctx == nullptr || out == nullptr) return false;
    audio_lm_audio_output pcm;
    if (!ctx->family->decode(&pcm)) return ctx->fail_family();
    return fill_pcm(pcm.pcm.data(), pcm.pcm.size(), pcm.sample_rate, pcm.n_channels, out) || ctx->fail("out of memory");
}

bool codec_audio_lm_stream_poll(struct codec_audio_lm * ctx, int32_t chunk_ms, struct codec_pcm_buffer * out) {
    if (ctx == nullptr || out == nullptr) return false;
    if (!ctx->streamer) ctx->streamer = std::make_unique<alm_streamer>(ctx->alm);
    std::vector<float> piece;
    int32_t sr = 0, nch = 1;
    if (!ctx->streamer->poll(chunk_ms, &piece, &sr, &nch)) return ctx->fail(ctx->streamer->error());
    return fill_pcm(piece.data(), piece.size(), sr, nch, out) || ctx->fail("out of memory");
}

bool codec_audio_lm_stream_finish(struct codec_audio_lm * ctx, struct codec_pcm_buffer * out) {
    if (ctx == nullptr || out == nullptr) return false;
    if (!ctx->streamer) ctx->streamer = std::make_unique<alm_streamer>(ctx->alm);
    std::vector<float> piece;
    int32_t sr = 0, nch = 1;
    if (!ctx->streamer->finish(&piece, &sr, &nch)) return ctx->fail(ctx->streamer->error());
    return fill_pcm(piece.data(), piece.size(), sr, nch, out) || ctx->fail("out of memory");
}

int32_t codec_audio_lm_decodable_frames(const struct codec_audio_lm * ctx) {
    return ctx != nullptr ? ctx->family->decodable_frames() : 0;
}

int32_t codec_audio_lm_hop(const struct codec_audio_lm * ctx) {
    return ctx != nullptr ? audio_lm_codec_hop(ctx->alm) : 0;
}

bool codec_audio_lm_decode_range(struct codec_audio_lm * ctx, int32_t frame_begin, int32_t frame_end,
                                 struct codec_pcm_buffer * out) {
    if (ctx == nullptr || out == nullptr) return false;
    audio_lm_audio_output pcm;
    if (!audio_lm_decode_audio_range(ctx->alm, frame_begin, frame_end, &pcm)) {
        return ctx->fail(audio_lm_last_error(ctx->alm));
    }
    return fill_pcm(pcm.pcm.data(), pcm.pcm.size(), pcm.sample_rate, pcm.n_channels, out) || ctx->fail("out of memory");
}

}  // extern "C"

namespace codec_common {

struct codec_audio_lm * alm_wrap_borrowed(audio_lm_context * alm, const std::string & backbone_path,
                                         std::string * err) {
    auto family = alm_make_family(alm, backbone_path, err);
    if (!family) return nullptr;
    auto * ctx = new codec_audio_lm();
    ctx->alm = alm;
    ctx->owns_alm = false;
    ctx->family = std::move(family);
    return ctx;
}

}  // namespace codec_common
