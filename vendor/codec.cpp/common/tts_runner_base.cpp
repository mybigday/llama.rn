// tts_runner_base.cpp — the parts of the reference TTS runner that need no
// llama.cpp backbone: request / voice marshalling onto codec_audio_lm and the
// self-contained synthesize (Pocket-TTS FlowLM).  Lives in the always-built
// codec_common library, so a CODEC_TTS_BACKBONE=OFF tts-cli keeps the
// self-contained models.

#include "tts_runner.h"

#include "codec_audio_lm.h"
#include "codec_common.h"
#include "utils/wav_io.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace codec_common {

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

const std::vector<tts_sampling_option> & tts_sampling_options() {
    using S = tts_sampling;
    static const std::vector<tts_sampling_option> opts = {
        {"--temp",               "temperature",        TTS_SAMPLING_TEMP,               &S::temp,               nullptr},
        {"--top-k",              "top_k",              TTS_SAMPLING_TOP_K,              nullptr,                &S::top_k},
        {"--top-p",              "top_p",              TTS_SAMPLING_TOP_P,              &S::top_p,              nullptr},
        {"--min-p",              "min_p",              TTS_SAMPLING_MIN_P,              &S::min_p,              nullptr},
        {"--repeat-last-n",      "repeat_last_n",      TTS_SAMPLING_PENALTY_LAST_N,     nullptr,                &S::penalty_last_n},
        {"--repeat-penalty",     "repeat_penalty",     TTS_SAMPLING_PENALTY_REPEAT,     &S::penalty_repeat,     nullptr},
        {"--presence-penalty",   "presence_penalty",   TTS_SAMPLING_PENALTY_PRESENT,    &S::penalty_present,    nullptr},
        {"--frequency-penalty",  "frequency_penalty",  TTS_SAMPLING_PENALTY_FREQ,       &S::penalty_freq,       nullptr},
        {"--dry-multiplier",     "dry_multiplier",     TTS_SAMPLING_DRY_MULTIPLIER,     &S::dry_multiplier,     nullptr},
        {"--dry-base",           "dry_base",           TTS_SAMPLING_DRY_BASE,           &S::dry_base,           nullptr},
        {"--dry-allowed-length", "dry_allowed_length", TTS_SAMPLING_DRY_ALLOWED_LENGTH, nullptr,                &S::dry_allowed_length},
        {"--dry-penalty-last-n", "dry_penalty_last_n", TTS_SAMPLING_DRY_PENALTY_LAST_N, nullptr,                &S::dry_penalty_last_n},
    };
    return opts;
}

std::vector<float> downmix_mono(const float * in, size_t n_frames, int32_t n_channels) {
    if (n_channels <= 1) return std::vector<float>(in, in + n_frames);
    std::vector<float> out(n_frames);
    for (size_t t = 0; t < n_frames; ++t) {
        float s = 0.0f;
        for (int32_t c = 0; c < n_channels; ++c) s += in[t * (size_t) n_channels + (size_t) c];
        out[t] = s / (float) n_channels;
    }
    return out;
}

bool load_ref_audio(const std::string & path, std::vector<float> & ref_pcm, int32_t * out_n, int32_t * out_sr,
                    std::string * err) {
    if (path.empty()) return true;
    codec_util_wav_data w;
    std::string werr;
    if (!codec_util_load_wav(path.c_str(), &w, &werr)) {
        *err = "failed to load " + path + ": " + werr;
        return false;
    }
    const int32_t nch = w.n_channels > 0 ? w.n_channels : 1;
    ref_pcm = downmix_mono(w.pcm.data(), w.pcm.size() / (size_t) nch, nch);
    if (out_n)  *out_n  = (int32_t) ref_pcm.size();
    if (out_sr) *out_sr = w.sample_rate;
    return true;
}

codec_audio_lm_tts_request tts_runner_request(const tts_runner_params & a) {
    codec_audio_lm_tts_request req = codec_audio_lm_default_tts_request();
    req.text     = a.text.c_str();
    req.instruct = a.instruct.empty() ? nullptr : a.instruct.c_str();
    req.voice    = a.voice_spk;
    req.min_frames    = a.min_frames;
    req.max_frames    = a.max_frames;
    req.lm_guidance   = a.lm_guidance;
    req.flow_guidance = a.flow_guidance;
    req.flow_steps    = a.flow_steps;
    return req;
}

bool tts_runner_make_voice(codec_audio_lm * alm, const tts_runner_params & a, tts_runner_voice * out, double * ms,
                           std::string * err) {
    const auto t0 = std::chrono::steady_clock::now();
    codec_audio_lm_caps caps{};
    codec_audio_lm_get_caps(alm, &caps);
    std::vector<float> ref_pcm;
    int32_t ref_n = 0, ref_sr = 0;
    if (!load_ref_audio(a.ref_audio_path, ref_pcm, &ref_n, &ref_sr, err)) return false;
    const char * name = a.voice.empty() ? nullptr : a.voice.c_str();
    if (!a.speaker_file.empty()) {
        out->owned = codec_audio_lm_load_voice(alm, a.speaker_file.c_str(), name);
        if (out->owned == nullptr) { *err = codec_audio_lm_last_error(alm); return false; }
        codec_log(CODEC_LOG_LEVEL_INFO, "voice: loaded %s%s%s\n", a.speaker_file.c_str(), name ? " : " : "", name ? name : "");
    } else if (!ref_pcm.empty() && caps.takes_ref_audio) {
        out->owned = codec_audio_lm_create_voice(alm, ref_pcm.data(), ref_n, ref_sr,
                                                 a.ref_text.empty() ? nullptr : a.ref_text.c_str());
        if (out->owned == nullptr) { *err = codec_audio_lm_last_error(alm); return false; }
        codec_log(CODEC_LOG_LEVEL_INFO, "voice: cloned from reference audio (%d samples @ %d Hz)\n", ref_n, ref_sr);
    } else {
        if (!ref_pcm.empty()) {
            codec_log(CODEC_LOG_LEVEL_WARN, "voice: this model cannot clone from reference audio; using the %s voice\n",
                      name ? name : "default");
        }
        if (name != nullptr) {
            out->use = codec_audio_lm_get_voice(alm, name);
            if (out->use == nullptr) { *err = codec_audio_lm_last_error(alm); return false; }
            codec_log(CODEC_LOG_LEVEL_INFO, "voice: %s\n", name);
        } else {
            out->use = a.voice_spk;
        }
    }
    if (out->owned != nullptr) out->use = out->owned;
    if (!a.save_speaker.empty()) {
        const codec_speaker * v = out->use ? out->use : codec_audio_lm_get_voice(alm, nullptr);
        if (v == nullptr || !codec_audio_lm_save_voice(alm, v, a.save_speaker.c_str(), "saved")) {
            *err = std::string("save voice: ") + codec_audio_lm_last_error(alm);
            return false;
        }
        codec_log(CODEC_LOG_LEVEL_INFO, "voice: saved to %s\n", a.save_speaker.c_str());
    }
    if (ms) *ms = ms_since(t0);
    return true;
}

bool tts_runner_take_pcm(codec_pcm_buffer * buf, tts_runner_result * out) {
    out->pcm.assign(buf->data, buf->data + (size_t) buf->n_samples * (size_t) std::max(1, buf->n_channels));
    out->sample_rate = buf->sample_rate;
    out->n_channels  = std::max(1, buf->n_channels);
    codec_pcm_buffer_free(buf);
    return true;
}

bool tts_runner_generate_self_contained(codec_audio_lm * alm, const tts_runner_params & a, tts_runner_result * out) {
    const codec_audio_lm_tts_request req = tts_runner_request(a);
    codec_audio_lm_format f;
    codec_audio_lm_completion c;
    if (!codec_audio_lm_tts_format(alm, &req, nullptr, &f) || !codec_audio_lm_tts_build(alm, &req, f.prompt, nullptr, &c) ||
        !codec_audio_lm_set_seed(alm, a.sampling.seed) ||
        !codec_audio_lm_generate_self_contained(alm)) {
        out->error = codec_audio_lm_last_error(alm);
        return false;
    }
    codec_pcm_buffer buf{};
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = codec_audio_lm_decode(alm, &buf);
    out->decode_ms += ms_since(t0);
    if (!ok) {
        out->error = std::string("decode: ") + codec_audio_lm_last_error(alm);
        return false;
    }
    tts_runner_take_pcm(&buf, out);
    out->n_frames    = codec_audio_lm_decodable_frames(alm);
    out->stop_reason = codec_audio_lm_stop_reason(alm);
    return true;
}

codec_audio_lm * tts_runner_open_codec(const tts_runner_params & a, std::string * err) {
    codec_audio_lm_params cp = codec_audio_lm_default_params();
    cp.codec_path     = a.codec_path.c_str();
    cp.backbone_path  = a.backbone_path.empty() ? nullptr : a.backbone_path.c_str();
    cp.kv_overrides   = a.kv_overrides.empty() ? nullptr : a.kv_overrides.data();
    cp.n_kv_overrides = (int32_t) a.kv_overrides.size();
    cp.n_threads      = a.n_threads;
    cp.use_gpu        = a.use_gpu;
    char e[512] = {0};
    codec_audio_lm * alm = codec_audio_lm_init(&cp, e, sizeof(e));
    if (alm == nullptr) *err = e;
    return alm;
}

int tts_runner_synthesize_selfcontained(const tts_runner_params & a, tts_runner_result * out) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    codec_audio_lm * alm = tts_runner_open_codec(a, &err);
    if (alm == nullptr) {
        out->error = err;
        return 1;
    }
    codec_audio_lm_caps caps{};
    codec_audio_lm_get_caps(alm, &caps);
    if (caps.needs_backbone) {
        codec_audio_lm_free(alm);
        return 0;
    }
    tts_runner_voice voice;
    if (!tts_runner_make_voice(alm, a, &voice, &out->speaker_ms, &out->error)) {
        codec_audio_lm_free(alm);
        return 1;
    }
    tts_runner_params av = a;
    av.voice_spk = voice.use;
    tts_runner_generate_self_contained(alm, av, out);
    codec_audio_lm_free(alm);
    out->total_ms = ms_since(t0);
    return 1;
}

}  // namespace codec_common
