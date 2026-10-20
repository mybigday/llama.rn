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

bool load_ref_audio(const std::string & path, std::vector<float> & ref_pcm, int32_t * out_n, int32_t * out_sr,
                    std::string * err) {
    if (path.empty()) return true;
    codec_example_wav_data w;
    std::string werr;
    if (!codec_example_load_wav_pcm16(path.c_str(), &w, &werr)) {
        *err = "failed to load " + path + ": " + werr;
        return false;
    }
    const int32_t nch = w.n_channels > 0 ? w.n_channels : 1;
    const int32_t nframes = (int32_t) (w.pcm_i16.size() / (size_t) nch);
    ref_pcm.assign((size_t) nframes, 0.0f);
    for (int32_t i = 0; i < nframes; ++i) {
        float acc = 0.0f;
        for (int32_t c = 0; c < nch; ++c) acc += w.pcm_i16[(size_t) i * nch + c] / 32768.0f;
        ref_pcm[(size_t) i] = acc / (float) nch;
    }
    if (out_n)  *out_n  = (int32_t) ref_pcm.size();
    if (out_sr) *out_sr = w.sample_rate;
    return true;
}

codec_audio_lm_tts_request tts_runner_request(const tts_runner_params & a) {
    codec_audio_lm_tts_request req = codec_audio_lm_default_tts_request();
    req.text     = a.text.c_str();
    req.instruct = a.instruct.empty() ? nullptr : a.instruct.c_str();
    req.preset   = a.preset.empty() ? nullptr : a.preset.c_str();
    req.seed     = a.seed;
    if (a.has_temp)        req.temperature = a.temp;
    if (a.has_top_p)       req.top_p = a.top_p;
    if (a.has_top_k)       req.top_k = a.top_k;
    if (a.has_min_p)       req.min_p = a.min_p;
    if (a.has_rep_penalty) req.repetition_penalty = a.repetition_penalty;
    if (a.min_len > 0)     req.min_frames = a.min_len;
    if (a.has_cfg_weight)  req.lm_guidance = a.cfg_weight;
    req.flow_guidance = a.cfg;
    req.flow_steps    = a.timesteps;
    req.max_frames    = a.max_frames;
    return req;
}

bool tts_runner_set_voice(codec_audio_lm * alm, const tts_runner_params & a, double * ms, std::string * err) {
    // The voice the request names (the model default when it names none).
    codec_audio_lm_tts_voice v = codec_audio_lm_default_tts_voice();
    std::vector<float> ref_pcm;
    int32_t ref_n = 0, ref_sr = 0;
    if (!load_ref_audio(a.ref_audio_path, ref_pcm, &ref_n, &ref_sr, err)) return false;
    v.voice           = a.voice.empty() ? nullptr : a.voice.c_str();
    v.speaker_file    = a.speaker_file.empty() ? nullptr : a.speaker_file.c_str();
    v.ref_pcm         = ref_pcm.empty() ? nullptr : ref_pcm.data();
    v.ref_n_samples   = ref_n;
    v.ref_sample_rate = ref_sr;
    v.ref_text        = a.ref_text.empty() ? nullptr : a.ref_text.c_str();
    v.save_speaker    = a.save_speaker.empty() ? nullptr : a.save_speaker.c_str();
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = codec_audio_lm_tts_set_voice(alm, &v);
    if (ms) *ms = ms_since(t0);
    if (!ok) *err = codec_audio_lm_last_error(alm);
    return ok;
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
    if (!codec_audio_lm_tts_prompt(alm, &req, nullptr) || !codec_audio_lm_generate_self_contained(alm)) {
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
    if (!tts_runner_set_voice(alm, a, &out->speaker_ms, &out->error)) {
        codec_audio_lm_free(alm);
        return 1;
    }
    tts_runner_generate_self_contained(alm, a, out);
    codec_audio_lm_free(alm);
    out->total_ms = ms_since(t0);
    return 1;
}

}  // namespace codec_common
