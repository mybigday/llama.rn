// C API over the tts_runner session (include/codec_tts.h).  Thin forwarding:
// converts the C structs to tts_runner_params and owns the returned strings.

#include "codec_tts.h"
#include "tts_runner.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

using namespace codec_common;

struct codec_tts {
    tts_runner_session * session = nullptr;
    tts_runner_params    load;            // load-time fields, reused per run
    std::string          error;
    tts_runner_info      info;
};

static void set_str(std::string & dst, const char * src) { dst = src != nullptr ? src : ""; }

// The bundled llama.cpp (and its own ggml) logs through codec_log as well,
// so one codec_log_set callback sees the whole pipeline.
static void llama_to_codec(enum ggml_log_level level, const char * text, void *) {
    codec_log((enum codec_log_level) level, "%s", text);
}

extern "C" {

struct codec_tts_params codec_tts_default_params(void) {
    codec_tts_params p{};
    return p;
}

struct codec_tts_request codec_tts_default_request(void) {
    codec_tts_request r{};
    r.temperature        = -1.0f;
    r.top_p              = -1.0f;
    r.top_k              = -1;
    r.min_p              = -1.0f;
    r.repetition_penalty = -1.0f;
    r.min_frames         = -1;
    r.lm_guidance        = -1.0f;
    r.flow_guidance      = -1.0f;
    r.flow_steps         = -1;
    return r;
}

struct codec_tts * codec_tts_load(const struct codec_tts_params * params, char * err, size_t err_size) {
    std::string e;
    codec_tts * t = nullptr;
    if (params == nullptr || params->codec_path == nullptr) {
        e = "codec_tts_load: codec_path is required";
    } else if ((t = new (std::nothrow) codec_tts()) == nullptr) {
        e = "codec_tts_load: out of memory";
    } else {
        llama_log_set(llama_to_codec, nullptr);
        tts_runner_params & p = t->load;
        p.codec_path = params->codec_path;
        set_str(p.backbone_path, params->backbone_path);
        if (params->kv_overrides != nullptr && params->n_kv_overrides > 0) {
            p.kv_overrides.assign(params->kv_overrides, params->kv_overrides + params->n_kv_overrides);
        }
        p.n_threads  = params->n_threads;
        p.use_gpu    = params->use_gpu;
        p.max_frames = params->max_frames;
        t->session = tts_runner_load(p, &e);
        if (t->session == nullptr) { delete t; t = nullptr; }
        else t->info = tts_runner_get_info(t->session);
    }
    if (t == nullptr && err != nullptr && err_size > 0) std::snprintf(err, err_size, "%s", e.c_str());
    return t;
}

void codec_tts_free(struct codec_tts * t) {
    if (t == nullptr) return;
    tts_runner_free(t->session);
    delete t;
}

bool codec_tts_run(struct codec_tts * t, const struct codec_tts_request * r,
                   struct codec_pcm_buffer * out_pcm, struct codec_tts_result * out_result) {
    if (t == nullptr) return false;
    if (r == nullptr || r->text == nullptr || out_pcm == nullptr) {
        t->error = "codec_tts_run: text and out_pcm are required";
        return false;
    }
    tts_runner_params p = t->load;
    p.text = r->text;
    set_str(p.voice, r->voice);
    set_str(p.speaker_file, r->speaker_file);
    set_str(p.ref_audio_path, r->ref_audio_path);
    set_str(p.ref_text, r->ref_text);
    set_str(p.save_speaker, r->save_speaker);
    set_str(p.instruct, r->instruct);
    set_str(p.preset, r->preset);
    set_str(p.grammar, r->grammar);
    p.seed = r->seed;
    if (r->temperature >= 0.0f)        { p.has_temp = true;        p.temp = r->temperature; }
    if (r->top_p >= 0.0f)              { p.has_top_p = true;       p.top_p = r->top_p; }
    if (r->top_k >= 0)                 { p.has_top_k = true;       p.top_k = r->top_k; }
    if (r->min_p >= 0.0f)              { p.has_min_p = true;       p.min_p = r->min_p; }
    if (r->repetition_penalty >= 0.0f) { p.has_rep_penalty = true; p.repetition_penalty = r->repetition_penalty; }
    if (r->lm_guidance >= 0.0f)        { p.has_cfg_weight = true;  p.cfg_weight = r->lm_guidance; }
    if (r->flow_guidance >= 0.0f) p.cfg = r->flow_guidance;
    if (r->flow_steps >= 0)       p.timesteps = r->flow_steps;
    p.min_len       = r->min_frames;
    p.segment_bytes = std::max(0, r->segment_bytes);
    if (r->on_audio != nullptr) {
        codec_tts_audio_callback cb = r->on_audio;
        void * ud = r->user_data;
        p.on_audio = [cb, ud](const float * pcm, int32_t n, int32_t sr, int32_t nch) { return cb(pcm, n, sr, nch, ud); };
        p.stream_chunk_ms = std::max(0, r->stream_chunk_ms);
    }

    tts_runner_result res;
    if (!tts_runner_run(t->session, p, &res)) {
        t->error = res.error.empty() ? "codec_tts_run failed" : res.error;
        return false;
    }
    out_pcm->data = static_cast<float *>(std::malloc(std::max<size_t>(1, res.pcm.size()) * sizeof(float)));
    if (out_pcm->data == nullptr) { t->error = "codec_tts_run: out of memory"; return false; }
    if (!res.pcm.empty()) std::memcpy(out_pcm->data, res.pcm.data(), res.pcm.size() * sizeof(float));
    out_pcm->n_channels  = res.n_channels;
    out_pcm->n_samples   = res.n_channels > 0 ? (int32_t) (res.pcm.size() / (size_t) res.n_channels) : 0;
    out_pcm->sample_rate = res.sample_rate;
    if (out_result != nullptr) {
        out_result->n_frames = res.n_frames;
        std::snprintf(out_result->stop_reason, sizeof(out_result->stop_reason), "%s", res.stop_reason.c_str());
        out_result->total_ms  = res.total_ms;
        out_result->decode_ms = res.decode_ms;
    }
    t->error.clear();
    return true;
}

const char * codec_tts_last_error(const struct codec_tts * t) {
    return t != nullptr ? t->error.c_str() : "null codec_tts";
}

int32_t codec_tts_voices(struct codec_tts * t, const char * speaker_file) {
    if (t == nullptr) return -1;
    t->info = tts_runner_get_info(t->session, speaker_file != nullptr ? speaker_file : "");
    if (!t->info.error.empty()) { t->error = t->info.error; return -1; }
    return (int32_t) t->info.voices.size();
}

const char * codec_tts_voice(const struct codec_tts * t, int32_t i) {
    if (t == nullptr || i < 0 || i >= (int32_t) t->info.voices.size()) return nullptr;
    return t->info.voices[(size_t) i].c_str();
}

const char * codec_tts_default_voice(const struct codec_tts * t) {
    return t != nullptr ? t->info.default_voice.c_str() : nullptr;
}

bool codec_tts_takes_instruct(const struct codec_tts * t) { return t != nullptr && t->info.takes_instruct; }

}  // extern "C"
