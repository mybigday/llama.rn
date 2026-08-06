// tts_runner_flow.cpp — self-contained (no-backbone) Pocket-TTS FlowLM
// synthesize.  FlowLM (codec.lm.kind="flow_lm") is a continuous-latent AR
// model whose AR transformer, text LUT, LSD flow head and EOS head all
// live in the codec GGUF, so there is NO llama.cpp backbone — everything
// runs through the codec C API + the codec_lm_flow_* helpers.  This TU has
// no llama dependency, so it lives in the always-built codec_common library
// and keeps the FlowLM path available even in CODEC_TTS_BACKBONE=OFF builds.
//
// Ported verbatim from examples/tts-cli.cpp's run_flow_lm_synthesize; the
// only change is that it fills a tts_runner_result (PCM + stats) instead of
// writing the WAV — the caller marshals output.

#include "tts_runner.h"

#include "codec.h"
#include "codec_lm.h"
#include "utils/wav_io.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace codec_common {

namespace {

// prepare_text_prompt mirrors pocket_tts/models/tts_model.py:prepare_text_prompt
// (strip, collapse double spaces, uppercase first letter, ensure trailing
// punctuation).  Returns the guessed frames_after_eos (3 if <=4 words else 1).
int flow_prepare_text(std::string & text) {
    // strip
    size_t b = text.find_first_not_of(" \t\r\n");
    size_t e = text.find_last_not_of(" \t\r\n");
    text = (b == std::string::npos) ? "" : text.substr(b, e - b + 1);
    // replace newlines with spaces + collapse double spaces
    for (char & c : text) if (c == '\n' || c == '\r') c = ' ';
    for (size_t i = text.find("  "); i != std::string::npos; i = text.find("  ")) text.replace(i, 2, " ");
    if (text.empty()) return 3;
    // word count
    int words = 0; bool in_w = false;
    for (char c : text) { if (c == ' ') in_w = false; else if (!in_w) { in_w = true; ++words; } }
    const int frames_after_eos_guess = (words <= 4) ? 3 : 1;
    // uppercase first letter
    if (text[0] >= 'a' && text[0] <= 'z') text[0] = (char) (text[0] - 'a' + 'A');
    // ensure trailing punctuation
    const char last = text.back();
    const bool alnum = (last >= 'a' && last <= 'z') || (last >= 'A' && last <= 'Z') || (last >= '0' && last <= '9');
    if (alnum) text += '.';
    return frames_after_eos_guess;
}

}  // namespace

int tts_runner_synthesize_selfcontained(const tts_runner_params & a,
                                        tts_runner_result * out) {
    // Load the codec model + context (self-contained).
    codec_model_params mparams = codec_model_default_params();
    mparams.use_gpu   = a.use_gpu;
    mparams.n_threads = a.n_threads > 0 ? a.n_threads : 1;
    codec_model * model = codec_model_load_from_file(a.codec_path.c_str(), mparams);
    if (!model) { out->error = "flow_lm: model load failed"; return 1; }

    codec_lm * lm = codec_lm_create(model);
    if (!lm) {
        // Not an LM-bearing codec, or a different kind — let the caller decide.
        codec_model_free(model);
        return 0;
    }
    const codec_lm_flow_info * fi = codec_lm_flow_get_info(lm);
    if (!fi) { codec_lm_free(lm); codec_model_free(model); return 0; }   // not flow_lm

    codec_context * cctx = codec_init_from_model(model, codec_context_default_params());
    if (!cctx) { out->error = "flow_lm: ctx init failed"; codec_lm_free(lm); codec_model_free(model); return 1; }

    const int32_t ldim    = fi->ldim;
    const int32_t d_model = fi->d_model;

    // ── Text prep + tokenize ────────────────────────────────────────
    std::string text = a.text;
    const int32_t fae_guess = flow_prepare_text(text) + 2;   // reference adds +2
    std::vector<int32_t> ids(512);
    int32_t n_tok = 0;
    if (codec_lm_flow_tokenize(lm, text.c_str(), ids.data(), (int32_t) ids.size(), &n_tok) != CODEC_STATUS_SUCCESS || n_tok <= 0) {
        out->error = std::string("flow_lm: tokenize failed: ") + codec_lm_get_last_error(lm);
        codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
    }
    ids.resize(n_tok);
    std::printf("flow_lm: text=\"%s\" -> %d tokens; d_model=%d ldim=%d\n", text.c_str(), n_tok, d_model, ldim);

    // ── Optional voice conditioning (--ref-audio) ───────────────────
    std::vector<float> voice_rows;
    int32_t n_voice = 0;
    if (!a.ref_audio_path.empty()) {
        codec_example_wav_data wav;
        std::string werr;
        if (!codec_example_load_wav_pcm16(a.ref_audio_path.c_str(), &wav, &werr)) {
            out->error = "flow_lm: ref-audio load failed: " + werr;
            codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
        }
        // Mono F32 (average channels), keep native rate — pocket_mimi encodes at
        // 24 kHz; the CLI assumes the ref is already 24 kHz (as kyutai voices are).
        const int32_t ch = wav.n_channels > 0 ? wav.n_channels : 1;
        const int32_t ns = (int32_t) (wav.pcm_i16.size() / (size_t) ch);
        std::vector<float> pcm((size_t) ns);
        for (int32_t i = 0; i < ns; ++i) {
            float s = 0; for (int32_t c = 0; c < ch; ++c) s += wav.pcm_i16[(size_t) i * ch + c];
            pcm[(size_t) i] = s / (float) ch / 32768.0f;
        }
        codec_audio au; au.data = pcm.data(); au.n_samples = ns;
        au.sample_rate = wav.sample_rate; au.n_channels = 1; au.pcm_type = CODEC_PCM_TYPE_F32;
        codec_token_buffer tb; std::memset(&tb, 0, sizeof(tb));
        codec_latent_buffer lb; std::memset(&lb, 0, sizeof(lb));
        if (codec_encode_latent(cctx, &au, &tb, &lb, codec_encode_default_params()) != CODEC_STATUS_SUCCESS) {
            out->error = std::string("flow_lm: ref-audio encode failed: ") + codec_get_last_error(cctx);
            codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
        }
        n_voice = lb.n_frames;
        voice_rows.resize((size_t) n_voice * d_model);
        // lb.data is channel-major [ldim, n_voice] (data[d*n_voice + t]).
        if (codec_lm_flow_speaker_rows(lm, lb.data, n_voice, voice_rows.data(), n_voice) != CODEC_STATUS_SUCCESS) {
            out->error = std::string("flow_lm: speaker_rows failed: ") + codec_lm_get_last_error(lm);
            codec_latent_buffer_free(&lb);
            codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
        }
        codec_latent_buffer_free(&lb);
        std::printf("flow_lm: voice conditioning from %s -> %d rows\n", a.ref_audio_path.c_str(), n_voice);
    }

    // ── State + prefill ─────────────────────────────────────────────
    codec_lm_state * st = codec_lm_state_new(lm);
    if (!st) { out->error = "flow_lm: state_new failed"; codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1; }
    if (codec_lm_flow_prefill(st, ids.data(), n_tok, n_voice > 0 ? voice_rows.data() : nullptr, n_voice) != CODEC_STATUS_SUCCESS) {
        out->error = std::string("flow_lm: prefill failed: ") + codec_lm_state_get_last_error(st);
        codec_lm_state_free(st); codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
    }

    // frames_after_eos policy: fixed from GGUF if >= 0, else the word-count guess.
    const int32_t frames_after_eos = (fi->frames_after_eos >= 0) ? fi->frames_after_eos : fae_guess;
    // max_gen_len estimate mirrors _estimate_max_gen_len (frame_rate=12.5,
    // ~3 tok/s + 2s padding); allow override via --max-frames.
    int32_t max_gen = a.max_frames > 0 ? a.max_frames
                    : (int32_t) std::ceil(((double) n_tok / 3.0 + 2.0) * 12.5);
    if (max_gen < 8) max_gen = 8;

    // ── AR loop ─────────────────────────────────────────────────────
    // The CLI owns the CFM init-noise RNG (seeded from --seed) so runs are
    // reproducible and seed-controllable; noise ~ N(0, temperature).  Uses an
    // explicit Box-Muller transform on a uniform mt19937 stream rather than
    // std::normal_distribution (whose libstdc++ polar method gives poor early
    // draws — the FlowLM is very sensitive to the first frames' noise quality).
    std::seed_seq seq{ (uint32_t) a.seed, 0x9E3779B9u, 0x243F6A88u, 0xB7E15162u };
    std::mt19937 rng(seq);
    const float noise_std = std::sqrt(fi->temperature);
    auto gauss = [&]() -> float {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float u1 = u(rng), u2 = u(rng);
        if (u1 < 1e-12f) u1 = 1e-12f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.28318530718f * u2) * noise_std;
    };
    std::vector<float> noise(ldim);
    std::vector<float> latents;   // frame-major [n_frames * ldim] (denormalized)
    std::vector<float> lat(ldim, 0.0f);
    int32_t n_frames = 0, eos_step = -1;
    const char * stop_reason = "max_frames";
    for (int32_t step = 0; step < max_gen; ++step) {
        for (int32_t d = 0; d < ldim; ++d) noise[(size_t) d] = gauss();
        int32_t is_eos = 0; float eos_logit = 0.0f;
        if (codec_lm_flow_step(st, noise.data(), lat.data(), &eos_logit, &is_eos) != CODEC_STATUS_SUCCESS) {
            char buf[128]; std::snprintf(buf, sizeof(buf), "flow_lm: step %d failed: ", step);
            out->error = std::string(buf) + codec_lm_state_get_last_error(st);
            codec_lm_state_free(st); codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
        }
        // Optional min-length guard: ignore EOS for the first --min-len frames.
        // The FlowLM is a short-utterance model whose EOS head can fire on frame
        // 0/1 for certain prompts + noise streams; --min-len lets the host force
        // a minimum audio length (mirrors the continuous_latent_cfm min_len knob).
        // Default 0 = honour EOS immediately (reference behaviour).
        const int32_t min_len = (a.min_len >= 0) ? a.min_len : 0;
        if (is_eos && eos_step < 0 && step >= min_len) eos_step = step;
        if (eos_step >= 0 && step >= eos_step + frames_after_eos) { stop_reason = "eos_head"; break; }
        // Denormalize the latent for Mimi decode and accumulate (channel-major).
        std::vector<float> den(ldim);
        codec_lm_flow_denorm_latent(lm, lat.data(), den.data());
        latents.insert(latents.end(), den.begin(), den.end());
        ++n_frames;
    }
    std::printf("flow_lm: AR done: %d frames, eos_step=%d, stop=%s\n", n_frames, eos_step, stop_reason);
    codec_lm_state_free(st);

    if (n_frames == 0) {
        out->error = "flow_lm: no frames generated";
        codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
    }

    // ── Decode: latents [ldim, n_frames] channel-major -> PCM ───────
    // codec_decode_quantized_representation expects channel-major buffer[d*T+t];
    // we accumulated frame-major [t*ldim+d], so transpose.
    std::vector<float> lat_cm((size_t) ldim * n_frames);
    for (int32_t t = 0; t < n_frames; ++t)
        for (int32_t d = 0; d < ldim; ++d)
            lat_cm[(size_t) d * n_frames + t] = latents[(size_t) t * ldim + d];

    codec_pcm_buffer pcm; std::memset(&pcm, 0, sizeof(pcm));
    if (codec_decode_quantized_representation(cctx, lat_cm.data(), ldim, n_frames, &pcm, codec_decode_default_params()) != CODEC_STATUS_SUCCESS) {
        out->error = std::string("flow_lm: decode failed: ") + codec_get_last_error(cctx);
        codec_free(cctx); codec_lm_free(lm); codec_model_free(model); return 1;
    }
    out->pcm.assign(pcm.data, pcm.data + pcm.n_samples);
    out->sample_rate = pcm.sample_rate;
    out->n_channels  = 1;
    out->n_frames    = n_frames;
    out->stop_reason = stop_reason;
    codec_pcm_buffer_free(&pcm);
    codec_free(cctx); codec_lm_free(lm); codec_model_free(model);
    return 1;
}

}  // namespace codec_common
