// tts_probe — TTS host driver for offline audio-quality verification.  Loads
// a backbone + codec pair through the same rn-llama path the app uses
// (initVocoder → getFormattedAudioCompletion → the completion loop → decode), and
// writes a mono 16-bit PCM WAV for an ASR / speaker-similarity check.
//
// Usage:
//   tts_probe --backbone LM.gguf --codec CODEC.gguf --text "..." \
//             [--n-predict N] [--n-min-predict N] [--threads N] [--out-wav PATH]
//
// Voices:
//   --voice NAME           builtin codec voice (or the speaker in --speaker-file)
//   --clone-audio WAV      registry speaker from reference audio (createSpeaker,
//                          as LlamaSpeaker does); --ref-text sets its transcript
//   --speaker-file PATH    registry speaker loaded from a speaker file
//   --save-speaker PATH    write the registry speaker to a speaker file
//   --kv-override K=T:V    codec GGUF metadata override (repeatable)
//   --instruct TEXT        how to speak (models with can_take_instruct)
//   --then TEXT            generate TEXT too, then decode both (the first one
//                          last, from its saved result) into OUT and OUT.then.wav
//
// --sampling JSON sets completion() sampling keys over the backbone GGUF's
// general.sampling.* ({"temperature":0.9,"top_k":50}); Chatterbox's guidance is
// --kv-override codec.lm.chatterbox.sampling.cfg_weight=float:0.
// Backbone: --ngl N, --cache-type T, --fa / --no-fa, --n-parallel N; --codec-gpu runs the
// codec on the GPU.  The completion runs with getFormattedAudioCompletion's
// completionParams, as the JS layer passes them, and the context's sampling;
// --seed N seeds it (the backbone's sampler and codec.cpp's).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-tts.h"
#include "common.h"
#include "sampling.h"
#include "utils/wav_io.h"
#include "nlohmann/json.hpp"

using namespace rnllama;
using json = nlohmann::ordered_json;

// completionParams onto the context (prompt, embedding, grammar and its
// Float32Array logit_bias), over the sampling the context already has: the
// backbone GGUF's general.sampling.* and --sampling.
static void apply_completion_params(llama_rn_context & ctx, const json & p, const std::vector<float> & logit_bias) {
    auto & sp = ctx.params.sampling;
    ctx.params.prompt    = p.value("prompt", std::string());
    ctx.params.embedding = p.value("embedding", false);
    llama_set_embeddings(ctx.ctx, ctx.params.embedding);
    sp.grammar = p.contains("grammar") ? common_grammar(COMMON_GRAMMAR_TYPE_USER, p["grammar"].get<std::string>())
                                       : common_grammar();
    sp.logit_bias.clear();
    for (size_t t = 0; t < logit_bias.size(); ++t) {
        if (logit_bias[t] != 0.0f) sp.logit_bias.push_back({(llama_token) t, logit_bias[t]});
    }
}

// --sampling JSON: completion() sampling keys onto the context.
static void apply_sampling(llama_rn_context & ctx, const json & p) {
    auto & sp = ctx.params.sampling;
    sp.temp               = p.value("temperature", sp.temp);
    sp.top_k              = p.value("top_k", sp.top_k);
    sp.top_p              = p.value("top_p", sp.top_p);
    sp.min_p              = p.value("min_p", sp.min_p);
    sp.penalty_last_n     = p.value("penalty_last_n", sp.penalty_last_n);
    sp.penalty_repeat     = p.value("penalty_repeat", sp.penalty_repeat);
    sp.penalty_freq       = p.value("penalty_freq", sp.penalty_freq);
    sp.penalty_present    = p.value("penalty_present", sp.penalty_present);
    sp.dry_multiplier     = p.value("dry_multiplier", sp.dry_multiplier);
    sp.dry_base           = p.value("dry_base", sp.dry_base);
    sp.dry_allowed_length = p.value("dry_allowed_length", sp.dry_allowed_length);
    sp.dry_penalty_last_n = p.value("dry_penalty_last_n", sp.dry_penalty_last_n);
    if (sp.penalty_last_n == -1) sp.penalty_last_n = (int32_t) llama_n_ctx(ctx.ctx);
    if (p.contains("samplers")) {
        std::vector<std::string> names;
        const std::string s = p["samplers"].get<std::string>();
        for (size_t at = 0; at <= s.size();) {
            const size_t e = std::min(s.find(';', at), s.size());
            if (e > at) names.push_back(s.substr(at, e - at));
            at = e + 1;
        }
        sp.samplers = common_sampler_types_from_names(names);
    }
}

// Minimal mono 16-bit PCM WAV writer (no libsndfile dep).
static bool write_wav_mono16(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t) pcm.size();
    const uint16_t channels = 1, bits = 16;
    const uint32_t byte_rate = (uint32_t) sample_rate * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t data_bytes = n * (bits / 8);
    const uint32_t riff_bytes = 36 + data_bytes;
    auto w32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    std::fwrite("RIFF", 1, 4, f); w32(riff_bytes); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(channels);
    w32((uint32_t) sample_rate); w32(byte_rate); w16(block_align); w16(bits);
    std::fwrite("data", 1, 4, f); w32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        float s = pcm[i];
        if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
        int16_t v = (int16_t) std::lround(s * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

// Mono float PCM from a 16-bit WAV.
static bool load_wav_mono(const std::string & path, std::vector<float> * pcm, int * sample_rate) {
    codec_example_wav_data wav;
    std::string werr;
    if (!codec_example_load_wav_pcm16(path.c_str(), &wav, &werr)) {
        std::fprintf(stderr, "[probe] failed to load %s: %s\n", path.c_str(), werr.c_str());
        return false;
    }
    const int32_t nch = wav.n_channels > 0 ? wav.n_channels : 1;
    const int32_t nframes = (int32_t) (wav.pcm_i16.size() / (size_t) nch);
    pcm->assign((size_t) nframes, 0.0f);
    for (int32_t i = 0; i < nframes; ++i) {
        float acc = 0.0f;
        for (int32_t c = 0; c < nch; ++c) acc += wav.pcm_i16[(size_t) i * (size_t) nch + (size_t) c] / 32768.0f;
        (*pcm)[(size_t) i] = acc / (float) nch;
    }
    *sample_rate = wav.sample_rate;
    std::printf("[probe] %s: %d frames @ %d Hz\n", path.c_str(), nframes, wav.sample_rate);
    return true;
}

int main(int argc, char ** argv) {
    std::string backbone_path;
    std::string codec_path;
    std::string text = "Hello world";
    int  n_predict = 0;      // 0: none (until the model stops or the context is full)
    int  n_min_predict = -1; // -1: the model's own minimum
    int  threads   = 8;
    uint32_t seed  = 0;      // 0: the model's default
    std::string sampling_json;   // completion() sampling keys over the GGUF's general.sampling.*
    std::string out_wav;
    std::string cache_type;  // e.g. "q8_0" to mirror the example app's TTS init
    bool force_fa = false;
    bool force_no_fa = false;
    int  n_gpu_layers = 0;
    int  n_parallel = 1;     // sequences the context decodes (2: Chatterbox's guidance pair)
    bool codec_gpu = false;
    std::string voice;
    std::string clone_audio_path;
    std::string ref_text;
    std::string instruct;
    std::string then_text;
    std::string speaker_file;
    std::string save_speaker;
    std::vector<std::string> kv_overrides;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0; };
        if      (is("--backbone")     && i + 1 < argc) backbone_path     = argv[++i];
        else if (is("--codec")        && i + 1 < argc) codec_path        = argv[++i];
        else if (is("--text")         && i + 1 < argc) text              = argv[++i];
        else if (is("--n-predict")    && i + 1 < argc) n_predict         = std::atoi(argv[++i]);
        else if (is("--n-min-predict") && i + 1 < argc) n_min_predict    = std::atoi(argv[++i]);
        else if (is("--threads")      && i + 1 < argc) threads           = std::atoi(argv[++i]);
        else if (is("--seed")         && i + 1 < argc) seed              = (uint32_t) std::atoll(argv[++i]);
        else if (is("--sampling")     && i + 1 < argc) sampling_json     = argv[++i];
        else if (is("--out-wav")      && i + 1 < argc) out_wav           = argv[++i];
        else if (is("--cache-type")   && i + 1 < argc) cache_type        = argv[++i];
        else if (is("--fa"))                           force_fa          = true;
        else if (is("--no-fa"))                        force_no_fa       = true;
        else if (is("--ngl")          && i + 1 < argc) n_gpu_layers      = std::atoi(argv[++i]);
        else if (is("--n-parallel")   && i + 1 < argc) n_parallel        = std::atoi(argv[++i]);
        else if (is("--codec-gpu"))                    codec_gpu         = true;
        else if (is("--voice")        && i + 1 < argc) voice             = argv[++i];
        else if (is("--clone-audio")  && i + 1 < argc) clone_audio_path  = argv[++i];
        else if (is("--ref-text")     && i + 1 < argc) ref_text          = argv[++i];
        else if (is("--instruct")     && i + 1 < argc) instruct          = argv[++i];
        else if (is("--then")         && i + 1 < argc) then_text         = argv[++i];
        else if (is("--speaker-file") && i + 1 < argc) speaker_file      = argv[++i];
        else if (is("--save-speaker") && i + 1 < argc) save_speaker      = argv[++i];
        else if (is("--kv-override")  && i + 1 < argc) kv_overrides.push_back(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if (backbone_path.empty() || codec_path.empty()) {
        std::fprintf(stderr,
            "usage: tts_probe --backbone LM.gguf --codec CODEC.gguf --text \"...\"\n"
            "                 [--voice NAME] [--clone-audio WAV [--ref-text T]] [--speaker-file PATH]\n"
            "                 [--n-predict N] [--n-min-predict N] [--threads N] [--out-wav PATH]\n");
        return 2;
    }
    for (const auto & p : { backbone_path, codec_path }) {
        if (!std::filesystem::exists(p)) {
            std::fprintf(stderr, "not found: %s\n", p.c_str());
            return 3;
        }
    }

    llama_rn_context ctx;
    common_params params;
    params.model.path = backbone_path;
    params.n_ctx = 4096;
    params.n_batch = 1024;
    params.cpuparams.n_threads = threads;
    params.n_gpu_layers = n_gpu_layers;
    params.n_parallel = n_parallel;
    params.no_kv_offload = (n_gpu_layers == 0);
    if (n_gpu_layers > 0) std::printf("[probe] n_gpu_layers = %d\n", n_gpu_layers);
    if (!cache_type.empty()) {
        params.cache_type_k = kv_cache_type_from_str(cache_type);
        params.cache_type_v = kv_cache_type_from_str(cache_type);
        std::printf("[probe] cache_type_k/v = %s\n", cache_type.c_str());
    }
    if (force_fa) params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (force_no_fa) params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    std::printf("[probe] loading backbone %s\n", backbone_path.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    if (!ctx.loadModel(params)) {
        std::fprintf(stderr, "loadModel failed\n");
        return 4;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[probe] backbone loaded in %.2fs\n", std::chrono::duration<double>(t1 - t0).count());

    std::printf("[probe] loading codec %s\n", codec_path.c_str());
    if (!ctx.initVocoder(codec_path, /*batch_size=*/-1, /*use_gpu=*/codec_gpu, kv_overrides)) {
        std::fprintf(stderr, "initVocoder failed\n");
        return 5;
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("[probe] codec loaded in %.2fs\n", std::chrono::duration<double>(t2 - t1).count());

    {
        const auto cap = ctx.tts_wrapper->getTTSCapabilities(&ctx);
        std::printf("[probe] tts type=%d prompt_kind=%s family=%s requires_phonemes=%d "
                    "can_speaker_from_audio=%d can_take_instruct=%d sample_rate=%d\n",
                    cap.type, cap.prompt_kind.c_str(), cap.family.c_str(), (int) cap.requires_phonemes,
                    (int) cap.can_speaker_from_audio, (int) cap.can_take_instruct, cap.sample_rate);
        for (const auto & v : cap.builtin_voices) {
            std::printf("[probe]   voice %s%s%s%s\n", v.name.c_str(), v.is_default ? " (default)" : "",
                        v.ref_text.empty() ? "" : " ref_text=", v.ref_text.c_str());
        }
    }

    // Registry speaker (the LlamaSpeaker path): from reference audio or a
    // speaker file.  Its id rides along like the JS layer passes it.
    int speaker_id = -1;
    try {
        if (!speaker_file.empty()) {
            for (const auto & v : ctx.tts_wrapper->listSpeakers(speaker_file)) {
                std::printf("[probe]   file speaker %s%s\n", v.name.c_str(), v.is_default ? " (default)" : "");
            }
            speaker_id = ctx.tts_wrapper->loadSpeaker(speaker_file, voice);
            voice.clear();
        } else if (!clone_audio_path.empty()) {
            std::vector<float> pcm;
            int sr = 0;
            if (!load_wav_mono(clone_audio_path, &pcm, &sr)) return 3;
            speaker_id = ctx.tts_wrapper->createSpeaker(&ctx, pcm, sr, ref_text, 0.5f, false, /*bake=*/false);
        }
        if (speaker_id >= 0) std::printf("[probe] registry speaker id=%d\n", speaker_id);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[probe] speaker setup failed: %s\n", e.what());
        return 3;
    }
    json speaker = voice.empty() ? json(nullptr) : json::object({{"codec_voice", voice}});

    // The same entry points the JS layer calls: arm, run the completion, keep
    // the result's decoder input (audio_tokens, or embeddings for latents).
    struct generation {
        std::vector<llama_token> tokens;
        std::vector<float> latents;
        int latent_dim = 0;
    };
    auto synthesize = [&](const std::string & t, generation * g) {
        try {
            if (!sampling_json.empty()) apply_sampling(ctx, json::parse(sampling_json));
            ctx.params.sampling.seed = seed > 0 ? seed : LLAMA_DEFAULT_SEED;
            // As JSIParams: -1 is the context size (llama.cpp's sampler would take it as off).
            if (ctx.params.sampling.penalty_last_n == -1) ctx.params.sampling.penalty_last_n = (int32_t) llama_n_ctx(ctx.ctx);
            const auto formatted = ctx.tts_wrapper->getFormattedAudioCompletion(&ctx, speaker, t, speaker_id, instruct);
            std::printf("[probe] flow=%s completionParams=%s\n", formatted.flow.c_str(),
                        formatted.completion_params.dump().substr(0, 400).c_str());
            ctx.completion->rewind();
            apply_completion_params(ctx, formatted.completion_params, formatted.logit_bias);
            // The completion's voice, as JSIParams sets it.
            ctx.tts_speaker_id = speaker_id;
            ctx.tts_voice = voice;
            ctx.params.n_predict = n_predict > 0 ? n_predict : -1;
            ctx.n_min_predict = n_min_predict;
            if (!ctx.completion->initSampling()) {
                std::fprintf(stderr, "initSampling failed\n");
                return false;
            }
            ctx.completion->beginCompletion();
            const auto t3 = std::chrono::steady_clock::now();
            ctx.completion->loadPrompt({});
            while (ctx.completion->has_next_token) (void) ctx.completion->doCompletion();
            ctx.completion->endCompletion();
            const auto t4 = std::chrono::steady_clock::now();
            // As the JSI completion result does.
            ctx.tts_wrapper->collectGeneration();
            g->tokens = ctx.tts_wrapper->audio_tokens;
            g->latents = ctx.tts_wrapper->audio_latents;
            g->latent_dim = ctx.tts_wrapper->audio_latent_dim;
            std::printf("\n[probe] === GENERATION ===\n");
            std::printf("  frames                : %d\n", ctx.tts_wrapper->n_frames);
            std::printf("  decoder input         : %s (%zu values)\n", g->latents.empty() ? "codes" : "latents",
                        g->latents.empty() ? g->tokens.size() : g->latents.size());
            std::printf("  STOP REASON           : %s\n", ctx.tts_wrapper->stop_reason.c_str());
            std::printf("  text head             : %s\n", ctx.completion->generated_text.substr(0, 120).c_str());
            std::printf("  total generation time : %.2fs\n", std::chrono::duration<double>(t4 - t3).count());
            return true;
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[probe] generation failed: %s\n", e.what());
            return false;
        }
    };
    auto decode = [&](const generation & g, std::vector<float> * pcm) {
        try {
            *pcm = g.latents.empty() ? ctx.tts_wrapper->decodeAudioTokens(&ctx, g.tokens)
                                     : ctx.tts_wrapper->decodeAudioEmbeddings(&ctx, g.latents, g.latent_dim);
            return true;
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[probe] decode failed: %s\n", e.what());
            return false;
        }
    };

    generation first;
    if (!synthesize(text, &first)) return 7;
    // --then: a second generation runs before the first one is decoded.
    generation second;
    if (!then_text.empty() && !synthesize(then_text, &second)) return 7;

    if (!save_speaker.empty() && speaker_id >= 0) {
        try {
            ctx.tts_wrapper->saveSpeaker(speaker_id, save_speaker, "");
            std::printf("[probe] speaker saved to %s\n", save_speaker.c_str());
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[probe] saveSpeaker failed: %s\n", e.what());
        }
    }

    if (out_wav.empty()) return 0;
    if (!then_text.empty()) {
        std::vector<float> pcm2;
        if (!decode(second, &pcm2)) return 8;
        const std::string path2 = out_wav + ".then.wav";
        write_wav_mono16(path2, pcm2, ctx.tts_wrapper->getAudioSampleRate());
        std::printf("[probe] --then audio: %s (%zu samples)\n", path2.c_str(), pcm2.size());
    }
    std::vector<float> pcm;
    if (!decode(first, &pcm)) return 8;

    const int sr = ctx.tts_wrapper->getAudioSampleRate();
    double sumsq = 0.0; bool has_nan = false; float peak = 0.0f;
    for (float s : pcm) {
        if (std::isnan(s) || std::isinf(s)) has_nan = true;
        sumsq += (double) s * s;
        peak = std::max(peak, std::fabs(s));
    }
    const double rms = pcm.empty() ? 0.0 : std::sqrt(sumsq / (double) pcm.size());
    if (!write_wav_mono16(out_wav, pcm, sr)) {
        std::fprintf(stderr, "[probe] failed to write %s\n", out_wav.c_str());
        return 9;
    }
    std::printf("\n[probe] === WAV ===\n");
    std::printf("  path        : %s\n", out_wav.c_str());
    std::printf("  sample_rate : %d Hz\n", sr);
    std::printf("  samples     : %zu\n", pcm.size());
    std::printf("  duration    : %.3f s\n", sr > 0 ? (double) pcm.size() / sr : 0.0);
    std::printf("  rms         : %.5f\n", rms);
    std::printf("  peak        : %.5f\n", peak);
    std::printf("  has_nan     : %d\n", (int) has_nan);
    return 0;
}
