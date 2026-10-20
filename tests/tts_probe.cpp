// tts_probe — TTS host driver for offline audio-quality verification.  Loads
// a backbone + codec pair through the same rn-llama path the app uses
// (initVocoder → getFormattedAudioCompletion → the completion loop → decode), and
// writes a mono 16-bit PCM WAV for an ASR / speaker-similarity check.
//
// Usage:
//   tts_probe --backbone LM.gguf --codec CODEC.gguf --text "..." \
//             [--n-predict N] [--threads N] [--out-wav PATH]
//
// Voices:
//   --voice NAME           builtin codec voice (or the speaker in --speaker-file)
//   --clone-audio WAV      registry speaker from reference audio (createSpeaker,
//                          as LlamaSpeaker does); --ref-text sets its transcript
//   --speaker-file PATH    registry speaker loaded from a speaker file
//   --save-speaker PATH    write the registry speaker to a speaker file
//   --kv-override K=T:V    codec GGUF metadata override (repeatable)
//
// Backbone: --ngl N, --cache-type T, --fa / --no-fa, --n-parallel N; --codec-gpu runs the
// codec on the GPU.  Sampling is the model's own (codec_audio_lm); --seed N
// sets its seed.

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
#include "utils/wav_io.h"
#include "nlohmann/json.hpp"

using namespace rnllama;
using json = nlohmann::ordered_json;

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
    int  n_predict = 0;      // 0: the model's frame cap
    int  threads   = 8;
    uint32_t seed  = 0;      // 0: the model's default
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
    std::string speaker_file;
    std::string save_speaker;
    std::vector<std::string> kv_overrides;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0; };
        if      (is("--backbone")     && i + 1 < argc) backbone_path     = argv[++i];
        else if (is("--codec")        && i + 1 < argc) codec_path        = argv[++i];
        else if (is("--text")         && i + 1 < argc) text              = argv[++i];
        else if (is("--n-predict")    && i + 1 < argc) n_predict         = std::atoi(argv[++i]);
        else if (is("--threads")      && i + 1 < argc) threads           = std::atoi(argv[++i]);
        else if (is("--seed")         && i + 1 < argc) seed              = (uint32_t) std::atoll(argv[++i]);
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
        else if (is("--speaker-file") && i + 1 < argc) speaker_file      = argv[++i];
        else if (is("--save-speaker") && i + 1 < argc) save_speaker      = argv[++i];
        else if (is("--kv-override")  && i + 1 < argc) kv_overrides.push_back(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if (backbone_path.empty() || codec_path.empty()) {
        std::fprintf(stderr,
            "usage: tts_probe --backbone LM.gguf --codec CODEC.gguf --text \"...\"\n"
            "                 [--voice NAME] [--clone-audio WAV [--ref-text T]] [--speaker-file PATH]\n"
            "                 [--n-predict N] [--threads N] [--out-wav PATH]\n");
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
                    "can_speaker_from_audio=%d takes_instruct=%d sample_rate=%d\n",
                    cap.type, cap.prompt_kind.c_str(), cap.family.c_str(), (int) cap.requires_phonemes,
                    (int) cap.can_speaker_from_audio, (int) cap.takes_instruct, cap.sample_rate);
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

    // The same entry points the JS layer calls.
    try {
        const auto formatted = ctx.tts_wrapper->getFormattedAudioCompletion(&ctx, speaker, text, speaker_id);
        std::printf("[probe] flow=%s embedding=%d\n", formatted.flow.c_str(), (int) formatted.embedding);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[probe] getFormattedAudioCompletion failed: %s\n", e.what());
        return 6;
    }
    ctx.completion->rewind();
    ctx.params.n_predict = n_predict > 0 ? n_predict : -1;
    if (seed > 0) ctx.params.sampling.seed = seed;
    if (!ctx.completion->initSampling()) {
        std::fprintf(stderr, "initSampling failed\n");
        return 6;
    }
    ctx.completion->beginCompletion();
    const auto t3 = std::chrono::steady_clock::now();
    try {
        ctx.completion->loadPrompt({});
        while (ctx.completion->has_next_token) (void) ctx.completion->doCompletion();
        ctx.completion->endCompletion();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[probe] generation failed: %s\n", e.what());
        return 7;
    }
    const auto t4 = std::chrono::steady_clock::now();

    std::printf("\n[probe] === GENERATION ===\n");
    std::printf("  frames                : %zu\n", ctx.tts_wrapper->audio_tokens.size());
    std::printf("  STOP REASON           : %s\n", ctx.tts_wrapper->stop_reason.c_str());
    std::printf("  total generation time : %.2fs\n", std::chrono::duration<double>(t4 - t3).count());

    if (!save_speaker.empty() && speaker_id >= 0) {
        try {
            ctx.tts_wrapper->saveSpeaker(speaker_id, save_speaker, "");
            std::printf("[probe] speaker saved to %s\n", save_speaker.c_str());
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[probe] saveSpeaker failed: %s\n", e.what());
        }
    }

    if (out_wav.empty()) return 0;
    std::vector<float> pcm;
    try {
        pcm = ctx.tts_wrapper->decodeAudioTokens(&ctx, ctx.tts_wrapper->audio_tokens);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[probe] decode failed: %s\n", e.what());
        return 8;
    }

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
