// bluemagpie_probe — driver for measuring per-step BlueMagpie latency.
// Loads the Barbet backbone + AudioVAE codec via the same rn-llama API path
// the app uses on device, sets a short max n_predict, feeds a tiny prompt,
// and prints per-step timings from `tryContinuousAudioStep`.
//
// Usage:
//   bluemagpie_probe --backbone BARBET.gguf --codec CODEC.gguf [--text "hi"] [--n-predict 30] [--threads 8]
//
// Design: we want realistic numbers, so the code path is exactly what device
// runs — `getFormattedAudioCompletion` + the completion loop.  Timing lives
// inside `tryContinuousAudioStep` (LOG_INFO calls) so no extra probe hooks
// are needed in production code.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// Minimal mono 16-bit PCM WAV writer (no deps).
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

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-tts.h"
#include "codec_lm.h"
#include "common.h"

using namespace rnllama;

int main(int argc, char ** argv) {
    std::string backbone_path;
    std::string codec_path;
    std::string text = "Hello world";
    int n_predict = 30;
    int threads = 8;
    std::string out_wav;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0; };
        if      (is("--backbone")  && i + 1 < argc) backbone_path = argv[++i];
        else if (is("--codec")     && i + 1 < argc) codec_path    = argv[++i];
        else if (is("--text")      && i + 1 < argc) text          = argv[++i];
        else if (is("--n-predict") && i + 1 < argc) n_predict     = std::atoi(argv[++i]);
        else if (is("--threads")   && i + 1 < argc) threads       = std::atoi(argv[++i]);
        else if (is("--out-wav")   && i + 1 < argc) out_wav       = argv[++i];
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if (backbone_path.empty() || codec_path.empty()) {
        std::fprintf(stderr,
            "usage: bluemagpie_probe --backbone PATH --codec PATH [--text ...] [--n-predict N] [--threads N]\n");
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
    params.embedding = true;              // continuous flow requires embedding=true
    params.cpuparams.n_threads = threads;
    params.cpuparams.n_threads = threads;
    params.n_gpu_layers = 0;
    params.no_kv_offload = true;

    std::printf("[probe] loading backbone %s\n", backbone_path.c_str());
    const auto t0 = std::chrono::steady_clock::now();
    if (!ctx.loadModel(params)) {
        std::fprintf(stderr, "loadModel failed\n");
        return 4;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("[probe] backbone loaded in %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count());

    std::printf("[probe] loading codec %s\n", codec_path.c_str());
    if (!ctx.initVocoder(codec_path, /*batch_size=*/-1, /*use_gpu=*/false)) {
        std::fprintf(stderr, "initVocoder failed\n");
        return 5;
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::printf("[probe] codec loaded in %.2fs\n",
                std::chrono::duration<double>(t2 - t1).count());

    // Build the prompt exactly like the JS path does — through
    // getFormattedAudioCompletion so BlueMagpie's builder fires.
    const auto formatted = ctx.tts_wrapper->getFormattedAudioCompletion(
        &ctx, /*speaker_json=*/"", text);
    std::printf("[probe] flow=%s embedding=%d prompt=%.60s...\n",
                formatted.flow.c_str(),
                (int) formatted.embedding,
                formatted.prompt.c_str());

    // Prime completion params.
    ctx.params.prompt    = formatted.prompt;
    ctx.params.n_predict = n_predict;
    ctx.params.embedding = true;
    ctx.params.sampling.temp  = 0.7f;
    ctx.params.sampling.top_p = 0.9f;
    llama_set_embeddings(ctx.ctx, true);

    ctx.completion->rewind();
    if (!ctx.completion->initSampling()) {
        std::fprintf(stderr, "initSampling failed\n");
        return 6;
    }
    ctx.completion->loadPrompt({});
    ctx.completion->beginCompletion();

    // Patch geometry from the codec_lm: each doCompletion step emits ONE
    // patch of `patch_size` latent frames.  n_predict caps the number of
    // PATCHES (one per generation step) so the "plausible patch count"
    // success criterion maps directly onto the loop cap.
    const ::codec_lm_info * lm_info =
        ctx.tts_wrapper->codec_lm ? ::codec_lm_get_info(ctx.tts_wrapper->codec_lm) : nullptr;
    const int patch_size = (lm_info && lm_info->patch_size > 0) ? lm_info->patch_size : 4;

    const auto t3 = std::chrono::steady_clock::now();
    int frames = 0;
    int patches = 0;
    while (ctx.completion->has_next_token && patches < n_predict) {
        (void) ctx.completion->doCompletion();
        frames = ctx.tts_wrapper->audio_embedding_dim > 0
            ? (int) (ctx.tts_wrapper->audio_embeddings.size() /
                     (size_t) ctx.tts_wrapper->audio_embedding_dim)
            : 0;
        patches = frames / patch_size;
        if (ctx.tts_wrapper->audio_embeddings_done) break;
    }
    ctx.completion->endCompletion();
    const auto t4 = std::chrono::steady_clock::now();

    const bool stop_head = ctx.tts_wrapper->audio_embeddings_done;
    const bool hit_cap   = !stop_head && patches >= n_predict;

    const double gen_s = std::chrono::duration<double>(t4 - t3).count();
    std::printf("\n[probe] === SUMMARY ===\n");
    std::printf("  patch_size            : %d frames/patch\n", patch_size);
    std::printf("  patches produced      : %d\n", patches);
    std::printf("  frames produced       : %d\n", frames);
    std::printf("  stopped_on_stop_head  : %d\n", (int) stop_head);
    std::printf("  hit_n_predict_cap     : %d\n", (int) hit_cap);
    std::printf("  STOP REASON           : %s\n",
                stop_head ? "STOP-HEAD" : (hit_cap ? "N-PREDICT-CAP" : "OTHER"));
    std::printf("  total generation time : %.2fs\n", gen_s);
    if (patches > 0) {
        std::printf("  avg per-patch (step)  : %.2fms\n", 1000.0 * gen_s / patches);
        // Each patch = patch_size frames; BlueMagpie AudioVAE ~= 160ms/patch.
        const double audio_s = patches * 0.160;
        std::printf("  approx audio produced : %.2fs\n", audio_s);
        std::printf("  RT factor             : %.2fx\n", audio_s / gen_s);
    }

    // Decode accumulated latents to PCM and write a WAV.  The accumulator
    // `audio_embeddings` is frame-major [T, latent_dim] — exactly what
    // decodeAudioEmbeddings() feeds to codec_decode_quantized_representation
    // (the codec performs the [T,D]->[D,T] transpose internally).
    if (!out_wav.empty()) {
        if (ctx.tts_wrapper->audio_embeddings.empty() || ctx.tts_wrapper->audio_embedding_dim <= 0) {
            std::fprintf(stderr, "[probe] no latents accumulated; skipping wav\n");
            return 7;
        }
        const auto td0 = std::chrono::steady_clock::now();
        std::vector<float> pcm = ctx.tts_wrapper->decodeAudioEmbeddings(
            &ctx, ctx.tts_wrapper->audio_embeddings, ctx.tts_wrapper->audio_embedding_dim);
        const auto td1 = std::chrono::steady_clock::now();
        if (pcm.empty()) {
            std::fprintf(stderr, "[probe] decodeAudioEmbeddings returned empty\n");
            return 8;
        }
        const int sr = ctx.tts_wrapper->getAudioSampleRate();
        // Sanity: RMS + NaN check.
        double sumsq = 0.0; bool has_nan = false; float peak = 0.0f;
        for (float s : pcm) {
            if (std::isnan(s) || std::isinf(s)) has_nan = true;
            sumsq += (double) s * s;
            float a = std::fabs(s);
            if (a > peak) peak = a;
        }
        const double rms = std::sqrt(sumsq / (double) pcm.size());
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
        std::printf("  decode time : %.2fs\n",
                    std::chrono::duration<double>(td1 - td0).count());
    }

    return 0;
}
