// tts_probe — general-purpose TTS host driver for offline audio-quality
// verification.  Loads any of the supported backbone + codec pairs via the
// exact same rn-llama API path the app uses on device, runs the standard
// completion loop (which auto-dispatches to the appropriate flow — token or
// continuous_embd — via getFormattedAudioCompletion), decodes the accumulated
// audio, and writes a mono 16-bit PCM WAV suitable for feeding into a
// downstream ASR / speaker-similarity pipeline.
//
// Design rationale: `getFormattedAudioCompletion` already knows how to build
// the correct prompt for every family (OuteTTS, NeuTTS, Soprano, CSM,
// Qwen3-TTS, MOSS-*, Chatterbox, BlueMagpie).  The completion loop then
// hooks the codec_lm state machine per-step for codec_lm-AR / continuous
// flows.  So this probe is really just a thin driver that:
//   1. sets up the backbone context + vocoder (codec),
//   2. calls getFormattedAudioCompletion() with the supplied text + speaker,
//   3. runs `doCompletion()` until has_next_token=false,
//   4. calls decodeAudioTokens() OR decodeAudioEmbeddings() based on the
//      flow the wrapper reports, then writes a WAV.
//
// Speaker JSON: for families that need it (OuteTTS legacy / V0.3 word-block
// voices, NeuTTS phonemized voices, etc.) pass --speaker-json PATH pointing
// at a JSON blob matching what the JS layer would have resolved from
// tts-voices.ts.  For families that don't (BlueMagpie, Soprano, ...) omit
// the flag.
//
// Usage:
//   tts_probe --backbone LM.gguf --codec CODEC.gguf --text "..." \
//             [--speaker-json PATH] [--n-predict N] [--threads N] \
//             [--out-wav PATH]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-tts.h"
#include "codec_lm.h"
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

static std::string slurp(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    std::string backbone_path;
    std::string codec_path;
    std::string text = "Hello world";
    std::string speaker_json_path;
    std::string ref_audio_path;   // WAV for voice-clone models (Qwen3-TTS etc)
    int  n_predict = 300;   // upper bound; per-family stop conditions dominate
    int  threads   = 8;
    std::string out_wav;
    // Sampling defaults match the RN example app's OuteTTS/BlueMagpie flow.
    // Override per-family via CLI to match e.g. NeuTTS's (1.0, 50, 1.0).
    float temp  = 0.7f;
    float top_p = 0.9f;
    int   top_k = 0;
    uint32_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0; };
        if      (is("--backbone")     && i + 1 < argc) backbone_path     = argv[++i];
        else if (is("--codec")        && i + 1 < argc) codec_path        = argv[++i];
        else if (is("--text")         && i + 1 < argc) text              = argv[++i];
        else if (is("--speaker-json") && i + 1 < argc) speaker_json_path = argv[++i];
        else if (is("--ref-audio")    && i + 1 < argc) ref_audio_path    = argv[++i];
        else if (is("--n-predict")    && i + 1 < argc) n_predict         = std::atoi(argv[++i]);
        else if (is("--threads")      && i + 1 < argc) threads           = std::atoi(argv[++i]);
        else if (is("--out-wav")      && i + 1 < argc) out_wav           = argv[++i];
        else if (is("--temp")         && i + 1 < argc) temp              = (float) std::atof(argv[++i]);
        else if (is("--top-p")        && i + 1 < argc) top_p             = (float) std::atof(argv[++i]);
        else if (is("--top-k")        && i + 1 < argc) top_k             = std::atoi(argv[++i]);
        else if (is("--seed")         && i + 1 < argc) seed              = (uint32_t) std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if (backbone_path.empty() || codec_path.empty()) {
        std::fprintf(stderr,
            "usage: tts_probe --backbone LM.gguf --codec CODEC.gguf --text \"...\"\n"
            "                 [--speaker-json PATH] [--n-predict N] [--threads N] [--out-wav PATH]\n");
        return 2;
    }
    for (const auto & p : { backbone_path, codec_path }) {
        if (!std::filesystem::exists(p)) {
            std::fprintf(stderr, "not found: %s\n", p.c_str());
            return 3;
        }
    }

    std::string speaker_json;
    if (!speaker_json_path.empty()) {
        speaker_json = slurp(speaker_json_path);
        if (speaker_json.empty()) {
            std::fprintf(stderr, "empty or missing speaker json: %s\n", speaker_json_path.c_str());
            return 3;
        }
    }

    llama_rn_context ctx;
    common_params params;
    params.model.path = backbone_path;
    params.n_ctx = 4096;
    params.n_batch = 1024;
    params.embedding = true;  // continuous flow requires; token flow ignores
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

    // Voice-clone models (Qwen3-TTS / MOSS-TTSD / Chatterbox): if the caller
    // provided --ref-audio, load the WAV, encode a speaker embedding via
    // ctx.tts_wrapper->encodeSpeaker, and inject the resulting x_vector into
    // the speaker JSON so getFormattedAudioCompletion picks it up.  Without
    // this, Qwen3-TTS-0.6B-Base falls into a degenerate short-output mode
    // (voice-clone-only training).
    if (!ref_audio_path.empty()) {
        codec_example_wav_data wav;
        std::string werr;
        if (!codec_example_load_wav_pcm16(ref_audio_path.c_str(), &wav, &werr)) {
            std::fprintf(stderr, "[probe] failed to load ref-audio %s: %s\n",
                         ref_audio_path.c_str(), werr.c_str());
            return 3;
        }
        // Downmix + F32 convert.
        const int32_t nch = wav.n_channels > 0 ? wav.n_channels : 1;
        const int32_t nframes = (int32_t) (wav.pcm_i16.size() / (size_t) nch);
        std::vector<float> pcm((size_t) nframes, 0.0f);
        for (int32_t i = 0; i < nframes; ++i) {
            float acc = 0.0f;
            for (int32_t c = 0; c < nch; ++c) {
                acc += wav.pcm_i16[(size_t) i * (size_t) nch + (size_t) c] / 32768.0f;
            }
            pcm[(size_t) i] = acc / (float) nch;
        }
        std::printf("[probe] ref-audio %s: %d frames @ %d Hz\n",
                    ref_audio_path.c_str(), nframes, wav.sample_rate);
        rnllama::llama_rn_encode_speaker_options sp_opts;
        sp_opts.pcm = std::move(pcm);
        sp_opts.input_sample_rate = wav.sample_rate;
        auto sp = ctx.tts_wrapper->encodeSpeaker(&ctx, sp_opts);
        if (!sp.speaker_emb.empty()) {
            std::printf("[probe] speaker_emb: %d rows × %d hidden (%zu floats)\n",
                        sp.speaker_n_rows, sp.speaker_hidden_dim, sp.speaker_emb.size());
            // Merge x_vector into the speaker JSON so build_talker_prefix picks it up.
            json spk_json = speaker_json.empty() ? json::object() : json::parse(speaker_json);
            spk_json["x_vector"] = sp.speaker_emb;
            speaker_json = spk_json.dump();
        } else {
            std::fprintf(stderr, "[probe] encodeSpeaker produced empty embedding; ignoring\n");
        }
    }

    // Build prompt via the exact same entry point the JS layer uses.
    const auto formatted = ctx.tts_wrapper->getFormattedAudioCompletion(
        &ctx, speaker_json, text);
    std::printf("[probe] flow=%s embedding=%d prompt.len=%zu\n",
                formatted.flow.c_str(),
                (int) formatted.embedding,
                formatted.prompt.size());
    if (formatted.flow.empty()) {
        std::fprintf(stderr, "[probe] getFormattedAudioCompletion returned empty flow — unknown TTS type\n");
        return 6;
    }

    // Prime completion params.  Token flow uses standard sampling; continuous
    // and codec_lm-AR flows ignore the backbone sampler (their hooks sample
    // per-codebook internally), so these values are only load-bearing for
    // OuteTTS / NeuTTS / Soprano.
    ctx.params.prompt     = formatted.prompt;
    ctx.params.n_predict  = n_predict;
    ctx.params.embedding  = formatted.embedding;
    ctx.params.sampling.grammar = formatted.grammar;
    ctx.params.sampling.temp    = temp;
    ctx.params.sampling.top_p   = top_p;
    if (top_k > 0) ctx.params.sampling.top_k = top_k;
    if (seed > 0) ctx.params.sampling.seed   = seed;
    llama_set_embeddings(ctx.ctx, formatted.embedding);

    ctx.completion->rewind();
    if (!ctx.completion->initSampling()) {
        std::fprintf(stderr, "initSampling failed\n");
        return 6;
    }
    ctx.completion->loadPrompt({});
    ctx.completion->beginCompletion();

    const bool is_continuous = (formatted.flow == "continuous_embd");
    // Soprano-style profiles emit `flow="tokens"` but their decode reads
    // from `completion->embeddings` instead of `audio_tokens`.  Treat them
    // like the continuous flow at decode time.
    const bool capture_embeddings = ctx.tts_wrapper->shouldCaptureAudioEmbeddings(&ctx);
    const ::codec_lm_info * lm_info = ctx.tts_wrapper->codec_lm
        ? ::codec_lm_get_info(ctx.tts_wrapper->codec_lm)
        : nullptr;
    const int patch_size = (lm_info && lm_info->patch_size > 0) ? lm_info->patch_size : 4;

    // Run the completion loop end-to-end.
    const auto t3 = std::chrono::steady_clock::now();
    int steps = 0;
    while (ctx.completion->has_next_token && steps < n_predict) {
        (void) ctx.completion->doCompletion();
        steps++;
        if (is_continuous && ctx.tts_wrapper->audio_embeddings_done) break;
    }
    ctx.completion->endCompletion();
    const auto t4 = std::chrono::steady_clock::now();
    const double gen_s = std::chrono::duration<double>(t4 - t3).count();

    // Report generation state.
    std::printf("\n[probe] === GENERATION ===\n");
    std::printf("  flow                  : %s\n", formatted.flow.c_str());
    if (is_continuous) {
        const int frames = ctx.tts_wrapper->audio_embedding_dim > 0
            ? (int) (ctx.tts_wrapper->audio_embeddings.size() /
                     (size_t) ctx.tts_wrapper->audio_embedding_dim)
            : 0;
        const int patches = frames / patch_size;
        const bool stopped = ctx.tts_wrapper->audio_embeddings_done;
        std::printf("  patch_size            : %d frames/patch\n", patch_size);
        std::printf("  patches produced      : %d\n", patches);
        std::printf("  frames produced       : %d\n", frames);
        std::printf("  stopped_on_stop_head  : %d\n", (int) stopped);
        std::printf("  STOP REASON           : %s\n",
                    stopped ? "STOP-HEAD" : (steps >= n_predict ? "N-PREDICT-CAP" : "OTHER"));
    } else {
        const int n_tokens = (int) ctx.tts_wrapper->audio_tokens.size();
        std::printf("  audio_tokens          : %d\n", n_tokens);
        std::printf("  completion_steps      : %d\n", steps);
        std::printf("  STOP REASON           : %s\n",
                    ctx.completion->has_next_token
                        ? "N-PREDICT-CAP"
                        : "COMPLETION-EOS");
    }
    std::printf("  total generation time : %.2fs\n", gen_s);

    if (out_wav.empty()) {
        // Nothing else to do — no WAV requested.
        return 0;
    }

    // Decode to PCM through the same helpers the app uses.
    std::vector<float> pcm;
    const auto td0 = std::chrono::steady_clock::now();
    if (is_continuous || capture_embeddings) {
        // Soprano's decodeAudioTokens internally routes to embeddings, so we
        // can just forward there — either through the explicit embeddings
        // helper (continuous flow, patches already accumulated in tts_wrapper)
        // or via the completion->embeddings buffer that captureHiddenStates
        // populated during the token loop.
        if (is_continuous) {
            if (ctx.tts_wrapper->audio_embeddings.empty() || ctx.tts_wrapper->audio_embedding_dim <= 0) {
                std::fprintf(stderr, "[probe] no latents accumulated; skipping wav\n");
                return 7;
            }
            pcm = ctx.tts_wrapper->decodeAudioEmbeddings(
                &ctx,
                ctx.tts_wrapper->audio_embeddings,
                ctx.tts_wrapper->audio_embedding_dim);
        } else {
            pcm = ctx.tts_wrapper->decodeAudioTokens(&ctx, {});
        }
    } else {
        if (ctx.tts_wrapper->audio_tokens.empty()) {
            std::fprintf(stderr, "[probe] no audio tokens produced; skipping wav\n");
            return 7;
        }
        pcm = ctx.tts_wrapper->decodeAudioTokens(&ctx, ctx.tts_wrapper->audio_tokens);
    }
    const auto td1 = std::chrono::steady_clock::now();
    if (pcm.empty()) {
        std::fprintf(stderr, "[probe] decode returned empty (see prior LOG_ERROR)\n");
        return 8;
    }

    const int sr = ctx.tts_wrapper->getAudioSampleRate();
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
    std::printf("  decode time : %.2fs\n", std::chrono::duration<double>(td1 - td0).count());
    return 0;
}
