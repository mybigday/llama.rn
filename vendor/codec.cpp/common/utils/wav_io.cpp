#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool codec_util_read_exact(FILE * fp, void * ptr, size_t n) {
    return std::fread(ptr, 1, n, fp) == n;
}

// One sample of `bits` bytes at `p` (little-endian) as a float in [-1, 1).
static float codec_util_wav_sample(const uint8_t * p, uint16_t bits, bool is_float) {
    if (is_float) {
        if (bits == 32) {
            float v;
            std::memcpy(&v, p, 4);
            return v;
        }
        double v;
        std::memcpy(&v, p, 8);
        return (float) v;
    }
    switch (bits) {
        case 8:  return ((int32_t) p[0] - 128) / 128.0f;  // 8-bit PCM is unsigned
        case 16: return (int16_t) (p[0] | (p[1] << 8)) / 32768.0f;
        case 24: return (int32_t) ((uint32_t) p[0] << 8 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 24) / 2147483648.0f;
        default: return (int32_t) ((uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24) / 2147483648.0f;
    }
}

bool codec_util_load_wav(const char * path, codec_util_wav_data * out, std::string * err) {
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        if (err != nullptr) {
            *err = "failed to open wav file";
        }
        return false;
    }

    char riff[4] = { 0 };
    uint32_t riff_size = 0;
    char wave[4] = { 0 };
    if (!codec_util_read_exact(fp, riff, 4) || !codec_util_read_exact(fp, &riff_size, 4) || !codec_util_read_exact(fp, wave, 4)) {
        std::fclose(fp);
        if (err != nullptr) {
            *err = "invalid wav header";
        }
        return false;
    }

    (void) riff_size;
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(fp);
        if (err != nullptr) {
            *err = "not a RIFF/WAVE file";
        }
        return false;
    }

    uint16_t audio_format = 0;
    uint16_t n_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<uint8_t> pcm_bytes;

    while (true) {
        char chunk_id[4] = { 0 };
        uint32_t chunk_size = 0;
        if (!codec_util_read_exact(fp, chunk_id, 4) || !codec_util_read_exact(fp, &chunk_size, 4)) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t block_align = 0;
            uint32_t byte_rate = 0;
            if (!codec_util_read_exact(fp, &audio_format, 2) ||
                !codec_util_read_exact(fp, &n_channels, 2) ||
                !codec_util_read_exact(fp, &sample_rate, 4) ||
                !codec_util_read_exact(fp, &byte_rate, 4) ||
                !codec_util_read_exact(fp, &block_align, 2) ||
                !codec_util_read_exact(fp, &bits_per_sample, 2)) {
                std::fclose(fp);
                if (err != nullptr) {
                    *err = "invalid fmt chunk";
                }
                return false;
            }
            (void) byte_rate;
            (void) block_align;
            // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the SubFormat GUID.
            long skip = (long) chunk_size - 16;
            if (audio_format == 0xFFFE && chunk_size >= 40) {
                uint8_t ext[24];
                if (!codec_util_read_exact(fp, ext, sizeof(ext))) {
                    std::fclose(fp);
                    if (err != nullptr) {
                        *err = "invalid fmt chunk";
                    }
                    return false;
                }
                audio_format = (uint16_t) (ext[8] | (ext[9] << 8));
                skip -= (long) sizeof(ext);
            }
            if (skip > 0) {
                std::fseek(fp, skip, SEEK_CUR);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            pcm_bytes.resize(chunk_size);
            if (chunk_size > 0 && !codec_util_read_exact(fp, pcm_bytes.data(), chunk_size)) {
                std::fclose(fp);
                if (err != nullptr) {
                    *err = "failed to read data chunk";
                }
                return false;
            }
        } else {
            std::fseek(fp, (long) chunk_size, SEEK_CUR);
        }

        if ((chunk_size & 1) != 0) {
            std::fseek(fp, 1, SEEK_CUR);
        }
    }

    std::fclose(fp);

    const bool is_float = audio_format == 3;
    const bool pcm_ok   = audio_format == 1 && (bits_per_sample == 8 || bits_per_sample == 16 ||
                                                bits_per_sample == 24 || bits_per_sample == 32);
    const bool float_ok = is_float && (bits_per_sample == 32 || bits_per_sample == 64);
    if (!pcm_ok && !float_ok) {
        if (err != nullptr) {
            *err = "unsupported wav format " + std::to_string(audio_format) + " (" +
                   std::to_string(bits_per_sample) + "-bit); PCM 8/16/24/32-bit and float 32/64-bit are supported";
        }
        return false;
    }

    if (n_channels == 0 || sample_rate == 0 || pcm_bytes.empty()) {
        if (err != nullptr) {
            *err = "missing fmt/data chunks";
        }
        return false;
    }

    out->sample_rate = (int32_t) sample_rate;
    out->n_channels = (int32_t) n_channels;
    const size_t bytes = bits_per_sample / 8;
    out->pcm.resize(pcm_bytes.size() / bytes);
    for (size_t i = 0; i < out->pcm.size(); ++i) {
        out->pcm[i] = codec_util_wav_sample(pcm_bytes.data() + i * bytes, bits_per_sample, is_float);
    }
    return true;
}

bool codec_util_write_wav_pcm16(const char * path, const float * pcm, int32_t n_samples, int32_t sample_rate, std::string * err, int32_t n_channels) {
    FILE * fp = std::fopen(path, "wb");
    if (fp == nullptr) {
        if (err != nullptr) {
            *err = "failed to open output wav";
        }
        return false;
    }

    const uint16_t audio_format = 1;
    const uint16_t nch = (uint16_t) std::max(1, n_channels);
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = nch * bits_per_sample / 8;
    const uint32_t byte_rate = (uint32_t) sample_rate * block_align;
    const uint32_t data_size = (uint32_t) n_samples * block_align;
    const uint32_t riff_size = 36 + data_size;

    std::fwrite("RIFF", 1, 4, fp);
    std::fwrite(&riff_size, 4, 1, fp);
    std::fwrite("WAVE", 1, 4, fp);
    std::fwrite("fmt ", 1, 4, fp);
    const uint32_t fmt_size = 16;
    std::fwrite(&fmt_size, 4, 1, fp);
    std::fwrite(&audio_format, 2, 1, fp);
    std::fwrite(&nch, 2, 1, fp);
    std::fwrite(&sample_rate, 4, 1, fp);
    std::fwrite(&byte_rate, 4, 1, fp);
    std::fwrite(&block_align, 2, 1, fp);
    std::fwrite(&bits_per_sample, 2, 1, fp);
    std::fwrite("data", 1, 4, fp);
    std::fwrite(&data_size, 4, 1, fp);

    const int32_t n_total = n_samples * (int32_t) nch;
    for (int32_t i = 0; i < n_total; ++i) {
        const float x = std::max(-1.0f, std::min(1.0f, pcm[i]));
        const int32_t q = (int32_t) std::lround(x * 32767.0f);
        const int16_t s = (int16_t) std::max(-32768, std::min(32767, q));
        std::fwrite(&s, sizeof(s), 1, fp);
    }

    std::fclose(fp);
    return true;
}
