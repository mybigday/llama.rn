#include "wav_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool codec_example_read_exact(FILE * fp, void * ptr, size_t n) {
    return std::fread(ptr, 1, n, fp) == n;
}

bool codec_example_load_wav_pcm16(const char * path, codec_example_wav_data * out, std::string * err) {
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
    if (!codec_example_read_exact(fp, riff, 4) || !codec_example_read_exact(fp, &riff_size, 4) || !codec_example_read_exact(fp, wave, 4)) {
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
        if (!codec_example_read_exact(fp, chunk_id, 4) || !codec_example_read_exact(fp, &chunk_size, 4)) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            uint16_t block_align = 0;
            uint32_t byte_rate = 0;
            if (!codec_example_read_exact(fp, &audio_format, 2) ||
                !codec_example_read_exact(fp, &n_channels, 2) ||
                !codec_example_read_exact(fp, &sample_rate, 4) ||
                !codec_example_read_exact(fp, &byte_rate, 4) ||
                !codec_example_read_exact(fp, &block_align, 2) ||
                !codec_example_read_exact(fp, &bits_per_sample, 2)) {
                std::fclose(fp);
                if (err != nullptr) {
                    *err = "invalid fmt chunk";
                }
                return false;
            }
            (void) byte_rate;
            (void) block_align;
            if (chunk_size > 16) {
                std::fseek(fp, (long) (chunk_size - 16), SEEK_CUR);
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            pcm_bytes.resize(chunk_size);
            if (chunk_size > 0 && !codec_example_read_exact(fp, pcm_bytes.data(), chunk_size)) {
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

    if (audio_format != 1 || bits_per_sample != 16) {
        if (err != nullptr) {
            *err = "only PCM 16-bit wav is supported";
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
    out->pcm_i16.resize(pcm_bytes.size() / sizeof(int16_t));
    std::memcpy(out->pcm_i16.data(), pcm_bytes.data(), pcm_bytes.size());
    return true;
}

bool codec_example_write_wav_pcm16(const char * path, const float * pcm, int32_t n_samples, int32_t sample_rate, std::string * err, int32_t n_channels) {
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
