#ifndef CODEC_COMMON_UTILS_WAV_IO_H
#define CODEC_COMMON_UTILS_WAV_IO_H

#include <cstdint>
#include <string>
#include <vector>

struct codec_util_wav_data {
    int32_t sample_rate = 0;
    int32_t n_channels = 0;
    std::vector<float> pcm;  // interleaved, [-1, 1)
};

// Reads PCM 8/16/24/32-bit and IEEE float 32/64-bit wav (incl. WAVE_FORMAT_EXTENSIBLE).
bool codec_util_load_wav(const char * path, codec_util_wav_data * out, std::string * err);
bool codec_util_write_wav_pcm16(const char * path, const float * pcm, int32_t n_samples, int32_t sample_rate, std::string * err, int32_t n_channels = 1);

#endif
