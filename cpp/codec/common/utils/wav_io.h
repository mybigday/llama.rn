#ifndef CODEC_EXAMPLES_UTILS_WAV_IO_H
#define CODEC_EXAMPLES_UTILS_WAV_IO_H

#include <cstdint>
#include <string>
#include <vector>

struct codec_example_wav_data {
    int32_t sample_rate = 0;
    int32_t n_channels = 0;
    std::vector<int16_t> pcm_i16;
};

bool codec_example_load_wav_pcm16(const char * path, codec_example_wav_data * out, std::string * err);
bool codec_example_write_wav_pcm16(const char * path, const float * pcm, int32_t n_samples, int32_t sample_rate, std::string * err, int32_t n_channels = 1);

#endif
