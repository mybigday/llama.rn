#ifndef CODEC_TTS_SEGMENT_H
#define CODEC_TTS_SEGMENT_H

// Text segmentation and PCM helpers shared by the codec_audio_lm families
// (tts_segment.cpp, part of libcodec).

#include <cstdint>
#include <string>
#include <vector>

namespace codec_common {

// Split synthesis text into segments of at most max_bytes UTF-8 bytes:
// sentences packed greedily, paragraph (blank-line) breaks always split, an
// over-long sentence cut at clause punctuation, then spaces (one with
// neither stays whole, over the budget — never cut mid-word).
// max_bytes <= 0 returns the trimmed text as one segment.
std::vector<std::string> tts_split_text(const std::string & text, int32_t max_bytes);

// A segment's edges for joining: leading / trailing silence trimmed (frames
// of 10 ms below -45 dBFS, keeping 20 ms of margin) when `trim`, then a
// linear fade of `fade_ms` at both ends.
std::vector<float> tts_shape_segment(const std::vector<float> & pcm, int32_t sample_rate, bool trim, int32_t fade_ms);

// Linear-interpolation resample of mono PCM (speaker encoders' input rates).
std::vector<float> resample_mono_f32(const std::vector<float> & in, int32_t in_sr, int32_t out_sr);

}  // namespace codec_common

#endif  // CODEC_TTS_SEGMENT_H
