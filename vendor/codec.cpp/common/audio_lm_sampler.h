#ifndef CODEC_AUDIO_LM_SAMPLER_H
#define CODEC_AUDIO_LM_SAMPLER_H

// Internal: samples the codec-side heads (codec_lm codebook logits, speech
// heads) inside codec_audio_lm, so the host only ever samples backbone tokens.
// The stages are llama.cpp's sampler algorithms (src/llama-sampler.cpp:
// penalties, temperature, top-k, top-p, min-p, dist, greedy) reproduced
// operation for operation, so a host that used llama_sampler on these logits
// before gets the same codes.

#include "codec_audio_lm.h"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace codec_common {

class alm_head_sampler {
public:
    enum stage { PENALTIES, TEMPERATURE, TOP_K, TOP_P, MIN_P };

    // The llama.cpp chain of `sp` (common_sampler_init): its stages in
    // sp.samplers order (llama.cpp's default when empty), then dist; temp <= 0
    // keeps the most likely candidate.  `penalties` false leaves the penalties
    // out (heads the model never penalizes).  False + *err when sp.samplers
    // names an unknown stage, or an active one the heads do not implement (DRY).
    bool init(const codec_audio_lm_sampler & sp, bool penalties, std::string * err);

    // Sample an index of logits[0..n) and record it for the penalties.
    int32_t sample(const float * logits, int32_t n);
    // Record a token the penalties should count without sampling it.
    void    accept(int32_t token);

    struct cand { int32_t id; float logit; float p; };

private:
    struct array { cand * data; size_t size; int64_t selected; bool sorted; };

    void penalties(array & cur) const;
    void top_p(array & cur);

    float   temp_ = 0.0f, top_p_ = 1.0f, min_p_ = 0.0f, repeat_ = 1.0f, freq_ = 0.0f, present_ = 0.0f;
    int32_t top_k_ = 0, last_n_ = 0;
    std::vector<stage> order_;
    std::mt19937 rng_;
    std::vector<cand> buf_, buf_sort_;
    std::vector<int32_t> prev_;                  // ring buffer of the last last_n_ tokens
    size_t prev_first_ = 0;
    std::unordered_map<int32_t, int> count_;
};

}  // namespace codec_common

#endif  // CODEC_AUDIO_LM_SAMPLER_H
