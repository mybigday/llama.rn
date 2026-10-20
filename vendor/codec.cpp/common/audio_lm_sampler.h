#ifndef CODEC_AUDIO_LM_SAMPLER_H
#define CODEC_AUDIO_LM_SAMPLER_H

// Internal: samples the codec-side heads (codec_lm codebook logits, speech
// heads) inside codec_audio_lm, so the host only ever samples backbone tokens.
// The stages are llama.cpp's sampler algorithms (src/llama-sampler.cpp:
// penalties, temperature, top-k, top-p, min-p, dist, greedy) reproduced
// operation for operation, so a host that used llama_sampler on these logits
// before gets the same codes.

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace codec_common {

class alm_head_sampler {
public:
    enum stage { PENALTIES, TEMPERATURE, TOP_K, TOP_P, MIN_P };

    // temp <= 0: greedy (argmax, no other stage).  penalty_last_n 0: whole history.
    void init(uint32_t seed, float temp, int32_t top_k, float top_p, float min_p, float penalty_repeat,
              int32_t penalty_last_n, std::vector<stage> order);
    // Default order: penalties -> temperature -> top-k -> min-p -> top-p.
    void init(uint32_t seed, float temp, int32_t top_k, float top_p, float min_p, float penalty_repeat,
              int32_t penalty_last_n);

    // Sample an index of logits[0..n) and record it for the penalties.
    int32_t sample(const float * logits, int32_t n);
    // Record a token the penalties should count without sampling it.
    void    accept(int32_t token);

    struct cand { int32_t id; float logit; float p; };

private:
    struct array { cand * data; size_t size; int64_t selected; bool sorted; };

    void penalties(array & cur) const;
    void top_p(array & cur);

    bool    greedy_ = true;
    float   temp_ = 0.0f, top_p_ = 1.0f, min_p_ = 0.0f, repeat_ = 1.0f;
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
