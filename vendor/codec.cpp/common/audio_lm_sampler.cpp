// llama.cpp's sampler stages (src/llama-sampler.cpp), reproduced for the
// codec-side heads; see audio_lm_sampler.h.

#include "audio_lm_sampler.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <sstream>

namespace codec_common {
namespace {

using cand = alm_head_sampler::cand;

constexpr int32_t k_full_history = 1 << 16;   // "whole history" window

bool by_logit(const cand & a, const cand & b) { return a.logit > b.logit; }

// llama_token_data_array_partial_sort: top npartial of cur into res (cur untouched).
void partial_sort(const cand * data, size_t size, int npartial, std::vector<cand> & res) {
    constexpr int   nbuckets     = 128;
    constexpr float bucket_low   = -10.0f;
    constexpr float bucket_high  =  10.0f;
    constexpr float bucket_scale = nbuckets / (bucket_high - bucket_low);
    constexpr float bucket_inter = -bucket_low * bucket_scale;

    std::vector<int> bucket_idx;
    std::vector<int> histo(nbuckets, 0);
    std::vector<cand *> bucket_ptrs;
    bucket_idx.reserve(size);
    for (int i = 0; i < (int) size; ++i) {
        int ib = int(bucket_scale * data[i].logit + bucket_inter);
        ib = std::max(0, std::min(nbuckets - 1, ib));
        bucket_idx.push_back(ib);
        ++histo[ib];
    }
    int nhave = 0;
    int ib = nbuckets - 1;
    for (; ib >= 0; --ib) {
        nhave += histo[ib];
        if (nhave >= npartial) break;
    }
    res.resize(nhave);
    cand * ptr = res.data();
    bucket_ptrs.reserve(nbuckets - ib);
    for (int j = nbuckets - 1; j >= ib; --j) {
        bucket_ptrs.push_back(ptr);
        ptr += histo[j];
    }
    for (int i = 0; i < (int) size; ++i) {
        const int j = bucket_idx[i];
        if (j >= ib) *bucket_ptrs[nbuckets - 1 - j]++ = data[i];
    }
    ptr = res.data();
    int ndone = 0;
    for (int j = nbuckets - 1; j > ib; --j) {
        std::sort(ptr, ptr + histo[j], by_logit);
        ptr += histo[j];
        ndone += histo[j];
    }
    std::partial_sort(ptr, ptr + npartial - ndone, ptr + histo[ib], by_logit);
}

// llama_token_data_array_partial_sort_inplace
void sort_inplace(cand * data, size_t & size, bool & sorted, size_t npartial) {
    if (npartial <= 128) {
        std::partial_sort(data, data + npartial, data + size, by_logit);
    } else {
        std::vector<cand> tmp;
        partial_sort(data, size, (int) npartial, tmp);
        std::copy(tmp.data(), tmp.data() + npartial, data);
    }
    size = npartial;
    sorted = true;
}

}  // namespace

bool alm_head_sampler::init(const codec_audio_lm_sampler & sp, bool penalties, std::string * err) {
    temp_    = sp.temp;
    top_k_   = sp.top_k;
    top_p_   = sp.top_p;
    min_p_   = sp.min_p;
    repeat_  = penalties ? sp.penalty_repeat : 1.0f;
    freq_    = penalties ? sp.penalty_freq : 0.0f;
    present_ = penalties ? sp.penalty_present : 0.0f;
    const bool any = repeat_ != 1.0f || freq_ != 0.0f || present_ != 0.0f;
    last_n_  = !any || sp.penalty_last_n == 0 ? 0 : (sp.penalty_last_n < 0 ? k_full_history : sp.penalty_last_n);
    order_.clear();
    // common_sampler_types_from_names, the stages a head can run; the others
    // are no-ops at llama.cpp's defaults.
    const std::string names = sp.samplers && *sp.samplers ? sp.samplers
                                                          : "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature";
    std::stringstream ss(names);
    for (std::string n; std::getline(ss, n, ';');) {
        if (n.empty()) continue;
        if (n == "penalties") order_.push_back(PENALTIES);
        else if (n == "top_k") order_.push_back(TOP_K);
        else if (n == "top_p" || n == "nucleus") order_.push_back(TOP_P);
        else if (n == "min_p") order_.push_back(MIN_P);
        else if (n == "temperature" || n == "temp") order_.push_back(TEMPERATURE);
        else if (n == "dry") {
            if (sp.dry_multiplier != 0.0f) {
                if (err) *err = "the codec-side heads have no DRY sampler (dry_multiplier must be 0)";
                return false;
            }
        } else if (n != "top_n_sigma" && n != "typ_p" && n != "typical_p" && n != "typical" && n != "xtc") {
            if (err) *err = "unknown sampler '" + n + "'";
            return false;
        }
    }
    // llama_sampler_init_dist(LLAMA_DEFAULT_SEED) seeds from the clock / device.
    rng_.seed(sp.seed == CODEC_AUDIO_LM_DEFAULT_SEED ? std::random_device{}() : sp.seed);
    prev_.clear();
    prev_first_ = 0;
    count_.clear();
    return true;
}

void alm_head_sampler::accept(int32_t token) {
    if (last_n_ == 0) return;
    // llama_sampler_penalties_accept
    count_[token]++;
    if (prev_.size() >= (size_t) last_n_) {
        const int32_t old = prev_[prev_first_];
        if (--count_[old] == 0) count_.erase(old);
        prev_[prev_first_] = token;
        prev_first_ = (prev_first_ + 1) % prev_.size();
    } else {
        prev_.push_back(token);
    }
}

void alm_head_sampler::penalties(array & cur) const {
    // llama_sampler_penalties_apply
    if (last_n_ == 0 || count_.empty()) return;
    for (size_t i = 0; i < cur.size; ++i) {
        const auto it = count_.find(cur.data[i].id);
        if (it == count_.end()) continue;
        if (cur.data[i].logit <= 0) cur.data[i].logit *= repeat_;
        else                        cur.data[i].logit /= repeat_;
        cur.data[i].logit -= float(it->second) * freq_ + float(it->second > 0) * present_;
    }
    cur.sorted = false;
}

void alm_head_sampler::top_p(array & cur) {
    // llama_sampler_top_p_apply with min_keep = 1
    if (top_p_ >= 1.0f) return;
    {   // llama_sampler_softmax_impl(cur, false)
        float max_l = cur.data[0].logit;
        if (!cur.sorted) for (size_t i = 1; i < cur.size; ++i) max_l = std::max(max_l, cur.data[i].logit);
        float cum = 0.0f;
        for (size_t i = 0; i < cur.size; ++i) { const float p = expf(cur.data[i].logit - max_l); cur.data[i].p = p; cum += p; }
        for (size_t i = 0; i < cur.size; ++i) cur.data[i].p /= cum;
    }
    size_t k = cur.size;
    cand * pdata = cur.data;
    if (!cur.sorted && cur.size > 1024) {
        k = std::min<size_t>(256, cur.size);
        partial_sort(cur.data, cur.size, (int) k, buf_sort_);
        pdata = buf_sort_.data();
    } else if (!cur.sorted) {
        sort_inplace(cur.data, cur.size, cur.sorted, k);
    }
    float cum_sum = 0.0f;
    size_t last_idx = cur.size;
    for (size_t i = 0; i < cur.size; ++i) {
        cum_sum += pdata[i].p;
        if (cum_sum >= top_p_ && i + 1 >= 1) { last_idx = i + 1; break; }
        if (!cur.sorted && i == k - 1) {
            k = cur.size;
            partial_sort(cur.data, cur.size, (int) k, buf_sort_);
            pdata = buf_sort_.data();
        }
    }
    if (!cur.sorted) {
        std::copy(buf_sort_.data(), buf_sort_.data() + last_idx, cur.data);
        cur.sorted = true;
    }
    cur.size = last_idx;
}

int32_t alm_head_sampler::sample(const float * logits, int32_t n) {
    if (n <= 0) return 0;
    buf_.resize((size_t) n);
    for (int32_t i = 0; i < n; ++i) buf_[(size_t) i] = {i, logits[i], 0.0f};
    array cur = {buf_.data(), (size_t) n, -1, false};

    for (stage s : order_) {
        switch (s) {
            case PENALTIES:
                penalties(cur);
                break;
            case TEMPERATURE:   // llama_sampler_temp_impl
                if (temp_ <= 0.0f) {   // keep only the most likely candidate
                    size_t best = 0;
                    for (size_t i = 1; i < cur.size; ++i) if (cur.data[i].logit > cur.data[best].logit) best = i;
                    cur.data[0] = cur.data[best];
                    cur.size = 1;
                    cur.sorted = true;
                } else {
                    for (size_t i = 0; i < cur.size; ++i) cur.data[i].logit /= temp_;
                }
                break;
            case TOP_K:         // llama_sampler_top_k_impl
                if (top_k_ > 0) {
                    const int32_t k = std::min(top_k_, (int32_t) cur.size);
                    if (!cur.sorted) sort_inplace(cur.data, cur.size, cur.sorted, (size_t) k);
                    cur.size = (size_t) k;
                }
                break;
            case TOP_P:
                if (top_p_ > 0.0f && top_p_ < 1.0f) top_p(cur);
                break;
            case MIN_P:         // llama_sampler_min_p_apply with min_keep = 1
                if (min_p_ > 0.0f && cur.size > 0) {
                    bool applied = false;
                    if (!cur.sorted) {
                        std::vector<cand> kept;
                        float max_logit = -FLT_MAX;
                        for (size_t i = 0; i < cur.size; ++i) max_logit = std::max(max_logit, cur.data[i].logit);
                        const float min_logit = max_logit + logf(min_p_);
                        for (size_t i = 0; i < cur.size; ++i) if (cur.data[i].logit >= min_logit) kept.push_back(cur.data[i]);
                        if (!kept.empty()) {
                            std::copy(kept.begin(), kept.end(), cur.data);
                            cur.size = kept.size();
                            applied = true;
                        }
                    }
                    if (!applied) {
                        if (!cur.sorted) sort_inplace(cur.data, cur.size, cur.sorted, cur.size);
                        const float min_logit = cur.data[0].logit + logf(min_p_);
                        size_t i = 1;
                        for (; i < cur.size; ++i) if (cur.data[i].logit < min_logit && i >= 1) break;
                        cur.size = i;
                    }
                }
                break;
        }
    }

    // llama_sampler_dist_apply
    cur.selected = 0;
    if (cur.size > 1) {
        float max_l = cur.data[0].logit;
        if (!cur.sorted) for (size_t i = 1; i < cur.size; ++i) max_l = std::max(max_l, cur.data[i].logit);
        double sum_cum = 0.0f;
        for (size_t i = 0; i < cur.size; ++i) {
            const float p = expf(cur.data[i].logit - max_l);
            cur.data[i].p = p;
            sum_cum += p;
        }
        std::uniform_real_distribution<double> dist(0.0f, 1.0f);
        const double rnd = dist(rng_);
        double sum_run = 0.0f;
        const double sum_tgt = sum_cum * rnd;
        bool found = false;
        for (size_t i = 0; i < cur.size; ++i) {
            if (!found) {
                sum_run += cur.data[i].p;
                if (sum_run >= sum_tgt) { cur.selected = (int64_t) i; found = true; }
            }
        }
        if (!found) cur.selected = (int64_t) cur.size - 1;
    }
    const int32_t id = cur.data[cur.selected].id;
    accept(id);
    return id;
}

}  // namespace codec_common
