#include "batch.h"

#include "../ops/safe_math.h"

#include <cstdlib>
#include <cstring>
#include <limits>

void codec_batch_reset(struct codec_batch * batch) {
    if (batch == nullptr) {
        return;
    }

    batch->n_seq = 0;
    batch->n_seq_alloc = 0;
    batch->n_seq_max = 0;

    batch->seq_id = nullptr;
    batch->n_frames = nullptr;
    batch->n_q = nullptr;

    batch->mode = CODEC_BATCH_MODE_CODES;

    batch->codes = nullptr;
    batch->codes_size = 0;
    batch->codes_used = 0;

    batch->latent = nullptr;
    batch->latent_dim = 0;
    batch->latent_size = 0;
    batch->latent_used = 0;

    batch->codes_offset = nullptr;
    batch->latent_offset = nullptr;

    batch->sample_rate = 0;
    batch->hop_size = 0;
}
struct codec_batch codec_batch_init_codes(int32_t n_seq_alloc, int32_t codes_alloc_total, int32_t n_seq_max) {
    struct codec_batch batch;
    codec_batch_reset(&batch);

    if (n_seq_alloc <= 0 || codes_alloc_total <= 0 || n_seq_max <= 0) {
        return batch;
    }

    batch.mode = CODEC_BATCH_MODE_CODES;
    batch.n_seq_alloc = n_seq_alloc;
    batch.n_seq_max = n_seq_max;
    batch.codes_size = codes_alloc_total;

    batch.seq_id = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.n_frames = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.n_q = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.codes_offset = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.latent_offset = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.codes = static_cast<int32_t *>(std::calloc((size_t)codes_alloc_total, sizeof(int32_t)));

    if (batch.seq_id == nullptr || batch.n_frames == nullptr || batch.n_q == nullptr ||
        batch.codes_offset == nullptr || batch.latent_offset == nullptr || batch.codes == nullptr) {
        codec_batch_free(batch);
        codec_batch_reset(&batch);
    }

    return batch;
}

struct codec_batch codec_batch_init_latent(int32_t n_seq_alloc, int32_t latent_dim, int32_t latent_alloc_total, int32_t n_seq_max) {
    struct codec_batch batch;
    codec_batch_reset(&batch);

    if (n_seq_alloc <= 0 || latent_dim <= 0 || latent_alloc_total <= 0 || n_seq_max <= 0) {
        return batch;
    }

    batch.mode = CODEC_BATCH_MODE_LATENT;
    batch.n_seq_alloc = n_seq_alloc;
    batch.n_seq_max = n_seq_max;
    batch.latent_dim = latent_dim;
    batch.latent_size = latent_alloc_total;

    batch.seq_id = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.n_frames = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.n_q = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.codes_offset = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.latent_offset = static_cast<int32_t *>(std::calloc((size_t)n_seq_alloc, sizeof(int32_t)));
    batch.latent = static_cast<float *>(std::calloc((size_t)latent_alloc_total, sizeof(float)));

    if (batch.seq_id == nullptr || batch.n_frames == nullptr || batch.n_q == nullptr ||
        batch.codes_offset == nullptr || batch.latent_offset == nullptr || batch.latent == nullptr) {
        codec_batch_free(batch);
        codec_batch_reset(&batch);
    }

    return batch;
}

void codec_batch_free(struct codec_batch batch) {
    std::free(batch.seq_id);
    std::free(batch.n_frames);
    std::free(batch.n_q);
    std::free(batch.codes);
    std::free(batch.latent);
    std::free(batch.codes_offset);
    std::free(batch.latent_offset);
}

int32_t codec_batch_add_seq_codes(
    struct codec_batch * batch,
    int32_t seq_id,
    int32_t n_frames,
    int32_t n_q,
    const int32_t * codes) {

    if (batch == nullptr || batch->mode != CODEC_BATCH_MODE_CODES) {
        return -1;
    }
    if (batch->n_seq_alloc <= 0 || batch->n_seq_max <= 0 || batch->seq_id == nullptr || batch->n_frames == nullptr ||
        batch->n_q == nullptr || batch->codes_offset == nullptr || batch->codes == nullptr) {
        return -1;
    }
    if (seq_id < 0 || seq_id >= batch->n_seq_max || n_frames <= 0 || n_q <= 0 || codes == nullptr) {
        return -1;
    }
    if (batch->n_seq < 0 || batch->n_seq >= batch->n_seq_alloc) {
        return -1;
    }

    int32_t seq_codes = 0;
    if (!codec_safe_mul_i32(n_frames, n_q, &seq_codes)) {
        return -1;
    }
    int32_t next_used = 0;
    if (!codec_safe_add_i32(batch->codes_used, seq_codes, &next_used)) {
        return -1;
    }
    if (batch->codes_size <= 0 || batch->codes_used < 0 || next_used > batch->codes_size) {
        return -1;
    }
    if (batch->codes_used > std::numeric_limits<int32_t>::max() / (int32_t)sizeof(int32_t)) {
        return -1;
    }

    const int32_t idx = batch->n_seq;
    batch->seq_id[idx] = seq_id;
    batch->n_frames[idx] = n_frames;
    batch->n_q[idx] = n_q;
    batch->codes_offset[idx] = batch->codes_used * (int32_t)sizeof(int32_t);
    if (batch->latent_offset != nullptr) {
        batch->latent_offset[idx] = 0;
    }

    std::memcpy(batch->codes + batch->codes_used, codes, (size_t)seq_codes * sizeof(int32_t));

    batch->codes_used = next_used;
    batch->n_seq = idx + 1;
    return idx;
}

int32_t codec_batch_add_seq_latent(
    struct codec_batch * batch,
    int32_t seq_id,
    int32_t n_frames,
    const float * latent,
    int32_t latent_dim) {

    if (batch == nullptr || batch->mode != CODEC_BATCH_MODE_LATENT) {
        return -1;
    }
    if (batch->n_seq_alloc <= 0 || batch->n_seq_max <= 0 || batch->seq_id == nullptr || batch->n_frames == nullptr ||
        batch->n_q == nullptr || batch->latent_offset == nullptr || batch->latent == nullptr) {
        return -1;
    }
    if (seq_id < 0 || seq_id >= batch->n_seq_max || n_frames <= 0 || latent == nullptr) {
        return -1;
    }
    if (batch->n_seq < 0 || batch->n_seq >= batch->n_seq_alloc) {
        return -1;
    }
    if (latent_dim <= 0 || batch->latent_dim <= 0 || latent_dim != batch->latent_dim) {
        return -1;
    }

    int32_t seq_latent = 0;
    if (!codec_safe_mul_i32(n_frames, latent_dim, &seq_latent)) {
        return -1;
    }
    int32_t next_used = 0;
    if (!codec_safe_add_i32(batch->latent_used, seq_latent, &next_used)) {
        return -1;
    }
    if (batch->latent_size <= 0 || batch->latent_used < 0 || next_used > batch->latent_size) {
        return -1;
    }
    if (batch->latent_used > std::numeric_limits<int32_t>::max() / (int32_t)sizeof(float)) {
        return -1;
    }

    const int32_t idx = batch->n_seq;
    batch->seq_id[idx] = seq_id;
    batch->n_frames[idx] = n_frames;
    batch->n_q[idx] = 0;
    batch->latent_offset[idx] = batch->latent_used * (int32_t)sizeof(float);
    if (batch->codes_offset != nullptr) {
        batch->codes_offset[idx] = 0;
    }

    std::memcpy(batch->latent + batch->latent_used, latent, (size_t)seq_latent * sizeof(float));

    batch->latent_used = next_used;
    batch->n_seq = idx + 1;
    return idx;
}

