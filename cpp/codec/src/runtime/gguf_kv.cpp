#include "lm_gguf_kv.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

char * codec_strdup(const char * s) {
    if (s == nullptr) {
        return nullptr;
    }

    const size_t n = std::strlen(s) + 1;
    char * out = static_cast<char *>(std::malloc(n));
    if (out != nullptr) {
        std::memcpy(out, s, n);
    }

    return out;
}

void codec_metadata_add(struct codec_lm_gguf_metadata * meta, const char * key, const std::string & value) {
    const size_t next = meta->n_items + 1;
    void * p = std::realloc(meta->items, next * sizeof(meta->items[0]));
    if (p == nullptr) {
        return;
    }

    meta->items = static_cast<struct codec_lm_gguf_kv *>(p);
    meta->items[meta->n_items].key = codec_strdup(key);
    meta->items[meta->n_items].value = codec_strdup(value.c_str());
    meta->n_items = next;
}

std::string codec_lm_gguf_value_to_string(struct lm_gguf_context * gf, int key_id) {
    std::ostringstream oss;

    switch (lm_gguf_get_kv_type(gf, key_id)) {
        case LM_GGUF_TYPE_UINT8:   oss << int(lm_gguf_get_val_u8(gf, key_id)); break;
        case LM_GGUF_TYPE_INT8:    oss << int(lm_gguf_get_val_i8(gf, key_id)); break;
        case LM_GGUF_TYPE_UINT16:  oss << lm_gguf_get_val_u16(gf, key_id); break;
        case LM_GGUF_TYPE_INT16:   oss << lm_gguf_get_val_i16(gf, key_id); break;
        case LM_GGUF_TYPE_UINT32:  oss << lm_gguf_get_val_u32(gf, key_id); break;
        case LM_GGUF_TYPE_INT32:   oss << lm_gguf_get_val_i32(gf, key_id); break;
        case LM_GGUF_TYPE_UINT64:  oss << lm_gguf_get_val_u64(gf, key_id); break;
        case LM_GGUF_TYPE_INT64:   oss << lm_gguf_get_val_i64(gf, key_id); break;
        case LM_GGUF_TYPE_FLOAT32: oss << lm_gguf_get_val_f32(gf, key_id); break;
        case LM_GGUF_TYPE_FLOAT64: oss << lm_gguf_get_val_f64(gf, key_id); break;
        case LM_GGUF_TYPE_BOOL:    oss << (lm_gguf_get_val_bool(gf, key_id) ? "true" : "false"); break;
        case LM_GGUF_TYPE_STRING:  oss << lm_gguf_get_val_str(gf, key_id); break;
        case LM_GGUF_TYPE_ARRAY: {
            oss << "<array:";
            oss << lm_gguf_type_name(lm_gguf_get_arr_type(gf, key_id));
            oss << ", n=" << (int)lm_gguf_get_arr_n(gf, key_id);
            oss << ">";
        } break;
        default:
            oss << "<unsupported>";
            break;
    }

    return oss.str();
}

void codec_collect_lm_gguf_metadata(struct codec_model * model) {
    const int n_kv = lm_gguf_get_n_kv(model->gguf);

    for (int i = 0; i < n_kv; ++i) {
        const char * key = lm_gguf_get_key(model->gguf, i);
        if (key == nullptr) {
            continue;
        }

        codec_metadata_add(&model->metadata, key, codec_lm_gguf_value_to_string(model->gguf, i));
    }
}

int32_t codec_read_i32_kv(struct lm_gguf_context * gf, const char * key, int32_t fallback) {
    const int key_id = lm_gguf_find_key(gf, key);
    if (key_id < 0) {
        return fallback;
    }

    const enum lm_gguf_type type = lm_gguf_get_kv_type(gf, key_id);
    if (type == LM_GGUF_TYPE_INT32) {
        return lm_gguf_get_val_i32(gf, key_id);
    }

    if (type == LM_GGUF_TYPE_UINT32) {
        return (int32_t)lm_gguf_get_val_u32(gf, key_id);
    }

    return fallback;
}

int32_t codec_read_i32_kv_any(struct lm_gguf_context * gf, const char * const * keys, size_t n_keys, int32_t fallback) {
    for (size_t i = 0; i < n_keys; ++i) {
        const int32_t val = codec_read_i32_kv(gf, keys[i], fallback);
        if (val != fallback) {
            return val;
        }
    }
    return fallback;
}

float codec_read_f32_kv(struct lm_gguf_context * gf, const char * key, float fallback) {
    const int key_id = lm_gguf_find_key(gf, key);
    if (key_id < 0) {
        return fallback;
    }

    const enum lm_gguf_type type = lm_gguf_get_kv_type(gf, key_id);
    if (type == LM_GGUF_TYPE_FLOAT32) {
        return lm_gguf_get_val_f32(gf, key_id);
    }
    if (type == LM_GGUF_TYPE_FLOAT64) {
        return (float)lm_gguf_get_val_f64(gf, key_id);
    }

    return fallback;
}

bool codec_read_bool_kv(struct lm_gguf_context * gf, const char * key, bool fallback) {
    const int key_id = lm_gguf_find_key(gf, key);
    if (key_id < 0) {
        return fallback;
    }

    if (lm_gguf_get_kv_type(gf, key_id) == LM_GGUF_TYPE_BOOL) {
        return lm_gguf_get_val_bool(gf, key_id);
    }

    return fallback;
}

int codec_count_tensors_with_prefix(const struct codec_model * model, const char * prefix) {
    int count = 0;
    for (int i = 0; i < model->n_tensors; ++i) {
        const char * name = lm_gguf_get_tensor_name(model->gguf, i);
        if (name != nullptr && std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            ++count;
        }
    }
    return count;
}

int32_t codec_infer_n_q_from_tensor_names(const struct codec_model * model) {
    int32_t max_layer = -1;

    for (int i = 0; i < model->n_tensors; ++i) {
        const char * name = lm_gguf_get_tensor_name(model->gguf, i);
        if (name == nullptr) {
            continue;
        }

        const char * prefix = "vq.vq.layers.";
        const size_t n_prefix = std::strlen(prefix);
        if (std::strncmp(name, prefix, n_prefix) != 0) {
            continue;
        }

        const char * p = name + n_prefix;
        int idx = 0;
        if (std::sscanf(p, "%d", &idx) == 1) {
            max_layer = std::max(max_layer, (int32_t)idx);
        }
    }

    return max_layer >= 0 ? (max_layer + 1) : 1;
}
