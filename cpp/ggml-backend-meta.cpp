#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct lm_ggml_backend_meta_device;
struct lm_ggml_backend_meta_buffer_type;
struct lm_ggml_backend_meta_buffer;
struct lm_ggml_backend_meta;

const char * lm_ggml_backend_meta_split_axis_name(enum lm_ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case LM_GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case LM_GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case LM_GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case LM_GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case LM_GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            LM_GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct lm_ggml_backend_meta_device_context {
    std::vector<lm_ggml_backend_dev_t>     simple_devs;
    lm_ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    lm_ggml_backend_meta_device_context(
            std::vector<lm_ggml_backend_dev_t> simple_devs, lm_ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += lm_ggml_backend_dev_name       (simple_devs[i]);
            description += lm_ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const lm_ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool lm_ggml_backend_dev_is_meta(lm_ggml_backend_dev_t dev);

static const char * lm_ggml_backend_meta_device_get_name(lm_ggml_backend_dev_t dev) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * lm_ggml_backend_meta_device_get_description(lm_ggml_backend_dev_t dev) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void lm_ggml_backend_meta_device_get_memory(lm_ggml_backend_dev_t dev, size_t * free, size_t * total) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (lm_ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        lm_ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum lm_ggml_backend_dev_type lm_ggml_backend_meta_device_get_type(lm_ggml_backend_dev_t dev) {
    return LM_GGML_BACKEND_DEVICE_TYPE_META;

    LM_GGML_UNUSED(dev);
}

static void lm_ggml_backend_meta_device_get_props(lm_ggml_backend_dev_t dev, lm_ggml_backend_dev_props * props) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = lm_ggml_backend_meta_device_get_name(dev);
    props->description = lm_ggml_backend_meta_device_get_description(dev);
    props->type        = lm_ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    lm_ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (lm_ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        lm_ggml_backend_dev_props tmp_props;
        lm_ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static lm_ggml_backend_t lm_ggml_backend_meta_device_init_backend(lm_ggml_backend_dev_t dev, const char * params);

static lm_ggml_backend_buffer_type_t lm_ggml_backend_meta_device_get_buffer_type(lm_ggml_backend_dev_t dev);

static lm_ggml_backend_buffer_type_t lm_ggml_backend_meta_device_get_host_buffer_type(lm_ggml_backend_dev_t dev);

static bool lm_ggml_backend_meta_device_supports_op(lm_ggml_backend_dev_t dev, const lm_ggml_tensor * op) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](lm_ggml_backend_dev_t simple_dev) { return lm_ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool lm_ggml_backend_meta_device_supports_buft(lm_ggml_backend_dev_t dev, lm_ggml_backend_buffer_type_t buft) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    lm_ggml_backend_dev_t dev_buft = lm_ggml_backend_buft_get_device(buft);
    if (!lm_ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const lm_ggml_backend_meta_device_context * meta_dev_ctx      = (const lm_ggml_backend_meta_device_context *) dev->context;
    const lm_ggml_backend_meta_device_context * meta_buft_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const lm_ggml_backend_device_i lm_ggml_backend_meta_device_iface = {
    /* .get_name             = */ lm_ggml_backend_meta_device_get_name,
    /* .get_description      = */ lm_ggml_backend_meta_device_get_description,
    /* .get_memory           = */ lm_ggml_backend_meta_device_get_memory,
    /* .get_type             = */ lm_ggml_backend_meta_device_get_type,
    /* .get_props            = */ lm_ggml_backend_meta_device_get_props,
    /* .init_backend         = */ lm_ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ lm_ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ lm_ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ lm_ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ lm_ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool lm_ggml_backend_dev_is_meta(lm_ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == lm_ggml_backend_meta_device_iface.get_name;
}

static size_t lm_ggml_backend_meta_dev_n_devs(lm_ggml_backend_dev_t meta_dev) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(meta_dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static lm_ggml_backend_dev_t lm_ggml_backend_meta_dev_simple_dev(lm_ggml_backend_dev_t meta_dev, size_t index) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(meta_dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) meta_dev->context;
    LM_GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

lm_ggml_backend_dev_t lm_ggml_backend_meta_device(
        lm_ggml_backend_dev_t * devs, size_t n_devs, lm_ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    LM_GGML_ASSERT(n_devs <= LM_GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<lm_ggml_backend_meta_device_context>>         ctxs;
    static std::map<lm_ggml_backend_meta_device_context, struct lm_ggml_backend_device> meta_devs;

    std::vector<lm_ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    lm_ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<lm_ggml_backend_meta_device_context>(ctx));

    struct lm_ggml_backend_device meta_dev = {
        /*iface  =*/ lm_ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct lm_ggml_backend_meta_buffer_type_context {
    std::vector<lm_ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    lm_ggml_backend_meta_buffer_type_context(std::vector<lm_ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += lm_ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const lm_ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t lm_ggml_backend_meta_buft_n_bufts(lm_ggml_backend_buffer_type_t meta_buft) {
    LM_GGML_ASSERT(lm_ggml_backend_buft_is_meta(meta_buft));
    const lm_ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const lm_ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * lm_ggml_backend_meta_buffer_type_get_name(lm_ggml_backend_buffer_type_t buft) {
    LM_GGML_ASSERT(lm_ggml_backend_buft_is_meta(buft));
    const lm_ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const lm_ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_meta_buft_simple_buft(lm_ggml_backend_buffer_type_t meta_buft, size_t index) {
    LM_GGML_ASSERT(lm_ggml_backend_buft_is_meta(meta_buft));
    const lm_ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const lm_ggml_backend_meta_buffer_type_context *) meta_buft->context;
    LM_GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static lm_ggml_backend_buffer_t lm_ggml_backend_meta_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size);

static size_t lm_ggml_backend_meta_buffer_type_get_alignment(lm_ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = lm_ggml_backend_buft_get_alignment(lm_ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        LM_GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t lm_ggml_backend_meta_buffer_type_get_max_size(lm_ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, lm_ggml_backend_buft_get_max_size(lm_ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t lm_ggml_backend_meta_buffer_type_get_alloc_size(lm_ggml_backend_buffer_type_t buft, const lm_ggml_tensor * tensor) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = lm_ggml_backend_buft_get_alloc_size(lm_ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool lm_ggml_backend_meta_buffer_type_is_host(lm_ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!lm_ggml_backend_buft_is_host(lm_ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct lm_ggml_backend_buffer_type_i lm_ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ lm_ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ lm_ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ lm_ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ lm_ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ lm_ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ lm_ggml_backend_meta_buffer_type_is_host,
};

bool lm_ggml_backend_buft_is_meta(lm_ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == lm_ggml_backend_meta_buffer_type_iface.get_name;
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_meta_device_get_buffer_type(lm_ggml_backend_dev_t dev) {
    static std::map<lm_ggml_backend_dev_t, struct lm_ggml_backend_buffer_type> meta_bufts;
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = lm_ggml_backend_meta_dev_n_devs(dev);
    std::vector<lm_ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(lm_ggml_backend_dev_buffer_type(lm_ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    lm_ggml_backend_meta_buffer_type_context * buft_ctx = new lm_ggml_backend_meta_buffer_type_context(simple_bufts);

    struct lm_ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ lm_ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static lm_ggml_backend_buffer_type_t lm_ggml_backend_meta_device_get_host_buffer_type(lm_ggml_backend_dev_t dev) {
    LM_GGML_ASSERT(lm_ggml_backend_dev_is_meta(dev));
    const lm_ggml_backend_meta_device_context * meta_dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;

    lm_ggml_backend_buffer_type_t host_buft = nullptr;
    for (lm_ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        lm_ggml_backend_buffer_type_t simple_host_buft = lm_ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

struct lm_ggml_backend_meta_buffer_context {
    static constexpr size_t nbtc = LM_GGML_TENSOR_SIZE - sizeof(lm_ggml_tensor::padding);

    std::map<std::pair<const lm_ggml_tensor *, bool>, std::pair<lm_ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
    std::map<          const lm_ggml_tensor *,        std::vector<lm_ggml_tensor *>>                           simple_tensors;

    struct buffer_config {
        lm_ggml_context          * ctx;
        lm_ggml_backend_buffer_t   buf;

        buffer_config(lm_ggml_context * ctx, lm_ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;

    int debug;

    lm_ggml_backend_meta_buffer_context() {
        const char * LM_GGML_META_DEBUG = getenv("LM_GGML_META_DEBUG");
        debug = LM_GGML_META_DEBUG ? atoi(LM_GGML_META_DEBUG) : 0;
    }
};

static void lm_ggml_backend_meta_buffer_free_buffer(lm_ggml_backend_buffer_t buffer) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_meta(buffer));
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        lm_ggml_backend_buffer_free(buf);
        lm_ggml_free(ctx);
    }
    delete buf_ctx;
}

static size_t lm_ggml_backend_meta_buffer_n_bufs(lm_ggml_backend_buffer_t meta_buf) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_meta(meta_buf));
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

static lm_ggml_backend_buffer_t lm_ggml_backend_meta_buffer_simple_buffer(lm_ggml_backend_buffer_t meta_buf, size_t index) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_meta(meta_buf));
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) meta_buf->context;
    LM_GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

static struct lm_ggml_tensor * lm_ggml_backend_meta_buffer_simple_tensor(const struct lm_ggml_tensor * tensor, size_t index) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_meta(tensor->buffer));
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) tensor->buffer->context;
    LM_GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct lm_ggml_backend_meta_split_state lm_ggml_backend_meta_get_split_state(const struct lm_ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = lm_ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const lm_ggml_backend_meta_split_state & a, const lm_ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> lm_ggml_backend_meta_split_state {
        lm_ggml_backend_meta_split_state ret = {LM_GGML_BACKEND_SPLIT_AXIS_NONE, {0}, 1};
        for (size_t i = 0; i < LM_GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == LM_GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
                break;
            }
        }
        if (ret.axis == LM_GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < LM_GGML_MAX_DIMS) {
            ret = {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        LM_GGML_ASSERT(ret.axis != LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        LM_GGML_ASSERT(src_ss[0].axis != LM_GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < LM_GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // LM_GGML_OP_ADD_ID
        }
        LM_GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        const lm_ggml_backend_meta_split_axis concat_axis = lm_ggml_backend_meta_split_axis(lm_ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < LM_GGML_MAX_DIMS) {
            LM_GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < LM_GGML_MAX_DIMS) {
            LM_GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1};
        }
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            lm_ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = LM_GGML_BACKEND_SPLIT_AXIS_0;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            lm_ggml_backend_meta_split_state ret = src_ss[1];
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_0) {
            LM_GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED : LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, 1};
        }
        LM_GGML_ABORT("fatal error");
        //return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
    };

    auto handle_cpy = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < LM_GGML_MAX_DIMS) {
            int64_t ne_split_src = tensor->src[0]->ne[0];
            for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                ne_split_src *= tensor->src[0]->ne[dim];
            }
            int64_t ne_split_dst = 1;
            for (int dim = 0; dim < LM_GGML_MAX_DIMS; dim++) {
                ne_split_dst *= tensor->ne[dim];
                if (ne_split_dst == ne_split_src) {
                    return {lm_ggml_backend_meta_split_axis(dim), {0}, 1};
                }
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_reshape = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case LM_GGML_BACKEND_SPLIT_AXIS_0:
            case LM_GGML_BACKEND_SPLIT_AXIS_1:
            case LM_GGML_BACKEND_SPLIT_AXIS_2:
            case LM_GGML_BACKEND_SPLIT_AXIS_3: {
                LM_GGML_ASSERT(!lm_ggml_is_permuted(tensor) && !lm_ggml_is_permuted(tensor->src[0]));
                if (src_ss[0].axis == lm_ggml_n_dims(tensor->src[0]) - 1) {
                    return {lm_ggml_backend_meta_split_axis(lm_ggml_n_dims(tensor) - 1), {0}, 1};
                }
                std::vector<int64_t> base_ne_in;
                base_ne_in.reserve(LM_GGML_MAX_DIMS - src_ss[0].axis);
                {
                    base_ne_in.push_back(1);
                    int dim = 0;
                    for (; dim <= src_ss[0].axis; dim++) {
                        base_ne_in[0] *= tensor->src[0]->ne[dim];
                    }
                    for (; dim <= LM_GGML_MAX_DIMS; dim++) {
                        base_ne_in.push_back(base_ne_in.back() * tensor->src[0]->ne[dim]);
                    }
                }
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < LM_GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    for (const int64_t & bni : base_ne_in) {
                        if (bni == base_ne_out_next) {
                            return {lm_ggml_backend_meta_split_axis(dim), {0}, 1};
                        }
                    }
                    if (base_ne_out_next > base_ne_in[0]) {
                        LM_GGML_ASSERT(dim + 1 < LM_GGML_MAX_DIMS);
                        return {lm_ggml_backend_meta_split_axis(dim + 1), {0}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                LM_GGML_ABORT("shape mismatch for %s", lm_ggml_op_name(tensor->op));
            }
            case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                LM_GGML_ABORT("fatal error");
                //return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_view = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (lm_ggml_is_contiguous(tensor) && lm_ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < LM_GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!lm_ggml_is_permuted(tensor) && !lm_ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < LM_GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < LM_GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {lm_ggml_backend_meta_split_axis(dim), {0}, 1};
                }
            }
            LM_GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        LM_GGML_ABORT("view of permuted tensor not implemented");
        //return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
    };

    auto handle_permute = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case LM_GGML_BACKEND_SPLIT_AXIS_0:
            case LM_GGML_BACKEND_SPLIT_AXIS_1:
            case LM_GGML_BACKEND_SPLIT_AXIS_2:
            case LM_GGML_BACKEND_SPLIT_AXIS_3: {
                return {lm_ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, 1};
            }
            case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                LM_GGML_ABORT("fatal error");
                //return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case LM_GGML_BACKEND_SPLIT_AXIS_0:
            case LM_GGML_BACKEND_SPLIT_AXIS_1: {
                return {lm_ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, 1};
            }
            case LM_GGML_BACKEND_SPLIT_AXIS_2:
            case LM_GGML_BACKEND_SPLIT_AXIS_3:
            case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                LM_GGML_ABORT("fatal error");
                //return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        LM_GGML_ASSERT(src_ss[0].axis != LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        LM_GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        LM_GGML_ASSERT(src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < LM_GGML_MAX_DIMS) {
            LM_GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            LM_GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        LM_GGML_ASSERT(                             src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_2);
        LM_GGML_ASSERT(                             src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_2);
        LM_GGML_ASSERT(                             src_ss[2].axis == LM_GGML_BACKEND_SPLIT_AXIS_2);
        LM_GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        LM_GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == LM_GGML_BACKEND_SPLIT_AXIS_0);
        return {LM_GGML_BACKEND_SPLIT_AXIS_1, {0}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_0) {
                return {LM_GGML_BACKEND_SPLIT_AXIS_1, {0}, 1};
            }
            if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_1) {
                return {LM_GGML_BACKEND_SPLIT_AXIS_0, {0}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<lm_ggml_backend_meta_split_state> & src_ss) -> lm_ggml_backend_meta_split_state {
        if (src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[2].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[4].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        LM_GGML_ASSERT(src_ss[0].axis == LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[1].axis == LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[2].axis == LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[3].axis == LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[4].axis == LM_GGML_BACKEND_SPLIT_AXIS_1);
        LM_GGML_ASSERT(src_ss[5].axis == LM_GGML_BACKEND_SPLIT_AXIS_2);
        return {LM_GGML_BACKEND_SPLIT_AXIS_0, {0}, 1};
    };

    auto calculate_split_state = [&]() -> lm_ggml_backend_meta_split_state {
        if (lm_ggml_nelements(tensor) == 0) {
            return {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        if (lm_ggml_backend_buffer_get_usage(tensor->buffer) != LM_GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_buft_get_device(lm_ggml_backend_buffer_get_type(tensor->buffer));
            const lm_ggml_backend_meta_device_context * dev_ctx = (const lm_ggml_backend_meta_device_context *) dev->context;
            lm_ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= LM_GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == LM_GGML_BACKEND_SPLIT_AXIS_0 ? lm_ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
                    LM_GGML_ASSERT(ret.ne[sj] % granularity == 0);
                    ne_sum += ret.ne[sj];
                }
                LM_GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<lm_ggml_backend_meta_split_state> src_ss(LM_GGML_MAX_SRC, {LM_GGML_BACKEND_SPLIT_AXIS_NONE, {0}, 1});
        for (size_t i = 0; i < LM_GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
                continue;
            }
            src_ss[i] = lm_ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
            LM_GGML_ASSERT(src_ss[i].axis != LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        lm_ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case LM_GGML_OP_NONE: {
                split_state = {LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1};
            } break;
            case LM_GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_ADD:
            case LM_GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case LM_GGML_OP_ADD1:
            case LM_GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SUB:
            case LM_GGML_OP_MUL:
            case LM_GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case LM_GGML_OP_SQR:
            case LM_GGML_OP_SQRT:
            case LM_GGML_OP_LOG:
            case LM_GGML_OP_SIN:
            case LM_GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SUM_ROWS:
            case LM_GGML_OP_CUMSUM:
            case LM_GGML_OP_MEAN:
            case LM_GGML_OP_ARGMAX:
            case LM_GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case LM_GGML_OP_REPEAT:
            case LM_GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case LM_GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_NORM:
            case LM_GGML_OP_RMS_NORM:
            case LM_GGML_OP_RMS_NORM_BACK:
            case LM_GGML_OP_GROUP_NORM:
            case LM_GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case LM_GGML_OP_MUL_MAT:
            case LM_GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case LM_GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case LM_GGML_OP_CONT:
            case LM_GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case LM_GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case LM_GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case LM_GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case LM_GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case LM_GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case LM_GGML_OP_DIAG:
            case LM_GGML_OP_DIAG_MASK_INF:
            case LM_GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SOFT_MAX:
            case LM_GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case LM_GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_CONV_TRANSPOSE_1D:
            case LM_GGML_OP_IM2COL:
            case LM_GGML_OP_IM2COL_BACK:
            case LM_GGML_OP_IM2COL_3D:
            case LM_GGML_OP_CONV_2D:
            case LM_GGML_OP_CONV_3D:
            case LM_GGML_OP_CONV_2D_DW:
            case LM_GGML_OP_CONV_TRANSPOSE_2D:
            case LM_GGML_OP_POOL_1D:
            case LM_GGML_OP_POOL_2D:
            case LM_GGML_OP_POOL_2D_BACK:
            case LM_GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case LM_GGML_OP_PAD_REFLECT_1D:
            case LM_GGML_OP_ROLL:
            case LM_GGML_OP_ARANGE:
            case LM_GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_ARGSORT:
            case LM_GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case LM_GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case LM_GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case LM_GGML_OP_SSM_SCAN:
            case LM_GGML_OP_WIN_PART:
            case LM_GGML_OP_WIN_UNPART:
            case LM_GGML_OP_GET_REL_POS:
            case LM_GGML_OP_ADD_REL_POS:
            case LM_GGML_OP_RWKV_WKV6:
            case LM_GGML_OP_GATED_LINEAR_ATTN:
            case LM_GGML_OP_RWKV_WKV7:
            case LM_GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case LM_GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case LM_GGML_OP_MAP_CUSTOM1:
            case LM_GGML_OP_MAP_CUSTOM2:
            case LM_GGML_OP_MAP_CUSTOM3:
            case LM_GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case LM_GGML_OP_CROSS_ENTROPY_LOSS:
            case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case LM_GGML_OP_OPT_STEP_ADAMW:
            case LM_GGML_OP_OPT_STEP_SGD:
            case LM_GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                LM_GGML_ABORT("ggml op not implemented: %s", lm_ggml_op_name(tensor->op));
                split_state = {LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < LM_GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = lm_ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < LM_GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= LM_GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            LM_GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_ss[i].axis] == 0);
                            split_state.ne[j] /= tensor->src[i]->ne[src_ss[i].axis];
                        }
                    }
                } else {
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j];
                        }
                        // Assert that ratio is consistent:
                        LM_GGML_ASSERT(split_state.ne[j] * tensor->src[i]->ne[src_ss[i].axis]
                                               == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            LM_GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < LM_GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor->src[0], true);
                const char * axis_name = lm_ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + lm_ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(buf_ctx->split_state_cache[key].first.ne[j]);
            }
            LM_GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, lm_ggml_op_name(tensor->op),
                lm_ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    lm_ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    LM_GGML_ASSERT(ret.axis != LM_GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < LM_GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
            ne_ret += ret.ne[sj];
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static void * lm_ggml_backend_meta_buffer_get_base(lm_ggml_backend_buffer_t buffer) {
    LM_GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum lm_ggml_status lm_ggml_backend_meta_buffer_init_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(lm_ggml_backend_buffer_is_meta(buffer));
    lm_ggml_backend_meta_buffer_context * buf_ctx = (lm_ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = lm_ggml_backend_meta_buffer_n_bufs(buffer);

    const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    LM_GGML_ASSERT(lm_ggml_nelements(tensor) == 0 || split_state.axis != LM_GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    LM_GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[LM_GGML_MAX_DIMS];
    size_t  nb[LM_GGML_MAX_DIMS];
    for (size_t k = 0; k < LM_GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<lm_ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        lm_ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        lm_ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        if (split_dim >= 0 && split_dim < LM_GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // LM_GGML_ASSERT(lm_ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j];
            }
            for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        lm_ggml_tensor * t_ij = lm_ggml_new_tensor(simple_ctx, tensor->type, LM_GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        lm_ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && lm_ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = lm_ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < LM_GGML_MAX_DIMS) {
                LM_GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = lm_ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                LM_GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < LM_GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) lm_ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(lm_ggml_backend_buffer_get_base(buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && lm_ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = lm_ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < LM_GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !lm_ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const lm_ggml_backend_meta_split_state split_state_src = lm_ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= LM_GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~LM_GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    buf_ctx->simple_tensors[tensor] = simple_tensors;

    return LM_GGML_STATUS_SUCCESS;
}

static void lm_ggml_backend_meta_buffer_set_tensor(lm_ggml_backend_buffer_t buffer, lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = lm_ggml_backend_meta_buffer_n_bufs(buffer);
    LM_GGML_ASSERT(lm_ggml_is_contiguous(tensor));

    const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (split_state.n_segments != 1) {
        LM_GGML_ASSERT(split_state.axis >= 0 && split_state.axis < LM_GGML_MAX_DIMS);
        LM_GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_0) {
            LM_GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            LM_GGML_ASSERT(offset % row_stride == 0);
            LM_GGML_ASSERT(size   % row_stride == 0);
            const int64_t r_start = offset / row_stride;
            const int64_t r_count = size   / row_stride;
            LM_GGML_ASSERT(r_start + r_count <= tensor->ne[1]);

            const int64_t blck_size = lm_ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    LM_GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                    lm_ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
            LM_GGML_ASSERT(offset_data*r_count == size);
            return;
        }
        LM_GGML_ASSERT(split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        LM_GGML_ASSERT(offset % row_stride == 0);
        LM_GGML_ASSERT(size   % row_stride == 0);
        const int64_t r_start = offset / row_stride;
        const int64_t r_count = size   / row_stride;
        LM_GGML_ASSERT(r_start + r_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                lm_ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                offset_data       += nbytes;
                simple_offsets[j] += nbytes;
            }
        }
        LM_GGML_ASSERT(offset_data*r_count == size);
        return;
    }

    switch (split_state.axis) {
        case LM_GGML_BACKEND_SPLIT_AXIS_0:
        case LM_GGML_BACKEND_SPLIT_AXIS_1:
        case LM_GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            LM_GGML_ASSERT(offset % chunk_size_full == 0);
            LM_GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                lm_ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            LM_GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                lm_ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            LM_GGML_ASSERT(tensor->type == LM_GGML_TYPE_F32);
            const int64_t ne = lm_ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                lm_ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            LM_GGML_ABORT("fatal error");
        }
    }
}

static void lm_ggml_backend_meta_buffer_get_tensor(lm_ggml_backend_buffer_t buffer, const lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = lm_ggml_backend_meta_buffer_n_bufs(buffer);
    LM_GGML_ASSERT(lm_ggml_is_contiguous(tensor));

    const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (split_state.n_segments != 1) {
        LM_GGML_ASSERT(split_state.axis >= 0 && split_state.axis < LM_GGML_MAX_DIMS);
        LM_GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_0) {
            LM_GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            LM_GGML_ASSERT(offset % row_stride == 0);
            LM_GGML_ASSERT(size   % row_stride == 0);
            const int64_t r_start = offset / row_stride;
            const int64_t r_count = size   / row_stride;
            LM_GGML_ASSERT(r_start + r_count <= tensor->ne[1]);

            const int64_t blck_size = lm_ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    LM_GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                    lm_ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
            LM_GGML_ASSERT(offset_data*r_count == size);
            return;
        }
        LM_GGML_ASSERT(split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        LM_GGML_ASSERT(offset % row_stride == 0);
        LM_GGML_ASSERT(size   % row_stride == 0);
        const int64_t r_start = offset / row_stride;
        const int64_t r_count = size   / row_stride;
        LM_GGML_ASSERT(r_start + r_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                lm_ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                offset_data       += nbytes;
                simple_offsets[j] += nbytes;
            }
        }
        LM_GGML_ASSERT(offset_data*r_count == size);
        return;
    }

    switch (split_state.axis) {
        case LM_GGML_BACKEND_SPLIT_AXIS_0:
        case LM_GGML_BACKEND_SPLIT_AXIS_1:
        case LM_GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            LM_GGML_ASSERT(offset % chunk_size_full == 0);
            LM_GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                lm_ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            LM_GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            lm_ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            LM_GGML_ABORT("fatal error");
        }
    }
}

static void lm_ggml_backend_meta_buffer_clear(lm_ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = lm_ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        lm_ggml_backend_buffer_clear(lm_ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void lm_ggml_backend_meta_buffer_reset(lm_ggml_backend_buffer_t buffer) {
    const size_t n_buffers = lm_ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        lm_ggml_backend_buffer_reset(lm_ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const lm_ggml_backend_buffer_i lm_ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ lm_ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ lm_ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ lm_ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ lm_ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ lm_ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ lm_ggml_backend_meta_buffer_clear,
    /* .reset           = */ lm_ggml_backend_meta_buffer_reset,
};

bool lm_ggml_backend_buffer_is_meta(lm_ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == lm_ggml_backend_meta_buffer_iface.free_buffer;
}

static lm_ggml_backend_buffer_t lm_ggml_backend_meta_buffer_type_alloc_buffer(lm_ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);

    lm_ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    lm_ggml_backend_meta_buffer_context * buf_ctx = new lm_ggml_backend_meta_buffer_context();
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        lm_ggml_backend_buffer_t simple_buf = lm_ggml_backend_buft_alloc_buffer(lm_ggml_backend_meta_buft_simple_buft(buft, i), size);
        max_size = std::max(max_size, lm_ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(lm_ggml_init(params), simple_buf);
    }

    return lm_ggml_backend_buffer_init(buft, lm_ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct lm_ggml_backend_buffer * lm_ggml_backend_meta_alloc_ctx_tensors_from_buft(struct lm_ggml_context * ctx, lm_ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = lm_ggml_backend_meta_buft_n_bufts(buft);

    lm_ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    lm_ggml_backend_meta_buffer_context * meta_buf_ctx = new lm_ggml_backend_meta_buffer_context();
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(lm_ggml_init(params), nullptr);
    }

    lm_ggml_backend_buffer_t meta_buf = lm_ggml_backend_buffer_init(buft, lm_ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (lm_ggml_tensor * t = lm_ggml_get_first_tensor(ctx); t != nullptr; t = lm_ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        lm_ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = lm_ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, lm_ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, lm_ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}

//
// meta backend
//

static lm_ggml_guid_t lm_ggml_backend_meta_guid() {
    static lm_ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct lm_ggml_backend_meta_context {
    struct cgraph_config {
        lm_ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<lm_ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        lm_ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<lm_ggml_tensor *>           nodes;
        std::vector<lm_ggml_backend_buffer_ptr> bufs;

        backend_config(lm_ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    lm_ggml_context_ptr            ctx;
    std::vector<lm_ggml_cgraph *>  cgraphs_aux;
    std::vector<lm_ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    lm_ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    lm_ggml_backend_meta_context(lm_ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = lm_ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<lm_ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            lm_ggml_backend_dev_t simple_dev = lm_ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += lm_ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(lm_ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            lm_ggml_backend_comm_init_t comm_init = (lm_ggml_backend_comm_init_t) lm_ggml_backend_reg_get_proc_address(
                lm_ggml_backend_dev_backend_reg(lm_ggml_backend_get_device(simple_backends[0])), "lm_ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (lm_ggml_backend_comm_allreduce_tensor_t)
                lm_ggml_backend_reg_get_proc_address(lm_ggml_backend_dev_backend_reg(
                    lm_ggml_backend_get_device(simple_backends[0])), "lm_ggml_backend_comm_allreduce_tensor");
            LM_GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~lm_ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            lm_ggml_backend_comm_free_t comm_free = (lm_ggml_backend_comm_free_t) lm_ggml_backend_reg_get_proc_address(
                lm_ggml_backend_dev_backend_reg(lm_ggml_backend_get_device(backend_configs[0].backend)), "lm_ggml_backend_comm_free");
            LM_GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            lm_ggml_backend_free(bc.backend);
        }
    }
};

static const char * lm_ggml_backend_meta_get_name(lm_ggml_backend_t backend) {
    LM_GGML_ASSERT(lm_ggml_backend_is_meta(backend));
    const lm_ggml_backend_meta_context * backend_ctx = (const lm_ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void lm_ggml_backend_meta_free(lm_ggml_backend_t backend) {
    LM_GGML_ASSERT(lm_ggml_backend_is_meta(backend));
    lm_ggml_backend_meta_context * backend_ctx = (lm_ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void lm_ggml_backend_meta_set_tensor_async(lm_ggml_backend_t backend, lm_ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = lm_ggml_backend_meta_n_backends(backend);
    LM_GGML_ASSERT(offset == 0);
    LM_GGML_ASSERT(lm_ggml_is_contiguous(tensor));

    const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    LM_GGML_ASSERT(split_state.n_segments == 1);

    switch (split_state.axis) {
        case LM_GGML_BACKEND_SPLIT_AXIS_0:
        case LM_GGML_BACKEND_SPLIT_AXIS_1:
        case LM_GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            LM_GGML_ASSERT(offset % chunk_size_full == 0);
            LM_GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                lm_ggml_backend_t simple_backend = lm_ggml_backend_meta_simple_backend(backend, j);
                lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                lm_ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            LM_GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                lm_ggml_backend_tensor_set_async(
                    lm_ggml_backend_meta_simple_backend(backend, j), lm_ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            LM_GGML_ABORT("fatal error");
        }
    }
}

static void lm_ggml_backend_meta_get_tensor_async(lm_ggml_backend_t backend, const lm_ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = lm_ggml_backend_meta_n_backends(backend);
    LM_GGML_ASSERT(offset == 0);
    LM_GGML_ASSERT(lm_ggml_is_contiguous(tensor));

    const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    LM_GGML_ASSERT(split_state.n_segments == 1);

    switch (split_state.axis) {
        case LM_GGML_BACKEND_SPLIT_AXIS_0:
        case LM_GGML_BACKEND_SPLIT_AXIS_1:
        case LM_GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            LM_GGML_ASSERT(offset % chunk_size_full == 0);
            LM_GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                lm_ggml_backend_t simple_backend = lm_ggml_backend_meta_simple_backend(backend, j);
                const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                lm_ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            LM_GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            lm_ggml_backend_t simple_backend = lm_ggml_backend_meta_simple_backend(backend, 0);
            const lm_ggml_tensor * simple_tensor = lm_ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            lm_ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            LM_GGML_ABORT("fatal error");
        }
    }
}

static void lm_ggml_backend_meta_synchronize(lm_ggml_backend_t backend) {
    const size_t n_backends = lm_ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        lm_ggml_backend_synchronize(lm_ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum lm_ggml_status lm_ggml_backend_meta_graph_compute(lm_ggml_backend_t backend, struct lm_ggml_cgraph * cgraph) {
    LM_GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = lm_ggml_backend_meta_n_backends(backend);
    lm_ggml_backend_meta_context * backend_ctx = (lm_ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                lm_ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == LM_GGML_OP_NONE && lm_ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call lm_ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = lm_ggml_backend_meta_buffer_simple_tensor(node, j);
                LM_GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                lm_ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = lm_ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        lm_ggml_tensor * next = cgraph->nodes[id+1];
                        if (lm_ggml_backend_meta_get_split_state(next, false).axis != LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < LM_GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (lm_ggml_backend_meta_get_split_state(next->src[s], false).axis != LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    lm_ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == LM_GGML_OP_ADD_ID && next->src[0] == node &&
                            lm_ggml_backend_meta_get_split_state(next->src[1], false).axis == LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            lm_ggml_backend_meta_get_split_state(next->src[2], false).axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = lm_ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    lm_ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == LM_GGML_OP_MUL && next->src[0] == node &&
                            lm_ggml_backend_meta_get_split_state(next->src[1], false).axis == LM_GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = lm_ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    lm_ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != LM_GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            lm_ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    lm_ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != LM_GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || lm_ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    lm_ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != LM_GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || lm_ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                lm_ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == LM_GGML_OP_NONE && lm_ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const lm_ggml_backend_meta_split_state split_state = lm_ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, lm_ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == LM_GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & LM_GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~LM_GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            LM_GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(lm_ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*lm_ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*lm_ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*lm_ggml_tensor_overhead();
            lm_ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(lm_ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = lm_ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = lm_ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = lm_ggml_new_tensor_1d(backend_ctx->ctx.get(), LM_GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                lm_ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                lm_ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    lm_ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = lm_ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = lm_ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = lm_ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](lm_ggml_tensor * t) -> lm_ggml_tensor * {
        lm_ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(lm_ggml_tensor));
        ret->op   = LM_GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < LM_GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](lm_ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        lm_ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || lm_ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(lm_ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = lm_ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> lm_ggml_cgraph * {
        lm_ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> lm_ggml_status {
        std::vector<lm_ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            lm_ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & LM_GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            lm_ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = LM_GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            lm_ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->flags |= LM_GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const lm_ggml_status status = lm_ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != LM_GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            lm_ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            lm_ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            LM_GGML_ASSERT(lm_ggml_is_contiguous(node_src));
            LM_GGML_ASSERT(lm_ggml_is_contiguous(node_dst));

            lm_ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            lm_ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            lm_ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = LM_GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= LM_GGML_TENSOR_FLAG_COMPUTE;
            lm_ggml_backend_view_init(node_red);

            lm_ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const lm_ggml_status status = lm_ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != LM_GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const lm_ggml_status status = lm_ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != LM_GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            lm_ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            lm_ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            lm_ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return LM_GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const lm_ggml_status status = lm_ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != LM_GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<lm_ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    lm_ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const lm_ggml_status status = allreduce_fallback(i);
                if (status != LM_GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return LM_GGML_STATUS_SUCCESS;
}

static const lm_ggml_backend_i lm_ggml_backend_meta_i = {
    /* .get_name                = */ lm_ggml_backend_meta_get_name,
    /* .free                    = */ lm_ggml_backend_meta_free,
    /* .set_tensor_async        = */ lm_ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ lm_ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ lm_ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ lm_ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool lm_ggml_backend_is_meta(lm_ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == lm_ggml_backend_meta_i.get_name;
}

static lm_ggml_backend_t lm_ggml_backend_meta_device_init_backend(lm_ggml_backend_dev_t dev, const char * params) {
    lm_ggml_backend_meta_context * backend_ctx = new lm_ggml_backend_meta_context(dev, params);

    lm_ggml_backend_t backend = new struct lm_ggml_backend;
    backend->guid    = lm_ggml_backend_meta_guid();
    backend->iface   = lm_ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t lm_ggml_backend_meta_n_backends(lm_ggml_backend_t meta_backend) {
    LM_GGML_ASSERT(lm_ggml_backend_is_meta(meta_backend));
    const lm_ggml_backend_meta_context * backend_ctx = (const lm_ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

lm_ggml_backend_t lm_ggml_backend_meta_simple_backend(lm_ggml_backend_t meta_backend, size_t index) {
    LM_GGML_ASSERT(lm_ggml_backend_is_meta(meta_backend));
    const lm_ggml_backend_meta_context * backend_ctx = (const lm_ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

