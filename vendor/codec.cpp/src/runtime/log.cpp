#include "codec.h"

#include <ggml.h>

#include <cstdarg>
#include <cstdio>
#include <vector>

namespace {

void default_log(enum codec_log_level level, const char * text, void *) {
    static enum codec_log_level last = CODEC_LOG_LEVEL_INFO;
    if (level != CODEC_LOG_LEVEL_CONT) last = level;
    FILE * f = (last == CODEC_LOG_LEVEL_WARN || last == CODEC_LOG_LEVEL_ERROR) ? stderr : stdout;
    std::fputs(text, f);
    std::fflush(f);
}

codec_log_callback g_cb   = default_log;
void *             g_user = nullptr;

void ggml_to_codec(enum ggml_log_level level, const char * text, void *) {
    g_cb((enum codec_log_level) level, text, g_user);
}

}  // namespace

extern "C" void codec_log_set(codec_log_callback callback, void * user_data) {
    g_cb   = callback != nullptr ? callback : default_log;
    g_user = callback != nullptr ? user_data : nullptr;
    // The backends libcodec runs on log through ggml: route them here too.
    if (callback != nullptr) ggml_log_set(ggml_to_codec, nullptr);
    else                     ggml_log_set(nullptr, nullptr);
}

extern "C" void codec_log(enum codec_log_level level, const char * fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n >= (int) sizeof(buf)) {
        std::vector<char> big((size_t) n + 1);
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        g_cb(level, big.data(), g_user);
    } else if (n >= 0) {
        g_cb(level, buf, g_user);
    }
    va_end(ap2);
}
