#include "perf_log.h"

#include <ggml.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

// Sticky cache for the env var: read once, sampled cheaply on each call.
// `nullptr` means "no path" (env var unset).
struct perf_log_state {
    const char * path = nullptr;   // owned by the env table
    bool         resolved = false;
    std::mutex   io_mu;
};

perf_log_state & state() {
    static perf_log_state s;
    return s;
}

const char * resolve_path() {
    auto & s = state();
    if (!s.resolved) {
        s.path = std::getenv("CODEC_PERF_LOG");
        // Treat empty string as unset.
        if (s.path != nullptr && s.path[0] == '\0') {
            s.path = nullptr;
        }
        s.resolved = true;
    }
    return s.path;
}

void escape_json(const std::string & in, std::string * out) {
    out->clear();
    out->reserve(in.size() + 2);
    for (char c : in) {
        switch (c) {
            case '"':  *out += "\\\""; break;
            case '\\': *out += "\\\\"; break;
            case '\n': *out += "\\n";  break;
            case '\r': *out += "\\r";  break;
            case '\t': *out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    *out += buf;
                } else {
                    *out += c;
                }
        }
    }
}

void emit_event(const char * phase, int64_t wall_us, const std::string & detail) {
    const char * path = resolve_path();
    if (path == nullptr || phase == nullptr) return;

    std::string esc_detail;
    escape_json(detail, &esc_detail);

    char head[128];
    int n;
    if (wall_us >= 0) {
        n = std::snprintf(head, sizeof(head),
                          "{\"phase\":\"%s\",\"wall_us\":%lld,\"detail\":\"",
                          phase, (long long) wall_us);
    } else {
        n = std::snprintf(head, sizeof(head),
                          "{\"phase\":\"%s\",\"detail\":\"",
                          phase);
    }
    if (n < 0) return;

    auto & s = state();
    std::lock_guard<std::mutex> lock(s.io_mu);
    FILE * f = std::fopen(path, "ab");
    if (f == nullptr) return;
    std::fwrite(head, 1, (size_t) n, f);
    std::fwrite(esc_detail.data(), 1, esc_detail.size(), f);
    std::fwrite("\"}\n", 1, 3, f);
    std::fclose(f);
}

}  // namespace

codec_perf_scope::codec_perf_scope(const char * phase_name)
    : phase(phase_name), t0_us(lm_ggml_time_us()) {}

codec_perf_scope::codec_perf_scope(const char * phase_name, std::string detail_str)
    : phase(phase_name), t0_us(lm_ggml_time_us()), detail(std::move(detail_str)) {}

codec_perf_scope::~codec_perf_scope() {
    if (resolve_path() == nullptr) return;
    const int64_t dt = lm_ggml_time_us() - t0_us;
    emit_event(phase, dt, detail);
}

void codec_perf_event(const char * phase, const std::string & detail) {
    if (resolve_path() == nullptr) return;
    emit_event(phase, /*wall_us=*/-1, detail);
}
