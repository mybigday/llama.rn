#ifndef CODEC_RUNTIME_PERF_LOG_H
#define CODEC_RUNTIME_PERF_LOG_H

#include <cstdint>
#include <string>

// Lightweight phase-timing hook.  When the `CODEC_PERF_LOG` environment
// variable is set to a file path, each completed `codec_perf_scope`
// instance appends a JSON-lines record like:
//
//   {"phase":"graph_compute","wall_us":1234,"detail":"kind=12"}
//
// to that file.  When the env var is unset (the common case) the scope is
// effectively a single `lm_ggml_time_us()` call at construction + a no-op at
// destruction — suitable for keeping in hot paths.
//
// Use the `CODEC_PERF_SCOPE` macro to record a phase with its enclosing
// scope's lifetime.  Pass an optional `detail` string for caller-specific
// context (graph kind, n_frames, etc.).

struct codec_perf_scope {
    const char * phase;
    int64_t t0_us;
    std::string detail;

    explicit codec_perf_scope(const char * phase_name);
    codec_perf_scope(const char * phase_name, std::string detail_str);
    ~codec_perf_scope();

    codec_perf_scope(const codec_perf_scope &)             = delete;
    codec_perf_scope & operator=(const codec_perf_scope &) = delete;
};

// Append an arbitrary structured event to the perf log (no implicit timing).
// Useful for one-off snapshots like "eval_arena_bytes=N" or cache-hit/miss.
void codec_perf_event(const char * phase, const std::string & detail);

#define CODEC_PERF_TOKEN_PASTE2(a, b) a##b
#define CODEC_PERF_TOKEN_PASTE(a, b)  CODEC_PERF_TOKEN_PASTE2(a, b)
#define CODEC_PERF_SCOPE(name)        codec_perf_scope CODEC_PERF_TOKEN_PASTE(_codec_perf_, __LINE__)((name))
#define CODEC_PERF_SCOPE_D(name, det) codec_perf_scope CODEC_PERF_TOKEN_PASTE(_codec_perf_, __LINE__)((name), (det))

#endif // CODEC_RUNTIME_PERF_LOG_H
