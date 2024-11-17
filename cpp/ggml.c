#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-threading.h"
#include "ggml.h"

// FIXME: required here for quantization functions
#include "ggml-quants.h"
#include "ggml-aarch64.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#endif

#define UNUSED LM_GGML_UNUSED

#if defined(_MSC_VER)
#define m512bh(p) p
#define m512i(p) p
#else
#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)
#endif

// precomputed f32 table for f16 (256 KB) (ggml-impl.h)
float lm_ggml_table_f32_f16[1 << 16];

#if (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && \
    (!defined(TARGET_OS_TV) && !defined(TARGET_OS_WATCH))
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>

#if defined(__ANDROID__)
#include <unwind.h>
#include <dlfcn.h>
#include <stdio.h>

struct backtrace_state {
    void ** current;
    void ** end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context* context, void* arg) {
    struct backtrace_state * state = (struct backtrace_state *)arg;
    uintptr_t pc = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) {
            return _URC_END_OF_STACK;
        } else {
            *state->current++ = (void*)pc;
        }
    }
    return _URC_NO_REASON;
}

static void lm_ggml_print_backtrace_symbols(void) {
    const int max = 100;
    void* buffer[max];

    struct backtrace_state state = {buffer, buffer + max};
    _Unwind_Backtrace(unwind_callback, &state);

    int count = state.current - buffer;

    for (int idx = 0; idx < count; ++idx) {
        const void * addr = buffer[idx];
        const char * symbol = "";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            symbol = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", idx, addr, symbol);
    }
}
#elif defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
static void lm_ggml_print_backtrace_symbols(void) {
    // void * trace[100];
    // int nptrs = backtrace(trace, sizeof(trace)/sizeof(trace[0]));
    // backtrace_symbols_fd(trace, nptrs, STDERR_FILENO);
}
#else
static void lm_ggml_print_backtrace_symbols(void) {
    // platform not supported
}
#endif

static void lm_ggml_print_backtrace(void) {
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        // try gdb
        execlp("gdb", "gdb", "--batch",
            "-ex", "set style enabled on",
            "-ex", attach,
            "-ex", "bt -frame-info source-and-location",
            "-ex", "detach",
            "-ex", "quit",
            (char *) NULL);
        // try lldb
        execlp("lldb", "lldb", "--batch",
            "-o", "bt",
            "-o", "quit",
            "-p", attach,
            (char *) NULL);
        exit(EXIT_FAILURE);
    } else {
        int wstatus;
        waitpid(pid, &wstatus, 0);
        if (WIFEXITED(wstatus)) {
            if (WEXITSTATUS(wstatus) == EXIT_FAILURE) {
                // gdb failed, fallback to backtrace_symbols
                lm_ggml_print_backtrace_symbols();
            }
        }
    }
}
#else
static void lm_ggml_print_backtrace(void) {
    // platform not supported
}
#endif

void lm_ggml_abort(const char * file, int line, const char * fmt, ...) {
    fflush(stdout);

    fprintf(stderr, "%s:%d: ", file, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");

    lm_ggml_print_backtrace();
    abort();
}

//
// logging
//

struct lm_ggml_logger_state {
    lm_ggml_log_callback log_callback;
    void * log_callback_user_data;
};
static struct lm_ggml_logger_state g_logger_state = {lm_ggml_log_callback_default, NULL};

static void lm_ggml_log_internal_v(enum lm_ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

void lm_ggml_log_internal(enum lm_ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    lm_ggml_log_internal_v(level, format, args);
    va_end(args);
}

void lm_ggml_log_callback_default(enum lm_ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

//
// end of logging block
//

#ifdef LM_GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define LM_GGML_SOFT_MAX_ACCELERATE
#endif


void * lm_ggml_aligned_malloc(size_t size) {
    const int alignment = 64;

#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
#else
    if (size == 0) {
        LM_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for lm_ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
  #ifdef LM_GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, alignment, size);
  #elif TARGET_OS_OSX
    LM_GGML_UNUSED(alignment);
    kern_return_t alloc_status = vm_allocate((vm_map_t) mach_task_self(), (vm_address_t *) &aligned_memory, size, VM_FLAGS_ANYWHERE);
    int result = EFAULT;
    switch (alloc_status) {
        case KERN_SUCCESS:
            result = 0;
            break;
        case KERN_INVALID_ADDRESS:
            result = EINVAL;
            break;
        case KERN_NO_SPACE:
            result = ENOMEM;
            break;
        default:
            result = EFAULT;
            break;
    }
  #else
    int result = posix_memalign(&aligned_memory, alignment, size);
  #endif
    if (result != 0) {
        // Handle allocation failure
        const char *error_desc = "unknown allocation error";
        switch (result) {
            case EINVAL:
                error_desc = "invalid alignment value";
                break;
            case ENOMEM:
                error_desc = "insufficient memory";
                break;
        }
        LM_GGML_LOG_ERROR("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        return NULL;
    }
    return aligned_memory;
#endif
}

void lm_ggml_aligned_free(void * ptr, size_t size) {
    LM_GGML_UNUSED(size);
#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#elif LM_GGML_USE_CPU_HBM
    if (ptr != NULL) {
        hbw_free(ptr);
    }
#elif TARGET_OS_OSX
    if (ptr != NULL) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
    }
#else
    free(ptr);
#endif
}


inline static void * lm_ggml_malloc(size_t size) {
    if (size == 0) {
        LM_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for lm_ggml_malloc!\n");
        return NULL;
    }
    void * result = malloc(size);
    if (result == NULL) {
        LM_GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        LM_GGML_ABORT("fatal error");
    }
    return result;
}

// calloc
inline static void * lm_ggml_calloc(size_t num, size_t size) {
    if (num == 0 || size == 0) {
        LM_GGML_LOG_WARN("Behavior may be unexpected when allocating 0 bytes for lm_ggml_calloc!\n");
        return NULL;
    }
    void * result = calloc(num, size);
    if (result == NULL) {
        LM_GGML_LOG_ERROR("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        LM_GGML_ABORT("fatal error");
    }
    return result;
}

#define LM_GGML_MALLOC(size)      lm_ggml_malloc(size)
#define LM_GGML_CALLOC(num, size) lm_ggml_calloc(num, size)

#define LM_GGML_FREE(ptr) free(ptr)

const char * lm_ggml_status_to_string(enum lm_ggml_status status) {
    switch (status) {
        case LM_GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
        case LM_GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
        case LM_GGML_STATUS_SUCCESS:      return "GGML status: success";
        case LM_GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
    }

    return "GGML status: unknown";
}

float lm_ggml_fp16_to_fp32(lm_ggml_fp16_t x) {
#define lm_ggml_fp16_to_fp32 do_not_use__lm_ggml_fp16_to_fp32__in_ggml
    return LM_GGML_FP16_TO_FP32(x);
}

lm_ggml_fp16_t lm_ggml_fp32_to_fp16(float x) {
#define lm_ggml_fp32_to_fp16 do_not_use__lm_ggml_fp32_to_fp16__in_ggml
    return LM_GGML_FP32_TO_FP16(x);
}

float lm_ggml_bf16_to_fp32(lm_ggml_bf16_t x) {
#define lm_ggml_bf16_to_fp32 do_not_use__lm_ggml_bf16_to_fp32__in_ggml
    return LM_GGML_BF16_TO_FP32(x);  // it just left shifts
}

lm_ggml_bf16_t lm_ggml_fp32_to_bf16(float x) {
#define lm_ggml_fp32_to_bf16 do_not_use__lm_ggml_fp32_to_bf16__in_ggml
    return LM_GGML_FP32_TO_BF16(x);
}

void lm_ggml_fp16_to_fp32_row(const lm_ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = LM_GGML_FP16_TO_FP32(x[i]);
    }
}

// FIXME: these functions must detect the instruction set at runtime, since they are part of the core ggml library
//        currently, the lm_ggml_cpu_has_* functions are entirely compile-time
void lm_ggml_fp32_to_fp16_row(const float * x, lm_ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
    //if (lm_ggml_cpu_has_f16c()) {
        for (; i + 7 < n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), y_vec);
        }
        for(; i + 3 < n; i += 4) {
            __m128 x_vec = _mm_loadu_ps(x + i);
            __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
            _mm_storel_epi64((__m128i *)(y + i), y_vec);
        }
    //}
#endif
    for (; i < n; i++) {
        y[i] = LM_GGML_FP32_TO_FP16(x[i]);
    }
}

void lm_ggml_bf16_to_fp32_row(const lm_ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX512F__)
    //if (lm_ggml_cpu_has_avx512()) {
        for (; i + 16 <= n; i += 16) {
            _mm512_storeu_ps(y + i,
                            _mm512_castsi512_ps(
                                _mm512_slli_epi32(
                                    _mm512_cvtepu16_epi32(
                                        _mm256_loadu_si256(
                                            (const __m256i *)(x + i))),
                                    16)));
        }
    //}
#endif
#if defined(__AVX2__)
    //if (lm_ggml_cpu_has_avx2()) {
        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(y + i,
                            _mm256_castsi256_ps(
                                _mm256_slli_epi32(
                                    _mm256_cvtepu16_epi32(
                                        _mm_loadu_si128(
                                            (const __m128i *)(x + i))),
                                    16)));
        }
    //}
#endif
    for (; i < n; i++) {
        y[i] = LM_GGML_BF16_TO_FP32(x[i]);
    }
}

void lm_ggml_fp32_to_bf16_row_ref(const float * x, lm_ggml_bf16_t * y, int64_t n) {
    for (int i = 0; i < n; i++) {
        y[i] = lm_ggml_compute_fp32_to_bf16(x[i]);
    }
}

void lm_ggml_fp32_to_bf16_row(const float * x, lm_ggml_bf16_t * y, int64_t n) {
  int i = 0;
#if defined(__AVX512BF16__)
  // subnormals are flushed to zero on this platform
  for (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
  }
#endif
    for (; i < n; i++) {
        y[i] = LM_GGML_FP32_TO_BF16(x[i]);
    }
}

bool lm_ggml_guid_matches(lm_ggml_guid_t guid_a, lm_ggml_guid_t guid_b) {
    return memcmp(guid_a, guid_b, sizeof(lm_ggml_guid)) == 0;
}

//
// timing
//

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
void lm_ggml_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
int64_t lm_ggml_time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
int64_t lm_ggml_time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#else
void lm_ggml_time_init(void) {}
int64_t lm_ggml_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

int64_t lm_ggml_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

int64_t lm_ggml_cycles(void) {
    return clock();
}

int64_t lm_ggml_cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}

//
// cross-platform UTF-8 file paths
//

#ifdef _WIN32
static wchar_t * lm_ggml_mbstowcs(const char * mbs) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, NULL, 0);
    if (!wlen) {
        errno = EINVAL;
        return NULL;
    }

    wchar_t * wbuf = LM_GGML_MALLOC(wlen * sizeof(wchar_t));
    wlen = MultiByteToWideChar(CP_UTF8, 0, mbs, -1, wbuf, wlen);
    if (!wlen) {
        LM_GGML_FREE(wbuf);
        errno = EINVAL;
        return NULL;
    }

    return wbuf;
}
#endif

FILE * lm_ggml_fopen(const char * fname, const char * mode) {
#ifdef _WIN32
    FILE * file = NULL;

    // convert fname (UTF-8)
    wchar_t * wfname = lm_ggml_mbstowcs(fname);
    if (wfname) {
        // convert mode (ANSI)
        wchar_t * wmode = LM_GGML_MALLOC((strlen(mode) + 1) * sizeof(wchar_t));
        wchar_t * wmode_p = wmode;
        do {
            *wmode_p++ = (wchar_t)*mode;
        } while (*mode++);

        // open file
        file = _wfopen(wfname, wmode);

        LM_GGML_FREE(wfname);
        LM_GGML_FREE(wmode);
    }

    return file;
#else
    return fopen(fname, mode);
#endif

}
static void lm_ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);
static void lm_ggml_vec_dot_f16(int n, float * restrict s, size_t bs, lm_ggml_fp16_t * restrict x, size_t bx, lm_ggml_fp16_t * restrict y, size_t by, int nrc);
static void lm_ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, lm_ggml_bf16_t * restrict x, size_t bx, lm_ggml_bf16_t * restrict y, size_t by, int nrc);

static const struct lm_ggml_type_traits type_traits[LM_GGML_TYPE_COUNT] = {
    [LM_GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_I64] = {
        .type_name                = "i64",
        .blck_size                = 1,
        .type_size                = sizeof(int64_t),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_F64] = {
        .type_name                = "f64",
        .blck_size                = 1,
        .type_size                = sizeof(double),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(lm_ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (lm_ggml_to_float_t) lm_ggml_fp16_to_fp32_row,
        .from_float_ref           = (lm_ggml_from_float_t) lm_ggml_fp32_to_fp16_row,
    },
    [LM_GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_0_ref,
    },
    [LM_GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_1,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_1_ref,
    },
    [4] = { // LM_GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [5] = { // LM_GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
    },
    [LM_GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_0_ref,
    },
    [LM_GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_1,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_1_ref,
    },
    [LM_GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q8_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q8_0_ref,
    },
    [LM_GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q8_1_ref,
    },
    [LM_GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q2_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q2_K_ref,
    },
    [LM_GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q3_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q3_K_ref,
    },
    [LM_GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_K_ref,
    },
    [LM_GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_K_ref,
    },
    [LM_GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q6_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q6_K_ref,
    },
    [LM_GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq3_xxs_ref,
    },
    [LM_GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq3_s,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq3_s_ref,
    },
    [LM_GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_s,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq2_s_ref,
    },
    [LM_GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq1_s,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq1_m,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq4_nl_ref,
    },
    [LM_GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq4_xs_ref,
    },
    [LM_GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
    },
    [LM_GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = sizeof(lm_ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (lm_ggml_to_float_t) lm_ggml_bf16_to_fp32_row,
        .from_float_ref           = (lm_ggml_from_float_t) lm_ggml_fp32_to_bf16_row_ref,
    },
    [LM_GGML_TYPE_Q4_0_4_4] = {
        .type_name                = "q4_0_4x4",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 4,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_Q4_0_4_8] = {
        .type_name                = "q4_0_4x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_Q4_0_8_8] = {
        .type_name                = "q4_0_8x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float_ref           = NULL,
    },
    [LM_GGML_TYPE_TQ1_0] = {
        .type_name                = "tq1_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq1_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_tq1_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_tq1_0_ref,
    },
    [LM_GGML_TYPE_TQ2_0] = {
        .type_name                = "tq2_0",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_tq2_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_tq2_0_ref,
    },
};

const struct lm_ggml_type_traits * lm_ggml_get_type_traits(enum lm_ggml_type type) {
    LM_GGML_ASSERT(type < LM_GGML_TYPE_COUNT);
    return &type_traits[type];
}

//
// ggml object
//

struct lm_ggml_object {
    size_t offs;
    size_t size;

    struct lm_ggml_object * next;

    enum lm_ggml_object_type type;

    char padding[4];
};

static const size_t LM_GGML_OBJECT_SIZE = sizeof(struct lm_ggml_object);

//
// ggml context
//

struct lm_ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct lm_ggml_object * objects_begin;
    struct lm_ggml_object * objects_end;
};

struct lm_ggml_context_container {
    bool used;

    struct lm_ggml_context context;
};

//
// data types
//

static const char * LM_GGML_OP_NAME[LM_GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "LEAKY_RELU",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",

    "UNARY",

    "MAP_UNARY",
    "MAP_BINARY",

    "MAP_CUSTOM1_F32",
    "MAP_CUSTOM2_F32",
    "MAP_CUSTOM3_F32",

    "MAP_CUSTOM1",
    "MAP_CUSTOM2",
    "MAP_CUSTOM3",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
};

static_assert(LM_GGML_OP_COUNT == 81, "LM_GGML_OP_COUNT != 81");

static const char * LM_GGML_OP_SYMBOL[LM_GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "sin(x)",
    "cos(x)",
    "Σx",
    "Σx_k",
    "Σx/n",
    "argmax(x)",
    "count_equal(x)",
    "repeat(x)",
    "repeat_back(x)",
    "concat(x, y)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",
    "group_norm(x)",

    "X*Y",
    "X[i]*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "clamp(x)",
    "conv_transpose_1d(x)",
    "im2col(x)",
    "im2col_back(x)",
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
    "pool_2d_back(x)",
    "upscale(x)",
    "pad(x)",
    "arange(start, stop, step)",
    "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",
    "leaky_relu(x)",

    "flash_attn_ext(x)",
    "flash_attn_back(x)",
    "ssm_conv(x)",
    "ssm_scan(x)",
    "win_part(x)",
    "win_unpart(x)",
    "get_rel_pos(x)",
    "add_rel_pos(x)",
    "rwkv_wkv6(k, v, r, tf, td, s)",

    "unary(x)",

    "f(x)",
    "f(x,y)",

    "custom_f32(x)",
    "custom_f32(x,y)",
    "custom_f32(x,y,z)",

    "custom(x)",
    "custom(x,y)",
    "custom(x,y,z)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
    "adamw(x)",
};

static_assert(LM_GGML_OP_COUNT == 81, "LM_GGML_OP_COUNT != 81");

static_assert(LM_GGML_OP_POOL_COUNT == 2, "LM_GGML_OP_POOL_COUNT != 2");


static const char * LM_GGML_UNARY_OP_NAME[LM_GGML_UNARY_OP_COUNT] = {
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "TANH",
    "ELU",
    "RELU",
    "SIGMOID",
    "GELU",
    "GELU_QUICK",
    "SILU",
    "HARDSWISH",
    "HARDSIGMOID",
    "EXP",
};

static_assert(LM_GGML_UNARY_OP_COUNT == 14, "LM_GGML_UNARY_OP_COUNT != 14");


static_assert(sizeof(struct lm_ggml_object)%LM_GGML_MEM_ALIGN == 0, "lm_ggml_object size must be a multiple of LM_GGML_MEM_ALIGN");
static_assert(sizeof(struct lm_ggml_tensor)%LM_GGML_MEM_ALIGN == 0, "lm_ggml_tensor size must be a multiple of LM_GGML_MEM_ALIGN");


////////////////////////////////////////////////////////////////////////////////

void lm_ggml_print_object(const struct lm_ggml_object * obj) {
    LM_GGML_LOG_INFO(" - lm_ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (const void *) obj->next);
}

void lm_ggml_print_objects(const struct lm_ggml_context * ctx) {
    struct lm_ggml_object * obj = ctx->objects_begin;

    LM_GGML_LOG_INFO("%s: objects in context %p:\n", __func__, (const void *) ctx);

    while (obj != NULL) {
        lm_ggml_print_object(obj);
        obj = obj->next;
    }

    LM_GGML_LOG_INFO("%s: --- end ---\n", __func__);
}

int64_t lm_ggml_nelements(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

int64_t lm_ggml_nrows(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

size_t lm_ggml_nbytes(const struct lm_ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = lm_ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = lm_ggml_type_size(tensor->type);
        for (int i = 0; i < LM_GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < LM_GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

size_t lm_ggml_nbytes_pad(const struct lm_ggml_tensor * tensor) {
    return LM_GGML_PAD(lm_ggml_nbytes(tensor), LM_GGML_MEM_ALIGN);
}

int64_t lm_ggml_blck_size(enum lm_ggml_type type) {
    return type_traits[type].blck_size;
}

size_t lm_ggml_type_size(enum lm_ggml_type type) {
    return type_traits[type].type_size;
}

size_t lm_ggml_row_size(enum lm_ggml_type type, int64_t ne) {
    assert(ne % lm_ggml_blck_size(type) == 0);
    return lm_ggml_type_size(type)*ne/lm_ggml_blck_size(type);
}

double lm_ggml_type_sizef(enum lm_ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

const char * lm_ggml_type_name(enum lm_ggml_type type) {
    return type < LM_GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

bool lm_ggml_is_quantized(enum lm_ggml_type type) {
    return type_traits[type].is_quantized;
}

const char * lm_ggml_op_name(enum lm_ggml_op op) {
    return LM_GGML_OP_NAME[op];
}

const char * lm_ggml_op_symbol(enum lm_ggml_op op) {
    return LM_GGML_OP_SYMBOL[op];
}

const char * lm_ggml_unary_op_name(enum lm_ggml_unary_op op) {
    return LM_GGML_UNARY_OP_NAME[op];
}

const char * lm_ggml_op_desc(const struct lm_ggml_tensor * t) {
    if (t->op == LM_GGML_OP_UNARY) {
        enum lm_ggml_unary_op uop = lm_ggml_get_unary_op(t);
        return lm_ggml_unary_op_name(uop);
    }
    return lm_ggml_op_name(t->op);
}

size_t lm_ggml_element_size(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_type_size(tensor->type);
}

bool lm_ggml_is_scalar(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool lm_ggml_is_vector(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool lm_ggml_is_matrix(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool lm_ggml_is_3d(const struct lm_ggml_tensor * tensor) {
    return tensor->ne[3] == 1;
}

int lm_ggml_n_dims(const struct lm_ggml_tensor * tensor) {
    for (int i = LM_GGML_MAX_DIMS - 1; i >= 1; --i) {
        if (tensor->ne[i] > 1) {
            return i + 1;
        }
    }
    return 1;
}

enum lm_ggml_type lm_ggml_ftype_to_lm_ggml_type(enum lm_ggml_ftype ftype) {
    enum lm_ggml_type wtype = LM_GGML_TYPE_COUNT;

    switch (ftype) {
        case LM_GGML_FTYPE_ALL_F32:              wtype = LM_GGML_TYPE_F32;   break;
        case LM_GGML_FTYPE_MOSTLY_F16:           wtype = LM_GGML_TYPE_F16;   break;
        case LM_GGML_FTYPE_MOSTLY_BF16:          wtype = LM_GGML_TYPE_BF16;  break;
        case LM_GGML_FTYPE_MOSTLY_Q4_0:          wtype = LM_GGML_TYPE_Q4_0;  break;
        case LM_GGML_FTYPE_MOSTLY_Q4_1:          wtype = LM_GGML_TYPE_Q4_1;  break;
        case LM_GGML_FTYPE_MOSTLY_Q5_0:          wtype = LM_GGML_TYPE_Q5_0;  break;
        case LM_GGML_FTYPE_MOSTLY_Q5_1:          wtype = LM_GGML_TYPE_Q5_1;  break;
        case LM_GGML_FTYPE_MOSTLY_Q8_0:          wtype = LM_GGML_TYPE_Q8_0;  break;
        case LM_GGML_FTYPE_MOSTLY_Q2_K:          wtype = LM_GGML_TYPE_Q2_K;  break;
        case LM_GGML_FTYPE_MOSTLY_Q3_K:          wtype = LM_GGML_TYPE_Q3_K;  break;
        case LM_GGML_FTYPE_MOSTLY_Q4_K:          wtype = LM_GGML_TYPE_Q4_K;  break;
        case LM_GGML_FTYPE_MOSTLY_Q5_K:          wtype = LM_GGML_TYPE_Q5_K;  break;
        case LM_GGML_FTYPE_MOSTLY_Q6_K:          wtype = LM_GGML_TYPE_Q6_K;  break;
        case LM_GGML_FTYPE_MOSTLY_IQ2_XXS:       wtype = LM_GGML_TYPE_IQ2_XXS;  break;
        case LM_GGML_FTYPE_MOSTLY_IQ2_XS:        wtype = LM_GGML_TYPE_IQ2_XS;   break;
        case LM_GGML_FTYPE_MOSTLY_IQ3_XXS:       wtype = LM_GGML_TYPE_IQ3_XXS;  break;
        case LM_GGML_FTYPE_MOSTLY_IQ1_S:         wtype = LM_GGML_TYPE_IQ1_S;    break;
        case LM_GGML_FTYPE_MOSTLY_IQ1_M:         wtype = LM_GGML_TYPE_IQ1_M;    break;
        case LM_GGML_FTYPE_MOSTLY_IQ4_NL:        wtype = LM_GGML_TYPE_IQ4_NL;   break;
        case LM_GGML_FTYPE_MOSTLY_IQ4_XS:        wtype = LM_GGML_TYPE_IQ4_XS;   break;
        case LM_GGML_FTYPE_MOSTLY_IQ3_S:         wtype = LM_GGML_TYPE_IQ3_S;    break;
        case LM_GGML_FTYPE_MOSTLY_IQ2_S:         wtype = LM_GGML_TYPE_IQ2_S;    break;
        case LM_GGML_FTYPE_MOSTLY_Q4_0_4_4:      wtype = LM_GGML_TYPE_Q4_0_4_4; break;
        case LM_GGML_FTYPE_MOSTLY_Q4_0_4_8:      wtype = LM_GGML_TYPE_Q4_0_4_8; break;
        case LM_GGML_FTYPE_MOSTLY_Q4_0_8_8:      wtype = LM_GGML_TYPE_Q4_0_8_8; break;
        case LM_GGML_FTYPE_UNKNOWN:              wtype = LM_GGML_TYPE_COUNT; break;
        case LM_GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: wtype = LM_GGML_TYPE_COUNT; break;
    }

    LM_GGML_ASSERT(wtype != LM_GGML_TYPE_COUNT);

    return wtype;
}

size_t lm_ggml_tensor_overhead(void) {
    return LM_GGML_OBJECT_SIZE + LM_GGML_TENSOR_SIZE;
}

bool lm_ggml_is_transposed(const struct lm_ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

static bool lm_ggml_is_contiguous_n(const struct lm_ggml_tensor * tensor, int n) {
    size_t next_nb = lm_ggml_type_size(tensor->type);
    if (tensor->ne[0] != lm_ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/lm_ggml_blck_size(tensor->type);
    for (int i = 1; i < LM_GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

bool lm_ggml_is_contiguous(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_0(tensor);
}

bool lm_ggml_is_contiguous_0(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 0);
}

bool lm_ggml_is_contiguous_1(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 1);
}

bool lm_ggml_is_contiguous_2(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 2);
}

bool lm_ggml_is_permuted(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->nb[0] > tensor->nb[1] || tensor->nb[1] > tensor->nb[2] || tensor->nb[2] > tensor->nb[3];
}

static inline bool lm_ggml_is_padded_1d(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == lm_ggml_type_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

bool lm_ggml_is_empty(const struct lm_ggml_tensor * tensor) {
    for (int i = 0; i < LM_GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            // empty if any dimension has no elements
            return true;
        }
    }
    return false;
}

bool lm_ggml_are_same_shape(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->ne[0] == t1->ne[0]) &&
        (t0->ne[1] == t1->ne[1]) &&
        (t0->ne[2] == t1->ne[2]) &&
        (t0->ne[3] == t1->ne[3]);
}

bool lm_ggml_are_same_stride(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return
        (t0->nb[0] == t1->nb[0]) &&
        (t0->nb[1] == t1->nb[1]) &&
        (t0->nb[2] == t1->nb[2]) &&
        (t0->nb[3] == t1->nb[3]);
}

// check if t1 can be represented as a repeatition of t0
bool lm_ggml_can_repeat(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return lm_ggml_is_empty(t0) ? lm_ggml_is_empty(t1) :
        (t1->ne[0]%t0->ne[0] == 0) &&
        (t1->ne[1]%t0->ne[1] == 0) &&
        (t1->ne[2]%t0->ne[2] == 0) &&
        (t1->ne[3]%t0->ne[3] == 0);
}

static inline bool lm_ggml_can_repeat_rows(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0] == t1->ne[0]) && lm_ggml_can_repeat(t0, t1);
}

// assert that pointer is aligned to LM_GGML_MEM_ALIGN
#define LM_GGML_ASSERT_ALIGNED(ptr) \
    LM_GGML_ASSERT(((uintptr_t) (ptr))%LM_GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct lm_ggml_context * lm_ggml_init(struct lm_ggml_init_params params) {
    static bool is_first_call = true;

    lm_ggml_critical_section_start();

    if (is_first_call) {
        // initialize time system (required on Windows)
        lm_ggml_time_init();

        for (int i = 0; i < (1 << 16); ++i) {
            union {
                uint16_t u16;
                lm_ggml_fp16_t fp16;
            } u = {i};
            lm_ggml_table_f32_f16[i] = LM_GGML_COMPUTE_FP16_TO_FP32(u.fp16);
        }

        is_first_call = false;
    }

    lm_ggml_critical_section_end();

    struct lm_ggml_context * ctx = LM_GGML_MALLOC(sizeof(struct lm_ggml_context));

    // allow to call lm_ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = LM_GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : LM_GGML_PAD(params.mem_size, LM_GGML_MEM_ALIGN);

    *ctx = (struct lm_ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : lm_ggml_aligned_malloc(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
    };

    LM_GGML_ASSERT(ctx->mem_buffer != NULL);

    LM_GGML_ASSERT_ALIGNED(ctx->mem_buffer);

    LM_GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    return ctx;
}

void lm_ggml_reset(struct lm_ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    ctx->n_objects     = 0;
    ctx->objects_begin = NULL;
    ctx->objects_end   = NULL;
}

void lm_ggml_free(struct lm_ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->mem_buffer_owned) {
        lm_ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
    }

    LM_GGML_FREE(ctx);
}

size_t lm_ggml_used_mem(const struct lm_ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool lm_ggml_get_no_alloc(struct lm_ggml_context * ctx) {
    return ctx->no_alloc;
}

void lm_ggml_set_no_alloc(struct lm_ggml_context * ctx, bool no_alloc) {
    ctx->no_alloc = no_alloc;
}

void * lm_ggml_get_mem_buffer(const struct lm_ggml_context * ctx) {
    return ctx->mem_buffer;
}

size_t lm_ggml_get_mem_size(const struct lm_ggml_context * ctx) {
    return ctx->mem_size;
}

size_t lm_ggml_get_max_tensor_size(const struct lm_ggml_context * ctx) {
    size_t max_size = 0;

    for (struct lm_ggml_tensor * tensor = lm_ggml_get_first_tensor(ctx); tensor != NULL; tensor = lm_ggml_get_next_tensor(ctx, tensor)) {
        size_t bytes = lm_ggml_nbytes(tensor);
        max_size = MAX(max_size, bytes);
    }

    return max_size;
}

////////////////////////////////////////////////////////////////////////////////

static struct lm_ggml_object * lm_ggml_new_object(struct lm_ggml_context * ctx, enum lm_ggml_object_type type, size_t size) {
    // always insert objects at the end of the context's memory pool
    struct lm_ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    // align to LM_GGML_MEM_ALIGN
    size_t size_needed = LM_GGML_PAD(size, LM_GGML_MEM_ALIGN);

    char * const mem_buffer = ctx->mem_buffer;
    struct lm_ggml_object * const obj_new = (struct lm_ggml_object *)(mem_buffer + cur_end);

    if (cur_end + size_needed + LM_GGML_OBJECT_SIZE > ctx->mem_size) {
        LM_GGML_LOG_WARN("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed + LM_GGML_OBJECT_SIZE, ctx->mem_size);
#ifndef NDEBUG
        LM_GGML_ABORT("not enough space in the context's memory pool");
#endif
        return NULL;
    }

    *obj_new = (struct lm_ggml_object) {
        .offs = cur_end + LM_GGML_OBJECT_SIZE,
        .size = size_needed,
        .next = NULL,
        .type = type,
    };

    LM_GGML_ASSERT_ALIGNED(mem_buffer + obj_new->offs);

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    return obj_new;
}

static struct lm_ggml_tensor * lm_ggml_new_tensor_impl(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type      type,
        int                   n_dims,
        const int64_t       * ne,
        struct lm_ggml_tensor  * view_src,
        size_t                view_offs) {

    LM_GGML_ASSERT(type >= 0 && type < LM_GGML_TYPE_COUNT);
    LM_GGML_ASSERT(n_dims >= 1 && n_dims <= LM_GGML_MAX_DIMS);

    // find the base tensor and absolute offset
    if (view_src != NULL && view_src->view_src != NULL) {
        view_offs += view_src->view_offs;
        view_src   = view_src->view_src;
    }

    size_t data_size = lm_ggml_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        data_size *= ne[i];
    }

    LM_GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= lm_ggml_nbytes(view_src));

    void * data = view_src != NULL ? view_src->data : NULL;
    if (data != NULL) {
        data = (char *) data + view_offs;
    }

    size_t obj_alloc_size = 0;

    if (view_src == NULL && !ctx->no_alloc) {
        // allocate tensor data in the context's memory pool
        obj_alloc_size = data_size;
    }

    struct lm_ggml_object * const obj_new = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_TENSOR, LM_GGML_TENSOR_SIZE + obj_alloc_size);
    LM_GGML_ASSERT(obj_new);

    struct lm_ggml_tensor * const result = (struct lm_ggml_tensor *)((char *)ctx->mem_buffer + obj_new->offs);

#ifdef __clang__
    // temporary until lm_ggml_tensor::backend is removed
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

    *result = (struct lm_ggml_tensor) {
        /*.type         =*/ type,
        /*.backend      =*/ LM_GGML_BACKEND_TYPE_CPU,
        /*.buffer       =*/ NULL,
        /*.ne           =*/ { 1, 1, 1, 1 },
        /*.nb           =*/ { 0, 0, 0, 0 },
        /*.op           =*/ LM_GGML_OP_NONE,
        /*.op_params    =*/ { 0 },
        /*.flags        =*/ 0,
        /*.grad         =*/ NULL,
        /*.src          =*/ { NULL },
        /*.view_src     =*/ view_src,
        /*.view_offs    =*/ view_offs,
        /*.data         =*/ obj_alloc_size > 0 ? (void *)(result + 1) : data,
        /*.name         =*/ { 0 },
        /*.extra        =*/ NULL,
        ///*.padding      =*/ { 0 },
    };

#ifdef __clang__
    #pragma clang diagnostic pop
#endif

    // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
    //LM_GGML_ASSERT_ALIGNED(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = lm_ggml_type_size(type);
    result->nb[1] = result->nb[0]*(result->ne[0]/lm_ggml_blck_size(type));
    for (int i = 2; i < LM_GGML_MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct lm_ggml_tensor * lm_ggml_new_tensor(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type      type,
        int                   n_dims,
        const int64_t       * ne) {
    return lm_ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct lm_ggml_tensor * lm_ggml_new_tensor_1d(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type      type,
        int64_t ne0) {
    return lm_ggml_new_tensor(ctx, type, 1, &ne0);
}

struct lm_ggml_tensor * lm_ggml_new_tensor_2d(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type      type,
        int64_t ne0,
        int64_t ne1) {
    const int64_t ne[2] = { ne0, ne1 };
    return lm_ggml_new_tensor(ctx, type, 2, ne);
}

struct lm_ggml_tensor * lm_ggml_new_tensor_3d(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type      type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    const int64_t ne[3] = { ne0, ne1, ne2 };
    return lm_ggml_new_tensor(ctx, type, 3, ne);
}

struct lm_ggml_tensor * lm_ggml_new_tensor_4d(
        struct lm_ggml_context * ctx,
        enum   lm_ggml_type type,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2,
        int64_t ne3) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return lm_ggml_new_tensor(ctx, type, 4, ne);
}

void * lm_ggml_new_buffer(struct lm_ggml_context * ctx, size_t nbytes) {
    struct lm_ggml_object * obj = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_WORK_BUFFER, nbytes);

    return (uint8_t *)ctx->mem_buffer + obj->offs;
}

struct lm_ggml_tensor * lm_ggml_dup_tensor(struct lm_ggml_context * ctx, const struct lm_ggml_tensor * src) {
    return lm_ggml_new_tensor(ctx, src->type, LM_GGML_MAX_DIMS, src->ne);
}

void lm_ggml_unravel_index(const struct lm_ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3) {
    const int64_t ne2 = tensor->ne[2];
    const int64_t ne1 = tensor->ne[1];
    const int64_t ne0 = tensor->ne[0];

    const int64_t i3_ = (i/(ne2*ne1*ne0));
    const int64_t i2_ = (i - i3_*ne2*ne1*ne0)/(ne1*ne0);
    const int64_t i1_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0)/ne0;
    const int64_t i0_ = (i - i3_*ne2*ne1*ne0 - i2_*ne1*ne0 - i1_*ne0);

    if (i0) {
        * i0 = i0_;
    }
    if (i1) {
        * i1 = i1_;
    }
    if (i2) {
        * i2 = i2_;
    }
    if (i3) {
        * i3 = i3_;
    }
}

void * lm_ggml_get_data(const struct lm_ggml_tensor * tensor) {
    return tensor->data;
}

float * lm_ggml_get_data_f32(const struct lm_ggml_tensor * tensor) {
    assert(tensor->type == LM_GGML_TYPE_F32);
    return (float *)(tensor->data);
}

enum lm_ggml_unary_op lm_ggml_get_unary_op(const struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor->op == LM_GGML_OP_UNARY);
    return (enum lm_ggml_unary_op) lm_ggml_get_op_params_i32(tensor, 0);
}

const char * lm_ggml_get_name(const struct lm_ggml_tensor * tensor) {
    return tensor->name;
}

struct lm_ggml_tensor * lm_ggml_set_name(struct lm_ggml_tensor * tensor, const char * name) {
    size_t i;
    for (i = 0; i < sizeof(tensor->name) - 1 && name[i] != '\0'; i++) {
        tensor->name[i] = name[i];
    }
    tensor->name[i] = '\0';
    return tensor;
}

struct lm_ggml_tensor * lm_ggml_format_name(struct lm_ggml_tensor * tensor, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
    va_end(args);
    return tensor;
}

struct lm_ggml_tensor * lm_ggml_view_tensor(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * src) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, src->type, LM_GGML_MAX_DIMS, src->ne, src, 0);
    lm_ggml_format_name(result, "%s (view)", src->name);

    for (int i = 0; i < LM_GGML_MAX_DIMS; i++) {
        result->nb[i] = src->nb[i];
    }

    return result;
}

struct lm_ggml_tensor * lm_ggml_get_first_tensor(const struct lm_ggml_context * ctx) {
    struct lm_ggml_object * obj = ctx->objects_begin;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == LM_GGML_OBJECT_TYPE_TENSOR) {
            return (struct lm_ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    return NULL;
}

struct lm_ggml_tensor * lm_ggml_get_next_tensor(const struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor) {
    struct lm_ggml_object * obj = (struct lm_ggml_object *) ((char *)tensor - LM_GGML_OBJECT_SIZE);
    obj = obj->next;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == LM_GGML_OBJECT_TYPE_TENSOR) {
            return (struct lm_ggml_tensor *)(mem_buffer + obj->offs);
        }

        obj = obj->next;
    }

    return NULL;
}

struct lm_ggml_tensor * lm_ggml_get_tensor(struct lm_ggml_context * ctx, const char * name) {
    struct lm_ggml_object * obj = ctx->objects_begin;

    char * const mem_buffer = ctx->mem_buffer;

    while (obj != NULL) {
        if (obj->type == LM_GGML_OBJECT_TYPE_TENSOR) {
            struct lm_ggml_tensor * cur = (struct lm_ggml_tensor *)(mem_buffer + obj->offs);
            if (strcmp(cur->name, name) == 0) {
                return cur;
            }
        }

        obj = obj->next;
    }

    return NULL;
}

////////////////////////////////////////////////////////////////////////////////

// lm_ggml_dup

static struct lm_ggml_tensor * lm_ggml_dup_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_DUP;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_dup(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_dup_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_dup_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_dup_impl(ctx, a, true);
}

// lm_ggml_add

static struct lm_ggml_tensor * lm_ggml_add_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_add_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_add_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_add_impl(ctx, a, b, true);
}

// lm_ggml_add_cast

static struct lm_ggml_tensor * lm_ggml_add_cast_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        enum   lm_ggml_type      type) {
    // TODO: support less-strict constraint
    //       LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));
    LM_GGML_ASSERT(lm_ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    LM_GGML_ASSERT(lm_ggml_is_quantized(a->type) ||
                a->type == LM_GGML_TYPE_F16 ||
                a->type == LM_GGML_TYPE_BF16);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, type, LM_GGML_MAX_DIMS, a->ne);

    result->op     = LM_GGML_OP_ADD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add_cast(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        enum   lm_ggml_type      type) {
    return lm_ggml_add_cast_impl(ctx, a, b, type);
}

// lm_ggml_add1

static struct lm_ggml_tensor * lm_ggml_add1_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_is_scalar(b));
    LM_GGML_ASSERT(lm_ggml_is_padded_1d(a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_ADD1;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add1(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_add1_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_add1_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_add1_impl(ctx, a, b, true);
}

// lm_ggml_acc

static struct lm_ggml_tensor * lm_ggml_acc_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_nelements(b) <= lm_ggml_nelements(a));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(a->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(b->type == LM_GGML_TYPE_F32);

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_ACC;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_acc(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct lm_ggml_tensor * lm_ggml_acc_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// lm_ggml_sub

static struct lm_ggml_tensor * lm_ggml_sub_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SUB;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_sub(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_sub_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_sub_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_sub_impl(ctx, a, b, true);
}

// lm_ggml_mul

static struct lm_ggml_tensor * lm_ggml_mul_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_MUL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_mul(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_mul_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_mul_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_mul_impl(ctx, a, b, true);
}

// lm_ggml_div

static struct lm_ggml_tensor * lm_ggml_div_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_DIV;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_div(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_div_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_div_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_div_impl(ctx, a, b, true);
}

// lm_ggml_sqr

static struct lm_ggml_tensor * lm_ggml_sqr_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SQR;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_sqr(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sqr_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_sqr_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sqr_impl(ctx, a, true);
}

// lm_ggml_sqrt

static struct lm_ggml_tensor * lm_ggml_sqrt_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SQRT;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_sqrt(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sqrt_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_sqrt_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sqrt_impl(ctx, a, true);
}

// lm_ggml_log

static struct lm_ggml_tensor * lm_ggml_log_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_LOG;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_log(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_log_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_log_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_log_impl(ctx, a, true);
}

// lm_ggml_sin

static struct lm_ggml_tensor * lm_ggml_sin_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SIN;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_sin(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sin_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_sin_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_sin_impl(ctx, a, true);
}

// lm_ggml_cos

static struct lm_ggml_tensor * lm_ggml_cos_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_COS;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_cos(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_cos_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_cos_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_cos_impl(ctx, a, true);
}

// lm_ggml_sum

struct lm_ggml_tensor * lm_ggml_sum(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = LM_GGML_OP_SUM;
    result->src[0] = a;

    return result;
}

// lm_ggml_sum_rows

struct lm_ggml_tensor * lm_ggml_sum_rows(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    int64_t ne[LM_GGML_MAX_DIMS] = { 1 };
    for (int i = 1; i < LM_GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, ne);

    result->op     = LM_GGML_OP_SUM_ROWS;
    result->src[0] = a;

    return result;
}

// lm_ggml_mean

struct lm_ggml_tensor * lm_ggml_mean(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op     = LM_GGML_OP_MEAN;
    result->src[0] = a;

    return result;
}

// lm_ggml_argmax

struct lm_ggml_tensor * lm_ggml_argmax(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    LM_GGML_ASSERT(lm_ggml_is_matrix(a));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, a->ne[1]);

    result->op     = LM_GGML_OP_ARGMAX;
    result->src[0] = a;

    return result;
}

// lm_ggml_count_equal

struct lm_ggml_tensor * lm_ggml_count_equal(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I64, 1);

    result->op     = LM_GGML_OP_COUNT_EQUAL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_repeat

struct lm_ggml_tensor * lm_ggml_repeat(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(a, b));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, b->ne);

    result->op     = LM_GGML_OP_REPEAT;
    result->src[0] = a;

    return result;
}

// lm_ggml_repeat_back

struct lm_ggml_tensor * lm_ggml_repeat_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, b->ne);

    result->op     = LM_GGML_OP_REPEAT_BACK;
    result->src[0] = a;

    return result;
}

// lm_ggml_concat

struct lm_ggml_tensor * lm_ggml_concat(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor  * a,
    struct lm_ggml_tensor  * b,
    int                   dim) {
    LM_GGML_ASSERT(dim >= 0 && dim < LM_GGML_MAX_DIMS);

    int64_t ne[LM_GGML_MAX_DIMS];
    for (int d = 0; d < LM_GGML_MAX_DIMS; ++d) {
        if (d == dim) {
            ne[d] = a->ne[d] + b->ne[d];
            continue;
        }
        LM_GGML_ASSERT(a->ne[d] == b->ne[d]);
        ne[d] = a->ne[d];
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, ne);

    lm_ggml_set_op_params_i32(result, 0, dim);

    result->op     = LM_GGML_OP_CONCAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_abs

struct lm_ggml_tensor * lm_ggml_abs(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_ABS);
}

struct lm_ggml_tensor * lm_ggml_abs_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_ABS);
}

// lm_ggml_sgn

struct lm_ggml_tensor * lm_ggml_sgn(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_SGN);
}

struct lm_ggml_tensor * lm_ggml_sgn_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_SGN);
}

// lm_ggml_neg

struct lm_ggml_tensor * lm_ggml_neg(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_NEG);
}

struct lm_ggml_tensor * lm_ggml_neg_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_NEG);
}

// lm_ggml_step

struct lm_ggml_tensor * lm_ggml_step(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_STEP);
}

struct lm_ggml_tensor * lm_ggml_step_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_STEP);
}

// lm_ggml_tanh

struct lm_ggml_tensor * lm_ggml_tanh(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_TANH);
}

struct lm_ggml_tensor * lm_ggml_tanh_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_TANH);
}

// lm_ggml_elu

struct lm_ggml_tensor * lm_ggml_elu(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_ELU);
}

struct lm_ggml_tensor * lm_ggml_elu_inplace(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_ELU);
}

// lm_ggml_relu

struct lm_ggml_tensor * lm_ggml_relu(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_RELU);
}

struct lm_ggml_tensor * lm_ggml_relu_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_RELU);
}

// lm_ggml_leaky_relu

struct lm_ggml_tensor * lm_ggml_leaky_relu(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 negative_slope,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &negative_slope, sizeof(negative_slope));

    result->op     = LM_GGML_OP_LEAKY_RELU;
    result->src[0] = a;

    return result;
}

// lm_ggml_sigmoid

struct lm_ggml_tensor * lm_ggml_sigmoid(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_SIGMOID);
}

struct lm_ggml_tensor * lm_ggml_sigmoid_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_SIGMOID);
}

// lm_ggml_gelu

struct lm_ggml_tensor * lm_ggml_gelu(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_GELU);
}

struct lm_ggml_tensor * lm_ggml_gelu_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_GELU);
}

// lm_ggml_gelu_quick

struct lm_ggml_tensor * lm_ggml_gelu_quick(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_GELU_QUICK);
}

struct lm_ggml_tensor * lm_ggml_gelu_quick_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_GELU_QUICK);
}

// lm_ggml_silu

struct lm_ggml_tensor * lm_ggml_silu(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_SILU);
}

struct lm_ggml_tensor * lm_ggml_silu_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_SILU);
}

// lm_ggml_silu_back

struct lm_ggml_tensor * lm_ggml_silu_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SILU_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// ggml hardswish

struct lm_ggml_tensor * lm_ggml_hardswish(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_HARDSWISH);
}

// ggml hardsigmoid

struct lm_ggml_tensor * lm_ggml_hardsigmoid(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_HARDSIGMOID);
}

// ggml exp

struct lm_ggml_tensor * lm_ggml_exp(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary(ctx, a, LM_GGML_UNARY_OP_EXP);
}

struct lm_ggml_tensor * lm_ggml_exp_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_unary_inplace(ctx, a, LM_GGML_UNARY_OP_EXP);
}

// lm_ggml_norm

static struct lm_ggml_tensor * lm_ggml_norm_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = LM_GGML_OP_NORM;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_norm(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps) {
    return lm_ggml_norm_impl(ctx, a, eps, false);
}

struct lm_ggml_tensor * lm_ggml_norm_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps) {
    return lm_ggml_norm_impl(ctx, a, eps, true);
}

// lm_ggml_rms_norm

static struct lm_ggml_tensor * lm_ggml_rms_norm_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = LM_GGML_OP_RMS_NORM;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_rms_norm(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps) {
    return lm_ggml_rms_norm_impl(ctx, a, eps, false);
}

struct lm_ggml_tensor * lm_ggml_rms_norm_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 eps) {
    return lm_ggml_rms_norm_impl(ctx, a, eps, true);
}

// lm_ggml_rms_norm_back

struct lm_ggml_tensor * lm_ggml_rms_norm_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        float                 eps) {
    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op     = LM_GGML_OP_RMS_NORM_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_group_norm

static struct lm_ggml_tensor * lm_ggml_group_norm_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_groups,
        float                 eps,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params_i32(result, 0, n_groups);
    lm_ggml_set_op_params_f32(result, 1, eps);

    result->op     = LM_GGML_OP_GROUP_NORM;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_group_norm(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_groups,
        float                 eps) {
    return lm_ggml_group_norm_impl(ctx, a, n_groups, eps, false);
}

struct lm_ggml_tensor * lm_ggml_group_norm_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_groups,
        float                 eps) {
    return lm_ggml_group_norm_impl(ctx, a, n_groups, eps, true);
}

// lm_ggml_mul_mat

static inline bool lm_ggml_can_mul_mat(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

struct lm_ggml_tensor * lm_ggml_mul_mat(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_mul_mat(a, b));
    LM_GGML_ASSERT(!lm_ggml_is_transposed(a));

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op     = LM_GGML_OP_MUL_MAT;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

void lm_ggml_mul_mat_set_prec(
        struct lm_ggml_tensor * a,
        enum lm_ggml_prec       prec) {
    LM_GGML_ASSERT(a->op == LM_GGML_OP_MUL_MAT);

    const int32_t prec_i32 = (int32_t) prec;

    lm_ggml_set_op_params_i32(a, 0, prec_i32);
}

// lm_ggml_mul_mat_id

/*
    c = lm_ggml_mul_mat_id(ctx, as, b, ids);

    as  -> [cols, rows, n_expert]
    ids -> [n_experts_used, n_tokens] (i32)
    b   -> [cols, n_expert_used, n_tokens]
    c   -> [rows, n_expert_used, n_tokens]

    in b, n_experts_used can be broadcasted to match the n_expert_used of ids

    c ~= as[:,:,i] @ b[:,i%r,t], i = ids[e,t] for all e,t in ids
*/
struct lm_ggml_tensor * lm_ggml_mul_mat_id(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * as,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * ids) {
    LM_GGML_ASSERT(!lm_ggml_is_transposed(as));
    LM_GGML_ASSERT(ids->type == LM_GGML_TYPE_I32);

    LM_GGML_ASSERT(as->ne[3] == 1); // as is 3d (one matrix per expert)
    LM_GGML_ASSERT(b->ne[3] == 1); // b is 3d
    LM_GGML_ASSERT(ids->ne[2] == 1 && ids->ne[3] == 1); // ids is 2d
    LM_GGML_ASSERT(ids->ne[1] == b->ne[2]); // must have an expert list per b row
    LM_GGML_ASSERT(as->ne[0] == b->ne[0]); // can_mul_mat
    LM_GGML_ASSERT(ids->ne[0] % b->ne[1] == 0); // can broadcast

    const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op     = LM_GGML_OP_MUL_MAT_ID;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}

// lm_ggml_out_prod

static inline bool lm_ggml_can_out_prod(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

struct lm_ggml_tensor * lm_ggml_out_prod(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_out_prod(a, b));
    LM_GGML_ASSERT(!lm_ggml_is_transposed(a));

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op     = LM_GGML_OP_OUT_PROD;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_scale

static struct lm_ggml_tensor * lm_ggml_scale_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 s,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_is_padded_1d(a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &s, sizeof(s));

    result->op     = LM_GGML_OP_SCALE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_scale(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 s) {
    return lm_ggml_scale_impl(ctx, a, s, false);
}

struct lm_ggml_tensor * lm_ggml_scale_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 s) {
    return lm_ggml_scale_impl(ctx, a, s, true);
}

// lm_ggml_set

static struct lm_ggml_tensor * lm_ggml_set_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_nelements(a) >= lm_ggml_nelements(b));

    // make a view of the destination
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    LM_GGML_ASSERT(offset < (size_t)(1 << 30));
    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_SET;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_set(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

struct lm_ggml_tensor * lm_ggml_set_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_1d_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

struct lm_ggml_tensor * lm_ggml_set_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_2d_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        size_t                nb1,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, true);
}

// lm_ggml_cpy

static struct lm_ggml_tensor * lm_ggml_cpy_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_nelements(a) == lm_ggml_nelements(b));

    // make a view of the destination
    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, b);
    if (strlen(b->name) > 0) {
        lm_ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } else {
        lm_ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op     = LM_GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_cpy(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_cpy_impl(ctx, a, b);
}

struct lm_ggml_tensor * lm_ggml_cast(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum   lm_ggml_type      type) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, type, LM_GGML_MAX_DIMS, a->ne);
    lm_ggml_format_name(result, "%s (copy)", a->name);

    result->op     = LM_GGML_OP_CPY;
    result->src[0] = a;
    result->src[1] = result;

    return result;
}

// lm_ggml_cont

static struct lm_ggml_tensor * lm_ggml_cont_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);
    lm_ggml_format_name(result, "%s (cont)", a->name);

    result->op     = LM_GGML_OP_CONT;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_cont(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    return lm_ggml_cont_impl(ctx, a);
}

// make contiguous, with new shape
LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0) {
    return lm_ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    return lm_ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

LM_GGML_API struct lm_ggml_tensor * lm_ggml_cont_3d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    return lm_ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

struct lm_ggml_tensor * lm_ggml_cont_4d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    LM_GGML_ASSERT(lm_ggml_nelements(a) == (ne0*ne1*ne2*ne3));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    lm_ggml_format_name(result, "%s (cont)", a->name);

    result->op     = LM_GGML_OP_CONT;
    result->src[0] = a;

    return result;
}

// lm_ggml_reshape

struct lm_ggml_tensor * lm_ggml_reshape(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    // as only the shape of b is relevant, and not its memory layout, b is allowed to be non contiguous.
    LM_GGML_ASSERT(lm_ggml_nelements(a) == lm_ggml_nelements(b));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, LM_GGML_MAX_DIMS, b->ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = LM_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_reshape_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_nelements(a) == ne0);

    const int64_t ne[1] = { ne0 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = LM_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_reshape_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_nelements(a) == ne0*ne1);

    const int64_t ne[2] = { ne0, ne1 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = LM_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_reshape_3d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_nelements(a) == ne0*ne1*ne2);

    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = LM_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_reshape_4d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_nelements(a) == ne0*ne1*ne2*ne3);

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op     = LM_GGML_OP_RESHAPE;
    result->src[0] = a;

    return result;
}

static struct lm_ggml_tensor * lm_ggml_view_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_dims,
        const int64_t       * ne,
        size_t                offset) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    lm_ggml_format_name(result, "%s (view)", a->name);

    lm_ggml_set_op_params(result, &offset, sizeof(offset));

    result->op     = LM_GGML_OP_VIEW;
    result->src[0] = a;

    return result;
}

// lm_ggml_view_1d

struct lm_ggml_tensor * lm_ggml_view_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        size_t                offset) {
    struct lm_ggml_tensor * result = lm_ggml_view_impl(ctx, a, 1, &ne0, offset);

    return result;
}

// lm_ggml_view_2d

struct lm_ggml_tensor * lm_ggml_view_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        size_t                nb1,
        size_t                offset) {
    const int64_t ne[2] = { ne0, ne1 };

    struct lm_ggml_tensor * result = lm_ggml_view_impl(ctx, a, 2, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    return result;
}

// lm_ggml_view_3d

struct lm_ggml_tensor * lm_ggml_view_3d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        size_t                nb1,
        size_t                nb2,
        size_t                offset) {
    const int64_t ne[3] = { ne0, ne1, ne2 };

    struct lm_ggml_tensor * result = lm_ggml_view_impl(ctx, a, 3, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = result->nb[2]*ne2;

    return result;
}

// lm_ggml_view_4d

struct lm_ggml_tensor * lm_ggml_view_4d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0,
        int64_t               ne1,
        int64_t               ne2,
        int64_t               ne3,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };

    struct lm_ggml_tensor * result = lm_ggml_view_impl(ctx, a, 4, ne, offset);

    result->nb[1] = nb1;
    result->nb[2] = nb2;
    result->nb[3] = nb3;

    return result;
}

// lm_ggml_permute

struct lm_ggml_tensor * lm_ggml_permute(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    LM_GGML_ASSERT(axis0 >= 0 && axis0 < LM_GGML_MAX_DIMS);
    LM_GGML_ASSERT(axis1 >= 0 && axis1 < LM_GGML_MAX_DIMS);
    LM_GGML_ASSERT(axis2 >= 0 && axis2 < LM_GGML_MAX_DIMS);
    LM_GGML_ASSERT(axis3 >= 0 && axis3 < LM_GGML_MAX_DIMS);

    LM_GGML_ASSERT(axis0 != axis1);
    LM_GGML_ASSERT(axis0 != axis2);
    LM_GGML_ASSERT(axis0 != axis3);
    LM_GGML_ASSERT(axis1 != axis2);
    LM_GGML_ASSERT(axis1 != axis3);
    LM_GGML_ASSERT(axis2 != axis3);

    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);
    lm_ggml_format_name(result, "%s (permuted)", a->name);

    int ne[LM_GGML_MAX_DIMS];
    int nb[LM_GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op     = LM_GGML_OP_PERMUTE;
    result->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    return result;
}

// lm_ggml_transpose

struct lm_ggml_tensor * lm_ggml_transpose(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);
    lm_ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op     = LM_GGML_OP_TRANSPOSE;
    result->src[0] = a;

    return result;
}

// lm_ggml_get_rows

struct lm_ggml_tensor * lm_ggml_get_rows(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(a->ne[2] == b->ne[1]);
    LM_GGML_ASSERT(b->ne[3] == 1);
    LM_GGML_ASSERT(b->type == LM_GGML_TYPE_I32);

    // TODO: implement non F32 return
    enum lm_ggml_type type = LM_GGML_TYPE_F32;
    if (a->type == LM_GGML_TYPE_I32) {
        type = a->type;
    }
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op     = LM_GGML_OP_GET_ROWS;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_get_rows_back

struct lm_ggml_tensor * lm_ggml_get_rows_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c) {
    LM_GGML_ASSERT(lm_ggml_is_matrix(a) && lm_ggml_is_vector(b) && b->type == LM_GGML_TYPE_I32);
    LM_GGML_ASSERT(lm_ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

    // TODO: implement non F32 return
    //struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op     = LM_GGML_OP_GET_ROWS_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_diag

struct lm_ggml_tensor * lm_ggml_diag(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    LM_GGML_ASSERT(a->ne[1] == 1);

    const int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, 4, ne);

    result->op     = LM_GGML_OP_DIAG;
    result->src[0] = a;

    return result;
}

// lm_ggml_diag_mask_inf

static struct lm_ggml_tensor * lm_ggml_diag_mask_inf_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_DIAG_MASK_INF;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_diag_mask_inf(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past) {
    return lm_ggml_diag_mask_inf_impl(ctx, a, n_past, false);
}

struct lm_ggml_tensor * lm_ggml_diag_mask_inf_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past) {
    return lm_ggml_diag_mask_inf_impl(ctx, a, n_past, true);
}

// lm_ggml_diag_mask_zero

static struct lm_ggml_tensor * lm_ggml_diag_mask_zero_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_DIAG_MASK_ZERO;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_diag_mask_zero(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past) {
    return lm_ggml_diag_mask_zero_impl(ctx, a, n_past, false);
}

struct lm_ggml_tensor * lm_ggml_diag_mask_zero_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past) {
    return lm_ggml_diag_mask_zero_impl(ctx, a, n_past, true);
}

// lm_ggml_soft_max

static struct lm_ggml_tensor * lm_ggml_soft_max_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));

    if (mask) {
        LM_GGML_ASSERT(mask->type == LM_GGML_TYPE_F16 || mask->type == LM_GGML_TYPE_F32);
        LM_GGML_ASSERT(lm_ggml_is_contiguous(mask));
        LM_GGML_ASSERT(lm_ggml_is_matrix(mask));
        LM_GGML_ASSERT(mask->ne[0] == a->ne[0]);
        LM_GGML_ASSERT(mask->ne[1] >= a->ne[1]);
    }

    if (max_bias > 0.0f) {
        LM_GGML_ASSERT(mask);
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    float params[] = { scale, max_bias };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_SOFT_MAX;
    result->src[0] = a;
    result->src[1] = mask;

    return result;
}

struct lm_ggml_tensor * lm_ggml_soft_max(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}

struct lm_ggml_tensor * lm_ggml_soft_max_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    return lm_ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, true);
}

struct lm_ggml_tensor * lm_ggml_soft_max_ext(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias) {
    return lm_ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

// lm_ggml_soft_max_back

static struct lm_ggml_tensor * lm_ggml_soft_max_back_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        bool                  inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_SOFT_MAX_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_soft_max_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_soft_max_back_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_soft_max_back_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_soft_max_back_impl(ctx, a, b, true);
}

// lm_ggml_rope

static struct lm_ggml_tensor * lm_ggml_rope_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        bool                  inplace) {
    LM_GGML_ASSERT((mode & 1) == 0 && "mode & 1 == 1 is no longer supported");

    LM_GGML_ASSERT(lm_ggml_is_vector(b));
    LM_GGML_ASSERT(b->type == LM_GGML_TYPE_I32);
    LM_GGML_ASSERT(a->ne[2] == b->ne[0]);

    if (c) {
        LM_GGML_ASSERT(c->type == LM_GGML_TYPE_F32);
        LM_GGML_ASSERT(c->ne[0] >= n_dims / 2);
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[11] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_ROPE;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct lm_ggml_tensor * lm_ggml_rope(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return lm_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    );
}

struct lm_ggml_tensor * lm_ggml_rope_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   n_dims,
        int                   mode) {
    return lm_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, true
    );
}

struct lm_ggml_tensor * lm_ggml_rope_ext(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return lm_ggml_rope_impl(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct lm_ggml_tensor * lm_ggml_rope_ext_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return lm_ggml_rope_impl(
        ctx, a, b, c, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

struct lm_ggml_tensor * lm_ggml_rope_custom(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return lm_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, false
    );
}

struct lm_ggml_tensor * lm_ggml_rope_custom_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    return lm_ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, true
    );
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float lm_ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

void lm_ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(lm_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(lm_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

// lm_ggml_rope_back

struct lm_ggml_tensor * lm_ggml_rope_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c,
        int                   n_dims,
        int                   mode,
        int                   n_ctx_orig,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow) {
    LM_GGML_ASSERT(lm_ggml_is_vector(b));
    LM_GGML_ASSERT(b->type == LM_GGML_TYPE_I32);
    LM_GGML_ASSERT(a->ne[2] == b->ne[0]);

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    int32_t params[11] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_ROPE_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

// lm_ggml_clamp

struct lm_ggml_tensor * lm_ggml_clamp(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 min,
        float                 max) {
    // TODO: when implement backward, fix this:
    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);

    float params[] = { min, max };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_CLAMP;
    result->src[0] = a;

    return result;
}

// lm_ggml_conv_1d

static int64_t lm_ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}

LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s0,
        int                   p0,
        int                   d0) {
    struct lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, LM_GGML_TYPE_F16); // [N, OL, IC * K]

    struct lm_ggml_tensor * result =
        lm_ggml_mul_mat(ctx,
                lm_ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])), // [N, OL, IC * K] => [N*OL, IC * K]
                lm_ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));                    // [OC，IC, K] => [OC, IC * K]

    result = lm_ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]); // [N, OC, OL]

    return result;
}

// lm_ggml_conv_1d_ph

struct lm_ggml_tensor* lm_ggml_conv_1d_ph(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s,
        int                   d) {
    return lm_ggml_conv_1d(ctx, a, b, s, a->ne[0] / 2, d);
}

// lm_ggml_conv_transpose_1d

static int64_t lm_ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

LM_GGML_API struct lm_ggml_tensor * lm_ggml_conv_transpose_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s0,
        int                   p0,
        int                   d0) {
    LM_GGML_ASSERT(lm_ggml_is_matrix(b));
    LM_GGML_ASSERT(a->ne[2] == b->ne[1]);
    LM_GGML_ASSERT(a->ne[3] == 1);

    LM_GGML_ASSERT(p0 == 0);
    LM_GGML_ASSERT(d0 == 1);

    const int64_t ne[4] = {
        lm_ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_CONV_TRANSPOSE_1D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_conv_depthwise

struct lm_ggml_tensor * lm_ggml_conv_depthwise_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1) {
    struct lm_ggml_tensor * new_a = lm_ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    struct lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, new_a,
                                        lm_ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                        s0, s1, p0, p1, d0, d1, true, LM_GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]
    struct lm_ggml_tensor * new_b = lm_ggml_reshape_4d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]); // [N * IC, OH, OW, KH * KW] => [N, IC, OH * OW, KH * KW]

    new_a = lm_ggml_reshape_4d(ctx, new_a, (new_a->ne[0] * new_a->ne[1]), new_a->ne[2],  new_a->ne[3], 1);                       // [OC，1, KH, KW] => [1, OC, 1, KH * KW]
    struct lm_ggml_tensor * result = lm_ggml_mul_mat(ctx, new_a, new_b);
    result = lm_ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]); // [N, OC, OH, OW]

    return result;
}
// lm_ggml_conv_2d

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct lm_ggml_tensor * lm_ggml_im2col(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D,
        enum lm_ggml_type        dst_type) {
    if(is_2D) {
        LM_GGML_ASSERT(a->ne[2] == b->ne[2]);
    } else {
        LM_GGML_ASSERT(a->ne[1] == b->ne[1]);
        LM_GGML_ASSERT(b->ne[3] == 1);
    }

    const int64_t OH = is_2D ? lm_ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW =         lm_ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    LM_GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    LM_GGML_ASSERT((OW > 0)           && "b too small compared to a");

    const int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_IM2COL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_im2col_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int64_t             * ne,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_IM2COL_BACK;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OC, OH, OW]
struct lm_ggml_tensor * lm_ggml_conv_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1) {
    struct lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type); // [N, OH, OW, IC * KH * KW]

    struct lm_ggml_tensor * result =
        lm_ggml_mul_mat(ctx,
                lm_ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
                lm_ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = lm_ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = lm_ggml_cont(ctx, lm_ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]


    return result;
}

// lm_ggml_conv_2d_sk_p0

struct lm_ggml_tensor * lm_ggml_conv_2d_sk_p0(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_conv_2d(ctx, a, b, a->ne[0], a->ne[1], 0, 0, 1, 1);
}

// lm_ggml_conv_2d_s1_ph

struct lm_ggml_tensor * lm_ggml_conv_2d_s1_ph(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    return lm_ggml_conv_2d(ctx, a, b, 1, 1, a->ne[0] / 2, a->ne[1] / 2, 1, 1);
}

// lm_ggml_conv_transpose_2d_p0

static int64_t lm_ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
    return (ins - 1) * s - 2 * p + ks;
}

struct lm_ggml_tensor * lm_ggml_conv_transpose_2d_p0(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        int                   stride) {
    LM_GGML_ASSERT(a->ne[3] == b->ne[2]);

    const int64_t ne[4] = {
        lm_ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        lm_ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    struct lm_ggml_tensor* result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    lm_ggml_set_op_params_i32(result, 0, stride);

    result->op     = LM_GGML_OP_CONV_TRANSPOSE_2D;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_pool_*

static int64_t lm_ggml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
    return (ins + 2 * p - ks) / s + 1;
}

// lm_ggml_pool_1d

struct lm_ggml_tensor * lm_ggml_pool_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_op_pool     op,
        int                   k0,
        int                   s0,
        int                   p0) {
    const int64_t ne[4] = {
        lm_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_POOL_1D;
    result->src[0] = a;

    return result;
}

// lm_ggml_pool_2d

struct lm_ggml_tensor * lm_ggml_pool_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_op_pool     op,
        int                   k0,
        int                   k1,
        int                   s0,
        int                   s1,
        float                 p0,
        float                 p1) {
    struct lm_ggml_tensor * result;
    const int64_t ne[4] = {
        lm_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        lm_ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_POOL_2D;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_pool_2d_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * af,
        enum lm_ggml_op_pool     op,
        int                   k0,
        int                   k1,
        int                   s0,
        int                   s1,
        float                 p0,
        float                 p1) {
    struct lm_ggml_tensor * result;
    result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, af->ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_POOL_2D_BACK;
    result->src[0] = a;
    result->src[1] = af;

    return result;
}

// lm_ggml_upscale

static struct lm_ggml_tensor * lm_ggml_upscale_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2,
        int                   ne3) {
    LM_GGML_ASSERT(a->ne[0] <= ne0);
    LM_GGML_ASSERT(a->ne[1] <= ne1);
    LM_GGML_ASSERT(a->ne[2] <= ne2);
    LM_GGML_ASSERT(a->ne[3] <= ne3);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);

    result->op     = LM_GGML_OP_UPSCALE;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_upscale(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   scale_factor) {
    return lm_ggml_upscale_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3]);
}

struct lm_ggml_tensor * lm_ggml_upscale_ext(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2,
        int                   ne3) {
    return lm_ggml_upscale_impl(ctx, a, ne0, ne1, ne2, ne3);
}

// lm_ggml_pad

struct lm_ggml_tensor * lm_ggml_pad(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   p0,
        int                   p1,
        int                   p2,
        int                   p3) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0,
            a->ne[1] + p1,
            a->ne[2] + p2,
            a->ne[3] + p3);

    result->op     = LM_GGML_OP_PAD;
    result->src[0] = a;

    return result;
}

// lm_ggml_arange

struct lm_ggml_tensor * lm_ggml_arange(
        struct lm_ggml_context * ctx,
        float                 start,
        float                 stop,
        float                 step) {
    LM_GGML_ASSERT(stop > start);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, steps);

    lm_ggml_set_op_params_f32(result, 0, start);
    lm_ggml_set_op_params_f32(result, 1, stop);
    lm_ggml_set_op_params_f32(result, 2, step);

    result->op = LM_GGML_OP_ARANGE;

    return result;
}

// lm_ggml_timestep_embedding

struct lm_ggml_tensor * lm_ggml_timestep_embedding(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * timesteps,
        int                   dim,
        int                   max_period) {
    int actual_dim = dim;
    if (dim % 2 != 0) {
        actual_dim = dim + 1;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, actual_dim, timesteps->ne[0]);

    lm_ggml_set_op_params_i32(result, 0, dim);
    lm_ggml_set_op_params_i32(result, 1, max_period);

    result->op     = LM_GGML_OP_TIMESTEP_EMBEDDING;
    result->src[0] = timesteps;

    return result;
}

// lm_ggml_argsort

struct lm_ggml_tensor * lm_ggml_argsort(
        struct lm_ggml_context  * ctx,
        struct lm_ggml_tensor   * a,
        enum lm_ggml_sort_order   order) {
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_I32, LM_GGML_MAX_DIMS, a->ne);

    lm_ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op     = LM_GGML_OP_ARGSORT;
    result->src[0] = a;

    return result;
}

// lm_ggml_top_k

struct lm_ggml_tensor * lm_ggml_top_k(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   k) {
    LM_GGML_ASSERT(a->ne[0] >= k);

    struct lm_ggml_tensor * result = lm_ggml_argsort(ctx, a, LM_GGML_SORT_ORDER_DESC);

    result = lm_ggml_view_4d(ctx, result,
                k, result->ne[1], result->ne[2], result->ne[3],
                   result->nb[1], result->nb[2], result->nb[3],
                0);

    return result;
}

// lm_ggml_flash_attn_ext

struct lm_ggml_tensor * lm_ggml_flash_attn_ext(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * q,
        struct lm_ggml_tensor  * k,
        struct lm_ggml_tensor  * v,
        struct lm_ggml_tensor  * mask,
        float                 scale,
        float                 max_bias,
        float                 logit_softcap) {
    LM_GGML_ASSERT(lm_ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    if (mask) {
        LM_GGML_ASSERT(lm_ggml_is_contiguous(mask));
        LM_GGML_ASSERT(mask->ne[2] == 1);
        LM_GGML_ASSERT(mask->ne[3] == 1);
        LM_GGML_ASSERT(mask->ne[1] >= LM_GGML_PAD(q->ne[1], LM_GGML_KQ_MASK_PAD) &&
                "the Flash-Attention kernel requires the mask to be padded to LM_GGML_KQ_MASK_PAD and at least n_queries big");
        //LM_GGML_ASSERT(lm_ggml_can_repeat_rows(mask, qk));
    }

    if (max_bias > 0.0f) {
        LM_GGML_ASSERT(mask);
    }

    bool is_node = false;

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { q->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    float params[] = { scale, max_bias, logit_softcap };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_FLASH_ATTN_EXT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = mask;

    return result;
}

void lm_ggml_flash_attn_ext_set_prec(
        struct lm_ggml_tensor * a,
        enum lm_ggml_prec       prec) {
    LM_GGML_ASSERT(a->op == LM_GGML_OP_FLASH_ATTN_EXT);

    const int32_t prec_i32 = (int32_t) prec;

    lm_ggml_set_op_params_i32(a, 3, prec_i32); // scale is on first pos, max_bias on second
}

enum lm_ggml_prec lm_ggml_flash_attn_ext_get_prec(
        const struct lm_ggml_tensor * a) {
    LM_GGML_ASSERT(a->op == LM_GGML_OP_FLASH_ATTN_EXT);

    const int32_t prec_i32 = lm_ggml_get_op_params_i32(a, 3);

    return (enum lm_ggml_prec) prec_i32;
}

// lm_ggml_flash_attn_back

struct lm_ggml_tensor * lm_ggml_flash_attn_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * q,
        struct lm_ggml_tensor  * k,
        struct lm_ggml_tensor  * v,
        struct lm_ggml_tensor  * d,
        bool                  masked) {
    LM_GGML_ABORT("TODO: adapt to lm_ggml_flash_attn_ext() changes");

    LM_GGML_ASSERT(lm_ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    // d shape [D,N,ne2,ne3]
    // q shape [D,N,ne2,ne3]
    // k shape [D,M,kvne2,ne3]
    // v shape [M,D,kvne2,ne3]

    const int64_t     D = q->ne[0];
    const int64_t     N = q->ne[1];
    const int64_t     M = k->ne[1];
    const int64_t   ne2 = q->ne[2];
    const int64_t   ne3 = q->ne[3];
    const int64_t kvne2 = k->ne[2];

    LM_GGML_ASSERT(k->ne[0] == D);
    LM_GGML_ASSERT(v->ne[0] == M);
    LM_GGML_ASSERT(v->ne[1] == D);
    LM_GGML_ASSERT(d->ne[0] == D);
    LM_GGML_ASSERT(d->ne[1] == N);
    LM_GGML_ASSERT(k->ne[2] == kvne2);
    LM_GGML_ASSERT(k->ne[3] == ne3);
    LM_GGML_ASSERT(v->ne[2] == kvne2);
    LM_GGML_ASSERT(v->ne[3] == ne3);
    LM_GGML_ASSERT(d->ne[2] == ne2);
    LM_GGML_ASSERT(d->ne[3] == ne3);

    LM_GGML_ASSERT(ne2 % kvne2 == 0);

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        // when using this operation (in backwards pass) these grads are set.
        // we don't want to create (big) grad of our result, so is_node is false.
        is_node = false;
    }

    // store gradients of q, k and v as continuous tensors concatenated in result.
    // note: v and gradv are actually transposed, i.e. v->ne[0] != D.
    const int64_t elem_q = lm_ggml_nelements(q);
    const int64_t elem_k = lm_ggml_nelements(k);
    const int64_t elem_v = lm_ggml_nelements(v);

    enum lm_ggml_type result_type = LM_GGML_TYPE_F32;
    LM_GGML_ASSERT(lm_ggml_blck_size(result_type) == 1);
    const size_t tsize = lm_ggml_type_size(result_type);

    const size_t offs_q = 0;
    const size_t offs_k = offs_q + LM_GGML_PAD(elem_q * tsize, LM_GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + LM_GGML_PAD(elem_k * tsize, LM_GGML_MEM_ALIGN);
    const size_t end    = offs_v + LM_GGML_PAD(elem_v * tsize, LM_GGML_MEM_ALIGN);

    const size_t nelements = (end + tsize - 1)/tsize;

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, nelements);

    int32_t masked_i = masked ? 1 : 0;
    lm_ggml_set_op_params(result, &masked_i, sizeof(masked_i));

    result->op   = LM_GGML_OP_FLASH_ATTN_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = d;

    return result;
}

// lm_ggml_ssm_conv

struct lm_ggml_tensor * lm_ggml_ssm_conv(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * sx,
        struct lm_ggml_tensor  * c) {
    LM_GGML_ASSERT(lm_ggml_is_3d(sx));
    LM_GGML_ASSERT(lm_ggml_is_matrix(c));

    const int64_t d_conv  = c->ne[0];
    const int64_t d_inner = c->ne[1];
    const int64_t n_t     = sx->ne[0] - d_conv + 1; // tokens per sequence
    const int64_t n_s     = sx->ne[2];

    // TODO: maybe support other strides than 1?
    // FIXME: this is always true?
    LM_GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);
    LM_GGML_ASSERT(sx->ne[1] == d_inner);
    LM_GGML_ASSERT(n_t >= 0);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_3d(ctx, LM_GGML_TYPE_F32, d_inner, n_t, n_s);

    result->op     = LM_GGML_OP_SSM_CONV;
    result->src[0] = sx;
    result->src[1] = c;

    return result;
}

// lm_ggml_ssm_scan

struct lm_ggml_tensor * lm_ggml_ssm_scan(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * s,
        struct lm_ggml_tensor  * x,
        struct lm_ggml_tensor  * dt,
        struct lm_ggml_tensor  * A,
        struct lm_ggml_tensor  * B,
        struct lm_ggml_tensor  * C) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(s));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(x));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dt));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(A));
    LM_GGML_ASSERT(lm_ggml_is_matrix(A));
    LM_GGML_ASSERT(lm_ggml_is_3d(B));
    LM_GGML_ASSERT(lm_ggml_is_3d(s));
    LM_GGML_ASSERT(B->nb[0] == lm_ggml_type_size(B->type));
    LM_GGML_ASSERT(C->nb[0] == lm_ggml_type_size(C->type));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(x, dt));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(B, C));

    {
        const int64_t d_state      = s->ne[0];
        const int64_t d_inner      = s->ne[1];
        const int64_t n_seq_tokens = x->ne[1];
        const int64_t n_seqs       = x->ne[2];

        LM_GGML_ASSERT(s->ne[2] == n_seqs);
        LM_GGML_ASSERT(x->ne[0] == d_inner);
        LM_GGML_ASSERT(A->ne[0] == d_state);
        LM_GGML_ASSERT(A->ne[1] == d_inner);
        LM_GGML_ASSERT(B->ne[0] == d_state);
        LM_GGML_ASSERT(B->ne[1] == n_seq_tokens);
        LM_GGML_ASSERT(B->ne[2] == n_seqs);
    }

    // concatenated y + ssm_states
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, lm_ggml_nelements(x) + lm_ggml_nelements(s));

    result->op   = LM_GGML_OP_SSM_SCAN;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;

    return result;
}

// lm_ggml_win_part

struct lm_ggml_tensor * lm_ggml_win_part(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   w) {
    LM_GGML_ASSERT(a->ne[3] == 1);
    LM_GGML_ASSERT(a->type  == LM_GGML_TYPE_F32);

    // padding
    const int px = (w - a->ne[1]%w)%w;
    const int py = (w - a->ne[2]%w)%w;

    const int npx = (px + a->ne[1])/w;
    const int npy = (py + a->ne[2])/w;
    const int np  = npx*npy;

    const int64_t ne[4] = { a->ne[0], w, w, np, };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { npx, npy, w };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_WIN_PART;
    result->src[0] = a;

    return result;
}

// lm_ggml_win_unpart

struct lm_ggml_tensor * lm_ggml_win_unpart(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   w0,
        int                   h0,
        int                   w) {
    LM_GGML_ASSERT(a->type == LM_GGML_TYPE_F32);

    const int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op     = LM_GGML_OP_WIN_UNPART;
    result->src[0] = a;

    return result;
}

// lm_ggml_get_rel_pos

struct lm_ggml_tensor * lm_ggml_get_rel_pos(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   qh,
        int                   kh) {
    LM_GGML_ASSERT(qh == kh);
    LM_GGML_ASSERT(2*MAX(qh, kh) - 1 == a->ne[1]);

    const int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F16, 3, ne);

    result->op     = LM_GGML_OP_GET_REL_POS;
    result->src[0] = a;

    return result;
}

// lm_ggml_add_rel_pos

static struct lm_ggml_tensor * lm_ggml_add_rel_pos_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * pw,
        struct lm_ggml_tensor  * ph,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(pw, ph));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(pw));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(ph));
    LM_GGML_ASSERT(ph->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(pw->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(pw->ne[3] == a->ne[2]);
    LM_GGML_ASSERT(pw->ne[0]*pw->ne[0] == a->ne[0]);
    LM_GGML_ASSERT(pw->ne[1]*pw->ne[2] == a->ne[1]);

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);
    lm_ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op     = LM_GGML_OP_ADD_REL_POS;
    result->src[0] = a;
    result->src[1] = pw;
    result->src[2] = ph;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add_rel_pos(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * pw,
        struct lm_ggml_tensor  * ph) {
    return lm_ggml_add_rel_pos_impl(ctx, a, pw, ph, false);
}

struct lm_ggml_tensor * lm_ggml_add_rel_pos_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * pw,
        struct lm_ggml_tensor  * ph) {
    return lm_ggml_add_rel_pos_impl(ctx, a, pw, ph, true);
}

// lm_ggml_rwkv_wkv6

struct lm_ggml_tensor * lm_ggml_rwkv_wkv6(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * k,
        struct lm_ggml_tensor  * v,
        struct lm_ggml_tensor  * r,
        struct lm_ggml_tensor  * tf,
        struct lm_ggml_tensor  * td,
        struct lm_ggml_tensor  * state) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(k));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(v));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(r));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(tf));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(td));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(state));

    const int64_t S = k->ne[0];
    const int64_t H = k->ne[2];
    const int64_t n_tokens = k->ne[3];
    const int64_t n_seqs = state->ne[1];
    {
        LM_GGML_ASSERT(k->ne[1] == 1);
        LM_GGML_ASSERT(v->ne[0] == 1 && v->ne[1] == S && v->ne[2] == H && v->ne[3] == n_tokens);
        LM_GGML_ASSERT(r->ne[0] == 1 && r->ne[1] == S && r->ne[2] == H && r->ne[3] == n_tokens);
        // TODO: RWKV v4 and v5
        LM_GGML_ASSERT(td->ne[0] == 1 && td->ne[1] == S && td->ne[2] == H && td->ne[3] == n_tokens);
        LM_GGML_ASSERT(lm_ggml_nelements(state) == S * S * H * n_seqs);
    }

    // concat output and new_state
    const int64_t ne[4] = { S * H, n_tokens + S * n_seqs, 1, 1 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op     = LM_GGML_OP_RWKV_WKV6;
    result->src[0] = k;
    result->src[1] = v;
    result->src[2] = r;
    result->src[3] = tf;
    result->src[4] = td;
    result->src[5] = state;

    return result;
}

// lm_ggml_unary

static struct lm_ggml_tensor * lm_ggml_unary_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op    op,
        bool                  inplace) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous_1(a));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op     = LM_GGML_OP_UNARY;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_unary(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op    op) {
    return lm_ggml_unary_impl(ctx, a, op, false);
}

struct lm_ggml_tensor * lm_ggml_unary_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op    op) {
    return lm_ggml_unary_impl(ctx, a, op, true);
}

// lm_ggml_map_unary

static struct lm_ggml_tensor * lm_ggml_map_unary_impl_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t   fun,
        bool                         inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = LM_GGML_OP_MAP_UNARY;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_unary_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t   fun) {
    return lm_ggml_map_unary_impl_f32(ctx, a, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_unary_inplace_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t   fun) {
    return lm_ggml_map_unary_impl_f32(ctx, a, fun, true);
}

// lm_ggml_map_binary

static struct lm_ggml_tensor * lm_ggml_map_binary_impl_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t   fun,
        bool                          inplace) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = LM_GGML_OP_MAP_BINARY;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_binary_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t   fun) {
    return lm_ggml_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_binary_inplace_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t   fun) {
    return lm_ggml_map_binary_impl_f32(ctx, a, b, fun, true);
}

// lm_ggml_map_custom1_f32

static struct lm_ggml_tensor * lm_ggml_map_custom1_impl_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_f32_t   fun,
        bool                           inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = LM_GGML_OP_MAP_CUSTOM1_F32;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom1_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_f32_t   fun) {
    return lm_ggml_map_custom1_impl_f32(ctx, a, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom1_inplace_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_f32_t   fun) {
    return lm_ggml_map_custom1_impl_f32(ctx, a, fun, true);
}

// lm_ggml_map_custom2_f32

static struct lm_ggml_tensor * lm_ggml_map_custom2_impl_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_f32_t   fun,
        bool                           inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = LM_GGML_OP_MAP_CUSTOM2_F32;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom2_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_f32_t   fun) {
    return lm_ggml_map_custom2_impl_f32(ctx, a, b, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom2_inplace_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_f32_t   fun) {
    return lm_ggml_map_custom2_impl_f32(ctx, a, b, fun, true);
}

// lm_ggml_map_custom3_f32

static struct lm_ggml_tensor * lm_ggml_map_custom3_impl_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_f32_t   fun,
        bool                           inplace) {
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op     = LM_GGML_OP_MAP_CUSTOM3_F32;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom3_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_f32_t   fun) {
    return lm_ggml_map_custom3_impl_f32(ctx, a, b, c, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom3_inplace_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_f32_t   fun) {
    return lm_ggml_map_custom3_impl_f32(ctx, a, b, c, fun, true);
}

// lm_ggml_map_custom1

static struct lm_ggml_tensor * lm_ggml_map_custom1_impl(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        const  lm_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = LM_GGML_OP_MAP_CUSTOM1;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom1(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        const  lm_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom1_inplace(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        const  lm_ggml_custom1_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// lm_ggml_map_custom2

static struct lm_ggml_tensor * lm_ggml_map_custom2_impl(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        const  lm_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = LM_GGML_OP_MAP_CUSTOM2;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom2(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        const  lm_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom2_inplace(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        const  lm_ggml_custom2_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// lm_ggml_map_custom3

static struct lm_ggml_tensor * lm_ggml_map_custom3_impl(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        struct lm_ggml_tensor       * c,
        const  lm_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata,
        bool                       inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op     = LM_GGML_OP_MAP_CUSTOM3;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom3(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        struct lm_ggml_tensor       * c,
        const  lm_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom3_inplace(
        struct lm_ggml_context      * ctx,
        struct lm_ggml_tensor       * a,
        struct lm_ggml_tensor       * b,
        struct lm_ggml_tensor       * c,
        const  lm_ggml_custom3_op_t   fun,
        int                        n_tasks,
        void                     * userdata) {
    return lm_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

// lm_ggml_cross_entropy_loss

struct lm_ggml_tensor * lm_ggml_cross_entropy_loss(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = LM_GGML_OP_CROSS_ENTROPY_LOSS;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_cross_entropy_loss_back

struct lm_ggml_tensor * lm_ggml_cross_entropy_loss_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        struct lm_ggml_tensor  * c) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
    LM_GGML_ASSERT(lm_ggml_is_scalar(c));

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    result->op     = LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

// opt_step_adamw

struct lm_ggml_tensor * lm_ggml_opt_step_adamw(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * grad,
        float                 alpha,
        float                 beta1,
        float                 beta2,
        float                 eps,
        float                 wd) {
    LM_GGML_ASSERT(a->flags & LM_GGML_TENSOR_FLAG_PARAM);
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, grad));
    LM_GGML_ASSERT(alpha >  0.0f);
    LM_GGML_ASSERT(beta1 >= 0.0f && beta1 <= 1.0f);
    LM_GGML_ASSERT(beta2 >= 0.0f && beta2 <= 1.0f);
    LM_GGML_ASSERT(eps   >= 0.0f);
    LM_GGML_ASSERT(wd    >= 0.0f && wd    <= 1.0f);

    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);

    const int64_t iter = 1;
    memcpy(&result->op_params[0], &iter, sizeof(int64_t));
    lm_ggml_set_op_params_f32(result, 2, alpha);
    lm_ggml_set_op_params_f32(result, 3, beta1);
    lm_ggml_set_op_params_f32(result, 4, beta2);
    lm_ggml_set_op_params_f32(result, 5, eps);
    lm_ggml_set_op_params_f32(result, 6, wd);

    result->op     = LM_GGML_OP_OPT_STEP_ADAMW;
    result->src[0] = a;
    result->src[1] = grad;
    result->src[2] = lm_ggml_dup_tensor(ctx, grad);
    result->src[3] = lm_ggml_dup_tensor(ctx, grad);

    return result;
}

////////////////////////////////////////////////////////////////////////////////

struct lm_ggml_hash_set lm_ggml_hash_set_new(size_t size) {
    size = lm_ggml_hash_size(size);
    struct lm_ggml_hash_set result;
    result.size = size;
    result.keys = LM_GGML_MALLOC(sizeof(struct lm_ggml_tensor *) * size);
    result.used = LM_GGML_CALLOC(lm_ggml_bitset_size(size), sizeof(lm_ggml_bitset_t));
    return result;
}

void lm_ggml_hash_set_reset(struct lm_ggml_hash_set * hash_set) {
    memset(hash_set->used, 0, sizeof(lm_ggml_bitset_t) * lm_ggml_bitset_size(hash_set->size));
}

void lm_ggml_hash_set_free(struct lm_ggml_hash_set * hash_set) {
    LM_GGML_FREE(hash_set->used);
    LM_GGML_FREE(hash_set->keys);
}

size_t lm_ggml_hash_size(size_t min_sz) {
    // next primes after powers of two
    static const size_t primes[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    static const size_t n_primes = sizeof(primes)/sizeof(primes[0]);

    // find the smallest prime that is larger or equal than min_sz
    size_t l = 0;
    size_t r = n_primes;
    while (l < r) {
        size_t m = (l + r)/2;
        if (primes[m] < min_sz) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    size_t sz = l < n_primes ? primes[l] : min_sz | 1;
    return sz;
}

struct hash_map {
    struct lm_ggml_hash_set set;
    struct lm_ggml_tensor ** vals;
};

static struct hash_map * lm_ggml_new_hash_map(size_t size) {
    struct hash_map * result = LM_GGML_MALLOC(sizeof(struct hash_map));
    result->set = lm_ggml_hash_set_new(size);
    result->vals = LM_GGML_CALLOC(result->set.size, sizeof(struct lm_ggml_tensor *));
    return result;
}

static void lm_ggml_hash_map_free(struct hash_map * map) {
    lm_ggml_hash_set_free(&map->set);
    LM_GGML_FREE(map->vals);
    LM_GGML_FREE(map);
}

// gradient checkpointing

static struct lm_ggml_tensor * lm_ggml_recompute_graph_node(
        struct lm_ggml_context * ctx,
        struct lm_ggml_cgraph  * graph,
        struct hash_map     * replacements,
        struct lm_ggml_tensor  * node) {

    if (node == NULL) {
        return NULL;
    }

    if (node->flags & LM_GGML_TENSOR_FLAG_PARAM) {
        return node;
    }

    if (!lm_ggml_hash_contains(&graph->visited_hash_set, node)) {
        return node;
    }

    int count_children = 0;
    for (int k = 0; k < LM_GGML_MAX_SRC; ++k) {
        if (node->src[k]) {
            ++count_children;
        }
    }

    if (count_children == 0) {
        return node;
    }

    size_t i = lm_ggml_hash_find(&replacements->set, node);
    LM_GGML_ASSERT(i != LM_GGML_HASHSET_FULL); // assert that not full
    if (replacements->set.keys[i] == node) {
        return replacements->vals[i];
    }

    struct lm_ggml_tensor * clone = lm_ggml_new_tensor(ctx, node->type, LM_GGML_MAX_DIMS, node->ne);

    // insert clone into replacements
    LM_GGML_ASSERT(replacements->set.keys[i] == NULL); // assert that we don't overwrite
    replacements->set.keys[i] = node;
    replacements->vals[i] = clone;

    clone->op       = node->op;
    clone->grad     = node->grad;
    clone->flags    = node->flags;
    clone->extra    = node->extra;
    for (int k = 0; k < LM_GGML_MAX_DIMS; ++k) {
        clone->nb[k] = node->nb[k];
    }
    for (int k = 0; k < LM_GGML_MAX_SRC; ++k) {
        clone->src[k] = lm_ggml_recompute_graph_node(ctx, graph, replacements, node->src[k]);
    }
    if (node->view_src != NULL) {
        clone->data = (node->view_src->data == NULL)
                        ? NULL // view_src not yet allocated
                        : (char *) node->view_src->data // view_src already allocated
                                 + node->view_offs;
        clone->view_src  = node->view_src;
        clone->view_offs = node->view_offs;
    }

    LM_GGML_ASSERT(sizeof(node->op_params) == sizeof(int32_t) * (LM_GGML_MAX_OP_PARAMS / sizeof(int32_t)));
    LM_GGML_ASSERT(sizeof(node->name)      == LM_GGML_MAX_NAME);
    memcpy(clone->op_params, node->op_params, sizeof(node->op_params));
    lm_ggml_format_name(clone, "%s (clone)", lm_ggml_get_name(node));

    return clone;
}

void lm_ggml_build_backward_gradient_checkpointing(
        struct lm_ggml_context   * ctx,
        struct lm_ggml_cgraph    * gf,
        struct lm_ggml_cgraph    * gb,
        struct lm_ggml_cgraph    * gb_tmp,
        struct lm_ggml_tensor  * * checkpoints,
        int                     n_checkpoints) {
    lm_ggml_graph_cpy(gf, gb_tmp);
    lm_ggml_build_backward_expand(ctx, gf, gb_tmp, false);

    if (n_checkpoints <= 0) {
        lm_ggml_graph_cpy(gb_tmp, gb);
        return;
    }

    struct hash_map * replacements = lm_ggml_new_hash_map(gf->n_nodes + gf->n_leafs + n_checkpoints);

    // insert checkpoints in replacements
    for (int i = 0; i < n_checkpoints; ++i) {
        size_t k = lm_ggml_hash_find(&replacements->set, checkpoints[i]);
        LM_GGML_ASSERT(k != LM_GGML_HASHSET_FULL); // assert that not full
        LM_GGML_ASSERT(replacements->set.keys[k] == NULL); // assert that we don't overwrite
        replacements->set.keys[k] = checkpoints[i];
        replacements->vals[k]     = checkpoints[i];
    }

    lm_ggml_graph_cpy(gf, gb);
    // rewrite gb_tmp->nodes[gf->n_nodes:gb_tmp->n_nodes],
    // replacing references to gb_tmp->nodes[0:gf->n_nodes] ( == gf->nodes[0:gf->n_nodes]),
    // by recomputing them from checkpoints
    for (int i = gf->n_nodes; i<gb_tmp->n_nodes; ++i) {
        struct lm_ggml_tensor * node = gb_tmp->nodes[i];
        for (int k = 0; k < LM_GGML_MAX_SRC; ++k) {
            // insert new tensors recomputing src, reusing already made replacements,
            // remember replacements: remember new tensors with mapping from corresponding gf nodes
            // recurse for input tensors,
            // unless (i.e. terminating when) input tensors are replacements (like checkpoints)
            node->src[k] = lm_ggml_recompute_graph_node(ctx, gf, replacements, node->src[k]);
        }
        // insert rewritten backward node with replacements made into resulting backward graph gb
        lm_ggml_build_forward_expand(gb, node);
    }

    lm_ggml_hash_map_free(replacements);
}

// utility functions to change gradients
// if a is in acc_table, modify gradients in-place and mark result as gradient accumulator
// else if a is in zero_table, replace a
// else, just add/subtract/etc. the gradients

static struct lm_ggml_tensor * lm_ggml_add_or_set(
        struct lm_ggml_context  * ctx,
        struct lm_ggml_tensor   * a,
        struct lm_ggml_tensor   * b,
        struct lm_ggml_hash_set * zero_table,
        struct lm_ggml_hash_set * acc_table) {
    if (lm_ggml_hash_contains(acc_table, a)) {
        struct lm_ggml_tensor * ret = lm_ggml_add_impl(ctx, a, b, true);
        const size_t insert_result = lm_ggml_hash_insert(acc_table, ret);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (lm_ggml_hash_contains(zero_table, a)) {
        return b;
    }
    return lm_ggml_add_impl(ctx, a, b, false);
}

static struct lm_ggml_tensor * lm_ggml_acc_or_set(
        struct lm_ggml_context  * ctx,
        struct lm_ggml_tensor   * a,
        struct lm_ggml_tensor   * b,
        const  size_t          nb1,
        const  size_t          nb2,
        const  size_t          nb3,
        const  size_t          offset,
        struct lm_ggml_hash_set * zero_table,
        struct lm_ggml_hash_set * acc_table) {
    if (lm_ggml_hash_contains(acc_table, a)) {
        struct lm_ggml_tensor * ret = lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
        const size_t insert_result = lm_ggml_hash_insert(acc_table, ret);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (lm_ggml_hash_contains(zero_table, a)) {
        struct lm_ggml_tensor * a_zero = lm_ggml_scale(ctx, a, 0.0f); // FIXME this is going to produce NaN if a contains inf/NaN
        return lm_ggml_acc_impl(ctx, a_zero, b, nb1, nb2, nb3, offset, false);
    }
    return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

static struct lm_ggml_tensor * lm_ggml_add1_or_set(
        struct lm_ggml_context  * ctx,
        struct lm_ggml_tensor   * a,
        struct lm_ggml_tensor   * b,
        struct lm_ggml_hash_set * zero_table,
        struct lm_ggml_hash_set * acc_table) {
    if (lm_ggml_hash_contains(acc_table, a)) {
        struct lm_ggml_tensor * ret = lm_ggml_add1_impl(ctx, a, b, true);
        const size_t insert_result = lm_ggml_hash_insert(acc_table, ret);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (lm_ggml_hash_contains(zero_table, a)) {
        return lm_ggml_repeat(ctx, b, a);
    }
    return lm_ggml_add1_impl(ctx, a, b, false);
}

static struct lm_ggml_tensor * lm_ggml_sub_or_set(
        struct lm_ggml_context  * ctx,
        struct lm_ggml_tensor   * a,
        struct lm_ggml_tensor   * b,
        struct lm_ggml_hash_set * zero_table,
        struct lm_ggml_hash_set * acc_table) {
    if (lm_ggml_hash_contains(acc_table, a)) {
        struct lm_ggml_tensor * ret = lm_ggml_sub_impl(ctx, a, b, true);
        const size_t insert_result = lm_ggml_hash_insert(acc_table, ret);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
        LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
        return ret;
    }
    if (lm_ggml_hash_contains(zero_table, a)) {
        return lm_ggml_neg(ctx, b);
    }
    return lm_ggml_sub_impl(ctx, a, b, false);
}

static void lm_ggml_compute_backward(struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor, struct lm_ggml_hash_set * zero_table, struct lm_ggml_hash_set * acc_table) {
    struct lm_ggml_tensor * src0 = tensor->src[0];
    struct lm_ggml_tensor * src1 = tensor->src[1];
    struct lm_ggml_tensor * src2 = tensor->src[2];

    switch (tensor->op) {
        case LM_GGML_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    if (lm_ggml_are_same_shape(src0, src1)) {
                        src1->grad = lm_ggml_add_or_set(ctx, src1->grad,                       tensor->grad,        zero_table, acc_table);
                    } else {
                        src1->grad = lm_ggml_add_or_set(ctx, src1->grad, lm_ggml_repeat_back(ctx, tensor->grad, src1), zero_table, acc_table);
                    }
                }
            } break;
        case LM_GGML_OP_ADD1:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad = lm_ggml_add_or_set(ctx,
                        src1->grad,
                        lm_ggml_mean(ctx, tensor->grad), // TODO: should probably be sum instead of mean
                        zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_ACC:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                    const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                    const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                    const size_t offset  = ((int32_t *) tensor->op_params)[3];

                    struct lm_ggml_tensor * tensor_grad_view = lm_ggml_view_4d(ctx,
                        tensor->grad,
                        src1->grad->ne[0],
                        src1->grad->ne[1],
                        src1->grad->ne[2],
                        src1->grad->ne[3],
                        nb1, nb2, nb3, offset);

                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                            src1->grad,
                            lm_ggml_reshape(ctx,
                                lm_ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad = lm_ggml_sub_or_set(ctx, src1->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_mul(ctx, src1, tensor->grad),
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                                src1->grad,
                                lm_ggml_mul(ctx, src0, tensor->grad),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_div(ctx, tensor->grad, src1),
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        lm_ggml_sub_or_set(ctx,
                                src1->grad,
                                lm_ggml_mul(ctx,
                                    tensor->grad,
                                    lm_ggml_div(ctx, tensor, src1)),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_scale(ctx,
                                    lm_ggml_mul(ctx, src0, tensor->grad),
                                    2.0f),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_scale(ctx,
                                    lm_ggml_div(ctx,
                                        tensor->grad,
                                        tensor),
                                    0.5f),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_LOG:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_div(ctx,
                                    tensor->grad,
                                    src0),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SIN:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_mul(ctx,
                                    tensor->grad,
                                    lm_ggml_cos(ctx, src0)),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_COS:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_sub_or_set(ctx,
                                src0->grad,
                                lm_ggml_mul(ctx,
                                    tensor->grad,
                                    lm_ggml_sin(ctx, src0)),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add1_or_set(ctx,
                                src0->grad,
                                tensor->grad,
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SUM_ROWS:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_repeat(ctx,
                                    tensor->grad,
                                    src0->grad),
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_MEAN:
        case LM_GGML_OP_ARGMAX:
        case LM_GGML_OP_COUNT_EQUAL:
            {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        case LM_GGML_OP_REPEAT:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_repeat_back(ctx, tensor->grad, src0->grad),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_REPEAT_BACK:
            {
                if (src0->grad) {
                    // TODO: test this
                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_repeat(ctx, tensor->grad, src0->grad),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_CONCAT:
            {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        case LM_GGML_OP_SILU_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_NORM:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_RMS_NORM:
            {
                // necessary for llama
                if (src0->grad) {
                    float eps;
                    memcpy(&eps, tensor->op_params, sizeof(float));

                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_rms_norm_back(ctx, src0, tensor->grad, eps),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_RMS_NORM_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_GROUP_NORM:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_MUL_MAT:
            {
                // https://cs231n.github.io/optimization-2/#staged
                // # forward pass
                // s0 = np.random.randn(5, 10)
                // s1 = np.random.randn(10, 3)
                // t = s0.dot(s1)

                // # now suppose we had the gradient on t from above in the circuit
                // dt = np.random.randn(*t.shape) # same shape as t
                // ds0 = dt.dot(s1.T) #.T gives the transpose of the matrix
                // ds1 = t.T.dot(dt)

                // tensor.shape [m,p,qq,rr]
                // src0.shape   [n,m,q1,r1]
                // src1.shape   [n,p,qq,rr]

                // necessary for llama
                if (src0->grad) {
                    struct lm_ggml_tensor * s1_tg =
                        lm_ggml_out_prod(ctx, // [n,m,qq,rr]
                            src1,          // [n,p,qq,rr]
                            tensor->grad); // [m,p,qq,rr]
                    const int64_t qq = s1_tg->ne[2];
                    const int64_t rr = s1_tg->ne[3];
                    const int64_t q1 = src0->ne[2];
                    const int64_t r1 = src0->ne[3];
                    const bool ne2_broadcasted = qq > q1;
                    const bool ne3_broadcasted = rr > r1;
                    if (ne2_broadcasted || ne3_broadcasted) {
                        // sum broadcast repetitions of s1_tg into shape of src0
                        s1_tg = lm_ggml_repeat_back(ctx, s1_tg, src0);
                    }
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad, // [n,m,q1,r1]
                                s1_tg,      // [n,m,q1,r1]
                                zero_table, acc_table);
                }
                if (src1->grad) {
                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                                src1->grad,                            // [n,p,qq,rr]
                                // lm_ggml_mul_mat(ctx,                   // [n,p,qq,rr]
                                //     lm_ggml_cont(ctx,                  // [m,n,q1,r1]
                                //         lm_ggml_transpose(ctx, src0)), // [m,n,q1,r1]
                                //     tensor->grad),                  // [m,p,qq,rr]

                                // // when src0 is bigger than tensor->grad (this is mostly the case in llama),
                                // // avoid transpose of src0, rather transpose smaller tensor->grad
                                // // and then use lm_ggml_out_prod
                                lm_ggml_out_prod(ctx,                  // [n,p,qq,rr]
                                    src0,                           // [n,m,q1,r1]
                                    lm_ggml_transpose(ctx,             // [p,m,qq,rr]
                                        tensor->grad)),             // [m,p,qq,rr]
                                zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_MUL_MAT_ID:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_OUT_PROD:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_SCALE:
            {
                // necessary for llama
                if (src0->grad) {
                    float s;
                    memcpy(&s, tensor->op_params, sizeof(float));

                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_scale_impl(ctx, tensor->grad, s, false),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SET:
            {
                const size_t nb1     = ((int32_t *) tensor->op_params)[0];
                const size_t nb2     = ((int32_t *) tensor->op_params)[1];
                const size_t nb3     = ((int32_t *) tensor->op_params)[2];
                const size_t offset  = ((int32_t *) tensor->op_params)[3];

                struct lm_ggml_tensor * tensor_grad_view = NULL;

                if (src0->grad || src1->grad) {
                    LM_GGML_ASSERT(src0->type == tensor->type);
                    LM_GGML_ASSERT(tensor->grad->type == tensor->type);
                    LM_GGML_ASSERT(!src1->grad || src1->grad->type == tensor->grad->type);

                    tensor_grad_view = lm_ggml_view_4d(ctx,
                        tensor->grad, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                        nb1, nb2, nb3, offset);
                }

                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx,
                        src0->grad,
                        lm_ggml_acc_impl(ctx,
                            tensor->grad,
                            lm_ggml_neg(ctx, tensor_grad_view),
                            nb1, nb2, nb3, offset, false),
                        zero_table, acc_table);
                }

                if (src1->grad) {
                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                            src1->grad,
                            lm_ggml_reshape(ctx,
                                lm_ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_CPY:
            {
                // necessary for llama
                // cpy overwrites value of src1 by src0 and returns view(src1)
                // the overwriting is mathematically equivalent to:
                // tensor = src0 * 1 + src1 * 0
                if (src0->grad) {
                    // dsrc0 = dtensor * 1
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
                if (src1->grad) {
                    // dsrc1 = dtensor * 0 -> noop
                }
            } break;
        case LM_GGML_OP_CONT:
            {
                // same as cpy
                if (src0->grad) {
                    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0->grad));
                    LM_GGML_ASSERT(lm_ggml_is_contiguous(tensor->grad));
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_RESHAPE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_reshape(ctx,
                                lm_ggml_is_contiguous(tensor->grad)
                                    ? tensor->grad
                                    : lm_ggml_cont(ctx, tensor->grad),
                                src0->grad),
                        zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_VIEW:
            {
                // necessary for llama
                if (src0->grad) {
                    size_t offset;

                    memcpy(&offset, tensor->op_params, sizeof(offset));

                    size_t nb1 = tensor->nb[1];
                    size_t nb2 = tensor->nb[2];
                    size_t nb3 = tensor->nb[3];

                    if (src0->type != src0->grad->type) {
                        // gradient is typically F32, but src0 could be other type
                        size_t ng = lm_ggml_element_size(src0->grad);
                        size_t n0 = lm_ggml_element_size(src0);
                        LM_GGML_ASSERT(offset % n0 == 0);
                        LM_GGML_ASSERT(nb1 % n0 == 0);
                        LM_GGML_ASSERT(nb2 % n0 == 0);
                        LM_GGML_ASSERT(nb3 % n0 == 0);
                        offset = (offset / n0) * ng;
                        nb1 = (nb1 / n0) * ng;
                        nb2 = (nb2 / n0) * ng;
                        nb3 = (nb3 / n0) * ng;
                    }

                    src0->grad = lm_ggml_acc_or_set(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_PERMUTE:
            {
                // necessary for llama
                if (src0->grad) {
                    int32_t * axes = (int32_t *) tensor->op_params;
                    int axis0 = axes[0] & 0x3;
                    int axis1 = axes[1] & 0x3;
                    int axis2 = axes[2] & 0x3;
                    int axis3 = axes[3] & 0x3;
                    int axes_backward[4] = {0,0,0,0};
                    axes_backward[axis0] = 0;
                    axes_backward[axis1] = 1;
                    axes_backward[axis2] = 2;
                    axes_backward[axis3] = 3;
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_permute(ctx,
                                tensor->grad,
                                axes_backward[0],
                                axes_backward[1],
                                axes_backward[2],
                                axes_backward[3]),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_TRANSPOSE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_transpose(ctx, tensor->grad),
                        zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                // necessary for llama (only for tokenizer)
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            // last lm_ggml_get_rows_back argument src0->grad is only
                            // necessary to setup correct output shape
                            lm_ggml_get_rows_back(ctx, tensor->grad, src1, src0->grad),
                        zero_table, acc_table);
                }
                if (src1->grad) {
                    // noop
                }
            } break;
        case LM_GGML_OP_GET_ROWS_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_DIAG:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_DIAG_MASK_INF:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            /* lm_ggml_diag_mask_inf_impl() shouldn't be here */
                            /* ref:  https://github.com/ggerganov/llama.cpp/pull/4203#discussion_r1412377992 */
                            lm_ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_DIAG_MASK_ZERO:
            {
                // necessary for llama
                if (src0->grad) {
                    const int n_past = ((int32_t *) tensor->op_params)[0];
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_diag_mask_zero_impl(ctx, tensor->grad, n_past, false),
                        zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_soft_max_back(ctx, tensor->grad, tensor),
                        zero_table, acc_table);
                }
                LM_GGML_ASSERT((!src1 || !src1->grad) && "backward pass for softmax mask not implemented");
            } break;
        case LM_GGML_OP_SOFT_MAX_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_ROPE:
            {
                // necessary for llama
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_rope_back(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow),
                            zero_table, acc_table);
                }
                LM_GGML_ASSERT((!src2 || !src2->grad) && "gradients for freq factors not implemented");
            } break;
        case LM_GGML_OP_ROPE_BACK:
            {
                if (src0->grad) {
                    //const int n_past = ((int32_t *) tensor->op_params)[0];
                    const int n_dims     = ((int32_t *) tensor->op_params)[1];
                    const int mode       = ((int32_t *) tensor->op_params)[2];
                    //const int n_ctx      = ((int32_t *) tensor->op_params)[3];
                    const int n_ctx_orig = ((int32_t *) tensor->op_params)[4];
                    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

                    memcpy(&freq_base,   (int32_t *) tensor->op_params +  5, sizeof(float));
                    memcpy(&freq_scale,  (int32_t *) tensor->op_params +  6, sizeof(float));
                    memcpy(&ext_factor,  (int32_t *) tensor->op_params +  7, sizeof(float));
                    memcpy(&attn_factor, (int32_t *) tensor->op_params +  8, sizeof(float));
                    memcpy(&beta_fast,   (int32_t *) tensor->op_params +  9, sizeof(float));
                    memcpy(&beta_slow,   (int32_t *) tensor->op_params + 10, sizeof(float));

                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_rope_impl(ctx,
                                tensor->grad,
                                src1,
                                src2,
                                n_dims,
                                mode,
                                n_ctx_orig,
                                freq_base,
                                freq_scale,
                                ext_factor,
                                attn_factor,
                                beta_fast,
                                beta_slow,
                                false),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_CLAMP:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_IM2COL:
            {
                if (src1->grad) {
                    const int32_t s0    = lm_ggml_get_op_params_i32(tensor, 0);
                    const int32_t s1    = lm_ggml_get_op_params_i32(tensor, 1);
                    const int32_t p0    = lm_ggml_get_op_params_i32(tensor, 2);
                    const int32_t p1    = lm_ggml_get_op_params_i32(tensor, 3);
                    const int32_t d0    = lm_ggml_get_op_params_i32(tensor, 4);
                    const int32_t d1    = lm_ggml_get_op_params_i32(tensor, 5);
                    const bool    is_2D = lm_ggml_get_op_params_i32(tensor, 6) == 1;

                    src1->grad = lm_ggml_add_or_set(ctx,
                            src1->grad,
                            lm_ggml_im2col_back(ctx, src0, tensor->grad, src1->ne, s0, s1, p0, p1, d0, d1, is_2D),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_IM2COL_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_CONV_TRANSPOSE_2D:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_POOL_1D:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_POOL_2D:
            {
                if (src0->grad) {
                    const enum lm_ggml_op_pool op = lm_ggml_get_op_params_i32(tensor, 0);
                    const      int32_t      k0 = lm_ggml_get_op_params_i32(tensor, 1);
                    const      int32_t      k1 = lm_ggml_get_op_params_i32(tensor, 2);
                    const      int32_t      s0 = lm_ggml_get_op_params_i32(tensor, 3);
                    const      int32_t      s1 = lm_ggml_get_op_params_i32(tensor, 4);
                    const      int32_t      p0 = lm_ggml_get_op_params_i32(tensor, 5);
                    const      int32_t      p1 = lm_ggml_get_op_params_i32(tensor, 6);

                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_pool_2d_back(ctx, tensor->grad, src0, op, k0, k1, s0, s1, p0, p1),
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_POOL_2D_BACK:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_UPSCALE:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_PAD:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_ARANGE:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_ARGSORT:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_LEAKY_RELU:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_FLASH_ATTN_EXT:
            {
                LM_GGML_ABORT("FA backward pass not adapted after rework");
                struct lm_ggml_tensor * flash_grad = NULL;
                if (src0->grad || src1->grad || tensor->src[2]->grad) {
                    int32_t t = lm_ggml_get_op_params_i32(tensor, 0);
                    LM_GGML_ASSERT(t == 0 || t == 1);
                    bool masked = t != 0;
                    flash_grad =
                        lm_ggml_flash_attn_back(ctx,
                            src0,
                            src1,
                            tensor->src[2],
                            tensor->grad,
                            masked);
                }

                const int64_t elem_q = lm_ggml_nelements(src0);
                const int64_t elem_k = lm_ggml_nelements(src1);
                const int64_t elem_v = lm_ggml_nelements(src2);

                enum lm_ggml_type result_type = flash_grad->type;
                LM_GGML_ASSERT(lm_ggml_blck_size(result_type) == 1);
                const size_t tsize = lm_ggml_type_size(result_type);

                const size_t offs_q = 0;
                const size_t offs_k = offs_q + LM_GGML_PAD(elem_q * tsize, LM_GGML_MEM_ALIGN);
                const size_t offs_v = offs_k + LM_GGML_PAD(elem_k * tsize, LM_GGML_MEM_ALIGN);

                if (src0->grad) {
                    struct lm_ggml_tensor * view_q = lm_ggml_view_1d(ctx, flash_grad, elem_q, offs_q);
                    struct lm_ggml_tensor * grad_q = lm_ggml_reshape(ctx, view_q, src0);
                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            grad_q,
                            zero_table, acc_table);
                }
                if (src1->grad) {
                    struct lm_ggml_tensor * view_k = lm_ggml_view_1d(ctx, flash_grad, elem_k, offs_k);
                    struct lm_ggml_tensor * grad_k = lm_ggml_reshape(ctx, view_k, src1);
                    src1->grad = lm_ggml_add_or_set(ctx,
                            src1->grad,
                            grad_k,
                            zero_table, acc_table);
                }
                if (src2->grad) {
                    struct lm_ggml_tensor * view_v = lm_ggml_view_1d(ctx, flash_grad, elem_v, offs_v);
                    struct lm_ggml_tensor * grad_v = lm_ggml_reshape(ctx, view_v, src2);
                    src2->grad = lm_ggml_add_or_set(ctx,
                            src2->grad,
                            grad_v,
                            zero_table, acc_table);
                }
            } break;
        case LM_GGML_OP_FLASH_ATTN_BACK:
            {
                LM_GGML_ABORT("fatal error"); // not supported
            }
        case LM_GGML_OP_SSM_CONV:
        case LM_GGML_OP_SSM_SCAN:
            {
                LM_GGML_ABORT("fatal error"); // TODO: not implemented
            }
        case LM_GGML_OP_WIN_PART:
        case LM_GGML_OP_WIN_UNPART:
        case LM_GGML_OP_UNARY:
            {
                switch (lm_ggml_get_unary_op(tensor)) {
                    case LM_GGML_UNARY_OP_ABS:
                        {
                            if (src0->grad) {
                                src0->grad =
                                    lm_ggml_add_or_set(ctx,
                                            src0->grad,
                                            lm_ggml_mul(ctx,
                                                lm_ggml_sgn(ctx, src0),
                                                tensor->grad),
                                            zero_table, acc_table);
                            }
                        } break;
                    case LM_GGML_UNARY_OP_SGN:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case LM_GGML_UNARY_OP_NEG:
                        {
                            if (src0->grad) {
                                src0->grad = lm_ggml_sub_or_set(ctx, src0->grad, tensor->grad, zero_table, acc_table);
                            }
                        } break;
                    case LM_GGML_UNARY_OP_STEP:
                        {
                            if (src0->grad) {
                                // noop
                            }
                        } break;
                    case LM_GGML_UNARY_OP_TANH:
                        {
                            LM_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case LM_GGML_UNARY_OP_ELU:
                        {
                            LM_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case LM_GGML_UNARY_OP_RELU:
                        {
                            if (src0->grad) {
                                src0->grad = lm_ggml_add_or_set(ctx,
                                        src0->grad,
                                        lm_ggml_mul(ctx,
                                            lm_ggml_step(ctx, src0),
                                            tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    case LM_GGML_UNARY_OP_SIGMOID:
                        {
                            LM_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case LM_GGML_UNARY_OP_GELU:
                        {
                            LM_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case LM_GGML_UNARY_OP_GELU_QUICK:
                        {
                            LM_GGML_ABORT("fatal error"); // TODO: not implemented
                        }
                    case LM_GGML_UNARY_OP_SILU:
                        {
                            // necessary for llama
                            if (src0->grad) {
                                src0->grad = lm_ggml_add_or_set(ctx,
                                        src0->grad,
                                        lm_ggml_silu_back(ctx, src0, tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    case LM_GGML_UNARY_OP_EXP:
                        {
                            if (src0->grad) {
                                src0->grad = lm_ggml_add_or_set(ctx,
                                        src0->grad,
                                        lm_ggml_mul(ctx, tensor, tensor->grad),
                                        zero_table, acc_table);
                            }
                        } break;
                    default:
                        LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_OP_GET_REL_POS:
        case LM_GGML_OP_ADD_REL_POS:
        case LM_GGML_OP_RWKV_WKV6:
        case LM_GGML_OP_MAP_UNARY:
        case LM_GGML_OP_MAP_BINARY:
        case LM_GGML_OP_MAP_CUSTOM1_F32:
        case LM_GGML_OP_MAP_CUSTOM2_F32:
        case LM_GGML_OP_MAP_CUSTOM3_F32:
        case LM_GGML_OP_MAP_CUSTOM1:
        case LM_GGML_OP_MAP_CUSTOM2:
        case LM_GGML_OP_MAP_CUSTOM3:
            {
                LM_GGML_ABORT("fatal error"); // not supported
            }
        case LM_GGML_OP_CROSS_ENTROPY_LOSS:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_cross_entropy_loss_back(ctx,
                                    src0,
                                    src1,
                                    tensor->grad),
                                zero_table, acc_table);
                }
                LM_GGML_ASSERT(!src1->grad && "backward pass for labels not implemented");
            } break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                LM_GGML_ABORT("fatal error"); // not supported
            }
        case LM_GGML_OP_OPT_STEP_ADAMW:
            {
                LM_GGML_ABORT("fatal error"); // not supported
            }
        case LM_GGML_OP_NONE:
            {
                // nop
            } break;
        case LM_GGML_OP_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    for (int i = 0; i < LM_GGML_MAX_SRC; ++i) {
        if (tensor->src[i] && tensor->src[i]->grad) {
            LM_GGML_ASSERT(lm_ggml_are_same_shape(tensor->src[i], tensor->src[i]->grad));
        }
    }
}

static void lm_ggml_visit_parents(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != LM_GGML_OP_NONE) {
            //LM_GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    if (lm_ggml_hash_insert(&cgraph->visited_hash_set, node) == LM_GGML_HASHSET_ALREADY_EXISTS) {
        return;
    }

    for (int i = 0; i < LM_GGML_MAX_SRC; ++i) {
        const int k =
            (cgraph->order == LM_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (cgraph->order == LM_GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (LM_GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i*/ i;
        if (node->src[k]) {
            lm_ggml_visit_parents(cgraph, node->src[k]);
        }
    }

    if (node->op == LM_GGML_OP_NONE && !(node->flags & LM_GGML_TENSOR_FLAG_PARAM)) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        LM_GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        if (strlen(node->name) == 0) {
            lm_ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
        }

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        LM_GGML_ASSERT(cgraph->n_nodes < cgraph->size);

        if (strlen(node->name) == 0) {
            lm_ggml_format_name(node, "node_%d", cgraph->n_nodes);
        }

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->n_nodes++;
    }
}

static void lm_ggml_build_forward_impl(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor, bool expand) {
    if (!expand) {
        // TODO: this branch isn't accessible anymore, maybe move this to lm_ggml_build_forward_expand
        lm_ggml_graph_clear(cgraph);
    }

    const int n0 = cgraph->n_nodes;

    lm_ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    LM_GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        LM_GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void lm_ggml_build_forward_expand(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor) {
    lm_ggml_build_forward_impl(cgraph, tensor, true);
}

void lm_ggml_build_backward_expand(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * gf, struct lm_ggml_cgraph * gb, bool accumulate) {
    LM_GGML_ASSERT(gf->n_nodes > 0);
    LM_GGML_ASSERT(gf->grads);

    for (int i = 0; i < gf->n_nodes; ++i) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        if (node->type == LM_GGML_TYPE_I32) {
            continue;
        }

        bool needs_grad = node->flags & LM_GGML_TENSOR_FLAG_PARAM;
        bool ignore_src[LM_GGML_MAX_SRC] = {false};
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            case LM_GGML_OP_IM2COL:      // only used for its shape
            case LM_GGML_OP_IM2COL_BACK: // same as IM2COL
                ignore_src[0] = true;
                break;
            case LM_GGML_OP_UNARY: {
                const enum lm_ggml_unary_op uop = lm_ggml_get_unary_op(node);
                // SGN and STEP unary ops are piecewise constant
                if (uop == LM_GGML_UNARY_OP_SGN || uop == LM_GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } break;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            case LM_GGML_OP_CPY:           // gradients in CPY target  are irrelevant
            case LM_GGML_OP_GET_ROWS:      // row indices not differentiable
            case LM_GGML_OP_GET_ROWS_BACK: // same as for GET_ROWS
            case LM_GGML_OP_ROPE:          // positions not differentiable
                ignore_src[1] = true;
                break;

            default:
                break;
        }
        for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->grad || ignore_src[j]) {
                continue;
            }
            LM_GGML_ASSERT(node->src[j]->type == LM_GGML_TYPE_F32 || node->src[j]->type == LM_GGML_TYPE_F16);
            needs_grad = true;
            break;
        }
        if (!needs_grad) {
            continue;
        }

        // inplace operations are currently not supported
        LM_GGML_ASSERT(!node->view_src || node->op == LM_GGML_OP_CPY || node->op == LM_GGML_OP_VIEW ||
            node->op == LM_GGML_OP_RESHAPE || node->op == LM_GGML_OP_PERMUTE || node->op == LM_GGML_OP_TRANSPOSE);

        // create a new tensor with the same type and shape as the node and set it as grad
        node->grad = lm_ggml_dup_tensor(ctx, node);
    }

    // keep tables of original gradients for replacement/accumulation logic
    struct lm_ggml_hash_set zero_table = lm_ggml_hash_set_new(gf->size);
    struct lm_ggml_hash_set acc_table  = lm_ggml_hash_set_new(gf->size);
    for (int i = 0; i < gf->n_nodes; i++) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        if (node->grad) {
            {
                const size_t insert_result = lm_ggml_hash_insert(&zero_table, node->grad);
                LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
                LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
            }

            // only gradients of trainable parameters should be accumulated
            if (accumulate && (node->flags & LM_GGML_TENSOR_FLAG_PARAM)) {
                const size_t insert_result = lm_ggml_hash_insert(&acc_table, node->grad);
                LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_FULL);
                LM_GGML_ASSERT(insert_result != LM_GGML_HASHSET_ALREADY_EXISTS);
            }
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by lm_ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        if (node->grad) {
            lm_ggml_compute_backward(ctx, node, &zero_table, &acc_table);
        }
    }

    for (int i = 0; i < gf->n_nodes; i++) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        if (node->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            LM_GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            lm_ggml_build_forward_expand(gb, node->grad);
        }
    }

    lm_ggml_hash_set_free(&zero_table);
    lm_ggml_hash_set_free(&acc_table);
}

void lm_ggml_build_opt_adamw(
        struct lm_ggml_context * ctx,
        struct lm_ggml_cgraph  * gf,
        struct lm_ggml_cgraph  * gb,
        float                 alpha,
        float                 beta1,
        float                 beta2,
        float                 eps,
        float                 wd) {
    for (int i = 0; i < gf->n_nodes; i++) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        if (node->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            LM_GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            struct lm_ggml_tensor * opt_step = lm_ggml_opt_step_adamw(ctx, node, node->grad, alpha, beta1, beta2, eps, wd);
            lm_ggml_build_forward_expand(gb, opt_step);
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {
    void * ptr = *p;
    ptr = (void *) LM_GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static size_t lm_ggml_graph_nbytes(size_t size, bool grads) {
    size_t hash_size = lm_ggml_hash_size(size * 2);
    void * p = 0;
    incr_ptr_aligned(&p, sizeof(struct lm_ggml_cgraph), 1);
    incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *)); // nodes
    incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *)); // leafs
    incr_ptr_aligned(&p, hash_size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *)); // hash keys
    if (grads) {
        incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *)); // grads
    }
    incr_ptr_aligned(&p, lm_ggml_bitset_size(hash_size) * sizeof(lm_ggml_bitset_t), sizeof(lm_ggml_bitset_t));

    size_t nbytes = (size_t) p;
    return nbytes;
}

size_t lm_ggml_graph_overhead_custom(size_t size, bool grads) {
    return LM_GGML_OBJECT_SIZE + LM_GGML_PAD(lm_ggml_graph_nbytes(size, grads), LM_GGML_MEM_ALIGN);
}

size_t lm_ggml_graph_overhead(void) {
    return lm_ggml_graph_overhead_custom(LM_GGML_DEFAULT_GRAPH_SIZE, false);
}

struct lm_ggml_cgraph * lm_ggml_new_graph_custom(struct lm_ggml_context * ctx, size_t size, bool grads) {
    const size_t obj_size = lm_ggml_graph_nbytes(size, grads);
    struct lm_ggml_object * obj = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_GRAPH, obj_size);
    struct lm_ggml_cgraph * cgraph = (struct lm_ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);

    // the size of the hash table is doubled since it needs to hold both nodes and leafs
    size_t hash_size = lm_ggml_hash_size(size * 2);

    void * p = cgraph + 1;

    struct lm_ggml_tensor ** nodes_ptr = incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *));
    struct lm_ggml_tensor ** leafs_ptr = incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *));
    struct lm_ggml_tensor ** hash_keys_ptr = incr_ptr_aligned(&p, hash_size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *));
    struct lm_ggml_tensor ** grads_ptr = grads ? incr_ptr_aligned(&p, size * sizeof(struct lm_ggml_tensor *), sizeof(struct lm_ggml_tensor *)) : NULL;
    lm_ggml_bitset_t * hash_used = incr_ptr_aligned(&p, lm_ggml_bitset_size(hash_size) * sizeof(lm_ggml_bitset_t), sizeof(lm_ggml_bitset_t));

    // check that we allocated the correct amount of memory
    assert(obj_size == (size_t)((char *)p - (char *)cgraph));

    *cgraph = (struct lm_ggml_cgraph) {
        /*.size         =*/ size,
        /*.n_nodes      =*/ 0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ nodes_ptr,
        /*.grads        =*/ grads_ptr,
        /*.leafs        =*/ leafs_ptr,
        /*.hash_table   =*/ { hash_size, hash_used, hash_keys_ptr },
        /*.order        =*/ LM_GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
    };

    lm_ggml_hash_set_reset(&cgraph->visited_hash_set);

    return cgraph;
}

struct lm_ggml_cgraph * lm_ggml_new_graph(struct lm_ggml_context * ctx) {
    return lm_ggml_new_graph_custom(ctx, LM_GGML_DEFAULT_GRAPH_SIZE, false);
}

struct lm_ggml_cgraph lm_ggml_graph_view(struct lm_ggml_cgraph * cgraph0, int i0, int i1) {
    struct lm_ggml_cgraph cgraph = {
        /*.size         =*/ 0,
        /*.n_nodes      =*/ i1 - i0,
        /*.n_leafs      =*/ 0,
        /*.nodes        =*/ cgraph0->nodes + i0,
        /*.grads        =*/ cgraph0->grads ? cgraph0->grads + i0 : NULL,
        /*.leafs        =*/ NULL,
        /*.hash_table   =*/ { 0, NULL, NULL },
        /*.order        =*/ cgraph0->order,
    };

    return cgraph;
}

void lm_ggml_graph_cpy(struct lm_ggml_cgraph * src, struct lm_ggml_cgraph * dst) {
    LM_GGML_ASSERT(dst->size >= src->n_leafs);
    LM_GGML_ASSERT(dst->size >= src->n_nodes);
    LM_GGML_ASSERT(dst->visited_hash_set.size >= src->visited_hash_set.size);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;
    dst->order   = src->order;

    for (int i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }

    for (int i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    if (src->grads) {
        LM_GGML_ASSERT(dst->grads != NULL);
        for (int i = 0; i < src->n_nodes; ++i) {
            dst->grads[i] = src->grads[i];
        }
    }

    for (size_t i = 0; i < src->visited_hash_set.size; ++i) {
        // copy all hashset keys (tensors) that are in use
        if (lm_ggml_bitset_get(src->visited_hash_set.used, i)) {
            lm_ggml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
        }
    }
}

struct lm_ggml_cgraph * lm_ggml_graph_dup(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_cgraph * result = lm_ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads != NULL);
    lm_ggml_graph_cpy(cgraph, result);
    return result;
}

struct lm_ggml_tensor * lm_ggml_set_zero(struct lm_ggml_tensor * tensor) {
    if (lm_ggml_is_empty(tensor)) {
        return tensor;
    }
    if (tensor->buffer) {
        lm_ggml_backend_tensor_memset(tensor, 0, 0, lm_ggml_nbytes(tensor));
    } else {
        LM_GGML_ASSERT(tensor->data);
        memset(tensor->data, 0, lm_ggml_nbytes(tensor));
    }
    return tensor;
}

void lm_ggml_graph_reset(struct lm_ggml_cgraph * cgraph) {
    LM_GGML_ASSERT(cgraph->grads != NULL);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        // initial gradients of loss should be 1, 0 otherwise
        if (node->grad) {
            if (node->flags & LM_GGML_TENSOR_FLAG_LOSS) {
                LM_GGML_ASSERT(node->grad->buffer);
                LM_GGML_ASSERT(node->type == LM_GGML_TYPE_F32);
                LM_GGML_ASSERT(lm_ggml_is_scalar(node));

                const float onef = 1.0f;
                lm_ggml_backend_tensor_set(node->grad, &onef, 0, lm_ggml_nbytes(node->grad));
            } else {
                lm_ggml_set_zero(node->grad);
            }
        }

        LM_GGML_ASSERT(node);
        if (node->op == LM_GGML_OP_OPT_STEP_ADAMW) {
            // set iteration to 1 and clear momenta
            lm_ggml_set_op_params_i32(node, 0, 1);
            lm_ggml_set_zero(node->src[2]);
            lm_ggml_set_zero(node->src[3]);
        }
    }
}

void lm_ggml_graph_clear(struct lm_ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    lm_ggml_hash_set_reset(&cgraph->visited_hash_set);
}

int lm_ggml_graph_size(struct lm_ggml_cgraph * cgraph) {
    return cgraph->size;
}

struct lm_ggml_tensor * lm_ggml_graph_node(struct lm_ggml_cgraph * cgraph, int i) {
    if (i < 0) {
        LM_GGML_ASSERT(cgraph->n_nodes + i >= 0);
        return cgraph->nodes[cgraph->n_nodes + i];
    }

    LM_GGML_ASSERT(i < cgraph->n_nodes);
    return cgraph->nodes[i];
}

struct lm_ggml_tensor ** lm_ggml_graph_nodes(struct lm_ggml_cgraph * cgraph) {
    return cgraph->nodes;
}

int lm_ggml_graph_n_nodes(struct lm_ggml_cgraph * cgraph) {
    return cgraph->n_nodes;
}

void lm_ggml_graph_add_node(struct lm_ggml_cgraph * cgraph, struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(cgraph->size > cgraph->n_nodes);
    cgraph->nodes[cgraph->n_nodes] = tensor;
    cgraph->n_nodes++;
}

struct lm_ggml_tensor * lm_ggml_graph_get_tensor(struct lm_ggml_cgraph * cgraph, const char * name) {
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct lm_ggml_tensor * leaf = cgraph->leafs[i];

        if (strcmp(leaf->name, name) == 0) {
            return leaf;
        }
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        if (strcmp(node->name, name) == 0) {
            return node;
        }
    }

    return NULL;
}

void lm_ggml_graph_print(const struct lm_ggml_cgraph * cgraph) {
    LM_GGML_LOG_INFO("=== GRAPH ===\n");

    LM_GGML_LOG_INFO("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        LM_GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                lm_ggml_op_name(node->op), (node->flags & LM_GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ");
    }

    LM_GGML_LOG_INFO("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct lm_ggml_tensor * node = cgraph->leafs[i];

        LM_GGML_LOG_INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                node->ne[0], node->ne[1],
                lm_ggml_op_name(node->op),
                lm_ggml_get_name(node));
    }

    LM_GGML_LOG_INFO("========================================\n");
}

// check if node is part of the graph
static bool lm_ggml_graph_find(const struct lm_ggml_cgraph * cgraph, const struct lm_ggml_tensor * node) {
    if (cgraph == NULL) {
        return true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return true;
        }
    }

    return false;
}

static struct lm_ggml_tensor * lm_ggml_graph_get_parent(const struct lm_ggml_cgraph * cgraph, const struct lm_ggml_tensor * node) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * parent = cgraph->nodes[i];

        if (parent->grad == node) {
            return parent;
        }
    }

    return NULL;
}

static void lm_ggml_graph_dump_dot_node_edge(FILE * fp, const struct lm_ggml_cgraph * gb, struct lm_ggml_tensor * node, struct lm_ggml_tensor * parent, const char * label)  {
    struct lm_ggml_tensor * gparent = lm_ggml_graph_get_parent(gb, node);
    struct lm_ggml_tensor * gparent0 = lm_ggml_graph_get_parent(gb, parent);
    fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"%s\"; ]\n",
            gparent0 ? (void *) gparent0 : (void *) parent,
            gparent0 ? "g" : "x",
            gparent ? (void *) gparent : (void *) node,
            gparent ? "g" : "x",
            gparent ? "empty" : "vee",
            gparent ? "dashed" : "solid",
            label);
}

static void lm_ggml_graph_dump_dot_leaf_edge(FILE * fp, struct lm_ggml_tensor * node, struct lm_ggml_tensor * parent, const char * label)  {
    fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"%s\"; ]\n",
            (void *) parent, "x",
            (void *) node, "x",
            label);
}

void lm_ggml_graph_dump_dot(const struct lm_ggml_cgraph * gb, const struct lm_ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = lm_ggml_fopen(filename, "w");
    LM_GGML_ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = TB;\n");

    for (int i = 0; i < gb->n_nodes; i++) {
        struct lm_ggml_tensor * node = gb->nodes[i];

        if (lm_ggml_graph_get_parent(gb, node) != NULL) {
            continue;
        }

        if (node->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            snprintf(color, sizeof(color), "yellow");
        } else if (node->grad) {
            if (lm_ggml_graph_find(gf, node)) {
                snprintf(color, sizeof(color), "green");
            } else {
                snprintf(color, sizeof(color), "lightblue");
            }
        } else {
            snprintf(color, sizeof(color), "white");
        }

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"",
                (void *) node, color);

        if (strlen(node->name) > 0) {
            fprintf(fp, "%s (%s)|", node->name, lm_ggml_type_name(node->type));
        } else {
            fprintf(fp, "(%s)|", lm_ggml_type_name(node->type));
        }

        if (lm_ggml_is_matrix(node)) {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], lm_ggml_op_symbol(node->op));
        } else {
            fprintf(fp, "%d [%" PRId64 ", %" PRId64 ", %" PRId64 "] | <x>%s", i, node->ne[0], node->ne[1], node->ne[2], lm_ggml_op_symbol(node->op));
        }

        if (node->grad) {
            fprintf(fp, " | <g>%s\"; ]\n", lm_ggml_op_symbol(node->grad->op));
        } else {
            fprintf(fp, "\"; ]\n");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct lm_ggml_tensor * node = gb->leafs[i];

        snprintf(color, sizeof(color), "pink");

        fprintf(fp, "  \"%p\" [ "
                    "style = filled; fillcolor = %s; shape = record; "
                    "label=\"<x>",
                (void *) node, color);

        if (strlen(node->name) > 0) {
            fprintf(fp, "%s (%s)|", node->name, lm_ggml_type_name(node->type));
        } else {
            fprintf(fp, "(%s)|", lm_ggml_type_name(node->type));
        }

        fprintf(fp, "CONST %d [%" PRId64 ", %" PRId64 "]", i, node->ne[0], node->ne[1]);
        if (lm_ggml_nelements(node) < 5 && node->data != NULL) {
            fprintf(fp, " | (");
            for (int j = 0; j < lm_ggml_nelements(node); j++) {
                // FIXME: use ggml-backend to obtain the tensor data
                //if (node->type == LM_GGML_TYPE_I8 || node->type == LM_GGML_TYPE_I16 || node->type == LM_GGML_TYPE_I32) {
                //    fprintf(fp, "%d", lm_ggml_get_i32_1d(node, j));
                //}
                //else if (node->type == LM_GGML_TYPE_F32 ||
                //         node->type == LM_GGML_TYPE_F16 ||
                //         node->type == LM_GGML_TYPE_BF16) {
                //    fprintf(fp, "%.1e", (double)lm_ggml_get_f32_1d(node, j));
                //}
                //else
                {
                    fprintf(fp, "#");
                }
                if (j < lm_ggml_nelements(node) - 1) {
                    fprintf(fp, ", ");
                }
            }
            fprintf(fp, ")");
        }
        fprintf(fp, "\"; ]\n");
    }

    for (int i = 0; i < gb->n_nodes; i++) {
        struct lm_ggml_tensor * node = gb->nodes[i];

        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                char label[16];
                snprintf(label, sizeof(label), "src %d", j);
                lm_ggml_graph_dump_dot_node_edge(fp, gb, node, node->src[j], label);
            }
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct lm_ggml_tensor * node = gb->leafs[i];

        for (int j = 0; j < LM_GGML_MAX_SRC; j++) {
            if (node->src[j]) {
                char label[16];
                snprintf(label, sizeof(label), "src %d", j);
                lm_ggml_graph_dump_dot_leaf_edge(fp, node, node->src[j], label);
            }
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    LM_GGML_LOG_INFO("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_set_input(struct lm_ggml_tensor * tensor) {
    tensor->flags |= LM_GGML_TENSOR_FLAG_INPUT;
}

void lm_ggml_set_output(struct lm_ggml_tensor * tensor) {
    tensor->flags |= LM_GGML_TENSOR_FLAG_OUTPUT;
}

void lm_ggml_set_param(struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor) {
    LM_GGML_UNUSED(ctx); // TODO: remove this parameter
    tensor->flags |= LM_GGML_TENSOR_FLAG_PARAM;
}

void lm_ggml_set_loss(struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(lm_ggml_is_scalar(tensor));
    LM_GGML_ASSERT(tensor->type == LM_GGML_TYPE_F32);
    tensor->flags |= LM_GGML_TENSOR_FLAG_LOSS;
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_quantize_init(enum lm_ggml_type type) {
    lm_ggml_critical_section_start();

    switch (type) {
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:   lm_iq2xs_init_impl(type); break;
        case LM_GGML_TYPE_IQ3_XXS: lm_iq3xs_init_impl(256); break;
        case LM_GGML_TYPE_IQ3_S:   lm_iq3xs_init_impl(512); break;
        default: // nothing
            break;
    }

    lm_ggml_critical_section_end();
}

void lm_ggml_quantize_free(void) {
    lm_ggml_critical_section_start();

    lm_iq2xs_free_impl(LM_GGML_TYPE_IQ2_XXS);
    lm_iq2xs_free_impl(LM_GGML_TYPE_IQ2_XS);
    lm_iq2xs_free_impl(LM_GGML_TYPE_IQ1_S);
    lm_iq3xs_free_impl(256);

    lm_ggml_critical_section_end();
}

bool lm_ggml_quantize_requires_imatrix(enum lm_ggml_type type) {
    return
        type == LM_GGML_TYPE_IQ2_XXS ||
        type == LM_GGML_TYPE_IQ2_XS  ||
        type == LM_GGML_TYPE_IQ1_S;//   ||
        //type == LM_GGML_TYPE_IQ1_M;
}

size_t lm_ggml_quantize_chunk(
        enum lm_ggml_type   type,
           const float * src,
                  void * dst,
               int64_t   start,
               int64_t   nrows,
               int64_t   n_per_row,
           const float * imatrix) {
    const int64_t n = (int64_t) nrows * n_per_row;

    if (lm_ggml_quantize_requires_imatrix(type)) {
        LM_GGML_ASSERT(imatrix != NULL);
    }

    LM_GGML_ASSERT(start % type_traits[type].blck_size == 0);
    LM_GGML_ASSERT(start % n_per_row == 0);

    lm_ggml_quantize_init(type); // this is noop if already initialized

    const size_t start_row = start / n_per_row;
    const size_t row_size  = lm_ggml_row_size(type, n_per_row);

    size_t result = 0;

    switch (type) {
        case LM_GGML_TYPE_Q4_0:    result = quantize_q4_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q4_1:    result = quantize_q4_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q5_0:    result = quantize_q5_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q5_1:    result = quantize_q5_1(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q8_0:    result = quantize_q8_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q2_K:    result = quantize_q2_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q3_K:    result = quantize_q3_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q4_K:    result = quantize_q4_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q5_K:    result = quantize_q5_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q6_K:    result = quantize_q6_K(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_TQ1_0:   result = quantize_tq1_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_TQ2_0:   result = quantize_tq2_0(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ2_XXS: result = quantize_iq2_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ2_XS:  result = quantize_iq2_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ3_XXS: result = quantize_iq3_xxs(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ3_S:   result = quantize_iq3_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ2_S:   result = quantize_iq2_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ1_S:   result = quantize_iq1_s  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ1_M:   result = quantize_iq1_m  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ4_NL:  result = quantize_iq4_nl (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_IQ4_XS:  result = quantize_iq4_xs (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q4_0_4_4: result = quantize_q4_0_4x4(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q4_0_4_8: result = quantize_q4_0_4x8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_Q4_0_8_8: result = quantize_q4_0_8x8(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;
        case LM_GGML_TYPE_F16:
            {
                size_t elemsize = sizeof(lm_ggml_fp16_t);
                lm_ggml_fp32_to_fp16_row(src + start, (lm_ggml_fp16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case LM_GGML_TYPE_BF16:
            {
                size_t elemsize = sizeof(lm_ggml_bf16_t);
                lm_ggml_fp32_to_bf16_row_ref(src + start, (lm_ggml_bf16_t *)dst + start, n);
                result = n * elemsize;
            } break;
        case LM_GGML_TYPE_F32:
            {
                size_t elemsize = sizeof(float);
                result = n * elemsize;
                memcpy((uint8_t *)dst + start * elemsize, src + start, result);
            } break;
        default:
            assert(false);
    }

    LM_GGML_ASSERT(result == nrows * row_size);

    return result;
}

////////////////////////////////////////////////////////////////////////////////

struct lm_gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

static const size_t LM_GGUF_TYPE_SIZE[LM_GGUF_TYPE_COUNT] = {
    [LM_GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [LM_GGUF_TYPE_INT8]    = sizeof(int8_t),
    [LM_GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [LM_GGUF_TYPE_INT16]   = sizeof(int16_t),
    [LM_GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [LM_GGUF_TYPE_INT32]   = sizeof(int32_t),
    [LM_GGUF_TYPE_FLOAT32] = sizeof(float),
    [LM_GGUF_TYPE_BOOL]    = sizeof(bool),
    [LM_GGUF_TYPE_STRING]  = sizeof(struct lm_gguf_str),
    [LM_GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [LM_GGUF_TYPE_INT64]   = sizeof(int64_t),
    [LM_GGUF_TYPE_FLOAT64] = sizeof(double),
    [LM_GGUF_TYPE_ARRAY]   = 0, // undefined
};
static_assert(LM_GGUF_TYPE_COUNT == 13, "LM_GGUF_TYPE_COUNT != 13");

static const char * LM_GGUF_TYPE_NAME[LM_GGUF_TYPE_COUNT] = {
    [LM_GGUF_TYPE_UINT8]   = "u8",
    [LM_GGUF_TYPE_INT8]    = "i8",
    [LM_GGUF_TYPE_UINT16]  = "u16",
    [LM_GGUF_TYPE_INT16]   = "i16",
    [LM_GGUF_TYPE_UINT32]  = "u32",
    [LM_GGUF_TYPE_INT32]   = "i32",
    [LM_GGUF_TYPE_FLOAT32] = "f32",
    [LM_GGUF_TYPE_BOOL]    = "bool",
    [LM_GGUF_TYPE_STRING]  = "str",
    [LM_GGUF_TYPE_ARRAY]   = "arr",
    [LM_GGUF_TYPE_UINT64]  = "u64",
    [LM_GGUF_TYPE_INT64]   = "i64",
    [LM_GGUF_TYPE_FLOAT64] = "f64",
};
static_assert(LM_GGUF_TYPE_COUNT == 13, "LM_GGUF_TYPE_COUNT != 13");

union lm_gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct lm_gguf_str str;

    struct {
        enum lm_gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};

struct lm_gguf_kv {
    struct lm_gguf_str key;

    enum  lm_gguf_type  type;
    union lm_gguf_value value;
};

struct lm_gguf_header {
    char magic[4];

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct lm_gguf_tensor_info {
    struct lm_gguf_str name;

    uint32_t n_dims;
    uint64_t ne[LM_GGML_MAX_DIMS];

    enum lm_ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};

struct lm_gguf_context {
    struct lm_gguf_header header;

    struct lm_gguf_kv          * kv;
    struct lm_gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};

static size_t lm_gguf_type_size(enum lm_gguf_type type) {
    LM_GGML_ASSERT(0 <= type && type < LM_GGUF_TYPE_COUNT);
    return LM_GGUF_TYPE_SIZE[type];
}

static bool lm_gguf_tensor_info_sanitize(struct lm_gguf_tensor_info * info) {
    if (info->n_dims > LM_GGML_MAX_DIMS) {
        fprintf(stderr, "%s: invalid number of dimensions (%" PRIu32 ")\n", __func__, info->n_dims);
        return false;
    }

    if (info->type < 0 || info->type >= LM_GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid type (%d)\n", __func__, info->type);
        return false;
    }

    if (strlen(info->name.data) >= LM_GGML_MAX_NAME) {
        fprintf(stderr, "%s: tensor '%s' name is too long\n", __func__, info->name.data);
        return false;
    }

    for (uint32_t i = 0; i < info->n_dims; ++i) {
        if (info->ne[i] <= 0) {
            fprintf(stderr, "%s: invalid number of elements (%" PRIu64 ")\n", __func__, info->ne[i]);
            return false;
        }
    }

    // prevent overflow for total number of elements
    if (INT64_MAX/info->ne[1] <= info->ne[0]) {
        fprintf(stderr, "%s: invalid number of elements (%" PRIu64 ")\n", __func__, info->ne[1]);
        return false;
    }

    if (INT64_MAX/info->ne[2] <= info->ne[0]*info->ne[1]) {
        fprintf(stderr, "%s: invalid number of elements (%" PRIu64 ")\n", __func__, info->ne[2]);
        return false;
    }

    if (INT64_MAX/info->ne[3] <= info->ne[0]*info->ne[1]*info->ne[2]) {
        fprintf(stderr, "%s: invalid number of elements (%" PRIu64 ")\n", __func__, info->ne[3]);
        return false;
    }

    return true;
}

static bool lm_gguf_fread_el(FILE * file, void * dst, size_t size, size_t * offset) {
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool lm_gguf_fread_str(FILE * file, struct lm_gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok = true;

    ok = ok && lm_gguf_fread_el(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\n", __func__, p->n);
        return false;
    }

    p->data = calloc(p->n + 1, 1);
    if (!p->data) {
        fprintf(stderr, "%s: failed to allocate memory for string of length %" PRIu64 "\n", __func__, p->n);
        return false;
    }

    ok = ok && lm_gguf_fread_el(file,  p->data, p->n, offset);

    return ok;
}

static void lm_gguf_free_kv(struct lm_gguf_kv * kv) {
    if (kv->key.data) {
        LM_GGML_FREE(kv->key.data);
    }

    if (kv->type == LM_GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
            LM_GGML_FREE(kv->value.str.data);
        }
    }

    if (kv->type == LM_GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
            if (kv->value.arr.type == LM_GGUF_TYPE_STRING) {
                for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                    struct lm_gguf_str * str = &((struct lm_gguf_str *) kv->value.arr.data)[j];
                    if (str->data) {
                        LM_GGML_FREE(str->data);
                    }
                }
            }
            LM_GGML_FREE(kv->value.arr.data);
        }
    }
}

struct lm_gguf_context * lm_gguf_init_empty(void) {
    struct lm_gguf_context * ctx = calloc(1, sizeof(struct lm_gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        return NULL;
    }

    memcpy(ctx->header.magic, LM_GGUF_MAGIC, sizeof(ctx->header.magic));
    ctx->header.version   = LM_GGUF_VERSION;
    ctx->header.n_tensors = 0;
    ctx->header.n_kv      = 0;

    ctx->kv    = NULL;
    ctx->infos = NULL;

    ctx->alignment = LM_GGUF_DEFAULT_ALIGNMENT;
    ctx->offset    = 0;
    ctx->size      = 0;

    ctx->data = NULL;

    return ctx;
}

struct lm_gguf_context * lm_gguf_init_from_file(const char * fname, struct lm_gguf_init_params params) {
    FILE * file = lm_ggml_fopen(fname, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    // offset from start of file
    size_t offset = 0;

    char magic[4];

    // check the magic before making allocations
    {
        lm_gguf_fread_el(file, &magic, sizeof(magic), &offset);

        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != LM_GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return NULL;
            }
        }
    }

    bool ok = true;

    struct lm_gguf_context * ctx = calloc(1, sizeof(struct lm_gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        fclose(file);
        return NULL;
    }

    // read the header
    {
        strncpy(ctx->header.magic, magic, 4);

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && lm_gguf_fread_el(file, &ctx->header.version,   sizeof(ctx->header.version),   &offset);
        ok = ok && lm_gguf_fread_el(file, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors), &offset);
        ok = ok && lm_gguf_fread_el(file, &ctx->header.n_kv,      sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }

        // sanity-checks to prevent from integer/buffer overflows

        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(struct lm_gguf_tensor_info));
        ok = ok && (ctx->header.n_tensors < (SIZE_MAX/2)/lm_ggml_tensor_overhead());
        ok = ok && (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct lm_gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }
    }

    // read the kv pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;

        ctx->kv = calloc(n_kv, sizeof(struct lm_gguf_kv));
        if (!ctx->kv) {
            fprintf(stderr, "%s: failed to allocate memory for kv pairs\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }

        for (uint64_t i = 0; i < n_kv; ++i) {
            struct lm_gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && lm_gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && lm_gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case LM_GGUF_TYPE_UINT8:   ok = ok && lm_gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case LM_GGUF_TYPE_INT8:    ok = ok && lm_gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case LM_GGUF_TYPE_UINT16:  ok = ok && lm_gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case LM_GGUF_TYPE_INT16:   ok = ok && lm_gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case LM_GGUF_TYPE_UINT32:  ok = ok && lm_gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case LM_GGUF_TYPE_INT32:   ok = ok && lm_gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case LM_GGUF_TYPE_FLOAT32: ok = ok && lm_gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case LM_GGUF_TYPE_UINT64:  ok = ok && lm_gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case LM_GGUF_TYPE_INT64:   ok = ok && lm_gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case LM_GGUF_TYPE_FLOAT64: ok = ok && lm_gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case LM_GGUF_TYPE_BOOL:    ok = ok && lm_gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case LM_GGUF_TYPE_STRING:  ok = ok && lm_gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case LM_GGUF_TYPE_ARRAY:
                    {
                        ok = ok && lm_gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                        ok = ok && lm_gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                        switch (kv->value.arr.type) {
                            case LM_GGUF_TYPE_UINT8:
                            case LM_GGUF_TYPE_INT8:
                            case LM_GGUF_TYPE_UINT16:
                            case LM_GGUF_TYPE_INT16:
                            case LM_GGUF_TYPE_UINT32:
                            case LM_GGUF_TYPE_INT32:
                            case LM_GGUF_TYPE_FLOAT32:
                            case LM_GGUF_TYPE_UINT64:
                            case LM_GGUF_TYPE_INT64:
                            case LM_GGUF_TYPE_FLOAT64:
                            case LM_GGUF_TYPE_BOOL:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/lm_gguf_type_size(kv->value.arr.type)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        lm_gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = calloc(kv->value.arr.n, lm_gguf_type_size(kv->value.arr.type));
                                    if (!kv->value.arr.data) {
                                        fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
                                        fclose(file);
                                        lm_gguf_free(ctx);
                                        return NULL;
                                    }

                                    ok = ok && lm_gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * lm_gguf_type_size(kv->value.arr.type), &offset);
                                } break;
                            case LM_GGUF_TYPE_STRING:
                                {
                                    // prevent from integer overflow in the malloc below
                                    if (kv->value.arr.n >= SIZE_MAX/sizeof(struct lm_gguf_str)) {
                                        fprintf(stderr, "%s: array size is too large (%" PRIu64 ")\n", __func__, kv->value.arr.n);
                                        fclose(file);
                                        lm_gguf_free(ctx);
                                        return NULL;
                                    }

                                    kv->value.arr.data = calloc(kv->value.arr.n, sizeof(struct lm_gguf_str));
                                    if (!kv->value.arr.data) {
                                        fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
                                        fclose(file);
                                        lm_gguf_free(ctx);
                                        return NULL;
                                    }

                                    for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && lm_gguf_fread_str(file, &((struct lm_gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case LM_GGUF_TYPE_ARRAY:
                            default:
                                {
                                    fprintf(stderr, "%s: invalid array type %d\n", __func__, kv->value.arr.type);
                                    ok = false;
                                } break;
                        }
                    } break;
                default:
                    {
                        fprintf(stderr, "%s: invalid type %d\n", __func__, kv->type);
                        ok = false;
                    } break;
            }

            if (!ok) {
                break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }
    }

    // read the tensor infos
    if (ctx->header.n_tensors > 0) {
        ctx->infos = calloc(ctx->header.n_tensors, sizeof(struct lm_gguf_tensor_info));
        if (!ctx->infos) {
            fprintf(stderr, "%s: failed to allocate memory for tensor infos\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct lm_gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && lm_gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && lm_gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= LM_GGML_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && lm_gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && lm_gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && lm_gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            ok = ok && lm_gguf_tensor_info_sanitize(info);

            // make sure there is no duplicated tensor names
            for (uint64_t j = 0; j < i && ok; ++j) {
                if (strcmp(info->name.data, ctx->infos[j].name.data) == 0) {
                    fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->name.data);
                    ok = false;
                }
            }

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                fclose(file);
                lm_gguf_free(ctx);
                return NULL;
            }
        }
    }

    ctx->alignment = LM_GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = lm_gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = lm_gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct lm_gguf_tensor_info * info = &ctx->infos[i];

            const int64_t ne =
                (int64_t) info->ne[0] *
                (int64_t) info->ne[1] *
                (int64_t) info->ne[2] *
                (int64_t) info->ne[3];

            if (lm_ggml_blck_size(info->type) == 0 || ne % lm_ggml_blck_size(info->type) != 0) {
                fprintf(stderr, "%s: tensor '%s' of type %d (%s) number of elements (%" PRId64 ") is not a multiple of block size (%" PRId64 ")\n",
                        __func__, info->name.data, (int) info->type, lm_ggml_type_name(info->type), ne, lm_ggml_blck_size(info->type));
                fclose(file);
                lm_gguf_free(ctx);
                return NULL;
            }

            const size_t size_cur = lm_ggml_row_size(info->type, ne);

            ctx->size += LM_GGML_PAD(size_cur, ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != NULL) {
        // if the provided lm_gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created lm_ggml_context as well, and point the "data" members of
        // the lm_ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new lm_ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (ctx->header.n_tensors    )*lm_ggml_tensor_overhead() :
            (ctx->header.n_tensors + 1)*lm_ggml_tensor_overhead() + ctx->size;

        struct lm_ggml_init_params pdata = {
            .mem_size   = mem_size,
            .mem_buffer = NULL,
            .no_alloc   = params.no_alloc,
        };

        *params.ctx = lm_ggml_init(pdata);
        if (*params.ctx == NULL) {
            fprintf(stderr, "%s: failed to initialize context\n", __func__);
            fclose(file);
            lm_gguf_free(ctx);
            return NULL;
        }

        struct lm_ggml_context * ctx_data = *params.ctx;

        struct lm_ggml_tensor * data = NULL;

        if (!params.no_alloc) {
            data = lm_ggml_new_tensor_1d(ctx_data, LM_GGML_TYPE_I8, ctx->size);

            ok = ok && data != NULL;

            // read the binary blob with the tensor data
            ok = ok && lm_gguf_fread_el(file, data->data, ctx->size, &offset);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                fclose(file);
                lm_ggml_free(ctx_data);
                lm_gguf_free(ctx);
                return NULL;
            }

            ctx->data = data->data;
        }

        lm_ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            const int64_t ne[LM_GGML_MAX_DIMS] = {
                ctx->infos[i].ne[0],
                ctx->infos[i].ne[1],
                ctx->infos[i].ne[2],
                ctx->infos[i].ne[3],
            };

            struct lm_ggml_tensor * cur = lm_ggml_new_tensor(ctx_data, ctx->infos[i].type, ctx->infos[i].n_dims, ne);

            ok = ok && cur != NULL;

            if (!ok) {
                break;
            }

            lm_ggml_set_name(cur, ctx->infos[i].name.data);

            // point the data member to the appropriate location in the binary blob using the tensor infos
            if (!params.no_alloc) {
              //cur->data = (char *) data->data + ctx->infos[i].offset - ctx->offset; // offset from start of file
                cur->data = (char *) data->data + ctx->infos[i].offset;               // offset from data
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read the tensor data\n", __func__);
            fclose(file);
            lm_ggml_free(ctx_data);
            lm_gguf_free(ctx);
            return NULL;
        }

        lm_ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    fclose(file);

    return ctx;
}

void lm_gguf_free(struct lm_gguf_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->kv) {
        // free string memory - not great..
        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            lm_gguf_free_kv(&ctx->kv[i]);
        }

        LM_GGML_FREE(ctx->kv);
    }

    if (ctx->infos) {
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct lm_gguf_tensor_info * info = &ctx->infos[i];

            if (info->name.data) {
                LM_GGML_FREE(info->name.data);
            }
        }

        LM_GGML_FREE(ctx->infos);
    }

    LM_GGML_FREE(ctx);
}

const char * lm_gguf_type_name(enum lm_gguf_type type) {
    return LM_GGUF_TYPE_NAME[type];
}

int lm_gguf_get_version(const struct lm_gguf_context * ctx) {
    return ctx->header.version;
}

size_t lm_gguf_get_alignment(const struct lm_gguf_context * ctx) {
    return ctx->alignment;
}

size_t lm_gguf_get_data_offset(const struct lm_gguf_context * ctx) {
    return ctx->offset;
}

void * lm_gguf_get_data(const struct lm_gguf_context * ctx) {
    return ctx->data;
}

int lm_gguf_get_n_kv(const struct lm_gguf_context * ctx) {
    return ctx->header.n_kv;
}

int lm_gguf_find_key(const struct lm_gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int keyfound = -1;

    const int n_kv = lm_gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, lm_gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * lm_gguf_get_key(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
}

enum lm_gguf_type lm_gguf_get_kv_type(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    return ctx->kv[key_id].type;
}

enum lm_gguf_type lm_gguf_get_arr_type(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.type;
}

const void * lm_gguf_get_arr_data(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.data;
}

const char * lm_gguf_get_arr_str(const struct lm_gguf_context * ctx, int key_id, int i) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_ARRAY);
    struct lm_gguf_kv * kv = &ctx->kv[key_id];
    struct lm_gguf_str * str = &((struct lm_gguf_str *) kv->value.arr.data)[i];
    return str->data;
}

int lm_gguf_get_arr_n(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_ARRAY);
    return ctx->kv[key_id].value.arr.n;
}

uint8_t lm_gguf_get_val_u8(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_UINT8);
    return ctx->kv[key_id].value.uint8;
}

int8_t lm_gguf_get_val_i8(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_INT8);
    return ctx->kv[key_id].value.int8;
}

uint16_t lm_gguf_get_val_u16(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_UINT16);
    return ctx->kv[key_id].value.uint16;
}

int16_t lm_gguf_get_val_i16(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_INT16);
    return ctx->kv[key_id].value.int16;
}

uint32_t lm_gguf_get_val_u32(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}

int32_t lm_gguf_get_val_i32(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_INT32);
    return ctx->kv[key_id].value.int32;
}

float lm_gguf_get_val_f32(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_FLOAT32);
    return ctx->kv[key_id].value.float32;
}

uint64_t lm_gguf_get_val_u64(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_UINT64);
    return ctx->kv[key_id].value.uint64;
}

int64_t lm_gguf_get_val_i64(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_INT64);
    return ctx->kv[key_id].value.int64;
}

double lm_gguf_get_val_f64(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_FLOAT64);
    return ctx->kv[key_id].value.float64;
}

bool lm_gguf_get_val_bool(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_BOOL);
    return ctx->kv[key_id].value.bool_;
}

const char * lm_gguf_get_val_str(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type == LM_GGUF_TYPE_STRING);
    return ctx->kv[key_id].value.str.data;
}

const void * lm_gguf_get_val_data(const struct lm_gguf_context * ctx, int key_id) {
    LM_GGML_ASSERT(key_id >= 0 && key_id < lm_gguf_get_n_kv(ctx));
    LM_GGML_ASSERT(ctx->kv[key_id].type != LM_GGUF_TYPE_ARRAY);
    LM_GGML_ASSERT(ctx->kv[key_id].type != LM_GGUF_TYPE_STRING);
    return &ctx->kv[key_id].value;
}

int lm_gguf_get_n_tensors(const struct lm_gguf_context * ctx) {
    return ctx->header.n_tensors;
}

int lm_gguf_find_tensor(const struct lm_gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int tensorfound = -1;

    const int n_tensors = lm_gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(name, lm_gguf_get_tensor_name(ctx, i)) == 0) {
            tensorfound = i;
            break;
        }
    }

    return tensorfound;
}

size_t lm_gguf_get_tensor_offset(const struct lm_gguf_context * ctx, int i) {
    return ctx->infos[i].offset;
}

char * lm_gguf_get_tensor_name(const struct lm_gguf_context * ctx, int i) {
    return ctx->infos[i].name.data;
}

enum lm_ggml_type lm_gguf_get_tensor_type(const struct lm_gguf_context * ctx, int i) {
    return ctx->infos[i].type;
}

// returns the index
static int lm_gguf_get_or_add_key(struct lm_gguf_context * ctx, const char * key) {
    const int idx = lm_gguf_find_key(ctx, key);
    if (idx >= 0) {
        return idx;
    }

    const int n_kv = lm_gguf_get_n_kv(ctx);

    ctx->kv = realloc(ctx->kv, (n_kv + 1) * sizeof(struct lm_gguf_kv));
    ctx->kv[n_kv].key.n    = strlen(key);
    ctx->kv[n_kv].key.data = strdup(key);
    ctx->header.n_kv++;

    return n_kv;
}

void lm_gguf_remove_key(struct lm_gguf_context * ctx, const char * key) {
    const int idx = lm_gguf_find_key(ctx, key);
    if (idx >= 0) {
        const int n_kv = lm_gguf_get_n_kv(ctx);
        lm_gguf_free_kv(&ctx->kv[idx]);
        for (int i = idx; i < n_kv-1; ++i) {
            ctx->kv[i] = ctx->kv[i+1];
        }
        ctx->kv = realloc(ctx->kv, (n_kv - 1) * sizeof(struct lm_gguf_kv));
        ctx->header.n_kv--;
    }
}

void lm_gguf_set_val_u8(struct lm_gguf_context * ctx, const char * key, uint8_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = LM_GGUF_TYPE_UINT8;
    ctx->kv[idx].value.uint8 = val;
}

void lm_gguf_set_val_i8(struct lm_gguf_context * ctx, const char * key, int8_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type       = LM_GGUF_TYPE_INT8;
    ctx->kv[idx].value.int8 = val;
}

void lm_gguf_set_val_u16(struct lm_gguf_context * ctx, const char * key, uint16_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = LM_GGUF_TYPE_UINT16;
    ctx->kv[idx].value.uint16 = val;
}

void lm_gguf_set_val_i16(struct lm_gguf_context * ctx, const char * key, int16_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = LM_GGUF_TYPE_INT16;
    ctx->kv[idx].value.int16 = val;
}

void lm_gguf_set_val_u32(struct lm_gguf_context * ctx, const char * key, uint32_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = LM_GGUF_TYPE_UINT32;
    ctx->kv[idx].value.uint32 = val;
}

void lm_gguf_set_val_i32(struct lm_gguf_context * ctx, const char * key, int32_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = LM_GGUF_TYPE_INT32;
    ctx->kv[idx].value.int32 = val;
}

void lm_gguf_set_val_f32(struct lm_gguf_context * ctx, const char * key, float val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = LM_GGUF_TYPE_FLOAT32;
    ctx->kv[idx].value.float32 = val;
}

void lm_gguf_set_val_u64(struct lm_gguf_context * ctx, const char * key, uint64_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type         = LM_GGUF_TYPE_UINT64;
    ctx->kv[idx].value.uint64 = val;
}

void lm_gguf_set_val_i64(struct lm_gguf_context * ctx, const char * key, int64_t val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = LM_GGUF_TYPE_INT64;
    ctx->kv[idx].value.int64 = val;
}

void lm_gguf_set_val_f64(struct lm_gguf_context * ctx, const char * key, double val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type          = LM_GGUF_TYPE_FLOAT64;
    ctx->kv[idx].value.float64 = val;
}

void lm_gguf_set_val_bool(struct lm_gguf_context * ctx, const char * key, bool val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type        = LM_GGUF_TYPE_BOOL;
    ctx->kv[idx].value.bool_ = val;
}

void lm_gguf_set_val_str(struct lm_gguf_context * ctx, const char * key, const char * val) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = LM_GGUF_TYPE_STRING;
    ctx->kv[idx].value.str.n    = strlen(val);
    ctx->kv[idx].value.str.data = strdup(val);
}

void lm_gguf_set_arr_data(struct lm_gguf_context * ctx, const char * key, enum lm_gguf_type type, const void * data, int n) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = LM_GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = type;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = LM_GGML_CALLOC(n, lm_gguf_type_size(type));
    memcpy(ctx->kv[idx].value.arr.data, data, n*lm_gguf_type_size(type));
}

void lm_gguf_set_arr_str(struct lm_gguf_context * ctx, const char * key, const char ** data, int n) {
    const int idx = lm_gguf_get_or_add_key(ctx, key);

    ctx->kv[idx].type           = LM_GGUF_TYPE_ARRAY;
    ctx->kv[idx].value.arr.type = LM_GGUF_TYPE_STRING;
    ctx->kv[idx].value.arr.n    = n;
    ctx->kv[idx].value.arr.data = LM_GGML_CALLOC(n, sizeof(struct lm_gguf_str));
    for (int i = 0; i < n; i++) {
        struct lm_gguf_str * str = &((struct lm_gguf_str *)ctx->kv[idx].value.arr.data)[i];
        str->n    = strlen(data[i]);
        str->data = strdup(data[i]);
    }
}

// set or add KV pairs from another context
void lm_gguf_set_kv(struct lm_gguf_context * ctx, struct lm_gguf_context * src) {
    for (uint32_t i = 0; i < src->header.n_kv; i++) {
        switch (src->kv[i].type) {
            case LM_GGUF_TYPE_UINT8:   lm_gguf_set_val_u8  (ctx, src->kv[i].key.data, src->kv[i].value.uint8);    break;
            case LM_GGUF_TYPE_INT8:    lm_gguf_set_val_i8  (ctx, src->kv[i].key.data, src->kv[i].value.int8);     break;
            case LM_GGUF_TYPE_UINT16:  lm_gguf_set_val_u16 (ctx, src->kv[i].key.data, src->kv[i].value.uint16);   break;
            case LM_GGUF_TYPE_INT16:   lm_gguf_set_val_i16 (ctx, src->kv[i].key.data, src->kv[i].value.int16);    break;
            case LM_GGUF_TYPE_UINT32:  lm_gguf_set_val_u32 (ctx, src->kv[i].key.data, src->kv[i].value.uint32);   break;
            case LM_GGUF_TYPE_INT32:   lm_gguf_set_val_i32 (ctx, src->kv[i].key.data, src->kv[i].value.int32);    break;
            case LM_GGUF_TYPE_FLOAT32: lm_gguf_set_val_f32 (ctx, src->kv[i].key.data, src->kv[i].value.float32);  break;
            case LM_GGUF_TYPE_UINT64:  lm_gguf_set_val_u64 (ctx, src->kv[i].key.data, src->kv[i].value.uint64);   break;
            case LM_GGUF_TYPE_INT64:   lm_gguf_set_val_i64 (ctx, src->kv[i].key.data, src->kv[i].value.int64);    break;
            case LM_GGUF_TYPE_FLOAT64: lm_gguf_set_val_f64 (ctx, src->kv[i].key.data, src->kv[i].value.float64);  break;
            case LM_GGUF_TYPE_BOOL:    lm_gguf_set_val_bool(ctx, src->kv[i].key.data, src->kv[i].value.bool_);    break;
            case LM_GGUF_TYPE_STRING:  lm_gguf_set_val_str (ctx, src->kv[i].key.data, src->kv[i].value.str.data); break;
            case LM_GGUF_TYPE_ARRAY:
                {
                    if (src->kv[i].value.arr.type == LM_GGUF_TYPE_STRING) {
                        const char ** data = LM_GGML_CALLOC(src->kv[i].value.arr.n, sizeof(char *));
                        for (uint32_t j = 0; j < src->kv[i].value.arr.n; j++) {
                            data[j] = ((struct lm_gguf_str *)src->kv[i].value.arr.data)[j].data;
                        }
                        lm_gguf_set_arr_str(ctx, src->kv[i].key.data, data, src->kv[i].value.arr.n);
                        LM_GGML_FREE((void *)data);
                    } else if (src->kv[i].value.arr.type == LM_GGUF_TYPE_ARRAY) {
                        LM_GGML_ABORT("nested arrays not supported");
                    } else {
                        lm_gguf_set_arr_data(ctx, src->kv[i].key.data, src->kv[i].value.arr.type, src->kv[i].value.arr.data, src->kv[i].value.arr.n);
                    }
                } break;
            default: LM_GGML_ABORT("invalid type");
        }
    }
}

void lm_gguf_add_tensor(
             struct lm_gguf_context * ctx,
        const struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(tensor);
    if (lm_gguf_find_tensor(ctx, tensor->name) != -1) {
        LM_GGML_ABORT("duplicated tensor name");
    }

    const int idx = ctx->header.n_tensors;
    ctx->infos = realloc(ctx->infos, (idx + 1)*sizeof(struct lm_gguf_tensor_info));

    ctx->infos[idx].name.n    = strlen(tensor->name);
    ctx->infos[idx].name.data = strdup(tensor->name);

    for (int i = 0; i < LM_GGML_MAX_DIMS; ++i) {
        ctx->infos[idx].ne[i] = 1;
    }

    ctx->infos[idx].n_dims = lm_ggml_n_dims(tensor);
    for (uint32_t i = 0; i < ctx->infos[idx].n_dims; i++) {
        ctx->infos[idx].ne[i] = tensor->ne[i];
    }

    ctx->infos[idx].type   = tensor->type;
    ctx->infos[idx].offset = 0;
    ctx->infos[idx].data   = tensor->data;
    ctx->infos[idx].size   = lm_ggml_nbytes(tensor);

    if (ctx->header.n_tensors > 0) {
        ctx->infos[idx].offset = ctx->infos[idx - 1].offset + LM_GGML_PAD(ctx->infos[idx - 1].size, ctx->alignment);
    }

    ctx->header.n_tensors++;
}

void lm_gguf_set_tensor_type(struct lm_gguf_context * ctx, const char * name, enum lm_ggml_type type) {
    const int idx = lm_gguf_find_tensor(ctx, name);
    if (idx < 0) {
        LM_GGML_ABORT("tensor not found");
    }

    ctx->infos[idx].type = type;
}

void lm_gguf_set_tensor_data(struct lm_gguf_context * ctx, const char * name, const void * data, size_t size) {
    const int idx = lm_gguf_find_tensor(ctx, name);
    if (idx < 0) {
        LM_GGML_ABORT("tensor not found");
    }

    ctx->infos[idx].data = data;
    ctx->infos[idx].size = size;

    // update offsets
    for (uint32_t i = idx + 1; i < ctx->header.n_tensors; ++i) {
        ctx->infos[i].offset = ctx->infos[i - 1].offset + LM_GGML_PAD(ctx->infos[i - 1].size, ctx->alignment);
    }
}

//static void lm_gguf_fwrite_str(FILE * file, const struct lm_gguf_str * val) {
//    fwrite(&val->n,   sizeof(val->n),    1, file);
//    fwrite(val->data, sizeof(char), val->n, file);
//}
//
//static void lm_gguf_fwrite_el(FILE * file, const void * val, size_t size) {
//    fwrite(val, sizeof(char), size, file);
//}

struct lm_gguf_buf {
    void * data;
    size_t size;
    size_t offset;
};

static struct lm_gguf_buf lm_gguf_buf_init(size_t size) {
    struct lm_gguf_buf buf = {
        /*buf.data   =*/ size == 0 ? NULL : LM_GGML_CALLOC(1, size),
        /*buf.size   =*/ size,
        /*buf.offset =*/ 0,
    };

    return buf;
}

static void lm_gguf_buf_free(struct lm_gguf_buf buf) {
    if (buf.data) {
        LM_GGML_FREE(buf.data);
    }
}

static void lm_gguf_buf_grow(struct lm_gguf_buf * buf, size_t size) {
    if (buf->offset + size > buf->size) {
        buf->size = 1.5*(buf->offset + size);
        if (buf->data) {
            buf->data = realloc(buf->data, buf->size);
        }
    }
}

static void lm_gguf_bwrite_str(struct lm_gguf_buf * buf, const struct lm_gguf_str * val) {
    lm_gguf_buf_grow(buf, sizeof(val->n) + val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, &val->n, sizeof(val->n));
    }
    buf->offset += sizeof(val->n);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val->data, val->n);
    }
    buf->offset += val->n;
}

static void lm_gguf_bwrite_el(struct lm_gguf_buf * buf, const void * val, size_t el_size) {
    lm_gguf_buf_grow(buf, el_size);

    if (buf->data) {
        memcpy((char *) buf->data + buf->offset, val, el_size);
    }
    buf->offset += el_size;
}

static void lm_gguf_write_to_buf(const struct lm_gguf_context * ctx, struct lm_gguf_buf * buf, bool only_meta) {
    // write header
    lm_gguf_bwrite_el(buf, &ctx->header.magic,     sizeof(ctx->header.magic));
    lm_gguf_bwrite_el(buf, &ctx->header.version,   sizeof(ctx->header.version));
    lm_gguf_bwrite_el(buf, &ctx->header.n_tensors, sizeof(ctx->header.n_tensors));
    lm_gguf_bwrite_el(buf, &ctx->header.n_kv,      sizeof(ctx->header.n_kv));

    // write key-value pairs
    for (uint32_t i = 0; i < ctx->header.n_kv; ++i) {
        struct lm_gguf_kv * kv = &ctx->kv[i];

        lm_gguf_bwrite_str(buf, &kv->key);
        lm_gguf_bwrite_el (buf, &kv->type, sizeof(kv->type));

        switch (kv->type) {
            case LM_GGUF_TYPE_UINT8:   lm_gguf_bwrite_el( buf, &kv->value.uint8,   sizeof(kv->value.uint8)  ); break;
            case LM_GGUF_TYPE_INT8:    lm_gguf_bwrite_el (buf, &kv->value.int8,    sizeof(kv->value.int8)   ); break;
            case LM_GGUF_TYPE_UINT16:  lm_gguf_bwrite_el (buf, &kv->value.uint16,  sizeof(kv->value.uint16) ); break;
            case LM_GGUF_TYPE_INT16:   lm_gguf_bwrite_el (buf, &kv->value.int16,   sizeof(kv->value.int16)  ); break;
            case LM_GGUF_TYPE_UINT32:  lm_gguf_bwrite_el (buf, &kv->value.uint32,  sizeof(kv->value.uint32) ); break;
            case LM_GGUF_TYPE_INT32:   lm_gguf_bwrite_el (buf, &kv->value.int32,   sizeof(kv->value.int32)  ); break;
            case LM_GGUF_TYPE_FLOAT32: lm_gguf_bwrite_el (buf, &kv->value.float32, sizeof(kv->value.float32)); break;
            case LM_GGUF_TYPE_UINT64:  lm_gguf_bwrite_el (buf, &kv->value.uint64,  sizeof(kv->value.uint64) ); break;
            case LM_GGUF_TYPE_INT64:   lm_gguf_bwrite_el (buf, &kv->value.int64,   sizeof(kv->value.int64)  ); break;
            case LM_GGUF_TYPE_FLOAT64: lm_gguf_bwrite_el (buf, &kv->value.float64, sizeof(kv->value.float64)); break;
            case LM_GGUF_TYPE_BOOL:    lm_gguf_bwrite_el (buf, &kv->value.bool_,   sizeof(kv->value.bool_)  ); break;
            case LM_GGUF_TYPE_STRING:  lm_gguf_bwrite_str(buf, &kv->value.str                               ); break;
            case LM_GGUF_TYPE_ARRAY:
                {
                    lm_gguf_bwrite_el(buf, &kv->value.arr.type, sizeof(kv->value.arr.type));
                    lm_gguf_bwrite_el(buf, &kv->value.arr.n,    sizeof(kv->value.arr.n)   );

                    switch (kv->value.arr.type) {
                        case LM_GGUF_TYPE_UINT8:
                        case LM_GGUF_TYPE_INT8:
                        case LM_GGUF_TYPE_UINT16:
                        case LM_GGUF_TYPE_INT16:
                        case LM_GGUF_TYPE_UINT32:
                        case LM_GGUF_TYPE_INT32:
                        case LM_GGUF_TYPE_FLOAT32:
                        case LM_GGUF_TYPE_UINT64:
                        case LM_GGUF_TYPE_INT64:
                        case LM_GGUF_TYPE_FLOAT64:
                        case LM_GGUF_TYPE_BOOL:
                            {
                                lm_gguf_bwrite_el(buf, kv->value.arr.data, kv->value.arr.n * lm_gguf_type_size(kv->value.arr.type));
                            } break;
                        case LM_GGUF_TYPE_STRING:
                            {
                                for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                                    lm_gguf_bwrite_str(buf, &((struct lm_gguf_str *) kv->value.arr.data)[j]);
                                }
                            } break;
                        case LM_GGUF_TYPE_ARRAY:
                        default: LM_GGML_ABORT("invalid type");
                    }
                } break;
            default: LM_GGML_ABORT("invalid type");
        }
    }

    // write tensor infos
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct lm_gguf_tensor_info * info = &ctx->infos[i];

        lm_gguf_bwrite_str(buf, &info->name);
        lm_gguf_bwrite_el (buf, &info->n_dims, sizeof(info->n_dims));
        for (uint32_t j = 0; j < info->n_dims; ++j) {
            lm_gguf_bwrite_el(buf, &info->ne[j], sizeof(info->ne[j]));
        }
        lm_gguf_bwrite_el(buf, &info->type,   sizeof(info->type));
        lm_gguf_bwrite_el(buf, &info->offset, sizeof(info->offset));
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset     = buf->offset;
        const size_t offset_pad = LM_GGML_PAD(offset, ctx->alignment);

        if (offset_pad != offset) {
            uint8_t pad = 0;
            for (size_t i = 0; i < offset_pad - offset; ++i) {
                lm_gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }
    }

    if (only_meta) {
        return;
    }

    size_t offset = 0;

    // write tensor data
    for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct lm_gguf_tensor_info * info = &ctx->infos[i];

        const size_t size     = info->size;
        const size_t size_pad = LM_GGML_PAD(size, ctx->alignment);

        lm_gguf_bwrite_el(buf, info->data, size);

        if (size_pad != size) {
            uint8_t pad = 0;
            for (size_t j = 0; j < size_pad - size; ++j) {
                lm_gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }

        LM_GGML_ASSERT(offset == info->offset);

        offset += size_pad;
    }
}

void lm_gguf_write_to_file(const struct lm_gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = lm_ggml_fopen(fname, "wb");
    if (!file) {
        LM_GGML_ABORT("failed to open file for writing");
    }

    struct lm_gguf_buf buf = lm_gguf_buf_init(16*1024);

    lm_gguf_write_to_buf(ctx, &buf, only_meta);

    fwrite(buf.data, 1, buf.offset, file);

    lm_gguf_buf_free(buf);

    fclose(file);
}

size_t lm_gguf_get_meta_size(const struct lm_gguf_context * ctx) {
    // no allocs - only compute size
    struct lm_gguf_buf buf = lm_gguf_buf_init(0);

    lm_gguf_write_to_buf(ctx, &buf, true);

    return buf.offset;
}

void lm_gguf_get_meta_data(const struct lm_gguf_context * ctx, void * data) {
    struct lm_gguf_buf buf = lm_gguf_buf_init(16*1024);

    lm_gguf_write_to_buf(ctx, &buf, true);

    memcpy(data, buf.data, buf.offset);

    lm_gguf_buf_free(buf);
}

void lm_ggml_log_set(lm_ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : lm_ggml_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}
