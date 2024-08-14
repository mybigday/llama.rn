#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
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

#ifdef LM_GGML_USE_OPENMP
#include <omp.h>
#endif

#ifdef LM_GGML_USE_METAL
#include <unistd.h>
#endif

#if defined(__ARM_FEATURE_SVE)
int lm_ggml_sve_cnt_b = 0;
#endif
#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef LM_GGML_USE_LLAMAFILE
#endif

#ifdef LM_GGML_USE_LLAMAFILE
#include <llamafile/sgemm.h>
#endif

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)

// disable POSIX deprecation warnings
// these functions are never going away, anyway
#pragma warning(disable: 4996)
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_sub(atomic_int * ptr, LONG dec) {
    return atomic_fetch_add(ptr, -(dec));
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else
#include <pthread.h>
#include <stdatomic.h>

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t lm_ggml_thread_t;

#ifdef LM_GGML_USE_CPU_HBM
#include <hbwmalloc.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if (defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && \
    (!defined(TARGET_OS_TV) && !defined(TARGET_OS_WATCH))

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

#define LM_GGML_DEBUG 0
#define LM_GGML_GELU_FP16
#define LM_GGML_GELU_QUICK_FP16

#define LM_GGML_SOFT_MAX_UNROLL 4
#define LM_GGML_VEC_DOT_UNROLL  2
#define LM_GGML_VEC_MAD_UNROLL  32

//
// logging
//

#if (LM_GGML_DEBUG >= 1)
#define LM_GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG(...)
#endif

#if (LM_GGML_DEBUG >= 5)
#define LM_GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG_5(...)
#endif

#if (LM_GGML_DEBUG >= 10)
#define LM_GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define LM_GGML_PRINT_DEBUG_10(...)
#endif

#define LM_GGML_PRINT(...) printf(__VA_ARGS__)

//
// end of logging block
//

#ifdef LM_GGML_USE_ACCELERATE
// uncomment to use vDSP for soft max computation
// note: not sure if it is actually faster
//#define LM_GGML_SOFT_MAX_ACCELERATE
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#define LM_GGML_ALIGNED_MALLOC(size) _aligned_malloc(size, LM_GGML_MEM_ALIGN)
#define LM_GGML_ALIGNED_FREE(ptr)    _aligned_free(ptr)
#else
inline static void * lm_ggml_aligned_malloc(size_t size) {
    if (size == 0) {
        LM_GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for lm_ggml_aligned_malloc!\n");
        return NULL;
    }
    void * aligned_memory = NULL;
#ifdef LM_GGML_USE_CPU_HBM
    int result = hbw_posix_memalign(&aligned_memory, 16, size);
#elif LM_GGML_USE_METAL
    int result = posix_memalign(&aligned_memory, sysconf(_SC_PAGESIZE), size);
#else
    int result = posix_memalign(&aligned_memory, LM_GGML_MEM_ALIGN, size);
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
        LM_GGML_PRINT("%s: %s (attempted to allocate %6.2f MB)\n", __func__, error_desc, size/(1024.0*1024.0));
        LM_GGML_ABORT("fatal error");
        return NULL;
    }
    return aligned_memory;
}
#define LM_GGML_ALIGNED_MALLOC(size) lm_ggml_aligned_malloc(size)
#ifdef LM_GGML_USE_CPU_HBM
#define LM_GGML_ALIGNED_FREE(ptr)    if(NULL != ptr) hbw_free(ptr)
#else
#define LM_GGML_ALIGNED_FREE(ptr)    free(ptr)
#endif
#endif

inline static void * lm_ggml_malloc(size_t size) {
    if (size == 0) {
        LM_GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for lm_ggml_malloc!\n");
        return NULL;
    }
    void * result = malloc(size);
    if (result == NULL) {
        LM_GGML_PRINT("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        LM_GGML_ABORT("fatal error");
    }
    return result;
}

// calloc
inline static void * lm_ggml_calloc(size_t num, size_t size) {
    if (num == 0 || size == 0) {
        LM_GGML_PRINT("WARNING: Behavior may be unexpected when allocating 0 bytes for lm_ggml_calloc!\n");
        return NULL;
    }
    void * result = calloc(num, size);
    if (result == NULL) {
        LM_GGML_PRINT("%s: failed to allocate %6.2f MB\n", __func__, size/(1024.0*1024.0));
        LM_GGML_ABORT("fatal error");
    }
    return result;
}

#define LM_GGML_MALLOC(size)      lm_ggml_malloc(size)
#define LM_GGML_CALLOC(num, size) lm_ggml_calloc(num, size)

#define LM_GGML_FREE(ptr) free(ptr)

#define UNUSED LM_GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

#if defined(LM_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double lm_ggml_float;

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

//
// global data
//

// precomputed gelu table for f16 (128 KB)
static lm_ggml_fp16_t lm_ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
static lm_ggml_fp16_t lm_ggml_table_gelu_quick_f16[1 << 16];

// precomputed f32 table for f16 (256 KB) (ggml-impl.h)
float lm_ggml_table_f32_f16[1 << 16];

LM_GGML_CALL const char * lm_ggml_status_to_string(enum lm_ggml_status status) {
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

void lm_ggml_fp32_to_fp16_row(const float * x, lm_ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
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
#endif
    for (; i < n; i++) {
        y[i] = LM_GGML_FP32_TO_FP16(x[i]);
    }
}

void lm_ggml_bf16_to_fp32_row(const lm_ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(y + i,
                         _mm512_castsi512_ps(
                             _mm512_slli_epi32(
                                 _mm512_cvtepu16_epi32(
                                     _mm256_loadu_si256(
                                         (const __m256i *)(x + i))),
                                 16)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i,
                         _mm256_castsi256_ps(
                             _mm256_slli_epi32(
                                 _mm256_cvtepu16_epi32(
                                     _mm_loadu_si128(
                                         (const __m128i *)(x + i))),
                                 16)));
    }
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

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

static void lm_ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);
static void lm_ggml_vec_dot_f16(int n, float * restrict s, size_t bs, lm_ggml_fp16_t * restrict x, size_t bx, lm_ggml_fp16_t * restrict y, size_t by, int nrc);
static void lm_ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, lm_ggml_bf16_t * restrict x, size_t bx, lm_ggml_bf16_t * restrict y, size_t by, int nrc);

static const lm_ggml_type_traits_t type_traits[LM_GGML_TYPE_COUNT] = {
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
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_f32,
        .vec_dot_type             = LM_GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(lm_ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (lm_ggml_to_float_t) lm_ggml_fp16_to_fp32_row,
        .from_float               = (lm_ggml_from_float_t) lm_ggml_fp32_to_fp16_row,
        .from_float_ref           = (lm_ggml_from_float_t) lm_ggml_fp32_to_fp16_row,
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_f16,
        .vec_dot_type             = LM_GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_0,
        .from_float               = quantize_row_q4_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_0_ref,
        .vec_dot                  = lm_ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_1,
        .from_float               = quantize_row_q4_1,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_1_ref,
        .vec_dot                  = lm_ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [4] = { // LM_GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = LM_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [5] = { // LM_GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = LM_GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_0,
        .from_float               = quantize_row_q5_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_0_ref,
        .vec_dot                  = lm_ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_1,
        .from_float               = quantize_row_q5_1,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_1_ref,
        .vec_dot                  = lm_ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q8_0,
        .from_float               = quantize_row_q8_0,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q8_0_ref,
        .from_float_to_mat        = quantize_mat_q8_0,
        .vec_dot                  = lm_ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [LM_GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_1,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q8_1_ref,
        .vec_dot_type             = LM_GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q2_K,
        .from_float               = quantize_row_q2_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q2_K_ref,
        .vec_dot                  = lm_ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q3_K,
        .from_float               = quantize_row_q3_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q3_K_ref,
        .vec_dot                  = lm_ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q4_K,
        .from_float               = quantize_row_q4_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q4_K_ref,
        .vec_dot                  = lm_ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q5_K,
        .from_float               = quantize_row_q5_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q5_K_ref,
        .vec_dot                  = lm_ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_q6_K,
        .from_float               = quantize_row_q6_K,
        .from_float_ref           = (lm_ggml_from_float_t) quantize_row_q6_K_ref,
        .vec_dot                  = lm_ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float               = quantize_row_iq3_xxs,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq3_xxs_ref,
        .vec_dot                  = lm_ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq3_s,
        .from_float               = quantize_row_iq3_s,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq3_s_ref,
        .vec_dot                  = lm_ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq2_s,
        .from_float               = quantize_row_iq2_s,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq2_s_ref,
        .vec_dot                  = lm_ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq1_s,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ1_M] = {
        .type_name                = "iq1_m",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_m),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq1_m,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = lm_ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float               = quantize_row_iq4_nl,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq4_nl_ref,
        .vec_dot                  = lm_ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (lm_ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float               = quantize_row_iq4_xs,
        .from_float_ref           = (lm_ggml_from_float_t)quantize_row_iq4_xs_ref,
        .vec_dot                  = lm_ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = LM_GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_K,
    },
    [LM_GGML_TYPE_BF16] = {
        .type_name                = "bf16",
        .blck_size                = 1,
        .type_size                = sizeof(lm_ggml_bf16_t),
        .is_quantized             = false,
        .to_float                 = (lm_ggml_to_float_t) lm_ggml_bf16_to_fp32_row,
        .from_float               = (lm_ggml_from_float_t) lm_ggml_fp32_to_bf16_row,
        .from_float_ref           = (lm_ggml_from_float_t) lm_ggml_fp32_to_bf16_row_ref,
        .vec_dot                  = (lm_ggml_vec_dot_t) lm_ggml_vec_dot_bf16,
        .vec_dot_type             = LM_GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [LM_GGML_TYPE_Q4_0_4_4] = {
        .type_name                = "q4_0_4x4",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 4,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = lm_ggml_gemv_q4_0_4x4_q8_0,
        .gemm                     = lm_ggml_gemm_q4_0_4x4_q8_0,
    },
    [LM_GGML_TYPE_Q4_0_4_8] = {
        .type_name                = "q4_0_4x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 4,
        .gemv                     = lm_ggml_gemv_q4_0_4x8_q8_0,
        .gemm                     = lm_ggml_gemm_q4_0_4x8_q8_0,
    },
    [LM_GGML_TYPE_Q4_0_8_8] = {
        .type_name                = "q4_0_8x8",
        .blck_size                = QK4_0,
        .blck_size_interleave     = 8,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_ref           = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = LM_GGML_TYPE_Q8_0,
        .nrows                    = 1,
        .ncols                    = 8,
        .gemv                     = lm_ggml_gemv_q4_0_8x8_q8_0,
        .gemm                     = lm_ggml_gemm_q4_0_8x8_q8_0,
    }
};

// For internal test use
lm_ggml_type_traits_t lm_ggml_internal_get_type_traits(enum lm_ggml_type type) {
    LM_GGML_ASSERT(type < LM_GGML_TYPE_COUNT);
    return type_traits[type];
}

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// LM_GGML_F32_STEP / LM_GGML_F16_STEP
//   number of elements to process in a single step
//
// LM_GGML_F32_EPR / LM_GGML_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define LM_GGML_SIMD

// F32 NEON

#define LM_GGML_F32_STEP 16
#define LM_GGML_F32_EPR  4

#define LM_GGML_F32x4              float32x4_t
#define LM_GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define LM_GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define LM_GGML_F32x4_LOAD         vld1q_f32
#define LM_GGML_F32x4_STORE        vst1q_f32
#define LM_GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define LM_GGML_F32x4_ADD          vaddq_f32
#define LM_GGML_F32x4_MUL          vmulq_f32
#define LM_GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define LM_GGML_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = LM_GGML_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vaddq_f32(x[i], x[offset+i]);   \
    }                                          \
    res = LM_GGML_F32x4_REDUCE_ONE(x[0]);         \
}

#define LM_GGML_F32_VEC        LM_GGML_F32x4
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x4_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x4_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define LM_GGML_F16_STEP 32
    #define LM_GGML_F16_EPR  8

    #define LM_GGML_F16x8              float16x8_t
    #define LM_GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define LM_GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define LM_GGML_F16x8_LOAD(x)      vld1q_f16((const lm_ggml_fp16_internal_t *)(x))
    #define LM_GGML_F16x8_STORE        vst1q_f16
    #define LM_GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define LM_GGML_F16x8_ADD          vaddq_f16
    #define LM_GGML_F16x8_MUL          vmulq_f16
    #define LM_GGML_F16x8_REDUCE(res, x)                             \
    do {                                                          \
        int offset = LM_GGML_F16_ARR >> 1;                           \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        offset >>= 1;                                             \
        for (int i = 0; i < offset; ++i) {                        \
            x[i] = vaddq_f16(x[i], x[offset+i]);                  \
        }                                                         \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 (x[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(x[0])); \
        res = (lm_ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } while (0)

    #define LM_GGML_F16_VEC                LM_GGML_F16x8
    #define LM_GGML_F16_VEC_ZERO           LM_GGML_F16x8_ZERO
    #define LM_GGML_F16_VEC_SET1           LM_GGML_F16x8_SET1
    #define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F16x8_LOAD(p)
    #define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F16x8_STORE((lm_ggml_fp16_internal_t *)(p), r[i])
    #define LM_GGML_F16_VEC_FMA            LM_GGML_F16x8_FMA
    #define LM_GGML_F16_VEC_ADD            LM_GGML_F16x8_ADD
    #define LM_GGML_F16_VEC_MUL            LM_GGML_F16x8_MUL
    #define LM_GGML_F16_VEC_REDUCE         LM_GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define LM_GGML_F16_STEP 16
    #define LM_GGML_F16_EPR  4

    #define LM_GGML_F32Cx4              float32x4_t
    #define LM_GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define LM_GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define LM_GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const lm_ggml_fp16_internal_t *)(x)))
    #define LM_GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define LM_GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define LM_GGML_F32Cx4_ADD          vaddq_f32
    #define LM_GGML_F32Cx4_MUL          vmulq_f32
    #define LM_GGML_F32Cx4_REDUCE       LM_GGML_F32x4_REDUCE

    #define LM_GGML_F16_VEC                LM_GGML_F32Cx4
    #define LM_GGML_F16_VEC_ZERO           LM_GGML_F32Cx4_ZERO
    #define LM_GGML_F16_VEC_SET1           LM_GGML_F32Cx4_SET1
    #define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F32Cx4_LOAD(p)
    #define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F32Cx4_STORE((lm_ggml_fp16_internal_t *)(p), r[i])
    #define LM_GGML_F16_VEC_FMA            LM_GGML_F32Cx4_FMA
    #define LM_GGML_F16_VEC_ADD            LM_GGML_F32Cx4_ADD
    #define LM_GGML_F16_VEC_MUL            LM_GGML_F32Cx4_MUL
    #define LM_GGML_F16_VEC_REDUCE         LM_GGML_F32Cx4_REDUCE
#endif

#elif defined(__AVX512F__)

#define LM_GGML_SIMD

// F32 AVX512

#define LM_GGML_F32_STEP 64
#define LM_GGML_F32_EPR  16

#define LM_GGML_F32x16         __m512
#define LM_GGML_F32x16_ZERO    _mm512_setzero_ps()
#define LM_GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#define LM_GGML_F32x16_LOAD    _mm512_loadu_ps
#define LM_GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#define LM_GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define LM_GGML_F32x16_ADD     _mm512_add_ps
#define LM_GGML_F32x16_MUL     _mm512_mul_ps
#define LM_GGML_F32x16_REDUCE(res, x)                                    \
do {                                                                  \
    int offset = LM_GGML_F32_ARR >> 1;                                   \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    res = _mm512_reduce_add_ps(x[0]);                                 \
} while (0)

// TODO: is this optimal ?

#define LM_GGML_F32_VEC        LM_GGML_F32x16
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x16_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x16_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x16_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x16_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x16_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x16_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x16_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x16_REDUCE

// F16 AVX512

// F16 AVX

#define LM_GGML_F16_STEP 64
#define LM_GGML_F16_EPR  16

// AVX512 has FP16 extension (AVX512_FP16) but I don't have it on my machine so I use FP32 instead

#define LM_GGML_F32Cx16             __m512
#define LM_GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#define LM_GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#define LM_GGML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define LM_GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#define LM_GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define LM_GGML_F32Cx16_ADD         _mm512_add_ps
#define LM_GGML_F32Cx16_MUL         _mm512_mul_ps
#define LM_GGML_F32Cx16_REDUCE(res, x)                               \
do {                                                              \
    int offset = LM_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    res = _mm512_reduce_add_ps(x[0]);                             \
} while (0)

#define LM_GGML_F16_VEC                LM_GGML_F32Cx16
#define LM_GGML_F16_VEC_ZERO           LM_GGML_F32Cx16_ZERO
#define LM_GGML_F16_VEC_SET1           LM_GGML_F32Cx16_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F32Cx16_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F32Cx16_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA            LM_GGML_F32Cx16_FMA
#define LM_GGML_F16_VEC_ADD            LM_GGML_F32Cx16_ADD
#define LM_GGML_F16_VEC_MUL            LM_GGML_F32Cx16_MUL
#define LM_GGML_F16_VEC_REDUCE         LM_GGML_F32Cx16_REDUCE

#elif defined(__AVX__)

#define LM_GGML_SIMD

// F32 AVX

#define LM_GGML_F32_STEP 32
#define LM_GGML_F32_EPR  8

#define LM_GGML_F32x8         __m256
#define LM_GGML_F32x8_ZERO    _mm256_setzero_ps()
#define LM_GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#define LM_GGML_F32x8_LOAD    _mm256_loadu_ps
#define LM_GGML_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define LM_GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define LM_GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define LM_GGML_F32x8_ADD     _mm256_add_ps
#define LM_GGML_F32x8_MUL     _mm256_mul_ps
#define LM_GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = LM_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (lm_ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} while (0)
// TODO: is this optimal ?

#define LM_GGML_F32_VEC        LM_GGML_F32x8
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x8_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x8_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x8_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x8_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x8_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x8_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x8_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x8_REDUCE

// F16 AVX

#define LM_GGML_F16_STEP 32
#define LM_GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define LM_GGML_F32Cx8             __m256
#define LM_GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#define LM_GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define LM_GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define LM_GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(lm_ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = LM_GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(lm_ggml_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = LM_GGML_FP32_TO_FP16(arr[i]);
}
#define LM_GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define LM_GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define LM_GGML_F32Cx8_FMA         LM_GGML_F32x8_FMA
#define LM_GGML_F32Cx8_ADD         _mm256_add_ps
#define LM_GGML_F32Cx8_MUL         _mm256_mul_ps
#define LM_GGML_F32Cx8_REDUCE      LM_GGML_F32x8_REDUCE

#define LM_GGML_F16_VEC                LM_GGML_F32Cx8
#define LM_GGML_F16_VEC_ZERO           LM_GGML_F32Cx8_ZERO
#define LM_GGML_F16_VEC_SET1           LM_GGML_F32Cx8_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F32Cx8_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F32Cx8_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA            LM_GGML_F32Cx8_FMA
#define LM_GGML_F16_VEC_ADD            LM_GGML_F32Cx8_ADD
#define LM_GGML_F16_VEC_MUL            LM_GGML_F32Cx8_MUL
#define LM_GGML_F16_VEC_REDUCE         LM_GGML_F32Cx8_REDUCE

#elif defined(__POWER9_VECTOR__)

#define LM_GGML_SIMD

// F32 POWER9

#define LM_GGML_F32_STEP 32
#define LM_GGML_F32_EPR  4

#define LM_GGML_F32x4              vector float
#define LM_GGML_F32x4_ZERO         0.0f
#define LM_GGML_F32x4_SET1         vec_splats
#define LM_GGML_F32x4_LOAD(p)      vec_xl(0, p)
#define LM_GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define LM_GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define LM_GGML_F32x4_ADD          vec_add
#define LM_GGML_F32x4_MUL          vec_mul
#define LM_GGML_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = LM_GGML_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}

#define LM_GGML_F32_VEC        LM_GGML_F32x4
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x4_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x4_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x4_REDUCE

// F16 POWER9
#define LM_GGML_F16_STEP       LM_GGML_F32_STEP
#define LM_GGML_F16_EPR        LM_GGML_F32_EPR
#define LM_GGML_F16_VEC        LM_GGML_F32x4
#define LM_GGML_F16_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F16_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F16_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F16_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F16_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F16_VEC_REDUCE LM_GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#define LM_GGML_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - LM_GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
#define LM_GGML_ENDIAN_BYTE(i) ((unsigned char *)&(uint16_t){1})[i]
#define LM_GGML_F16_VEC_STORE(p, r, i)                             \
  if (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - LM_GGML_ENDIAN_BYTE(1)],  \
                                   r[i - LM_GGML_ENDIAN_BYTE(0)]), \
            0, p - LM_GGML_F16_EPR)

#elif defined(__wasm_simd128__)

#define LM_GGML_SIMD

// F32 WASM

#define LM_GGML_F32_STEP 16
#define LM_GGML_F32_EPR  4

#define LM_GGML_F32x4              v128_t
#define LM_GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#define LM_GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#define LM_GGML_F32x4_LOAD         wasm_v128_load
#define LM_GGML_F32x4_STORE        wasm_v128_store
#define LM_GGML_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#define LM_GGML_F32x4_ADD          wasm_f32x4_add
#define LM_GGML_F32x4_MUL          wasm_f32x4_mul
#define LM_GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    int offset = LM_GGML_F32_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define LM_GGML_F32_VEC        LM_GGML_F32x4
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x4_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x4_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x4_REDUCE

// F16 WASM

#define LM_GGML_F16_STEP 16
#define LM_GGML_F16_EPR  4

inline static v128_t __wasm_f16x4_load(const lm_ggml_fp16_t * p) {
    float tmp[4];

    tmp[0] = LM_GGML_FP16_TO_FP32(p[0]);
    tmp[1] = LM_GGML_FP16_TO_FP32(p[1]);
    tmp[2] = LM_GGML_FP16_TO_FP32(p[2]);
    tmp[3] = LM_GGML_FP16_TO_FP32(p[3]);

    return wasm_v128_load(tmp);
}

inline static void __wasm_f16x4_store(lm_ggml_fp16_t * p, v128_t x) {
    float tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = LM_GGML_FP32_TO_FP16(tmp[0]);
    p[1] = LM_GGML_FP32_TO_FP16(tmp[1]);
    p[2] = LM_GGML_FP32_TO_FP16(tmp[2]);
    p[3] = LM_GGML_FP32_TO_FP16(tmp[3]);
}

#define LM_GGML_F16x4             v128_t
#define LM_GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#define LM_GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#define LM_GGML_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#define LM_GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#define LM_GGML_F16x4_FMA         LM_GGML_F32x4_FMA
#define LM_GGML_F16x4_ADD         wasm_f32x4_add
#define LM_GGML_F16x4_MUL         wasm_f32x4_mul
#define LM_GGML_F16x4_REDUCE(res, x)                  \
{                                                  \
    int offset = LM_GGML_F16_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define LM_GGML_F16_VEC                LM_GGML_F16x4
#define LM_GGML_F16_VEC_ZERO           LM_GGML_F16x4_ZERO
#define LM_GGML_F16_VEC_SET1           LM_GGML_F16x4_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F16x4_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F16x4_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA            LM_GGML_F16x4_FMA
#define LM_GGML_F16_VEC_ADD            LM_GGML_F16x4_ADD
#define LM_GGML_F16_VEC_MUL            LM_GGML_F16x4_MUL
#define LM_GGML_F16_VEC_REDUCE         LM_GGML_F16x4_REDUCE

#elif defined(__SSE3__)

#define LM_GGML_SIMD

// F32 SSE

#define LM_GGML_F32_STEP 32
#define LM_GGML_F32_EPR  4

#define LM_GGML_F32x4         __m128
#define LM_GGML_F32x4_ZERO    _mm_setzero_ps()
#define LM_GGML_F32x4_SET1(x) _mm_set1_ps(x)
#define LM_GGML_F32x4_LOAD    _mm_loadu_ps
#define LM_GGML_F32x4_STORE   _mm_storeu_ps
#if defined(__FMA__)
    // TODO: Does this work?
    #define LM_GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
    #define LM_GGML_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define LM_GGML_F32x4_ADD     _mm_add_ps
#define LM_GGML_F32x4_MUL     _mm_mul_ps
#define LM_GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    int offset = LM_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (lm_ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#define LM_GGML_F32_VEC        LM_GGML_F32x4
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x4_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x4_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x4_REDUCE

// F16 SSE

#define LM_GGML_F16_STEP 32
#define LM_GGML_F16_EPR  4

static inline __m128 __sse_f16x4_load(lm_ggml_fp16_t *x) {
    float tmp[4];

    tmp[0] = LM_GGML_FP16_TO_FP32(x[0]);
    tmp[1] = LM_GGML_FP16_TO_FP32(x[1]);
    tmp[2] = LM_GGML_FP16_TO_FP32(x[2]);
    tmp[3] = LM_GGML_FP16_TO_FP32(x[3]);

    return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(lm_ggml_fp16_t *x, __m128 y) {
    float arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = LM_GGML_FP32_TO_FP16(arr[0]);
    x[1] = LM_GGML_FP32_TO_FP16(arr[1]);
    x[2] = LM_GGML_FP32_TO_FP16(arr[2]);
    x[3] = LM_GGML_FP32_TO_FP16(arr[3]);
}

#define LM_GGML_F32Cx4             __m128
#define LM_GGML_F32Cx4_ZERO        _mm_setzero_ps()
#define LM_GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#define LM_GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#define LM_GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define LM_GGML_F32Cx4_FMA         LM_GGML_F32x4_FMA
#define LM_GGML_F32Cx4_ADD         _mm_add_ps
#define LM_GGML_F32Cx4_MUL         _mm_mul_ps
#define LM_GGML_F32Cx4_REDUCE      LM_GGML_F32x4_REDUCE

#define LM_GGML_F16_VEC                 LM_GGML_F32Cx4
#define LM_GGML_F16_VEC_ZERO            LM_GGML_F32Cx4_ZERO
#define LM_GGML_F16_VEC_SET1            LM_GGML_F32Cx4_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)      LM_GGML_F32Cx4_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i)  LM_GGML_F32Cx4_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA             LM_GGML_F32Cx4_FMA
#define LM_GGML_F16_VEC_ADD             LM_GGML_F32Cx4_ADD
#define LM_GGML_F16_VEC_MUL             LM_GGML_F32Cx4_MUL
#define LM_GGML_F16_VEC_REDUCE          LM_GGML_F32Cx4_REDUCE

#elif defined(__loongarch_asx)

#define LM_GGML_SIMD

// F32 LASX
#define LM_GGML_F32_STEP 32
#define LM_GGML_F32_EPR  8

#define LM_GGML_F32x8         __m256
#define LM_GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#define LM_GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#define LM_GGML_F32x8_LOAD(x) (__m256)__lasx_xvld((x), 0)
#define LM_GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#define LM_GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#define LM_GGML_F32x8_ADD     __lasx_xvfadd_s
#define LM_GGML_F32x8_MUL     __lasx_xvfmul_s
#define LM_GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = LM_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    float *tmp_p = (float *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} while (0)
// TODO: is this optimal ?

#define LM_GGML_F32_VEC        LM_GGML_F32x8
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x8_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x8_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x8_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x8_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x8_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x8_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x8_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x8_REDUCE

// F16 LASX

#define LM_GGML_F16_STEP 32
#define LM_GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define LM_GGML_F32Cx8          __m256
#define LM_GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#define LM_GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplgr2vr_w((x))

static inline __m256 __lasx_f32cx8_load(const lm_ggml_fp16_t * x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = LM_GGML_FP16_TO_FP32(x[i]);
    }

    return (__m256)__lasx_xvld(tmp, 0);
}
static inline void __lasx_f32cx8_store(lm_ggml_fp16_t * x, __m256 y) {
    float arr[8];

    __lasx_xvst(y, arr, 0);

    for (int i = 0; i < 8; i++) {
        x[i] = LM_GGML_FP32_TO_FP16(arr[i]);
    }
}
#define LM_GGML_F32Cx8_LOAD(x)     __lasx_f32cx8_load(x)
#define LM_GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#define LM_GGML_F32Cx8_FMA         LM_GGML_F32x8_FMA
#define LM_GGML_F32Cx8_ADD         __lasx_xvfadd_s
#define LM_GGML_F32Cx8_MUL         __lasx_xvfmul_s
#define LM_GGML_F32Cx8_REDUCE      LM_GGML_F32x8_REDUCE

#define LM_GGML_F16_VEC                LM_GGML_F32Cx8
#define LM_GGML_F16_VEC_ZERO           LM_GGML_F32Cx8_ZERO
#define LM_GGML_F16_VEC_SET1           LM_GGML_F32Cx8_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)     LM_GGML_F32Cx8_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i) LM_GGML_F32Cx8_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA            LM_GGML_F32Cx8_FMA
#define LM_GGML_F16_VEC_ADD            LM_GGML_F32Cx8_ADD
#define LM_GGML_F16_VEC_MUL            LM_GGML_F32Cx8_MUL
#define LM_GGML_F16_VEC_REDUCE         LM_GGML_F32Cx8_REDUCE

#elif defined(__loongarch_sx)

#define LM_GGML_SIMD

// F32 LSX

#define LM_GGML_F32_STEP 32
#define LM_GGML_F32_EPR  4

#define LM_GGML_F32x4         __m128
#define LM_GGML_F32x4_ZERO    __lsx_vldi(0)
#define LM_GGML_F32x4_SET1(x) __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define LM_GGML_F32x4_LOAD(x) __lsx_vld((x), 0)
#define LM_GGML_F32x4_STORE((x),(y))   __lsx_vst((y), (x), 0)
#define LM_GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#define LM_GGML_F32x4_ADD     __lsx_vfadd_s
#define LM_GGML_F32x4_MUL     __lsx_vfmul_s
#define LM_GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    int offset = LM_GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lsx_vfadd_s(x[i], x[offset+i]);                     \
    }                                                             \
    __m128i tmp = __lsx_vsrli_d((__m128i)x[0], 32); \
    tmp = (__m128i)__lsx_vfadd_s((__m128)tmp, x[0]); \
    tmp = __lsx_vpickev_w(__lsx_vldi(0), tmp); \
    const __m128 t0 = __lsx_vshuf4i_w(tmp, 0x88); \
    tmp = __lsx_vsrli_d((__m128i)t0, 32); \
    tmp = (__m128i)__lsx_vfadd_s((__m128)tmp, t0); \
    tmp = __lsx_vpickev_w(__lsx_vldi(0), tmp); \
    res = (lm_ggml_float) __lsx_vpickve2gr_w(__lsx_vshuf4i_w(tmp, 0x88), 0);        \
}

#define LM_GGML_F32_VEC        LM_GGML_F32x4
#define LM_GGML_F32_VEC_ZERO   LM_GGML_F32x4_ZERO
#define LM_GGML_F32_VEC_SET1   LM_GGML_F32x4_SET1
#define LM_GGML_F32_VEC_LOAD   LM_GGML_F32x4_LOAD
#define LM_GGML_F32_VEC_STORE  LM_GGML_F32x4_STORE
#define LM_GGML_F32_VEC_FMA    LM_GGML_F32x4_FMA
#define LM_GGML_F32_VEC_ADD    LM_GGML_F32x4_ADD
#define LM_GGML_F32_VEC_MUL    LM_GGML_F32x4_MUL
#define LM_GGML_F32_VEC_REDUCE LM_GGML_F32x4_REDUCE

// F16 LSX

#define LM_GGML_F16_STEP 32
#define LM_GGML_F16_EPR  4

static inline __m128 __lsx_f16x4_load(const lm_ggml_fp16_t * x) {
    float tmp[4];

    tmp[0] = LM_GGML_FP16_TO_FP32(x[0]);
    tmp[1] = LM_GGML_FP16_TO_FP32(x[1]);
    tmp[2] = LM_GGML_FP16_TO_FP32(x[2]);
    tmp[3] = LM_GGML_FP16_TO_FP32(x[3]);

    return __lsx_vld(tmp, 0);
}

static inline void __lsx_f16x4_store(lm_ggml_fp16_t * x, __m128 y) {
    float arr[4];

    __lsx_vst(y, arr, 0);

    x[0] = LM_GGML_FP32_TO_FP16(arr[0]);
    x[1] = LM_GGML_FP32_TO_FP16(arr[1]);
    x[2] = LM_GGML_FP32_TO_FP16(arr[2]);
    x[3] = LM_GGML_FP32_TO_FP16(arr[3]);
}

#define LM_GGML_F32Cx4             __m128
#define LM_GGML_F32Cx4_ZERO        __lsx_vldi(0)
#define LM_GGML_F32Cx4_SET1(x)     __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define LM_GGML_F32Cx4_LOAD(x)     __lsx_f16x4_load(x)
#define LM_GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#define LM_GGML_F32Cx4_FMA         LM_GGML_F32x4_FMA
#define LM_GGML_F32Cx4_ADD         __lsx_vfadd_s
#define LM_GGML_F32Cx4_MUL         __lsx_vfmul_s
#define LM_GGML_F32Cx4_REDUCE      LM_GGML_F32x4_REDUCE

#define LM_GGML_F16_VEC                 LM_GGML_F32Cx4
#define LM_GGML_F16_VEC_ZERO            LM_GGML_F32Cx4_ZERO
#define LM_GGML_F16_VEC_SET1            LM_GGML_F32Cx4_SET1
#define LM_GGML_F16_VEC_LOAD(p, i)      LM_GGML_F32Cx4_LOAD(p)
#define LM_GGML_F16_VEC_STORE(p, r, i)  LM_GGML_F32Cx4_STORE(p, r[i])
#define LM_GGML_F16_VEC_FMA             LM_GGML_F32Cx4_FMA
#define LM_GGML_F16_VEC_ADD             LM_GGML_F32Cx4_ADD
#define LM_GGML_F16_VEC_MUL             LM_GGML_F32Cx4_MUL
#define LM_GGML_F16_VEC_REDUCE          LM_GGML_F32Cx4_REDUCE

#endif

// LM_GGML_F32_ARR / LM_GGML_F16_ARR
//   number of registers to use per step
#ifdef LM_GGML_SIMD
#define LM_GGML_F32_ARR (LM_GGML_F32_STEP/LM_GGML_F32_EPR)
#define LM_GGML_F16_ARR (LM_GGML_F16_STEP/LM_GGML_F16_EPR)
#endif

//
// ggml context
//

struct lm_ggml_context {
    size_t mem_size;
    void* mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    bool   no_alloc_save; // this is used to save the no_alloc state when using scratch buffers

    int    n_objects;

    struct lm_ggml_object * objects_begin;
    struct lm_ggml_object * objects_end;

    struct lm_ggml_scratch scratch;
    struct lm_ggml_scratch scratch_save;
};

struct lm_ggml_context_container {
    bool used;

    struct lm_ggml_context context;
};

struct lm_ggml_compute_state_shared {
    const struct lm_ggml_cgraph * cgraph;
    const struct lm_ggml_cplan * cplan;

    int n_threads;

    // synchronization primitives
    atomic_int n_barrier;
    atomic_int n_barrier_passed;

    lm_ggml_abort_callback abort_callback; // abort lm_ggml_graph_compute when true
    void * abort_callback_data;

    atomic_int current_chunk; // currently processing chunk during mul_mat, shared between all the threads

    enum lm_ggml_status ec;
};

struct lm_ggml_compute_state {
    lm_ggml_thread_t thrd;
    int ith;
    struct lm_ggml_compute_state_shared * shared;
};

struct lm_ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct lm_ggml_compute_state_shared * shared;
};

//
// fundamental operations
//

inline static void lm_ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_set_i32(const int n, int32_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_set_f16(const int n, lm_ggml_fp16_t * x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_set_bf16(const int n, lm_ggml_bf16_t * x, const lm_ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void lm_ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
inline static void lm_ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void lm_ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void lm_ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void lm_ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void lm_ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void lm_ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void lm_ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void lm_ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void lm_ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }

static void lm_ggml_vec_dot_f32(int n, float * restrict s, size_t bs, const float * restrict x, size_t bx, const float * restrict y, size_t by, int nrc) {
   assert(nrc == 1);
   UNUSED(nrc);
   UNUSED(bx);
   UNUSED(by);
   UNUSED(bs);

#if defined(LM_GGML_SIMD)
    float sumf = 0.0f;
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC sum[LM_GGML_F32_ARR] = { LM_GGML_F32_VEC_ZERO };

    LM_GGML_F32_VEC ax[LM_GGML_F32_ARR];
    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ax[j] = LM_GGML_F32_VEC_LOAD(x + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);

            sum[j] = LM_GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    LM_GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    lm_ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (lm_ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

static void lm_ggml_vec_dot_bf16(int n, float * restrict s, size_t bs, lm_ggml_bf16_t * restrict x, size_t bx, lm_ggml_bf16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    int i = 0;
    lm_ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (lm_ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (lm_ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (lm_ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (lm_ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (lm_ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (lm_ggml_float)(LM_GGML_BF16_TO_FP32(x[i]) *
                             LM_GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

static void lm_ggml_vec_dot_f16(int n, float * restrict s, size_t bs, lm_ggml_fp16_t * restrict x, size_t bx, lm_ggml_fp16_t * restrict y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    lm_ggml_float sumf = 0.0;

#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC sum[LM_GGML_F16_ARR] = { LM_GGML_F16_VEC_ZERO };

    LM_GGML_F16_VEC ax[LM_GGML_F16_ARR];
    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ax[j] = LM_GGML_F16_VEC_LOAD(x + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);

            sum[j] = LM_GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    LM_GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[i])*LM_GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[i])*LM_GGML_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

// compute LM_GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void lm_ggml_vec_dot_f16_unroll(const int n, const int xs, float * restrict s, void * restrict xv, lm_ggml_fp16_t * restrict y) {
    lm_ggml_float sumf[LM_GGML_VEC_DOT_UNROLL] = { 0.0 };

    lm_ggml_fp16_t * restrict x[LM_GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < LM_GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (lm_ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC sum[LM_GGML_VEC_DOT_UNROLL][LM_GGML_F16_ARR] = { { LM_GGML_F16_VEC_ZERO } };

    LM_GGML_F16_VEC ax[LM_GGML_F16_ARR];
    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);

            for (int k = 0; k < LM_GGML_VEC_DOT_UNROLL; ++k) {
                ax[j] = LM_GGML_F16_VEC_LOAD(x[k] + i + j*LM_GGML_F16_EPR, j);

                sum[k][j] = LM_GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
            }
        }
    }

    // reduce sum0..sum3 to sum0
    for (int k = 0; k < LM_GGML_VEC_DOT_UNROLL; ++k) {
        LM_GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < LM_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[j][i])*LM_GGML_FP16_TO_FP32(y[i]));
        }
    }
#else
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < LM_GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (lm_ggml_float)(LM_GGML_FP16_TO_FP32(x[j][i])*LM_GGML_FP16_TO_FP32(y[i]));
        }
    }
#endif

    for (int i = 0; i < LM_GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = sumf[i];
    }
}

inline static void lm_ggml_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx = LM_GGML_F32_VEC_SET1(v);

    LM_GGML_F32_VEC ax[LM_GGML_F32_ARR];
    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ax[j] = LM_GGML_F32_VEC_LOAD(x + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_FMA(ay[j], ax[j], vx);

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] += x[i]*v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void lm_ggml_vec_mad_f16(const int n, lm_ggml_fp16_t * restrict y, const lm_ggml_fp16_t * restrict x, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC vx = LM_GGML_F16_VEC_SET1(v);

    LM_GGML_F16_VEC ax[LM_GGML_F16_ARR];
    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ax[j] = LM_GGML_F16_VEC_LOAD(x + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            LM_GGML_F16_VEC_STORE(y + i + j*LM_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i]) + LM_GGML_FP16_TO_FP32(x[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i]) + LM_GGML_FP16_TO_FP32(x[i])*v);
    }
#endif
}

// xs and vs are byte strides of x and v
inline static void lm_ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * restrict y, const float * restrict xv, const float * restrict vv) {

    const float * restrict x[LM_GGML_VEC_MAD_UNROLL];
    const float * restrict v[LM_GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < LM_GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx[LM_GGML_VEC_MAD_UNROLL];

    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        vx[k] = LM_GGML_F32_VEC_SET1(v[k][0]);
    }

    LM_GGML_F32_VEC ax[LM_GGML_VEC_MAD_UNROLL][LM_GGML_F32_ARR];
    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);

            for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
                ax[k][j] = LM_GGML_F32_VEC_LOAD(x[k] + i + j*LM_GGML_F32_EPR);
                ay[j] = LM_GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
            }

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = np; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#else
    // scalar
    for (int k = 0; k < LM_GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

//inline static void lm_ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void lm_ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(LM_GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F32_STEP - 1));

    LM_GGML_F32_VEC vx = LM_GGML_F32_VEC_SET1(v);

    LM_GGML_F32_VEC ay[LM_GGML_F32_ARR];

    for (int i = 0; i < np; i += LM_GGML_F32_STEP) {
        for (int j = 0; j < LM_GGML_F32_ARR; j++) {
            ay[j] = LM_GGML_F32_VEC_LOAD(y + i + j*LM_GGML_F32_EPR);
            ay[j] = LM_GGML_F32_VEC_MUL(ay[j], vx);

            LM_GGML_F32_VEC_STORE(y + i + j*LM_GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void lm_ggml_vec_scale_f16(const int n, lm_ggml_fp16_t * y, const float v) {
#if defined(LM_GGML_SIMD)
    const int np = (n & ~(LM_GGML_F16_STEP - 1));

    LM_GGML_F16_VEC vx = LM_GGML_F16_VEC_SET1(v);

    LM_GGML_F16_VEC ay[LM_GGML_F16_ARR];

    for (int i = 0; i < np; i += LM_GGML_F16_STEP) {
        for (int j = 0; j < LM_GGML_F16_ARR; j++) {
            ay[j] = LM_GGML_F16_VEC_LOAD(y + i + j*LM_GGML_F16_EPR, j);
            ay[j] = LM_GGML_F16_VEC_MUL(ay[j], vx);

            LM_GGML_F16_VEC_STORE(y + i + j*LM_GGML_F16_EPR, ay, j);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i])*v);
    }
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(y[i])*v);
    }
#endif
}

inline static void lm_ggml_vec_norm_f32 (const int n, float * s, const float * x) { lm_ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void lm_ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void lm_ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void lm_ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);   }
inline static void lm_ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void lm_ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void lm_ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void lm_ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void lm_ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void lm_ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void lm_ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void lm_ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
// TODO: optimize performance
inline static void lm_ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void lm_ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

inline static float lm_ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void lm_ggml_vec_gelu_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_table_gelu_f16[i16[i]];
    }
}

#ifdef LM_GGML_GELU_FP16
inline static void lm_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            lm_ggml_fp16_t fp16 = LM_GGML_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = LM_GGML_FP16_TO_FP32(lm_ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void lm_ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_gelu_f32(x[i]);
    }
}
#endif

inline static float lm_ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

//inline static void lm_ggml_vec_gelu_quick_f16(const int n, lm_ggml_fp16_t * y, const lm_ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = lm_ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

#ifdef LM_GGML_GELU_QUICK_FP16
inline static void lm_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        lm_ggml_fp16_t fp16 = LM_GGML_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = LM_GGML_FP16_TO_FP32(lm_ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void lm_ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = lm_ggml_gelu_quick_f32(x[i]);
    }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float lm_ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggerganov/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t lm_ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t lm_ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = lm_ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 lm_ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 lm_ggml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 lm_ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 lm_ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 lm_ggml_v_expf(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c))
        return MADD128(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 lm_ggml_v_silu(__m128 x) {
    const __m128 one = _mm_set1_ps(1);
    const __m128 zero = _mm_setzero_ps();
    const __m128 neg_x = _mm_sub_ps(zero, x);
    const __m128 exp_neg_x = lm_ggml_v_expf(neg_x);
    const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    return _mm_div_ps(x, one_plus_exp_neg_x);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__

static void lm_ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, lm_ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, lm_ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, lm_ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, lm_ggml_v_silu(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = lm_ggml_silu_f32(x[i]);
    }
}

static lm_ggml_float lm_ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    lm_ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = lm_ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (lm_ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = lm_ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (lm_ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = lm_ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (lm_ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = lm_ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (lm_ggml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (lm_ggml_float)val;
        y[i] = val;
    }
    return sum;
}

inline static float lm_ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static void lm_ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = lm_ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void lm_ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef LM_GGML_USE_ACCELERATE
    lm_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (lm_ggml_float)x[i];
    }
    *s = sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void lm_ggml_vec_sum_f32_ggf(const int n, lm_ggml_float * s, const float * x) {
    lm_ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (lm_ggml_float)x[i];
    }
    *s = sum;
}

inline static void lm_ggml_vec_sum_f16_ggf(const int n, float * s, const lm_ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += LM_GGML_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void lm_ggml_vec_sum_bf16_ggf(const int n, float * s, const lm_ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += LM_GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void lm_ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef LM_GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void lm_ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    lm_ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void lm_ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

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
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
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
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
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
};

static_assert(LM_GGML_OP_COUNT == 74, "LM_GGML_OP_COUNT != 74");

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
    "Σx",
    "Σx_k",
    "Σx/n",
    "argmax(x)",
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
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
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
};

static_assert(LM_GGML_OP_COUNT == 74, "LM_GGML_OP_COUNT != 74");

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
};

static_assert(LM_GGML_UNARY_OP_COUNT == 13, "LM_GGML_UNARY_OP_COUNT != 13");


static_assert(sizeof(struct lm_ggml_object)%LM_GGML_MEM_ALIGN == 0, "lm_ggml_object size must be a multiple of LM_GGML_MEM_ALIGN");
static_assert(sizeof(struct lm_ggml_tensor)%LM_GGML_MEM_ALIGN == 0, "lm_ggml_tensor size must be a multiple of LM_GGML_MEM_ALIGN");

//
// NUMA support
//

#define LM_GGML_NUMA_MAX_NODES 8
#define LM_GGML_NUMA_MAX_CPUS 512

struct lm_ggml_numa_node {
    uint32_t cpus[LM_GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct lm_ggml_numa_nodes {
    enum lm_ggml_numa_strategy numa_strategy;
    struct lm_ggml_numa_node nodes[LM_GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct lm_ggml_state {
    struct lm_ggml_context_container contexts[LM_GGML_MAX_CONTEXTS];
    struct lm_ggml_numa_nodes numa;
};

// global state
static struct lm_ggml_state g_state;
static atomic_flag g_state_critical = ATOMIC_FLAG_INIT;

// critical section via spin lock
inline static void lm_ggml_critical_section_start(void) {
    while (atomic_flag_test_and_set(&g_state_critical)) {
        // spin
        sched_yield();
    }
}

#ifdef LM_GGML_USE_OPENMP
static void lm_ggml_barrier(struct lm_ggml_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }

    #pragma omp barrier
}
#else
static void lm_ggml_barrier(struct lm_ggml_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }

    atomic_int * n_barrier = &shared->n_barrier;
    atomic_int * n_barrier_passed = &shared->n_barrier_passed;

    int n_threads = shared->n_threads;
    int passed_old = atomic_load(n_barrier_passed);

    if (atomic_fetch_add(n_barrier, 1) == n_threads - 1) {
        // last thread
        atomic_store(n_barrier, 0);
        atomic_fetch_add(n_barrier_passed, 1);
    } else {
        // wait for other threads
        const int n_spin_before_sleep = 100000;
        while (true) {
            for (int i = 0; i < n_spin_before_sleep; i++) {
                if (atomic_load(n_barrier_passed) != passed_old) {
                    return;
                }
            #if defined(__SSE3__)
                _mm_pause();
            #endif
            }
            sched_yield();
        }
    }
}
#endif

// TODO: make this somehow automatically executed
//       some sort of "sentry" mechanism
inline static void lm_ggml_critical_section_end(void) {
    atomic_flag_clear(&g_state_critical);
}

#if defined(__gnu_linux__)
static cpu_set_t lm_ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t lm_ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void lm_ggml_numa_init(enum lm_ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "lm_ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    LM_GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = lm_ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < LM_GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < LM_GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    LM_GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 28) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    LM_GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct lm_ggml_numa_node * node = &g_state.numa.nodes[n];
        LM_GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            LM_GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                LM_GGML_PRINT_DEBUG(" %u", c);
            }
        }
        LM_GGML_PRINT_DEBUG("\n");
    }

    if (lm_ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                LM_GGML_PRINT("WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool lm_ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_print_object(const struct lm_ggml_object * obj) {
    LM_GGML_PRINT(" - lm_ggml_object: type = %d, offset = %zu, size = %zu, next = %p\n",
            obj->type, obj->offs, obj->size, (const void *) obj->next);
}

void lm_ggml_print_objects(const struct lm_ggml_context * ctx) {
    struct lm_ggml_object * obj = ctx->objects_begin;

    LM_GGML_PRINT("%s: objects in context %p:\n", __func__, (const void *) ctx);

    while (obj != NULL) {
        lm_ggml_print_object(obj);
        obj = obj->next;
    }

    LM_GGML_PRINT("%s: --- end ---\n", __func__);
}

LM_GGML_CALL int64_t lm_ggml_nelements(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

LM_GGML_CALL int64_t lm_ggml_nrows(const struct lm_ggml_tensor * tensor) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

LM_GGML_CALL size_t lm_ggml_nbytes(const struct lm_ggml_tensor * tensor) {
    size_t nbytes;
    size_t blck_size = lm_ggml_blck_size(tensor->type);
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

LM_GGML_CALL int64_t lm_ggml_blck_size(enum lm_ggml_type type) {
    return type_traits[type].blck_size;
}

LM_GGML_CALL size_t lm_ggml_type_size(enum lm_ggml_type type) {
    return type_traits[type].type_size;
}

LM_GGML_CALL size_t lm_ggml_row_size(enum lm_ggml_type type, int64_t ne) {
    assert(ne % lm_ggml_blck_size(type) == 0);
    return lm_ggml_type_size(type)*ne/lm_ggml_blck_size(type);
}

double lm_ggml_type_sizef(enum lm_ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

LM_GGML_CALL const char * lm_ggml_type_name(enum lm_ggml_type type) {
    return type_traits[type].type_name;
}

LM_GGML_CALL bool lm_ggml_is_quantized(enum lm_ggml_type type) {
    return type_traits[type].is_quantized;
}

LM_GGML_CALL const char * lm_ggml_op_name(enum lm_ggml_op op) {
    return LM_GGML_OP_NAME[op];
}

const char * lm_ggml_op_symbol(enum lm_ggml_op op) {
    return LM_GGML_OP_SYMBOL[op];
}

const char * lm_ggml_unary_op_name(enum lm_ggml_unary_op op) {
    return LM_GGML_UNARY_OP_NAME[op];
}

LM_GGML_CALL const char * lm_ggml_op_desc(const struct lm_ggml_tensor * t) {
    if (t->op == LM_GGML_OP_UNARY) {
        enum lm_ggml_unary_op uop = lm_ggml_get_unary_op(t);
        return lm_ggml_unary_op_name(uop);
    }
    return lm_ggml_op_name(t->op);
}

LM_GGML_CALL size_t lm_ggml_element_size(const struct lm_ggml_tensor * tensor) {
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

static inline bool lm_ggml_can_mul_mat(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[0]           == t1->ne[0])  &&
           (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
}

static inline bool lm_ggml_can_out_prod(const struct lm_ggml_tensor * t0, const struct lm_ggml_tensor * t1) {
    static_assert(LM_GGML_MAX_DIMS == 4, "LM_GGML_MAX_DIMS is not 4 - update this function");

    return (t0->ne[1] == t1->ne[1])   &&
           (t1->ne[2]%t0->ne[2] == 0) && // verify t0 is broadcastable
           (t1->ne[3]%t0->ne[3] == 0);
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

LM_GGML_CALL bool lm_ggml_is_transposed(const struct lm_ggml_tensor * tensor) {
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

LM_GGML_CALL bool lm_ggml_is_contiguous(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_0(tensor);
}

LM_GGML_CALL bool lm_ggml_is_contiguous_0(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 0);
}

LM_GGML_CALL bool lm_ggml_is_contiguous_1(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 1);
}

LM_GGML_CALL bool lm_ggml_is_contiguous_2(const struct lm_ggml_tensor * tensor) {
    return lm_ggml_is_contiguous_n(tensor, 2);
}

LM_GGML_CALL bool lm_ggml_is_permuted(const struct lm_ggml_tensor * tensor) {
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

LM_GGML_CALL bool lm_ggml_is_empty(const struct lm_ggml_tensor * tensor) {
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

static inline int lm_ggml_up32(int n) {
    return (n + 31) & ~31;
}

//static inline int lm_ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

static inline int lm_ggml_up(int n, int m) {
    // assert m is a power of 2
    LM_GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

// assert that pointer is aligned to LM_GGML_MEM_ALIGN
#define LM_GGML_ASSERT_ALIGNED(ptr) \
    LM_GGML_ASSERT(((uintptr_t) (ptr))%LM_GGML_MEM_ALIGN == 0)

////////////////////////////////////////////////////////////////////////////////

struct lm_ggml_context * lm_ggml_init(struct lm_ggml_init_params params) {
    // make this function thread safe
    lm_ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize time system (required on Windows)
        lm_ggml_time_init();

        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = lm_ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    lm_ggml_fp16_t fp16;
                } u = {i};
                float f = lm_ggml_table_f32_f16[i] = LM_GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                lm_ggml_table_gelu_f16[i] = LM_GGML_FP32_TO_FP16(lm_ggml_gelu_f32(f));
                lm_ggml_table_gelu_quick_f16[i] = LM_GGML_FP32_TO_FP16(lm_ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = lm_ggml_time_us(); UNUSED(t_end);

            LM_GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        // initialize g_state
        {
            const uint64_t t_start = lm_ggml_time_us(); UNUSED(t_start);

            g_state = (struct lm_ggml_state) {
                /*.contexts =*/ { { 0 } },
                /*.numa =*/ {
                    .n_nodes = 0,
                    .total_cpus = 0,
                },
            };

            for (int i = 0; i < LM_GGML_MAX_CONTEXTS; ++i) {
                g_state.contexts[i].used = false;
            }

            const uint64_t t_end = lm_ggml_time_us(); UNUSED(t_end);

            LM_GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
        }

        is_first_call = false;
    }

    // find non-used context in g_state
    struct lm_ggml_context * ctx = NULL;

    for (int i = 0; i < LM_GGML_MAX_CONTEXTS; i++) {
        if (!g_state.contexts[i].used) {
            g_state.contexts[i].used = true;
            ctx = &g_state.contexts[i].context;

            LM_GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
            break;
        }
    }

    if (ctx == NULL) {
        LM_GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

        lm_ggml_critical_section_end();

        return NULL;
    }

    // allow to call lm_ggml_init with 0 size
    if (params.mem_size == 0) {
        params.mem_size = LM_GGML_MEM_ALIGN;
    }

    const size_t mem_size = params.mem_buffer ? params.mem_size : LM_GGML_PAD(params.mem_size, LM_GGML_MEM_ALIGN);

    *ctx = (struct lm_ggml_context) {
        /*.mem_size           =*/ mem_size,
        /*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : LM_GGML_ALIGNED_MALLOC(mem_size),
        /*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
        /*.no_alloc           =*/ params.no_alloc,
        /*.no_alloc_save      =*/ params.no_alloc,
        /*.n_objects          =*/ 0,
        /*.objects_begin      =*/ NULL,
        /*.objects_end        =*/ NULL,
        /*.scratch            =*/ { 0, 0, NULL, },
        /*.scratch_save       =*/ { 0, 0, NULL, },
    };

    LM_GGML_ASSERT(ctx->mem_buffer != NULL);

    LM_GGML_ASSERT_ALIGNED(ctx->mem_buffer);

#if defined(__ARM_FEATURE_SVE)
    if (!lm_ggml_sve_cnt_b) {
        lm_ggml_sve_cnt_b = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
    }
#endif

    LM_GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

    lm_ggml_critical_section_end();

    return ctx;
}

void lm_ggml_free(struct lm_ggml_context * ctx) {
    if (ctx == NULL) {
        return;
    }

    // make this function thread safe
    lm_ggml_critical_section_start();

    bool found = false;

    for (int i = 0; i < LM_GGML_MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            LM_GGML_PRINT_DEBUG("%s: context %d has been freed. memory used = %zu\n",
                    __func__, i, lm_ggml_used_mem(ctx));

            if (ctx->mem_buffer_owned) {
                LM_GGML_ALIGNED_FREE(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        LM_GGML_PRINT_DEBUG("%s: context not found\n", __func__);
    }

    lm_ggml_critical_section_end();
}

size_t lm_ggml_used_mem(const struct lm_ggml_context * ctx) {
    return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

size_t lm_ggml_set_scratch(struct lm_ggml_context * ctx, struct lm_ggml_scratch scratch) {
    const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

    ctx->scratch = scratch;

    return result;
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

// IMPORTANT:
// when creating "opt" tensors, always save and load the scratch buffer
// this is an error prone process, but it is necessary to support inplace
// operators when using scratch buffers
// TODO: implement a better way
static void lm_ggml_scratch_save(struct lm_ggml_context * ctx) {
    // this is needed to allow opt tensors to store their data
    // TODO: again, need to find a better way
    ctx->no_alloc_save = ctx->no_alloc;
    ctx->no_alloc      = false;

    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;
}

static void lm_ggml_scratch_load(struct lm_ggml_context * ctx) {
    ctx->no_alloc = ctx->no_alloc_save;

    ctx->scratch = ctx->scratch_save;
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
        LM_GGML_PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                __func__, cur_end + size_needed, ctx->mem_size);
        assert(false);
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

    assert(n_dims >= 1 && n_dims <= LM_GGML_MAX_DIMS);

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
        if (ctx->scratch.data != NULL) {
            // allocate tensor data in the scratch buffer
            if (ctx->scratch.offs + data_size > ctx->scratch.size) {
                LM_GGML_PRINT("%s: not enough space in the scratch memory pool (needed %zu, available %zu)\n",
                        __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
                assert(false);
                return NULL;
            }

            data = (char * const) ctx->scratch.data + ctx->scratch.offs;

            ctx->scratch.offs += data_size;
        } else {
            // allocate tensor data in the context's memory pool
            obj_alloc_size = data_size;
        }
    }

    struct lm_ggml_object * const obj_new = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_TENSOR, LM_GGML_TENSOR_SIZE + obj_alloc_size);

    // TODO: for recoverable errors, we would need to free the data allocated from the scratch buffer here

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

struct lm_ggml_tensor * lm_ggml_new_i32(struct lm_ggml_context * ctx, int32_t value) {
    lm_ggml_scratch_save(ctx);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, 1);

    lm_ggml_scratch_load(ctx);

    lm_ggml_set_i32(result, value);

    return result;
}

struct lm_ggml_tensor * lm_ggml_new_f32(struct lm_ggml_context * ctx, float value) {
    lm_ggml_scratch_save(ctx);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, 1);

    lm_ggml_scratch_load(ctx);

    lm_ggml_set_f32(result, value);

    return result;
}

struct lm_ggml_tensor * lm_ggml_dup_tensor(struct lm_ggml_context * ctx, const struct lm_ggml_tensor * src) {
    return lm_ggml_new_tensor(ctx, src->type, LM_GGML_MAX_DIMS, src->ne);
}

static void lm_ggml_set_op_params(struct lm_ggml_tensor * tensor, const void * params, size_t params_size) {
    LM_GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= LM_GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t lm_ggml_get_op_params_i32(const struct lm_ggml_tensor * tensor, uint32_t i) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((const int32_t *)(tensor->op_params))[i];
}

static float lm_ggml_get_op_params_f32(const struct lm_ggml_tensor * tensor, uint32_t i) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}

static void lm_ggml_set_op_params_i32(struct lm_ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void lm_ggml_set_op_params_f32(struct lm_ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < LM_GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}

struct lm_ggml_tensor * lm_ggml_set_zero(struct lm_ggml_tensor * tensor) {
    memset(tensor->data, 0, lm_ggml_nbytes(tensor));
    return tensor;
}

struct lm_ggml_tensor * lm_ggml_set_i32 (struct lm_ggml_tensor * tensor, int32_t value) {
    const int n     = lm_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f16(nc, (lm_ggml_fp16_t *)(data + i*n1), LM_GGML_FP32_TO_FP16(value));
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_bf16(nc, (lm_ggml_bf16_t *)(data + i*n1), LM_GGML_FP32_TO_BF16(value));
                }
            } break;
        case LM_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct lm_ggml_tensor * lm_ggml_set_f32(struct lm_ggml_tensor * tensor, float value) {
    const int n     = lm_ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case LM_GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f16(nc, (lm_ggml_fp16_t *)(data + i*n1), LM_GGML_FP32_TO_FP16(value));
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_bf16(nc, (lm_ggml_bf16_t *)(data + i*n1), LM_GGML_FP32_TO_BF16(value));
                }
            } break;
        case LM_GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    lm_ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    return tensor;
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

int32_t lm_ggml_get_i32_1d(const struct lm_ggml_tensor * tensor, int i) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return lm_ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_F16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                return LM_GGML_FP16_TO_FP32(((lm_ggml_fp16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_BF16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_F32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

void lm_ggml_set_i32_1d(const struct lm_ggml_tensor * tensor, int i, int32_t value) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        lm_ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_fp16_t));
                ((lm_ggml_fp16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(lm_ggml_bf16_t));
                ((lm_ggml_bf16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                LM_GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

int32_t lm_ggml_get_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case LM_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case LM_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case LM_GGML_TYPE_F16:
            return LM_GGML_FP16_TO_FP32(((lm_ggml_fp16_t *) data)[0]);
        case LM_GGML_TYPE_BF16:
            return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *) data)[0]);
        case LM_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            LM_GGML_ABORT("fatal error");
    }
}

void lm_ggml_set_i32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(data))[0] = LM_GGML_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(data))[0] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

float lm_ggml_get_f32_1d(const struct lm_ggml_tensor * tensor, int i) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return lm_ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case LM_GGML_TYPE_F16:
            {
                return LM_GGML_FP16_TO_FP32(((lm_ggml_fp16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_BF16:
            {
                return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *)(tensor->data))[i]);
            }
        case LM_GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

void lm_ggml_set_f32_1d(const struct lm_ggml_tensor * tensor, int i, float value) {
    if (!lm_ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        lm_ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        lm_ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(tensor->data))[i] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

float lm_ggml_get_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case LM_GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case LM_GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case LM_GGML_TYPE_F16:
            return LM_GGML_FP16_TO_FP32(((lm_ggml_fp16_t *) data)[0]);
        case LM_GGML_TYPE_BF16:
            return LM_GGML_BF16_TO_FP32(((lm_ggml_bf16_t *) data)[0]);
        case LM_GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            LM_GGML_ABORT("fatal error");
    }
}

void lm_ggml_set_f32_nd(const struct lm_ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case LM_GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case LM_GGML_TYPE_F16:
            {
                ((lm_ggml_fp16_t *)(data))[0] = LM_GGML_FP32_TO_FP16(value);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                ((lm_ggml_bf16_t *)(data))[0] = LM_GGML_FP32_TO_BF16(value);
            } break;
        case LM_GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

void * lm_ggml_get_data(const struct lm_ggml_tensor * tensor) {
    return tensor->data;
}

float * lm_ggml_get_data_f32(const struct lm_ggml_tensor * tensor) {
    assert(tensor->type == LM_GGML_TYPE_F32);
    return (float *)(tensor->data);
}

LM_GGML_CALL enum lm_ggml_unary_op lm_ggml_get_unary_op(const struct lm_ggml_tensor * tensor) {
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
        struct lm_ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_DUP;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_dup(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    return lm_ggml_dup_impl(ctx, a, false);
}

struct lm_ggml_tensor * lm_ggml_dup_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    return lm_ggml_dup_impl(ctx, a, true);
}

// lm_ggml_add

static struct lm_ggml_tensor * lm_ggml_add_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        // TODO: support backward pass for broadcasting
        LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_ADD;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_add_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_add_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_add_impl(ctx, a, b, true);
}

// lm_ggml_add_cast

static struct lm_ggml_tensor * lm_ggml_add_cast_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        enum   lm_ggml_type     type) {
    // TODO: support less-strict constraint
    //       LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));
    LM_GGML_ASSERT(lm_ggml_can_repeat_rows(b, a));

    // currently only supported for quantized input and f16
    LM_GGML_ASSERT(lm_ggml_is_quantized(a->type) ||
                a->type == LM_GGML_TYPE_F16 ||
                a->type == LM_GGML_TYPE_BF16);

    bool is_node = false;

    if (a->grad || b->grad) {
        // TODO: support backward pass for broadcasting
        LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, type, LM_GGML_MAX_DIMS, a->ne);

    result->op   = LM_GGML_OP_ADD;
    result->grad = is_node ? lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, LM_GGML_MAX_DIMS, a->ne) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add_cast(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        enum   lm_ggml_type     type) {
    return lm_ggml_add_cast_impl(ctx, a, b, type);
}

// lm_ggml_add1

static struct lm_ggml_tensor * lm_ggml_add1_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_is_scalar(b));
    LM_GGML_ASSERT(lm_ggml_is_padded_1d(a));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_ADD1;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_add1(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_add1_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_add1_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_add1_impl(ctx, a, b, true);
}

// lm_ggml_acc

static struct lm_ggml_tensor * lm_ggml_acc_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        size_t               nb1,
        size_t               nb2,
        size_t               nb3,
        size_t               offset,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_nelements(b) <= lm_ggml_nelements(a));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(a->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(b->type == LM_GGML_TYPE_F32);

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_ACC;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_acc(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        size_t               nb1,
        size_t               nb2,
        size_t               nb3,
        size_t               offset) {
    return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct lm_ggml_tensor * lm_ggml_acc_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        size_t               nb1,
        size_t               nb2,
        size_t               nb3,
        size_t               offset) {
    return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

// lm_ggml_sub

static struct lm_ggml_tensor * lm_ggml_sub_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_SUB;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_sub(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_sub_impl(ctx, a, b, false);
}

struct lm_ggml_tensor * lm_ggml_sub_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    return lm_ggml_sub_impl(ctx, a, b, true);
}

// lm_ggml_mul

static struct lm_ggml_tensor * lm_ggml_mul_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        // TODO: support backward pass for broadcasting
        LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
        is_node = true;
    }

    if (inplace) {
        LM_GGML_ASSERT(!is_node);
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_MUL;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        LM_GGML_ASSERT(!is_node);
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_DIV;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        struct lm_ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_SQR;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        struct lm_ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_SQRT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_LOG;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

// lm_ggml_sum

struct lm_ggml_tensor * lm_ggml_sum(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op   = LM_GGML_OP_SUM;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_sum_rows

struct lm_ggml_tensor * lm_ggml_sum_rows(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    int64_t ne[LM_GGML_MAX_DIMS] = { 1 };
    for (int i = 1; i < LM_GGML_MAX_DIMS; ++i) {
        ne[i] = a->ne[i];
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, ne);

    result->op   = LM_GGML_OP_SUM_ROWS;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_mean

struct lm_ggml_tensor * lm_ggml_mean(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement
        is_node = true;
    }

    int64_t ne[4] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op   = LM_GGML_OP_MEAN;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_argmax

struct lm_ggml_tensor * lm_ggml_argmax(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a) {
    LM_GGML_ASSERT(lm_ggml_is_matrix(a));
    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error");
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_I32, a->ne[1]);

    result->op   = LM_GGML_OP_ARGMAX;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_repeat

struct lm_ggml_tensor * lm_ggml_repeat(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(a, b));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, b->ne);

    result->op   = LM_GGML_OP_REPEAT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_repeat_back

struct lm_ggml_tensor * lm_ggml_repeat_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        struct lm_ggml_tensor * b) {
    LM_GGML_ASSERT(lm_ggml_can_repeat(b, a));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (lm_ggml_are_same_shape(a, b) && !is_node) {
        return a;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, b->ne);

    result->op   = LM_GGML_OP_REPEAT_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_concat

struct lm_ggml_tensor * lm_ggml_concat(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    struct lm_ggml_tensor * b,
    int dim) {
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

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, LM_GGML_MAX_DIMS, ne);

    lm_ggml_set_op_params_i32(result, 0, dim);

    result->op = LM_GGML_OP_CONCAT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        struct lm_ggml_tensor  * a, float negative_slope, bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &negative_slope, sizeof(negative_slope));

    result->op   = LM_GGML_OP_LEAKY_RELU;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
    bool is_node = false;

    if (a->grad || b->grad) {
        // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_SILU_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

// lm_ggml_norm

static struct lm_ggml_tensor * lm_ggml_norm_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float eps,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op   = LM_GGML_OP_NORM;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_norm(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float eps) {
    return lm_ggml_norm_impl(ctx, a, eps, false);
}

struct lm_ggml_tensor * lm_ggml_norm_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float eps) {
    return lm_ggml_norm_impl(ctx, a, eps, true);
}

// lm_ggml_rms_norm

static struct lm_ggml_tensor * lm_ggml_rms_norm_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float eps,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op   = LM_GGML_OP_RMS_NORM;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_rms_norm(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float  eps) {
    return lm_ggml_rms_norm_impl(ctx, a, eps, false);
}

struct lm_ggml_tensor * lm_ggml_rms_norm_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float eps) {
    return lm_ggml_rms_norm_impl(ctx, a, eps, true);
}

// lm_ggml_rms_norm_back

struct lm_ggml_tensor * lm_ggml_rms_norm_back(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b,
        float  eps) {
    bool is_node = false;

    if (a->grad) {
        // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &eps, sizeof(eps));

    result->op   = LM_GGML_OP_RMS_NORM_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_group_norm

static struct lm_ggml_tensor * lm_ggml_group_norm_impl(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int n_groups,
    bool inplace) {

    bool is_node = false;
    if (!inplace && (a->grad)) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op_params[0] = n_groups;

    result->op = LM_GGML_OP_GROUP_NORM;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_group_norm(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int n_groups) {
    return lm_ggml_group_norm_impl(ctx, a, n_groups, false);
}

struct lm_ggml_tensor * lm_ggml_group_norm_inplace(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int n_groups) {
    return lm_ggml_group_norm_impl(ctx, a, n_groups, true);
}

// lm_ggml_mul_mat

struct lm_ggml_tensor * lm_ggml_mul_mat(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_mul_mat(a, b));
    LM_GGML_ASSERT(!lm_ggml_is_transposed(a));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    const int64_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op   = LM_GGML_OP_MUL_MAT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (as->grad || b->grad) {
        is_node = true;
    }

    const int64_t ne[4] = { as->ne[1], ids->ne[0], b->ne[2], 1 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op   = LM_GGML_OP_MUL_MAT_ID;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = as;
    result->src[1] = b;
    result->src[2] = ids;

    return result;
}

// lm_ggml_out_prod

struct lm_ggml_tensor * lm_ggml_out_prod(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        struct lm_ggml_tensor  * b) {
    LM_GGML_ASSERT(lm_ggml_can_out_prod(a, b));
    LM_GGML_ASSERT(!lm_ggml_is_transposed(a));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    // a is broadcastable to b for ne[2] and ne[3] -> use b->ne[2] and b->ne[3]
    const int64_t ne[4] = { a->ne[0], b->ne[0], b->ne[2], b->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    result->op   = LM_GGML_OP_OUT_PROD;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_scale

static struct lm_ggml_tensor * lm_ggml_scale_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 s,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_is_padded_1d(a));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, &s, sizeof(s));

    result->op   = LM_GGML_OP_SCALE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_scale(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        float                s) {
    return lm_ggml_scale_impl(ctx, a, s, false);
}

struct lm_ggml_tensor * lm_ggml_scale_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        float                s) {
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
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_nelements(a) >= lm_ggml_nelements(b));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    // make a view of the destination
    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { nb1, nb2, nb3, offset, inplace ? 1 : 0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_SET;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_set(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
        size_t                nb1,
        size_t                nb2,
        size_t                nb3,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, nb2, nb3, offset, true);
}

struct lm_ggml_tensor * lm_ggml_set_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_1d_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], offset, true);
}

struct lm_ggml_tensor * lm_ggml_set_2d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
        size_t                nb1,
        size_t                offset) {
    return lm_ggml_set_impl(ctx, a, b, nb1, a->nb[2], a->nb[3], offset, false);
}

struct lm_ggml_tensor * lm_ggml_set_2d_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor *  a,
        struct lm_ggml_tensor *  b,
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

    bool is_node = false;

    if (a->grad || b->grad) {
        // inplace is false and either one have a grad
        is_node = true;
    }

    // make a view of the destination
    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, b);
    if (strlen(b->name) > 0) {
        lm_ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
    } else {
        lm_ggml_format_name(result, "%s (copy)", a->name);
    }

    result->op   = LM_GGML_OP_CPY;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
    bool is_node = false;

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, type, LM_GGML_MAX_DIMS, a->ne);
    lm_ggml_format_name(result, "%s (copy)", a->name);

    result->op   = LM_GGML_OP_CPY;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = result;

    return result;
}

// lm_ggml_cont

static struct lm_ggml_tensor * lm_ggml_cont_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);
    lm_ggml_format_name(result, "%s (cont)", a->name);

    result->op   = LM_GGML_OP_CONT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type, ne0, ne1, ne2, ne3);
    lm_ggml_format_name(result, "%s (cont)", a->name);

    result->op   = LM_GGML_OP_CONT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (b->grad) {
        // gradient propagation is not supported
        //LM_GGML_ABORT("fatal error");
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, LM_GGML_MAX_DIMS, b->ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op   = LM_GGML_OP_RESHAPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_reshape_1d(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int64_t               ne0) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(a));
    LM_GGML_ASSERT(lm_ggml_nelements(a) == ne0);

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int64_t ne[1] = { ne0 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op   = LM_GGML_OP_RESHAPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int64_t ne[2] = { ne0, ne1 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op   = LM_GGML_OP_RESHAPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int64_t ne[3] = { ne0, ne1, ne2 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op   = LM_GGML_OP_RESHAPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, 4, ne, a, 0);
    lm_ggml_format_name(result, "%s (reshaped)", a->name);

    result->op   = LM_GGML_OP_RESHAPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

static struct lm_ggml_tensor * lm_ggml_view_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_dims,
        const int64_t       * ne,
        size_t                offset) {

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
    lm_ggml_format_name(result, "%s (view)", a->name);

    lm_ggml_set_op_params(result, &offset, sizeof(offset));

    result->op   = LM_GGML_OP_VIEW;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

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

    result->op   = LM_GGML_OP_PERMUTE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    return result;
}

// lm_ggml_transpose

struct lm_ggml_tensor * lm_ggml_transpose(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);
    lm_ggml_format_name(result, "%s (transposed)", a->name);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op   = LM_GGML_OP_TRANSPOSE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    // TODO: implement non F32 return
    enum lm_ggml_type type = LM_GGML_TYPE_F32;
    if (a->type == LM_GGML_TYPE_I32) {
        type = a->type;
    }
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, type, a->ne[0], b->ne[0], b->ne[1], b->ne[2]);

    result->op   = LM_GGML_OP_GET_ROWS;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    // TODO: implement non F32 return
    //struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, c->ne[0], c->ne[1]);

    result->op   = LM_GGML_OP_GET_ROWS_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_diag

struct lm_ggml_tensor * lm_ggml_diag(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a) {
    LM_GGML_ASSERT(a->ne[1] == 1);
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    const int64_t ne[4] = { a->ne[0], a->ne[0], a->ne[2], a->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, a->type, 4, ne);

    result->op   = LM_GGML_OP_DIAG;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_diag_mask_inf

static struct lm_ggml_tensor * lm_ggml_diag_mask_inf_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   n_past,
        bool                  inplace) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_DIAG_MASK_INF;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    int32_t params[] = { n_past };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_DIAG_MASK_ZERO;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    float params[] = { scale, max_bias };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_SOFT_MAX;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true; // TODO : implement backward pass
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_SOFT_MAX_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        is_node = true;
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

    result->op   = LM_GGML_OP_ROPE;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
    LM_GGML_ASSERT(c == NULL && "freq factors not implemented yet");

    LM_GGML_ASSERT((mode & 4) == 0 && "lm_ggml_rope_back() for ChatGLM not implemented yet");

    bool is_node = false;

    if (a->grad) {
        is_node = false; // TODO: implement backward
    }

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    int32_t params[11] = { /*n_past*/ 0, n_dims, mode, /*n_ctx*/ 0, n_ctx_orig };
    memcpy(params +  5, &freq_base,    sizeof(float));
    memcpy(params +  6, &freq_scale,   sizeof(float));
    memcpy(params +  7, &ext_factor,   sizeof(float));
    memcpy(params +  8, &attn_factor,  sizeof(float));
    memcpy(params +  9, &beta_fast,    sizeof(float));
    memcpy(params + 10, &beta_slow,    sizeof(float));
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_ROPE_BACK;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_clamp

struct lm_ggml_tensor * lm_ggml_clamp(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        float                 min,
        float                 max) {
    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    struct lm_ggml_tensor * result = lm_ggml_view_tensor(ctx, a);

    float params[] = { min, max };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_CLAMP;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad || b->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = {
        lm_ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
        a->ne[1], b->ne[2], 1,
    };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { s0, p0, d0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op = LM_GGML_OP_CONV_TRANSPOSE_1D;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_conv_depthwise
struct lm_ggml_tensor * lm_ggml_conv_depthwise_2d(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    struct lm_ggml_tensor * b,
    int                  s0,
    int                  s1,
    int                  p0,
    int                  p1,
    int                  d0,
    int                  d1) {

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
    int                  s0,
    int                  s1,
    int                  p0,
    int                  p1,
    int                  d0,
    int                  d1,
    bool                 is_2D,
    enum lm_ggml_type       dst_type) {

    if(is_2D) {
        LM_GGML_ASSERT(a->ne[2] == b->ne[2]);
    } else {
        LM_GGML_ASSERT(a->ne[1] == b->ne[1]);
    }
    bool is_node = false;

    if (a->grad || b->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t OH = is_2D ? lm_ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const int64_t OW =         lm_ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    const int64_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, dst_type, 4, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op = LM_GGML_OP_IM2COL;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        int                  s0,
        int                  s1,
        int                  p0,
        int                  p1,
        int                  d0,
        int                  d1) {
    struct lm_ggml_tensor * im2col = lm_ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, LM_GGML_TYPE_F16); // [N, OH, OW, IC * KH * KW]

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

    bool is_node = false;

    if (a->grad || b->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = {
        lm_ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
        lm_ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
        a->ne[2], b->ne[3],
    };

    struct lm_ggml_tensor* result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    lm_ggml_set_op_params_i32(result, 0, stride);

    result->op = LM_GGML_OP_CONV_TRANSPOSE_2D;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = {
        lm_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        a->ne[1],
        a->ne[2],
        a->ne[3],
    };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    int32_t params[] = { op, k0, s0, p0 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op = LM_GGML_OP_POOL_1D;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result;
    const int64_t ne[3] = {
        lm_ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        lm_ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
    };
    result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 3, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op = LM_GGML_OP_POOL_2D;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    return result;
}

// lm_ggml_upscale

static struct lm_ggml_tensor * lm_ggml_upscale_impl(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int ne0,
    int ne1,
    int ne2,
    int ne3) {
    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    LM_GGML_ASSERT(a->ne[0] <= ne0);
    LM_GGML_ASSERT(a->ne[1] <= ne1);
    LM_GGML_ASSERT(a->ne[2] <= ne2);
    LM_GGML_ASSERT(a->ne[3] <= ne3);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type,
            ne0,
            ne1,
            ne2,
            ne3
            );

    result->op = LM_GGML_OP_UPSCALE;

    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_upscale(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int scale_factor) {
    return lm_ggml_upscale_impl(ctx, a, a->ne[0] * scale_factor, a->ne[1] * scale_factor, a->ne[2], a->ne[3]);
}

struct lm_ggml_tensor * lm_ggml_upscale_ext(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor * a,
    int ne0,
    int ne1,
    int ne2,
    int ne3) {
    return lm_ggml_upscale_impl(ctx, a, ne0, ne1, ne2, ne3);
}

// lm_ggml_pad

struct lm_ggml_tensor * lm_ggml_pad(
    struct lm_ggml_context * ctx,
    struct lm_ggml_tensor  * a,
    int p0, int p1, int p2, int p3) {
    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_4d(ctx, a->type,
            a->ne[0] + p0,
            a->ne[1] + p1,
            a->ne[2] + p2,
            a->ne[3] + p3);

    result->op = LM_GGML_OP_PAD;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

// lm_ggml_arange

struct lm_ggml_tensor * lm_ggml_arange(
    struct lm_ggml_context * ctx,
    float start,
    float stop,
    float step) {

    LM_GGML_ASSERT(stop > start);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, steps);

    result->op = LM_GGML_OP_ARANGE;
    lm_ggml_set_op_params_f32(result, 0, start);
    lm_ggml_set_op_params_f32(result, 1, stop);
    lm_ggml_set_op_params_f32(result, 2, step);

    return result;
}

// lm_ggml_timestep_embedding

struct lm_ggml_tensor * lm_ggml_timestep_embedding(
            struct lm_ggml_context * ctx,
            struct lm_ggml_tensor  * timesteps,
            int                   dim,
            int                   max_period) {
    bool is_node = false;

    if (timesteps->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    int actual_dim = dim;
    if (dim % 2 != 0) {
        actual_dim = dim + 1;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_2d(ctx, LM_GGML_TYPE_F32, actual_dim, timesteps->ne[0]);

    result->op = LM_GGML_OP_TIMESTEP_EMBEDDING;
    lm_ggml_set_op_params_i32(result, 0, dim);
    lm_ggml_set_op_params_i32(result, 1, max_period);

    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = timesteps;

    return result;
}

// lm_ggml_argsort

struct lm_ggml_tensor * lm_ggml_argsort(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_sort_order  order) {
    bool is_node = false;

    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_I32, LM_GGML_MAX_DIMS, a->ne);

    lm_ggml_set_op_params_i32(result, 0, (int32_t) order);

    result->op   = LM_GGML_OP_ARGSORT;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        float                 max_bias) {
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

    if (q->grad || k->grad || v->grad) {
        is_node = true;
    }

    // permute(0, 2, 1, 3)
    int64_t ne[4] = { q->ne[0], q->ne[2], q->ne[1], q->ne[3] };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 4, ne);

    float params[] = { scale, max_bias };
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

    lm_ggml_set_op_params_i32(a, 2, prec_i32); // scale is on first pos, max_bias on second
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
        struct lm_ggml_tensor  * s,
        struct lm_ggml_tensor  * x,
        struct lm_ggml_tensor  * c,
        struct lm_ggml_tensor  * sq) {
    LM_GGML_ASSERT(lm_ggml_is_3d(s));
    LM_GGML_ASSERT(lm_ggml_is_matrix(x));
    LM_GGML_ASSERT(lm_ggml_is_matrix(c));
    LM_GGML_ASSERT(lm_ggml_is_matrix(sq));
    LM_GGML_ASSERT(sq->type == LM_GGML_TYPE_I32);

    const int64_t d_conv   = c->ne[0];
    const int64_t d_inner  = c->ne[1];
    const int64_t n_tokens = x->ne[1];
    const int64_t n_kv     = s->ne[2];

    LM_GGML_ASSERT( s->ne[0] == d_conv - 1);
    LM_GGML_ASSERT( s->ne[1] == d_inner);
    LM_GGML_ASSERT( x->ne[0] == d_inner);
    LM_GGML_ASSERT(sq->ne[0] == n_kv);
    LM_GGML_ASSERT(sq->ne[1] == n_tokens);

    bool is_node = false;

    if (s->grad || x->grad || c->grad || sq->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement
        is_node = true;
    }

    // 2-in-1 concatenated x and conv_states, {d_inner, n_tokens} with {d_conv, d_inner, n_kv}
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, (d_inner*n_tokens) + (d_conv*d_inner*n_kv));

    result->op   = LM_GGML_OP_SSM_CONV;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = c;
    result->src[3] = sq;

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
        struct lm_ggml_tensor  * C,
        struct lm_ggml_tensor  * sq) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous(s));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(x));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dt));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(A));
    LM_GGML_ASSERT(sq->type == LM_GGML_TYPE_I32);
    LM_GGML_ASSERT(B->nb[0] == lm_ggml_type_size(B->type));
    LM_GGML_ASSERT(C->nb[0] == lm_ggml_type_size(C->type));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(x, dt));

    {
        const int64_t d_state  = s->ne[0];
        const int64_t d_inner  = s->ne[1];
        const int64_t n_tokens = x->ne[1];

        LM_GGML_ASSERT(x->ne[0] == d_inner);
        LM_GGML_ASSERT(A->ne[0] == d_state);
        LM_GGML_ASSERT(A->ne[1] == d_inner);
        LM_GGML_ASSERT(B->ne[0] == d_state);
        LM_GGML_ASSERT(B->ne[1] == n_tokens);
        LM_GGML_ASSERT(C->ne[0] == d_state);
        LM_GGML_ASSERT(C->ne[1] == n_tokens);
    }

    bool is_node = false;

    if (s->grad || x->grad || dt->grad || A->grad || B->grad || C->grad || sq->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement
        is_node = true;
    }

    // 2-in-1 concatenated y and ssm_states, {d_inner, n_tokens} with {d_state, d_inner, n_kv}
    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, lm_ggml_nelements(x) + lm_ggml_nelements(s));

    result->op   = LM_GGML_OP_SSM_SCAN;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = s;
    result->src[1] = x;
    result->src[2] = dt;
    result->src[3] = A;
    result->src[4] = B;
    result->src[5] = C;
    result->src[6] = sq;

    return result;
}

// lm_ggml_win_part

struct lm_ggml_tensor * lm_ggml_win_part(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        int                   w) {
    LM_GGML_ASSERT(a->ne[3] == 1);
    LM_GGML_ASSERT(a->type  == LM_GGML_TYPE_F32);

    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

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

    result->op   = LM_GGML_OP_WIN_PART;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = { a->ne[0], w0, h0, 1, };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F32, 3, ne);

    int32_t params[] = { w };
    lm_ggml_set_op_params(result, params, sizeof(params));

    result->op   = LM_GGML_OP_WIN_UNPART;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (a->grad) {
        LM_GGML_ABORT("fatal error"); // TODO: implement backward
        is_node = true;
    }

    const int64_t ne[4] = { a->ne[0], kh, qh, 1, };
    struct lm_ggml_tensor * result = lm_ggml_new_tensor(ctx, LM_GGML_TYPE_F16, 3, ne);

    result->op   = LM_GGML_OP_GET_REL_POS;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

    bool is_node = false;

    if (!inplace && (a->grad || pw->grad || ph->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);
    lm_ggml_set_op_params_i32(result, 0, inplace ? 1 : 0);

    result->op   = LM_GGML_OP_ADD_REL_POS;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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

// lm_ggml_unary

static struct lm_ggml_tensor * lm_ggml_unary_impl(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * a,
        enum lm_ggml_unary_op op,
        bool inplace) {
    LM_GGML_ASSERT(lm_ggml_is_contiguous_1(a));

    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params_i32(result, 0, (int32_t) op);

    result->op   = LM_GGML_OP_UNARY;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_unary(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op op) {
    return lm_ggml_unary_impl(ctx, a, op, false);
}

struct lm_ggml_tensor * lm_ggml_unary_inplace(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor  * a,
        enum lm_ggml_unary_op op) {
    return lm_ggml_unary_impl(ctx, a, op, true);
}

// lm_ggml_map_unary

static struct lm_ggml_tensor * lm_ggml_map_unary_impl_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t fun,
        bool   inplace) {
    bool is_node = false;

    if (!inplace && a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op = LM_GGML_OP_MAP_UNARY;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_unary_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t fun) {
    return lm_ggml_map_unary_impl_f32(ctx, a, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_unary_inplace_f32(
        struct lm_ggml_context        * ctx,
        struct lm_ggml_tensor         * a,
        const  lm_ggml_unary_op_f32_t fun) {
    return lm_ggml_map_unary_impl_f32(ctx, a, fun, true);
}

// lm_ggml_map_binary

static struct lm_ggml_tensor * lm_ggml_map_binary_impl_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t fun,
        bool   inplace) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op = LM_GGML_OP_MAP_BINARY;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_binary_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t fun) {
    return lm_ggml_map_binary_impl_f32(ctx, a, b, fun, false);
}

struct lm_ggml_tensor * lm_ggml_map_binary_inplace_f32(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        const  lm_ggml_binary_op_f32_t fun) {
    return lm_ggml_map_binary_impl_f32(ctx, a, b, fun, true);
}

// lm_ggml_map_custom1_f32

static struct lm_ggml_tensor * lm_ggml_map_custom1_impl_f32(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_f32_t   fun,
        bool   inplace) {
    bool is_node = false;

    if (!inplace && a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op = LM_GGML_OP_MAP_CUSTOM1_F32;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        bool   inplace) {
    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op = LM_GGML_OP_MAP_CUSTOM2_F32;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
        bool   inplace) {
    bool is_node = false;

    if (!inplace && (a->grad || b->grad || c->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    lm_ggml_set_op_params(result, (const void *) &fun, sizeof(fun));

    result->op = LM_GGML_OP_MAP_CUSTOM3_F32;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
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
struct lm_ggml_map_custom1_op_params {
    lm_ggml_custom1_op_t fun;
    int n_tasks;
    void * userdata;
};

static struct lm_ggml_tensor * lm_ggml_map_custom1_impl(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_t       fun,
        int                            n_tasks,
        void                         * userdata,
        bool                           inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    bool is_node = false;

    if (!inplace && a->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom1_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op = LM_GGML_OP_MAP_CUSTOM1;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom1(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom1_inplace(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        const  lm_ggml_custom1_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom1_impl(ctx, a, fun, n_tasks, userdata, true);
}

// lm_ggml_map_custom2

struct lm_ggml_map_custom2_op_params {
    lm_ggml_custom2_op_t fun;
    int n_tasks;
    void * userdata;
};

static struct lm_ggml_tensor * lm_ggml_map_custom2_impl(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_t       fun,
        int                            n_tasks,
        void                         * userdata,
        bool                           inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom2_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op = LM_GGML_OP_MAP_CUSTOM2;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom2(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom2_inplace(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        const  lm_ggml_custom2_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom2_impl(ctx, a, b, fun, n_tasks, userdata, true);
}

// lm_ggml_map_custom3

struct lm_ggml_map_custom3_op_params {
    lm_ggml_custom3_op_t fun;
    int n_tasks;
    void * userdata;
};

static struct lm_ggml_tensor * lm_ggml_map_custom3_impl(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_t       fun,
        int                            n_tasks,
        void                         * userdata,
        bool                           inplace) {
    LM_GGML_ASSERT(n_tasks == LM_GGML_N_TASKS_MAX || n_tasks > 0);

    bool is_node = false;

    if (!inplace && (a->grad || b->grad || c->grad)) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = inplace ? lm_ggml_view_tensor(ctx, a) : lm_ggml_dup_tensor(ctx, a);

    struct lm_ggml_map_custom3_op_params params = {
        /*.fun      =*/ fun,
        /*.n_tasks  =*/ n_tasks,
        /*.userdata =*/ userdata
    };
    lm_ggml_set_op_params(result, (const void *) &params, sizeof(params));

    result->op = LM_GGML_OP_MAP_CUSTOM3;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

struct lm_ggml_tensor * lm_ggml_map_custom3(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, false);
}

struct lm_ggml_tensor * lm_ggml_map_custom3_inplace(
        struct lm_ggml_context          * ctx,
        struct lm_ggml_tensor           * a,
        struct lm_ggml_tensor           * b,
        struct lm_ggml_tensor           * c,
        const  lm_ggml_custom3_op_t       fun,
        int                            n_tasks,
        void                         * userdata) {
    return lm_ggml_map_custom3_impl(ctx, a, b, c, fun, n_tasks, userdata, true);
}

// lm_ggml_cross_entropy_loss

struct lm_ggml_tensor * lm_ggml_cross_entropy_loss(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct lm_ggml_tensor * result = lm_ggml_new_tensor_1d(ctx, a->type, 1);

    result->op   = LM_GGML_OP_CROSS_ENTROPY_LOSS;
    result->grad = is_node ? lm_ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
}

// lm_ggml_cross_entropy_loss_back

struct lm_ggml_tensor * lm_ggml_cross_entropy_loss_back(
        struct lm_ggml_context         * ctx,
        struct lm_ggml_tensor          * a,
        struct lm_ggml_tensor          * b,
        struct lm_ggml_tensor          * c) {
    LM_GGML_ASSERT(lm_ggml_are_same_shape(a, b));
    LM_GGML_ASSERT(lm_ggml_is_scalar(c));

    struct lm_ggml_tensor * result = lm_ggml_dup_tensor(ctx, a);

    result->op   = LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK;
    result->grad = NULL;
    result->src[0] = a;
    result->src[1] = b;
    result->src[2] = c;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_set_param(
        struct lm_ggml_context * ctx,
        struct lm_ggml_tensor * tensor) {
    tensor->flags |= LM_GGML_TENSOR_FLAG_PARAM;

    LM_GGML_ASSERT(tensor->grad == NULL);
    tensor->grad = lm_ggml_dup_tensor(ctx, tensor);
    lm_ggml_format_name(tensor->grad, "%s (grad)", tensor->name);
}

// lm_ggml_compute_forward_dup

static void lm_ggml_compute_forward_dup_same_cont(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst) && lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(src0->type == dst->type);

    const size_t nb00 = src0->nb[0];
    const size_t nb0 = dst->nb[0];

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    // parallelize by elements
    const int ne = lm_ggml_nelements(dst);
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = MIN(ie0 + dr, ne);

    if (ie0 < ie1) {
        memcpy(
            ((char *)  dst->data + ie0*nb0),
            ((char *) src0->data + ie0*nb00),
            (ie1 - ie0) * lm_ggml_type_size(src0->type));
    }
}

static void lm_ggml_compute_forward_dup_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    if (lm_ggml_is_contiguous(src0) && lm_ggml_is_contiguous(dst) && src0->type == dst->type) {
        lm_ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == lm_ggml_type_size(src0->type) && nb0 == lm_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (lm_ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(lm_ggml_fp16_t)) {
            if (dst->type == LM_GGML_TYPE_F16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = LM_GGML_FP16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (type_traits[dst->type].from_float) {
                lm_ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
                float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / lm_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = LM_GGML_FP16_TO_FP32(src0_ptr[i00]);
                            }

                            quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = LM_GGML_FP16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F16) {
                size_t id = 0;
                lm_ggml_fp16_t * dst_ptr = (lm_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == LM_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(lm_ggml_fp16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == LM_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(float *) dst_ptr = LM_GGML_FP16_TO_FP32(*(const lm_ggml_fp16_t *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        LM_GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void lm_ggml_compute_forward_dup_bf16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    if (lm_ggml_is_contiguous(src0) && lm_ggml_is_contiguous(dst) && src0->type == dst->type) {
        lm_ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == lm_ggml_type_size(src0->type) && nb0 == lm_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

    if (lm_ggml_is_contiguous(dst)) {
        if (nb00 == sizeof(lm_ggml_bf16_t)) {
            if (dst->type == LM_GGML_TYPE_BF16) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F16) {
                size_t id = 0;
                lm_ggml_fp16_t * dst_ptr = (lm_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = LM_GGML_FP32_TO_FP16(LM_GGML_BF16_TO_FP32(src0_ptr[i00]));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            for (int i00 = 0; i00 < ne00; i00++) {
                                dst_ptr[id] = LM_GGML_BF16_TO_FP32(src0_ptr[i00]);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (type_traits[dst->type].from_float) {
                lm_ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
                float * src0_f32 = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / lm_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                            for (int i00 = 0; i00 < ne00; i00++) {
                                src0_f32[i00] = LM_GGML_BF16_TO_FP32(src0_ptr[i00]);
                            }

                            quantize_row_q(src0_f32, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = LM_GGML_BF16_TO_FP32(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_BF16) {
                size_t id = 0;
                lm_ggml_bf16_t * dst_ptr = (lm_ggml_bf16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F16) {
                size_t id = 0;
                lm_ggml_fp16_t * dst_ptr = (lm_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = LM_GGML_FP32_TO_FP16(LM_GGML_BF16_TO_FP32(*src0_ptr));
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        }
        return;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == LM_GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(lm_ggml_bf16_t));

                        if (++i10 == ne00) {
                            i10 = 0;
                            if (++i11 == ne01) {
                                i11 = 0;
                                if (++i12 == ne02) {
                                    i12 = 0;
                                    if (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == LM_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(lm_ggml_fp16_t *) dst_ptr = LM_GGML_FP32_TO_FP16(LM_GGML_BF16_TO_FP32(*(const lm_ggml_bf16_t *) src0_ptr));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == LM_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(float *) dst_ptr = LM_GGML_BF16_TO_FP32(*(const lm_ggml_bf16_t *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        LM_GGML_ABORT("fatal error"); // TODO: implement
    }
}

static void lm_ggml_compute_forward_dup_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads

    if (lm_ggml_is_contiguous(src0) && lm_ggml_is_contiguous(dst) && src0->type == dst->type) {
        lm_ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == lm_ggml_type_size(src0->type) && nb0 == lm_ggml_type_size(dst->type)) {
        // copy by rows
        const size_t rs = ne00*nb00;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (lm_ggml_is_contiguous(dst)) {
        // TODO: simplify
        if (nb00 == sizeof(float)) {
            if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                const size_t rs = ne00 * nb00;
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else if (type_traits[dst->type].from_float) {
                lm_ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;

                size_t id = 0;
                size_t rs = nb0 * (ne00 / lm_ggml_blck_size(dst->type));
                char * dst_ptr = (char *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            quantize_row_q(src0_ptr, dst_ptr + id, ne00);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            if (dst->type == LM_GGML_TYPE_F32) {
                size_t id = 0;
                float * dst_ptr = (float *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = *src0_ptr;
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_F16) {
                size_t id = 0;
                lm_ggml_fp16_t * dst_ptr = (lm_ggml_fp16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = LM_GGML_FP32_TO_FP16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else if (dst->type == LM_GGML_TYPE_BF16) {
                size_t id = 0;
                lm_ggml_bf16_t * dst_ptr = (lm_ggml_bf16_t *) dst->data;

                for (int i03 = 0; i03 < ne03; i03++) {
                    for (int i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        for (int i01 = ir0; i01 < ir1; i01++) {
                            for (int i00 = 0; i00 < ne00; i00++) {
                                const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                                dst_ptr[id] = LM_GGML_FP32_TO_BF16(*src0_ptr);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            } else {
                LM_GGML_ABORT("fatal error"); // TODO: implement
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    if (dst->type == LM_GGML_TYPE_F32) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, sizeof(float));

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == LM_GGML_TYPE_F16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(lm_ggml_fp16_t *) dst_ptr = LM_GGML_FP32_TO_FP16(*(const float *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else if (dst->type == LM_GGML_TYPE_BF16) {
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        *(lm_ggml_bf16_t *) dst_ptr = LM_GGML_FP32_TO_BF16(*(const float *) src0_ptr);

                        if (++i10 == ne0) {
                            i10 = 0;
                            if (++i11 == ne1) {
                                i11 = 0;
                                if (++i12 == ne2) {
                                    i12 = 0;
                                    if (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                while (i10 >= ne0) {
                    i10 -= ne0;
                    if (++i11 == ne1) {
                        i11 = 0;
                        if (++i12 == ne2) {
                            i12 = 0;
                            if (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    } else {
        LM_GGML_ABORT("fatal error"); // TODO: implement
    }
}

// A simplified version of lm_ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void lm_ggml_compute_forward_dup_bytes(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));
    LM_GGML_ASSERT(src0->type == dst->type);

    if (lm_ggml_is_contiguous(src0) && lm_ggml_is_contiguous(dst)) {
        lm_ggml_compute_forward_dup_same_cont(params, dst);
        return;
    }

    LM_GGML_TENSOR_UNARY_OP_LOCALS;

    const size_t type_size = lm_ggml_type_size(src0->type);
    const int ith = params->ith; // thread index
    const int nth = params->nth; // number of threads


    // parallelize by rows
    const int nr = ne01;
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        const size_t rs = ne00 * type_size;
        for (int64_t i03 = 0; i03 < ne03; i03++) {
            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        return;
    }

    if (lm_ggml_is_contiguous(dst)) {
        size_t id = 0;
        char * dst_ptr = (char *) dst->data;
        const size_t rs = ne00 * type_size;

        if (nb00 == type_size) {
            // src0 is contigous on first dimension, copy by rows
            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } else {
            //printf("%s: this is not optimal - fix me\n", __func__);

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    for (int64_t i01 = ir0; i01 < ir1; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            const char * src0_ptr = (char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        return;
    }

    // dst counters

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            i10 += ne00 * ir0;
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            for (int64_t i01 = ir0; i01 < ir1; i01++) {
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    const char * src0_ptr = ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          char * dst_ptr  = ((char *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    if (++i10 == ne0) {
                        i10 = 0;
                        if (++i11 == ne1) {
                            i11 = 0;
                            if (++i12 == ne2) {
                                i12 = 0;
                                if (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            i10 += ne00 * (ne01 - ir1);
            while (i10 >= ne0) {
                i10 -= ne0;
                if (++i11 == ne1) {
                    i11 = 0;
                    if (++i12 == ne2) {
                        i12 = 0;
                        if (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_dup(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (src0->type == dst->type) {
        lm_ggml_compute_forward_dup_bytes(params, dst);
        return;
    }

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_dup_f16(params, dst);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                lm_ggml_compute_forward_dup_bf16(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_dup_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_add

static void lm_ggml_compute_forward_add_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_can_repeat(src1, src0) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef LM_GGML_USE_ACCELERATE
                vDSP_vadd(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                lm_ggml_vec_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }
}

static void lm_ggml_compute_forward_add_f16_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

    if (dst->type == LM_GGML_TYPE_F32) {
        LM_GGML_ASSERT( nb0 == sizeof(float));
    }
    else {
        LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_F16);
        LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_fp16_t));
    }

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == LM_GGML_TYPE_F16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                lm_ggml_fp16_t * dst_ptr  = (lm_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        } else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                float *       dst_ptr  = (float *)       ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = LM_GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    else {
        // src1 is not contiguous
        LM_GGML_ABORT("fatal error");
    }
}

static void lm_ggml_compute_forward_add_bf16_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

    if (dst->type == LM_GGML_TYPE_F32) {
        LM_GGML_ASSERT( nb0 == sizeof(float));
    }
    else {
        LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_BF16);
        LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_bf16_t));
    }

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        if (dst->type == LM_GGML_TYPE_BF16) {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                lm_ggml_bf16_t * dst_ptr  = (lm_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = LM_GGML_FP32_TO_BF16(LM_GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
                }
            }
        } else {
            for (int ir = ir0; ir < ir1; ++ir) {
                // src0, src1 and dst are same shape => same indices
                const int i3 = ir/(ne2*ne1);
                const int i2 = (ir - i3*ne2*ne1)/ne1;
                const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

                float *       dst_ptr  = (float *)       ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
                lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                float *       src1_ptr = (float *)       ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

                for (int i = 0; i < ne0; i++) {
                    dst_ptr[i] = LM_GGML_BF16_TO_FP32(src0_ptr[i]) + src1_ptr[i];
                }
            }
        }
    }
    else {
        // src1 is not contiguous
        LM_GGML_ABORT("fatal error");
    }
}

static void lm_ggml_compute_forward_add_f16_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_F16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(lm_ggml_fp16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

            lm_ggml_fp16_t * dst_ptr  = (lm_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
            lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
            lm_ggml_fp16_t * src1_ptr = (lm_ggml_fp16_t *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(src0_ptr[i]) + LM_GGML_FP16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    else {
        // src1 is not contiguous
        LM_GGML_ABORT("fatal error");
    }
}

static void lm_ggml_compute_forward_add_bf16_bf16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_BF16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_bf16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(lm_ggml_bf16_t)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

            lm_ggml_bf16_t * dst_ptr  = (lm_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1);
            lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
            lm_ggml_bf16_t * src1_ptr = (lm_ggml_bf16_t *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11);

            for (int i = 0; i < ne0; i++) {
                dst_ptr[i] = LM_GGML_FP32_TO_BF16(LM_GGML_BF16_TO_FP32(src0_ptr[i]) + LM_GGML_BF16_TO_FP32(src1_ptr[i]));
            }
        }
    }
    else {
        // src1 is not contiguous
        LM_GGML_ABORT("fatal error");
    }
}

static void lm_ggml_compute_forward_add_q_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum lm_ggml_type type = src0->type;
    const enum lm_ggml_type dtype = dst->type;
    lm_ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
    lm_ggml_from_float_t const quantize_row_q = type_traits[dtype].from_float;

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    LM_GGML_ASSERT(lm_ggml_is_quantized(src0->type));
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        const int i03 = ir/(ne02*ne01);
        const int i02 = (ir - i03*ne02*ne01)/ne01;
        const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        const int i13 = i03;
        const int i12 = i02;
        const int i11 = i01;

        const int i3 = i03;
        const int i2 = i02;
        const int i1 = i01;

        void  * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        float * src1_row = (float *)((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        void  * dst_row  = (void *) ((char *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        assert(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        lm_ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        if (quantize_row_q != NULL) {
            quantize_row_q(wdata, dst_row, ne00);
        } else {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

static void lm_ggml_compute_forward_add(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                if (src1->type == LM_GGML_TYPE_F32) {
                    lm_ggml_compute_forward_add_f32(params, dst);
                }
                else {
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_TYPE_F16:
            {
                if (src1->type == LM_GGML_TYPE_F16) {
                    lm_ggml_compute_forward_add_f16_f16(params, dst);
                }
                else if (src1->type == LM_GGML_TYPE_F32) {
                    lm_ggml_compute_forward_add_f16_f32(params, dst);
                }
                else {
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                if (src1->type == LM_GGML_TYPE_BF16) {
                    lm_ggml_compute_forward_add_bf16_bf16(params, dst);
                }
                else if (src1->type == LM_GGML_TYPE_F32) {
                    lm_ggml_compute_forward_add_bf16_f32(params, dst);
                }
                else {
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
            {
                lm_ggml_compute_forward_add_q_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_add1

static void lm_ggml_compute_forward_add1_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef LM_GGML_USE_ACCELERATE
        UNUSED(lm_ggml_vec_add1_f32);

        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (float *) ((char *) src1->data), 0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#else
        lm_ggml_vec_add1_f32(ne0,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(float *) src1->data);
#endif
    }
}

static void lm_ggml_compute_forward_add1_f16_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_F16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        lm_ggml_fp16_t * dst_ptr  = (lm_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void lm_ggml_compute_forward_add1_f16_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    // scalar to add
    const float v = LM_GGML_FP16_TO_FP32(*(lm_ggml_fp16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_F16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        lm_ggml_fp16_t * dst_ptr  = (lm_ggml_fp16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        lm_ggml_fp16_t * src0_ptr = (lm_ggml_fp16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = LM_GGML_FP32_TO_FP16(LM_GGML_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void lm_ggml_compute_forward_add1_q_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const enum lm_ggml_type type = src0->type;
    lm_ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
    lm_ggml_from_float_t const quantize_row_q = type_traits[type].from_float;

    // we don't support permuted src0
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    LM_GGML_ASSERT(lm_ggml_is_quantized(src0->type));
    LM_GGML_ASSERT(dst->type == src0->type);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        void  * src0_row = (void *) ((char *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        void  * dst_row  = (void *) ((char *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        assert(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        lm_ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

static void lm_ggml_compute_forward_add1_bf16_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    // scalar to add
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_BF16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_bf16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        lm_ggml_bf16_t * dst_ptr  = (lm_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = LM_GGML_FP32_TO_BF16(LM_GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void lm_ggml_compute_forward_add1_bf16_bf16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_scalar(src1));

    // scalar to add
    const float v = LM_GGML_BF16_TO_FP32(*(lm_ggml_bf16_t *) src1->data);

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_BF16);
    LM_GGML_ASSERT(dst->type  == LM_GGML_TYPE_BF16);

    LM_GGML_ASSERT( nb0 == sizeof(lm_ggml_bf16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_bf16_t));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        const int i3 = ir/(ne2*ne1);
        const int i2 = (ir - i3*ne2*ne1)/ne1;
        const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

        lm_ggml_bf16_t * dst_ptr  = (lm_ggml_bf16_t *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        lm_ggml_bf16_t * src0_ptr = (lm_ggml_bf16_t *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        for (int i = 0; i < ne0; i++) {
            dst_ptr[i] = LM_GGML_FP32_TO_BF16(LM_GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

static void lm_ggml_compute_forward_add1(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_add1_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
            {
                if (src1->type == LM_GGML_TYPE_F16) {
                    lm_ggml_compute_forward_add1_f16_f16(params, dst);
                }
                else if (src1->type == LM_GGML_TYPE_F32) {
                    lm_ggml_compute_forward_add1_f16_f32(params, dst);
                }
                else {
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_TYPE_BF16:
            {
                if (src1->type == LM_GGML_TYPE_BF16) {
                    lm_ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                else if (src1->type == LM_GGML_TYPE_F32) {
                    lm_ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                else {
                    LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q8_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
            {
                lm_ggml_compute_forward_add1_q_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_acc

static void lm_ggml_compute_forward_acc_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst) && lm_ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                lm_ggml_nbytes(dst));
        }
        lm_ggml_barrier(params->shared);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = lm_ggml_nrows(src1);
    const int nc = src1->ne[0];

    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    const size_t nb0 = lm_ggml_element_size(src0);

    const size_t nb00 = nb0;
    const size_t nb01 = nb1;
    const size_t nb02 = nb2;
    const size_t nb03 = nb3;

    LM_GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < lm_ggml_nbytes(dst));
    LM_GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < lm_ggml_nbytes(src0));

    LM_GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef LM_GGML_USE_ACCELERATE
        vDSP_vadd(
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#else
        lm_ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

static void lm_ggml_compute_forward_acc(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_acc_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_BF16:
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q8_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sub

static void lm_ggml_compute_forward_sub_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int nr  = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int ir = 0; ir < nr; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef LM_GGML_USE_ACCELERATE
            vDSP_vsub(
                    (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                    (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                    (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                    ne0);
#else
            lm_ggml_vec_sub_f32(ne0,
                    (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                    (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                    (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
                // }
            // }
        }
    } else {
        // src1 is not contiguous
        for (int ir = 0; ir < nr; ++ir) {
            // src0, src1 and dst are same shape => same indices
            const int i3 = ir/(ne2*ne1);
            const int i2 = (ir - i3*ne2*ne1)/ne1;
            const int i1 = (ir - i3*ne2*ne1 - i2*ne1);

            float * dst_ptr  = (float *) ((char *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
            for (int i0 = 0; i0 < ne0; i0++) {
                float * src1_ptr = (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10);

                dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
            }
        }
    }
}

static void lm_ggml_compute_forward_sub(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sub_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_mul

static void lm_ggml_compute_forward_mul_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_can_repeat(src1, src0) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0 ; r < nr0; ++r) {
#ifdef LM_GGML_USE_ACCELERATE
                UNUSED(lm_ggml_vec_mul_f32);

                vDSP_vmul(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                lm_ggml_vec_mul_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
            }
        }
    }
}

static void lm_ggml_compute_forward_mul(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32 && "only f32 src1 supported for now");

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_mul_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_div

static void lm_ggml_compute_forward_div_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_can_repeat(src1, src0) && lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = lm_ggml_nrows(src0);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef LM_GGML_USE_ACCELERATE
                UNUSED(lm_ggml_vec_div_f32);

                vDSP_vdiv(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
#else
                lm_ggml_vec_div_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
            }
        }
    }
}

static void lm_ggml_compute_forward_div(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_div_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sqr

static void lm_ggml_compute_forward_sqr_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_are_same_shape(src0, dst));

    const int n     = lm_ggml_nrows(src0);
    const int nc    = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_sqr_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_sqr(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sqr_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sqrt

static void lm_ggml_compute_forward_sqrt_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_sqrt_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_sqrt(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sqrt_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_log

static void lm_ggml_compute_forward_log_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    LM_GGML_ASSERT( dst->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_log_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_log(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_log_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sum

static void lm_ggml_compute_forward_sum_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_scalar(dst));


    assert(lm_ggml_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    lm_ggml_float sum     = 0;
    lm_ggml_float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                lm_ggml_vec_sum_f32_ggf(ne00,
                        &row_sum,
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                sum += row_sum;
            }
        }
    }
    ((float *) dst->data)[0] = sum;
}

static void lm_ggml_compute_forward_sum_f16(
    const struct lm_ggml_compute_params * params,
          struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(lm_ggml_fp16_t));

    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                lm_ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (lm_ggml_fp16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((lm_ggml_fp16_t *) dst->data)[0] = LM_GGML_FP32_TO_FP16(sum);
}

static void lm_ggml_compute_forward_sum_bf16(
    const struct lm_ggml_compute_params * params,
          struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_scalar(dst));

    assert(src0->nb[0] == sizeof(lm_ggml_bf16_t));

    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    float sum = 0;
    float row_sum = 0;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                lm_ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (lm_ggml_bf16_t *) ((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((lm_ggml_bf16_t *) dst->data)[0] = LM_GGML_FP32_TO_BF16(sum);
}

static void lm_ggml_compute_forward_sum(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sum_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_sum_f16(params, dst);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                lm_ggml_compute_forward_sum_bf16(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sum_rows

static void lm_ggml_compute_forward_sum_rows_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));
    LM_GGML_ASSERT(dst->nb[0] == sizeof(float));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(ne0 == 1);
    LM_GGML_ASSERT(ne1 == ne01);
    LM_GGML_ASSERT(ne2 == ne02);
    LM_GGML_ASSERT(ne3 == ne03);

    for (int64_t i3 = 0; i3 < ne03; i3++) {
        for (int64_t i2 = 0; i2 < ne02; i2++) {
            for (int64_t i1 = 0; i1 < ne01; i1++) {
                float * src_row = (float *) ((char *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                float * dst_row = (float *) ((char *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                float row_sum = 0;
                lm_ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

static void lm_ggml_compute_forward_sum_rows(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sum_rows_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_mean

static void lm_ggml_compute_forward_mean_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    UNUSED(ne0);
    UNUSED(ne1);
    UNUSED(ne2);
    UNUSED(ne3);

    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                lm_ggml_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

static void lm_ggml_compute_forward_mean(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_mean_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_argmax

static void lm_ggml_compute_forward_argmax_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    const size_t nb01 = src0->nb[1];
    const size_t nb0 = dst->nb[0];

    for (int64_t i1 = 0; i1 < ne01; i1++) {
        float * src = (float *) ((char *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((char *)  dst->data + i1*nb0);
        int v = 0;
        lm_ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

static void lm_ggml_compute_forward_argmax(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_argmax_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_repeat

static void lm_ggml_compute_forward_repeat_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_can_repeat(src0, dst));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in lm_ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    LM_GGML_ASSERT(nb0  == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                lm_ggml_vec_cpy_f32(ne00,
                                        (float *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (float *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_repeat_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_can_repeat(src0, dst));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in lm_ggml_can_repeat
    const int nr0 = (int)(ne0/ne00);
    const int nr1 = (int)(ne1/ne01);
    const int nr2 = (int)(ne2/ne02);
    const int nr3 = (int)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    LM_GGML_ASSERT(nb0  == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3;  i3++) {
        for                     (int k3 = 0; k3 < ne03; k3++) {
            for                 (int i2 = 0; i2 < nr2;  i2++) {
                for             (int k2 = 0; k2 < ne02; k2++) {
                    for         (int i1 = 0; i1 < nr1;  i1++) {
                        for     (int k1 = 0; k1 < ne01; k1++) {
                            for (int i0 = 0; i0 < nr0;  i0++) {
                                lm_ggml_fp16_t * y = (lm_ggml_fp16_t *) ((char *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                lm_ggml_fp16_t * x = (lm_ggml_fp16_t *) ((char *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // lm_ggml_vec_cpy_f16(ne00, y, x)
                                for (int i = 0; i < ne00; ++i) {
                                    y[i]  = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_repeat(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_BF16:
        case LM_GGML_TYPE_I16:
            {
                lm_ggml_compute_forward_repeat_f16(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
        case LM_GGML_TYPE_I32:
            {
                lm_ggml_compute_forward_repeat_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_repeat_back

static void lm_ggml_compute_forward_repeat_back_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_can_repeat(dst, src0));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in lm_ggml_can_repeat
    const int nr0 = (int)(ne00/ne0);
    const int nr1 = (int)(ne01/ne1);
    const int nr2 = (int)(ne02/ne2);
    const int nr3 = (int)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    LM_GGML_ASSERT(nb0  == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    if (lm_ggml_is_contiguous(dst)) {
        lm_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    } else {
        for         (int k3 = 0; k3 < ne3; k3++) {
            for     (int k2 = 0; k2 < ne2; k2++) {
                for (int k1 = 0; k1 < ne1; k1++) {
                    lm_ggml_vec_set_f32(ne0,
                        (float *) ((char *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    for                         (int i3 = 0; i3 < nr3; i3++) {
        for                     (int k3 = 0; k3 < ne3; k3++) {
            for                 (int i2 = 0; i2 < nr2; i2++) {
                for             (int k2 = 0; k2 < ne2; k2++) {
                    for         (int i1 = 0; i1 < nr1; i1++) {
                        for     (int k1 = 0; k1 < ne1; k1++) {
                            for (int i0 = 0; i0 < nr0; i0++) {
                                lm_ggml_vec_acc_f32(ne0,
                                        (float *) ((char *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (float *) ((char *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_repeat_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_repeat_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_concat

static void lm_ggml_compute_forward_concat_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    // TODO: support for transposed / permuted tensors
    LM_GGML_ASSERT(nb0  == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    const int32_t dim = lm_ggml_get_op_params_i32(dst, 0);

    LM_GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    const float * x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = ith; i2 < ne2; i2 += nth) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (const float *) ((const char *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } else {
                        x = (const float *) ((const char *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    float * y = (float *)((char *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_concat(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
        case LM_GGML_TYPE_I32:
            {
                lm_ggml_compute_forward_concat_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_abs

static void lm_ggml_compute_forward_abs_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_abs_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_abs(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_abs_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sgn

static void lm_ggml_compute_forward_sgn_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_sgn_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_sgn(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sgn_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_neg

static void lm_ggml_compute_forward_neg_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_neg_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_neg(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_neg_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_step

static void lm_ggml_compute_forward_step_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_step_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_step(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_step_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_tanh

static void lm_ggml_compute_forward_tanh_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_tanh_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_tanh(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_tanh_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_elu

static void lm_ggml_compute_forward_elu_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_elu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_elu(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_elu_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_relu

static void lm_ggml_compute_forward_relu_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_relu(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_relu_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_sigmoid

static void lm_ggml_compute_forward_sigmoid_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_sigmoid_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_sigmoid(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_sigmoid_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_gelu

static void lm_ggml_compute_forward_gelu_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        lm_ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void lm_ggml_compute_forward_gelu(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_gelu_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_gelu_quick

static void lm_ggml_compute_forward_gelu_quick_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        lm_ggml_vec_gelu_quick_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void lm_ggml_compute_forward_gelu_quick(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_gelu_quick_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_silu

static void lm_ggml_compute_forward_silu_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        lm_ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*(dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void lm_ggml_compute_forward_silu(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_silu_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}
// lm_ggml_compute_forward_leaky_relu

static void lm_ggml_compute_forward_leaky_relu_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_leaky_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

static void lm_ggml_compute_forward_leaky_relu(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_leaky_relu_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_silu_back

static void lm_ggml_compute_forward_silu_back_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * grad = dst->src[1];

    assert(lm_ggml_is_contiguous_1(grad));
    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));
    assert(lm_ggml_are_same_shape(src0, grad));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        lm_ggml_vec_silu_backward_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])),
                (float *) ((char *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void lm_ggml_compute_forward_silu_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_silu_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


static void lm_ggml_compute_forward_hardswish_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_hardswish_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}
static void lm_ggml_compute_forward_hardswish(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_hardswish_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

static void lm_ggml_compute_forward_hardsigmoid_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        lm_ggml_vec_hardsigmoid_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_hardsigmoid(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_hardsigmoid_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


// lm_ggml_compute_forward_norm

static void lm_ggml_compute_forward_norm_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    LM_GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                lm_ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (lm_ggml_float)x[i00];
                }

                float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                lm_ggml_float sum2 = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += (lm_ggml_float)(v*v);
                }

                float variance = sum2/ne00;
                const float scale = 1.0f/sqrtf(variance + eps);

                lm_ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void lm_ggml_compute_forward_norm(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_norm_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_group_rms_norm

static void lm_ggml_compute_forward_rms_norm_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    LM_GGML_ASSERT(eps > 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                lm_ggml_float sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (lm_ggml_float)(x[i00] * x[i00]);
                }

                const float mean = sum/ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0f/sqrtf(mean + eps);

                lm_ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void lm_ggml_compute_forward_rms_norm(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_rms_norm_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

static void lm_ggml_compute_forward_rms_norm_back_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst) && lm_ggml_are_same_shape(src0, src1));

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                const int64_t i11 = i01;
                const int64_t i12 = i02;
                const int64_t i13 = i03;

                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                const float * dz = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                lm_ggml_float sum_xx  = 0.0;
                lm_ggml_float sum_xdz = 0.0;

                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (lm_ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (lm_ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                const float mean_eps = (float)(sum_xx)/ne00 + eps;
                const float sum_eps  = (float)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement lm_ggml_rms and compose lm_ggml_rms_norm using lm_ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                const float rrms     = 1.0f / sqrtf(mean_eps);
                //const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

                {
                    // z = rms_norm(x)
                    //
                    // rms_norm(src0) =
                    //     scale(
                    //         src0,
                    //         div(
                    //             1,
                    //             sqrt(
                    //                 add(
                    //                     scale(
                    //                         sum(
                    //                             sqr(
                    //                                 src0)),
                    //                         (1.0/N)),
                    //                     eps))));

                    // postorder:
                    // ## op    args         grad
                    // 00 param src0         grad[#00]
                    // 01 const 1
                    // 02 sqr   (#00)        grad[#02]
                    // 03 sum   (#02)        grad[#03]
                    // 04 const 1/N
                    // 05 scale (#03, #04)   grad[#05]
                    // 06 const eps
                    // 07 add   (#05, #06)   grad[#07]
                    // 08 sqrt  (#07)        grad[#08]
                    // 09 div   (#01,#08)    grad[#09]
                    // 10 scale (#00,#09)    grad[#10]
                    //
                    // backward pass, given grad[#10]
                    // #10: scale
                    // grad[#00] += scale(grad[#10],#09)
                    // grad[#09] += sum(mul(grad[#10],#00))
                    // #09: div
                    // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
                    // #08: sqrt
                    // grad[#07] += mul(grad[#08], div(0.5, #08))
                    // #07: add
                    // grad[#05] += grad[#07]
                    // #05: scale
                    // grad[#03] += scale(grad[#05],#04)
                    // #03: sum
                    // grad[#02] += repeat(grad[#03], #02)
                    // #02:
                    // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
                    //
                    // substitute and simplify:
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#02] = repeat(grad[#03], #02)
                    // grad[#02] = repeat(scale(grad[#05],#04), #02)
                    // grad[#02] = repeat(scale(grad[#07],#04), #02)
                    // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N))), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum(mul(dz,x)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum_xdz * div(-1,rms*N*mean_eps))
                    // a = b*c + d*e
                    // a = b*c*f/f + d*e*f/f
                    // a = (b*c*f + d*e*f)*(1/f)
                    // a = (b*c*(1/c) + d*e*(1/c))*(1/(1/c))
                    // a = (b + d*e/c)*c
                    // b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps)
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
                    // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
                    // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
                    // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
                    // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
                    // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
                    // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                    // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                }
                // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                // post-order:
                // dx := x
                // dx := scale(dx,-mean_xdz/mean_eps)
                // dx := add(dx, dz)
                // dx := scale(dx, rrms)
                float * dx = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                lm_ggml_vec_cpy_f32  (ne00, dx, x);
                // lm_ggml_vec_scale_f32(ne00, dx, -mean_xdz/mean_eps);
                lm_ggml_vec_scale_f32(ne00, dx, (float)(-sum_xdz)/sum_eps);
                lm_ggml_vec_acc_f32  (ne00, dx, dz);
                lm_ggml_vec_scale_f32(ne00, dx, rrms);
            }
        }
    }
}

static void lm_ggml_compute_forward_rms_norm_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_rms_norm_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_group_norm

static void lm_ggml_compute_forward_group_norm_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const float eps = 1e-6f; // TODO: make this a parameter

    // TODO: optimize

    int n_channels = src0->ne[2];
    int n_groups = dst->op_params[0];
    int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    for (int i = ith; i < n_groups; i += nth) {
        int start = i * n_channels_per_group;
        int end = start + n_channels_per_group;
        if (end > n_channels) {
            end = n_channels;
        }
        int step = end - start;

        for (int64_t i03 = 0; i03 < ne03; i03++) {
            lm_ggml_float sum = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    lm_ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (lm_ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            const float mean = sum / (ne00 * ne01 * step);

            lm_ggml_float sum2 = 0.0;
            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * x = (float *)((char *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    lm_ggml_float sumr = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        float v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (lm_ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            const float variance = sum2 / (ne00 * ne01 * step);
            const float scale = 1.0f / sqrtf(variance + eps);

            for (int64_t i02 = start; i02 < end; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    float * y = (float *)((char *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    lm_ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_group_norm(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_group_norm_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_mul_mat

static void lm_ggml_compute_forward_mul_mat_one_chunk(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const enum lm_ggml_type type = src0->type;

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    lm_ggml_vec_dot_t const vec_dot      = type_traits[type].vec_dot;
    enum lm_ggml_type const vec_dot_type = type_traits[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_mul_mat(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum lm_ggml_type type = src0->type;

    enum lm_ggml_type           const vec_dot_type         = type_traits[type].vec_dot_type;
    lm_ggml_from_float_t        const from_float           = type_traits[vec_dot_type].from_float;
    lm_ggml_from_float_to_mat_t const from_float_to_mat    = type_traits[vec_dot_type].from_float_to_mat;
    int64_t                  const vec_dot_num_rows     = type_traits[type].nrows;
    int64_t                  const matmul_num_cols      = type_traits[type].ncols;
    int64_t                  const blck_size_interleave = type_traits[type].blck_size_interleave;
    lm_ggml_gemv_t              const gemv                 = type_traits[type].gemv;
    lm_ggml_gemm_t              const gemm                 = type_traits[type].gemm;

    LM_GGML_ASSERT(ne0 == ne01);
    LM_GGML_ASSERT(ne1 == ne11);
    LM_GGML_ASSERT(ne2 == ne12);
    LM_GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));
    LM_GGML_ASSERT(nb10 == lm_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

#if LM_GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(ne01, ne11, ne00/lm_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/lm_ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/lm_ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/lm_ggml_type_size(dst->type),
                                     ith, nth,
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw1 = lm_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                int64_t i11_processed = 0;
                if ((lm_ggml_n_dims(src1) == 2) && from_float_to_mat && gemm) {
                    for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                        from_float_to_mat((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                                          (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                          4, ne10, blck_size_interleave);
                    }
                    i11_processed = ne11 - ne11 % 4;
                }
                for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                           (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                           ne10);
                }
            }
        }
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store(&params->shared->current_chunk, nth);
    }

    lm_ggml_barrier(params->shared);

#if LM_GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(ne01, ne11, ne00/lm_ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/lm_ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/lm_ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/lm_ggml_type_size(dst->type),
                                     ith, nth,
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
    int64_t num_rows_per_vec_dot = vec_dot_num_rows;
    // TODO: currently the mmla kernels support only even numbered rows/cols.
    // this check can be removed once they are extended to support odd numbered rows/cols too
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        num_rows_per_vec_dot = 1;
    }

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggerganov/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || lm_ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    if ((lm_ggml_n_dims(src0) == 2) && gemv) {
        const void * src1_wdata      = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t src1_col_stride = lm_ggml_is_contiguous(src1) || src1->type != vec_dot_type ? lm_ggml_row_size(vec_dot_type, ne10) : nb11;
        int64_t src0_start = (ith * ne01) / nth;
        int64_t src0_end   = ((ith + 1) * ne01) / nth;
        src0_start = (src0_start % matmul_num_cols) ? src0_start + matmul_num_cols - (src0_start % matmul_num_cols): src0_start;
        src0_end   = (src0_end   % matmul_num_cols) ? src0_end   + matmul_num_cols - (src0_end   % matmul_num_cols): src0_end;
        if (src0_start >= src0_end) return;

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (gemm && (ne11 > 3)) {
            gemm(ne00, (float *)((char *) dst->data) + src0_start, ne01, (const char *) src0->data + src0_start * nb01,
                 (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
        }
        for (int iter = gemm ? ne11 - ne11 % 4 : 0; iter < ne11; iter++) {
            gemv(ne00, (float *)((char *) dst->data + (iter * nb1)) + src0_start, ne01,
                 (const char *) src0->data + src0_start * nb01, (const char *) src1_wdata + (src1_col_stride * iter), 1,
                 src0_end - src0_start);
        }
        return;
    }

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        lm_ggml_compute_forward_mul_mat_one_chunk(params, dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add(&params->shared->current_chunk, 1);
    }
}

// lm_ggml_compute_forward_mul_mat_id

static void lm_ggml_compute_forward_mul_mat_id(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * ids = dst->src[2];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum lm_ggml_type type = src0->type;

    const bool src1_cont = lm_ggml_is_contiguous(src1);

    lm_ggml_vec_dot_t    const vec_dot         = type_traits[type].vec_dot;
    enum lm_ggml_type    const vec_dot_type    = type_traits[type].vec_dot_type;
    lm_ggml_from_float_t const from_float      = type_traits[vec_dot_type].from_float;
    int64_t           const matmul_num_cols = type_traits[type].ncols;
    lm_ggml_gemv_t       const gemv            = type_traits[type].gemv;

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));
    LM_GGML_ASSERT(nb10 == lm_ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    char * wdata_src1_end = (src1->type == vec_dot_type) ?
            (char *) params->wdata :
            (char *) params->wdata + LM_GGML_PAD(lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(src1)), sizeof(int64_t));

    struct mmid_row_mapping {
        int32_t i1;
        int32_t i2;
    };

    int64_t * matrix_row_counts = (int64_t *) (wdata_src1_end); // [n_as]
    struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *)(matrix_row_counts + n_as); // [n_as][ne11]

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw1 = lm_ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
    }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ne12 + (i1)]

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    lm_ggml_barrier(params->shared);

    // compute each matrix multiplication in sequence
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a*nb02;

        const void * wdata    = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = lm_ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01; // src0 rows
        const int64_t nr1 = cne1; // src1 rows

        if (((lm_ggml_n_dims(src0) - 1) == 2) && gemv) {
            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;
            src0_cur_start = (src0_cur_start % matmul_num_cols) ? src0_cur_start + matmul_num_cols - (src0_cur_start % matmul_num_cols): src0_cur_start;
            src0_cur_end   = (src0_cur_end % matmul_num_cols) ? src0_cur_end + matmul_num_cols - (src0_cur_end % matmul_num_cols): src0_cur_end;
            if (src0_cur_start >= src0_cur_end) return;

            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11        + i12 * ne11) * row_size
                    : (i11 * nb11 + i12 * nb12));

                gemv(ne00, (float *)((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                     (const char *) src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
            continue;
        }

        // distribute the thread work across the inner or outer loop based on which one is larger

        const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

        const int64_t ith0 = ith % nth0;
        const int64_t ith1 = ith / nth0;

        const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
        const int64_t dr1 = (nr1 + nth1 - 1)/nth1;

        const int64_t ir010 = dr0*ith0;
        const int64_t ir011 = MIN(ir010 + dr0, nr0);

        const int64_t ir110 = dr1*ith1;
        const int64_t ir111 = MIN(ir110 + dr1, nr1);

        // threads with no work simply yield (not sure if it helps)
        //if (ir010 >= ir011 || ir110 >= ir111) {
        //    sched_yield();
        //    continue;
        //}

        // block-tiling attempt
        const int64_t blck_0 = 16;
        const int64_t blck_1 = 16;

        // attempt to reduce false-sharing (does not seem to make a difference)
        float tmp[16];

        for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
            for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
                for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
                    const int64_t _i12 = ir1; // logical row index for this expert

                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                    const int id       = row_mapping.i1; // selected expert index

                    const int64_t  i11 = id % ne11;
                    const int64_t  i12 = row_mapping.i2; // row index in src1

                    const int64_t  i1 = id;  // selected expert index
                    const int64_t  i2 = i12; // row

                    // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                    //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                    //       the original src1 data pointer, so we should index using the indices directly
                    // TODO: this is a bit of a hack, we should probably have a better way to handle this
                    const char * src1_col = (const char *) wdata +
                        (src1_cont || src1->type != vec_dot_type
                        ? (i11      + i12*ne11)*row_size
                        : (i11*nb11 + i12*nb12));

                    float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                    //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                    //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                    //}

                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                    }

                    memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir011) - iir0)*sizeof(float));
                }
            }
        }
    }

#undef MMID_MATRIX_ROW
}

// lm_ggml_compute_forward_out_prod

static void lm_ggml_compute_forward_out_prod_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_ASSERT(ne0  == ne00);
    LM_GGML_ASSERT(ne1  == ne10);
    LM_GGML_ASSERT(ne2  == ne02);
    LM_GGML_ASSERT(ne02 == ne12);
    LM_GGML_ASSERT(ne3  == ne13);
    LM_GGML_ASSERT(ne03 == ne13);

    // we don't support permuted src0 or src1
    LM_GGML_ASSERT(nb00 == sizeof(float));

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    // LM_GGML_ASSERT(nb0 <= nb1);
    // LM_GGML_ASSERT(nb1 <= nb2);
    // LM_GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        lm_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    }
    lm_ggml_barrier(params->shared);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    const int64_t blck_0 = MAX(LM_GGML_VEC_MAD_UNROLL, 32);
    const int64_t blck_1 = 16;

    for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        const int64_t bir1 = MIN(bir + blck_1, ir1);
        for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            const int64_t bne01 = MIN(bi01 + blck_0, ne01);
            for (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                const int64_t i3 = ir/(ne2*ne1);
                const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                const int64_t i02 = i2;
                const int64_t i03 = i3;

                //const int64_t i10 = i1;
                const int64_t i12 = i2;
                const int64_t i13 = i3;

#if LM_GGML_VEC_MAD_UNROLL > 2
                const int64_t bne01_unroll = bne01 - (bne01 % LM_GGML_VEC_MAD_UNROLL);
                for (int64_t i01 = bi01; i01 < bne01_unroll; i01 += LM_GGML_VEC_MAD_UNROLL) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    lm_ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                for (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    lm_ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#else
                for (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    const int64_t i11 = i01;

                    float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    lm_ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

static void lm_ggml_compute_forward_out_prod_q_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS;

    const int ith = params->ith;
    const int nth = params->nth;

    const enum lm_ggml_type type = src0->type;
    lm_ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;

    LM_GGML_ASSERT(ne02 == ne12);
    LM_GGML_ASSERT(ne03 == ne13);
    LM_GGML_ASSERT(ne2  == ne12);
    LM_GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    LM_GGML_ASSERT(nb00 == lm_ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    // LM_GGML_ASSERT(nb0 <= nb1);
    // LM_GGML_ASSERT(nb1 <= nb2);
    // LM_GGML_ASSERT(nb2 <= nb3);

    LM_GGML_ASSERT(ne0 == ne00);
    LM_GGML_ASSERT(ne1 == ne10);
    LM_GGML_ASSERT(ne2 == ne02);
    LM_GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        lm_ggml_vec_set_f32(ne0*ne1*ne2*ne3, dst->data, 0);
    }
    lm_ggml_barrier(params->shared);

    // parallelize by last three dimensions

    // total rows in dst
    const int64_t nr = ne1*ne2*ne3;

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    float * wdata = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        const int64_t i3 = ir/(ne2*ne1);
        const int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        const int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        const int64_t i02 = i2;
        const int64_t i03 = i3;

        //const int64_t i10 = i1;
        const int64_t i12 = i2;
        const int64_t i13 = i3;

        for (int64_t i01 = 0; i01 < ne01; ++i01) {
            const int64_t i11 = i01;

            float * s0 = (float *) ((char *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            float * s1 = (float *) ((char *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            float * d  = (float *) ((char *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            lm_ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

static void lm_ggml_compute_forward_out_prod(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
            {
                lm_ggml_compute_forward_out_prod_q_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
            {
                LM_GGML_ABORT("fatal error"); // todo
                // lm_ggml_compute_forward_out_prod_f16_f32(params, dst);
            }
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_out_prod_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_scale

static void lm_ggml_compute_forward_scale_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));

    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    for (int i1 = ir0; i1 < ir1; i1++) {
        if (dst->data != src0->data) {
            // src0 is same shape as dst => same indices
            memcpy((char *)dst->data + i1*nb1, (char *)src0->data + i1*nb01, nc * sizeof(float));
        }
        lm_ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*nb1), v);
    }
}

static void lm_ggml_compute_forward_scale(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_scale_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_set

static void lm_ggml_compute_forward_set_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst) && lm_ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    if (!inplace) {
        if (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                lm_ggml_nbytes(dst));
        }
        lm_ggml_barrier(params->shared);
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = lm_ggml_nrows(src1);
    const int nc = src1->ne[0];

    LM_GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    const size_t nb0 = lm_ggml_element_size(src0);

    const int im0 = (ne10 == 0 ? 0 : ne10-1);
    const int im1 = (ne11 == 0 ? 0 : ne11-1);
    const int im2 = (ne12 == 0 ? 0 : ne12-1);
    const int im3 = (ne13 == 0 ? 0 : ne13-1);

    LM_GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= lm_ggml_nbytes(dst));

    LM_GGML_ASSERT(nb10 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        const int i3 = ir/(ne12*ne11);
        const int i2 = (ir - i3*ne12*ne11)/ne11;
        const int i1 = (ir - i3*ne12*ne11 - i2*ne11);

        lm_ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (float *) ((char *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

static void lm_ggml_compute_forward_set(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_set_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_BF16:
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q8_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_cpy

static void lm_ggml_compute_forward_cpy(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    lm_ggml_compute_forward_dup(params, dst);
}

// lm_ggml_compute_forward_cont

static void lm_ggml_compute_forward_cont(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    lm_ggml_compute_forward_dup(params, dst);
}

// lm_ggml_compute_forward_reshape

static void lm_ggml_compute_forward_reshape(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// lm_ggml_compute_forward_view

static void lm_ggml_compute_forward_view(
        const struct lm_ggml_compute_params * params,
        const struct lm_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// lm_ggml_compute_forward_permute

static void lm_ggml_compute_forward_permute(
        const struct lm_ggml_compute_params * params,
        const struct lm_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// lm_ggml_compute_forward_transpose

static void lm_ggml_compute_forward_transpose(
        const struct lm_ggml_compute_params * params,
        const struct lm_ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// lm_ggml_compute_forward_get_rows

static void lm_ggml_compute_forward_get_rows_q(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = lm_ggml_nelements(src1);

    const enum lm_ggml_type type = src0->type;
    lm_ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == lm_ggml_type_size(type));
    assert(lm_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        assert(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void lm_ggml_compute_forward_get_rows_f16(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = lm_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(lm_ggml_fp16_t));
    assert(lm_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        assert(i01 >= 0 && i01 < ne01);

        lm_ggml_fp16_to_fp32_row(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void lm_ggml_compute_forward_get_rows_bf16(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = lm_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(lm_ggml_bf16_t));
    assert(lm_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        assert(i01 >= 0 && i01 < ne01);

        lm_ggml_bf16_to_fp32_row(
                (const void *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

static void lm_ggml_compute_forward_get_rows_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t nc = ne00;
    const int64_t nr = lm_ggml_nelements(src1);

    assert(ne0  == nc);
    assert(ne02 == ne11);
    assert(nb00 == sizeof(float));
    assert(lm_ggml_nrows(dst) == nr);

    const int ith = params->ith;
    const int nth = params->nth;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int64_t i = ir0; i < ir1; ++i) {
        const int64_t i12 = i/(ne11*ne10);
        const int64_t i11 = (i - i12*ne11*ne10)/ne10;
        const int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        const int64_t i01 = *(int32_t *) ((char *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        assert(i01 >= 0 && i01 < ne01);

        lm_ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (float *) ((char *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

static void lm_ggml_compute_forward_get_rows(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q8_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
            {
                lm_ggml_compute_forward_get_rows_q(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_get_rows_f16(params, dst);
            } break;
        case LM_GGML_TYPE_BF16:
            {
                lm_ggml_compute_forward_get_rows_bf16(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
        case LM_GGML_TYPE_I32:
            {
                lm_ggml_compute_forward_get_rows_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// lm_ggml_compute_forward_get_rows_back

static void lm_ggml_compute_forward_get_rows_back_f32_f16(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));

    // lm_ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, lm_ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nelements(src1);

    LM_GGML_ASSERT( dst->ne[0] == nc);
    LM_GGML_ASSERT(src0->nb[0] == sizeof(lm_ggml_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            lm_ggml_fp16_t v = ((lm_ggml_fp16_t *) ((char *) src0->data + i*src0->nb[1]))[j];
            ((float *) ((char *) dst->data + r*dst->nb[1]))[j] += LM_GGML_FP16_TO_FP32(v);
        }
    }
}

static void lm_ggml_compute_forward_get_rows_back_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));

    // lm_ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, lm_ggml_nbytes(dst));

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nelements(src1);

    LM_GGML_ASSERT( dst->ne[0] == nc);
    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        lm_ggml_vec_add_f32(nc,
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *)  dst->data + r*dst->nb[1]),
                (float *) ((char *) src0->data + i*src0->nb[1]));
    }
}

static void lm_ggml_compute_forward_get_rows_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_get_rows_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// lm_ggml_compute_forward_diag

static void lm_ggml_compute_forward_diag_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(ne00 == ne0);
    LM_GGML_ASSERT(ne00 == ne1);
    LM_GGML_ASSERT(ne01 == 1);
    LM_GGML_ASSERT(ne02 == ne2);
    LM_GGML_ASSERT(ne03 == ne3);

    LM_GGML_ASSERT(nb00 == sizeof(float));
    LM_GGML_ASSERT(nb0  == sizeof(float));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                float * d = (float *)((char *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                float * s = (float *)((char *) src0->data + i3*nb03 + i2*nb02);
                for (int i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                for (int i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_diag(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_diag_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_diag_mask_inf

static void lm_ggml_compute_forward_diag_mask_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const float value) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    const int ith = params->ith;
    const int nth = params->nth;

    const int  n_past  = ((int32_t *) dst->op_params)[0];
    const bool inplace = src0->data == dst->data;

    LM_GGML_ASSERT(n_past >= 0);

    if (!inplace) {
        if (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            LM_GGML_ASSERT(lm_ggml_nelements(dst) == lm_ggml_nelements(src0));
            LM_GGML_ASSERT(lm_ggml_is_contiguous(dst) && lm_ggml_is_contiguous(src0));
            memcpy(
                ((char *)  dst->data),
                ((char *) src0->data),
                lm_ggml_nbytes(dst));
        }
        lm_ggml_barrier(params->shared);
    }

    // TODO: handle transposed/permuted matrices

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    LM_GGML_ASSERT( dst->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = ith; j < nr; j += nth) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_diag_mask_inf(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

static void lm_ggml_compute_forward_diag_mask_zero(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_soft_max

static void lm_ggml_compute_forward_soft_max_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    assert(lm_ggml_is_contiguous(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    //const int64_t ne11 = src1 ? src1->ne[1] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    const uint32_t n_head      = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * wp = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;

    const bool use_f16 = (src1 && src1->type == LM_GGML_TYPE_F16);

    for (int i1 = ir0; i1 < ir1; i1++) {
        // ALiBi
        const uint32_t h = (i1/ne01)%ne02; // head
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * dp = (float *)((char *)  dst->data +  i1*dst->nb[1]);

        // broadcast the mask across rows
        lm_ggml_fp16_t * mp_f16 = src1 ? (lm_ggml_fp16_t *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;
        float       * mp_f32 = src1 ? (float       *)((char *) src1->data) + (i1%ne01)*ne00 : NULL;

        lm_ggml_vec_cpy_f32  (nc, wp, sp);
        lm_ggml_vec_scale_f32(nc, wp, scale);
        if (mp_f32) {
            if (use_f16) {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*LM_GGML_FP16_TO_FP32(mp_f16[i]);
                }
            } else {
                for (int i = 0; i < nc; ++i) {
                    wp[i] += slope*mp_f32[i];
                }
            }
        }

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(wp[i]));
        }
#endif

        float max = -INFINITY;
        lm_ggml_vec_max_f32(nc, &max, wp);

        lm_ggml_float sum = lm_ggml_vec_soft_max_f32(nc, dp, wp, max);
        assert(sum > 0.0);

        sum = 1.0/sum;
        lm_ggml_vec_scale_f32(nc, dp, sum);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(dp[i]));
            assert(!isinf(dp[i]));
        }
#endif
    }
}

static void lm_ggml_compute_forward_soft_max(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_soft_max_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


// lm_ggml_compute_forward_soft_max_back

static void lm_ggml_compute_forward_soft_max_back_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, dst));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src1, dst));

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *dy = (float *)((char *) src0->data + i1*src0->nb[1]);
        float *y  = (float *)((char *) src1->data + i1*src1->nb[1]);
        float *dx = (float *)((char *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(dy[i]));
            assert(!isnan(y[i]));
        }
#endif
        // Jii = yi - yi*yi
        // Jij = -yi*yj
        // J = diag(y)-y.T*y
        // dx = J * dy
        // dxk = sum_i(Jki * dyi)
        // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*dyk
        // dxk = -yk * sum_i(yi * dyi) + yk*dyk
        // dxk = -yk * dot(y, dy) + yk*dyk
        // dxk = yk * (- dot(y, dy) + dyk)
        // dxk = yk * (dyk - dot(y, dy))
        //
        // post-order:
        // dot_y_dy := dot(y, dy)
        // dx := dy
        // dx := dx - dot_y_dy
        // dx := dx * y

        // linear runtime, no additional memory
        float dot_y_dy = 0;
        lm_ggml_vec_dot_f32 (nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        lm_ggml_vec_cpy_f32 (nc, dx, dy);
        lm_ggml_vec_acc1_f32(nc, dx, -dot_y_dy);
        lm_ggml_vec_mul_f32 (nc, dx, dx, y);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(dx[i]));
            assert(!isinf(dx[i]));
        }
#endif
    }
}

static void lm_ggml_compute_forward_soft_max_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_soft_max_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_clamp

static void lm_ggml_compute_forward_clamp_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    float min;
    float max;
    memcpy(&min, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max, (float *) dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    LM_GGML_ASSERT( nb0 == sizeof(float));
    LM_GGML_ASSERT(nb00 == sizeof(float));

    for (int j = ith; j < n; j += nth) {
        float * dst_ptr  = (float *) ((char *)  dst->data + j*nb1);
        float * src0_ptr = (float *) ((char *) src0->data + j*nb01);

        for (int i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

static void lm_ggml_compute_forward_clamp(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_clamp_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_BF16:
        case LM_GGML_TYPE_Q4_0:
        case LM_GGML_TYPE_Q4_1:
        case LM_GGML_TYPE_Q5_0:
        case LM_GGML_TYPE_Q5_1:
        case LM_GGML_TYPE_Q8_0:
        case LM_GGML_TYPE_Q8_1:
        case LM_GGML_TYPE_Q2_K:
        case LM_GGML_TYPE_Q3_K:
        case LM_GGML_TYPE_Q4_K:
        case LM_GGML_TYPE_Q5_K:
        case LM_GGML_TYPE_Q6_K:
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ3_XXS:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:
        case LM_GGML_TYPE_IQ4_NL:
        case LM_GGML_TYPE_IQ4_XS:
        case LM_GGML_TYPE_IQ3_S:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_Q8_K:
        case LM_GGML_TYPE_Q4_0_4_4:
        case LM_GGML_TYPE_Q4_0_4_8:
        case LM_GGML_TYPE_Q4_0_8_8:
        case LM_GGML_TYPE_I8:
        case LM_GGML_TYPE_I16:
        case LM_GGML_TYPE_I32:
        case LM_GGML_TYPE_I64:
        case LM_GGML_TYPE_F64:
        case LM_GGML_TYPE_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_rope

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / MAX(0.001f, high - low);
    return 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float lm_ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float)M_PI)) / (2 * logf(base));
}

static void lm_ggml_rope_cache_init(
     float theta_base, float freq_scale, const float * freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
     float * cache, float sin_sign, float theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    float theta = theta_base;
    for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        const float ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

LM_GGML_CALL void lm_ggml_rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    float start = floorf(lm_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end   =  ceilf(lm_ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = MAX(0, start);
    dims[1] = MIN(n_dims - 1, end);
}

static void lm_ggml_compute_forward_rope_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const bool forward) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    LM_GGML_ASSERT(nb00 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = lm_ggml_nrows(dst);

    LM_GGML_ASSERT(n_dims <= ne0);
    LM_GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    lm_ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & 2;

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        LM_GGML_ASSERT(src2->type == LM_GGML_TYPE_F32);
        LM_GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            lm_ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = src[0];
                        const float x1 = src[1];

                        dst_data[0] = x0*cos_theta - x1*sin_theta;
                        dst_data[1] = x0*sin_theta + x1*cos_theta;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = src[0];
                        const float x1 = src[n_dims/2];

                        dst_data[0]        = x0*cos_theta - x1*sin_theta;
                        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;
                    }
                }

                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const float * const src = (float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    float * dst_data  = (float *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

// TODO: deduplicate f16/f32 code
static void lm_ggml_compute_forward_rope_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const bool forward) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * src2 = dst->src[2];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    LM_GGML_ASSERT(nb0 == sizeof(lm_ggml_fp16_t));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nr = lm_ggml_nrows(dst);

    LM_GGML_ASSERT(n_dims <= ne0);
    LM_GGML_ASSERT(n_dims % 2 == 0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    int ir = 0;

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    float corr_dims[2];
    lm_ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const bool is_neox = mode & 2;

    const float * freq_factors = NULL;
    if (src2 != NULL) {
        LM_GGML_ASSERT(src2->type == LM_GGML_TYPE_F32);
        LM_GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (const float *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    const float sin_sign = forward ? 1.0f : -1.0f;

    const int32_t * pos = (const int32_t *) src1->data;

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            const int64_t p = pos[i2];

            float * cache = (float *) params->wdata + (ne0 + CACHE_LINE_SIZE_F32)*ith;
            lm_ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);

            for (int64_t i1 = 0; i1 < ne1; i1++) {
                if (ir++ < ir0) continue;
                if (ir   > ir1) break;

                if (!is_neox) {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const lm_ggml_fp16_t * const src = (lm_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                              lm_ggml_fp16_t * dst_data  = (lm_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        const float x0 = LM_GGML_FP16_TO_FP32(src[0]);
                        const float x1 = LM_GGML_FP16_TO_FP32(src[1]);

                        dst_data[0] = LM_GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[1] = LM_GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                } else {
                    for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
                        const int64_t ic = i0/2;

                        const float cos_theta = cache[i0 + 0];
                        const float sin_theta = cache[i0 + 1];

                        const lm_ggml_fp16_t * const src = (lm_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
                        lm_ggml_fp16_t * dst_data  = (lm_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

                        const float x0 = LM_GGML_FP16_TO_FP32(src[0]);
                        const float x1 = LM_GGML_FP16_TO_FP32(src[n_dims/2]);

                        dst_data[0]        = LM_GGML_FP32_TO_FP16(x0*cos_theta - x1*sin_theta);
                        dst_data[n_dims/2] = LM_GGML_FP32_TO_FP16(x0*sin_theta + x1*cos_theta);
                    }
                }

                for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
                    const lm_ggml_fp16_t * const src = (lm_ggml_fp16_t *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    lm_ggml_fp16_t * dst_data  = (lm_ggml_fp16_t *)((char *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                    dst_data[0] = src[0];
                    dst_data[1] = src[1];
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_rope(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_rope_f16(params, dst, true);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_rope_f32(params, dst, true);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_rope_back

static void lm_ggml_compute_forward_rope_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_rope_f16(params, dst, false);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_rope_f32(params, dst, false);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_conv_transpose_1d

static void lm_ggml_compute_forward_conv_transpose_1d_f16_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const lm_ggml_fp16_t * const src = (lm_ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    lm_ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) params->wdata + nk;
            lm_ggml_fp16_t * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = LM_GGML_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, lm_ggml_nbytes(dst));
    }
    lm_ggml_barrier(params->shared);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    lm_ggml_fp16_t * const wdata     = (lm_ggml_fp16_t *) params->wdata + 0;
    lm_ggml_fp16_t * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        lm_ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                lm_ggml_vec_dot_f16(ne02, &v, 0,
                        (lm_ggml_fp16_t *)    wdata_src + i1n, 0,
                        (lm_ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

static void lm_ggml_compute_forward_conv_transpose_1d_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02;

    LM_GGML_ASSERT(nb00 == sizeof(float));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int64_t i02 = 0; i02 < ne02; i02++) {
                for (int64_t i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i01*ne00*ne02;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + nk;
            float * dst_data = wdata;

            for (int64_t i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                for (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, lm_ggml_nbytes(dst));
    }
    lm_ggml_barrier(params->shared);

    const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

    // total rows in dst
    const int nr = ne1;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float * const wdata     = (float *) params->wdata + 0;
    float * const wdata_src = wdata + nk;

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        float * wdata_kernel = wdata + i1*ne02*ne00;
        for (int i10 = 0; i10 < ne10; i10++) {
            const int i1n = i10*ne11;
            for (int i00 = 0; i00 < ne00; i00++) {
                float v = 0;
                lm_ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

static void lm_ggml_compute_forward_conv_transpose_1d(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void lm_ggml_compute_forward_im2col_f32(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

    LM_GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        float * const wdata = (float *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        float * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void lm_ggml_compute_forward_im2col_f16(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F16);

    LM_GGML_TENSOR_BINARY_OP_LOCALS;

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t s1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];
    const int32_t p1 = ((const int32_t *)(dst->op_params))[3];
    const int32_t d0 = ((const int32_t *)(dst->op_params))[4];
    const int32_t d1 = ((const int32_t *)(dst->op_params))[5];
    const bool is_2D = ((const int32_t *)(dst->op_params))[6] == 1;

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t N  = is_2D ? ne13 : ne12;
    const int64_t IC = is_2D ? ne12 : ne11;
    const int64_t IH = is_2D ? ne11 : 1;
    const int64_t IW = ne10;

    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    const int64_t OH = is_2D ? ne2 : 1;
    const int64_t OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) dst->data;

        for (int64_t in = 0; in < N; in++) {
            for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                for (int64_t iow = 0; iow < OW; iow++) {
                    for (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        lm_ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        const float * const src_data = (float *)((char *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            for (int64_t ikw = 0; ikw < KW; ikw++) {
                                const int64_t iiw = iow*s0 + ikw*d0 - p0;
                                const int64_t iih = ioh*s1 + ikh*d1 - p1;

                                if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } else {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = LM_GGML_FP32_TO_FP16(src_data[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_im2col(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {
    switch (dst->type) {
        case LM_GGML_TYPE_F16:
            {
                lm_ggml_compute_forward_im2col_f16(params, dst);
            } break;
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_im2col_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


// lm_ggml_compute_forward_conv_transpose_2d

static void lm_ggml_compute_forward_conv_transpose_2d(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F16);
    LM_GGML_ASSERT(src1->type == LM_GGML_TYPE_F32);
    LM_GGML_ASSERT( dst->type == LM_GGML_TYPE_F32);

    LM_GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00*ne01*ne02*ne03;

    LM_GGML_ASSERT(nb00 == sizeof(lm_ggml_fp16_t));
    LM_GGML_ASSERT(nb10 == sizeof(float));

    if (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) params->wdata + 0;

            for (int64_t i03 = 0; i03 < ne03; i03++) {
                for (int64_t i02 = 0; i02 < ne02; i02++) {
                    const lm_ggml_fp16_t * const src = (lm_ggml_fp16_t *)((char *) src0->data + i03*nb03 + i02*nb02);
                    lm_ggml_fp16_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    for (int64_t i01 = 0; i01 < ne01; i01++) {
                        for (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
        {
            lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) params->wdata + nk;
            for (int i12 = 0; i12 < ne12; i12++) {
                for (int i11 = 0; i11 < ne11; i11++) {
                    const float * const src = (float *)((char *) src1->data + i12*nb12 + i11*nb11);
                    lm_ggml_fp16_t * dst_data = wdata + i11*ne10*ne12;
                    for (int i10 = 0; i10 < ne10; i10++) {
                        dst_data[i10*ne12 + i12] = LM_GGML_FP32_TO_FP16(src[i10]);
                    }
                }
            }
        }

        memset(dst->data, 0, lm_ggml_nbytes(dst));
    }
    lm_ggml_barrier(params->shared);

    const int32_t stride = lm_ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    const int np = ne2;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    lm_ggml_fp16_t * const wdata = (lm_ggml_fp16_t *) params->wdata + 0;
    lm_ggml_fp16_t * const wdata_src = wdata + nk;

    for (int i2 = ip0; i2 < ip1; i2++) { // Cout
        float * dst_data = (float *)((char *) dst->data + i2*nb2);
        lm_ggml_fp16_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        for (int i11 = 0; i11 < ne11; i11++) {
            for (int i10 = 0; i10 < ne10; i10++) {
                const int i1n = i11*ne10*ne12 + i10*ne12;
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i00 = 0; i00 < ne00; i00++) {
                        float v = 0;
                        lm_ggml_vec_dot_f16(ne03, &v, 0,
                                wdata_src + i1n, 0,
                                wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                        dst_data[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                    }
                }
            }
        }
    }
}

// lm_ggml_compute_forward_pool_1d_sk_p0

static void lm_ggml_compute_forward_pool_1d_sk_p0(
        const struct lm_ggml_compute_params * params,
        const enum lm_ggml_op_pool op,
        const int k,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src = dst->src[0];

    assert(src->type == LM_GGML_TYPE_F32 || src->type == LM_GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const char * cdata = (const char *)src->data;
    const char * const data_end = cdata + lm_ggml_nbytes(src);
    float * drow = (float *)dst->data;

    const int64_t rs = dst->ne[0];

    while (cdata < data_end) {
        const void * srow = (const void *)cdata;
        int j = 0;
        for (int64_t i = 0; i < rs; ++i) {
            switch (op) {
                case LM_GGML_OP_POOL_AVG:   drow[i] = 0;        break;
                case LM_GGML_OP_POOL_MAX:   drow[i] = -FLT_MAX; break;
                case LM_GGML_OP_POOL_COUNT: LM_GGML_ABORT("fatal error");
            }
            for (int ki = 0; ki < k; ++ki) {
                const float srow_j = (src->type == LM_GGML_TYPE_F32) ? ((const float*)srow)[j] : LM_GGML_FP16_TO_FP32(((const lm_ggml_fp16_t*)srow)[j]);
                switch (op) {
                    case LM_GGML_OP_POOL_AVG:                         drow[i] += srow_j; break;
                    case LM_GGML_OP_POOL_MAX:   if (srow_j > drow[i]) drow[i]  = srow_j; break;
                    case LM_GGML_OP_POOL_COUNT:                       LM_GGML_ABORT("fatal error");
                }
                ++j;
            }
            switch (op) {
                case LM_GGML_OP_POOL_AVG:         drow[i] /= k; break;
                case LM_GGML_OP_POOL_MAX:                       break;
                case LM_GGML_OP_POOL_COUNT: LM_GGML_ABORT("fatal error");
            }
        }

        cdata += src->nb[1];
        drow  += rs;
    }
}

// lm_ggml_compute_forward_pool_1d

static void lm_ggml_compute_forward_pool_1d(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum lm_ggml_op_pool op = opts[0];
    const int k0 = opts[1];
    const int s0 = opts[2];
    const int p0 = opts[3];
    LM_GGML_ASSERT(p0 == 0); // padding not supported
    LM_GGML_ASSERT(k0 == s0); // only s = k supported

    lm_ggml_compute_forward_pool_1d_sk_p0(params, op, k0, dst);
}

// lm_ggml_compute_forward_pool_2d

static void lm_ggml_compute_forward_pool_2d(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src = dst->src[0];

    assert(src->type == LM_GGML_TYPE_F32 || src->type == LM_GGML_TYPE_F16);

    if (params->ith != 0) {
        return;
    }

    const int32_t * opts = (const int32_t *)dst->op_params;
    enum lm_ggml_op_pool op = opts[0];
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];
    const char * cdata = (const char*)src->data;
    const char * const data_end = cdata + lm_ggml_nbytes(src);

    const int64_t px = dst->ne[0];
    const int64_t py = dst->ne[1];
    const int64_t pa = px * py;

    float * dplane = (float *)dst->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            float * const drow = dplane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                float * const out =  drow + ox;
                switch (op) {
                    case LM_GGML_OP_POOL_AVG:     *out = 0;        break;
                    case LM_GGML_OP_POOL_MAX:     *out = -FLT_MAX; break;
                    case LM_GGML_OP_POOL_COUNT: LM_GGML_ABORT("fatal error");
                }

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                for (int ky = 0; ky < k1; ++ky) {
                    if (iy + ky < 0 || iy + ky >= src->ne[1]) continue;
                    const void * srow = (const void *)(cdata + src->nb[1] * (iy + ky));
                    for (int kx = 0; kx < k0; ++kx) {
                        int j = ix + kx;
                        if (j < 0 || j >= src->ne[0]) continue;
                        const float srow_j = (src->type == LM_GGML_TYPE_F32) ? ((const float*)srow)[j] : LM_GGML_FP16_TO_FP32(((const lm_ggml_fp16_t*)srow)[j]);
                        switch (op) {
                            case LM_GGML_OP_POOL_AVG:                     *out += srow_j; break;
                            case LM_GGML_OP_POOL_MAX: if (srow_j > *out)  *out  = srow_j; break;
                            case LM_GGML_OP_POOL_COUNT:               LM_GGML_ABORT("fatal error");
                        }
                    }
                }
                switch (op) {
                    case LM_GGML_OP_POOL_AVG:           *out /= ka; break;
                    case LM_GGML_OP_POOL_MAX:                       break;
                    case LM_GGML_OP_POOL_COUNT: LM_GGML_ABORT("fatal error");
                }
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// lm_ggml_compute_forward_upscale

static void lm_ggml_compute_forward_upscale_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(src0->type == LM_GGML_TYPE_F32);

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const float sf0 = (float)ne0/src0->ne[0];
    const float sf1 = (float)ne1/src0->ne[1];
    const float sf2 = (float)ne2/src0->ne[2];
    const float sf3 = (float)ne3/src0->ne[3];

    // TODO: optimize

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        const int64_t i03 = i3 / sf3;
        for (int64_t i2 = ith; i2 < ne2; i2 += nth) {
            const int64_t i02 = i2 / sf2;
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const int64_t i01 = i1 / sf1;
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    const int64_t i00 = i0 / sf0;

                    const float * x = (float *)((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          float * y = (float *)((char *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_upscale(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_upscale_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


// lm_ggml_compute_forward_pad

static void lm_ggml_compute_forward_pad_f32(
    const struct lm_ggml_compute_params * params,
          struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));
    LM_GGML_ASSERT( dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    float * dst_ptr = (float *) dst->data;

    // TODO: optimize

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = ith; i1 < ne1; i1 += nth) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;

                    const float * src_ptr = (const float *)((char *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);

                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        dst_ptr[dst_idx] = *src_ptr;
                    } else {
                        dst_ptr[dst_idx] = 0;
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_pad(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_pad_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}


// lm_ggml_compute_forward_arange

static void lm_ggml_compute_forward_arange_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    LM_GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const float start = lm_ggml_get_op_params_f32(dst, 0);
    const float stop  = lm_ggml_get_op_params_f32(dst, 1);
    const float step  = lm_ggml_get_op_params_f32(dst, 2);

    const int64_t steps = (int64_t) ceilf((stop - start) / step);

    LM_GGML_ASSERT(lm_ggml_nelements(dst) == steps);

    for (int64_t i = ith; i < steps; i+= nth) {
        float value = start + step * i;
        ((float *)dst->data)[i] = value;
    }
}

static void lm_ggml_compute_forward_arange(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {
    switch (dst->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_arange_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

static void lm_ggml_compute_forward_timestep_embedding_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const int dim = lm_ggml_get_op_params_i32(dst, 0);
    const int max_period = lm_ggml_get_op_params_i32(dst, 1);

    int half = dim / 2;

    for (int64_t i = 0; i < ne00; i++) {
        float * embed_data = (float *)((char *)  dst->data +  i*nb1);
        for (int64_t j = ith; j < half; j += nth) {
            float timestep = ((float *)src0->data)[i];
            float freq = (float)expf(-logf(max_period) * j / half);
            float arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        if (dim % 2 != 0 && ith == 0) {
            embed_data[dim] = 0.f;
        }
    }
}

static void lm_ggml_compute_forward_timestep_embedding(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_timestep_embedding_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_argsort

static void lm_ggml_compute_forward_argsort_f32(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    LM_GGML_ASSERT(nb0 == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nr = lm_ggml_nrows(src0);

    enum lm_ggml_sort_order order = (enum lm_ggml_sort_order) lm_ggml_get_op_params_i32(dst, 0);

    for (int64_t i = ith; i < nr; i += nth) {
        int32_t * dst_data = (int32_t *)((char *) dst->data + i*nb1);
        const float * src_data = (float *)((char *) src0->data + i*nb01);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        // C doesn't have a functional sort, so we do a bubble sort instead
        for (int64_t j = 0; j < ne0; j++) {
            for (int64_t k = j + 1; k < ne0; k++) {
                if ((order == LM_GGML_SORT_ORDER_ASC  && src_data[dst_data[j]] > src_data[dst_data[k]]) ||
                    (order == LM_GGML_SORT_ORDER_DESC && src_data[dst_data[j]] < src_data[dst_data[k]])) {
                    int32_t tmp = dst_data[j];
                    dst_data[j] = dst_data[k];
                    dst_data[k] = tmp;
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_argsort(
    const struct lm_ggml_compute_params * params,
    struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_argsort_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_flash_attn_ext

static void lm_ggml_compute_forward_flash_attn_ext_f16(
        const struct lm_ggml_compute_params * params,
        const struct lm_ggml_tensor * q,
        const struct lm_ggml_tensor * k,
        const struct lm_ggml_tensor * v,
        const struct lm_ggml_tensor * mask,
        struct lm_ggml_tensor * dst) {

    LM_GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;

    LM_GGML_ASSERT(ne0 == D);
    LM_GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    LM_GGML_ASSERT(nbq0 == lm_ggml_type_size(q->type));
    LM_GGML_ASSERT(nbk0 == lm_ggml_type_size(k->type));
    LM_GGML_ASSERT(nbv0 == lm_ggml_type_size(v->type));

    LM_GGML_ASSERT(neq0 == D);
    LM_GGML_ASSERT(nek0 == D);
    LM_GGML_ASSERT(nev0 == D);

    LM_GGML_ASSERT(neq1 == N);
    LM_GGML_ASSERT(nev0 == D);

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    const int64_t rk2 = neq2/nek2;
    const int64_t rk3 = neq3/nek3;

    const int64_t rv2 = neq2/nev2;
    const int64_t rv3 = neq3/nev3;

    // parallelize by q rows using lm_ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const uint32_t n_head      = neq2;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    enum lm_ggml_type    const k_vec_dot_type = type_traits[k->type].vec_dot_type;
    lm_ggml_from_float_t const q_to_vec_dot   = type_traits[k_vec_dot_type].from_float;
    lm_ggml_vec_dot_t    const kq_vec_dot     = type_traits[k->type].vec_dot;
    lm_ggml_to_float_t   const v_to_float     = type_traits[v->type].to_float;

    // loop over n_batch and n_head
    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        const uint32_t h = iq2; // head index
        const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        float S = 0.0f;      // sum
        float M = -INFINITY; // maximum KQ value

        float       * VKQ32 = (float       *) params->wdata + ith*(3*D + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*D); // (temporary) FP32 V buffer
        lm_ggml_fp16_t * VKQ16 = (lm_ggml_fp16_t *) (VKQ32 + 1*D); // (temporary) FP16 VKQ accumulator
        lm_ggml_fp16_t * Q_q   = (lm_ggml_fp16_t *) (VKQ32 + 2*D); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == LM_GGML_TYPE_F16) {
            memset(VKQ16, 0, D*sizeof(lm_ggml_fp16_t));
        } else {
            memset(VKQ32, 0, D*sizeof(float));
        }

        const lm_ggml_fp16_t * mp = mask ? (lm_ggml_fp16_t *)((char *) mask->data + iq1*mask->nb[1]) : NULL;

        // k indices
        const int ik3 = iq3 / rk3;
        const int ik2 = iq2 / rk2;

        // v indices
        const int iv3 = iq3 / rv3;
        const int iv2 = iq2 / rv2;

        const float * pq = (const float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, D);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf
        for (int64_t ic = 0; ic < nek1; ++ic) {
            const float mv = mp ? slope*LM_GGML_FP16_TO_FP32(mp[ic]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }

            float s; // KQ value

            const char * k_data = (const char *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(D, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale + mv; // scale KQ value and apply mask

            const float Mold = M;

            float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            float vs = 1.0f; // post-softmax KQ value, expf(s - M)

            const char * v_data = ((const char *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            if (v->type== LM_GGML_TYPE_F16) {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    lm_ggml_vec_scale_f16(D, VKQ16, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                lm_ggml_vec_mad_f16(D, VKQ16, (const lm_ggml_fp16_t *) v_data, vs);
            } else {
                if (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    lm_ggml_vec_scale_f32(D, VKQ32, ms);
                } else {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                v_to_float(v_data, V32, D);

                // V += v*expf(s - M)
                lm_ggml_vec_mad_f32(D, VKQ32, V32, vs);
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        if (v->type == LM_GGML_TYPE_F16) {
            for (int64_t d = 0; d < D; ++d) {
                VKQ32[d] = LM_GGML_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // V /= S
        const float S_inv = 1.0f/S;
        lm_ggml_vec_scale_f32(D, VKQ32, S_inv);

        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // original
        //memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

        // permute(0, 2, 1, 3)
        memcpy((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
    }
}

static void lm_ggml_compute_forward_flash_attn_ext(
        const struct lm_ggml_compute_params * params,
        const struct lm_ggml_tensor * q,
        const struct lm_ggml_tensor * k,
        const struct lm_ggml_tensor * v,
        const struct lm_ggml_tensor * mask,
        struct lm_ggml_tensor * dst) {
    switch (dst->op_params[2]) {
        case LM_GGML_PREC_DEFAULT:
        case LM_GGML_PREC_F32:
            {
                // uses F32 accumulators
                lm_ggml_compute_forward_flash_attn_ext_f16(params, q, k, v, mask, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_flash_attn_back

static void lm_ggml_compute_forward_flash_attn_back_f32(
        const struct lm_ggml_compute_params * params,
        const bool masked,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * q = dst->src[0];
    const struct lm_ggml_tensor * k = dst->src[1];
    const struct lm_ggml_tensor * v = dst->src[2];
    const struct lm_ggml_tensor * d = dst->src[3];

    LM_GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, ned, d,   ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nbd, d,   nb)
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    LM_GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t D = neq0;
    const int64_t N = neq1;
    const int64_t P = nek1 - N;
    const int64_t M = P + N;

    const int Mup  = lm_ggml_up(M, LM_GGML_SOFT_MAX_UNROLL);
    const int mxDM = MAX(D, Mup);

    // LM_GGML_ASSERT(ne0 == D);
    // LM_GGML_ASSERT(ne1 == N);
    LM_GGML_ASSERT(P >= 0);

    LM_GGML_ASSERT(nbq0 == sizeof(float));
    LM_GGML_ASSERT(nbk0 == sizeof(float));
    LM_GGML_ASSERT(nbv0 == sizeof(float));

    LM_GGML_ASSERT(neq0 == D);
    LM_GGML_ASSERT(nek0 == D);
    LM_GGML_ASSERT(nev1 == D);
    LM_GGML_ASSERT(ned0 == D);

    LM_GGML_ASSERT(neq1 == N);
    LM_GGML_ASSERT(nek1 == N + P);
    LM_GGML_ASSERT(nev1 == D);
    LM_GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    LM_GGML_ASSERT(nb0 == sizeof(float));
    LM_GGML_ASSERT(nb0 <= nb1);
    LM_GGML_ASSERT(nb1 <= nb2);
    LM_GGML_ASSERT(nb2 <= nb3);

    if (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    lm_ggml_barrier(params->shared);

    const int64_t elem_q = lm_ggml_nelements(q);
    const int64_t elem_k = lm_ggml_nelements(k);

    enum lm_ggml_type result_type = dst->type;
    LM_GGML_ASSERT(lm_ggml_blck_size(result_type) == 1);
    const size_t tsize = lm_ggml_type_size(result_type);

    const size_t offs_q = 0;
    const size_t offs_k = offs_q + LM_GGML_PAD(elem_q * tsize, LM_GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + LM_GGML_PAD(elem_k * tsize, LM_GGML_MEM_ALIGN);

    void * grad_q = (char *) dst->data;
    void * grad_k = (char *) dst->data + offs_k;
    void * grad_v = (char *) dst->data + offs_v;

    const size_t nbgq1 = nb0*neq0;
    const size_t nbgq2 = nb0*neq0*neq1;
    const size_t nbgq3 = nb0*neq0*neq1*neq2;

    const size_t nbgk1 = nb0*nek0;
    const size_t nbgk2 = nb0*nek0*nek1;
    const size_t nbgk3 = nb0*nek0*nek1*neq2;

    const size_t nbgv1 = nb0*nev0;
    const size_t nbgv2 = nb0*nev0*nev1;
    const size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using lm_ggml_vec_dot_f32

    // total rows in k
    const int nr = nek2*nek3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    int nrep = neq2/nek2;

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int ik3 = ir/(nek2);
        const int ik2 = ir - ik3*nek2;

        const int iq3 = ik3;
        const int id3 = ik3;
        const int iv3 = ik3;
        const int iv2 = ik2;

        for (int irep = 0; irep < nrep; ++irep) {
            const int iq2 = ik2 + irep*nek2;
            const int id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            for (int iq1 = 0; iq1 < neq1; ++iq1) {
                const int id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                float * S  = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 0*(mxDM+CACHE_LINE_SIZE_F32);
                float * SM = (float *) params->wdata + ith*2*(mxDM + CACHE_LINE_SIZE_F32) + 1*(mxDM+CACHE_LINE_SIZE_F32);

                for (int i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                const int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    const int ik1 = ic;

                    // S indices
                    const int i1 = ik1;

                    lm_ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                lm_ggml_vec_scale_f32(masked_begin, S, scale);

                for (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    float max = -INFINITY;
                    lm_ggml_vec_max_f32(masked_begin, &max, S);

                    lm_ggml_float sum = 0.0;
                    {
#ifdef LM_GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        lm_ggml_vec_sum_f32(Mup, &sum, SM);
#else
                        sum = lm_ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    assert(sum > 0.0);

                    sum = 1.0/sum;
                    lm_ggml_vec_scale_f32(masked_begin, SM, sum);

                }

                // step-by-step explanation
                {
                    // forward-process                    shape      grads from backward process
                    // parallel_for ik2,ik3:
                    //  for irep:
                    //   iq2 = ik2 + irep*nek2
                    //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
                    //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
                    //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
                    //   for iq1:
                    //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
                    //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
                    //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
                    //    S0     = -Inf                   [D,1,1,1]
                    //   ~S1[i]  = dot(kcur[:D,i], qcur)
                    //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
                    //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
                    //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
                    //   ~S5[i]  = dot(vcur[:,i], S4)
                    //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
                    //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
                    //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
                    // dst                               backward-/ grad[dst]                 = d
                    //
                    // output gradients with their dependencies:
                    //
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S4]   = grad[S5] @ vcur
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[vcur] = grad[S5].T @ S4
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // in post-order:
                    //
                    // S1         = qcur @ kcur.T
                    // S2         = S1 * scale
                    // S3         = diag_mask_inf(S2, P)
                    // S4         = softmax(S3)
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // using less variables (SM=S4):
                    //
                    // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
                    // SM            = softmax(S)
                    // S             = d[:D,iq1,iq2,iq3] @ vcur
                    // dot_SM_gradSM = dot(SM, S)
                    // S             = SM * (S - dot(SM, S))
                    // S             = diag_mask_zero(S, P) * scale
                    //
                    // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
                    // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
                    // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
                }

                // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // for ic:
                //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
                // exclude known future zero S[..] values from operation
                lm_ggml_vec_set_f32(masked_begin, S, 0);
                for (int64_t ic = 0; ic < D; ++ic) {
                    lm_ggml_vec_mad_f32(masked_begin,
                            S,
                             (float *) ((char *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                float dot_SM_gradSM = 0;
                lm_ggml_vec_dot_f32 (masked_begin, &dot_SM_gradSM, 0, SM, 0, S, 0, 1);
                lm_ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
                lm_ggml_vec_mul_f32 (masked_begin, S, S, SM);

                // S = diag_mask_zero(S, P) * scale
                // already done by above lm_ggml_vec_set_f32

                // exclude known zero S[..] values from operation
                lm_ggml_vec_scale_f32(masked_begin, S, scale);

                // S    shape [M,1]
                // SM   shape [M,1]
                // kcur shape [D,M]
                // qcur shape [D,1]
                // vcur shape [M,D]

                // grad[q][:D,iq1,iq2,iq3] += S @ kcur
                // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
                // for ic:
                //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    lm_ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (float *) ((char *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                for (int64_t ic = 0; ic < masked_begin; ++ic) {
                    lm_ggml_vec_mad_f32(D,
                            (float *) ((char *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (float *) ((char *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                for (int64_t ic = 0; ic < D; ++ic) {
                    lm_ggml_vec_mad_f32(masked_begin,
                            (float *) ((char *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(float *) ((char *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_flash_attn_back(
        const struct lm_ggml_compute_params * params,
        const bool masked,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * q = dst->src[0];

    switch (q->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_ssm_conv

static void lm_ggml_compute_forward_ssm_conv_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    const struct lm_ggml_tensor * src0 = dst->src[0]; // conv_state
    const struct lm_ggml_tensor * src1 = dst->src[1]; // x
    const struct lm_ggml_tensor * src2 = dst->src[2]; // conv1d.weight
    const struct lm_ggml_tensor * src3 = dst->src[3]; // state_seq

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc   = src2->ne[0]; // d_conv
    const int nr   = src0->ne[1]; // d_inner
    const int n_t  = src1->ne[1]; // n_tokens
    const int n_kv = src0->ne[2]; // max number of sequences in the batch

    LM_GGML_ASSERT((nr*n_t) + (nc*nr*n_kv) == lm_ggml_nelements(dst));
    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src1->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src2->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src3->nb[0] == sizeof(int32_t));
    LM_GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    // for use with the destination state offset between sequences
    LM_GGML_ASSERT(src2->nb[2] == src2->ne[1]*src2->ne[0]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    if (n_kv > 1) {
        // multiple sequences means it's hard to know when it's the first time a state is read,
        // so copy them all over to the destination, just to be sure.
        for (int i3 = 0; i3 < n_kv; ++i3) {
            float * s0 = (float *) ((char *) src0->data + ir0*(src0->nb[1]) + i3*(src0->nb[2]));
            float * s  = (float *) ((char *)  dst->data + ir0*(src2->nb[1]) + i3*(src2->nb[2]) + nr*n_t*sizeof(float));
            // can't use memcpy because of d_conv vs d_conv - 1
            for (int i1 = 0; i1 < ir; ++i1) {
                for (int i0 = 0; i0 < nc - 1; ++i0) {
                    // copy s0 to last (d_conv - 1) columns of s
                    s[1 + i0 + i1*nc] = s0[i0 + i1*(nc - 1)];
                }
            }
        }
    }

    for (int i2 = 0; i2 < n_t; ++i2) {
        int32_t * sq = (int32_t *) ((char *) src3->data +  i2*(src3->nb[1])); // {n_kv, n_tokens}
        float *   x  = (float *)   ((char *)  dst->data + ir0*sizeof(float) + i2*(nr*sizeof(float))); // {d_inner, n_tokens}
        float *   s  = (float *)   ((char *)  dst->data + ir0*(src2->nb[1]) + sq[0]*(src2->nb[2]) + nr*n_t*sizeof(float)); // {d_conv, d_inner, n_kv}
        float *   s0; // {d_conv - 1, d_inner, n_kv}
        float *   x0 = (float *)   ((char *) src1->data + ir0*(src1->nb[0]) + i2*(src1->nb[1])); // {d_inner, n_tokens}
        float *   c  = (float *)   ((char *) src2->data + ir0*(src2->nb[1])); // {d_conv, d_inner}
        int ne0s0;

        LM_GGML_ASSERT(0 <= sq[0] && sq[0] < n_kv);

        // avoid needing to copy the state for the first token
        if (i2 == 0) {
            s0 = (float *) ((char *) src0->data + ir0*(src0->nb[1]) + sq[0]*(src0->nb[2])); // {d_conv - 1, d_inner, n_kv}
            ne0s0 = src0->ne[0];
        } else {
            // the source is the last (d_conv - 1) columns of the destination
            s0 = s + 1;
            ne0s0 = nc;
        }

        // d_inner
        for (int i1 = 0; i1 < ir; ++i1) {
            // shift state left
            for (int i0 = 0; i0 < nc - 1; ++i0) {
                s[i0 + i1*nc] = s0[i0 + i1*ne0s0];
            }
            // insert x on the last column
            s[(nc - 1) + i1*nc] = x0[i1];
        }

        // handle copies when there are multiple output states
        for (int i3 = 1; i3 < n_kv; ++i3) {
            int32_t seq = sq[i3];
            if (0 <= seq && seq < n_kv) {
                float * s1 = s + (seq - sq[0])*nc*nr;
                memcpy(s1, s, nc*ir*sizeof(float));
            } else {
                // stop at negative or too big seq_ids
                break;
            }
        }

        // it seems a little faster when this is separate from the state shift
        for (int i1 = 0; i1 < ir; ++i1) {
            // rowwise dot product
            float sumf = 0.0f;
            for (int i0 = 0; i0 < nc; ++i0) {
                int i = i0 + i1*nc;
                sumf += s[i] * c[i];
            }
            x[i1] = sumf;
        }
    }
}

static void lm_ggml_compute_forward_ssm_conv(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_ssm_conv_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_ssm_scan

static void lm_ggml_compute_forward_ssm_scan_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    const struct lm_ggml_tensor * src0 = dst->src[0]; // s
    const struct lm_ggml_tensor * src1 = dst->src[1]; // x
    const struct lm_ggml_tensor * src2 = dst->src[2]; // dt
    const struct lm_ggml_tensor * src3 = dst->src[3]; // A
    const struct lm_ggml_tensor * src4 = dst->src[4]; // B
    const struct lm_ggml_tensor * src5 = dst->src[5]; // C
    const struct lm_ggml_tensor * src6 = dst->src[6]; // sq

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t nc   = src0->ne[0]; // d_state
    const int64_t nr   = src0->ne[1]; // d_inner
    const int64_t n_t  = src1->ne[1]; // number of tokens in the batch
    const int64_t n_kv = src0->ne[2]; // max number of sequences in the batch

    LM_GGML_ASSERT(lm_ggml_nelements(src1) + lm_ggml_nelements(src0) == lm_ggml_nelements(dst));
    LM_GGML_ASSERT(src0->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src1->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src2->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src3->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src4->nb[0] == sizeof(float));
    LM_GGML_ASSERT(src5->nb[0] == sizeof(float));
    // required for the dot product between s and C, and when copying the states
    LM_GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));
    // required for per-sequence offsets for states
    LM_GGML_ASSERT(src0->nb[2] == src0->ne[0]*src0->ne[1]*sizeof(float));
    // required to get correct offset for state destination (i.e. src1->nb[2])
    LM_GGML_ASSERT(src1->nb[2] == src1->ne[0]*src1->ne[1]*sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    const int ir  = ir1 - ir0;

    if (n_kv > 1) {
        // it's hard to know if the source states have already been copied
        // when there are multiple, so copy them already.
        for (int i3 = 0; i3 < n_kv; ++i3) {
            float * s0 = (float *) ((char *) src0->data + ir0*(src0->nb[1]) + i3*(src0->nb[2]));
            float * s  = (float *) ((char *)  dst->data + ir0*(src0->nb[1]) + i3*(src0->nb[2]) + src1->nb[2]);
            memcpy(s, s0, nc*ir*sizeof(float));
        }
    }

    for (int i2 = 0; i2 < n_t; ++i2) {
        int32_t * sq = (int32_t *) ((char *) src6->data +  i2*(src6->nb[1])); // {n_kv, n_tokens}
        float *   y  = (float *)   ((char *)  dst->data + ir0*(src1->nb[0]) +    i2*(src1->nb[1])); // {d_inner, n_tokens}
        float *   s  = (float *)   ((char *)  dst->data + ir0*(src0->nb[1]) + sq[0]*(src0->nb[2]) + src1->nb[2]); // {d_state, d_inner, n_kv}
        float *   s0;
        float *   x  = (float *)   ((char *) src1->data + ir0*(src1->nb[0]) + i2*(src1->nb[1])); // {d_inner, n_tokens}
        float *   dt = (float *)   ((char *) src2->data + ir0*(src2->nb[0]) + i2*(src2->nb[1])); // {d_inner, n_tokens}
        float *   A  = (float *)   ((char *) src3->data + ir0*(src3->nb[1])); // {d_state, d_inner}
        float *   B  = (float *)   ((char *) src4->data +  i2*(src4->nb[1])); // {d_state, n_tokens}
        float *   C  = (float *)   ((char *) src5->data +  i2*(src5->nb[1])); // {d_state, n_tokens}

        LM_GGML_ASSERT(0 <= sq[0] && sq[0] < n_kv);

        // avoid needing to copy the state for the first token
        if (i2 == 0) {
            s0 = (float *) ((char *) src0->data + ir0*(src0->nb[1]) + sq[0]*(src0->nb[2])); // {d_state, d_inner, n_kv}
        } else {
            // otherwise the source is the same as the destination
            s0 = s;
        }

        // d_inner
        for (int i1 = 0; i1 < ir; ++i1) {
            // ref: https://github.com/state-spaces/mamba/blob/34076d664838588a3c97727b263478ab9f621a07/mamba_ssm/ops/triton/selective_state_update.py#L78
            float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
            float x_dt = x[i1] * dt_soft_plus;
            float sumf = 0.0f;
            // d_state
            for (int i0 = 0; i0 < nc; ++i0) {
                int i = i0 + i1*nc;
                // state = prev_state * dA + dB * x
                float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
                // y = rowwise_dotprod(state, C)
                sumf += state * C[i0];
                s[i] = state;
            }
            y[i1] = sumf;
        }

        // handle copies when there are multiple output states
        for (int i3 = 1; i3 < n_kv; ++i3) {
            int32_t seq = sq[i3];
            if (0 <= seq && seq < n_kv) {
                float * s1 = s + (seq - sq[0])*nc*nr;
                memcpy(s1, s, nc*ir*sizeof(float));
            } else {
                // stop at negative or too big seq_ids
                break;
            }
        }
    }
}

static void lm_ggml_compute_forward_ssm_scan(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    switch (dst->src[0]->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_ssm_scan_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_win_part

static void lm_ggml_compute_forward_win_part_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    UNUSED(params);

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t nep0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t nep1 = ((const int32_t *)(dst->op_params))[1];
    const int32_t w    = ((const int32_t *)(dst->op_params))[2];

    assert(ne00 == ne0);
    assert(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    for (int py = 0; py < nep1; ++py) {
        for (int px = 0; px < nep0; ++px) {
            const int64_t i3 = py*nep0 + px;
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                for (int64_t i1 = 0; i1 < ne1; ++i1) {
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        const int64_t i02 = py*w + i2;
                        const int64_t i01 = px*w + i1;
                        const int64_t i00 = i0;

                        const int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        const int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        if (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((float *) dst->data)[i] = 0.0f;
                        } else {
                            ((float *) dst->data)[i] = ((float *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_win_part(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_win_part_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_win_unpart

static void lm_ggml_compute_forward_win_unpart_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    UNUSED(params);

    const struct lm_ggml_tensor * src0 = dst->src[0];

    LM_GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    LM_GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    const int32_t w = ((const int32_t *)(dst->op_params))[0];

    // padding
    const int px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    const int npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    assert(ne0 == ne00);

    // TODO: optimize / multi-thread
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int ip2 = i2/w;
                const int ip1 = i1/w;

                const int64_t i02 = i2%w;
                const int64_t i01 = i1%w;
                const int64_t i00 = i0;

                const int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                const int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((float *) dst->data)[j] = ((float *) src0->data)[i];
            }
        }
    }
}

static void lm_ggml_compute_forward_win_unpart(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_win_unpart_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

//gmml_compute_forward_unary

static void lm_ggml_compute_forward_unary(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const enum lm_ggml_unary_op op = lm_ggml_get_unary_op(dst);

    switch (op) {
        case LM_GGML_UNARY_OP_ABS:
            {
                lm_ggml_compute_forward_abs(params, dst);
            } break;
        case LM_GGML_UNARY_OP_SGN:
            {
                lm_ggml_compute_forward_sgn(params, dst);
            } break;
        case LM_GGML_UNARY_OP_NEG:
            {
                lm_ggml_compute_forward_neg(params, dst);
            } break;
        case LM_GGML_UNARY_OP_STEP:
            {
                lm_ggml_compute_forward_step(params, dst);
            } break;
        case LM_GGML_UNARY_OP_TANH:
            {
                lm_ggml_compute_forward_tanh(params, dst);
            } break;
        case LM_GGML_UNARY_OP_ELU:
            {
                lm_ggml_compute_forward_elu(params, dst);
            } break;
        case LM_GGML_UNARY_OP_RELU:
            {
                lm_ggml_compute_forward_relu(params, dst);
            } break;
        case LM_GGML_UNARY_OP_SIGMOID:
            {
                lm_ggml_compute_forward_sigmoid(params, dst);
            } break;
        case LM_GGML_UNARY_OP_GELU:
            {
                lm_ggml_compute_forward_gelu(params, dst);
            } break;
        case LM_GGML_UNARY_OP_GELU_QUICK:
            {
                lm_ggml_compute_forward_gelu_quick(params, dst);
            } break;
        case LM_GGML_UNARY_OP_SILU:
            {
                lm_ggml_compute_forward_silu(params, dst);
            } break;
        case LM_GGML_UNARY_OP_HARDSWISH:
            {
                lm_ggml_compute_forward_hardswish(params, dst);
            } break;
        case LM_GGML_UNARY_OP_HARDSIGMOID:
            {
                lm_ggml_compute_forward_hardsigmoid(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_get_rel_pos

static void lm_ggml_compute_forward_get_rel_pos_f16(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {
    UNUSED(params);

    const struct lm_ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    LM_GGML_TENSOR_UNARY_OP_LOCALS

    const int64_t w = ne1;

    lm_ggml_fp16_t * src0_data = (lm_ggml_fp16_t *) src0->data;
    lm_ggml_fp16_t * dst_data  = (lm_ggml_fp16_t *) dst->data;

    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t pos = (w - i1 - 1) + i2;
            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

static void lm_ggml_compute_forward_get_rel_pos(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F16:
        case LM_GGML_TYPE_BF16:
            {
                lm_ggml_compute_forward_get_rel_pos_f16(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_add_rel_pos

static void lm_ggml_compute_forward_add_rel_pos_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * src2 = dst->src[2];

    const bool inplace = (bool) ((int32_t *) dst->op_params)[0];
    if (!inplace) {
        if (params->ith == 0) {
            memcpy((char *) dst->data, (char *) src0->data, lm_ggml_nbytes(dst));
        }
        lm_ggml_barrier(params->shared);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    float * src1_data = (float *) src1->data;
    float * src2_data = (float *) src2->data;
    float * dst_data  = (float *) dst->data;

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    const int ith = params->ith;
    const int nth = params->nth;

    // total patches in dst
    const int np = ne13;

    // patches per thread
    const int dp = (np + nth - 1)/nth;

    // patch range for this thread
    const int ip0 = dp*ith;
    const int ip1 = MIN(ip0 + dp, np);

    for (int64_t i13 = ip0; i13 < ip1; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = 0; i11 < ne11; ++i11) {
                const int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                for (int64_t i10 = 0; i10 < ne10; ++i10) {
                    const int64_t jp0  = jp1 + i10;
                    const float src1_e = src1_data[jp0];
                    const float src2_e = src2_data[jp0];

                    const int64_t jdh = jp0 * ne10;
                    const int64_t jdw = jdh - (ne10 - 1) * i10;

                    for (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

static void lm_ggml_compute_forward_add_rel_pos(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_add_rel_pos_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_map_unary

static void lm_ggml_compute_forward_map_unary_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_unary_op_f32_t fun) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void lm_ggml_compute_forward_map_unary(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_unary_op_f32_t fun) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_map_unary_f32(params, dst, fun);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_map_binary

static void lm_ggml_compute_forward_map_binary_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_binary_op_f32_t fun) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    assert(lm_ggml_is_contiguous_1(src0));
    assert(lm_ggml_is_contiguous_1(src1));
    assert(lm_ggml_is_contiguous_1(dst));
    assert(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int n  = lm_ggml_nrows(src0);
    const int nc = src0->ne[0];

    for (int i = 0; i < n; i++) {
        fun(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void lm_ggml_compute_forward_map_binary(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_binary_op_f32_t fun) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_map_binary_f32(params, dst, fun);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_map_custom1

static void lm_ggml_compute_forward_map_custom1_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_custom1_op_f32_t fun) {

    const struct lm_ggml_tensor * a = dst->src[0];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a);
}

// lm_ggml_compute_forward_map_custom2

static void lm_ggml_compute_forward_map_custom2_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_custom2_op_f32_t fun) {

    const struct lm_ggml_tensor * a = dst->src[0];
    const struct lm_ggml_tensor * b = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a, b);
}

// lm_ggml_compute_forward_map_custom3

static void lm_ggml_compute_forward_map_custom3_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst,
        const lm_ggml_custom3_op_f32_t fun) {

    const struct lm_ggml_tensor * a = dst->src[0];
    const struct lm_ggml_tensor * b = dst->src[1];
    const struct lm_ggml_tensor * c = dst->src[1];

    if (params->ith != 0) {
        return;
    }

    fun(dst, a, b, c);
}

// lm_ggml_compute_forward_map_custom1

static void lm_ggml_compute_forward_map_custom1(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * a = dst->src[0];

    struct lm_ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// lm_ggml_compute_forward_map_custom2

static void lm_ggml_compute_forward_map_custom2(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * a = dst->src[0];
    const struct lm_ggml_tensor * b = dst->src[1];

    struct lm_ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// lm_ggml_compute_forward_map_custom3

static void lm_ggml_compute_forward_map_custom3(
        const struct lm_ggml_compute_params * params,
              struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * a = dst->src[0];
    const struct lm_ggml_tensor * b = dst->src[1];
    const struct lm_ggml_tensor * c = dst->src[2];

    struct lm_ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, sizeof(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// lm_ggml_compute_forward_cross_entropy_loss

static void lm_ggml_compute_forward_cross_entropy_loss_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];

    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));
    LM_GGML_ASSERT(lm_ggml_is_scalar(dst));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1));

    const int ith = params->ith;
    const int nth = params->nth;

    float * sums = (float *) params->wdata;

    // TODO: handle transposed/permuted matrices
    const int nc = src0->ne[0];
    const int nr = lm_ggml_nrows(src0);

    LM_GGML_ASSERT(params->wsize >= sizeof(float) * (nth + nth * nc));

    if (ith == 0) {
        memset(sums, 0, sizeof(float) * (nth + nth * nc));
    }
    lm_ggml_barrier(params->shared);

    const double eps = 1e-9;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * s0 = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * s1 = (float *)((char *) src1->data + i1*src1->nb[1]);
        float * st = ((float *) params->wdata) + nth + ith*nc;

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        // soft_max
        float max = -INFINITY;
        lm_ggml_vec_max_f32(nc, &max, s0);
        lm_ggml_float sum = lm_ggml_vec_soft_max_f32(nc, st, s0, max);
        assert(sum > 0.0);
        sum = (1.0 - eps) / sum;

        // avoid log(0) by rescaling from [0..1] to [eps..1]
        lm_ggml_vec_scale_f32(nc, st, sum);
        lm_ggml_vec_add1_f32(nc, st, st, eps);
        lm_ggml_vec_log_f32(nc, st, st);
        lm_ggml_vec_mul_f32(nc, st, st, s1);

        float st_sum = 0;
        lm_ggml_vec_sum_f32(nc, &st_sum, st);
        sums[ith] += st_sum;

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(st[i]));
            assert(!isinf(st[i]));
        }
#endif
    }
    lm_ggml_barrier(params->shared);

    if (ith == 0) {
        float * dp = (float *) dst->data;
        lm_ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (float) nr;
    }
}

static void lm_ggml_compute_forward_cross_entropy_loss(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

// lm_ggml_compute_forward_cross_entropy_loss_back

static void lm_ggml_compute_forward_cross_entropy_loss_back_f32(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];
    const struct lm_ggml_tensor * src1 = dst->src[1];
    const struct lm_ggml_tensor * opt0 = dst->src[2];

    LM_GGML_ASSERT(lm_ggml_is_contiguous(dst));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src0));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(src1));
    LM_GGML_ASSERT(lm_ggml_is_contiguous(opt0));
    LM_GGML_ASSERT(lm_ggml_are_same_shape(src0, src1) && lm_ggml_are_same_shape(src0, dst));

    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const double eps = 1e-9;

    // TODO: handle transposed/permuted matrices
    const int64_t nc = src0->ne[0];
    const int64_t nr = lm_ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    float * d   = (float *) opt0->data;

    for (int64_t i1 = ir0; i1 < ir1; i1++) {
        float * ds0 = (float *)((char *) dst->data  + i1*dst->nb[1]);
        float * s0  = (float *)((char *) src0->data + i1*src0->nb[1]);
        float * s1  = (float *)((char *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(s0[i]));
            assert(!isnan(s1[i]));
        }
#endif

        // soft_max
        float max = -INFINITY;
        lm_ggml_vec_max_f32(nc, &max, s0);
        lm_ggml_float sum = lm_ggml_vec_soft_max_f32(nc, ds0, s0, max);
        assert(sum > 0.0);
        sum = (1.0 - eps) / sum;

        // grad(src0) = (softmax(src0) - src1) * grad(cross_entropy_loss(src0, src1)) / nr
        lm_ggml_vec_scale_f32(nc, ds0, sum);
        lm_ggml_vec_add1_f32(nc, ds0, ds0, eps);
        lm_ggml_vec_sub_f32(nc, ds0, ds0, s1);
        lm_ggml_vec_scale_f32(nc, ds0, d[0] / (float) nr);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(ds0[i]));
            assert(!isinf(ds0[i]));
        }
#endif
    }
}

static void lm_ggml_compute_forward_cross_entropy_loss_back(
        const struct lm_ggml_compute_params * params,
        struct lm_ggml_tensor * dst) {

    const struct lm_ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case LM_GGML_TYPE_F32:
            {
                lm_ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } break;
        default:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
}

/////////////////////////////////

static void lm_ggml_compute_forward(struct lm_ggml_compute_params * params, struct lm_ggml_tensor * tensor) {
    LM_GGML_ASSERT(params);

    if (tensor->op == LM_GGML_OP_NONE || lm_ggml_is_empty(tensor)) {
        return;
    }

    switch (tensor->op) {
        case LM_GGML_OP_DUP:
            {
                lm_ggml_compute_forward_dup(params, tensor);
            } break;
        case LM_GGML_OP_ADD:
            {
                lm_ggml_compute_forward_add(params, tensor);
            } break;
        case LM_GGML_OP_ADD1:
            {
                lm_ggml_compute_forward_add1(params, tensor);
            } break;
        case LM_GGML_OP_ACC:
            {
                lm_ggml_compute_forward_acc(params, tensor);
            } break;
        case LM_GGML_OP_SUB:
            {
                lm_ggml_compute_forward_sub(params, tensor);
            } break;
        case LM_GGML_OP_MUL:
            {
                lm_ggml_compute_forward_mul(params, tensor);
            } break;
        case LM_GGML_OP_DIV:
            {
                lm_ggml_compute_forward_div(params, tensor);
            } break;
        case LM_GGML_OP_SQR:
            {
                lm_ggml_compute_forward_sqr(params, tensor);
            } break;
        case LM_GGML_OP_SQRT:
            {
                lm_ggml_compute_forward_sqrt(params, tensor);
            } break;
        case LM_GGML_OP_LOG:
            {
                lm_ggml_compute_forward_log(params, tensor);
            } break;
        case LM_GGML_OP_SUM:
            {
                lm_ggml_compute_forward_sum(params, tensor);
            } break;
        case LM_GGML_OP_SUM_ROWS:
            {
                lm_ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case LM_GGML_OP_MEAN:
            {
                lm_ggml_compute_forward_mean(params, tensor);
            } break;
        case LM_GGML_OP_ARGMAX:
            {
                lm_ggml_compute_forward_argmax(params, tensor);
            } break;
        case LM_GGML_OP_REPEAT:
            {
                lm_ggml_compute_forward_repeat(params, tensor);
            } break;
        case LM_GGML_OP_REPEAT_BACK:
            {
                lm_ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case LM_GGML_OP_CONCAT:
            {
                lm_ggml_compute_forward_concat(params, tensor);
            } break;
        case LM_GGML_OP_SILU_BACK:
            {
                lm_ggml_compute_forward_silu_back(params, tensor);
            } break;
        case LM_GGML_OP_NORM:
            {
                lm_ggml_compute_forward_norm(params, tensor);
            } break;
        case LM_GGML_OP_RMS_NORM:
            {
                lm_ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case LM_GGML_OP_RMS_NORM_BACK:
            {
                lm_ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case LM_GGML_OP_GROUP_NORM:
            {
                lm_ggml_compute_forward_group_norm(params, tensor);
            } break;
        case LM_GGML_OP_MUL_MAT:
            {
                lm_ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case LM_GGML_OP_MUL_MAT_ID:
            {
                lm_ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case LM_GGML_OP_OUT_PROD:
            {
                lm_ggml_compute_forward_out_prod(params, tensor);
            } break;
        case LM_GGML_OP_SCALE:
            {
                lm_ggml_compute_forward_scale(params, tensor);
            } break;
        case LM_GGML_OP_SET:
            {
                lm_ggml_compute_forward_set(params, tensor);
            } break;
        case LM_GGML_OP_CPY:
            {
                lm_ggml_compute_forward_cpy(params, tensor);
            } break;
        case LM_GGML_OP_CONT:
            {
                lm_ggml_compute_forward_cont(params, tensor);
            } break;
        case LM_GGML_OP_RESHAPE:
            {
                lm_ggml_compute_forward_reshape(params, tensor);
            } break;
        case LM_GGML_OP_VIEW:
            {
                lm_ggml_compute_forward_view(params, tensor);
            } break;
        case LM_GGML_OP_PERMUTE:
            {
                lm_ggml_compute_forward_permute(params, tensor);
            } break;
        case LM_GGML_OP_TRANSPOSE:
            {
                lm_ggml_compute_forward_transpose(params, tensor);
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                lm_ggml_compute_forward_get_rows(params, tensor);
            } break;
        case LM_GGML_OP_GET_ROWS_BACK:
            {
                lm_ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case LM_GGML_OP_DIAG:
            {
                lm_ggml_compute_forward_diag(params, tensor);
            } break;
        case LM_GGML_OP_DIAG_MASK_INF:
            {
                lm_ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case LM_GGML_OP_DIAG_MASK_ZERO:
            {
                lm_ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                lm_ggml_compute_forward_soft_max(params, tensor);
            } break;
        case LM_GGML_OP_SOFT_MAX_BACK:
            {
                lm_ggml_compute_forward_soft_max_back(params, tensor);
            } break;
        case LM_GGML_OP_ROPE:
            {
                lm_ggml_compute_forward_rope(params, tensor);
            } break;
        case LM_GGML_OP_ROPE_BACK:
            {
                lm_ggml_compute_forward_rope_back(params, tensor);
            } break;
        case LM_GGML_OP_CLAMP:
            {
                lm_ggml_compute_forward_clamp(params, tensor);
            } break;
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
            {
                lm_ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case LM_GGML_OP_IM2COL:
            {
                lm_ggml_compute_forward_im2col(params, tensor);
            } break;
        case LM_GGML_OP_CONV_TRANSPOSE_2D:
            {
                lm_ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case LM_GGML_OP_POOL_1D:
            {
                lm_ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case LM_GGML_OP_POOL_2D:
            {
                lm_ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case LM_GGML_OP_UPSCALE:
            {
                lm_ggml_compute_forward_upscale(params, tensor);
            } break;
        case LM_GGML_OP_PAD:
            {
                lm_ggml_compute_forward_pad(params, tensor);
            } break;
        case LM_GGML_OP_ARANGE:
            {
                lm_ggml_compute_forward_arange(params, tensor);
            } break;
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
            {
                lm_ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case LM_GGML_OP_ARGSORT:
            {
                lm_ggml_compute_forward_argsort(params, tensor);
            } break;
        case LM_GGML_OP_LEAKY_RELU:
            {
                lm_ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case LM_GGML_OP_FLASH_ATTN_EXT:
            {
                lm_ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
            } break;
        case LM_GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = lm_ggml_get_op_params_i32(tensor, 0);
                LM_GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                lm_ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case LM_GGML_OP_SSM_CONV:
            {
                lm_ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case LM_GGML_OP_SSM_SCAN:
            {
                lm_ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case LM_GGML_OP_WIN_PART:
            {
                lm_ggml_compute_forward_win_part(params, tensor);
            } break;
        case LM_GGML_OP_WIN_UNPART:
            {
                lm_ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case LM_GGML_OP_UNARY:
            {
                lm_ggml_compute_forward_unary(params, tensor);
            } break;
        case LM_GGML_OP_GET_REL_POS:
            {
                lm_ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case LM_GGML_OP_ADD_REL_POS:
            {
                lm_ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case LM_GGML_OP_MAP_UNARY:
            {
                lm_ggml_unary_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                lm_ggml_compute_forward_map_unary(params, tensor, fun);
            }
            break;
        case LM_GGML_OP_MAP_BINARY:
            {
                lm_ggml_binary_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                lm_ggml_compute_forward_map_binary(params, tensor, fun);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM1_F32:
            {
                lm_ggml_custom1_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                lm_ggml_compute_forward_map_custom1_f32(params, tensor, fun);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM2_F32:
            {
                lm_ggml_custom2_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                lm_ggml_compute_forward_map_custom2_f32(params, tensor, fun);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM3_F32:
            {
                lm_ggml_custom3_op_f32_t fun;
                memcpy(&fun, tensor->op_params, sizeof(fun));
                lm_ggml_compute_forward_map_custom3_f32(params, tensor, fun);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM1:
            {
                lm_ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM2:
            {
                lm_ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case LM_GGML_OP_MAP_CUSTOM3:
            {
                lm_ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS:
            {
                lm_ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                lm_ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case LM_GGML_OP_NONE:
            {
                // nop
            } break;
        case LM_GGML_OP_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
    }
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
    lm_ggml_build_backward_expand(ctx, gf, gb_tmp, true);

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

// functions to change gradients considering the case that input a might be initial gradient with zero value

static struct lm_ggml_tensor * lm_ggml_add_or_set(struct lm_ggml_context * ctx, struct lm_ggml_tensor * a, struct lm_ggml_tensor * b, struct lm_ggml_hash_set * zero_table) {
    if (lm_ggml_hash_contains(zero_table, a)) {
        return b;
    } else {
        return lm_ggml_add_impl(ctx, a, b, false);
    }
}

static struct lm_ggml_tensor * lm_ggml_acc_or_set(struct lm_ggml_context * ctx, struct lm_ggml_tensor * a, struct lm_ggml_tensor * b, size_t nb1, size_t nb2, size_t nb3, size_t offset, struct lm_ggml_hash_set * zero_table) {
    if (lm_ggml_hash_contains(zero_table, a)) {
        struct lm_ggml_tensor * a_zero = lm_ggml_scale(ctx, a, 0.0f);
        return lm_ggml_acc_impl(ctx, a_zero, b, nb1, nb2, nb3, offset, false);
    } else {
        return lm_ggml_acc_impl(ctx, a, b, nb1, nb2, nb3, offset, false);
    }
}

static struct lm_ggml_tensor * lm_ggml_add1_or_set(struct lm_ggml_context * ctx, struct lm_ggml_tensor * a, struct lm_ggml_tensor * b, struct lm_ggml_hash_set * zero_table) {
    if (lm_ggml_hash_contains(zero_table, a)) {
        return lm_ggml_repeat(ctx, b, a);
    } else {
        return lm_ggml_add1_impl(ctx, a, b, false);
    }
}

static struct lm_ggml_tensor * lm_ggml_sub_or_set(struct lm_ggml_context * ctx, struct lm_ggml_tensor * a, struct lm_ggml_tensor * b, struct lm_ggml_hash_set * zero_table) {
    if (lm_ggml_hash_contains(zero_table, a)) {
        return lm_ggml_neg(ctx, b);
    } else {
        return lm_ggml_sub_impl(ctx, a, b, false);
    }
}

static void lm_ggml_compute_backward(struct lm_ggml_context * ctx, struct lm_ggml_tensor * tensor, struct lm_ggml_hash_set * zero_table) {
    struct lm_ggml_tensor * src0 = tensor->src[0];
    struct lm_ggml_tensor * src1 = tensor->src[1];
    struct lm_ggml_tensor * src2 = tensor->src[2];

    switch (tensor->op) {
        case LM_GGML_OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
            } break;
        case LM_GGML_OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    src1->grad = lm_ggml_add_or_set(ctx, src1->grad, tensor->grad, zero_table);
                }
            } break;
        case LM_GGML_OP_ADD1:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    src1->grad = lm_ggml_add_or_set(ctx,
                        src1->grad,
                        lm_ggml_mean(ctx, tensor->grad), // TODO: should probably be sum instead of mean
                        zero_table);
                }
            } break;
        case LM_GGML_OP_ACC:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
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
                            zero_table);
                }
            } break;
        case LM_GGML_OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
                }
                if (src1->grad) {
                    src1->grad = lm_ggml_sub_or_set(ctx, src1->grad, tensor->grad, zero_table);
                }
            } break;
        case LM_GGML_OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_mul(ctx, src1, tensor->grad),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                                src1->grad,
                                lm_ggml_mul(ctx, src0, tensor->grad),
                                zero_table);
                }
            } break;
        case LM_GGML_OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx,
                                src0->grad,
                                lm_ggml_div(ctx, tensor->grad, src1),
                                zero_table);
                }
                if (src1->grad) {
                    src1->grad =
                        lm_ggml_sub_or_set(ctx,
                                src1->grad,
                                lm_ggml_mul(ctx,
                                    tensor->grad,
                                    lm_ggml_div(ctx, tensor, src1)),
                                zero_table);
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
                                zero_table);
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
                                zero_table);
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
                                zero_table);
                }
            } break;
        case LM_GGML_OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add1_or_set(ctx,
                                src0->grad,
                                tensor->grad,
                                zero_table);
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
                                zero_table);
                }
            } break;
        case LM_GGML_OP_MEAN:
        case LM_GGML_OP_ARGMAX:
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
                            zero_table);
                }
            } break;
        case LM_GGML_OP_REPEAT_BACK:
            {
                if (src0->grad) {
                    // TODO: test this
                    src0->grad = lm_ggml_add_or_set(ctx,
                            src0->grad,
                            lm_ggml_repeat(ctx, tensor->grad, src0->grad),
                            zero_table);
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
                            zero_table);
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
                                zero_table);
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
                                zero_table);
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
                            zero_table);
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
                    LM_GGML_ASSERT(tensor->grad->type == src1->grad->type);

                    tensor_grad_view = lm_ggml_view_4d(ctx,
                        tensor->grad,
                        src1->grad->ne[0],
                        src1->grad->ne[1],
                        src1->grad->ne[2],
                        src1->grad->ne[3],
                        nb1, nb2, nb3, offset);
                }

                if (src0->grad) {
                    src0->grad = lm_ggml_add_or_set(ctx,
                        src0->grad,
                        lm_ggml_acc_impl(ctx,
                            tensor->grad,
                            lm_ggml_neg(ctx, tensor_grad_view),
                            nb1, nb2, nb3, offset, false),
                        zero_table);
                }

                if (src1->grad) {
                    src1->grad =
                        lm_ggml_add_or_set(ctx,
                            src1->grad,
                            lm_ggml_reshape(ctx,
                                lm_ggml_cont(ctx, tensor_grad_view),
                                src1->grad),
                            zero_table);
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
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
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
                    src0->grad = lm_ggml_add_or_set(ctx, src0->grad, tensor->grad, zero_table);
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
                        zero_table);
                }
            } break;
        case LM_GGML_OP_VIEW:
            {
                // necessary for llama
                if (src0->grad) {
                    size_t offset;

                    memcpy(&offset, tensor->op_params, sizeof(offset));

                    size_t nb1     = tensor->nb[1];
                    size_t nb2     = tensor->nb[2];
                    size_t nb3     = tensor->nb[3];

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

                    src0->grad = lm_ggml_acc_or_set(ctx, src0->grad, tensor->grad, nb1, nb2, nb3, offset, zero_table);
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
                            zero_table);
                }
            } break;
        case LM_GGML_OP_TRANSPOSE:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_transpose(ctx, tensor->grad),
                        zero_table);
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
                        zero_table);
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
                        zero_table);
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
                        zero_table);
                }
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                // necessary for llama
                if (src0->grad) {
                    src0->grad =
                        lm_ggml_add_or_set(ctx, src0->grad,
                            lm_ggml_soft_max_back(ctx, tensor->grad, tensor),
                        zero_table);
                }

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
                            zero_table);
                }
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
                            zero_table);
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
                            zero_table);
                }
                if (src1->grad) {
                    struct lm_ggml_tensor * view_k = lm_ggml_view_1d(ctx, flash_grad, elem_k, offs_k);
                    struct lm_ggml_tensor * grad_k = lm_ggml_reshape(ctx, view_k, src1);
                    src1->grad = lm_ggml_add_or_set(ctx,
                            src1->grad,
                            grad_k,
                            zero_table);
                }
                if (src2->grad) {
                    struct lm_ggml_tensor * view_v = lm_ggml_view_1d(ctx, flash_grad, elem_v, offs_v);
                    struct lm_ggml_tensor * grad_v = lm_ggml_reshape(ctx, view_v, src2);
                    src2->grad = lm_ggml_add_or_set(ctx,
                            src2->grad,
                            grad_v,
                            zero_table);
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
                                            zero_table);
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
                                src0->grad = lm_ggml_sub_or_set(ctx, src0->grad, tensor->grad, zero_table);
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
                                        zero_table);
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
                                        zero_table);
                            }
                        } break;
                    default:
                        LM_GGML_ABORT("fatal error");
                }
            } break;
        case LM_GGML_OP_GET_REL_POS:
        case LM_GGML_OP_ADD_REL_POS:
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
                                zero_table);
                }
            } break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
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

    if (node->op == LM_GGML_OP_NONE && node->grad == NULL) {
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
        if (cgraph->grads) {
            cgraph->grads[cgraph->n_nodes] = node->grad;
        }
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

void lm_ggml_build_backward_expand(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * gf, struct lm_ggml_cgraph * gb, bool keep) {
    LM_GGML_ASSERT(gf->n_nodes > 0);

    // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
    if (keep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct lm_ggml_tensor * node = gf->nodes[i];

            if (node->grad) {
                node->grad = lm_ggml_dup_tensor(ctx, node);
                gf->grads[i] = node->grad;
            }
        }
    }

    // remember original gradients which start with zero values
    struct lm_ggml_hash_set zero_table = lm_ggml_hash_set_new(gf->size);
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->grads[i]) {
            lm_ggml_hash_insert(&zero_table, gf->grads[i]);
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct lm_ggml_tensor * node = gf->nodes[i];

        // inplace operations to add gradients are not created by lm_ggml_compute_backward
        // use allocator to automatically make inplace operations
        if (node->grad) {
            lm_ggml_compute_backward(ctx, node, &zero_table);
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
        if (src->visited_hash_set.keys[i]) {
            lm_ggml_hash_insert(&dst->visited_hash_set, src->visited_hash_set.keys[i]);
        }
    }
}

struct lm_ggml_cgraph * lm_ggml_graph_dup(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph) {
    struct lm_ggml_cgraph * result = lm_ggml_new_graph_custom(ctx, cgraph->size, cgraph->grads != NULL);
    lm_ggml_graph_cpy(cgraph, result);
    return result;
}

void lm_ggml_graph_reset(struct lm_ggml_cgraph * cgraph) {
    LM_GGML_ASSERT(cgraph->grads != NULL);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * grad = cgraph->grads[i];

        if (grad) {
            lm_ggml_set_zero(grad);
        }
    }
}

void lm_ggml_graph_clear(struct lm_ggml_cgraph * cgraph) {
    cgraph->n_leafs = 0;
    cgraph->n_nodes = 0;
    lm_ggml_hash_set_reset(&cgraph->visited_hash_set);
}

//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//

#ifdef __APPLE__

//#include <os/lock.h>
//
//typedef os_unfair_lock lm_ggml_lock_t;
//
//#define lm_ggml_lock_init(x)    UNUSED(x)
//#define lm_ggml_lock_destroy(x) UNUSED(x)
//#define lm_ggml_lock_lock       os_unfair_lock_lock
//#define lm_ggml_lock_unlock     os_unfair_lock_unlock
//
//#define LM_GGML_LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int lm_ggml_lock_t;

#define lm_ggml_lock_init(x)    UNUSED(x)
#define lm_ggml_lock_destroy(x) UNUSED(x)
#define lm_ggml_lock_lock(x)    UNUSED(x)
#define lm_ggml_lock_unlock(x)  UNUSED(x)

#define LM_GGML_LOCK_INITIALIZER 0

#define lm_ggml_thread_create pthread_create
#define lm_ggml_thread_join   pthread_join

#else

//typedef pthread_spinlock_t lm_ggml_lock_t;

//#define lm_ggml_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define lm_ggml_lock_destroy pthread_spin_destroy
//#define lm_ggml_lock_lock    pthread_spin_lock
//#define lm_ggml_lock_unlock  pthread_spin_unlock

typedef int lm_ggml_lock_t;

#define lm_ggml_lock_init(x)    UNUSED(x)
#define lm_ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define lm_ggml_lock_lock(x)    _mm_pause()
#else
#define lm_ggml_lock_lock(x)    UNUSED(x)
#endif
#define lm_ggml_lock_unlock(x)  UNUSED(x)

#define LM_GGML_LOCK_INITIALIZER 0

#define lm_ggml_thread_create pthread_create
#define lm_ggml_thread_join   pthread_join

#endif

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!lm_ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case LM_GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case LM_GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case LM_GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct lm_ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!lm_ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int lm_ggml_get_n_tasks(struct lm_ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (lm_ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case LM_GGML_OP_CPY:
        case LM_GGML_OP_DUP:
        case LM_GGML_OP_CONT:
        case LM_GGML_OP_ADD:
        case LM_GGML_OP_ADD1:
        case LM_GGML_OP_ACC:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_SUB:
        case LM_GGML_OP_SQR:
        case LM_GGML_OP_SQRT:
        case LM_GGML_OP_LOG:
        case LM_GGML_OP_SUM:
        case LM_GGML_OP_SUM_ROWS:
        case LM_GGML_OP_MEAN:
        case LM_GGML_OP_ARGMAX:
        case LM_GGML_OP_REPEAT:
        case LM_GGML_OP_REPEAT_BACK:
        case LM_GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_UNARY:
            switch (lm_ggml_get_unary_op(node)) {
                case LM_GGML_UNARY_OP_ABS:
                case LM_GGML_UNARY_OP_SGN:
                case LM_GGML_UNARY_OP_NEG:
                case LM_GGML_UNARY_OP_STEP:
                case LM_GGML_UNARY_OP_TANH:
                case LM_GGML_UNARY_OP_ELU:
                case LM_GGML_UNARY_OP_RELU:
                case LM_GGML_UNARY_OP_SIGMOID:
                case LM_GGML_UNARY_OP_HARDSWISH:
                case LM_GGML_UNARY_OP_HARDSIGMOID:
                    {
                        n_tasks = 1;
                    } break;

                case LM_GGML_UNARY_OP_GELU:
                case LM_GGML_UNARY_OP_GELU_QUICK:
                case LM_GGML_UNARY_OP_SILU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    LM_GGML_ABORT("fatal error");
            }
            break;
        case LM_GGML_OP_SILU_BACK:
        case LM_GGML_OP_MUL:
        case LM_GGML_OP_DIV:
        case LM_GGML_OP_NORM:
        case LM_GGML_OP_RMS_NORM:
        case LM_GGML_OP_RMS_NORM_BACK:
        case LM_GGML_OP_GROUP_NORM:
        case LM_GGML_OP_CONCAT:
        case LM_GGML_OP_MUL_MAT:
        case LM_GGML_OP_MUL_MAT_ID:
        case LM_GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case LM_GGML_OP_SCALE:
        case LM_GGML_OP_SET:
        case LM_GGML_OP_RESHAPE:
        case LM_GGML_OP_VIEW:
        case LM_GGML_OP_PERMUTE:
        case LM_GGML_OP_TRANSPOSE:
        case LM_GGML_OP_GET_ROWS_BACK:
        case LM_GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_DIAG_MASK_ZERO:
        case LM_GGML_OP_DIAG_MASK_INF:
        case LM_GGML_OP_SOFT_MAX_BACK:
        case LM_GGML_OP_ROPE:
        case LM_GGML_OP_ROPE_BACK:
        case LM_GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case LM_GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, lm_ggml_nrows(node->src[0]));
            } break;
        case LM_GGML_OP_IM2COL:
        case LM_GGML_OP_CONV_TRANSPOSE_1D:
        case LM_GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_POOL_1D:
        case LM_GGML_OP_POOL_2D:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_UPSCALE:
        case LM_GGML_OP_PAD:
        case LM_GGML_OP_ARANGE:
        case LM_GGML_OP_TIMESTEP_EMBEDDING:
        case LM_GGML_OP_ARGSORT:
        case LM_GGML_OP_FLASH_ATTN_EXT:
        case LM_GGML_OP_FLASH_ATTN_BACK:
        case LM_GGML_OP_SSM_CONV:
        case LM_GGML_OP_SSM_SCAN:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_WIN_PART:
        case LM_GGML_OP_WIN_UNPART:
        case LM_GGML_OP_GET_REL_POS:
        case LM_GGML_OP_MAP_UNARY:
        case LM_GGML_OP_MAP_BINARY:
        case LM_GGML_OP_MAP_CUSTOM1_F32:
        case LM_GGML_OP_MAP_CUSTOM2_F32:
        case LM_GGML_OP_MAP_CUSTOM3_F32:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_MAP_CUSTOM1:
            {
                struct lm_ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_MAP_CUSTOM2:
            {
                struct lm_ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_MAP_CUSTOM3:
            {
                struct lm_ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == LM_GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case LM_GGML_OP_CROSS_ENTROPY_LOSS:
        case LM_GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                n_tasks = n_threads;
            } break;
        case LM_GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case LM_GGML_OP_COUNT:
            {
                LM_GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < LM_GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", lm_ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                LM_GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

struct lm_ggml_cplan lm_ggml_graph_plan(const struct lm_ggml_cgraph * cgraph, int n_threads) {
    if (n_threads <= 0) {
        n_threads = LM_GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct lm_ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct lm_ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = lm_ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        switch (node->op) {
            case LM_GGML_OP_CPY:
            case LM_GGML_OP_DUP:
                {
                    if (lm_ggml_is_quantized(node->type) ||
                        // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                        (node->src[0]->type == LM_GGML_TYPE_F16  && node->src[1] && node->src[1]->type == LM_GGML_TYPE_BF16) ||
                        (node->src[0]->type == LM_GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == LM_GGML_TYPE_F16)) {
                        cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    }
                } break;
            case LM_GGML_OP_ADD:
            case LM_GGML_OP_ADD1:
                {
                    if (lm_ggml_is_quantized(node->src[0]->type)) {
                        cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                    }
                } break;
            case LM_GGML_OP_ACC:
                {
                    if (lm_ggml_is_quantized(node->src[0]->type)) {
                        cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                    }
                } break;
            case LM_GGML_OP_MUL_MAT:
                {
                    const enum lm_ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

                    if (node->src[1]->type != vec_dot_type) {
                        cur = lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(node->src[1]));
                    }
                } break;
            case LM_GGML_OP_MUL_MAT_ID:
                {
                    cur = 0;
                    const struct lm_ggml_tensor * src0 = node->src[0];
                    const struct lm_ggml_tensor * src1 = node->src[1];
                    const enum lm_ggml_type vec_dot_type = type_traits[src0->type].vec_dot_type;
                    if (src1->type != vec_dot_type) {
                        cur += lm_ggml_row_size(vec_dot_type, lm_ggml_nelements(src1));
                    }
                    const int n_as = src0->ne[2];
                    cur += LM_GGML_PAD(cur, sizeof(int64_t));       // align
                    cur += n_as * sizeof(int64_t);               // matrix_row_counts
                    cur += n_as * src1->ne[2] * sizeof(int64_t); // matrix_rows
                } break;
            case LM_GGML_OP_OUT_PROD:
                {
                    if (lm_ggml_is_quantized(node->src[0]->type)) {
                        cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                    }
                } break;
            case LM_GGML_OP_SOFT_MAX:
            case LM_GGML_OP_ROPE:
                {
                    cur = lm_ggml_type_size(LM_GGML_TYPE_F32) * node->ne[0] * n_tasks;
                } break;
            case LM_GGML_OP_CONV_TRANSPOSE_1D:
                {
                    LM_GGML_ASSERT(node->src[0]->ne[3] == 1);
                    LM_GGML_ASSERT(node->src[1]->ne[2] == 1);
                    LM_GGML_ASSERT(node->src[1]->ne[3] == 1);

                    const int64_t ne00 = node->src[0]->ne[0];  // K
                    const int64_t ne01 = node->src[0]->ne[1];  // Cout
                    const int64_t ne02 = node->src[0]->ne[2];  // Cin

                    const int64_t ne10 = node->src[1]->ne[0];  // L
                    const int64_t ne11 = node->src[1]->ne[1];  // Cin

                    if ((node->src[0]->type == LM_GGML_TYPE_F16 ||
                         node->src[0]->type == LM_GGML_TYPE_BF16) &&
                        node->src[1]->type == LM_GGML_TYPE_F32) {
                        cur += sizeof(lm_ggml_fp16_t)*ne00*ne01*ne02;
                        cur += sizeof(lm_ggml_fp16_t)*ne10*ne11;
                    } else if (node->src[0]->type == LM_GGML_TYPE_F32 &&
                               node->src[1]->type == LM_GGML_TYPE_F32) {
                        cur += sizeof(float)*ne00*ne01*ne02;
                        cur += sizeof(float)*ne10*ne11;
                    } else {
                        LM_GGML_ABORT("fatal error");
                    }
                } break;
            case LM_GGML_OP_CONV_TRANSPOSE_2D:
                {
                    const int64_t ne00 = node->src[0]->ne[0]; // W
                    const int64_t ne01 = node->src[0]->ne[1]; // H
                    const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                    const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                    const int64_t ne10 = node->src[1]->ne[0]; // W
                    const int64_t ne11 = node->src[1]->ne[1]; // H
                    const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                    cur += sizeof(lm_ggml_fp16_t)*ne00*ne01*ne02*ne03;
                    cur += sizeof(lm_ggml_fp16_t)*ne10*ne11*ne12;
                } break;
            case LM_GGML_OP_FLASH_ATTN_EXT:
                {
                    const int64_t ne00 = node->src[0]->ne[0]; // D

                    cur = 3*sizeof(float)*ne00*n_tasks; // 3x head size/thread
                } break;
            case LM_GGML_OP_FLASH_ATTN_BACK:
                {
                    const int64_t    D = node->src[0]->ne[0];
                    const int64_t ne11 = lm_ggml_up(node->src[1]->ne[1], LM_GGML_SOFT_MAX_UNROLL);
                    const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in lm_ggml_compute_forward_flash_attn_back
                    if (node->src[1]->type == LM_GGML_TYPE_F32) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    } else if (node->src[1]->type == LM_GGML_TYPE_F16) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    } else if (node->src[1]->type == LM_GGML_TYPE_BF16) {
                        cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                        cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                    }
                } break;

            case LM_GGML_OP_CROSS_ENTROPY_LOSS:
                {
                    cur = lm_ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                } break;
            case LM_GGML_OP_COUNT:
                {
                    LM_GGML_ABORT("fatal error");
                }
            default:
                break;
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads - 1);
    }

    cplan.n_threads = MIN(max_tasks, n_threads);
    cplan.work_size = work_size;
    cplan.work_data = NULL;

    return cplan;
}

static thread_ret_t lm_ggml_graph_compute_thread(void * data) {
    struct lm_ggml_compute_state * state = (struct lm_ggml_compute_state *) data;

    const struct lm_ggml_cgraph * cgraph = state->shared->cgraph;
    const struct lm_ggml_cplan  * cplan  = state->shared->cplan;

    set_numa_thread_affinity(state->ith);

    struct lm_ggml_compute_params params = {
        /*.ith   =*/ state->ith,
        /*.nth   =*/ state->shared->n_threads,
        /*.wsize =*/ cplan->work_size,
        /*.wdata =*/ cplan->work_data,
        /*.shared=*/ state->shared,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct lm_ggml_tensor * node = cgraph->nodes[node_n];

        lm_ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
            state->shared->ec = LM_GGML_STATUS_ABORTED;
        }

        lm_ggml_barrier(state->shared);

        if (state->shared->ec != LM_GGML_STATUS_SUCCESS) {
            break;
        }
    }

    return 0;
}

enum lm_ggml_status lm_ggml_graph_compute(struct lm_ggml_cgraph * cgraph, struct lm_ggml_cplan * cplan) {
    LM_GGML_ASSERT(cplan);
    LM_GGML_ASSERT(cplan->n_threads > 0);
    LM_GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads = cplan->n_threads;

    struct lm_ggml_compute_state_shared state_shared = {
        /*.cgraph                  =*/ cgraph,
        /*.cgraph_plan             =*/ cplan,
        /*.n_threads               =*/ n_threads,
        /*.n_barrier               =*/ 0,
        /*.n_barrier_passed        =*/ 0,
        /*.abort_callback          =*/ NULL,
        /*.abort_callback_data     =*/ NULL,
        /*.current_chunk           =*/ 0,
        /*.ec                      =*/ LM_GGML_STATUS_SUCCESS,
    };

#ifdef LM_GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                state_shared.n_threads = n_threads;
            }

            struct lm_ggml_compute_state worker = {
                .thrd   = 0,
                .ith    = omp_get_thread_num(),
                .shared = &state_shared,
            };
            lm_ggml_graph_compute_thread(&worker);
        }
    } else {
        struct lm_ggml_compute_state worker = {
            .thrd   = 0,
            .ith    = 0,
            .shared = &state_shared,
        };
        lm_ggml_graph_compute_thread(&worker);
    }
#else
    struct lm_ggml_compute_state * workers = alloca(sizeof(struct lm_ggml_compute_state)*n_threads);

    for (int j = 0; j < n_threads; ++j) {
        workers[j] = (struct lm_ggml_compute_state) {
            .thrd   = 0,
            .ith    = j,
            .shared = &state_shared,
        };
    }

    // create thread pool
    for (int j = 1; j < n_threads; ++j) {
        const int rc = lm_ggml_thread_create(&workers[j].thrd, NULL, lm_ggml_graph_compute_thread, &workers[j]);
        LM_GGML_ASSERT(rc == 0);
        UNUSED(rc);
    }

    // this is a work thread too
    lm_ggml_graph_compute_thread(&workers[0]);

    // join or kill thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; j++) {
            const int rc = lm_ggml_thread_join(workers[j].thrd, NULL);
            LM_GGML_ASSERT(rc == 0);
            UNUSED(rc);
        }
    }
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    return state_shared.ec;
}

enum lm_ggml_status lm_ggml_graph_compute_with_ctx(struct lm_ggml_context * ctx, struct lm_ggml_cgraph * cgraph, int n_threads) {
    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(cgraph, n_threads);

    struct lm_ggml_object * obj = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);

    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    return lm_ggml_graph_compute(cgraph, &cplan);
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

static void lm_ggml_graph_export_leaf(const struct lm_ggml_tensor * tensor, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
            lm_ggml_type_name(tensor->type),
            lm_ggml_op_name  (tensor->op),
            lm_ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);
}

static void lm_ggml_graph_export_node(const struct lm_ggml_tensor * tensor, const char * arg, FILE * fout) {
    const int64_t * ne = tensor->ne;
    const size_t  * nb = tensor->nb;

    fprintf(fout, "%-6s %-6s %-12s %8d %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %16zu %16zu %16zu %16zu %16p %32s\n",
            arg,
            lm_ggml_type_name(tensor->type),
            lm_ggml_op_name  (tensor->op),
            lm_ggml_n_dims(tensor),
            ne[0], ne[1], ne[2], ne[3],
            nb[0], nb[1], nb[2], nb[3],
            tensor->data,
            tensor->name);
}

void lm_ggml_graph_export(const struct lm_ggml_cgraph * cgraph, const char * fname) {
    uint64_t size_eval = 0;

    // compute size of intermediate results
    // TODO: does not take into account scratch buffers !!!!
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        size_eval += lm_ggml_nbytes_pad(cgraph->nodes[i]);
    }

    // print
    {
        FILE * fout = stdout;

        fprintf(fout, "\n");
        fprintf(fout, "%-16s %8x\n", "magic",        LM_GGML_FILE_MAGIC);
        fprintf(fout, "%-16s %8d\n", "version",      LM_GGML_FILE_VERSION);
        fprintf(fout, "%-16s %8d\n", "leafs",        cgraph->n_leafs);
        fprintf(fout, "%-16s %8d\n", "nodes",        cgraph->n_nodes);
        fprintf(fout, "%-16s %" PRIu64 "\n", "eval", size_eval);

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %16s %16s\n",
                "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_leafs; ++i) {
            lm_ggml_graph_export_leaf(cgraph->leafs[i], fout);

            LM_GGML_ASSERT(cgraph->leafs[i]->op   == LM_GGML_OP_NONE);
            LM_GGML_ASSERT(cgraph->leafs[i]->src[0] == NULL);
            LM_GGML_ASSERT(cgraph->leafs[i]->src[1] == NULL);
        }

        // header
        fprintf(fout, "\n");
        fprintf(fout, "%-6s %-6s %-12s %8s %8s %8s %8s %8s %16s %16s %16s %16s %8s %16s %16s\n",
                "ARG", "TYPE", "OP", "NDIMS", "NE0", "NE1", "NE2", "NE3", "NB0", "NB1", "NB2", "NB3", "NTASKS", "DATA", "NAME");

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            lm_ggml_graph_export_node(cgraph->nodes[i], "DST", fout);

            for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    lm_ggml_graph_export_node(cgraph->nodes[i]->src[j], "SRC", fout);
                }
            }

            fprintf(fout, "\n");
        }

        fprintf(fout, "\n");
    }

    // write binary data
    {
        FILE * fout = lm_ggml_fopen(fname, "wb");

        if (!fout) {
            fprintf(stderr, "%s: failed to open %s: %s\n", __func__, fname, strerror(errno));
            return;
        }

        // header
        {
            const uint32_t magic   = LM_GGML_FILE_MAGIC;
            const uint32_t version = LM_GGML_FILE_VERSION;
            const uint32_t n_leafs = cgraph->n_leafs;
            const uint32_t n_nodes = cgraph->n_nodes;

            fwrite(&magic,     sizeof(uint32_t), 1, fout);
            fwrite(&version,   sizeof(uint32_t), 1, fout);
            fwrite(&n_leafs,   sizeof(uint32_t), 1, fout);
            fwrite(&n_nodes,   sizeof(uint32_t), 1, fout);
            fwrite(&size_eval, sizeof(uint64_t), 1, fout);
        }

        // leafs
        {
            for (int i = 0; i < cgraph->n_leafs; ++i) {
                const struct lm_ggml_tensor * tensor = cgraph->leafs[i];

                const uint32_t type   = tensor->type;
                const uint32_t op     = tensor->op;

                fwrite(&type,   sizeof(uint32_t), 1, fout);
                fwrite(&op,     sizeof(uint32_t), 1, fout);

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    const uint64_t ne = tensor->ne[j];
                    const uint64_t nb = tensor->nb[j];

                    fwrite(&ne, sizeof(uint64_t), 1, fout);
                    fwrite(&nb, sizeof(uint64_t), 1, fout);
                }

                fwrite(tensor->name,      sizeof(char), LM_GGML_MAX_NAME,      fout);
                fwrite(tensor->op_params, sizeof(char), LM_GGML_MAX_OP_PARAMS, fout);

                // dump the data
                // TODO: pad this to 32 byte boundary
                {
                    const size_t size = lm_ggml_nbytes(tensor);

                    fwrite(tensor->data, sizeof(char), size, fout);
                }
            }
        }

        // nodes
        {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const struct lm_ggml_tensor * tensor = cgraph->nodes[i];

                const uint32_t type   = tensor->type;
                const uint32_t op     = tensor->op;

                fwrite(&type,   sizeof(uint32_t), 1, fout);
                fwrite(&op,     sizeof(uint32_t), 1, fout);

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    const uint64_t ne = tensor->ne[j];
                    const uint64_t nb = tensor->nb[j];

                    fwrite(&ne, sizeof(uint64_t), 1, fout);
                    fwrite(&nb, sizeof(uint64_t), 1, fout);
                }

                fwrite(tensor->name,      sizeof(char), LM_GGML_MAX_NAME,      fout);
                fwrite(tensor->op_params, sizeof(char), LM_GGML_MAX_OP_PARAMS, fout);

                // output the op arguments
                {
                    struct lm_ggml_tensor * args[LM_GGML_MAX_SRC] = { NULL };

                    for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
                        args[j] = tensor->src[j];
                    }

                    for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
                        if (args[j]) {
                            int32_t idx = -1;

                            // check if leaf
                            {
                                for (int k = 0; k < cgraph->n_leafs; ++k) {
                                    if (args[j] == cgraph->leafs[k]) {
                                        idx = k;
                                        break;
                                    }
                                }
                            }

                            // check if node
                            if (idx == -1) {
                                for (int k = 0; k < cgraph->n_nodes; ++k) {
                                    if (args[j] == cgraph->nodes[k]) {
                                        idx = cgraph->n_leafs + k;
                                        break;
                                    }
                                }
                            }

                            if (idx == -1) {
                                fprintf(stderr, "%s: failed to find tensor, arg = %d, node = %d\n", __func__, j, i);
                                fclose(fout);
                                return;
                            }

                            fwrite(&idx, sizeof(int32_t), 1, fout);
                        } else {
                            const int32_t nul = -1;

                            fwrite(&nul, sizeof(int32_t), 1, fout);
                        }
                    }
                }
            }
        }

        fclose(fout);
    }
}

struct lm_ggml_cgraph * lm_ggml_graph_import(const char * fname, struct lm_ggml_context ** ctx_data, struct lm_ggml_context ** ctx_eval) {
    assert(*ctx_data == NULL);
    assert(*ctx_eval == NULL);

    struct lm_ggml_cgraph * result = NULL;

    struct lm_ggml_tensor * data = NULL;

    // read file into data
    {
        FILE * fin = lm_ggml_fopen(fname, "rb");
        if (!fin) {
            fprintf(stderr, "%s: failed to open %s: %s\n", __func__, fname, strerror(errno));
            return result;
        }

        size_t fsize = 0;

        fseek(fin, 0, SEEK_END);
        fsize = ftell(fin);
        fseek(fin, 0, SEEK_SET);

        // create the data context
        {
            const size_t overhead = 1*lm_ggml_tensor_overhead();

            struct lm_ggml_init_params params = {
                .mem_size   = fsize + overhead,
                .mem_buffer = NULL,
                .no_alloc   = false,
            };

            *ctx_data = lm_ggml_init(params);

            if (!*ctx_data) {
                fprintf(stderr, "%s: failed to create ggml context\n", __func__);
                fclose(fin);
                return result;
            }
        }

        data = lm_ggml_new_tensor_1d(*ctx_data, LM_GGML_TYPE_I8, fsize);

        {
            const size_t ret = fread(data->data, sizeof(char), fsize, fin);
            if (ret != fsize) {
                fprintf(stderr, "%s: failed to read %s\n", __func__, fname);
                fclose(fin);
                return result;
            }
        }

        fclose(fin);
    }

    // populate result
    {
        char * ptr = (char *) data->data;

        const uint32_t magic = *(const uint32_t *) ptr; ptr += sizeof(magic);

        if (magic != LM_GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid magic number, got %08x\n", __func__, magic);
            return result;
        }

        const uint32_t version = *(const uint32_t *) ptr; ptr += sizeof(version);

        if (version != LM_GGML_FILE_VERSION) {
            fprintf(stderr, "%s: invalid version number\n", __func__);
            return result;
        }

        const uint32_t n_leafs   = *(const uint32_t *) ptr; ptr += sizeof(n_leafs);
        const uint32_t n_nodes   = *(const uint32_t *) ptr; ptr += sizeof(n_nodes);
        const uint64_t size_eval = *(const uint64_t *) ptr; ptr += sizeof(size_eval);
        const int     graph_size = MAX(n_leafs, n_nodes);

        // create the data context
        {
            const size_t overhead = (n_leafs + n_nodes)*lm_ggml_tensor_overhead() + lm_ggml_graph_overhead_custom(graph_size, false);

            struct lm_ggml_init_params params = {
                .mem_size   = size_eval + overhead,
                .mem_buffer = NULL,
                .no_alloc   = true,
            };

            *ctx_eval = lm_ggml_init(params);

            if (!*ctx_eval) {
                fprintf(stderr, "%s: failed to create ggml context\n", __func__);
                return result;
            }
        }

        result = lm_ggml_new_graph_custom(*ctx_eval, graph_size, false);

        result->n_leafs = n_leafs;
        result->n_nodes = n_nodes;


        // leafs
        {
            uint32_t type;
            uint32_t op;

            for (uint32_t i = 0; i < n_leafs; ++i) {
                type   = *(const uint32_t *) ptr; ptr += sizeof(type);
                op     = *(const uint32_t *) ptr; ptr += sizeof(op);

                int64_t ne[LM_GGML_MAX_DIMS];
                size_t  nb[LM_GGML_MAX_DIMS];

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    uint64_t ne_cur;
                    uint64_t nb_cur;

                    ne_cur = *(const uint64_t *) ptr; ptr += sizeof(ne_cur);
                    nb_cur = *(const uint64_t *) ptr; ptr += sizeof(nb_cur);

                    ne[j] = ne_cur;
                    nb[j] = nb_cur;
                }

                struct lm_ggml_tensor * tensor = lm_ggml_new_tensor(*ctx_eval, (enum lm_ggml_type) type, LM_GGML_MAX_DIMS, ne);

                tensor->op = (enum lm_ggml_op) op;

                memcpy(tensor->name,      ptr, LM_GGML_MAX_NAME);      ptr += LM_GGML_MAX_NAME;
                memcpy(tensor->op_params, ptr, LM_GGML_MAX_OP_PARAMS); ptr += LM_GGML_MAX_OP_PARAMS;

                tensor->data = (void *) ptr;

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    tensor->nb[j] = nb[j];
                }

                result->leafs[i] = tensor;

                ptr += lm_ggml_nbytes(tensor);

                fprintf(stderr, "%s: loaded leaf %u: '%16s', %9zu bytes\n", __func__, i, tensor->name, lm_ggml_nbytes(tensor));
            }
        }

        lm_ggml_set_no_alloc(*ctx_eval, false);

        // nodes
        {
            uint32_t type;
            uint32_t op;

            for (uint32_t i = 0; i < n_nodes; ++i) {
                type   = *(const uint32_t *) ptr; ptr += sizeof(type);
                op     = *(const uint32_t *) ptr; ptr += sizeof(op);

                enum lm_ggml_op eop = (enum lm_ggml_op) op;

                int64_t ne[LM_GGML_MAX_DIMS];
                size_t  nb[LM_GGML_MAX_DIMS];

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    uint64_t ne_cur;
                    uint64_t nb_cur;

                    ne_cur = *(const uint64_t *) ptr; ptr += sizeof(ne_cur);
                    nb_cur = *(const uint64_t *) ptr; ptr += sizeof(nb_cur);

                    ne[j] = ne_cur;
                    nb[j] = nb_cur;
                }

                const char * ptr_name      = ptr; ptr += LM_GGML_MAX_NAME;
                const char * ptr_op_params = ptr; ptr += LM_GGML_MAX_OP_PARAMS;

                const int32_t * ptr_arg_idx = (const int32_t *) ptr; ptr += LM_GGML_MAX_SRC*sizeof(int32_t);

                struct lm_ggml_tensor * args[LM_GGML_MAX_SRC] = { NULL };

                // parse args
                for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
                    const int32_t arg_idx = ptr_arg_idx[j];

                    if (arg_idx == -1) {
                        continue;
                    }

                    if (arg_idx < result->n_leafs) {
                        args[j] = result->leafs[arg_idx];
                    } else {
                        args[j] = result->nodes[arg_idx - result->n_leafs];
                    }
                }

                // create the tensor
                // "view" operations are handled differently
                // TODO: handle inplace ops - currently a copy is always made

                struct lm_ggml_tensor * tensor = NULL;

                switch (eop) {
                    // TODO: implement other view ops
                    case LM_GGML_OP_RESHAPE:
                        {
                            tensor = lm_ggml_reshape_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3]);
                        } break;
                    case LM_GGML_OP_VIEW:
                        {
                            tensor = lm_ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);

                            size_t offs;
                            memcpy(&offs, ptr_op_params, sizeof(offs));

                            tensor->data = ((char *) tensor->data) + offs;
                        } break;
                    case LM_GGML_OP_TRANSPOSE:
                        {
                            tensor = lm_ggml_transpose(*ctx_eval, args[0]);
                        } break;
                    case LM_GGML_OP_PERMUTE:
                        {
                            tensor = lm_ggml_view_4d(*ctx_eval, args[0], ne[0], ne[1], ne[2], ne[3], 0, 0, 0, 0);
                        } break;
                    default:
                        {
                            tensor = lm_ggml_new_tensor(*ctx_eval, (enum lm_ggml_type) type, LM_GGML_MAX_DIMS, ne);

                            tensor->op = eop;
                        } break;
                }

                memcpy(tensor->name,      ptr_name,      LM_GGML_MAX_NAME);
                memcpy(tensor->op_params, ptr_op_params, LM_GGML_MAX_OP_PARAMS);

                for (int j = 0; j < LM_GGML_MAX_DIMS; ++j) {
                    tensor->nb[j] = nb[j];
                }

                for (int j = 0; j < LM_GGML_MAX_SRC; ++j) {
                    tensor->src[j] = args[j];
                }

                result->nodes[i] = tensor;

                fprintf(stderr, "%s: loaded node %u: '%16s', %9zu bytes\n", __func__, i, tensor->name, lm_ggml_nbytes(tensor));
            }
        }
    }

    return result;
}

void lm_ggml_graph_print(const struct lm_ggml_cgraph * cgraph) {
    LM_GGML_PRINT("=== GRAPH ===\n");

    LM_GGML_PRINT("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct lm_ggml_tensor * node = cgraph->nodes[i];

        LM_GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                lm_ggml_op_name(node->op), (node->flags & LM_GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ");
    }

    LM_GGML_PRINT("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct lm_ggml_tensor * node = cgraph->leafs[i];

        LM_GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                i,
                node->ne[0], node->ne[1],
                lm_ggml_op_name(node->op),
                lm_ggml_get_name(node));
    }

    LM_GGML_PRINT("========================================\n");
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
                if (node->type == LM_GGML_TYPE_I8 || node->type == LM_GGML_TYPE_I16 || node->type == LM_GGML_TYPE_I32) {
                    fprintf(fp, "%d", lm_ggml_get_i32_1d(node, j));
                }
                else if (node->type == LM_GGML_TYPE_F32 ||
                         node->type == LM_GGML_TYPE_F16 ||
                         node->type == LM_GGML_TYPE_BF16) {
                    fprintf(fp, "%.1e", (double)lm_ggml_get_f32_1d(node, j));
                }
                else {
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

    LM_GGML_PRINT("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void lm_ggml_opt_set_params(int np, struct lm_ggml_tensor * const ps[], const float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = lm_ggml_nelements(ps[p]) ;
        // TODO: add function to set tensor from array
        for (int64_t j = 0; j < ne; ++j) {
            lm_ggml_set_f32_1d(ps[p], j, x[i++]);
        }
    }
}

static void lm_ggml_opt_get_params(int np, struct lm_ggml_tensor * const ps[], float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = lm_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            x[i++] = lm_ggml_get_f32_1d(ps[p], j);
        }
    }
}

static void lm_ggml_opt_get_grad(int np, struct lm_ggml_tensor * const ps[], float * g) {
    int64_t i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = lm_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            g[i++] = lm_ggml_get_f32_1d(ps[p]->grad, j);
        }
    }
}

static void lm_ggml_opt_acc_grad(int np, struct lm_ggml_tensor * const ps[], float * g, float scale) {
    int64_t i = 0;
    for (int p = 0; p < np; ++p) {
        const int64_t ne = lm_ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int64_t j = 0; j < ne; ++j) {
            g[i++] += lm_ggml_get_f32_1d(ps[p]->grad, j) * scale;
        }
    }
}

//
// Using AdamW - ref: https://arxiv.org/pdf/1711.05101v3.pdf
//
// (Original Adam - ref: https://arxiv.org/pdf/1412.6980.pdf)
//

static enum lm_ggml_opt_result lm_ggml_opt_adam(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_context * opt,
        struct lm_ggml_opt_params params,
        struct lm_ggml_tensor * f,
        struct lm_ggml_cgraph * gf,
        struct lm_ggml_cgraph * gb,
        lm_ggml_opt_callback callback,
        void * callback_data) {
    LM_GGML_ASSERT(lm_ggml_is_scalar(f));

    // these will store the parameters we want to optimize
    struct lm_ggml_tensor * ps[LM_GGML_MAX_PARAMS];

    int np = 0;
    int64_t nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            LM_GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            LM_GGML_ASSERT(np < LM_GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += lm_ggml_nelements(gf->nodes[i]);
        }
    }

    if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past)) {
        int iter = opt->iter;
        lm_ggml_opt_init(opt->ctx, opt, params, nx);
        opt->iter = iter;
    }

    // constants
    float sched = params.adam.sched;
    const float alpha = params.adam.alpha;
    const float decay = params.adam.decay * alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;
    const float gclip = params.adam.gclip;
    const int decay_min_ndim = params.adam.decay_min_ndim;
    const int n_accum = MAX(1, params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    float * g  = opt->adam.g->data;  // gradients
    float * m  = opt->adam.m->data;  // first moment
    float * v  = opt->adam.v->data;  // second moment

    float * pf = params.past > 0 ? opt->adam.pf->data : NULL; // past function values

    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(gb, params.n_threads);
    struct lm_ggml_object * obj = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    bool cancel = false;

    // compute the function value
    float fx = 0;
    lm_ggml_set_zero(opt->adam.g);
    for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
        if (callback) {
            callback(callback_data, accum_step, &sched, &cancel);
            if (cancel) {
                return LM_GGML_OPT_RESULT_CANCEL;
            }
        }
        // lm_ggml_graph_reset  (gf);
        lm_ggml_set_f32      (f->grad, 1.0f);
        lm_ggml_graph_compute(gb, &cplan);
        lm_ggml_opt_acc_grad(np, ps, g, accum_norm);
        fx += lm_ggml_get_f32_1d(f, 0);
    }
    fx *= accum_norm;

    opt->adam.fx_prev = fx;
    opt->adam.fx_best = opt->adam.fx_prev;
    if (pf) {
        pf[opt->iter % params.past] = opt->adam.fx_prev;
    }

    opt->loss_before = opt->adam.fx_prev;
    opt->loss_after  = opt->adam.fx_prev;

    // initialize
    if (opt->just_initialized) {
        opt->adam.n_no_improvement = 0;
        opt->just_initialized = false;
    }

    float * fx_best = &opt->adam.fx_best;
    float * fx_prev = &opt->adam.fx_prev;
    int * n_no_improvement = &opt->adam.n_no_improvement;

    int iter0 = opt->iter;

    // run the optimizer
    for (int t = 0; t < params.adam.n_iter; ++t) {
        opt->iter = iter0 + t + 1;
        LM_GGML_PRINT_DEBUG  ("=== iter %d ===\n", t);

        LM_GGML_PRINT_DEBUG  ("f      = %10.6f\n", lm_ggml_get_f32_1d(f, 0));
        LM_GGML_PRINT_DEBUG_5("df/dx0 = %10.6f\n", lm_ggml_get_f32_1d(ps[0]->grad, 0));
        LM_GGML_PRINT_DEBUG_5("df/dx1 = %10.6f\n", lm_ggml_get_f32_1d(ps[1]->grad, 0));

        for (int i = 0; i < np; ++i) {
            LM_GGML_PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,
                    lm_ggml_get_f32_1d(ps[i], 0), lm_ggml_get_f32_1d(ps[i]->grad, 0));
        }

        const int64_t t_start_wall = lm_ggml_time_us();
        const int64_t t_start_cpu = lm_ggml_cycles();
        UNUSED(t_start_wall);
        UNUSED(t_start_cpu);

        {
            float gnorm = 1.0f;
            if (gclip > 0.0f) {
                // gradient clipping
                lm_ggml_float sum = 0.0;
                for (int64_t i = 0; i < nx; ++i) {
                    sum += (lm_ggml_float)(g[i]*g[i]);
                }
                lm_ggml_float norm = sqrt(sum);
                if (norm > (lm_ggml_float) gclip) {
                    gnorm = (float) ((lm_ggml_float) gclip / norm);
                }
            }
            const float beta1h = alpha*sched/(1.0f - powf(beta1, opt->iter));
            const float beta2h =        1.0f/(1.0f - powf(beta2, opt->iter));
            int64_t i = 0;
            for (int p = 0; p < np; ++p) {
                const int64_t ne = lm_ggml_nelements(ps[p]);
                const float p_decay = ((lm_ggml_n_dims(ps[p]) >= decay_min_ndim) ? decay : 0.0f) * sched;
                for (int64_t j = 0; j < ne; ++j) {
                    float x  = lm_ggml_get_f32_1d(ps[p], j);
                    float g_ = g[i]*gnorm;
                    m[i] = m[i]*beta1 +    g_*(1.0f - beta1);
                    v[i] = v[i]*beta2 + g_*g_*(1.0f - beta2);
                    float mh = m[i]*beta1h;
                    float vh = v[i]*beta2h;
                    vh = sqrtf(vh) + eps;
                    x  = x*(1.0f - p_decay) - mh/vh;
                    lm_ggml_set_f32_1d(ps[p], j, x);
                    ++i;
                }
            }
        }

        fx = 0;
        lm_ggml_set_zero(opt->adam.g);
        for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
            if (callback) {
                callback(callback_data, accum_step, &sched, &cancel);
                if (cancel) {
                    return LM_GGML_OPT_RESULT_CANCEL;;
                }
            }
            // lm_ggml_graph_reset  (gf);
            lm_ggml_set_f32      (f->grad, 1.0f);
            lm_ggml_graph_compute(gb, &cplan);
            lm_ggml_opt_acc_grad(np, ps, g, accum_norm);
            fx += lm_ggml_get_f32_1d(f, 0);
        }
        fx *= accum_norm;

        opt->loss_after = fx;

        // check convergence
        if (fabsf(fx - fx_prev[0])/fx < params.adam.eps_f) {
            LM_GGML_PRINT_DEBUG("converged\n");

            return LM_GGML_OPT_RESULT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= iter0 + t) {
                const float rate = (pf[(iter0 + t)%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return LM_GGML_OPT_RESULT_OK;
                }
            }

            pf[(iter0 + t)%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx_best[0] > fx) {
                fx_best[0] = fx;
                n_no_improvement[0] = 0;
            } else {
                ++n_no_improvement[0];

                if (n_no_improvement[0] >= params.max_no_improvement) {
                    return LM_GGML_OPT_RESULT_OK;
                }
            }
        }

        fx_prev[0] = fx;

        {
            const int64_t t_end_cpu = lm_ggml_cycles();
            LM_GGML_PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            UNUSED(t_end_cpu);

            const int64_t t_end_wall = lm_ggml_time_us();
            LM_GGML_PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            UNUSED(t_end_wall);
        }
    }

    return LM_GGML_OPT_RESULT_DID_NOT_CONVERGE;
}

//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct lm_ggml_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

static enum lm_ggml_opt_result linesearch_backtracking(
        const struct lm_ggml_opt_params * params,
        int nx,
        float * x,
        float * fx,
        float * g,
        float * d,
        float * step,
        const float * xp,
        struct lm_ggml_tensor * f,
        struct lm_ggml_cgraph * gb,
        struct lm_ggml_cplan  * cplan,
        const int np,
        struct lm_ggml_tensor * ps[],
        bool * cancel,
        lm_ggml_opt_callback callback,
        void * callback_data) {
    int count = 0;

    float width  = 0.0f;
    float dg     = 0.0f;
    float finit  = 0.0f;
    float dginit = 0.0f;
    float dgtest = 0.0f;

    const float dec = 0.5f;
    const float inc = 2.1f;

    const int n_accum = MAX(1, params->n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    if (*step <= 0.f) {
        return LM_GGML_LINESEARCH_INVALID_PARAMETERS;
    }

    // compute the initial gradient in the search direction
    lm_ggml_vec_dot_f32(nx, &dginit, 0, g, 0, d, 0, 1);

    // make sure that d points to a descent direction
    if (0 < dginit) {
        return LM_GGML_LINESEARCH_FAIL;
    }

    // initialize local variables
    finit = *fx;
    dgtest = params->lbfgs.ftol*dginit;

    while (true) {
        lm_ggml_vec_cpy_f32(nx, x, xp);
        lm_ggml_vec_mad_f32(nx, x, d, *step);

        // evaluate the function and gradient values
        {
            lm_ggml_opt_set_params(np, ps, x);

            *fx = 0;
            memset(g, 0, sizeof(float)*nx);
            for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
                if (callback) {
                    // LBFG-S does not support learning rate -> ignore learning schedule
                    float sched = 0;
                    callback(callback_data, accum_step, &sched, cancel);
                    if (*cancel) {
                        return LM_GGML_OPT_RESULT_CANCEL;
                    }
                }
                // lm_ggml_graph_reset  (gf);
                lm_ggml_set_f32      (f->grad, 1.0f);
                lm_ggml_graph_compute(gb, cplan);
                lm_ggml_opt_acc_grad(np, ps, g, accum_norm);
                *fx += lm_ggml_get_f32_1d(f, 0);
            }
            *fx *= accum_norm;

        }

        ++count;

        if (*fx > finit + (*step)*dgtest) {
            width = dec;
        } else {
            // Armijo condition is satisfied
            if (params->lbfgs.linesearch == LM_GGML_LINESEARCH_BACKTRACKING_ARMIJO) {
                return count;
            }

            lm_ggml_vec_dot_f32(nx, &dg, 0, g, 0, d, 0, 1);

            // check the Wolfe condition
            if (dg < params->lbfgs.wolfe * dginit) {
                width = inc;
            } else {
                if(params->lbfgs.linesearch == LM_GGML_LINESEARCH_BACKTRACKING_WOLFE) {
                    // regular Wolfe conditions
                    return count;
                }

                if(dg > -params->lbfgs.wolfe*dginit) {
                    width = dec;
                } else {
                    // strong Wolfe condition (LM_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    return count;
                }
            }
        }

        if (*step < params->lbfgs.min_step) {
            return LM_GGML_LINESEARCH_MINIMUM_STEP;
        }
        if (*step > params->lbfgs.max_step) {
            return LM_GGML_LINESEARCH_MAXIMUM_STEP;
        }
        if (params->lbfgs.max_linesearch <= count) {
            return LM_GGML_LINESEARCH_MAXIMUM_ITERATIONS;
        }

        (*step) *= width;
    }

    LM_GGML_ABORT("line search failed");

    //return LM_GGML_LINESEARCH_FAIL;
}

static enum lm_ggml_opt_result lm_ggml_opt_lbfgs(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_context * opt,
        struct lm_ggml_opt_params params,
        struct lm_ggml_tensor * f,
        struct lm_ggml_cgraph * gf,
        struct lm_ggml_cgraph * gb,
        lm_ggml_opt_callback callback,
        void * callback_data) {
    if (params.lbfgs.linesearch == LM_GGML_LINESEARCH_BACKTRACKING_WOLFE ||
        params.lbfgs.linesearch == LM_GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1.f <= params.lbfgs.wolfe) {
            return LM_GGML_OPT_RESULT_INVALID_WOLFE;
        }
    }

    const int m = params.lbfgs.m;

    // these will store the parameters we want to optimize
    struct lm_ggml_tensor * ps[LM_GGML_MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & LM_GGML_TENSOR_FLAG_PARAM) {
            LM_GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            LM_GGML_ASSERT(np < LM_GGML_MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += lm_ggml_nelements(gf->nodes[i]);
        }
    }

    if ((opt->params.type != params.type) || (opt->nx != nx) || (opt->params.past != params.past) || (opt->params.lbfgs.m != params.lbfgs.m)) {
        int iter = opt->iter;
        lm_ggml_opt_init(ctx, opt, params, nx);
        opt->iter = iter;
    }

    struct lm_ggml_cplan cplan = lm_ggml_graph_plan(gb, params.n_threads);
    struct lm_ggml_object * obj = lm_ggml_new_object(ctx, LM_GGML_OBJECT_TYPE_WORK_BUFFER, cplan.work_size);
    cplan.work_data = (uint8_t *)ctx->mem_buffer + obj->offs;

    float * x  = opt->lbfgs.x->data;  // current parameters
    float * xp = opt->lbfgs.xp->data; // previous parameters
    float * g  = opt->lbfgs.g->data;  // current gradient
    float * gp = opt->lbfgs.gp->data; // previous gradient
    float * d  = opt->lbfgs.d->data;  // search direction

    float * pf = params.past > 0 ? opt->lbfgs.pf->data : NULL; // past function values

    const int n_accum = MAX(1, params.n_gradient_accumulation);
    const float accum_norm = 1.0f / (float) n_accum;

    float fx    = 0.0f; // cost function value
    float xnorm = 0.0f; // ||x||
    float gnorm = 0.0f; // ||g||

    // initialize x from the graph nodes
    lm_ggml_opt_get_params(np, ps, x);

    // the L-BFGS memory
    float * lm_alpha = opt->lbfgs.lmal->data;
    float * lm_ys    = opt->lbfgs.lmys->data;
    float * lm_s     = opt->lbfgs.lms->data;
    float * lm_y     = opt->lbfgs.lmy->data;

    bool cancel = false;

    // evaluate the function value and its gradient
    {
        lm_ggml_opt_set_params(np, ps, x);

        fx = 0;
        memset(g, 0, sizeof(float)*nx);
        for (int accum_step = 0; accum_step < n_accum; ++accum_step) {
            if (callback) {
                // LBFG-S does not support learning rate -> ignore learning schedule
                float sched = 0;
                callback(callback_data, accum_step, &sched, &cancel);
                if (cancel) {
                    return LM_GGML_OPT_RESULT_CANCEL;
                }
            }
            // lm_ggml_graph_reset  (gf);
            lm_ggml_set_f32      (f->grad, 1.0f);
            lm_ggml_graph_compute(gb, &cplan);
            lm_ggml_opt_acc_grad(np, ps, g, accum_norm);
            fx += lm_ggml_get_f32_1d(f, 0);
        }
        fx *= accum_norm;

        opt->loss_before = fx;
        opt->loss_after  = fx;
    }

    // search direction = -gradient
    lm_ggml_vec_neg_f32(nx, d, g);

    // ||x||, ||g||
    lm_ggml_vec_norm_f32(nx, &xnorm, x);
    lm_ggml_vec_norm_f32(nx, &gnorm, g);

    if (xnorm < 1.0f) {
        xnorm = 1.0f;
    }

    // already optimized
    if (gnorm/xnorm <= params.lbfgs.eps) {
        return LM_GGML_OPT_RESULT_OK;
    }

    if (opt->just_initialized) {
        if (pf) {
            pf[0] = fx;
        }
        opt->lbfgs.fx_best = fx;

        // initial step
        lm_ggml_vec_norm_inv_f32(nx, &opt->lbfgs.step, d);
        opt->lbfgs.j                = 0;
        opt->lbfgs.k                = 1;
        opt->lbfgs.end              = 0;
        opt->lbfgs.n_no_improvement = 0;
        opt->just_initialized       = false;
    }

    float * fx_best        = &opt->lbfgs.fx_best;
    float * step           = &opt->lbfgs.step;
    int * j                = &opt->lbfgs.j;
    int * k                = &opt->lbfgs.k;
    int * end              = &opt->lbfgs.end;
    int * n_no_improvement = &opt->lbfgs.n_no_improvement;

    int ls     = 0;
    int bound  = 0;

    float ys   = 0.0f;
    float yy   = 0.0f;
    float beta = 0.0f;

    int it = 0;

    while (true) {
        // store the current position and gradient vectors
        lm_ggml_vec_cpy_f32(nx, xp, x);
        lm_ggml_vec_cpy_f32(nx, gp, g);

        // TODO: instead of passing &cancel here, use the return code of the linesearch
        //       to determine if the optimization should be cancelled
        //       this is a simple change, but not doing this atm, since I don't have a nice
        //       way to test and don't want to break something with so many changes lined up
        ls = linesearch_backtracking(&params, nx, x, &fx, g, d, step, xp, f, gb, &cplan, np, ps, &cancel, callback, callback_data);
        if (cancel) {
            return LM_GGML_OPT_RESULT_CANCEL;
        }

        if (ls < 0) {
            // linesearch failed - go back to the previous point and return
            lm_ggml_vec_cpy_f32(nx, x, xp);
            lm_ggml_vec_cpy_f32(nx, g, gp);

            return ls;
        }

        opt->loss_after = fx;

        lm_ggml_vec_norm_f32(nx, &xnorm, x);
        lm_ggml_vec_norm_f32(nx, &gnorm, g);

        LM_GGML_PRINT_DEBUG("f = %10.6f\n", lm_ggml_get_f32_1d(f, 0));

        if (xnorm < 1.0f) {
            xnorm = 1.0f;
        }
        if (gnorm/xnorm <= params.lbfgs.eps) {
            // converged
            return LM_GGML_OPT_RESULT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= k[0]) {
                const float rate = (pf[k[0]%params.past] - fx)/fx;

                if (fabsf(rate) < params.delta) {
                    return LM_GGML_OPT_RESULT_OK;
                }
            }

            pf[k[0]%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx < fx_best[0]) {
                fx_best[0] = fx;
                n_no_improvement[0] = 0;
            } else {
                n_no_improvement[0]++;

                if (n_no_improvement[0] >= params.max_no_improvement) {
                    return LM_GGML_OPT_RESULT_OK;
                }
            }
        }

        if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < it + 1) {
            // reached the maximum number of iterations
            return LM_GGML_OPT_RESULT_DID_NOT_CONVERGE;
        }

        // update vectors s and y:
        //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        //   y_{k+1} = g_{k+1} - g_{k}.
        //
        lm_ggml_vec_sub_f32(nx, &lm_s[end[0]*nx], x, xp);
        lm_ggml_vec_sub_f32(nx, &lm_y[end[0]*nx], g, gp);

        // compute scalars ys and yy:
        //     ys = y^t \cdot s    -> 1 / \rho.
        //     yy = y^t \cdot y.
        //
        lm_ggml_vec_dot_f32(nx, &ys, 0, &lm_y[end[0]*nx], 0, &lm_s[end[0]*nx], 0, 1);
        lm_ggml_vec_dot_f32(nx, &yy, 0, &lm_y[end[0]*nx], 0, &lm_y[end[0]*nx], 0, 1);

        lm_ys[end[0]] = ys;

        // find new search direction
        //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

        bound = (m <= k[0]) ? m : k[0];
        k[0]++;
        it++;
        end[0] = (end[0] + 1)%m;

        // initialize search direction with -g
        lm_ggml_vec_neg_f32(nx, d, g);

        j[0] = end[0];
        for (int i = 0; i < bound; ++i) {
            j[0] = (j[0] + m - 1) % m;
            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
            lm_ggml_vec_dot_f32(nx, &lm_alpha[j[0]], 0, &lm_s[j[0]*nx], 0, d, 0, 1);
            lm_alpha[j[0]] /= lm_ys[j[0]];
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}
            lm_ggml_vec_mad_f32(nx, d, &lm_y[j[0]*nx], -lm_alpha[j[0]]);
        }

        lm_ggml_vec_scale_f32(nx, d, ys/yy);

        for (int i = 0; i < bound; ++i) {
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
            lm_ggml_vec_dot_f32(nx, &beta, 0, &lm_y[j[0]*nx], 0, d, 0, 1);
            beta /= lm_ys[j[0]];
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
            lm_ggml_vec_mad_f32(nx, d, &lm_s[j[0]*nx], lm_alpha[j[0]] - beta);
            j[0] = (j[0] + 1)%m;
        }

        step[0] = 1.0;
    }

    LM_GGML_ABORT("lbfgs failed");

    //return LM_GGML_OPT_RESULT_DID_NOT_CONVERGE;
}

struct lm_ggml_opt_params lm_ggml_opt_default_params(enum lm_ggml_opt_type type) {
    struct lm_ggml_opt_params result;

    switch (type) {
        case LM_GGML_OPT_TYPE_ADAM:
            {
                result = (struct lm_ggml_opt_params) {
                    .type       = LM_GGML_OPT_TYPE_ADAM,
                    .graph_size = LM_GGML_DEFAULT_GRAPH_SIZE,
                    .n_threads  = 1, // FIXME: LM_GGML_DEFAULT_N_THREADS ?
                    .past       = 0,
                    .delta      = 1e-5f,

                    .max_no_improvement = 100,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .n_gradient_accumulation = 1,

                    .adam = {
                        .n_iter = 10000,
                        .sched  = 1.000f,
                        .decay  = 0.0f,
                        .decay_min_ndim = 2,
                        .alpha  = 0.001f,
                        .beta1  = 0.9f,
                        .beta2  = 0.999f,
                        .eps    = 1e-8f,
                        .eps_f  = 1e-5f,
                        .eps_g  = 1e-3f,
                        .gclip  = 0.0f,
                    },
                };
            } break;
        case LM_GGML_OPT_TYPE_LBFGS:
            {
                result = (struct lm_ggml_opt_params) {
                    .type       = LM_GGML_OPT_TYPE_LBFGS,
                    .graph_size = LM_GGML_DEFAULT_GRAPH_SIZE,
                    .n_threads  = 1,
                    .past       = 0,
                    .delta      = 1e-5f,

                    .max_no_improvement = 0,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .n_gradient_accumulation = 1,

                    .lbfgs = {
                        .m              = 6,
                        .n_iter         = 100,
                        .max_linesearch = 20,

                        .eps      = 1e-5f,
                        .ftol     = 1e-4f,
                        .wolfe    = 0.9f,
                        .min_step = 1e-20f,
                        .max_step = 1e+20f,

                        .linesearch = LM_GGML_LINESEARCH_DEFAULT,
                    },
                };
            } break;
    }

    return result;
}

LM_GGML_API void lm_ggml_opt_init(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_context * opt,
        struct lm_ggml_opt_params params,
        int64_t nx) {
    opt->ctx = ctx;
    opt->params = params;
    opt->iter = 0;
    opt->nx = nx;
    opt->just_initialized = true;
    if (opt->ctx == NULL) {
        struct lm_ggml_init_params ctx_opt_params;
        if (opt->params.type == LM_GGML_OPT_TYPE_ADAM) {
            ctx_opt_params.mem_size = LM_GGML_MEM_ALIGN*3 + lm_ggml_tensor_overhead()*3 + lm_ggml_type_size(LM_GGML_TYPE_F32)*nx*3;
            if (opt->params.past > 0) {
                ctx_opt_params.mem_size += LM_GGML_MEM_ALIGN + lm_ggml_tensor_overhead() + lm_ggml_type_size(LM_GGML_TYPE_F32)*opt->params.past;
            }
        } else if (opt->params.type == LM_GGML_OPT_TYPE_LBFGS) {
            ctx_opt_params.mem_size = LM_GGML_MEM_ALIGN*9 + lm_ggml_tensor_overhead()*9 + lm_ggml_type_size(LM_GGML_TYPE_F32)*(nx*5 + opt->params.lbfgs.m*2 + nx*opt->params.lbfgs.m*2);
            if (opt->params.past > 0) {
                ctx_opt_params.mem_size += LM_GGML_MEM_ALIGN + lm_ggml_tensor_overhead() + lm_ggml_type_size(LM_GGML_TYPE_F32)*opt->params.past;
            }
        }
        ctx_opt_params.mem_buffer = NULL;
        ctx_opt_params.no_alloc   = false;

        opt->ctx = lm_ggml_init(ctx_opt_params);
    }
    switch (opt->params.type) {
        case LM_GGML_OPT_TYPE_ADAM:
            {
                opt->adam.g  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->adam.m  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->adam.v  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->adam.pf = params.past > 0
                    ? lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, params.past)
                    : NULL;
                lm_ggml_set_zero(opt->adam.m);
                lm_ggml_set_zero(opt->adam.v);
                if (opt->adam.pf) {
                    lm_ggml_set_zero(opt->adam.pf);
                }
            } break;
        case LM_GGML_OPT_TYPE_LBFGS:
            {
                opt->lbfgs.x  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->lbfgs.xp = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->lbfgs.g  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->lbfgs.gp = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->lbfgs.d  = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, nx);
                opt->lbfgs.pf = params.past > 0
                    ? lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, params.past)
                    : NULL;
                opt->lbfgs.lmal = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, params.lbfgs.m);
                opt->lbfgs.lmys = lm_ggml_new_tensor_1d(opt->ctx, LM_GGML_TYPE_F32, params.lbfgs.m);
                opt->lbfgs.lms  = lm_ggml_new_tensor_2d(opt->ctx, LM_GGML_TYPE_F32, nx, params.lbfgs.m);
                opt->lbfgs.lmy  = lm_ggml_new_tensor_2d(opt->ctx, LM_GGML_TYPE_F32, nx, params.lbfgs.m);
                lm_ggml_set_zero(opt->lbfgs.x);
                lm_ggml_set_zero(opt->lbfgs.xp);
                lm_ggml_set_zero(opt->lbfgs.g);
                lm_ggml_set_zero(opt->lbfgs.gp);
                lm_ggml_set_zero(opt->lbfgs.d);
                if (opt->lbfgs.pf) {
                    lm_ggml_set_zero(opt->lbfgs.pf);
                }
                lm_ggml_set_zero(opt->lbfgs.lmal);
                lm_ggml_set_zero(opt->lbfgs.lmys);
                lm_ggml_set_zero(opt->lbfgs.lms);
                lm_ggml_set_zero(opt->lbfgs.lmy);
            } break;
    }
}

enum lm_ggml_opt_result lm_ggml_opt(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_params params,
        struct lm_ggml_tensor * f) {
    bool free_ctx = false;
    if (ctx == NULL) {
        struct lm_ggml_init_params params_ctx = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        ctx = lm_ggml_init(params_ctx);
        if (ctx == NULL) {
            return LM_GGML_OPT_RESULT_NO_CONTEXT;
        }

        free_ctx = true;
    }

    enum lm_ggml_opt_result result = LM_GGML_OPT_RESULT_OK;

    struct lm_ggml_opt_context * opt = (struct lm_ggml_opt_context *) alloca(sizeof(struct lm_ggml_opt_context));

    lm_ggml_opt_init(ctx, opt, params, 0);
    result = lm_ggml_opt_resume(ctx, opt, f);

    if (free_ctx) {
        lm_ggml_free(ctx);
    }

    return result;
}

enum lm_ggml_opt_result lm_ggml_opt_resume(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_context * opt,
        struct lm_ggml_tensor * f) {

    // build forward + backward compute graphs
    struct lm_ggml_cgraph * gf = lm_ggml_new_graph_custom(ctx, opt->params.graph_size, true);
    lm_ggml_build_forward_expand(gf, f);

    struct lm_ggml_cgraph * gb = lm_ggml_graph_dup(ctx, gf);
    lm_ggml_build_backward_expand(ctx, gf, gb, true);

    return lm_ggml_opt_resume_g(ctx, opt, f, gf, gb, NULL, NULL);
}

enum lm_ggml_opt_result lm_ggml_opt_resume_g(
        struct lm_ggml_context * ctx,
        struct lm_ggml_opt_context * opt,
        struct lm_ggml_tensor * f,
        struct lm_ggml_cgraph * gf,
        struct lm_ggml_cgraph * gb,
        lm_ggml_opt_callback callback,
        void * callback_data) {

    // build forward + backward compute graphs
    enum lm_ggml_opt_result result = LM_GGML_OPT_RESULT_OK;

    switch (opt->params.type) {
        case LM_GGML_OPT_TYPE_ADAM:
            {
                result = lm_ggml_opt_adam(ctx, opt, opt->params, f, gf, gb, callback, callback_data);
            } break;
        case LM_GGML_OPT_TYPE_LBFGS:
            {
                result = lm_ggml_opt_lbfgs(ctx, opt, opt->params, f, gf, gb, callback, callback_data);
            } break;
    }

    if (opt->params.print_forward_graph) {
        lm_ggml_graph_print   (gf);
        lm_ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
    }

    if (opt->params.print_backward_graph) {
        lm_ggml_graph_print   (gb);
        lm_ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_set_input(struct lm_ggml_tensor * tensor) {
    tensor->flags |= LM_GGML_TENSOR_FLAG_INPUT;
}

void lm_ggml_set_output(struct lm_ggml_tensor * tensor) {
    tensor->flags |= LM_GGML_TENSOR_FLAG_OUTPUT;
}

////////////////////////////////////////////////////////////////////////////////

void lm_ggml_quantize_init(enum lm_ggml_type type) {
    lm_ggml_critical_section_start();

    switch (type) {
        case LM_GGML_TYPE_IQ2_XXS:
        case LM_GGML_TYPE_IQ2_XS:
        case LM_GGML_TYPE_IQ2_S:
        case LM_GGML_TYPE_IQ1_S:
        case LM_GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); break;
        case LM_GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); break;
        case LM_GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); break;
        default: // nothing
            break;
    }

    lm_ggml_critical_section_end();
}

void lm_ggml_quantize_free(void) {
    lm_ggml_critical_section_start();

    iq2xs_free_impl(LM_GGML_TYPE_IQ2_XXS);
    iq2xs_free_impl(LM_GGML_TYPE_IQ2_XS);
    iq2xs_free_impl(LM_GGML_TYPE_IQ1_S);
    iq3xs_free_impl(256);

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

static void lm_gguf_tensor_info_sanitize(struct lm_gguf_tensor_info * info) {
    LM_GGML_ASSERT(info->n_dims <= LM_GGML_MAX_DIMS);
    LM_GGML_ASSERT(0 <= info->type && info->type < LM_GGML_TYPE_COUNT);

    for (uint32_t i = 0; i < info->n_dims; ++i) {
        LM_GGML_ASSERT(info->ne[i] > 0);
    }

    // prevent overflow for total number of elements
    LM_GGML_ASSERT(INT64_MAX/info->ne[1] > info->ne[0]);
    LM_GGML_ASSERT(INT64_MAX/info->ne[2] > info->ne[0]*info->ne[1]);
    LM_GGML_ASSERT(INT64_MAX/info->ne[3] > info->ne[0]*info->ne[1]*info->ne[2]);
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

    p->data = LM_GGML_CALLOC(p->n + 1, 1);

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
    struct lm_gguf_context * ctx = LM_GGML_CALLOC(1, sizeof(struct lm_gguf_context));

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

    struct lm_gguf_context * ctx = LM_GGML_CALLOC(1, sizeof(struct lm_gguf_context));

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

        // header.n_kv will hold the actual value of pairs that were successfully read in the loop below
        ctx->header.n_kv = 0;
        ctx->kv = LM_GGML_CALLOC(n_kv, sizeof(struct lm_gguf_kv));

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

                                    kv->value.arr.data = LM_GGML_CALLOC(kv->value.arr.n, lm_gguf_type_size(kv->value.arr.type));

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

                                    kv->value.arr.data = LM_GGML_CALLOC(kv->value.arr.n, sizeof(struct lm_gguf_str));

                                    for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                        ok = ok && lm_gguf_fread_str(file, &((struct lm_gguf_str *) kv->value.arr.data)[j], &offset);
                                    }
                                } break;
                            case LM_GGUF_TYPE_ARRAY:
                            default: LM_GGML_ABORT("invalid type");
                        }
                    } break;
                default: LM_GGML_ABORT("invalid type");
            }

            if (!ok) {
                break;
            }

            ctx->header.n_kv++;
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
        ctx->infos = LM_GGML_CALLOC(ctx->header.n_tensors, sizeof(struct lm_gguf_tensor_info));

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

            // TODO: return an error instead of crashing with LM_GGML_ASSERT
            lm_gguf_tensor_info_sanitize(info);

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

            if (ne % lm_ggml_blck_size(info->type) != 0) {
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

////////////////////////////////////////////////////////////////////////////////

int lm_ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_neon(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_sve(void) {
#if defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_metal(void) {
#if defined(LM_GGML_USE_METAL)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_blas(void) {
#if defined(LM_GGML_USE_BLAS) || defined(LM_GGML_USE_CUDA) || defined(LM_GGML_USE_VULKAN) || defined(LM_GGML_USE_SYCL)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_cuda(void) {
#if defined(LM_GGML_USE_CUDA)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_vulkan(void) {
#if defined(LM_GGML_USE_VULKAN)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_kompute(void) {
#if defined(LM_GGML_USE_KOMPUTE)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_sycl(void) {
#if defined(LM_GGML_USE_SYCL)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_rpc(void) {
#if defined(LM_GGML_USE_RPC)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_cann(void) {
#if defined(LM_GGML_USE_CANN)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_llamafile(void) {
#if defined(LM_GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_gpublas(void) {
    return lm_ggml_cpu_has_cuda() || lm_ggml_cpu_has_vulkan() || lm_ggml_cpu_has_kompute() || lm_ggml_cpu_has_sycl();
}

int lm_ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int lm_ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
