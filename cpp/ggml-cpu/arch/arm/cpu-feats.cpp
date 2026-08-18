#include "ggml-backend-impl.h"
#include "ggml-feats.h"

#if defined(__aarch64__) || defined(_M_ARM64)

static int lm_ggml_backend_cpu_aarch64_score() {
    int score = 1;
    const lm_ggml_feats_arch64_runtime_t af = lm_ggml_feats_get_arch64_runtime();
    LM_GGML_UNUSED(af);

#ifdef LM_GGML_USE_DOTPROD
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif
#ifdef LM_GGML_USE_FP16_VECTOR_ARITHMETIC
    if (!af.has_fp16) { return 0; }
    score += 1<<2;
#endif
#ifdef LM_GGML_USE_SVE
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif
#ifdef LM_GGML_USE_MATMUL_INT8
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif
#ifdef LM_GGML_USE_SVE2
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif
#ifdef LM_GGML_USE_SME
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif

    return score;
}

LM_GGML_BACKEND_DL_SCORE_IMPL(lm_ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__) || defined(_M_ARM64)
