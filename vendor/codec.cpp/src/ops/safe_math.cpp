#include "safe_math.h"

#include <limits>

bool codec_safe_add_i32(int32_t a, int32_t b, int32_t * out) {
    if (out == nullptr) {
        return false;
    }
    if (b > 0 && a > std::numeric_limits<int32_t>::max() - b) {
        return false;
    }
    if (b < 0 && a < std::numeric_limits<int32_t>::min() - b) {
        return false;
    }
    *out = a + b;
    return true;
}

bool codec_safe_mul_i32(int32_t a, int32_t b, int32_t * out) {
    if (out == nullptr) {
        return false;
    }
    if (a < 0 || b < 0) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > std::numeric_limits<int32_t>::max() / b) {
        return false;
    }
    *out = a * b;
    return true;
}
