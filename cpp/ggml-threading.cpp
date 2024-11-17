#include "ggml-threading.h"
#include <mutex>

std::mutex lm_ggml_critical_section_mutex;

void lm_ggml_critical_section_start() {
    lm_ggml_critical_section_mutex.lock();
}

void lm_ggml_critical_section_end(void) {
    lm_ggml_critical_section_mutex.unlock();
}
