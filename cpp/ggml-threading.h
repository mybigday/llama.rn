#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void lm_ggml_critical_section_start(void);
void lm_ggml_critical_section_end(void);

#ifdef __cplusplus
}
#endif
