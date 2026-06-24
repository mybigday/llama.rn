#include "models.h"
#include <cmath>

lm_ggml_cgraph * clip_graph_gemma4ua::build() {
    lm_ggml_tensor * inp = build_inp_raw(1);

    auto cur = lm_ggml_cont(ctx0, lm_ggml_permute(ctx0, inp, 1, 0, 2, 3));

    // Gemma4UnifiedMultimodalEmbedder
    {
        // embedding_pre_projection_norm
        cur = lm_ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = build_mm(model.mm_input_proj_w, cur);
        cb(cur, "projected", -1);
    }

    lm_ggml_build_forward_expand(gf, cur);
    return gf;
}
