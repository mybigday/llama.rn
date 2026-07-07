// barbet_hidden_dump — dump per-position backbone hiddens for parity debugging.
//
// Tokenizes the BlueMagpie prompt exactly like the completion path does
// (add_bos = vocab default, parse_special = true), runs one llama_decode
// with a sched eval callback that captures `l_out-<il>` and `result_norm`,
// and prints per-position L2 norms + the raw final-norm hiddens (to a .npy)
// so a Python script can compare against stop_gen.npz `prefill_hiddens`.
//
// Usage:
//   barbet_hidden_dump --backbone BARBET.gguf --text "..." [--out hid.bin] [--layer N]
//
// Output file (--out): raw float32, [n_tokens, n_embd] row-major, of result_norm.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "common.h"

struct capture {
    std::string want;             // tensor name prefix to grab (e.g. "result_norm")
    std::vector<float> data;      // captured (flattened)
    int64_t ne0 = 0, ne1 = 0;
};

static capture g_final;           // result_norm
static int     g_dump_layer = -1; // if >=0 also capture l_out-<layer>
static capture g_layer;

static bool g_verbose_names = false;

static bool eval_cb(struct lm_ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) {
        if (g_verbose_names && t->name && t->name[0]) std::printf("[cb] %s\n", t->name);
        // Observe result_norm and the requested layer's l_out.
        if (t->name && (std::strcmp(t->name, "result_norm") == 0 ||
                        std::strcmp(t->name, "result_embd") == 0)) return true;
        if (g_dump_layer >= 0 && t->name) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "l_out-%d", g_dump_layer);
            if (std::strcmp(t->name, buf) == 0) return true;
        }
        return false;
    }
    // ask == false: data is ready.
    capture * dst = nullptr;
    if (std::strcmp(t->name, "result_norm") == 0 ||
        std::strcmp(t->name, "result_embd") == 0) dst = &g_final;
    else dst = &g_layer;
    dst->ne0 = t->ne[0];
    dst->ne1 = t->ne[1];
    const size_t n = (size_t) t->ne[0] * t->ne[1];
    dst->data.resize(n);
    lm_ggml_backend_tensor_get(t, dst->data.data(), 0, n * sizeof(float));
    return true;
}

int main(int argc, char ** argv) {
    std::string backbone, text = "你好，今天天氣真好。";
    std::string out_path;
    for (int i = 1; i < argc; ++i) {
        auto is = [&](const char * k){ return std::strcmp(argv[i], k) == 0; };
        if      (is("--backbone") && i+1 < argc) backbone = argv[++i];
        else if (is("--text")     && i+1 < argc) text     = argv[++i];
        else if (is("--out")      && i+1 < argc) out_path = argv[++i];
        else if (is("--layer")    && i+1 < argc) g_dump_layer = std::atoi(argv[++i]);
        else if (is("--names"))                   g_verbose_names = true;
    }
    if (backbone.empty()) { std::fprintf(stderr, "need --backbone\n"); return 2; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(backbone.c_str(), mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 1024;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;   // per-token final-norm hiddens
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "ctx failed\n"); return 4; }
    llama_set_embeddings(ctx, true);

    // NOTE: <|bm_spk|>/<|bm_audio_start|> are the converter-baked CONTROL
    // strings for the Megatron padding-region ids (spk=114826,
    // audio_start=114822) that the model was actually trained on — NOT the
    // tokenizer.json <|speaker|>/<|audio_start|> entries.  Requires a GGUF
    // converted after the bm_special registration landed.
    const std::string prompt = "<|bm_spk|>" + text + "<|bm_audio_start|>";
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> toks = common_tokenize(ctx, prompt, add_bos, /*parse_special=*/true);

    std::printf("[dump] add_bos=%d n_tokens=%d\n", (int) add_bos, (int) toks.size());
    std::printf("[dump] tokens:");
    for (auto tk : toks) std::printf(" %d", tk);
    std::printf("\n[dump] pieces:");
    for (auto tk : toks) {
        char buf[128];
        int n = llama_token_to_piece(vocab, tk, buf, sizeof(buf), 0, true);
        std::printf(" [%.*s]", n > 0 ? n : 0, buf);
    }
    std::printf("\n");

    bool stepwise = false;
    std::string embd_bin;   // raw f32 [n_steps, n_embd]: inject after the prompt
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--stepwise") == 0) stepwise = true;
        if (std::strcmp(argv[i], "--embd-bin") == 0 && i + 1 < argc) embd_bin = argv[i + 1];
    }

    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    if (!stepwise) {
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = (llama_pos) i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;   // output all positions
        }
        batch.n_tokens = (int) toks.size();
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 5; }
    } else {
        // One llama_decode per token: exercises the incremental (batch=1)
        // recurrent-state path that generation feedback steps use.  Collect
        // each position's hidden as it is produced.
        const int64_t H2 = llama_model_n_embd(model);
        g_final.ne0 = H2;
        g_final.ne1 = (int64_t) toks.size();
        g_final.data.resize((size_t) H2 * toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            batch.token[0]     = toks[i];
            batch.pos[0]       = (llama_pos) i;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            batch.n_tokens     = 1;
            if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "step decode failed @%zu\n", i); return 5; }
            const float * e = llama_get_embeddings_ith(ctx, -1);
            if (!e) { std::fprintf(stderr, "no embd @%zu\n", i); return 6; }
            std::memcpy(g_final.data.data() + i * (size_t) H2, e, (size_t) H2 * sizeof(float));
        }
    }

    // Embd-injection mode: after the prompt, feed each row of --embd-bin as a
    // single-position embd batch (mirrors llama.rn's feedback injection) and
    // capture the produced hidden.  Overwrites g_final with [n_steps, H].
    if (!embd_bin.empty()) {
        const int64_t H2 = llama_model_n_embd(model);
        FILE * f = std::fopen(embd_bin.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", embd_bin.c_str()); return 7; }
        std::fseek(f, 0, SEEK_END);
        const long fsz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        const int64_t n_steps = fsz / (int64_t) (H2 * sizeof(float));
        std::vector<float> embds((size_t) n_steps * H2);
        if (std::fread(embds.data(), sizeof(float), embds.size(), f) != embds.size()) { std::fclose(f); return 7; }
        std::fclose(f);
        std::printf("[dump] injecting %lld embd steps after the %zu-token prompt\n",
                    (long long) n_steps, toks.size());
        llama_batch eb = llama_batch_init(1, (int32_t) H2, 1);
        std::vector<float> hidden_out((size_t) n_steps * H2);
        for (int64_t sidx = 0; sidx < n_steps; ++sidx) {
            eb.n_tokens    = 1;
            eb.token       = nullptr;
            std::memcpy(eb.embd, embds.data() + sidx * H2, (size_t) H2 * sizeof(float));
            eb.pos[0]       = (llama_pos) (toks.size() + (size_t) sidx);
            eb.n_seq_id[0]  = 1;
            eb.seq_id[0][0] = 0;
            eb.logits[0]    = 1;
            if (llama_decode(ctx, eb) != 0) { std::fprintf(stderr, "embd decode failed @%lld\n", (long long) sidx); return 8; }
            const float * e = llama_get_embeddings_ith(ctx, -1);
            if (!e) { std::fprintf(stderr, "no embd out @%lld\n", (long long) sidx); return 8; }
            std::memcpy(hidden_out.data() + sidx * H2, e, (size_t) H2 * sizeof(float));
        }
        llama_batch_free(eb);
        g_final.ne0 = H2; g_final.ne1 = n_steps;
        g_final.data = std::move(hidden_out);
    }

    // Primary capture path: per-position embeddings straight from the context
    // (embeddings=true + logits on every position).  The cb_eval capture is
    // kept as a secondary source for --layer dumps.
    if (g_final.data.empty()) {
        const int64_t H2 = llama_model_n_embd(model);
        g_final.ne0 = H2;
        g_final.ne1 = (int64_t) toks.size();
        g_final.data.resize((size_t) H2 * toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            const float * e = llama_get_embeddings_ith(ctx, (int32_t) i);
            if (e == nullptr) { std::fprintf(stderr, "no embd at pos %zu\n", i); return 6; }
            std::memcpy(g_final.data.data() + i * (size_t) H2, e, (size_t) H2 * sizeof(float));
        }
    }
    std::printf("[dump] result_norm ne=[%lld,%lld]\n",
                (long long) g_final.ne0, (long long) g_final.ne1);
    const int64_t H = g_final.ne0;
    const int64_t T = g_final.ne1;
    std::printf("[dump] per-position L2 norm (result_norm):\n");
    for (int64_t p = 0; p < T; ++p) {
        double s = 0;
        for (int64_t k = 0; k < H; ++k) { float v = g_final.data[p*H + k]; s += (double) v*v; }
        std::printf("  pos %lld : %.4f\n", (long long) p, std::sqrt(s));
    }
    if (g_dump_layer >= 0 && !g_layer.data.empty()) {
        std::printf("[dump] l_out-%d per-position L2:\n", g_dump_layer);
        const int64_t Hl = g_layer.ne0, Tl = g_layer.ne1;
        for (int64_t p = 0; p < Tl; ++p) {
            double s = 0;
            for (int64_t k = 0; k < Hl; ++k) { float v = g_layer.data[p*Hl+k]; s += (double)v*v; }
            std::printf("  pos %lld : %.4f\n", (long long) p, std::sqrt(s));
        }
    }

    if (!out_path.empty()) {
        // Prefer result_norm; fall back to the captured layer (l_out-<layer>)
        // when the final-norm node was renamed by the embeddings pooling path.
        const capture & src = !g_final.data.empty() ? g_final : g_layer;
        FILE * f = std::fopen(out_path.c_str(), "wb");
        if (f) {
            int64_t To = src.ne1, Ho = src.ne0;
            std::fwrite(&To, sizeof(int64_t), 1, f);
            std::fwrite(&Ho, sizeof(int64_t), 1, f);
            std::fwrite(src.data.data(), sizeof(float), (size_t) To*Ho, f);
            std::fclose(f);
            std::printf("[dump] wrote %s (%s ne=[%lld,%lld])\n", out_path.c_str(),
                        !g_final.data.empty() ? "result_norm" : "l_out",
                        (long long) Ho, (long long) To);
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
