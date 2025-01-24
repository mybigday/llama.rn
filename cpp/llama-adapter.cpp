#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <algorithm>
#include <map>
#include <cassert>
#include <stdexcept>

// vec

struct lm_ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

struct lm_ggml_tensor * llama_adapter_cvec::apply_to(struct lm_ggml_context * ctx, struct lm_ggml_tensor * cur, int  il) const {
    lm_ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = lm_ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    LM_GGML_ASSERT(tensors.empty());
    LM_GGML_ASSERT(ctxs.empty());
    LM_GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<lm_ggml_backend_buffer_type_t, lm_ggml_context *> ctx_map;
    auto ctx_for_buft = [&](lm_ggml_backend_buffer_type_t buft) -> lm_ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            struct lm_ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer*lm_ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            lm_ggml_context * ctx = lm_ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer);
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer; il++) {
        lm_ggml_backend_buffer_type_t buft = model.select_buft(il);
        lm_ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        lm_ggml_tensor * tensor = lm_ggml_new_tensor_1d(ctx, LM_GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        lm_ggml_backend_buffer_type_t buft = it.first;
        lm_ggml_context * ctx = it.second;
        lm_ggml_backend_buffer_t buf = lm_ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        lm_ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

int32_t llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return 0;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return 1;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return 1;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer; il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            lm_ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * lm_ggml_element_size(tensors[il]));
        }
    }

    return 0;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(struct lm_ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static void llama_adapter_lora_init_impl(struct llama_model & model, const char * path_lora, struct llama_adapter_lora & adapter) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    lm_ggml_context * ctx_init;
    struct lm_gguf_init_params meta_lm_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    lm_gguf_context_ptr ctx_gguf { lm_gguf_init_from_file(path_lora, meta_lm_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter file from " + std::string(path_lora));
    }

    lm_ggml_context_ptr ctx { ctx_init };

    // check metadata
    {
        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = lm_gguf_find_key(ctx_gguf.get(), key.c_str());
            return id < 0 ? "" : std::string(lm_gguf_get_val_str(ctx_gguf.get(), id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = lm_gguf_find_key(ctx_gguf.get(), key.c_str());
            return id < 0 ? 0.0f : lm_gguf_get_val_f32(ctx_gguf.get(), id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));
    }

    int n_tensors = lm_gguf_get_n_tensors(ctx_gguf.get());

    // contexts for each buffer type
    std::map<lm_ggml_backend_buffer_type_t, lm_ggml_context *> ctx_map;
    auto ctx_for_buft = [&](lm_ggml_backend_buffer_type_t buft) -> lm_ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            struct lm_ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*lm_ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            lm_ggml_context * buft_ctx = lm_ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (lm_ggml_tensor * cur = lm_ggml_get_first_tensor(ctx.get()); cur; cur = lm_ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        const auto * model_tensor = model.get_tensor(name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }

        struct lm_ggml_context * dev_ctx = ctx_for_buft(lm_ggml_backend_buffer_get_type(model_tensor->buffer));
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        struct lm_ggml_tensor * tensor_a = lm_ggml_dup_tensor(dev_ctx, w.a);
        struct lm_ggml_tensor * tensor_b = lm_ggml_dup_tensor(dev_ctx, w.b);
        lm_ggml_set_name(tensor_a, w.a->name);
        lm_ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            lm_ggml_backend_buffer_type_t buft = it.first;
            lm_ggml_context * ctx_dev = it.second;
            lm_ggml_backend_buffer_ptr buf { lm_ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, lm_ggml_backend_buffer_name(buf.get()), lm_ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        llama_file lm_gguf_file(path_lora, "rb");
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](struct lm_ggml_tensor * orig, struct lm_ggml_tensor * dev) {
            size_t offs = lm_gguf_get_data_offset(ctx_gguf.get()) + lm_gguf_get_tensor_offset(ctx_gguf.get(), lm_gguf_find_tensor(ctx_gguf.get(), orig->name));
            size_t size = lm_ggml_nbytes(orig);
            read_buf.resize(size);
            lm_gguf_file.seek(offs, SEEK_SET);
            lm_gguf_file.read_raw(read_buf.data(), size);
            lm_ggml_backend_tensor_set(dev, read_buf.data(), 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

struct llama_adapter_lora * llama_adapter_lora_init(struct llama_model * model, const char * path_lora) {
    struct llama_adapter_lora * adapter = new llama_adapter_lora();

    try {
        llama_adapter_lora_init_impl(*model, path_lora, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

void llama_adapter_lora_free(struct llama_adapter_lora * adapter) {
    delete adapter;
}
