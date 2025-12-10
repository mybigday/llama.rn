#include "JSIParams.h"
#include <cmath>
#include <algorithm>
#include <list>
#include <thread>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace rnllama_jsi {

#if defined(__ANDROID__)
    static inline int int_min(int a, int b) {
        return a < b ? a : b;
    }

    static void set_best_cores(cpu_params & params, int n_threads) {
        const int max_threads = (int) std::thread::hardware_concurrency();

        int default_n_threads = 0;
#if defined(LM_GGML_USE_HEXAGON)
        default_n_threads = 6;
        if (max_threads > 0) {
            default_n_threads = int_min(default_n_threads, max_threads);
        }
#else
        default_n_threads = max_threads == 4 ? 2 : int_min(4, max_threads);
#endif

        const int target_threads = (max_threads > 0 && n_threads > 0)
            ? int_min(n_threads, max_threads)
            : default_n_threads;

        params.n_threads = target_threads;

        std::vector<std::pair<int, int>> cores;
        for (int i = 0; i < max_threads; ++i) {
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq");
            int freq;
            if (f >> freq) {
                cores.emplace_back(freq, i);
            }
        }

        std::sort(cores.rbegin(), cores.rend());
        std::fill(std::begin(params.cpumask), std::end(params.cpumask), false);

        for (int i = 0; i < target_threads && i < (int) cores.size(); ++i) {
            params.cpumask[cores[i].second] = true;
        }

        params.strict_cpu = true;
        params.mask_valid = true;
    }
#endif

    std::string getPropertyAsString(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, const std::string& defaultValue) {
        if (obj.hasProperty(runtime, name)) {
            auto val = obj.getProperty(runtime, name);
            if (val.isString()) {
                return val.getString(runtime).utf8(runtime);
            }
        }
        return defaultValue;
    }

    int getPropertyAsInt(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, int defaultValue) {
        if (obj.hasProperty(runtime, name)) {
            auto val = obj.getProperty(runtime, name);
            if (val.isNumber()) {
                return (int)val.getNumber();
            }
        }
        return defaultValue;
    }

    double getPropertyAsDouble(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, double defaultValue) {
        if (obj.hasProperty(runtime, name)) {
            auto val = obj.getProperty(runtime, name);
            if (val.isNumber()) {
                return val.getNumber();
            }
        }
        return defaultValue;
    }

    bool getPropertyAsBool(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, bool defaultValue) {
        if (obj.hasProperty(runtime, name)) {
            auto val = obj.getProperty(runtime, name);
            if (val.isBool()) {
                return val.getBool();
            }
        }
        return defaultValue;
    }

    float getPropertyAsFloat(jsi::Runtime& runtime, const jsi::Object& obj, const char* name, float defaultValue) {
        if (obj.hasProperty(runtime, name)) {
            auto val = obj.getProperty(runtime, name);
            if (val.isNumber()) {
                return (float)val.getNumber();
            }
        }
        return defaultValue;
    }

    void parseCommonParams(jsi::Runtime& runtime, const jsi::Object& params, common_params& cparams) {
        // Model path
        cparams.model.path = getPropertyAsString(runtime, params, "model");
        cparams.vocab_only = getPropertyAsBool(runtime, params, "vocab_only", false);
        if (cparams.vocab_only) {
            cparams.warmup = false;
        }

        cparams.n_ctx = getPropertyAsInt(runtime, params, "n_ctx", cparams.n_ctx);
        cparams.n_batch = getPropertyAsInt(runtime, params, "n_batch", cparams.n_batch);
        cparams.n_ubatch = getPropertyAsInt(runtime, params, "n_ubatch", cparams.n_ubatch);
        cparams.n_parallel = getPropertyAsInt(runtime, params, "n_parallel", cparams.n_parallel);
        cparams.cpuparams.n_threads = getPropertyAsInt(runtime, params, "n_threads", cparams.cpuparams.n_threads);

        std::string cpuMask = getPropertyAsString(runtime, params, "cpu_mask");
#if defined(__ANDROID__)
        set_best_cores(cparams.cpuparams, cparams.cpuparams.n_threads);
#endif

        cparams.n_gpu_layers = getPropertyAsInt(runtime, params, "n_gpu_layers", cparams.n_gpu_layers);
        if (!cpuMask.empty()) {
            bool cpumask[LM_GGML_MAX_N_THREADS] = {false};
            if (parse_cpu_mask(cpuMask, cpumask)) {
                std::copy(std::begin(cpumask), std::end(cpumask), std::begin(cparams.cpuparams.cpumask));
                cparams.cpuparams.mask_valid = true;
            }
        }
        cparams.cpuparams.strict_cpu = getPropertyAsBool(runtime, params, "cpu_strict", cparams.cpuparams.strict_cpu);

        // Chat template
        std::string chatTemplate = getPropertyAsString(runtime, params, "chat_template");
        if (!chatTemplate.empty()) {
            cparams.chat_template = chatTemplate;
        }

        cparams.use_mlock = getPropertyAsBool(runtime, params, "use_mlock", cparams.use_mlock);
        cparams.use_mmap = getPropertyAsBool(runtime, params, "use_mmap", cparams.use_mmap);

        if (params.hasProperty(runtime, "flash_attn")) {
            bool fa = getPropertyAsBool(runtime, params, "flash_attn", false);
            cparams.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        if (params.hasProperty(runtime, "flash_attn_type")) {
            std::string fa = getPropertyAsString(runtime, params, "flash_attn_type");
             cparams.flash_attn_type = static_cast<enum llama_flash_attn_type>(rnllama::flash_attn_type_from_str(fa));
        }

        std::string ck = getPropertyAsString(runtime, params, "cache_type_k");
        if (!ck.empty()) cparams.cache_type_k = rnllama::kv_cache_type_from_str(ck);

        std::string cv = getPropertyAsString(runtime, params, "cache_type_v");
        if (!cv.empty()) cparams.cache_type_v = rnllama::kv_cache_type_from_str(cv);

        cparams.ctx_shift = getPropertyAsBool(runtime, params, "ctx_shift", cparams.ctx_shift);
        cparams.kv_unified = getPropertyAsBool(runtime, params, "kv_unified", cparams.kv_unified);
        cparams.swa_full = getPropertyAsBool(runtime, params, "swa_full", cparams.swa_full);

        if (params.hasProperty(runtime, "embedding") && getPropertyAsBool(runtime, params, "embedding")) {
            cparams.embedding = true;
            cparams.n_ubatch = cparams.n_batch; // Default for non-causal
            cparams.embd_normalize = getPropertyAsInt(runtime, params, "embd_normalize", cparams.embd_normalize);
        }

        int pooling_type = getPropertyAsInt(runtime, params, "pooling_type", -1);
        if (pooling_type >= 0) {
            cparams.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
        }

        cparams.rope_freq_base = getPropertyAsFloat(runtime, params, "rope_freq_base", cparams.rope_freq_base);
        cparams.rope_freq_scale = getPropertyAsFloat(runtime, params, "rope_freq_scale", cparams.rope_freq_scale);

        int n_cpu_moe = getPropertyAsInt(runtime, params, "n_cpu_moe", 0);
        if (n_cpu_moe > 0) {
            static std::list<std::string> buft_overrides;
            for (int i = 0; i < n_cpu_moe; ++i) {
                std::string pattern = "blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate)_exps";
                buft_overrides.push_back(pattern);
                cparams.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), lm_ggml_backend_cpu_buffer_type()});
            }
            cparams.tensor_buft_overrides.push_back({nullptr, nullptr});
        }

        // LoRA
        if (params.hasProperty(runtime, "lora")) {
            std::string loraPath = getPropertyAsString(runtime, params, "lora");
            if (!loraPath.empty()) {
                common_adapter_lora_info la;
                la.path = loraPath;
                la.scale = getPropertyAsFloat(runtime, params, "lora_scaled", 1.0f);
                cparams.lora_adapters.push_back(la);
            }
        }

        if (params.hasProperty(runtime, "lora_list")) {
            jsi::Array loraList = params.getProperty(runtime, "lora_list").asObject(runtime).asArray(runtime);
            for (size_t i = 0; i < loraList.size(runtime); i++) {
                jsi::Object item = loraList.getValueAtIndex(runtime, i).asObject(runtime);
                std::string path = getPropertyAsString(runtime, item, "path");
                if (!path.empty()) {
                    common_adapter_lora_info la;
                    la.path = path;
                    la.scale = getPropertyAsFloat(runtime, item, "scaled", 1.0f);
                    cparams.lora_adapters.push_back(la);
                }
            }
        }
    }

    void parseCompletionParams(jsi::Runtime& runtime, const jsi::Object& params, rnllama::llama_rn_context* ctx) {
        if (!ctx) return;

        ctx->params.prompt = getPropertyAsString(runtime, params, "prompt");

        auto& sparams = ctx->params.sampling;
        sparams.seed = getPropertyAsInt(runtime, params, "seed", -1);
        ctx->params.n_predict = getPropertyAsInt(runtime, params, "n_predict", ctx->params.n_predict);
        ctx->params.sampling.ignore_eos = getPropertyAsBool(runtime, params, "ignore_eos", ctx->params.sampling.ignore_eos);

        sparams.temp = getPropertyAsDouble(runtime, params, "temperature", sparams.temp);
        sparams.n_probs = getPropertyAsInt(runtime, params, "n_probs", sparams.n_probs);

        sparams.penalty_last_n = getPropertyAsInt(runtime, params, "penalty_last_n", sparams.penalty_last_n);
        sparams.penalty_repeat = getPropertyAsDouble(runtime, params, "penalty_repeat", sparams.penalty_repeat);
        sparams.penalty_freq = getPropertyAsDouble(runtime, params, "penalty_freq", sparams.penalty_freq);
        sparams.penalty_present = getPropertyAsDouble(runtime, params, "penalty_present", sparams.penalty_present);

        sparams.mirostat = getPropertyAsInt(runtime, params, "mirostat", sparams.mirostat);
        sparams.mirostat_tau = getPropertyAsDouble(runtime, params, "mirostat_tau", sparams.mirostat_tau);
        sparams.mirostat_eta = getPropertyAsDouble(runtime, params, "mirostat_eta", sparams.mirostat_eta);

        sparams.top_k = getPropertyAsInt(runtime, params, "top_k", sparams.top_k);
        sparams.top_p = getPropertyAsDouble(runtime, params, "top_p", sparams.top_p);
        sparams.min_p = getPropertyAsDouble(runtime, params, "min_p", sparams.min_p);

        sparams.xtc_threshold = getPropertyAsDouble(runtime, params, "xtc_threshold", sparams.xtc_threshold);
        sparams.xtc_probability = getPropertyAsDouble(runtime, params, "xtc_probability", sparams.xtc_probability);
        sparams.typ_p = getPropertyAsDouble(runtime, params, "typical_p", sparams.typ_p);

        sparams.dry_multiplier = getPropertyAsDouble(runtime, params, "dry_multiplier", sparams.dry_multiplier);
        sparams.dry_base = getPropertyAsDouble(runtime, params, "dry_base", sparams.dry_base);
        sparams.dry_allowed_length = getPropertyAsInt(runtime, params, "dry_allowed_length", sparams.dry_allowed_length);
        sparams.dry_penalty_last_n = getPropertyAsInt(runtime, params, "dry_penalty_last_n", sparams.dry_penalty_last_n);
        if (params.hasProperty(runtime, "dry_sequence_breakers")) {
            auto breakersVal = params.getProperty(runtime, "dry_sequence_breakers");
            if (breakersVal.isObject() && breakersVal.asObject(runtime).isArray(runtime)) {
                jsi::Array breakers = breakersVal.asObject(runtime).asArray(runtime);
                sparams.dry_sequence_breakers.clear();
                for (size_t i = 0; i < breakers.size(runtime); i++) {
                    auto breakerVal = breakers.getValueAtIndex(runtime, i);
                    if (breakerVal.isString()) {
                        sparams.dry_sequence_breakers.push_back(breakerVal.asString(runtime).utf8(runtime));
                    }
                }
            }
        }

        sparams.top_n_sigma = getPropertyAsDouble(runtime, params, "top_n_sigma", sparams.top_n_sigma);

        // Grammar
        std::string grammar = getPropertyAsString(runtime, params, "grammar");
        if (!grammar.empty()) {
            sparams.grammar = grammar;
        }

        std::string jsonSchema = getPropertyAsString(runtime, params, "json_schema");
        if (!jsonSchema.empty() && grammar.empty()) {
            sparams.grammar = json_schema_to_grammar(json::parse(jsonSchema));
        }

        sparams.grammar_lazy = getPropertyAsBool(runtime, params, "grammar_lazy", false);

        if (params.hasProperty(runtime, "preserved_tokens")) {
            sparams.preserved_tokens.clear();
            auto preservedVal = params.getProperty(runtime, "preserved_tokens");
            if (preservedVal.isObject() && preservedVal.asObject(runtime).isArray(runtime)) {
                jsi::Array preserved = preservedVal.asObject(runtime).asArray(runtime);
                for (size_t i = 0; i < preserved.size(runtime); ++i) {
                    auto tokenVal = preserved.getValueAtIndex(runtime, i);
                    if (!tokenVal.isString()) {
                        continue;
                    }
                    std::string tokenStr = tokenVal.asString(runtime).utf8(runtime);
                    auto ids = common_tokenize(ctx->ctx, tokenStr.c_str(), /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        sparams.preserved_tokens.insert(ids[0]);
                    }
                }
            }
        }

        if (params.hasProperty(runtime, "grammar_triggers")) {
            sparams.grammar_triggers.clear();
            auto triggersVal = params.getProperty(runtime, "grammar_triggers");
            if (triggersVal.isObject() && triggersVal.asObject(runtime).isArray(runtime)) {
                jsi::Array triggers = triggersVal.asObject(runtime).asArray(runtime);
                for (size_t i = 0; i < triggers.size(runtime); ++i) {
                    auto triggerVal = triggers.getValueAtIndex(runtime, i);
                    if (!triggerVal.isObject()) {
                        continue;
                    }
                    jsi::Object triggerObj = triggerVal.asObject(runtime);
                    auto type = static_cast<common_grammar_trigger_type>(getPropertyAsInt(runtime, triggerObj, "type", 0));
                    std::string word = getPropertyAsString(runtime, triggerObj, "value");
                    if (word.empty()) {
                        continue;
                    }

                    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                        auto ids = common_tokenize(ctx->ctx, word.c_str(), /* add_special= */ false, /* parse_special= */ true);
                        if (ids.size() == 1) {
                            const llama_token token = ids[0];
                            if (sparams.preserved_tokens.find(token) == sparams.preserved_tokens.end()) {
                                throw std::runtime_error("Grammar trigger word should be marked as preserved token");
                            }
                            common_grammar_trigger trigger;
                            trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                            trigger.value = word;
                            trigger.token = token;
                            sparams.grammar_triggers.push_back(std::move(trigger));
                        } else {
                            sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                        }
                    } else {
                        common_grammar_trigger trigger;
                        trigger.type = type;
                        trigger.value = word;
                        if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                            trigger.token = (llama_token)getPropertyAsInt(runtime, triggerObj, "token", 0);
                        }
                        sparams.grammar_triggers.push_back(std::move(trigger));
                    }
                }
            }
        }

        // Logit bias
        sparams.logit_bias.clear();
        const llama_model * model = llama_get_model(ctx->ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        if (ctx->params.sampling.ignore_eos) {
            sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
        }

        if (params.hasProperty(runtime, "logit_bias")) {
            auto logitBias = params.getProperty(runtime, "logit_bias").asObject(runtime).asArray(runtime);
            for (size_t i = 0; i < logitBias.size(runtime); i++) {
                auto el = logitBias.getValueAtIndex(runtime, i).asObject(runtime).asArray(runtime);
                if (el.size(runtime) == 2) {
                    int tok = (int)el.getValueAtIndex(runtime, 0).asNumber();
                    auto val = el.getValueAtIndex(runtime, 1);
                    if (val.isNumber()) {
                        sparams.logit_bias[tok].bias = val.asNumber();
                    } else if (val.isBool() && !val.getBool()) {
                        sparams.logit_bias[tok].bias = -INFINITY;
                    }
                }
            }
        }

        ctx->params.antiprompt.clear();
        if (params.hasProperty(runtime, "stop")) {
            jsi::Array stop = params.getProperty(runtime, "stop").asObject(runtime).asArray(runtime);
            for (size_t i = 0; i < stop.size(runtime); i++) {
                ctx->params.antiprompt.push_back(stop.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
            }
        }

#if defined(__ANDROID__)
        if (params.hasProperty(runtime, "n_threads")) {
            int nThreads = getPropertyAsInt(runtime, params, "n_threads", ctx->params.cpuparams.n_threads);
            set_best_cores(ctx->params.cpuparams, nThreads);
        }
#else
        if (params.hasProperty(runtime, "n_threads")) {
            int nThreads = getPropertyAsInt(runtime, params, "n_threads", ctx->params.cpuparams.n_threads);
            const int maxThreads = (int) std::thread::hardware_concurrency();
            const int defaultNThreads = nThreads == 4 ? 2 : (maxThreads > 0 ? std::min(4, maxThreads) : 4);
            ctx->params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
        }
#endif
    }
}
