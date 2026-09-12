#include "JSIParams.h"
#if defined(RNLLAMA_USE_FRAMEWORK_HEADERS)
#include <rnllama/speculative.h>
#else
#include "speculative.h"
#endif
#include <cmath>
#include <algorithm>
#include <list>
#include <thread>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>

namespace rnllama_jsi {

#if defined(__ANDROID__)
    static inline int int_min(int a, int b) {
        return a < b ? a : b;
    }

    static void set_best_cores(common_cpu_params & params, int n_threads) {
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

#if defined(__APPLE__)
    static int default_apple_n_threads() {
        return std::max(1, common_cpu_get_num_math() / 2);
    }
#endif

    // ---- json lookups -----------------------------------------------------

    static const json* findProperty(const json& obj, const char* name) {
        if (!obj.is_object()) return nullptr;
        auto it = obj.find(name);
        return it == obj.end() ? nullptr : &*it;
    }

    std::string getPropertyAsString(const json& obj, const char* name, const std::string& defaultValue) {
        const json* val = findProperty(obj, name);
        return (val && val->is_string()) ? val->get<std::string>() : defaultValue;
    }

    int getPropertyAsInt(const json& obj, const char* name, int defaultValue) {
        const json* val = findProperty(obj, name);
        return (val && val->is_number()) ? (int) val->get<double>() : defaultValue;
    }

    double getPropertyAsDouble(const json& obj, const char* name, double defaultValue) {
        const json* val = findProperty(obj, name);
        return (val && val->is_number()) ? val->get<double>() : defaultValue;
    }

    bool getPropertyAsBool(const json& obj, const char* name, bool defaultValue) {
        const json* val = findProperty(obj, name);
        return (val && val->is_boolean()) ? val->get<bool>() : defaultValue;
    }

    float getPropertyAsFloat(const json& obj, const char* name, float defaultValue) {
        const json* val = findProperty(obj, name);
        return (val && val->is_number()) ? (float) val->get<double>() : defaultValue;
    }

    static bool hasProperty(const json& obj, const char* name) {
        return findProperty(obj, name) != nullptr;
    }

    std::vector<common_adapter_lora_info> parseLoraAdapters(const json& list) {
        std::vector<common_adapter_lora_info> adapters;
        if (!list.is_array()) return adapters;
        for (const auto& item : list) {
            if (!item.is_object()) continue;
            std::string path = getPropertyAsString(item, "path");
            if (path.empty()) continue;
            common_adapter_lora_info la;
            la.path = path;
            la.scale = getPropertyAsFloat(item, "scaled", 1.0f);
            adapters.push_back(la);
        }
        return adapters;
    }

    // ---- speculative decoding options -----------------------------------

    static std::string normalizeSpeculativeTypeName(std::string name) {
        if (name == "mtp") {
            return "draft-mtp";
        }
        return name;
    }

    static void addSpeculativeTypeName(std::vector<std::string>& typeNames, std::string name) {
        name = normalizeSpeculativeTypeName(std::move(name));
        if (std::find(typeNames.begin(), typeNames.end(), name) == typeNames.end()) {
            typeNames.push_back(std::move(name));
        }
    }

    // Accepts a single type name or an array of names.
    static void addSpeculativeTypeNamesFromValue(const json& value, std::vector<std::string>& typeNames) {
        if (value.is_string()) {
            addSpeculativeTypeName(typeNames, value.get<std::string>());
            return;
        }
        if (value.is_array()) {
            for (const auto& item : value) {
                if (item.is_string()) {
                    addSpeculativeTypeName(typeNames, item.get<std::string>());
                }
            }
        }
    }

    static void applySpeculativeDraftOptions(const json& obj, common_params_speculative_draft& draft) {
        draft.mparams.path = getPropertyAsString(obj, "model", draft.mparams.path);
        draft.mparams.path = getPropertyAsString(obj, "path", draft.mparams.path);
        draft.mparams.path = getPropertyAsString(obj, "model_draft", draft.mparams.path);
        draft.mparams.path = getPropertyAsString(obj, "draft_model", draft.mparams.path);
        draft.n_max = getPropertyAsInt(obj, "n_max", draft.n_max);
        draft.n_min = getPropertyAsInt(obj, "n_min", draft.n_min);
        draft.p_min = getPropertyAsFloat(obj, "p_min", draft.p_min);
        draft.p_split = getPropertyAsFloat(obj, "p_split", draft.p_split);
        draft.n_gpu_layers = getPropertyAsInt(obj, "n_gpu_layers", draft.n_gpu_layers);

        std::string cacheTypeK = getPropertyAsString(obj, "cache_type_k");
        if (!cacheTypeK.empty()) {
            draft.cache_type_k = rnllama::kv_cache_type_from_str(cacheTypeK);
        }

        std::string cacheTypeV = getPropertyAsString(obj, "cache_type_v");
        if (!cacheTypeV.empty()) {
            draft.cache_type_v = rnllama::kv_cache_type_from_str(cacheTypeV);
        }
    }

    bool hasSpeculativeType(const common_params_speculative& speculative, common_speculative_type type) {
        return std::find(speculative.types.begin(), speculative.types.end(), type) != speculative.types.end();
    }

    static void applySpeculativeTypeNames(
        common_params_speculative& speculative,
        const std::vector<std::string>& typeNames
    ) {
        if (typeNames.empty()) {
            return;
        }
        speculative.types = common_speculative_types_from_names(typeNames);
    }

    static void applySpeculativeOptions(const json& params, common_params& cparams) {
        std::vector<std::string> typeNames;

        if (const json* specType = findProperty(params, "spec_type")) {
            addSpeculativeTypeNamesFromValue(*specType, typeNames);
        }

        // `speculative` accepts a bool, a type name, or an options object.
        if (const json* value = findProperty(params, "speculative"); value && !value->is_null()) {
            if (value->is_boolean()) {
                addSpeculativeTypeName(typeNames, value->get<bool>() ? "draft-mtp" : "none");
            } else if (value->is_string()) {
                addSpeculativeTypeName(typeNames, value->get<std::string>());
            } else if (value->is_object()) {
                const json& speculative = *value;
                bool enabled = false;
                bool hasEnabled = false;
                bool hasExplicitType = false;

                if (const json* enabledValue = findProperty(speculative, "enabled"); enabledValue && enabledValue->is_boolean()) {
                    enabled = enabledValue->get<bool>();
                    hasEnabled = true;
                }

                if (const json* type = findProperty(speculative, "type")) {
                    const size_t oldSize = typeNames.size();
                    addSpeculativeTypeNamesFromValue(*type, typeNames);
                    hasExplicitType = hasExplicitType || typeNames.size() != oldSize;
                }

                if (const json* types = findProperty(speculative, "types")) {
                    const size_t oldSize = typeNames.size();
                    addSpeculativeTypeNamesFromValue(*types, typeNames);
                    hasExplicitType = hasExplicitType || typeNames.size() != oldSize;
                }

                if (hasEnabled) {
                    if (!enabled) {
                        addSpeculativeTypeName(typeNames, "none");
                    } else if (!hasExplicitType) {
                        addSpeculativeTypeName(typeNames, "draft-mtp");
                    }
                }

                applySpeculativeDraftOptions(speculative, cparams.speculative.draft);
                if (const json* draftValue = findProperty(speculative, "draft"); draftValue && draftValue->is_object()) {
                    applySpeculativeDraftOptions(*draftValue, cparams.speculative.draft);
                }
            }
        }

        cparams.speculative.draft.n_max = getPropertyAsInt(
            params, "spec_draft_n_max", cparams.speculative.draft.n_max);
        cparams.speculative.draft.n_max = getPropertyAsInt(
            params, "speculative.n_max", cparams.speculative.draft.n_max);
        cparams.speculative.draft.n_min = getPropertyAsInt(
            params, "spec_draft_n_min", cparams.speculative.draft.n_min);
        cparams.speculative.draft.n_min = getPropertyAsInt(
            params, "speculative.n_min", cparams.speculative.draft.n_min);
        cparams.speculative.draft.p_min = getPropertyAsFloat(
            params, "spec_draft_p_min", cparams.speculative.draft.p_min);
        cparams.speculative.draft.p_min = getPropertyAsFloat(
            params, "speculative.p_min", cparams.speculative.draft.p_min);
        cparams.speculative.draft.p_split = getPropertyAsFloat(
            params, "spec_draft_p_split", cparams.speculative.draft.p_split);
        cparams.speculative.draft.p_split = getPropertyAsFloat(
            params, "speculative.p_split", cparams.speculative.draft.p_split);
        cparams.speculative.draft.mparams.path = getPropertyAsString(
            params, "model_draft", cparams.speculative.draft.mparams.path);
        cparams.speculative.draft.mparams.path = getPropertyAsString(
            params, "draft_model", cparams.speculative.draft.mparams.path);
        cparams.speculative.draft.n_gpu_layers = getPropertyAsInt(
            params, "spec_draft_n_gpu_layers", cparams.speculative.draft.n_gpu_layers);

        std::string draftCacheTypeK = getPropertyAsString(params, "spec_draft_cache_type_k");
        if (!draftCacheTypeK.empty()) {
            cparams.speculative.draft.cache_type_k = rnllama::kv_cache_type_from_str(draftCacheTypeK);
        }

        std::string draftCacheTypeV = getPropertyAsString(params, "spec_draft_cache_type_v");
        if (!draftCacheTypeV.empty()) {
            cparams.speculative.draft.cache_type_v = rnllama::kv_cache_type_from_str(draftCacheTypeV);
        }

        applySpeculativeTypeNames(cparams.speculative, typeNames);

        if (hasSpeculativeType(cparams.speculative, COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
            cparams.speculative.draft.n_max <= 0) {
            throw std::invalid_argument("MTP requires spec_draft_n_max > 0");
        }
    }

    // ---- context params ---------------------------------------------------

    void parseCommonParams(const json& params, common_params& cparams) {
        cparams.fit_params = false;

        // Model path
        cparams.model.path = getPropertyAsString(params, "model");
        cparams.vocab_only = getPropertyAsBool(params, "vocab_only", false);
        if (cparams.vocab_only) {
            cparams.warmup = false;
        }

        cparams.n_ctx = getPropertyAsInt(params, "n_ctx", cparams.n_ctx);

        // For vocab_only models, ensure n_ctx is set because:
        // 1. vocab_only models have n_ctx_train = 0 (no tensors loaded)
        // 2. Context creation fails if both n_ctx and n_ctx_train are 0
        // Use 512 as a minimal default - sufficient for tokenization
        if (cparams.vocab_only && cparams.n_ctx == 0) {
            cparams.n_ctx = 512;
        }
        cparams.n_batch = getPropertyAsInt(params, "n_batch", cparams.n_batch);
        cparams.n_ubatch = getPropertyAsInt(params, "n_ubatch", cparams.n_ubatch);
        cparams.n_parallel = getPropertyAsInt(params, "n_parallel", cparams.n_parallel);
        cparams.cpuparams.n_threads = getPropertyAsInt(params, "n_threads", cparams.cpuparams.n_threads);
        std::string cpuMask = getPropertyAsString(params, "cpu_mask");
#if defined(__ANDROID__)
        set_best_cores(cparams.cpuparams, cparams.cpuparams.n_threads);
#elif defined(__APPLE__)
        if (cparams.cpuparams.n_threads < 0) {
            cparams.cpuparams.n_threads = default_apple_n_threads();
        }
#endif

        cparams.n_gpu_layers = getPropertyAsInt(params, "n_gpu_layers", cparams.n_gpu_layers);
        if (!cpuMask.empty()) {
            bool cpumask[LM_GGML_MAX_N_THREADS] = {false};
            if (parse_cpu_mask(cpuMask, cpumask)) {
                std::copy(std::begin(cpumask), std::end(cpumask), std::begin(cparams.cpuparams.cpumask));
                cparams.cpuparams.mask_valid = true;
            }
        }
        cparams.cpuparams.strict_cpu = getPropertyAsBool(params, "cpu_strict", cparams.cpuparams.strict_cpu);

        // Chat template
        std::string chatTemplate = getPropertyAsString(params, "chat_template");
        if (!chatTemplate.empty()) {
            cparams.chat_template = chatTemplate;
        }

        bool useMmap = cparams.load_mode == LLAMA_LOAD_MODE_MMAP ||
            cparams.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        bool useMlock = cparams.load_mode == LLAMA_LOAD_MODE_MLOCK ||
            cparams.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
        useMlock = getPropertyAsBool(params, "use_mlock", useMlock);
        useMmap = getPropertyAsBool(params, "use_mmap", useMmap);
        if (useMmap && useMlock) {
            cparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
        } else if (useMmap) {
            cparams.load_mode = LLAMA_LOAD_MODE_MMAP;
        } else if (useMlock) {
            cparams.load_mode = LLAMA_LOAD_MODE_MLOCK;
        } else {
            cparams.load_mode = LLAMA_LOAD_MODE_NONE;
        }
        cparams.no_extra_bufts = getPropertyAsBool(params, "no_extra_bufts", cparams.no_extra_bufts);

        if (hasProperty(params, "flash_attn")) {
            bool fa = getPropertyAsBool(params, "flash_attn", false);
            cparams.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        if (hasProperty(params, "flash_attn_type")) {
            std::string fa = getPropertyAsString(params, "flash_attn_type");
            cparams.flash_attn_type = static_cast<enum llama_flash_attn_type>(rnllama::flash_attn_type_from_str(fa));
        }

        std::string ck = getPropertyAsString(params, "cache_type_k");
        if (!ck.empty()) cparams.cache_type_k = rnllama::kv_cache_type_from_str(ck);

        std::string cv = getPropertyAsString(params, "cache_type_v");
        if (!cv.empty()) cparams.cache_type_v = rnllama::kv_cache_type_from_str(cv);

        cparams.ctx_shift = getPropertyAsBool(params, "ctx_shift", cparams.ctx_shift);
        cparams.kv_unified = getPropertyAsBool(params, "kv_unified", cparams.kv_unified);
        cparams.swa_full = getPropertyAsBool(params, "swa_full", cparams.swa_full);

        if (getPropertyAsBool(params, "embedding", false)) {
            cparams.embedding = true;
            cparams.n_ubatch = cparams.n_batch; // Default for non-causal
            cparams.embd_normalize = getPropertyAsInt(params, "embd_normalize", cparams.embd_normalize);
        }

        int pooling_type = getPropertyAsInt(params, "pooling_type", -1);
        if (pooling_type >= 0) {
            cparams.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
        }

        cparams.rope_freq_base = getPropertyAsFloat(params, "rope_freq_base", cparams.rope_freq_base);
        cparams.rope_freq_scale = getPropertyAsFloat(params, "rope_freq_scale", cparams.rope_freq_scale);

        int n_cpu_moe = getPropertyAsInt(params, "n_cpu_moe", 0);
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
        std::string loraPath = getPropertyAsString(params, "lora");
        if (!loraPath.empty()) {
            common_adapter_lora_info la;
            la.path = loraPath;
            la.scale = getPropertyAsFloat(params, "lora_scaled", 1.0f);
            cparams.lora_adapters.push_back(la);
        }

        if (const json* loraList = findProperty(params, "lora_list")) {
            for (auto& la : parseLoraAdapters(*loraList)) {
                cparams.lora_adapters.push_back(la);
            }
        }

        applySpeculativeOptions(params, cparams);
    }

    // ---- completion params ------------------------------------------------

    void parseCompletionParams(const json& params, rnllama::llama_rn_context* ctx) {
        if (!ctx) return;

        ctx->params.prompt = getPropertyAsString(params, "prompt");

        auto& sparams = ctx->params.sampling;
        sparams.seed = getPropertyAsInt(params, "seed", -1);
        ctx->params.n_predict = getPropertyAsInt(params, "n_predict", ctx->params.n_predict);
        ctx->params.sampling.ignore_eos = getPropertyAsBool(params, "ignore_eos", ctx->params.sampling.ignore_eos);
        ctx->params.embedding = getPropertyAsBool(params, "embedding", false);
        llama_set_embeddings(ctx->ctx, ctx->params.embedding);
        applySpeculativeOptions(params, ctx->params);

        sparams.temp = getPropertyAsDouble(params, "temperature", sparams.temp);
        sparams.n_probs = getPropertyAsInt(params, "n_probs", sparams.n_probs);

        sparams.penalty_last_n = getPropertyAsInt(params, "penalty_last_n", sparams.penalty_last_n);
        sparams.penalty_repeat = getPropertyAsDouble(params, "penalty_repeat", sparams.penalty_repeat);
        sparams.penalty_freq = getPropertyAsDouble(params, "penalty_freq", sparams.penalty_freq);
        sparams.penalty_present = getPropertyAsDouble(params, "penalty_present", sparams.penalty_present);

        sparams.mirostat = getPropertyAsInt(params, "mirostat", sparams.mirostat);
        sparams.mirostat_tau = getPropertyAsDouble(params, "mirostat_tau", sparams.mirostat_tau);
        sparams.mirostat_eta = getPropertyAsDouble(params, "mirostat_eta", sparams.mirostat_eta);

        sparams.top_k = getPropertyAsInt(params, "top_k", sparams.top_k);
        sparams.top_p = getPropertyAsDouble(params, "top_p", sparams.top_p);
        sparams.min_p = getPropertyAsDouble(params, "min_p", sparams.min_p);

        sparams.xtc_threshold = getPropertyAsDouble(params, "xtc_threshold", sparams.xtc_threshold);
        sparams.xtc_probability = getPropertyAsDouble(params, "xtc_probability", sparams.xtc_probability);
        sparams.typ_p = getPropertyAsDouble(params, "typical_p", sparams.typ_p);

        sparams.dry_multiplier = getPropertyAsDouble(params, "dry_multiplier", sparams.dry_multiplier);
        sparams.dry_base = getPropertyAsDouble(params, "dry_base", sparams.dry_base);
        sparams.dry_allowed_length = getPropertyAsInt(params, "dry_allowed_length", sparams.dry_allowed_length);
        sparams.dry_penalty_last_n = getPropertyAsInt(params, "dry_penalty_last_n", sparams.dry_penalty_last_n);
        if (const json* breakers = findProperty(params, "dry_sequence_breakers"); breakers && breakers->is_array()) {
            sparams.dry_sequence_breakers.clear();
            for (const auto& breaker : *breakers) {
                if (breaker.is_string()) {
                    sparams.dry_sequence_breakers.push_back(breaker.get<std::string>());
                }
            }
        }

        sparams.top_n_sigma = getPropertyAsDouble(params, "top_n_sigma", sparams.top_n_sigma);

        // Grammar
        sparams.grammar = {};
        sparams.generation_prompt.clear();
        sparams.grammar_triggers.clear();
        sparams.preserved_tokens.clear();
        sparams.reasoning_budget_tokens = -1;
        sparams.reasoning_budget_activate_immediately = false;
        sparams.reasoning_budget_start.clear();
        sparams.reasoning_budget_end.clear();
        sparams.reasoning_budget_forced.clear();

        std::string grammar = getPropertyAsString(params, "grammar");
        if (!grammar.empty()) {
            sparams.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar)};
        }

        std::string jsonSchema = getPropertyAsString(params, "json_schema");
        if (!jsonSchema.empty() && sparams.grammar.empty()) {
#if defined(RNLLAMA_HAS_COMMON_JSON)
            sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(common_json::parse(jsonSchema))};
#else
            sparams.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(nlohmann::ordered_json::parse(jsonSchema))};
#endif
        }

        sparams.generation_prompt = getPropertyAsString(params, "generation_prompt");

        const int thinkingBudgetTokens = getPropertyAsInt(params, "thinking_budget_tokens", -1);
        if (thinkingBudgetTokens >= 0) {
            const std::string thinkingEndTag = getPropertyAsString(params, "thinking_end_tag");
            if (!thinkingEndTag.empty()) {
                const std::string thinkingStartTag = getPropertyAsString(params, "thinking_start_tag");
                const std::string thinkingBudgetMessage = getPropertyAsString(params, "thinking_budget_message");

                if (!thinkingStartTag.empty()) {
                    sparams.reasoning_budget_start = common_tokenize(
                        ctx->ctx, thinkingStartTag, /* add_special= */ false, /* parse_special= */ true);
                }
                auto reasoningBudgetEnd = common_tokenize(
                    ctx->ctx, thinkingEndTag, /* add_special= */ false, /* parse_special= */ true);
                if (!reasoningBudgetEnd.empty()) {
                    sparams.reasoning_budget_end.push_back(std::move(reasoningBudgetEnd));
                }
                sparams.reasoning_budget_forced = common_tokenize(
                    ctx->ctx, thinkingBudgetMessage + thinkingEndTag, /* add_special= */ false, /* parse_special= */ true);

                if (!sparams.reasoning_budget_end.empty() && !sparams.reasoning_budget_forced.empty()) {
                    sparams.reasoning_budget_tokens = thinkingBudgetTokens;
                    sparams.reasoning_budget_activate_immediately = getPropertyAsBool(
                        params, "thinking_forced_open", false);
                } else {
                    sparams.reasoning_budget_start.clear();
                    sparams.reasoning_budget_end.clear();
                    sparams.reasoning_budget_forced.clear();
                }
            }
        }

        sparams.grammar_lazy = getPropertyAsBool(params, "grammar_lazy", false);

        if (const json* preserved = findProperty(params, "preserved_tokens"); preserved && preserved->is_array()) {
            for (const auto& token : *preserved) {
                if (!token.is_string()) {
                    continue;
                }
                auto ids = common_tokenize(ctx->ctx, token.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    sparams.preserved_tokens.insert(ids[0]);
                }
            }
        }

        if (const json* triggers = findProperty(params, "grammar_triggers"); triggers && triggers->is_array()) {
            for (const auto& triggerObj : *triggers) {
                if (!triggerObj.is_object()) {
                    continue;
                }
                auto type = static_cast<common_grammar_trigger_type>(getPropertyAsInt(triggerObj, "type", 0));
                std::string word = getPropertyAsString(triggerObj, "value");
                if (word.empty()) {
                    continue;
                }

                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    auto ids = common_tokenize(ctx->ctx, word, /* add_special= */ false, /* parse_special= */ true);
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
                        trigger.token = (llama_token) getPropertyAsInt(triggerObj, "token", 0);
                    }
                    sparams.grammar_triggers.push_back(std::move(trigger));
                }
            }
        }

        // Logit bias: [[token, bias | false], ...]
        sparams.logit_bias.clear();
        const llama_model * model = llama_get_model(ctx->ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        if (ctx->params.sampling.ignore_eos) {
            sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
        }

        if (const json* logitBias = findProperty(params, "logit_bias"); logitBias && logitBias->is_array()) {
            for (const auto& el : *logitBias) {
                if (!el.is_array() || el.size() != 2 || !el[0].is_number()) {
                    continue;
                }
                int tok = (int) el[0].get<double>();
                const json& val = el[1];
                if (val.is_number()) {
                    sparams.logit_bias[tok].bias = val.get<double>();
                } else if (val.is_boolean() && !val.get<bool>()) {
                    sparams.logit_bias[tok].bias = -INFINITY;
                }
            }
        }

        ctx->params.antiprompt.clear();
        if (const json* stop = findProperty(params, "stop"); stop && stop->is_array()) {
            for (const auto& word : *stop) {
                if (word.is_string()) {
                    ctx->params.antiprompt.push_back(word.get<std::string>());
                }
            }
        }

        if (hasProperty(params, "n_threads")) {
            int nThreads = getPropertyAsInt(params, "n_threads", ctx->params.cpuparams.n_threads);
#if defined(__ANDROID__)
            set_best_cores(ctx->params.cpuparams, nThreads);
#else
            const int maxThreads = (int) std::thread::hardware_concurrency();
            const int defaultNThreads = nThreads == 4 ? 2 : (maxThreads > 0 ? std::min(4, maxThreads) : 4);
            ctx->params.cpuparams.n_threads = nThreads > 0 ? nThreads : defaultNThreads;
#endif
        }

        // TTS speaker id: thread from completion options into tts_wrapper so
        // rn-completion.cpp can resolve the speaker via getSpeaker(pending_speaker_id).
        // Default -1 means no speaker override.
        if (ctx->tts_wrapper != nullptr) {
            ctx->tts_wrapper->pending_speaker_id = getPropertyAsInt(params, "speakerId", -1);
        }
    }
}
