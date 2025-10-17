#include <jni.h>
// #include <android/asset_manager.h>
// #include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <sys/sysinfo.h>
#include <string>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <list>
#include <nlohmann/json.hpp>
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "llama-impl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "rn-llama.h"
#include "rn-completion.h"
#include "rn-slot-manager.h"
#include "jni-utils.h"
#define UNUSED(x) (void)(x)
#define TAG "RNLLAMA_ANDROID_JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,    TAG, __VA_ARGS__)
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

static void rnllama_log_callback_default(lm_ggml_log_level level, const char * fmt, void * data) {
    if (level == LM_GGML_LOG_LEVEL_ERROR)     __android_log_print(ANDROID_LOG_ERROR, TAG, fmt, data);
    else if (level == LM_GGML_LOG_LEVEL_INFO) __android_log_print(ANDROID_LOG_INFO, TAG, fmt, data);
    else if (level == LM_GGML_LOG_LEVEL_WARN) __android_log_print(ANDROID_LOG_WARN, TAG, fmt, data);
    else __android_log_print(ANDROID_LOG_DEFAULT, TAG, fmt, data);
}

extern "C" {

// sets cpu mask to use best performing cors
void set_best_cores(struct cpu_params &params, int n) {
    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);
    params.n_threads = n > 0 && n <= max_threads ? n : default_n_threads;

    std::vector<std::pair<int,int>> cores; // {freq, id}

    for (int i = 0; i < max_threads; i++) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/cpuinfo_max_freq");
        int freq;
        if (f >> freq) {
            cores.emplace_back(freq, i);
        }
    }

    std::sort(cores.rbegin(), cores.rend());
    std::fill(std::begin(params.cpumask), std::end(params.cpumask), false);

    for (int i = 0; i < n && i < (int)cores.size(); i++) {
        LOGI("Using core %d with frequency %d", cores[i].second, cores[i].first);
        params.cpumask[cores[i].second] = true;
    }
    params.strict_cpu = true;
    params.mask_valid = true;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_modelInfo(
    JNIEnv *env,
    jobject thiz,
    jstring model_path_str,
    jobjectArray skip
) {
    UNUSED(thiz);

    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);

    std::vector<std::string> skip_vec;
    int skip_len = env->GetArrayLength(skip);
    for (int i = 0; i < skip_len; i++) {
        jstring skip_str = (jstring) env->GetObjectArrayElement(skip, i);
        const char *skip_chars = env->GetStringUTFChars(skip_str, nullptr);
        skip_vec.push_back(skip_chars);
        env->ReleaseStringUTFChars(skip_str, skip_chars);
    }

    struct lm_gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };
    struct lm_gguf_context * ctx = lm_gguf_init_from_file(model_path_chars, params);

    if (!ctx) {
        LOGI("%s: failed to load '%s'\n", __func__, model_path_chars);
        return nullptr;
    }

    auto info = writablemap::createWriteableMap(env);
    writablemap::putInt(env, info, "version", lm_gguf_get_version(ctx));
    writablemap::putInt(env, info, "alignment", lm_gguf_get_alignment(ctx));
    writablemap::putInt(env, info, "data_offset", lm_gguf_get_data_offset(ctx));
    {
        const int n_kv = lm_gguf_get_n_kv(ctx);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = lm_gguf_get_key(ctx, i);

            bool skipped = false;
            if (skip_len > 0) {
                for (int j = 0; j < skip_len; j++) {
                    if (skip_vec[j] == key) {
                        skipped = true;
                        break;
                    }
                }
            }

            if (skipped) {
                continue;
            }

            const std::string value = lm_gguf_kv_to_str(ctx, i);
            writablemap::putString(env, info, key, value.c_str());
        }
    }

    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    lm_gguf_free(ctx);

    return reinterpret_cast<jobject>(info);
}

JNIEXPORT jstring JNICALL
Java_com_rnllama_LlamaContext_getBackendDevicesInfo(
    JNIEnv *env,
    jobject thiz
) {
    UNUSED(thiz);
    std::string devices_json_str = rnllama::get_backend_devices_info();
    return env->NewStringUTF(devices_json_str.c_str());
}

struct callback_context {
    JNIEnv *env;
    rnllama::llama_rn_context *llama;
    jobject callback;
};

std::unordered_map<long, rnllama::llama_rn_context *> context_map;

// Utility function to extract media_paths array from ReadableMap
std::vector<std::string> extract_media_paths(JNIEnv *env, jobject params) {
    std::vector<std::string> media_paths_vec;
    if (readablemap::hasKey(env, params, "media_paths")) {
        jobject media_paths = readablemap::getArray(env, params, "media_paths");

        if (media_paths != nullptr) {
            int size = readablearray::size(env, media_paths);
            for (int i = 0; i < size; i++) {
                jstring path_str = readablearray::getString(env, media_paths, i);
                const char *path_chars = env->GetStringUTFChars(path_str, nullptr);
                media_paths_vec.push_back(path_chars);
                env->ReleaseStringUTFChars(path_str, path_chars);
            }
        }
    }
    return media_paths_vec;
}

// Utility function to extract common sampling parameters from ReadableMap
void extract_sampling_params(
    JNIEnv *env,
    jobject params,
    rnllama::llama_rn_context* llama,
    common_params_sampling& sparams,
    common_params& cparams
) {
    // Basic sampling parameters
    if (readablemap::hasKey(env, params, "n_predict")) {
        cparams.n_predict = readablemap::getInt(env, params, "n_predict", -1);
    }
    if (readablemap::hasKey(env, params, "temperature")) {
        sparams.temp = readablemap::getFloat(env, params, "temperature", 0.7f);
    }
    if (readablemap::hasKey(env, params, "top_k")) {
        sparams.top_k = readablemap::getInt(env, params, "top_k", 40);
    }
    if (readablemap::hasKey(env, params, "top_p")) {
        sparams.top_p = readablemap::getFloat(env, params, "top_p", 0.95f);
    }
    if (readablemap::hasKey(env, params, "min_p")) {
        sparams.min_p = readablemap::getFloat(env, params, "min_p", 0.05f);
    }
    if (readablemap::hasKey(env, params, "xtc_threshold")) {
        sparams.xtc_threshold = readablemap::getFloat(env, params, "xtc_threshold", 0.1f);
    }
    if (readablemap::hasKey(env, params, "xtc_probability")) {
        sparams.xtc_probability = readablemap::getFloat(env, params, "xtc_probability", 0.0f);
    }
    if (readablemap::hasKey(env, params, "typical_p")) {
        sparams.typ_p = readablemap::getFloat(env, params, "typical_p", 1.0f);
    }

    // Penalty parameters
    if (readablemap::hasKey(env, params, "penalty_last_n")) {
        sparams.penalty_last_n = readablemap::getInt(env, params, "penalty_last_n", 64);
    }
    if (readablemap::hasKey(env, params, "penalty_repeat")) {
        sparams.penalty_repeat = readablemap::getFloat(env, params, "penalty_repeat", 1.0f);
    }
    if (readablemap::hasKey(env, params, "penalty_freq")) {
        sparams.penalty_freq = readablemap::getFloat(env, params, "penalty_freq", 0.0f);
    }
    if (readablemap::hasKey(env, params, "penalty_present")) {
        sparams.penalty_present = readablemap::getFloat(env, params, "penalty_present", 0.0f);
    }

    // Mirostat parameters
    if (readablemap::hasKey(env, params, "mirostat")) {
        sparams.mirostat = readablemap::getInt(env, params, "mirostat", 0);
    }
    if (readablemap::hasKey(env, params, "mirostat_tau")) {
        sparams.mirostat_tau = readablemap::getFloat(env, params, "mirostat_tau", 5.0f);
    }
    if (readablemap::hasKey(env, params, "mirostat_eta")) {
        sparams.mirostat_eta = readablemap::getFloat(env, params, "mirostat_eta", 0.1f);
    }

    // DRY parameters
    if (readablemap::hasKey(env, params, "dry_multiplier")) {
        sparams.dry_multiplier = readablemap::getFloat(env, params, "dry_multiplier", 0.0f);
    }
    if (readablemap::hasKey(env, params, "dry_base")) {
        sparams.dry_base = readablemap::getFloat(env, params, "dry_base", 1.75f);
    }
    if (readablemap::hasKey(env, params, "dry_allowed_length")) {
        sparams.dry_allowed_length = readablemap::getInt(env, params, "dry_allowed_length", 2);
    }
    if (readablemap::hasKey(env, params, "dry_penalty_last_n")) {
        sparams.dry_penalty_last_n = readablemap::getInt(env, params, "dry_penalty_last_n", -1);
    }

    // DRY sequence breakers
    if (readablemap::hasKey(env, params, "dry_sequence_breakers")) {
        jobject dry_sequence_breakers = readablemap::getArray(env, params, "dry_sequence_breakers");

        sparams.dry_sequence_breakers.clear();
        int size = readablearray::size(env, dry_sequence_breakers);
        for (int i = 0; i < size; i++) {
            jstring str = readablearray::getString(env, dry_sequence_breakers, i);
            const char *chars = env->GetStringUTFChars(str, nullptr);
            sparams.dry_sequence_breakers.push_back(chars);
            env->ReleaseStringUTFChars(str, chars);
        }
    }

    // Other parameters
    if (readablemap::hasKey(env, params, "top_n_sigma")) {
        sparams.top_n_sigma = readablemap::getFloat(env, params, "top_n_sigma", -1.0f);
    }
    if (readablemap::hasKey(env, params, "n_probs")) {
        sparams.n_probs = readablemap::getInt(env, params, "n_probs", 0);
    }
    if (readablemap::hasKey(env, params, "seed")) {
        jint seed = readablemap::getInt(env, params, "seed", -1);
        sparams.seed = (seed == -1) ? time(NULL) : seed;
    }
    if (readablemap::hasKey(env, params, "ignore_eos")) {
        sparams.ignore_eos = readablemap::getBool(env, params, "ignore_eos", false);
    }

    // Grammar parameters
    if (readablemap::hasKey(env, params, "grammar")) {
        jstring grammar = readablemap::getString(env, params, "grammar", nullptr);
        const char *grammar_chars = env->GetStringUTFChars(grammar, nullptr);
        if (grammar_chars && grammar_chars[0] != '\0') {
            sparams.grammar = grammar_chars;
        }
        env->ReleaseStringUTFChars(grammar, grammar_chars);
    }

    if (readablemap::hasKey(env, params, "grammar_lazy")) {
        sparams.grammar_lazy = readablemap::getBool(env, params, "grammar_lazy", false);
    }

    // JSON schema to grammar conversion
    if (readablemap::hasKey(env, params, "json_schema") && !readablemap::hasKey(env, params, "grammar")) {
        jstring json_schema = readablemap::getString(env, params, "json_schema", nullptr);
        const char *json_schema_chars = env->GetStringUTFChars(json_schema, nullptr);
        if (json_schema_chars && json_schema_chars[0] != '\0') {
            auto schema = json::parse(json_schema_chars);
            sparams.grammar = json_schema_to_grammar(schema);
        }
        env->ReleaseStringUTFChars(json_schema, json_schema_chars);
    }

    // Stop sequences
    if (readablemap::hasKey(env, params, "stop")) {
        jobject stop_array = readablemap::getArray(env, params, "stop");

        cparams.antiprompt.clear();
        int size = readablearray::size(env, stop_array);
        for (int i = 0; i < size; i++) {
            jstring str = readablearray::getString(env, stop_array, i);
            const char *chars = env->GetStringUTFChars(str, nullptr);
            cparams.antiprompt.push_back(chars);
            env->ReleaseStringUTFChars(str, chars);
        }
    }

    // Logit bias
    if (readablemap::hasKey(env, params, "logit_bias")) {
        jobject logit_bias_array = readablemap::getArray(env, params, "logit_bias");

        int size = readablearray::size(env, logit_bias_array);
        for (int i = 0; i < size; i++) {
            jobject row_array = readablearray::getMap(env, logit_bias_array, i);
            jclass arrayClass = env->GetObjectClass(row_array);
            jmethodID getDoubleMethod = env->GetMethodID(arrayClass, "getDouble", "(I)D");

            jint tok = (jint) env->CallDoubleMethod(row_array, getDoubleMethod, 0);
            jdouble bias_d = env->CallDoubleMethod(row_array, getDoubleMethod, 1);

            if (std::isfinite(bias_d)) {
                sparams.logit_bias[tok].bias = bias_d;
            } else {
                sparams.logit_bias[tok].bias = -INFINITY;
            }
        }
    }

    // Ignore EOS logit bias
    if (readablemap::hasKey(env, params, "ignore_eos") && readablemap::getBool(env, params, "ignore_eos", false)) {
        const struct llama_vocab *vocab = llama_model_get_vocab(llama->model);
        sparams.logit_bias[llama_vocab_eos(vocab)].bias = -INFINITY;
    }

    // Set preserved_tokens if provided
    if (readablemap::hasKey(env, params, "preserved_tokens")) {
        jobject preserved_tokens = readablemap::getArray(env, params, "preserved_tokens");

        int preserved_tokens_size = readablearray::size(env, preserved_tokens);
        for (int i = 0; i < preserved_tokens_size; i++) {
            jstring preserved_token = readablearray::getString(env, preserved_tokens, i);
            auto ids = common_tokenize(llama->ctx, env->GetStringUTFChars(preserved_token, nullptr), /* add_special= */ false, /* parse_special= */ true);
            if (ids.size() == 1) {
                sparams.preserved_tokens.insert(ids[0]);
            }
        }
    }

    // Set grammar_triggers if provided
    if (readablemap::hasKey(env, params, "grammar_triggers")) {
        jobject grammar_triggers = readablemap::getArray(env, params, "grammar_triggers");

        int grammar_triggers_size = readablearray::size(env, grammar_triggers);
        for (int i = 0; i < grammar_triggers_size; i++) {
            auto trigger_map = readablearray::getMap(env, grammar_triggers, i);
            const auto type = static_cast<common_grammar_trigger_type>(readablemap::getInt(env, trigger_map, "type", 0));
            jstring trigger_word = readablemap::getString(env, trigger_map, "value", nullptr);
            const char *trigger_word_chars = env->GetStringUTFChars(trigger_word, nullptr);
            const std::string word = trigger_word_chars;
            env->ReleaseStringUTFChars(trigger_word, trigger_word_chars);

            if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                // Try to tokenize - if single token, use TOKEN type, otherwise WORD type
                auto ids = common_tokenize(llama->ctx, word.c_str(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    common_grammar_trigger trigger;
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    trigger.value = word;
                    trigger.token = ids[0];
                    sparams.grammar_triggers.push_back(std::move(trigger));
                } else {
                    sparams.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                }
            } else {
                // TOKEN, PATTERN, or PATTERN_FULL
                common_grammar_trigger trigger;
                trigger.type = type;
                trigger.value = word;
                if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
                    // Extract token if provided
                    trigger.token = (llama_token) readablemap::getInt(env, trigger_map, "token", 0);
                }
                sparams.grammar_triggers.push_back(std::move(trigger));
            }
        }
    }
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_initContext(
    JNIEnv *env,
    jobject thiz,
    jobject params_map,
    jobject load_progress_callback
) {
    UNUSED(thiz);

    // Extract required model parameter
    jstring model_path_str = readablemap::getString(env, params_map, "model", nullptr);
    if (!model_path_str) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "_error", "Missing required parameter: model");
        return result;
    }

    common_params defaultParams;

    // Set model path
    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    defaultParams.model.path = model_path_chars;

    // Extract optional chat_template
    jstring chat_template = nullptr;
    const char *chat_template_chars = "";
    if (readablemap::hasKey(env, params_map, "chat_template")) {
        chat_template = readablemap::getString(env, params_map, "chat_template", nullptr);
        if (chat_template) {
            chat_template_chars = env->GetStringUTFChars(chat_template, nullptr);
            defaultParams.chat_template = chat_template_chars;
        }
    }

    // Extract vocab_only (affects warmup setting)
    bool vocab_only = false;
    if (readablemap::hasKey(env, params_map, "vocab_only")) {
        vocab_only = readablemap::getBool(env, params_map, "vocab_only", false);
        defaultParams.vocab_only = vocab_only;
        if (vocab_only) {
            defaultParams.warmup = false;
        }
    }

    // Extract context parameters
    if (readablemap::hasKey(env, params_map, "n_ctx")) {
        defaultParams.n_ctx = readablemap::getInt(env, params_map, "n_ctx", 512);
    }
    if (readablemap::hasKey(env, params_map, "n_batch")) {
        defaultParams.n_batch = readablemap::getInt(env, params_map, "n_batch", 512);
    }
    if (readablemap::hasKey(env, params_map, "n_ubatch")) {
        defaultParams.n_ubatch = readablemap::getInt(env, params_map, "n_ubatch", 512);
    }
    if (readablemap::hasKey(env, params_map, "n_parallel")) {
        jint n_parallel = readablemap::getInt(env, params_map, "n_parallel", 0);
        if (n_parallel > 0) {
            defaultParams.n_parallel = n_parallel;
        }
    }
    if (readablemap::hasKey(env, params_map, "ctx_shift")) {
        defaultParams.ctx_shift = readablemap::getBool(env, params_map, "ctx_shift", true);
    }
    if (readablemap::hasKey(env, params_map, "kv_unified")) {
        defaultParams.kv_unified = readablemap::getBool(env, params_map, "kv_unified", false);
    }
    if (readablemap::hasKey(env, params_map, "swa_full")) {
        defaultParams.swa_full = readablemap::getBool(env, params_map, "swa_full", false);
    }

    // Handle n_cpu_moe parameter
    if (readablemap::hasKey(env, params_map, "n_cpu_moe")) {
        jint n_cpu_moe = readablemap::getInt(env, params_map, "n_cpu_moe", 0);
        if (n_cpu_moe > 0) {
            for (int i = 0; i < n_cpu_moe; ++i) {
                static std::list<std::string> buft_overrides;
                std::string pattern = "blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate)_exps";
                buft_overrides.push_back(pattern);
                defaultParams.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), lm_ggml_backend_cpu_buffer_type()});
            }
            defaultParams.tensor_buft_overrides.push_back({nullptr, nullptr});
        }
    }

    // Extract pooling_type
    if (readablemap::hasKey(env, params_map, "pooling_type")) {
        jint pooling_type = readablemap::getInt(env, params_map, "pooling_type", -1);
        if (pooling_type != -1) {
            defaultParams.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
        }
    }

    // Extract embedding parameters
    bool embedding = false;
    if (readablemap::hasKey(env, params_map, "embedding")) {
        embedding = readablemap::getBool(env, params_map, "embedding", false);
        defaultParams.embedding = embedding;
    }
    if (readablemap::hasKey(env, params_map, "embd_normalize")) {
        jint embd_normalize = readablemap::getInt(env, params_map, "embd_normalize", -1);
        if (embd_normalize != -1) {
            defaultParams.embd_normalize = embd_normalize;
        }
    }
    if (embedding) {
        // For non-causal models, batch size must be equal to ubatch size
        defaultParams.n_ubatch = defaultParams.n_batch;
    }

    jint n_threads = defaultParams.cpuparams.n_threads;
    if (readablemap::hasKey(env, params_map, "n_threads")) {
        n_threads = readablemap::getInt(env, params_map, "n_threads", n_threads);
    }
    set_best_cores(defaultParams.cpuparams, n_threads);

    // Extract GPU parameters
    if (readablemap::hasKey(env, params_map, "n_gpu_layers")) {
        defaultParams.n_gpu_layers = readablemap::getInt(env, params_map, "n_gpu_layers", 0);
    }

    // Extract flash attention parameters
    jstring flash_attn_type_str = nullptr;
    const char *flash_attn_type_chars = nullptr;
    if (readablemap::hasKey(env, params_map, "flash_attn_type")) {
        flash_attn_type_str = readablemap::getString(env, params_map, "flash_attn_type", nullptr);
        if (flash_attn_type_str) {
            flash_attn_type_chars = env->GetStringUTFChars(flash_attn_type_str, nullptr);
            if (flash_attn_type_chars && flash_attn_type_chars[0] != '\0') {
                defaultParams.flash_attn_type = static_cast<enum llama_flash_attn_type>(rnllama::flash_attn_type_from_str(flash_attn_type_chars));
            }
        }
    } else if (readablemap::hasKey(env, params_map, "flash_attn")) {
        // DEPRECATED: use flash_attn_type instead
        bool flash_attn = readablemap::getBool(env, params_map, "flash_attn", false);
        defaultParams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    // Extract cache type parameters
    jstring cache_type_k = nullptr;
    const char *cache_type_k_chars = nullptr;
    if (readablemap::hasKey(env, params_map, "cache_type_k")) {
        cache_type_k = readablemap::getString(env, params_map, "cache_type_k", nullptr);
        if (cache_type_k) {
            cache_type_k_chars = env->GetStringUTFChars(cache_type_k, nullptr);
            if (cache_type_k_chars) {
                defaultParams.cache_type_k = rnllama::kv_cache_type_from_str(cache_type_k_chars);
            }
        }
    }

    jstring cache_type_v = nullptr;
    const char *cache_type_v_chars = nullptr;
    if (readablemap::hasKey(env, params_map, "cache_type_v")) {
        cache_type_v = readablemap::getString(env, params_map, "cache_type_v", nullptr);
        if (cache_type_v) {
            cache_type_v_chars = env->GetStringUTFChars(cache_type_v, nullptr);
            if (cache_type_v_chars) {
                defaultParams.cache_type_v = rnllama::kv_cache_type_from_str(cache_type_v_chars);
            }
        }
    }

    // Extract memory parameters
    if (readablemap::hasKey(env, params_map, "use_mlock")) {
        defaultParams.use_mlock = readablemap::getBool(env, params_map, "use_mlock", true);
    }
    if (readablemap::hasKey(env, params_map, "use_mmap")) {
        defaultParams.use_mmap = readablemap::getBool(env, params_map, "use_mmap", true);
    }

    // Extract RoPE parameters
    if (readablemap::hasKey(env, params_map, "rope_freq_base")) {
        defaultParams.rope_freq_base = readablemap::getFloat(env, params_map, "rope_freq_base", 0.0f);
    }
    if (readablemap::hasKey(env, params_map, "rope_freq_scale")) {
        defaultParams.rope_freq_scale = readablemap::getFloat(env, params_map, "rope_freq_scale", 0.0f);
    }

    auto llama = new rnllama::llama_rn_context();
    llama->is_load_interrupted = false;
    llama->loading_progress = 0;

    if (load_progress_callback != nullptr) {
        defaultParams.progress_callback = [](float progress, void * user_data) {
            callback_context *cb_ctx = (callback_context *)user_data;
            JNIEnv *env = cb_ctx->env;
            auto llama = cb_ctx->llama;
            jobject callback = cb_ctx->callback;
            int percentage = (int) (100 * progress);
            if (percentage > llama->loading_progress) {
                llama->loading_progress = percentage;
                jclass callback_class = env->GetObjectClass(callback);
                jmethodID onLoadProgress = env->GetMethodID(callback_class, "onLoadProgress", "(I)V");
                env->CallVoidMethod(callback, onLoadProgress, percentage);
            }
            return !llama->is_load_interrupted;
        };

        callback_context *cb_ctx = new callback_context;
        cb_ctx->env = env;
        cb_ctx->llama = llama;
        cb_ctx->callback = env->NewGlobalRef(load_progress_callback);
        defaultParams.progress_callback_user_data = cb_ctx;
    }

    bool is_model_loaded = llama->loadModel(defaultParams);

    // Cleanup string resources (lora cleaned up later after applyLoraAdapters)
    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    if (chat_template) {
        env->ReleaseStringUTFChars(chat_template, chat_template_chars);
    }
    if (flash_attn_type_str && flash_attn_type_chars) {
        env->ReleaseStringUTFChars(flash_attn_type_str, flash_attn_type_chars);
    }
    if (cache_type_k && cache_type_k_chars) {
        env->ReleaseStringUTFChars(cache_type_k, cache_type_k_chars);
    }
    if (cache_type_v && cache_type_v_chars) {
        env->ReleaseStringUTFChars(cache_type_v, cache_type_v_chars);
    }

    LOGI("[RNLlama] is_model_loaded %s", (is_model_loaded ? "true" : "false"));
    if (is_model_loaded) {
        if (embedding && llama_model_has_encoder(llama->model) && llama_model_has_decoder(llama->model)) {
            LOGI("[RNLlama] computing embeddings in encoder-decoder models is not supported");
            llama_free(llama->ctx);
            context_map.erase((long) llama->ctx);
            delete llama;
            return nullptr;
        }
        context_map[(long) llama->ctx] = llama;
    } else {
        llama_free(llama->ctx);
        delete llama;
        return nullptr;
    }

    // Extract lora parameters
    std::vector<common_adapter_lora_info> lora;
    jstring lora_str = nullptr;
    const char *lora_chars = nullptr;

    if (readablemap::hasKey(env, params_map, "lora")) {
        lora_str = readablemap::getString(env, params_map, "lora", nullptr);
        if (lora_str) {
            lora_chars = env->GetStringUTFChars(lora_str, nullptr);
            if (lora_chars != nullptr && lora_chars[0] != '\0') {
                common_adapter_lora_info la;
                la.path = lora_chars;
                // Get lora_scaled if provided
                if (readablemap::hasKey(env, params_map, "lora_scaled")) {
                    la.scale = readablemap::getFloat(env, params_map, "lora_scaled", 1.0f);
                } else {
                    la.scale = 1.0f;
                }
                lora.push_back(la);
            }
        }
    }

    // Handle lora_list array
    if (readablemap::hasKey(env, params_map, "lora_list")) {
        jobject lora_list = readablemap::getArray(env, params_map, "lora_list");

        if (lora_list != nullptr) {
            // lora_adapters: ReadableArray<ReadableMap>
            int lora_list_size = readablearray::size(env, lora_list);
            for (int i = 0; i < lora_list_size; i++) {
                jobject lora_adapter = readablearray::getMap(env, lora_list, i);
                jstring path = readablemap::getString(env, lora_adapter, "path", nullptr);
                if (path != nullptr) {
                    const char *path_chars = env->GetStringUTFChars(path, nullptr);
                    common_adapter_lora_info la;
                    la.path = path_chars;
                    la.scale = readablemap::getFloat(env, lora_adapter, "scaled", 1.0f);
                    lora.push_back(la);
                    env->ReleaseStringUTFChars(path, path_chars);
                }
            }
        }
    }
    int result = llama->applyLoraAdapters(lora);

    // Cleanup lora string resources
    if (lora_str && lora_chars) {
        env->ReleaseStringUTFChars(lora_str, lora_chars);
    }

    if (result != 0) {
      LOGI("[RNLlama] Failed to apply lora adapters");
      llama_free(llama->ctx);
      context_map.erase((long) llama->ctx);
      delete llama;
      return nullptr;
    }

    bool gpu_used = false;
    bool gpu_device_available = false;
    std::string reason_no_gpu = "";
    std::string gpu_device_name = "";

    bool has_explicit_devices = !llama->params.devices.empty();
    bool explicit_gpu_requested = false;
    if (has_explicit_devices) {
        for (auto dev : llama->params.devices) {
            auto dev_type = lm_ggml_backend_dev_type(dev);
            if (dev_type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                explicit_gpu_requested = true;
                break;
            }
        }
    }

    if (llama->llama_init.model) {
        const auto &model_devices = llama->llama_init.model->devices;
        for (auto dev : model_devices) {
            auto dev_type = lm_ggml_backend_dev_type(dev);
            if (dev_type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpu_used = true;
                if (gpu_device_name.empty()) {
                    const char *used_name = lm_ggml_backend_dev_name(dev);
                    if (used_name != nullptr) {
                        gpu_device_name = used_name;
                    }
                }
            }
        }
    }

#ifdef LM_GGML_USE_OPENCL
    const size_t backend_dev_count = lm_ggml_backend_dev_count();
    for (size_t i = 0; i < backend_dev_count; ++i) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        auto dev_type = lm_ggml_backend_dev_type(dev);
        if (dev_type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_device_available = true;
        }
    }
#else
    gpu_device_available = false;
#endif

    if (!gpu_used) {
#ifdef LM_GGML_USE_OPENCL
        if (!gpu_device_available) {
            reason_no_gpu = "No compatible OpenCL GPU detected";
        }
#else
        reason_no_gpu = "OpenCL backend not enabled in this build";
#endif
        if (reason_no_gpu.empty() && explicit_gpu_requested) {
            reason_no_gpu = "GPU requested but not used";
        }
    }

    auto result_map = writablemap::createWriteableMap(env);

    const auto context_ptr = reinterpret_cast<intptr_t>(llama->ctx);
    const std::string context_str = std::to_string(context_ptr);

    writablemap::putString(env, result_map, "context", context_str.c_str());
    writablemap::putBoolean(env, result_map, "gpu", gpu_used);
    writablemap::putString(env, result_map, "reasonNoGPU", reason_no_gpu.c_str());
    if (gpu_used && !gpu_device_name.empty()) {
        writablemap::putString(env, result_map, "gpuDevice", gpu_device_name.c_str());
    }

    return result_map;
}


JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_interruptLoad(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama) {
        llama->is_load_interrupted = true;
    }
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_loadModelDetails(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    int count = llama_model_meta_count(llama->model);
    auto meta = writablemap::createWriteableMap(env);
    for (int i = 0; i < count; i++) {
        char key[256];
        llama_model_meta_key_by_index(llama->model, i, key, sizeof(key));
        char val[16384];  // gpt-oss's chat template is 12kb
        llama_model_meta_val_str_by_index(llama->model, i, val, sizeof(val));

        writablemap::putString(env, meta, key, val);
    }

    auto result = writablemap::createWriteableMap(env);

    char desc[1024];
    llama_model_desc(llama->model, desc, sizeof(desc));

    writablemap::putString(env, result, "desc", desc);
    writablemap::putDouble(env, result, "size", llama_model_size(llama->model));
    writablemap::putDouble(env, result, "nEmbd", llama_model_n_embd(llama->model));
    writablemap::putDouble(env, result, "nParams", llama_model_n_params(llama->model));
    auto chat_templates = writablemap::createWriteableMap(env);
    writablemap::putBoolean(env, chat_templates, "llamaChat", llama->validateModelChatTemplate(false, nullptr));

    auto minja = writablemap::createWriteableMap(env);
    writablemap::putBoolean(env, minja, "default", llama->validateModelChatTemplate(true, nullptr));

    auto default_caps = writablemap::createWriteableMap(env);

    auto default_tmpl = llama->templates.get()->template_default.get();
    auto default_tmpl_caps = default_tmpl->original_caps();
    writablemap::putBoolean(env, default_caps, "tools", default_tmpl_caps.supports_tools);
    writablemap::putBoolean(env, default_caps, "toolCalls", default_tmpl_caps.supports_tool_calls);
    writablemap::putBoolean(env, default_caps, "parallelToolCalls", default_tmpl_caps.supports_parallel_tool_calls);
    writablemap::putBoolean(env, default_caps, "toolResponses", default_tmpl_caps.supports_tool_responses);
    writablemap::putBoolean(env, default_caps, "systemRole", default_tmpl_caps.supports_system_role);
    writablemap::putBoolean(env, default_caps, "toolCallId", default_tmpl_caps.supports_tool_call_id);
    writablemap::putMap(env, minja, "defaultCaps", default_caps);

    writablemap::putBoolean(env, minja, "toolUse", llama->validateModelChatTemplate(true, "tool_use"));
    auto tool_use_tmpl = llama->templates.get()->template_tool_use.get();
    if (tool_use_tmpl != nullptr) {
      auto tool_use_caps = writablemap::createWriteableMap(env);
      auto tool_use_tmpl_caps = tool_use_tmpl->original_caps();
      writablemap::putBoolean(env, tool_use_caps, "tools", tool_use_tmpl_caps.supports_tools);
      writablemap::putBoolean(env, tool_use_caps, "toolCalls", tool_use_tmpl_caps.supports_tool_calls);
      writablemap::putBoolean(env, tool_use_caps, "parallelToolCalls", tool_use_tmpl_caps.supports_parallel_tool_calls);
      writablemap::putBoolean(env, tool_use_caps, "systemRole", tool_use_tmpl_caps.supports_system_role);
      writablemap::putBoolean(env, tool_use_caps, "toolResponses", tool_use_tmpl_caps.supports_tool_responses);
      writablemap::putBoolean(env, tool_use_caps, "toolCallId", tool_use_tmpl_caps.supports_tool_call_id);
      writablemap::putMap(env, minja, "toolUseCaps", tool_use_caps);
    }

    writablemap::putMap(env, chat_templates, "minja", minja);
    writablemap::putMap(env, result, "metadata", meta);
    writablemap::putMap(env, result, "chatTemplates", chat_templates);

    // deprecated
    writablemap::putBoolean(env, result, "isChatTemplateSupported", llama->validateModelChatTemplate(false, nullptr));

    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_getFormattedChatWithJinja(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages,
    jstring chat_template,
    jobject params
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *messages_chars = env->GetStringUTFChars(messages, nullptr);
    const char *tmpl_chars = env->GetStringUTFChars(chat_template, nullptr);

    // Extract optional string parameters
    jstring json_schema = readablemap::hasKey(env, params, "json_schema") ?
        readablemap::getString(env, params, "json_schema", nullptr) : nullptr;
    const char *json_schema_chars = json_schema ? env->GetStringUTFChars(json_schema, nullptr) : "";

    jstring tools = readablemap::hasKey(env, params, "tools") ?
        readablemap::getString(env, params, "tools", nullptr) : nullptr;
    const char *tools_chars = tools ? env->GetStringUTFChars(tools, nullptr) : "";

    jstring tool_choice = readablemap::hasKey(env, params, "tool_choice") ?
        readablemap::getString(env, params, "tool_choice", nullptr) : nullptr;
    const char *tool_choice_chars = tool_choice ? env->GetStringUTFChars(tool_choice, nullptr) : "";

    jstring now_str = readablemap::hasKey(env, params, "now_str") ?
        readablemap::getString(env, params, "now_str", nullptr) : nullptr;
    const char *now_chars = now_str ? env->GetStringUTFChars(now_str, nullptr) : "";

    jstring chat_template_kwargs = readablemap::hasKey(env, params, "chat_template_kwargs") ?
        readablemap::getString(env, params, "chat_template_kwargs", nullptr) : nullptr;
    const char *kwargs_chars = chat_template_kwargs ? env->GetStringUTFChars(chat_template_kwargs, nullptr) : "";

    // Extract boolean parameters
    jboolean parallel_tool_calls = readablemap::hasKey(env, params, "parallel_tool_calls") ?
        readablemap::getBool(env, params, "parallel_tool_calls", false) : false;

    jboolean enable_thinking = readablemap::hasKey(env, params, "enable_thinking") ?
        readablemap::getBool(env, params, "enable_thinking", false) : false;

    jboolean add_generation_prompt = readablemap::hasKey(env, params, "add_generation_prompt") ?
        readablemap::getBool(env, params, "add_generation_prompt", true) : true;

    std::map<std::string, std::string> kwargs_map;
    if (strlen(kwargs_chars) > 0) {
        try {
            auto kwargs_json = json::parse(kwargs_chars);
            for (auto& [key, value] : kwargs_json.items()) {
                if (value.is_string()) {
                    kwargs_map[key] = value.get<std::string>();
                }
            }
        } catch (...) {
            // Ignore JSON parsing errors for kwargs
        }
    }

    auto result = writablemap::createWriteableMap(env);
    try {
        auto formatted = llama->getFormattedChatWithJinja(
            messages_chars,
            tmpl_chars,
            json_schema_chars,
            tools_chars,
            parallel_tool_calls,
            tool_choice_chars,
            enable_thinking,
            add_generation_prompt,
            now_chars,
            kwargs_map
        );
        writablemap::putString(env, result, "prompt", formatted.prompt.c_str());
        writablemap::putInt(env, result, "chat_format", static_cast<int>(formatted.format));
        writablemap::putString(env, result, "grammar", formatted.grammar.c_str());
        writablemap::putBoolean(env, result, "grammar_lazy", formatted.grammar_lazy);
        auto grammar_triggers = writablearray::createWritableArray(env);
        for (const auto &trigger : formatted.grammar_triggers) {
            auto trigger_map = writablemap::createWriteableMap(env);
            writablemap::putInt(env, trigger_map, "type", trigger.type);
            writablemap::putString(env, trigger_map, "value", trigger.value.c_str());
            writablemap::putInt(env, trigger_map, "token", trigger.token);
            writablearray::pushMap(env, grammar_triggers, trigger_map);
        }
        writablemap::putBoolean(env, result, "thinking_forced_open", formatted.thinking_forced_open);
        writablemap::putArray(env, result, "grammar_triggers", grammar_triggers);
        auto preserved_tokens = writablearray::createWritableArray(env);
        for (const auto &token : formatted.preserved_tokens) {
            writablearray::pushString(env, preserved_tokens, token.c_str());
        }
        writablemap::putArray(env, result, "preserved_tokens", preserved_tokens);
        auto additional_stops = writablearray::createWritableArray(env);
        for (const auto &stop : formatted.additional_stops) {
            writablearray::pushString(env, additional_stops, stop.c_str());
        }
        writablemap::putArray(env, result, "additional_stops", additional_stops);
    } catch (const nlohmann::json_abi_v3_12_0::detail::parse_error& e) {
        std::string errorMessage = "JSON parse error in getFormattedChat: " + std::string(e.what());
        writablemap::putString(env, result, "_error", errorMessage.c_str());
        writablemap::putString(env, result, "_error_type", "json_parse_error");
        LOGI("[RNLlama] JSON parse error: %s", e.what());
    } catch (const std::invalid_argument& e) {
        std::string errorMessage = "Invalid argument in getFormattedChat: " + std::string(e.what());
        writablemap::putString(env, result, "_error", errorMessage.c_str());
        writablemap::putString(env, result, "_error_type", "invalid_argument");
        LOGI("[RNLlama] Invalid argument: %s", e.what());
    } catch (const std::runtime_error& e) {
        std::string errorMessage = "Runtime error in getFormattedChat: " + std::string(e.what());
        writablemap::putString(env, result, "_error", errorMessage.c_str());
        writablemap::putString(env, result, "_error_type", "runtime_error");
        LOGI("[RNLlama] Runtime error: %s", e.what());
    } catch (const std::exception& e) {
        std::string errorMessage = "C++ exception in getFormattedChat: " + std::string(e.what());
        writablemap::putString(env, result, "_error", errorMessage.c_str());
        writablemap::putString(env, result, "_error_type", "cpp_exception");
        LOGI("[RNLlama] C++ exception: %s", e.what());
    }
    // Cleanup - release string resources
    env->ReleaseStringUTFChars(messages, messages_chars);
    if (chat_template) env->ReleaseStringUTFChars(chat_template, tmpl_chars);
    if (json_schema) env->ReleaseStringUTFChars(json_schema, json_schema_chars);
    if (tools) env->ReleaseStringUTFChars(tools, tools_chars);
    if (tool_choice) env->ReleaseStringUTFChars(tool_choice, tool_choice_chars);
    if (now_str) env->ReleaseStringUTFChars(now_str, now_chars);
    if (chat_template_kwargs) env->ReleaseStringUTFChars(chat_template_kwargs, kwargs_chars);
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jstring JNICALL
Java_com_rnllama_LlamaContext_getFormattedChat(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring messages,
    jstring chat_template
) {
    UNUSED(thiz);

    try {
        auto llama = context_map[(long) context_ptr];
        if (!llama) {
            LOGI("[RNLlama] Error: Context pointer %ld not found in context_map", (long) context_ptr);
            env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Invalid context pointer in getFormattedChat");
            return nullptr;
        }

        const char *messages_chars = env->GetStringUTFChars(messages, nullptr);
        const char *tmpl_chars = env->GetStringUTFChars(chat_template, nullptr);

        if (!messages_chars) {
            env->ReleaseStringUTFChars(chat_template, tmpl_chars);
            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Messages parameter is null in getFormattedChat");
            return nullptr;
        }

        std::string formatted_chat = llama->getFormattedChat(messages_chars, tmpl_chars);

        env->ReleaseStringUTFChars(messages, messages_chars);
        env->ReleaseStringUTFChars(chat_template, tmpl_chars);

        std::string sanitized = rnbridge::sanitize_utf8_for_jni(formatted_chat.c_str());
        return env->NewStringUTF(sanitized.c_str());
    } catch (const nlohmann::json_abi_v3_12_0::detail::parse_error& e) {
        LOGI("[RNLlama] JSON parse error in getFormattedChat: %s", e.what());
        std::string errorMessage = "JSON parse error in getFormattedChat: " + std::string(e.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), errorMessage.c_str());
        return nullptr;
    } catch (const std::invalid_argument& e) {
        LOGI("[RNLlama] Invalid argument in getFormattedChat: %s", e.what());
        std::string errorMessage = "Invalid argument in getFormattedChat: " + std::string(e.what());
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), errorMessage.c_str());
        return nullptr;
    } catch (const std::runtime_error& e) {
        LOGI("[RNLlama] Runtime error in getFormattedChat: %s", e.what());
        std::string errorMessage = "Runtime error in getFormattedChat: " + std::string(e.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), errorMessage.c_str());
        return nullptr;
    } catch (const std::exception& e) {
        LOGI("[RNLlama] C++ exception in getFormattedChat: %s", e.what());
        std::string errorMessage = "C++ exception in getFormattedChat: " + std::string(e.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), errorMessage.c_str());
        return nullptr;
    }
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_loadSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    if (llama->completion == nullptr) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Context has been released");
        return reinterpret_cast<jobject>(result);
    }

    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    auto result = writablemap::createWriteableMap(env);
    size_t n_token_count_out = 0;
    llama->completion->embd.resize(llama->params.n_ctx);
    if (!llama_state_load_file(llama->ctx, path_chars, llama->completion->embd.data(), llama->completion->embd.capacity(), &n_token_count_out)) {
      env->ReleaseStringUTFChars(path, path_chars);

      writablemap::putString(env, result, "error", "Failed to load session");
      return reinterpret_cast<jobject>(result);
    }
    llama->completion->embd.resize(n_token_count_out);
    env->ReleaseStringUTFChars(path, path_chars);

    // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
    auto null_token_iter = std::find(llama->completion->embd.begin(), llama->completion->embd.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != llama->completion->embd.end()) {
        llama->completion->embd.resize(std::distance(llama->completion->embd.begin(), null_token_iter));
    }

    const std::string text = rnllama::tokens_to_str(llama->ctx, llama->completion->embd.cbegin(), llama->completion->embd.cend());
    writablemap::putInt(env, result, "tokens_loaded", n_token_count_out);
    writablemap::putString(env, result, "prompt", text.c_str());
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_saveSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path,
    jint size
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    auto result = writablemap::createWriteableMap(env);

    if (llama->completion == nullptr) {
        writablemap::putString(env, result, "error", "Context has been released");
        writablemap::putInt(env, result, "tokens_saved", 0);
        return result;
    }

    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    std::vector<llama_token> session_tokens = llama->completion->embd;

    // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
    auto null_token_iter = std::find(session_tokens.begin(), session_tokens.end(), LLAMA_TOKEN_NULL);
    if (null_token_iter != session_tokens.end()) {
        session_tokens.resize(std::distance(session_tokens.begin(), null_token_iter));
    }

    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_state_save_file(llama->ctx, path_chars, session_tokens.data(), save_size)) {
      env->ReleaseStringUTFChars(path, path_chars);
      writablemap::putString(env, result, "error", "Failed to save session file");
      writablemap::putInt(env, result, "tokens_saved", 0);
      return result;
    }

    env->ReleaseStringUTFChars(path, path_chars);
    writablemap::putInt(env, result, "tokens_saved", save_size);
    return result;
}

static inline jobject tokenProbsToMap(
  JNIEnv *env,
  rnllama::llama_rn_context *llama,
  std::vector<rnllama::completion_token_output> probs
) {
    auto result = writablearray::createWritableArray(env);
    for (const auto &prob : probs) {
        auto probsForToken = writablearray::createWritableArray(env);
        for (const auto &p : prob.probs) {
            std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, p.tok);
            auto probResult = writablemap::createWriteableMap(env);
            writablemap::putString(env, probResult, "tok_str", tokStr.c_str());
            writablemap::putDouble(env, probResult, "prob", p.prob);
            writablearray::pushMap(env, probsForToken, probResult);
        }
        std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, prob.tok);
        auto tokenResult = writablemap::createWriteableMap(env);
        writablemap::putString(env, tokenResult, "content", tokStr.c_str());
        writablemap::putArray(env, tokenResult, "probs", probsForToken);
        writablearray::pushMap(env, result, tokenResult);
    }
    return result;
}

static inline jobject tokensToArray(
    JNIEnv *env,
    rnllama::llama_rn_context *llama,
    std::vector<llama_token> tokens
) {
    auto result = writablearray::createWritableArray(env);
    for (const auto &token : tokens) {
        writablearray::pushInt(env, result, token);
    }
    return result;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_doCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jobject params,
    jobject partial_completion_callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    if (llama->completion == nullptr) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Context has been released");
        return reinterpret_cast<jobject>(result);
    }

    llama->completion->rewind();

    //llama_reset_timings(llama->ctx);

    // Extract required parameters
    jstring prompt = readablemap::getString(env, params, "prompt", nullptr);
    if (!prompt) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Missing required parameter: prompt");
        return reinterpret_cast<jobject>(result);
    }

    const char *prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    llama->params.prompt = prompt_chars;

    // Extract optional prefill_text
    if (readablemap::hasKey(env, params, "prefill_text")) {
        jstring prefill_text = readablemap::getString(env, params, "prefill_text", nullptr);
        const char *prefill_text_chars = env->GetStringUTFChars(prefill_text, nullptr);
        if (prefill_text_chars) {
            llama->completion->prefill_text = prefill_text_chars;
        }
        env->ReleaseStringUTFChars(prefill_text, prefill_text_chars);
    }

    // Extract guide_tokens if provided
    if (readablemap::hasKey(env, params, "guide_tokens")) {
        jobject guide_tokens_array = readablemap::getArray(env, params, "guide_tokens");

        int size = readablearray::size(env, guide_tokens_array);
        std::vector<llama_token> guide_tokens_vector(size);
        for (int i = 0; i < size; i++) {
            jclass arrayClass = env->GetObjectClass(guide_tokens_array);
            jmethodID getDoubleMethod = env->GetMethodID(arrayClass, "getDouble", "(I)D");
            guide_tokens_vector[i] = (jint) env->CallDoubleMethod(guide_tokens_array, getDoubleMethod, i);
        }
        if (llama->tts_wrapper != nullptr) {
            llama->tts_wrapper->setGuideTokens(guide_tokens_vector);
        }
    }

    // Extract media_paths if provided
    std::vector<std::string> media_paths_vector = extract_media_paths(env, params);
    if (!media_paths_vector.empty() && !llama->isMultimodalEnabled()) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Multimodal support not enabled. Call initMultimodal first.");
        env->ReleaseStringUTFChars(prompt, prompt_chars);
        return reinterpret_cast<jobject>(result);
    }

    // Set seed if provided
    if (readablemap::hasKey(env, params, "seed")) {
        jint seed = readablemap::getInt(env, params, "seed", -1);
        llama->params.sampling.seed = (seed == -1) ? time(NULL) : seed;
    }

    if (readablemap::hasKey(env, params, "n_threads")) {
        jint n_threads = readablemap::getInt(env, params, "n_threads", 0);
        set_best_cores(llama->params.cpuparams, n_threads);
    }

    // Set n_predict if provided
    if (readablemap::hasKey(env, params, "n_predict")) {
        llama->params.n_predict = readablemap::getInt(env, params, "n_predict", -1);
    }

    // Set ignore_eos if provided
    if (readablemap::hasKey(env, params, "ignore_eos")) {
        llama->params.sampling.ignore_eos = readablemap::getBool(env, params, "ignore_eos", false);
    }

    auto & sparams = llama->params.sampling;

    extract_sampling_params(env, params, llama, sparams, llama->params);

    const llama_model * model = llama_get_model(llama->ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (!llama->completion->initSampling()) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Failed to initialize sampling");
        return reinterpret_cast<jobject>(result);
    }

    // Extract chat_format if provided (defaults to 0)
    jint chat_format = 0;
    if (readablemap::hasKey(env, params, "chat_format")) {
        chat_format = readablemap::getInt(env, params, "chat_format", 0);
    }

    // Extract reasoning_format if provided (defaults to "none")
    common_reasoning_format reasoning_format_enum = common_reasoning_format_from_name("none");
    if (readablemap::hasKey(env, params, "reasoning_format")) {
        jstring reasoning_format = readablemap::getString(env, params, "reasoning_format", nullptr);
        const char *reasoning_format_chars = env->GetStringUTFChars(reasoning_format, nullptr);
        if (reasoning_format_chars) {
            std::string reasoning_format_str = reasoning_format_chars;
            reasoning_format_enum = common_reasoning_format_from_name(reasoning_format_str);
            env->ReleaseStringUTFChars(reasoning_format, reasoning_format_chars);
        }
    }

    // Extract thinking_forced_open if provided (defaults to false)
    jboolean thinking_forced_open = false;
    if (readablemap::hasKey(env, params, "thinking_forced_open")) {
        thinking_forced_open = readablemap::getBool(env, params, "thinking_forced_open", false);
    }

    llama->completion->beginCompletion(chat_format, reasoning_format_enum, thinking_forced_open);
    try {
        llama->completion->loadPrompt(media_paths_vector);
    } catch (const std::exception &e) {
        llama->completion->endCompletion();
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", e.what());
        return reinterpret_cast<jobject>(result);
    } catch (const std::runtime_error& e) {
        llama->completion->endCompletion();
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", e.what());
        return reinterpret_cast<jobject>(result);
    }

    if (llama->completion->context_full) {
        llama->completion->endCompletion();
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Context is full");
        return reinterpret_cast<jobject>(result);
    }

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->completion->has_next_token && !llama->completion->is_interrupted) {
        const rnllama::completion_token_output token_with_probs = llama->completion->doCompletion();
        if (token_with_probs.tok == -1 || llama->completion->incomplete) {
            continue;
        }
        const std::string token_text = common_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->completion->generated_text.size());

        const std::string str_test = llama->completion->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->completion->generated_text.erase(
                llama->completion->generated_text.begin() + pos + stop_pos,
                llama->completion->generated_text.end());
            pos = std::min(sent_count, llama->completion->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->completion->findStoppingStrings(str_test, token_text.size(),
                rnllama::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->completion->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->completion->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<rnllama::completion_token_output> probs_output = {};

            auto tokenResult = writablemap::createWriteableMap(env);
            writablemap::putString(env, tokenResult, "token", to_send.c_str());

            if (llama->params.sampling.n_probs > 0) {
              const std::vector<llama_token> to_send_toks = common_tokenize(llama->ctx, to_send, false);
              size_t probs_pos = std::min(sent_token_probs_index, llama->completion->generated_token_probs.size());
              size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->completion->generated_token_probs.size());
              if (probs_pos < probs_stop_pos) {
                  probs_output = std::vector<rnllama::completion_token_output>(llama->completion->generated_token_probs.begin() + probs_pos, llama->completion->generated_token_probs.begin() + probs_stop_pos);
              }
              sent_token_probs_index = probs_stop_pos;

              writablemap::putArray(env, tokenResult, "completion_probabilities", tokenProbsToMap(env, llama, probs_output));
            }

            auto partial_output = llama->completion->parseChatOutput(true);
            if (!partial_output.content.empty()) {
                writablemap::putString(env, tokenResult, "content", partial_output.content.c_str());
            }

            if (!partial_output.reasoning_content.empty()) {
                writablemap::putString(env, tokenResult, "reasoning_content", partial_output.reasoning_content.c_str());
            }
            if (!partial_output.tool_calls.empty()) {
                auto toolCallsArray = writablearray::createWritableArray(env);
                for (const auto& tc : partial_output.tool_calls) {
                    auto toolCall = writablemap::createWriteableMap(env);
                    writablemap::putString(env, toolCall, "type", "function");
                    auto functionMap = writablemap::createWriteableMap(env);
                    writablemap::putString(env, functionMap, "name", tc.name.c_str());
                    writablemap::putString(env, functionMap, "arguments", tc.arguments.c_str());
                    writablemap::putMap(env, toolCall, "function", functionMap);
                    if (!tc.id.empty()) {
                      writablemap::putString(env, toolCall, "id", tc.id.c_str());
                    }
                    writablearray::pushMap(env, toolCallsArray, toolCall);
                }
                writablemap::putArray(env, tokenResult, "tool_calls", toolCallsArray);
            }
            if (!partial_output.accumulated_text.empty()) {
                writablemap::putString(env, tokenResult, "accumulated_text", partial_output.accumulated_text.c_str());
            }

            jclass cb_class = env->GetObjectClass(partial_completion_callback);
            jmethodID onPartialCompletion = env->GetMethodID(cb_class, "onPartialCompletion", "(Lcom/facebook/react/bridge/WritableMap;)V");
            env->CallVoidMethod(partial_completion_callback, onPartialCompletion, tokenResult);
        }
    }

    // Release prompt_chars
    env->ReleaseStringUTFChars(prompt, prompt_chars);

    llama_perf_context_print(llama->ctx);
    llama->completion->endCompletion();

    auto toolCalls = writablearray::createWritableArray(env);
    std::string reasoningContent = "";
    std::string content;
    auto toolCallsSize = 0;
    if (!llama->completion->is_interrupted) {
        try {
            auto final_output = llama->completion->parseChatOutput(false);
            if (!final_output.reasoning_content.empty()) {
                reasoningContent = final_output.reasoning_content;
            }
            content = final_output.content;
            for (const auto &tc : final_output.tool_calls) {
                auto toolCall = writablemap::createWriteableMap(env);
                writablemap::putString(env, toolCall, "type", "function");
                auto functionMap = writablemap::createWriteableMap(env);
                writablemap::putString(env, functionMap, "name", tc.name.c_str());
                writablemap::putString(env, functionMap, "arguments", tc.arguments.c_str());
                writablemap::putMap(env, toolCall, "function", functionMap);
                if (!tc.id.empty()) {
                    writablemap::putString(env, toolCall, "id", tc.id.c_str());
                }
                writablearray::pushMap(env, toolCalls, toolCall);
                toolCallsSize++;
            }
        } catch (const std::exception &e) {
        } catch (...) {
        }
    }

    auto result = writablemap::createWriteableMap(env);
    writablemap::putInt(env, result, "chat_format", chat_format);
    writablemap::putString(env, result, "text", llama->completion->generated_text.c_str());
    if (!content.empty()) {
        writablemap::putString(env, result, "content", content.c_str());
    }
    if (!reasoningContent.empty()) {
        writablemap::putString(env, result, "reasoning_content", reasoningContent.c_str());
    }
    if (toolCallsSize > 0) {
        writablemap::putArray(env, result, "tool_calls", toolCalls);
    }
    if (llama->tts_wrapper != nullptr) {
        std::vector<llama_token> audio_tokens = llama->tts_wrapper->audio_tokens;
        writablemap::putArray(env, result, "audio_tokens", tokensToArray(env, llama, audio_tokens));
    }
    writablemap::putArray(env, result, "completion_probabilities", tokenProbsToMap(env, llama, llama->completion->generated_token_probs));
    writablemap::putInt(env, result, "tokens_predicted", llama->completion->num_tokens_predicted);
    writablemap::putInt(env, result, "tokens_evaluated", llama->completion->num_prompt_tokens);
    writablemap::putInt(env, result, "truncated", llama->completion->truncated);
    writablemap::putBoolean(env, result, "context_full", llama->completion->context_full);
    writablemap::putBoolean(env, result, "interrupted", llama->completion->is_interrupted);
    writablemap::putInt(env, result, "stopped_eos", llama->completion->stopped_eos);
    writablemap::putInt(env, result, "stopped_word", llama->completion->stopped_word);
    writablemap::putInt(env, result, "stopped_limit", llama->completion->stopped_limit);
    writablemap::putString(env, result, "stopping_word", llama->completion->stopping_word.c_str());
    writablemap::putInt(env, result, "tokens_cached", llama->completion->n_past);

    const auto timings_token = llama_perf_context(llama -> ctx);

    auto timingsResult = writablemap::createWriteableMap(env);
    writablemap::putInt(env, timingsResult, "prompt_n", timings_token.n_p_eval);
    writablemap::putInt(env, timingsResult, "prompt_ms", timings_token.t_p_eval_ms);
    writablemap::putInt(env, timingsResult, "prompt_per_token_ms", timings_token.t_p_eval_ms / timings_token.n_p_eval);
    writablemap::putDouble(env, timingsResult, "prompt_per_second", 1e3 / timings_token.t_p_eval_ms * timings_token.n_p_eval);
    writablemap::putInt(env, timingsResult, "predicted_n", timings_token.n_eval);
    writablemap::putInt(env, timingsResult, "predicted_ms", timings_token.t_eval_ms);
    writablemap::putInt(env, timingsResult, "predicted_per_token_ms", timings_token.t_eval_ms / timings_token.n_eval);
    writablemap::putDouble(env, timingsResult, "predicted_per_second", 1e3 / timings_token.t_eval_ms * timings_token.n_eval);

    writablemap::putMap(env, result, "timings", timingsResult);

    return reinterpret_cast<jobject>(result);
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_stopCompletion(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama->completion == nullptr) {
        return;
    }
    llama->completion->is_interrupted = true;
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_isPredicting(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama->completion == nullptr) {
        return false;
    }
    return llama->completion->is_predicting;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_tokenize(
        JNIEnv *env, jobject thiz, jlong context_ptr, jstring text, jobjectArray media_paths) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *text_chars = env->GetStringUTFChars(text, nullptr);
    std::vector<std::string> media_paths_vector;
    for (int i = 0; i < env->GetArrayLength(media_paths); i++) {
        jstring image_path = (jstring) env->GetObjectArrayElement(media_paths, i);
        const char *image_path_chars = env->GetStringUTFChars(image_path, nullptr);
        media_paths_vector.push_back(image_path_chars);
        env->ReleaseStringUTFChars(image_path, image_path_chars);
    }
    auto tokenize_result = llama->tokenize(text_chars, media_paths_vector);

    auto result = writablemap::createWriteableMap(env);

    auto tokens = writablearray::createWritableArray(env);
    for (const auto &tok : tokenize_result.tokens) {
      writablearray::pushInt(env, tokens, tok);
    }
    writablemap::putArray(env, result, "tokens", tokens);

    writablemap::putBoolean(env, result, "has_media", tokenize_result.has_media);

    auto bitmap_hashes = writablearray::createWritableArray(env);
    for (const auto &hash : tokenize_result.bitmap_hashes) {
      writablearray::pushString(env, bitmap_hashes, hash.c_str());
    }
    writablemap::putArray(env, result, "bitmap_hashes", bitmap_hashes);

    auto chunk_pos = writablearray::createWritableArray(env);
    for (const auto &pos : tokenize_result.chunk_pos) {
      writablearray::pushInt(env, chunk_pos, pos);
    }
    writablemap::putArray(env, result, "chunk_pos", chunk_pos);

    auto chunk_pos_media = writablearray::createWritableArray(env);
    for (const auto &pos : tokenize_result.chunk_pos_media) {
      writablearray::pushInt(env, chunk_pos_media, pos);
    }
    writablemap::putArray(env, result, "chunk_pos_media", chunk_pos_media);

    env->ReleaseStringUTFChars(text, text_chars);
    return result;
}

JNIEXPORT jstring JNICALL
Java_com_rnllama_LlamaContext_detokenize(
        JNIEnv *env, jobject thiz, jlong context_ptr, jintArray tokens) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    jsize tokens_len = env->GetArrayLength(tokens);
    jint *tokens_ptr = env->GetIntArrayElements(tokens, 0);
    std::vector<llama_token> toks;
    for (int i = 0; i < tokens_len; i++) {
        toks.push_back(tokens_ptr[i]);
    }

    auto text = rnllama::tokens_to_str(llama->ctx, toks.cbegin(), toks.cend());

    env->ReleaseIntArrayElements(tokens, tokens_ptr, 0);

    std::string sanitized = rnbridge::sanitize_utf8_for_jni(text.c_str());
    return env->NewStringUTF(sanitized.c_str());
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_embedding(
        JNIEnv *env, jobject thiz,
        jlong context_ptr,
        jstring text,
        jint embd_normalize
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    if (llama->completion == nullptr) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Context has been released");
        return result;
    }
    if (llama->completion->is_predicting) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Context is predicting");
        return result;
    }
    if (llama->params.embedding != true) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Embedding is not enabled");
        return result;
    }

    common_params embdParams;
    embdParams.embedding = true;
    embdParams.embd_normalize = llama->params.embd_normalize;
    if (embd_normalize != -1) {
      embdParams.embd_normalize = embd_normalize;
    }

    const char *text_chars = env->GetStringUTFChars(text, nullptr);

    llama->params.prompt = text_chars;
    llama->params.n_predict = 0;

    auto result = writablemap::createWriteableMap(env);
    try {
        std::vector<float> embedding = llama->completion->embedding(embdParams);

        auto embeddings = writablearray::createWritableArray(env);
        for (const auto &val : embedding) {
          writablearray::pushDouble(env, embeddings, (double) val);
        }
        writablemap::putArray(env, result, "embedding", embeddings);

        auto promptTokens = writablearray::createWritableArray(env);
        for (const auto &tok : llama->completion->embd) {
          writablearray::pushString(env, promptTokens, common_token_to_piece(llama->ctx, tok).c_str());
        }
        writablemap::putArray(env, result, "prompt_tokens", promptTokens);
    } catch (const std::exception &e) {
        llama->completion->endCompletion();
        writablemap::putString(env, result, "error", e.what());
    } catch (const std::runtime_error& e) {
        llama->completion->endCompletion();
        writablemap::putString(env, result, "error", e.what());
    }
    env->ReleaseStringUTFChars(text, text_chars);
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_rerank(
        JNIEnv *env, jobject thiz,
        jlong context_ptr,
        jstring query,
        jobjectArray documents,
        jint normalize
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    auto response = writablemap::createWriteableMap(env);

    if (llama->completion == nullptr) {
        writablemap::putString(env, response, "error", "Context has been released");
        return response;
    }
    if (llama->completion->is_predicting) {
        writablemap::putString(env, response, "error", "Context is predicting");
        return response;
    }
    if (llama->params.embedding != true) {
        writablemap::putString(env, response, "error", "Embedding is not enabled");
        return response;
    }

    const char *query_chars = env->GetStringUTFChars(query, nullptr);

    // Convert Java string array to C++ vector
    std::vector<std::string> documents_vector;
    int documents_size = env->GetArrayLength(documents);
    for (int i = 0; i < documents_size; i++) {
        jstring document = (jstring) env->GetObjectArrayElement(documents, i);
        const char *document_chars = env->GetStringUTFChars(document, nullptr);
        documents_vector.push_back(document_chars);
        env->ReleaseStringUTFChars(document, document_chars);
    }

    auto result = writablearray::createWritableArray(env);

    try {
        std::vector<float> scores = llama->completion->rerank(query_chars, documents_vector);

        for (size_t i = 0; i < scores.size(); i++) {
            auto item = writablemap::createWriteableMap(env);
            writablemap::putDouble(env, item, "score", (double) scores[i]);
            writablemap::putInt(env, item, "index", (int) i);
            writablearray::pushMap(env, result, item);
        }
        writablemap::putArray(env, response, "result", result);
    } catch (const std::exception &e) {
        writablemap::putString(env, response, "error", e.what());
        auto emptyResult = writablearray::createWritableArray(env);
        writablemap::putArray(env, response, "result", emptyResult);
    } catch (const std::runtime_error& e) {
        writablemap::putString(env, response, "error", e.what());
        auto emptyResult = writablearray::createWritableArray(env);
        writablemap::putArray(env, response, "result", emptyResult);
    }

    env->ReleaseStringUTFChars(query, query_chars);
    return response;
}

JNIEXPORT jstring JNICALL
Java_com_rnllama_LlamaContext_bench(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jint pp,
    jint tg,
    jint pl,
    jint nr
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama->completion == nullptr) {
        return env->NewStringUTF("");
    }
    std::string result = llama->completion->bench(pp, tg, pl, nr);
    std::string sanitized = rnbridge::sanitize_utf8_for_jni(result.c_str());
    return env->NewStringUTF(sanitized.c_str());
}

JNIEXPORT jint JNICALL
Java_com_rnllama_LlamaContext_applyLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr, jobjectArray loraAdapters) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    // lora_adapters: ReadableArray<ReadableMap>
    std::vector<common_adapter_lora_info> lora_adapters;
    int lora_adapters_size = readablearray::size(env, loraAdapters);
    for (int i = 0; i < lora_adapters_size; i++) {
        jobject lora_adapter = readablearray::getMap(env, loraAdapters, i);
        jstring path = readablemap::getString(env, lora_adapter, "path", nullptr);
        if (path != nullptr) {
          const char *path_chars = env->GetStringUTFChars(path, nullptr);
          env->ReleaseStringUTFChars(path, path_chars);
          float scaled = readablemap::getFloat(env, lora_adapter, "scaled", 1.0f);
          common_adapter_lora_info la;
          la.path = path_chars;
          la.scale = scaled;
          lora_adapters.push_back(la);
        }
    }
    return llama->applyLoraAdapters(lora_adapters);
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_removeLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->removeLoraAdapters();
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_getLoadedLoraAdapters(
    JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    auto loaded_lora_adapters = llama->getLoadedLoraAdapters();
    auto result = writablearray::createWritableArray(env);
    for (common_adapter_lora_info &la : loaded_lora_adapters) {
        auto map = writablemap::createWriteableMap(env);
        writablemap::putString(env, map, "path", la.path.c_str());
        writablemap::putDouble(env, map, "scaled", la.scale);
        writablearray::pushMap(env, result, map);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    context_map.erase((long) llama->ctx);
    delete llama;
}

struct log_callback_context {
    JavaVM *jvm;
    jobject callback;
};

static void rnllama_log_callback_to_j(lm_ggml_log_level level, const char * text, void * data) {
    auto level_c = "";
    if (level == LM_GGML_LOG_LEVEL_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, text, nullptr);
        level_c = "error";
    } else if (level == LM_GGML_LOG_LEVEL_INFO) {
        __android_log_print(ANDROID_LOG_INFO, TAG, text, nullptr);
        level_c = "info";
    } else if (level == LM_GGML_LOG_LEVEL_WARN) {
        __android_log_print(ANDROID_LOG_WARN, TAG, text, nullptr);
        level_c = "warn";
    } else {
        __android_log_print(ANDROID_LOG_DEFAULT, TAG, text, nullptr);
    }

    log_callback_context *cb_ctx = (log_callback_context *) data;

    JNIEnv *env;
    bool need_detach = false;
    int getEnvResult = cb_ctx->jvm->GetEnv((void**)&env, JNI_VERSION_1_6);

    if (getEnvResult == JNI_EDETACHED) {
        if (cb_ctx->jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            need_detach = true;
        } else {
            return;
        }
    } else if (getEnvResult != JNI_OK) {
        return;
    }

    jobject callback = cb_ctx->callback;
    jclass cb_class = env->GetObjectClass(callback);
    jmethodID emitNativeLog = env->GetMethodID(cb_class, "emitNativeLog", "(Ljava/lang/String;Ljava/lang/String;)V");

    jstring level_str = env->NewStringUTF(level_c);
    std::string sanitized_text = rnbridge::sanitize_utf8_for_jni(text);
    jstring text_str = env->NewStringUTF(sanitized_text.c_str());
    env->CallVoidMethod(callback, emitNativeLog, level_str, text_str);
    env->DeleteLocalRef(level_str);
    env->DeleteLocalRef(text_str);

    if (need_detach) {
        cb_ctx->jvm->DetachCurrentThread();
    }
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_setupLog(JNIEnv *env, jobject thiz, jobject logCallback) {
    UNUSED(thiz);

    log_callback_context *cb_ctx = new log_callback_context;

    JavaVM *jvm;
    env->GetJavaVM(&jvm);
    cb_ctx->jvm = jvm;
    cb_ctx->callback = env->NewGlobalRef(logCallback);

    llama_log_set(rnllama_log_callback_to_j, cb_ctx);
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_unsetLog(JNIEnv *env, jobject thiz) {
    UNUSED(env);
    UNUSED(thiz);
    llama_log_set(rnllama_log_callback_default, NULL);
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_initMultimodal(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring mmproj_path,
    jboolean mmproj_use_gpu
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *mmproj_path_chars = env->GetStringUTFChars(mmproj_path, nullptr);
    bool result = llama->initMultimodal(mmproj_path_chars, mmproj_use_gpu);
    env->ReleaseStringUTFChars(mmproj_path, mmproj_path_chars);

    return result;
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_isMultimodalEnabled(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->isMultimodalEnabled();
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_getMultimodalSupport(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    auto result = writablemap::createWriteableMap(env);
    writablemap::putBoolean(env, result, "vision", llama->isMultimodalSupportVision());
    writablemap::putBoolean(env, result, "audio", llama->isMultimodalSupportAudio());
    return result;
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_releaseMultimodal(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->releaseMultimodal();
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_initVocoder(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring vocoder_model_path,
    jint batch_size
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    const char *vocoder_model_path_chars = env->GetStringUTFChars(vocoder_model_path, nullptr);
    bool result = llama->initVocoder(vocoder_model_path_chars, batch_size);
    env->ReleaseStringUTFChars(vocoder_model_path, vocoder_model_path_chars);
    return result;
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_releaseVocoder(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->releaseVocoder();
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_isVocoderEnabled(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->isVocoderEnabled();
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_getFormattedAudioCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring speaker_json_str,
    jstring text_to_speak
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    const char *speaker_json_str_chars = env->GetStringUTFChars(speaker_json_str, nullptr);
    const char *text_to_speak_chars = env->GetStringUTFChars(text_to_speak, nullptr);

    auto result = writablemap::createWriteableMap(env);
    try {
        auto audio_result = llama->tts_wrapper->getFormattedAudioCompletion(llama, speaker_json_str_chars, text_to_speak_chars);
        writablemap::putString(env, result, "prompt", audio_result.prompt.c_str());
        if (audio_result.grammar != nullptr) {
            writablemap::putString(env, result, "grammar", audio_result.grammar);
        }
    } catch (const std::exception &e) {
        env->ReleaseStringUTFChars(speaker_json_str, speaker_json_str_chars);
        env->ReleaseStringUTFChars(text_to_speak, text_to_speak_chars);
        jclass exceptionClass = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exceptionClass, e.what());
        return nullptr;
    }

    env->ReleaseStringUTFChars(speaker_json_str, speaker_json_str_chars);
    env->ReleaseStringUTFChars(text_to_speak, text_to_speak_chars);
    return result;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_getAudioCompletionGuideTokens(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring text_to_speak
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    const char *text_to_speak_chars = env->GetStringUTFChars(text_to_speak, nullptr);
    std::vector<llama_token> guide_tokens = llama->tts_wrapper->getAudioCompletionGuideTokens(llama, text_to_speak_chars);
    env->ReleaseStringUTFChars(text_to_speak, text_to_speak_chars);
    auto result = writablearray::createWritableArray(env);
    for (const auto &val : guide_tokens) {
        writablearray::pushInt(env, result, (int) val);
    }
    return result;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_decodeAudioTokens(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jintArray tokens
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    jsize tokens_size = env->GetArrayLength(tokens);
    jint *tokens_ptr = env->GetIntArrayElements(tokens, nullptr);
    std::vector<llama_token> tokens_vec(tokens_size);
    for (int i = 0; i < tokens_size; i++) {
        tokens_vec[i] = tokens_ptr[i];
    }
    env->ReleaseIntArrayElements(tokens, tokens_ptr, 0);
    std::vector<float> audio = llama->tts_wrapper->decodeAudioTokens(llama, tokens_vec);
    auto result = writablearray::createWritableArray(env);
    for (const auto &val : audio) {
      writablearray::pushDouble(env, result, (double) val);
    }
    return result;
}

// Parallel decoding support
JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_enableParallelMode(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jint n_parallel,
    jint n_batch
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama) {
        LOGE("enableParallelMode: Invalid context pointer");
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "Invalid context pointer");
        return;
    }

    try {
        llama->enableParallelMode(n_parallel, n_batch);
    } catch (const std::runtime_error& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
    }
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_startProcessingLoop(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        return;
    }
    llama->slot_manager->start_processing_loop();
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_stopProcessingLoop(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        return;
    }
    llama->slot_manager->stop_processing_loop();
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_updateSlots(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        return;
    }
    llama->slot_manager->update_slots();
}

// Global context for JNI callbacks
struct jni_callback_context {
    JNIEnv *env;
    JavaVM *jvm;
    jobject partial_callback;
    jobject complete_callback;
    int request_id;
};

// Map to store callback contexts
std::unordered_map<int32_t, std::shared_ptr<jni_callback_context>> jni_callback_map;

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_doQueueCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jobject params_map,
    jobject partial_completion_callback,
    jobject completion_callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Invalid context or parallel mode not enabled");
        return result;
    }

    try {
        // Extract required prompt parameter
        jstring prompt = readablemap::getString(env, params_map, "prompt", nullptr);
        if (!prompt) {
            auto result = writablemap::createWriteableMap(env);
            writablemap::putString(env, result, "error", "Missing required parameter: prompt");
            return result;
        }
        const char *prompt_chars = env->GetStringUTFChars(prompt, nullptr);

        // Build params from llama context defaults
        common_params params = llama->params;
        common_params_sampling& sparams = params.sampling;

        if (readablemap::hasKey(env, params_map, "n_threads")) {
            jint n_threads = params.cpuparams.n_threads;
            n_threads = readablemap::getInt(env, params_map, "n_threads", params.cpuparams.n_threads);
            set_best_cores(llama->params.cpuparams, n_threads);
        }

        // TODO: Support guide_tokens (for TTS)
        extract_sampling_params(env, params_map, llama, sparams, params);

        // Extract chat_format, reasoning_format, thinking_forced_open
        jint chat_format = 0;
        if (readablemap::hasKey(env, params_map, "chat_format")) {
            chat_format = readablemap::getInt(env, params_map, "chat_format", 0);
        }

        common_reasoning_format reasoning_format_enum = COMMON_REASONING_FORMAT_NONE;
        if (readablemap::hasKey(env, params_map, "reasoning_format")) {
            jstring reasoning_format = readablemap::getString(env, params_map, "reasoning_format", nullptr);
            if (reasoning_format) {
                const char *reasoning_format_chars = env->GetStringUTFChars(reasoning_format, nullptr);
                std::string reasoning_format_str = reasoning_format_chars;
                reasoning_format_enum = common_reasoning_format_from_name(reasoning_format_str);
                env->ReleaseStringUTFChars(reasoning_format, reasoning_format_chars);
            }
        }

        jboolean thinking_forced_open = false;
        if (readablemap::hasKey(env, params_map, "thinking_forced_open")) {
            thinking_forced_open = readablemap::getBool(env, params_map, "thinking_forced_open", false);
        }

        // Extract prefill_text
        const char *prefill_text_chars = "";
        jstring prefill_text = nullptr;
        if (readablemap::hasKey(env, params_map, "prefill_text")) {
            prefill_text = readablemap::getString(env, params_map, "prefill_text", nullptr);
            if (prefill_text) {
                prefill_text_chars = env->GetStringUTFChars(prefill_text, nullptr);
            }
        }

        // Extract media_paths array
        std::vector<std::string> media_paths_vec = extract_media_paths(env, params_map);
        if (!media_paths_vec.empty() && !llama->isMultimodalEnabled()) {
            throw std::runtime_error("Multimodal support not enabled. Call initMultimodal first.");
        }

        // Tokenize prompt using llama->tokenize (handles multimodal content)
        rnllama::llama_rn_tokenize_result tokenize_result = llama->tokenize(
            prompt_chars ? prompt_chars : "",
            media_paths_vec
        );

        // Get JavaVM for later callback
        JavaVM *jvm;
        env->GetJavaVM(&jvm);

        // Create callback context
        auto cb_ctx = std::make_shared<jni_callback_context>();
        cb_ctx->jvm = jvm;
        cb_ctx->partial_callback = env->NewGlobalRef(partial_completion_callback);
        cb_ctx->complete_callback = env->NewGlobalRef(completion_callback);
        cb_ctx->request_id = -1; // Will be set after queueing

        // Token callback
        auto on_token = [cb_ctx, llama](const rnllama::completion_token_output& token_output) {
            JNIEnv *env_cb;
            bool attached = false;

            // Attach to JVM if needed
            int getEnvResult = cb_ctx->jvm->GetEnv((void**)&env_cb, JNI_VERSION_1_6);
            if (getEnvResult == JNI_EDETACHED) {
                cb_ctx->jvm->AttachCurrentThread(&env_cb, nullptr);
                attached = true;
            }

            // Build token result map
            auto tokenResult = writablemap::createWriteableMap(env_cb);
            writablemap::putInt(env_cb, tokenResult, "requestId", token_output.request_id);
            // Use the pre-decoded text from the token output
            writablemap::putString(env_cb, tokenResult, "token", token_output.text.c_str());

            // Add probabilities if available
            if (!token_output.probs.empty()) {
                auto probsArray = writablearray::createWritableArray(env_cb);
                for (const auto &p : token_output.probs) {
                    auto probMap = writablemap::createWriteableMap(env_cb);
                    std::string tok_str = rnllama::tokens_to_output_formatted_string(llama->ctx, p.tok);
                    writablemap::putString(env_cb, probMap, "tok_str", tok_str.c_str());
                    writablemap::putDouble(env_cb, probMap, "prob", p.prob);
                    writablearray::pushMap(env_cb, probsArray, probMap);
                }
                writablemap::putArray(env_cb, tokenResult, "probs", probsArray);
            }

            // Find slot and parse chat output
            rnllama::completion_chat_output parsed_output;
            bool has_parsed_output = false;
            if (llama->slot_manager) {
                auto* slot = llama->slot_manager->get_slot_by_request_id(token_output.request_id);
                if (slot) {
                    parsed_output = slot->parseChatOutput(true);  // is_partial = true
                    has_parsed_output = true;
                }
            }

            // Add parsed chat output (content, reasoning_content, tool_calls, accumulated_text)
            if (has_parsed_output) {
                if (!parsed_output.content.empty()) {
                    writablemap::putString(env_cb, tokenResult, "content", parsed_output.content.c_str());
                }
                if (!parsed_output.reasoning_content.empty()) {
                    writablemap::putString(env_cb, tokenResult, "reasoning_content", parsed_output.reasoning_content.c_str());
                }
                if (!parsed_output.tool_calls.empty()) {
                    auto toolCallsArray = writablearray::createWritableArray(env_cb);
                    for (const auto &tc : parsed_output.tool_calls) {
                        auto toolCallMap = writablemap::createWriteableMap(env_cb);
                        writablemap::putString(env_cb, toolCallMap, "type", "function");
                        auto functionMap = writablemap::createWriteableMap(env_cb);
                        writablemap::putString(env_cb, functionMap, "name", tc.name.c_str());
                        writablemap::putString(env_cb, functionMap, "arguments", tc.arguments.c_str());
                        writablemap::putMap(env_cb, toolCallMap, "function", functionMap);
                        if (!tc.id.empty()) {
                            writablemap::putString(env_cb, toolCallMap, "id", tc.id.c_str());
                        }
                        writablearray::pushMap(env_cb, toolCallsArray, toolCallMap);
                    }
                    writablemap::putArray(env_cb, tokenResult, "tool_calls", toolCallsArray);
                }
                if (!parsed_output.accumulated_text.empty()) {
                    writablemap::putString(env_cb, tokenResult, "accumulated_text", parsed_output.accumulated_text.c_str());
                }
            }

            // Call Java callback
            jclass callbackClass = env_cb->GetObjectClass(cb_ctx->partial_callback);
            jmethodID onPartialMethod = env_cb->GetMethodID(callbackClass, "onPartialCompletion", "(Lcom/facebook/react/bridge/WritableMap;)V");
            if (onPartialMethod) {
                env_cb->CallVoidMethod(cb_ctx->partial_callback, onPartialMethod, tokenResult);
            }

            if (attached) {
                cb_ctx->jvm->DetachCurrentThread();
            }
        };

        // Completion callback
        auto on_complete = [cb_ctx, llama](rnllama::llama_rn_slot* slot) {
            JNIEnv *env_cb;
            bool attached = false;

            int getEnvResult = cb_ctx->jvm->GetEnv((void**)&env_cb, JNI_VERSION_1_6);
            if (getEnvResult == JNI_EDETACHED) {
                cb_ctx->jvm->AttachCurrentThread(&env_cb, nullptr);
                attached = true;
            }

            // Build completion result (similar to doCompletion)
            auto result = writablemap::createWriteableMap(env_cb);
            writablemap::putInt(env_cb, result, "requestId", slot->request_id);
            std::string text = rnllama::tokens_to_str(llama->ctx, slot->generated_tokens.begin(), slot->generated_tokens.end());
            writablemap::putString(env_cb, result, "text", text.c_str());
            writablemap::putInt(env_cb, result, "tokens_predicted", slot->generated_tokens.size());
            writablemap::putInt(env_cb, result, "tokens_evaluated", slot->prompt_tokens.size());
            writablemap::putBoolean(env_cb, result, "truncated", false);
            writablemap::putBoolean(env_cb, result, "stopped_eos", slot->stopped_eos);
            writablemap::putString(env_cb, result, "stopping_word", slot->stopping_word.c_str());

            // Add error message if present
            if (!slot->error_message.empty()) {
                writablemap::putString(env_cb, result, "error", slot->error_message.c_str());
            }

            // Add timings
            rnllama::slot_timings timings = slot->get_timings();
            auto timingsMap = writablemap::createWriteableMap(env_cb);
            writablemap::putInt(env_cb, timingsMap, "cache_n", timings.cache_n);
            writablemap::putInt(env_cb, timingsMap, "prompt_n", timings.prompt_n);
            writablemap::putDouble(env_cb, timingsMap, "prompt_ms", timings.prompt_ms);
            writablemap::putDouble(env_cb, timingsMap, "prompt_per_token_ms", timings.prompt_per_token_ms);
            writablemap::putDouble(env_cb, timingsMap, "prompt_per_second", timings.prompt_per_second);
            writablemap::putInt(env_cb, timingsMap, "predicted_n", timings.predicted_n);
            writablemap::putDouble(env_cb, timingsMap, "predicted_ms", timings.predicted_ms);
            writablemap::putDouble(env_cb, timingsMap, "predicted_per_token_ms", timings.predicted_per_token_ms);
            writablemap::putDouble(env_cb, timingsMap, "predicted_per_second", timings.predicted_per_second);
            writablemap::putMap(env_cb, result, "timings", timingsMap);

            // Parse final chat output
            rnllama::completion_chat_output final_output;
            bool has_final_output = false;
            try {
                final_output = slot->parseChatOutput(false);  // is_partial = false
                has_final_output = true;
            } catch (...) {
                // Ignore parsing errors
            }

            // Add parsed chat output (final)
            if (has_final_output) {
                if (!final_output.content.empty()) {
                    writablemap::putString(env_cb, result, "content", final_output.content.c_str());
                }
                if (!final_output.reasoning_content.empty()) {
                    writablemap::putString(env_cb, result, "reasoning_content", final_output.reasoning_content.c_str());
                }
                if (!final_output.tool_calls.empty()) {
                    auto toolCallsArray = writablearray::createWritableArray(env_cb);
                    for (const auto &tc : final_output.tool_calls) {
                        auto toolCallMap = writablemap::createWriteableMap(env_cb);
                        writablemap::putString(env_cb, toolCallMap, "type", "function");
                        auto functionMap = writablemap::createWriteableMap(env_cb);
                        writablemap::putString(env_cb, functionMap, "name", tc.name.c_str());
                        writablemap::putString(env_cb, functionMap, "arguments", tc.arguments.c_str());
                        writablemap::putMap(env_cb, toolCallMap, "function", functionMap);
                        if (!tc.id.empty()) {
                            writablemap::putString(env_cb, toolCallMap, "id", tc.id.c_str());
                        }
                        writablearray::pushMap(env_cb, toolCallsArray, toolCallMap);
                    }
                    writablemap::putArray(env_cb, result, "tool_calls", toolCallsArray);
                }
            }

            // Call Java callback
            jclass callbackClass = env_cb->GetObjectClass(cb_ctx->complete_callback);
            jmethodID onCompleteMethod = env_cb->GetMethodID(callbackClass, "onComplete", "(Lcom/facebook/react/bridge/WritableMap;)V");
            if (onCompleteMethod) {
                env_cb->CallVoidMethod(cb_ctx->complete_callback, onCompleteMethod, result);
            }

            // Cleanup
            env_cb->DeleteGlobalRef(cb_ctx->partial_callback);
            env_cb->DeleteGlobalRef(cb_ctx->complete_callback);
            jni_callback_map.erase(cb_ctx->request_id);

            if (attached) {
                cb_ctx->jvm->DetachCurrentThread();
            }
        };

        // Convert prefill_text to std::string
        std::string prefill_text_str = prefill_text_chars ? prefill_text_chars : "";

        // Extract state paths
        std::string load_state_path;
        std::string save_state_path;
        int32_t save_state_size = -1;

        if (readablemap::hasKey(env, params_map, "load_state_path")) {
            jstring load_path = readablemap::getString(env, params_map, "load_state_path", nullptr);
            if (load_path) {
                const char *load_path_chars = env->GetStringUTFChars(load_path, nullptr);
                load_state_path = load_path_chars;
                // Remove file:// prefix if present
                if (load_state_path.find("file://") == 0) {
                    load_state_path = load_state_path.substr(7);
                }
                env->ReleaseStringUTFChars(load_path, load_path_chars);
            }
        }

        if (readablemap::hasKey(env, params_map, "save_state_path")) {
            jstring save_path = readablemap::getString(env, params_map, "save_state_path", nullptr);
            if (save_path) {
                const char *save_path_chars = env->GetStringUTFChars(save_path, nullptr);
                save_state_path = save_path_chars;
                // Remove file:// prefix if present
                if (save_state_path.find("file://") == 0) {
                    save_state_path = save_state_path.substr(7);
                }
                env->ReleaseStringUTFChars(save_path, save_path_chars);
            }
        }

        if (readablemap::hasKey(env, params_map, "save_state_size")) {
            save_state_size = readablemap::getInt(env, params_map, "save_state_size", -1);
        }

        // Queue the request with all required parameters
        int32_t request_id = llama->slot_manager->queue_request(
            params,
            tokenize_result.tokens,
            tokenize_result.has_media ? media_paths_vec : std::vector<std::string>(),
            prompt_chars,
            chat_format,
            reasoning_format_enum,
            thinking_forced_open,
            prefill_text_str,
            load_state_path,
            save_state_path,
            save_state_size,
            on_token,
            on_complete
        );
        cb_ctx->request_id = request_id;
        jni_callback_map[request_id] = cb_ctx;

        // Cleanup string resources
        env->ReleaseStringUTFChars(prompt, prompt_chars);
        if (prefill_text) {
            env->ReleaseStringUTFChars(prefill_text, prefill_text_chars);
        }

        auto result = writablemap::createWriteableMap(env);
        writablemap::putInt(env, result, "requestId", request_id);
        return result;
    } catch (const std::exception &e) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", e.what());
        return result;
    }
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_doCancelRequest(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jint request_id
) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        return;
    }
    llama->slot_manager->cancel_request(request_id);
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_doQueueEmbedding(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring text,
    jint embd_normalize,
    jobject callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Invalid context or parallel mode not enabled");
        return result;
    }

    try {
        const char *text_chars = env->GetStringUTFChars(text, nullptr);
        const llama_vocab* vocab = llama_model_get_vocab(llama->model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const bool is_enc_dec = llama_model_has_encoder(llama->model);
        std::vector<llama_token> tokens = common_tokenize(
            llama->ctx,
            text_chars,
            add_bos || is_enc_dec,
            true
        );
        env->ReleaseStringUTFChars(text, text_chars);

        // Get JavaVM for callback
        JavaVM *jvm;
        env->GetJavaVM(&jvm);

        // Create global ref for callback
        jobject callback_ref = env->NewGlobalRef(callback);

        // Queue embedding request
        int32_t request_id = llama->slot_manager->queue_embedding_request(
            tokens,
            embd_normalize,
            [jvm, callback_ref](int32_t request_id, const std::vector<float>& embedding) {
                // Copy embedding vector to avoid dangling reference
                std::vector<float> embedding_copy = embedding;

                JNIEnv *env_cb;
                bool attached = false;

                int getEnvResult = jvm->GetEnv((void**)&env_cb, JNI_VERSION_1_6);
                if (getEnvResult == JNI_EDETACHED) {
                    jvm->AttachCurrentThread(&env_cb, nullptr);
                    attached = true;
                }

                // Create embedding array
                auto embeddingArray = writablearray::createWritableArray(env_cb);
                for (float val : embedding_copy) {
                    writablearray::pushDouble(env_cb, embeddingArray, val);
                }

                // Call Java callback with request_id parameter
                jclass callbackClass = env_cb->GetObjectClass(callback_ref);
                jmethodID onResultMethod = env_cb->GetMethodID(callbackClass, "onResult", "(ILcom/facebook/react/bridge/WritableArray;)V");
                if (onResultMethod) {
                    env_cb->CallVoidMethod(callback_ref, onResultMethod, request_id, embeddingArray);
                }

                // Clean up global ref
                env_cb->DeleteGlobalRef(callback_ref);

                if (attached) {
                    jvm->DetachCurrentThread();
                }
            }
        );

        auto result = writablemap::createWriteableMap(env);
        writablemap::putInt(env, result, "requestId", request_id);
        return result;
    } catch (const std::exception& e) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", e.what());
        return result;
    }
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_doQueueRerank(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring query,
    jobjectArray documents,
    jint normalize,
    jobject callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (!llama || !llama->slot_manager) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", "Invalid context or parallel mode not enabled");
        return result;
    }

    try {
        // Convert query
        const char *query_chars = env->GetStringUTFChars(query, nullptr);
        std::string query_str(query_chars);
        env->ReleaseStringUTFChars(query, query_chars);

        // Convert documents array
        std::vector<std::string> documents_vec;
        jsize doc_count = env->GetArrayLength(documents);
        for (jsize i = 0; i < doc_count; i++) {
            jstring doc_str = (jstring) env->GetObjectArrayElement(documents, i);
            const char *doc_chars = env->GetStringUTFChars(doc_str, nullptr);
            documents_vec.push_back(doc_chars);
            env->ReleaseStringUTFChars(doc_str, doc_chars);
        }

        // Get JavaVM for callback
        JavaVM *jvm;
        env->GetJavaVM(&jvm);

        // Create global ref for callback
        jobject callback_ref = env->NewGlobalRef(callback);

        // Queue rerank request
        int32_t request_id = llama->slot_manager->queue_rerank_request(
            query_str,
            documents_vec,
            normalize,
            [jvm, callback_ref](int32_t request_id, const std::vector<float>& scores) {
                // Copy scores vector to avoid dangling reference
                std::vector<float> scores_copy = scores;

                JNIEnv *env_cb;
                bool attached = false;

                int getEnvResult = jvm->GetEnv((void**)&env_cb, JNI_VERSION_1_6);
                if (getEnvResult == JNI_EDETACHED) {
                    jvm->AttachCurrentThread(&env_cb, nullptr);
                    attached = true;
                }

                // Create results array
                auto resultsArray = writablearray::createWritableArray(env_cb);
                for (size_t i = 0; i < scores_copy.size(); i++) {
                    auto resultMap = writablemap::createWriteableMap(env_cb);
                    writablemap::putDouble(env_cb, resultMap, "score", scores_copy[i]);
                    writablemap::putInt(env_cb, resultMap, "index", (int)i);
                    writablearray::pushMap(env_cb, resultsArray, resultMap);
                }

                // Call Java callback with request_id parameter
                jclass callbackClass = env_cb->GetObjectClass(callback_ref);
                jmethodID onResultsMethod = env_cb->GetMethodID(callbackClass, "onResults", "(ILcom/facebook/react/bridge/WritableArray;)V");
                if (onResultsMethod) {
                    env_cb->CallVoidMethod(callback_ref, onResultsMethod, request_id, resultsArray);
                }

                // Clean up global ref
                env_cb->DeleteGlobalRef(callback_ref);

                if (attached) {
                    jvm->DetachCurrentThread();
                }
            }
        );

        auto result = writablemap::createWriteableMap(env);
        writablemap::putInt(env, result, "requestId", request_id);
        return result;
    } catch (const std::exception& e) {
        auto result = writablemap::createWriteableMap(env);
        writablemap::putString(env, result, "error", e.what());
        return result;
    }
}

// JNI_OnLoad: Called when the library is loaded
// Initialize React Native bridge utilities
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
    UNUSED(reserved);
    JNIEnv* env;

    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        LOGE("JNI_OnLoad: Failed to get JNIEnv");
        return JNI_ERR;
    }

    // Initialize rnbridge namespace with cached React Native class references
    if (!rnbridge::initialize(env)) {
        LOGE("JNI_OnLoad: Failed to initialize rnbridge");
        return JNI_ERR;
    }

    LOGI("JNI_OnLoad: Successfully initialized");
    return JNI_VERSION_1_6;
}

// JNI_OnUnload: Called when the library is unloaded
JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
    UNUSED(reserved);
    JNIEnv* env;

    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return;
    }

    // Cleanup rnbridge namespace
    rnbridge::cleanup(env);
    LOGI("JNI_OnUnload: Cleaned up rnbridge");
}

} // extern "C"
