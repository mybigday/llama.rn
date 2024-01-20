#include <jni.h>
// #include <android/asset_manager.h>
// #include <android/asset_manager_jni.h>
#include <android/log.h>
#include <cstdlib>
#include <sys/sysinfo.h>
#include <string>
#include <thread>
#include <unordered_map>
#include "llama.h"
#include "rn-llama.hpp"
#include "ggml.h"

#define UNUSED(x) (void)(x)
#define TAG "RNLLAMA_ANDROID_JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

extern "C" {

// Method to create WritableMap
static inline jobject createWriteableMap(JNIEnv *env) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
    jmethodID init = env->GetStaticMethodID(mapClass, "createMap", "()Lcom/facebook/react/bridge/WritableMap;");
    jobject map = env->CallStaticObjectMethod(mapClass, init);
    return map;
}

// Method to put string into WritableMap
static inline void putString(JNIEnv *env, jobject map, const char *key, const char *value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putStringMethod = env->GetMethodID(mapClass, "putString", "(Ljava/lang/String;Ljava/lang/String;)V");

    jstring jKey = env->NewStringUTF(key);
    jstring jValue = env->NewStringUTF(value);

    env->CallVoidMethod(map, putStringMethod, jKey, jValue);
}

// Method to put int into WritableMap
static inline void putInt(JNIEnv *env, jobject map, const char *key, int value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putIntMethod = env->GetMethodID(mapClass, "putInt", "(Ljava/lang/String;I)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putIntMethod, jKey, value);
}

// Method to put double into WritableMap
static inline void putDouble(JNIEnv *env, jobject map, const char *key, double value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putDoubleMethod = env->GetMethodID(mapClass, "putDouble", "(Ljava/lang/String;D)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putDoubleMethod, jKey, value);
}

// Method to put WriteableMap into WritableMap
static inline void putMap(JNIEnv *env, jobject map, const char *key, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putMapMethod = env->GetMethodID(mapClass, "putMap", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableMap;)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putMapMethod, jKey, value);
}

// Method to create WritableArray
static inline jobject createWritableArray(JNIEnv *env) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
    jmethodID init = env->GetStaticMethodID(mapClass, "createArray", "()Lcom/facebook/react/bridge/WritableArray;");
    jobject map = env->CallStaticObjectMethod(mapClass, init);
    return map;
}

// Method to push int into WritableArray
static inline void pushInt(JNIEnv *env, jobject arr, int value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushIntMethod = env->GetMethodID(mapClass, "pushInt", "(I)V");

    env->CallVoidMethod(arr, pushIntMethod, value);
}

// Method to push double into WritableArray
static inline void pushDouble(JNIEnv *env, jobject arr, double value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushDoubleMethod = env->GetMethodID(mapClass, "pushDouble", "(D)V");

    env->CallVoidMethod(arr, pushDoubleMethod, value);
}

// Method to push WritableMap into WritableArray
static inline void pushMap(JNIEnv *env, jobject arr, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    jmethodID pushMapMethod = env->GetMethodID(mapClass, "pushMap", "(Lcom/facebook/react/bridge/WritableMap;)V");

    env->CallVoidMethod(arr, pushMapMethod, value);
}

// Method to put WritableArray into WritableMap
static inline void putArray(JNIEnv *env, jobject map, const char *key, jobject value) {
    jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    jmethodID putArrayMethod = env->GetMethodID(mapClass, "putArray", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableArray;)V");

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putArrayMethod, jKey, value);
}


std::unordered_map<long, rnllama::llama_rn_context *> context_map;

JNIEXPORT jlong JNICALL
Java_com_rnllama_LlamaContext_initContext(
    JNIEnv *env,
    jobject thiz,
    jstring model_path_str,
    jboolean embedding,
    jint n_ctx,
    jint n_batch,
    jint n_threads,
    jint n_gpu_layers, // TODO: Support this
    jboolean use_mlock,
    jboolean use_mmap,
    jstring lora_str,
    jfloat lora_scaled,
    jstring lora_base_str,
    jfloat rope_freq_base,
    jfloat rope_freq_scale
) {
    UNUSED(thiz);

    gpt_params defaultParams;

    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);
    defaultParams.model = model_path_chars;

    defaultParams.embedding = embedding;

    defaultParams.n_ctx = n_ctx;
    defaultParams.n_batch = n_batch;

    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);
    defaultParams.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    defaultParams.n_gpu_layers = n_gpu_layers;

    defaultParams.use_mlock = use_mlock;
    defaultParams.use_mmap = use_mmap;

    const char *lora_chars = env->GetStringUTFChars(lora_str, nullptr);
    const char *lora_base_chars = env->GetStringUTFChars(lora_base_str, nullptr);
    if (lora_chars != nullptr && lora_chars[0] != '\0') {
        defaultParams.lora_adapter.push_back({lora_chars, lora_scaled});
        defaultParams.lora_base = lora_base_chars;
        defaultParams.use_mmap = false;
    }

    defaultParams.rope_freq_base = rope_freq_base;
    defaultParams.rope_freq_scale = rope_freq_scale;

    auto llama = new rnllama::llama_rn_context();
    bool is_model_loaded = llama->loadModel(defaultParams);

    LOGI("[RNLlama] is_model_loaded %s", (is_model_loaded ? "true" : "false"));
    if (is_model_loaded) {
      context_map[(long) llama->ctx] = llama;
    } else {
      llama_free(llama->ctx);
    }

    env->ReleaseStringUTFChars(model_path_str, model_path_chars);
    env->ReleaseStringUTFChars(lora_str, lora_chars);
    env->ReleaseStringUTFChars(lora_base_str, lora_base_chars);

    return reinterpret_cast<jlong>(llama->ctx);
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
    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    auto result = createWriteableMap(env);
    size_t n_token_count_out = 0;
    llama->embd.resize(llama->params.n_ctx);
    if (!llama_load_session_file(llama->ctx, path_chars, llama->embd.data(), llama->embd.capacity(), &n_token_count_out)) {
      env->ReleaseStringUTFChars(path, path_chars);

      putString(env, result, "error", "Failed to load session");
      return reinterpret_cast<jobject>(result);
    }
    llama->embd.resize(n_token_count_out);
    env->ReleaseStringUTFChars(path, path_chars);

    const std::string text = rnllama::tokens_to_str(llama->ctx, llama->embd.cbegin(), llama->embd.cend());
    putInt(env, result, "tokens_loaded", n_token_count_out);
    putString(env, result, "prompt", text.c_str());
    return reinterpret_cast<jobject>(result);
}

JNIEXPORT jint JNICALL
Java_com_rnllama_LlamaContext_saveSession(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring path,
    jint size
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *path_chars = env->GetStringUTFChars(path, nullptr);

    std::vector<llama_token> session_tokens = llama->embd;
    int default_size = session_tokens.size();
    int save_size = size > 0 && size <= default_size ? size : default_size;
    if (!llama_save_session_file(llama->ctx, path_chars, session_tokens.data(), save_size)) {
      env->ReleaseStringUTFChars(path, path_chars);
      return -1;
    }

    env->ReleaseStringUTFChars(path, path_chars);
    return session_tokens.size();
}

static inline jobject tokenProbsToMap(
  JNIEnv *env,
  rnllama::llama_rn_context *llama,
  std::vector<rnllama::completion_token_output> probs
) {
    auto result = createWritableArray(env);
    for (const auto &prob : probs) {
        auto probsForToken = createWritableArray(env);
        for (const auto &p : prob.probs) {
            std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, p.tok);
            auto probResult = createWriteableMap(env);
            putString(env, probResult, "tok_str", tokStr.c_str());
            putDouble(env, probResult, "prob", p.prob);
            pushMap(env, probsForToken, probResult);
        }
        std::string tokStr = rnllama::tokens_to_output_formatted_string(llama->ctx, prob.tok);
        auto tokenResult = createWriteableMap(env);
        putString(env, tokenResult, "content", tokStr.c_str());
        putArray(env, tokenResult, "probs", probsForToken);
        pushMap(env, result, tokenResult);
    }
    return result;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_doCompletion(
    JNIEnv *env,
    jobject thiz,
    jlong context_ptr,
    jstring prompt,
    jstring grammar,
    jfloat temperature,
    jint n_threads,
    jint n_predict,
    jint n_probs,
    jint penalty_last_n,
    jfloat penalty_repeat,
    jfloat penalty_freq,
    jfloat penalty_present,
    jfloat mirostat,
    jfloat mirostat_tau,
    jfloat mirostat_eta,
    jboolean penalize_nl,
    jint top_k,
    jfloat top_p,
    jfloat min_p,
    jfloat tfs_z,
    jfloat typical_p,
    jint seed,
    jobjectArray stop,
    jboolean ignore_eos,
    jobjectArray logit_bias,
    jobject partial_completion_callback
) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    llama->rewind();

    llama_reset_timings(llama->ctx);

    llama->params.prompt = env->GetStringUTFChars(prompt, nullptr);
    llama->params.seed = seed;

    int max_threads = std::thread::hardware_concurrency();
    // Use 2 threads by default on 4-core devices, 4 threads on more cores
    int default_n_threads = max_threads == 4 ? 2 : min(4, max_threads);
    llama->params.n_threads = n_threads > 0 ? n_threads : default_n_threads;

    llama->params.n_predict = n_predict;
    llama->params.ignore_eos = ignore_eos;

    auto & sparams = llama->params.sparams;
    sparams.temp = temperature;
    sparams.penalty_last_n = penalty_last_n;
    sparams.penalty_repeat = penalty_repeat;
    sparams.penalty_freq = penalty_freq;
    sparams.penalty_present = penalty_present;
    sparams.mirostat = mirostat;
    sparams.mirostat_tau = mirostat_tau;
    sparams.mirostat_eta = mirostat_eta;
    sparams.penalize_nl = penalize_nl;
    sparams.top_k = top_k;
    sparams.top_p = top_p;
    sparams.min_p = min_p;
    sparams.tfs_z = tfs_z;
    sparams.typical_p = typical_p;
    sparams.n_probs = n_probs;
    sparams.grammar = env->GetStringUTFChars(grammar, nullptr);

    sparams.logit_bias.clear();
    if (ignore_eos) {
        sparams.logit_bias[llama_token_eos(llama->model)] = -INFINITY;
    }

    const int n_vocab = llama_n_vocab(llama_get_model(llama->ctx));
    jsize logit_bias_len = env->GetArrayLength(logit_bias);

    for (jsize i = 0; i < logit_bias_len; i++) {
        jdoubleArray el = (jdoubleArray) env->GetObjectArrayElement(logit_bias, i);
        if (el && env->GetArrayLength(el) == 2) {
            jdouble* doubleArray = env->GetDoubleArrayElements(el, 0);

            llama_token tok = static_cast<llama_token>(doubleArray[0]);
            if (tok >= 0 && tok < n_vocab) {
                if (doubleArray[1] != 0) {  // If the second element is not false (0)
                    sparams.logit_bias[tok] = doubleArray[1];
                } else {
                    sparams.logit_bias[tok] = -INFINITY;
                }
            }

            env->ReleaseDoubleArrayElements(el, doubleArray, 0);
        }
        env->DeleteLocalRef(el);
    }

    llama->params.antiprompt.clear();
    int stop_len = env->GetArrayLength(stop);
    for (int i = 0; i < stop_len; i++) {
        jstring stop_str = (jstring) env->GetObjectArrayElement(stop, i);
        const char *stop_chars = env->GetStringUTFChars(stop_str, nullptr);
        llama->params.antiprompt.push_back(stop_chars);
        env->ReleaseStringUTFChars(stop_str, stop_chars);
    }

    if (!llama->initSampling()) {
        auto result = createWriteableMap(env);
        putString(env, result, "error", "Failed to initialize sampling");
        return reinterpret_cast<jobject>(result);
    }
    llama->beginCompletion();
    llama->loadPrompt();

    size_t sent_count = 0;
    size_t sent_token_probs_index = 0;

    while (llama->has_next_token && !llama->is_interrupted) {
        const rnllama::completion_token_output token_with_probs = llama->doCompletion();
        if (token_with_probs.tok == -1 || llama->multibyte_pending > 0) {
            continue;
        }
        const std::string token_text = llama_token_to_piece(llama->ctx, token_with_probs.tok);

        size_t pos = std::min(sent_count, llama->generated_text.size());

        const std::string str_test = llama->generated_text.substr(pos);
        bool is_stop_full = false;
        size_t stop_pos =
            llama->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
        if (stop_pos != std::string::npos) {
            is_stop_full = true;
            llama->generated_text.erase(
                llama->generated_text.begin() + pos + stop_pos,
                llama->generated_text.end());
            pos = std::min(sent_count, llama->generated_text.size());
        } else {
            is_stop_full = false;
            stop_pos = llama->findStoppingStrings(str_test, token_text.size(),
                rnllama::STOP_PARTIAL);
        }

        if (
            stop_pos == std::string::npos ||
            // Send rest of the text if we are at the end of the generation
            (!llama->has_next_token && !is_stop_full && stop_pos > 0)
        ) {
            const std::string to_send = llama->generated_text.substr(pos, std::string::npos);

            sent_count += to_send.size();

            std::vector<rnllama::completion_token_output> probs_output = {};

            auto tokenResult = createWriteableMap(env);
            putString(env, tokenResult, "token", to_send.c_str());

            if (llama->params.sparams.n_probs > 0) {
              const std::vector<llama_token> to_send_toks = llama_tokenize(llama->ctx, to_send, false);
              size_t probs_pos = std::min(sent_token_probs_index, llama->generated_token_probs.size());
              size_t probs_stop_pos = std::min(sent_token_probs_index + to_send_toks.size(), llama->generated_token_probs.size());
              if (probs_pos < probs_stop_pos) {
                  probs_output = std::vector<rnllama::completion_token_output>(llama->generated_token_probs.begin() + probs_pos, llama->generated_token_probs.begin() + probs_stop_pos);
              }
              sent_token_probs_index = probs_stop_pos;

              putArray(env, tokenResult, "completion_probabilities", tokenProbsToMap(env, llama, probs_output));
            }

            jclass cb_class = env->GetObjectClass(partial_completion_callback);
            jmethodID onPartialCompletion = env->GetMethodID(cb_class, "onPartialCompletion", "(Lcom/facebook/react/bridge/WritableMap;)V");
            env->CallVoidMethod(partial_completion_callback, onPartialCompletion, tokenResult);
        }
    }

    llama_print_timings(llama->ctx);
    llama->is_predicting = false;

    auto result = createWriteableMap(env);
    putString(env, result, "text", llama->generated_text.c_str());
    putArray(env, result, "completion_probabilities", tokenProbsToMap(env, llama, llama->generated_token_probs));
    putInt(env, result, "tokens_predicted", llama->num_tokens_predicted);
    putInt(env, result, "tokens_evaluated", llama->num_prompt_tokens);
    putInt(env, result, "truncated", llama->truncated);
    putInt(env, result, "stopped_eos", llama->stopped_eos);
    putInt(env, result, "stopped_word", llama->stopped_word);
    putInt(env, result, "stopped_limit", llama->stopped_limit);
    putString(env, result, "stopping_word", llama->stopping_word.c_str());
    putInt(env, result, "tokens_cached", llama->n_past);

    const auto timings = llama_get_timings(llama->ctx);
    auto timingsResult = createWriteableMap(env);
    putInt(env, timingsResult, "prompt_n", timings.n_p_eval);
    putInt(env, timingsResult, "prompt_ms", timings.t_p_eval_ms);
    putInt(env, timingsResult, "prompt_per_token_ms", timings.t_p_eval_ms / timings.n_p_eval);
    putDouble(env, timingsResult, "prompt_per_second", 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    putInt(env, timingsResult, "predicted_n", timings.n_eval);
    putInt(env, timingsResult, "predicted_ms", timings.t_eval_ms);
    putInt(env, timingsResult, "predicted_per_token_ms", timings.t_eval_ms / timings.n_eval);
    putDouble(env, timingsResult, "predicted_per_second", 1e3 / timings.t_eval_ms * timings.n_eval);

    putMap(env, result, "timings", timingsResult);

    return reinterpret_cast<jobject>(result);
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_stopCompletion(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    llama->is_interrupted = true;
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_isPredicting(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->is_predicting;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_tokenize(
        JNIEnv *env, jobject thiz, jlong context_ptr, jstring text) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *text_chars = env->GetStringUTFChars(text, nullptr);

    const std::vector<llama_token> toks = llama_tokenize(
        llama->ctx,
        text_chars,
        false
    );

    jobject result = createWritableArray(env);
    for (const auto &tok : toks) {
      pushInt(env, result, tok);
    }

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

    return env->NewStringUTF(text.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_rnllama_LlamaContext_isEmbeddingEnabled(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    return llama->params.embedding;
}

JNIEXPORT jobject JNICALL
Java_com_rnllama_LlamaContext_embedding(
        JNIEnv *env, jobject thiz, jlong context_ptr, jstring text) {
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];

    const char *text_chars = env->GetStringUTFChars(text, nullptr);

    llama->rewind();

    llama_reset_timings(llama->ctx);

    llama->params.prompt = text_chars;

    llama->params.n_predict = 0;
    llama->beginCompletion();
    llama->loadPrompt();
    llama->doCompletion();

    std::vector<float> embedding = llama->getEmbedding();

    jobject result = createWritableArray(env);

    for (const auto &val : embedding) {
      pushDouble(env, result, (double) val);
    }

    env->ReleaseStringUTFChars(text, text_chars);
    return result;
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
    std::string result = llama->bench(pp, tg, pl, nr);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_rnllama_LlamaContext_freeContext(
        JNIEnv *env, jobject thiz, jlong context_ptr) {
    UNUSED(env);
    UNUSED(thiz);
    auto llama = context_map[(long) context_ptr];
    if (llama->model) {
        llama_free_model(llama->model);
    }
    if (llama->ctx) {
        llama_free(llama->ctx);
    }
    if (llama->ctx_sampling != nullptr)
    {
        llama_sampling_free(llama->ctx_sampling);
    }
    context_map.erase((long) llama->ctx);
}

} // extern "C"
