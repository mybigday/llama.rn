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
#define TAG "JNI"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,     TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,     TAG, __VA_ARGS__)

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

extern "C" {

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
    jboolean memory_f16,
    jstring lora,
    jstring lora_base,
    jfloat rope_freq_base,
    jfloat rope_freq_scale
) {
    UNUSED(thiz);
    struct llama_context *context = nullptr;
    const char *model_path_chars = env->GetStringUTFChars(model_path_str, nullptr);

    gpt_params defaultParams;

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

    defaultParams.memory_f16 = memory_f16;

    // auto lora_str = lora != nullptr ? env->GetStringUTFChars(lora, nullptr) : nullptr;
    // auto lora_base_str = lora_base != nullptr ? env->GetStringUTFChars(lora_base, nullptr) : nullptr;

    // defaultParams.lora_adapter = lora_str;
    // defaultParams.lora_base = lora_base_str;

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
    // if (lora_str != nullptr) {
    //     env->ReleaseStringUTFChars(lora, lora_str);
    // }
    // if (lora_base_str != nullptr) {
    //     env->ReleaseStringUTFChars(lora_base, lora_base_str);
    // }
    if (llama->ctx == nullptr) {
        return 0;
    }
    return reinterpret_cast<jlong>(llama->ctx);
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
    context_map.erase((long) llama->ctx);
}

} // extern "C"
