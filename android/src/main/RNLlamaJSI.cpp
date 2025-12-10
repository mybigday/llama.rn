#include "jsi/RNLlamaJSI.h"
#include <fbjni/fbjni.h>
#include <ReactCommon/CallInvokerHolder.h>
#include "llama.h"
#include "llama-impl.h"
#include <android/log.h>
#include <string>
#include <cstdint>

extern "C" JNIEXPORT void JNICALL

Java_com_rnllama_RNLlamaModule_installJSIBindings(
    JNIEnv *env,
    jobject thiz,
    jlong runtimePtr,
    jobject callInvokerHolder
) {
    if (runtimePtr == 0 || callInvokerHolder == nullptr) {
        return;
    }

    auto runtime = reinterpret_cast<facebook::jsi::Runtime*>(runtimePtr);

    // Unwrap CallInvokerHolder
    auto holder = facebook::jni::alias_ref<facebook::react::CallInvokerHolder::javaobject>{
        reinterpret_cast<facebook::react::CallInvokerHolder::javaobject>(callInvokerHolder)
    };
    auto callInvoker = holder->cthis()->getCallInvoker();
    if (!callInvoker) {
        return;
    }

    callInvoker->invokeAsync([runtime, callInvoker]() {
        rnllama_jsi::installJSIBindings(*runtime, callInvoker);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_rnllama_RNLlama_nativeSetLoadedLibrary(JNIEnv *env, jclass /*clazz*/, jstring name) {
    const char *chars = name ? env->GetStringUTFChars(name, nullptr) : nullptr;
    std::string libName = chars ? std::string(chars) : std::string();
    rnllama_jsi::setAndroidLoadedLibrary(libName);
    if (chars) {
        env->ReleaseStringUTFChars(name, chars);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_rnllama_RNLlamaModule_cleanupJSIBindings(JNIEnv *env, jobject /*thiz*/) {
    (void) env;
    rnllama_jsi::cleanupJSIBindings();
}
