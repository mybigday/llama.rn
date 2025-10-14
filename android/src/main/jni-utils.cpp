#include "jni-utils.h"
#include <android/log.h>
#include <cstring>

#define TAG "RNLLAMA_JNI_UTILS"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace rnbridge {

// Sanitize UTF-8 string for JNI NewStringUTF
// Replaces invalid UTF-8 sequences with '?' to prevent JNI errors
std::string sanitize_utf8_for_jni(const char* text) {
    if (!text) return "";

    std::string result;
    result.reserve(strlen(text));

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(text);
    size_t i = 0;

    while (bytes[i] != 0) {
        unsigned char c = bytes[i];

        // ASCII (0x00-0x7F)
        if (c <= 0x7F) {
            result += static_cast<char>(c);
            i++;
        }
        // 2-byte sequence (0xC0-0xDF)
        else if ((c & 0xE0) == 0xC0) {
            if (bytes[i+1] != 0 && (bytes[i+1] & 0xC0) == 0x80) {
                result += static_cast<char>(bytes[i]);
                result += static_cast<char>(bytes[i+1]);
                i += 2;
            } else {
                result += '?';  // Invalid sequence
                i++;
            }
        }
        // 3-byte sequence (0xE0-0xEF)
        else if ((c & 0xF0) == 0xE0) {
            if (bytes[i+1] != 0 && (bytes[i+1] & 0xC0) == 0x80 &&
                bytes[i+2] != 0 && (bytes[i+2] & 0xC0) == 0x80) {
                result += static_cast<char>(bytes[i]);
                result += static_cast<char>(bytes[i+1]);
                result += static_cast<char>(bytes[i+2]);
                i += 3;
            } else {
                result += '?';  // Invalid sequence
                i++;
            }
        }
        // 4-byte sequence (0xF0-0xF7)
        else if ((c & 0xF8) == 0xF0) {
            if (bytes[i+1] != 0 && (bytes[i+1] & 0xC0) == 0x80 &&
                bytes[i+2] != 0 && (bytes[i+2] & 0xC0) == 0x80 &&
                bytes[i+3] != 0 && (bytes[i+3] & 0xC0) == 0x80) {
                result += static_cast<char>(bytes[i]);
                result += static_cast<char>(bytes[i+1]);
                result += static_cast<char>(bytes[i+2]);
                result += static_cast<char>(bytes[i+3]);
                i += 4;
            } else {
                result += '?';  // Invalid sequence
                i++;
            }
        }
        // Invalid start byte
        else {
            result += '?';
            i++;
        }
    }

    return result;
}

using namespace internal;

// Initialize cached class references - must be called during JNI_OnLoad
bool initialize(JNIEnv* env) {
    jclass argumentsClass;
    jclass writableMapClass;
    jclass writableArrayClass;

    // Cache Arguments class and methods
    argumentsClass = env->FindClass("com/facebook/react/bridge/Arguments");
    if (argumentsClass == nullptr) {
        LOGE("initialize: Failed to find Arguments class");
        return false;
    }

    g_ArgumentsClass = reinterpret_cast<jclass>(env->NewGlobalRef(argumentsClass));
    env->DeleteLocalRef(argumentsClass);

    if (g_ArgumentsClass == nullptr) {
        LOGE("initialize: Failed to create global ref for Arguments class");
        return false;
    }

    g_createMapMethod = env->GetStaticMethodID(g_ArgumentsClass, "createMap", "()Lcom/facebook/react/bridge/WritableMap;");
    if (g_createMapMethod == nullptr) {
        LOGE("initialize: Failed to find createMap method");
        goto cleanup;
    }

    g_createArrayMethod = env->GetStaticMethodID(g_ArgumentsClass, "createArray", "()Lcom/facebook/react/bridge/WritableArray;");
    if (g_createArrayMethod == nullptr) {
        LOGE("initialize: Failed to find createArray method");
        goto cleanup;
    }

    // Cache WritableMap class and methods
    writableMapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
    if (writableMapClass == nullptr) {
        LOGE("initialize: Failed to find WritableMap class");
        goto cleanup;
    }

    g_WritableMapClass = reinterpret_cast<jclass>(env->NewGlobalRef(writableMapClass));
    env->DeleteLocalRef(writableMapClass);

    if (g_WritableMapClass == nullptr) {
        LOGE("initialize: Failed to create global ref for WritableMap class");
        goto cleanup;
    }

    g_putStringMethod = env->GetMethodID(g_WritableMapClass, "putString", "(Ljava/lang/String;Ljava/lang/String;)V");
    g_putIntMethod = env->GetMethodID(g_WritableMapClass, "putInt", "(Ljava/lang/String;I)V");
    g_putDoubleMethod = env->GetMethodID(g_WritableMapClass, "putDouble", "(Ljava/lang/String;D)V");
    g_putBooleanMethod = env->GetMethodID(g_WritableMapClass, "putBoolean", "(Ljava/lang/String;Z)V");
    g_putMapMethod = env->GetMethodID(g_WritableMapClass, "putMap", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableMap;)V");
    g_putArrayMethod = env->GetMethodID(g_WritableMapClass, "putArray", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableArray;)V");

    if (g_putStringMethod == nullptr || g_putIntMethod == nullptr || g_putDoubleMethod == nullptr ||
        g_putBooleanMethod == nullptr || g_putMapMethod == nullptr || g_putArrayMethod == nullptr) {
        LOGE("initialize: Failed to find WritableMap methods");
        goto cleanup;
    }

    // Cache WritableArray class and methods
    writableArrayClass = env->FindClass("com/facebook/react/bridge/WritableArray");
    if (writableArrayClass == nullptr) {
        LOGE("initialize: Failed to find WritableArray class");
        goto cleanup;
    }

    g_WritableArrayClass = reinterpret_cast<jclass>(env->NewGlobalRef(writableArrayClass));
    env->DeleteLocalRef(writableArrayClass);

    if (g_WritableArrayClass == nullptr) {
        LOGE("initialize: Failed to create global ref for WritableArray class");
        goto cleanup;
    }

    g_pushIntMethod = env->GetMethodID(g_WritableArrayClass, "pushInt", "(I)V");
    g_pushDoubleMethod = env->GetMethodID(g_WritableArrayClass, "pushDouble", "(D)V");
    g_pushStringMethod = env->GetMethodID(g_WritableArrayClass, "pushString", "(Ljava/lang/String;)V");
    g_pushMapMethod = env->GetMethodID(g_WritableArrayClass, "pushMap", "(Lcom/facebook/react/bridge/ReadableMap;)V");

    if (g_pushIntMethod == nullptr || g_pushDoubleMethod == nullptr ||
        g_pushStringMethod == nullptr || g_pushMapMethod == nullptr) {
        LOGE("initialize: Failed to find WritableArray methods");
        goto cleanup;
    }

    LOGI("Successfully cached React Native class references");
    return true;

cleanup:
    cleanup(env);
    return false;
}

// Cleanup cached class references - called during JNI_OnUnload
void cleanup(JNIEnv* env) {
    if (g_ArgumentsClass != nullptr) {
        env->DeleteGlobalRef(g_ArgumentsClass);
        g_ArgumentsClass = nullptr;
    }
    if (g_WritableMapClass != nullptr) {
        env->DeleteGlobalRef(g_WritableMapClass);
        g_WritableMapClass = nullptr;
    }
    if (g_WritableArrayClass != nullptr) {
        env->DeleteGlobalRef(g_WritableArrayClass);
        g_WritableArrayClass = nullptr;
    }
    g_createMapMethod = nullptr;
    g_createArrayMethod = nullptr;
    g_putStringMethod = nullptr;
    g_putIntMethod = nullptr;
    g_putDoubleMethod = nullptr;
    g_putBooleanMethod = nullptr;
    g_putMapMethod = nullptr;
    g_putArrayMethod = nullptr;
    g_pushIntMethod = nullptr;
    g_pushDoubleMethod = nullptr;
    g_pushStringMethod = nullptr;
    g_pushMapMethod = nullptr;
}

// WritableMap creation and manipulation

jobject createMap(JNIEnv *env) {
    if (g_ArgumentsClass == nullptr || g_createMapMethod == nullptr) {
        // Fallback to dynamic lookup if not initialized
        jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
        if (mapClass == nullptr) {
            LOGE("createMap: Failed to find Arguments class");
            return nullptr;
        }
        jmethodID init = env->GetStaticMethodID(mapClass, "createMap", "()Lcom/facebook/react/bridge/WritableMap;");
        if (init == nullptr) {
            LOGE("createMap: Failed to find createMap method");
            return nullptr;
        }
        jobject map = env->CallStaticObjectMethod(mapClass, init);
        return map;
    }

    // Use cached references
    jobject map = env->CallStaticObjectMethod(g_ArgumentsClass, g_createMapMethod);
    return map;
}

void putString(JNIEnv *env, jobject map, const char *key, const char *value) {
    jmethodID putStringMethod = g_putStringMethod;
    if (putStringMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putStringMethod = env->GetMethodID(mapClass, "putString", "(Ljava/lang/String;Ljava/lang/String;)V");
    }

    jstring jKey = env->NewStringUTF(key);
    std::string sanitized_value = sanitize_utf8_for_jni(value);
    jstring jValue = env->NewStringUTF(sanitized_value.c_str());

    env->CallVoidMethod(map, putStringMethod, jKey, jValue);
}

void putInt(JNIEnv *env, jobject map, const char *key, int value) {
    jmethodID putIntMethod = g_putIntMethod;
    if (putIntMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putIntMethod = env->GetMethodID(mapClass, "putInt", "(Ljava/lang/String;I)V");
    }

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putIntMethod, jKey, value);
}

void putDouble(JNIEnv *env, jobject map, const char *key, double value) {
    jmethodID putDoubleMethod = g_putDoubleMethod;
    if (putDoubleMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putDoubleMethod = env->GetMethodID(mapClass, "putDouble", "(Ljava/lang/String;D)V");
    }

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putDoubleMethod, jKey, value);
}

void putBoolean(JNIEnv *env, jobject map, const char *key, bool value) {
    jmethodID putBooleanMethod = g_putBooleanMethod;
    if (putBooleanMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putBooleanMethod = env->GetMethodID(mapClass, "putBoolean", "(Ljava/lang/String;Z)V");
    }

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putBooleanMethod, jKey, value);
}

void putMap(JNIEnv *env, jobject map, const char *key, jobject value) {
    jmethodID putMapMethod = g_putMapMethod;
    if (putMapMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putMapMethod = env->GetMethodID(mapClass, "putMap", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableMap;)V");
    }

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putMapMethod, jKey, value);
}

void putArray(JNIEnv *env, jobject map, const char *key, jobject value) {
    jmethodID putArrayMethod = g_putArrayMethod;
    if (putArrayMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableMap");
        putArrayMethod = env->GetMethodID(mapClass, "putArray", "(Ljava/lang/String;Lcom/facebook/react/bridge/ReadableArray;)V");
    }

    jstring jKey = env->NewStringUTF(key);

    env->CallVoidMethod(map, putArrayMethod, jKey, value);
}

// WritableArray creation and manipulation

jobject createArray(JNIEnv *env) {
    if (g_ArgumentsClass == nullptr || g_createArrayMethod == nullptr) {
        // Fallback to dynamic lookup if not initialized
        jclass mapClass = env->FindClass("com/facebook/react/bridge/Arguments");
        if (mapClass == nullptr) {
            LOGE("createArray: Failed to find Arguments class");
            return nullptr;
        }
        jmethodID init = env->GetStaticMethodID(mapClass, "createArray", "()Lcom/facebook/react/bridge/WritableArray;");
        if (init == nullptr) {
            LOGE("createArray: Failed to find createArray method");
            return nullptr;
        }
        jobject map = env->CallStaticObjectMethod(mapClass, init);
        return map;
    }

    // Use cached references
    jobject map = env->CallStaticObjectMethod(g_ArgumentsClass, g_createArrayMethod);
    return map;
}

void pushInt(JNIEnv *env, jobject arr, int value) {
    jmethodID pushIntMethod = g_pushIntMethod;
    if (pushIntMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
        pushIntMethod = env->GetMethodID(mapClass, "pushInt", "(I)V");
    }

    env->CallVoidMethod(arr, pushIntMethod, value);
}

void pushDouble(JNIEnv *env, jobject arr, double value) {
    jmethodID pushDoubleMethod = g_pushDoubleMethod;
    if (pushDoubleMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
        pushDoubleMethod = env->GetMethodID(mapClass, "pushDouble", "(D)V");
    }

    env->CallVoidMethod(arr, pushDoubleMethod, value);
}

void pushString(JNIEnv *env, jobject arr, const char *value) {
    jmethodID pushStringMethod = g_pushStringMethod;
    if (pushStringMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
        pushStringMethod = env->GetMethodID(mapClass, "pushString", "(Ljava/lang/String;)V");
    }

    std::string sanitized_value = sanitize_utf8_for_jni(value);
    jstring jValue = env->NewStringUTF(sanitized_value.c_str());
    env->CallVoidMethod(arr, pushStringMethod, jValue);
}

void pushMap(JNIEnv *env, jobject arr, jobject value) {
    jmethodID pushMapMethod = g_pushMapMethod;
    if (pushMapMethod == nullptr) {
        jclass mapClass = env->FindClass("com/facebook/react/bridge/WritableArray");
        pushMapMethod = env->GetMethodID(mapClass, "pushMap", "(Lcom/facebook/react/bridge/ReadableMap;)V");
    }

    env->CallVoidMethod(arr, pushMapMethod, value);
}

} // namespace rnbridge
