#ifndef JNI_UTILS_H
#define JNI_UTILS_H

#include <jni.h>
#include <string>

// WritableMap and WritableArray utilities with cached class references

namespace rnbridge {

// Sanitize UTF-8 string for JNI NewStringUTF
// Replaces invalid UTF-8 sequences with '?' to prevent JNI errors
std::string sanitize_utf8_for_jni(const char* text);

// Global references to React Native classes (cached during initialization)
namespace internal {
    static jclass g_ArgumentsClass = nullptr;
    static jmethodID g_createMapMethod = nullptr;
    static jmethodID g_createArrayMethod = nullptr;

    static jclass g_WritableMapClass = nullptr;
    static jmethodID g_putStringMethod = nullptr;
    static jmethodID g_putIntMethod = nullptr;
    static jmethodID g_putDoubleMethod = nullptr;
    static jmethodID g_putBooleanMethod = nullptr;
    static jmethodID g_putMapMethod = nullptr;
    static jmethodID g_putArrayMethod = nullptr;

    static jclass g_WritableArrayClass = nullptr;
    static jmethodID g_pushIntMethod = nullptr;
    static jmethodID g_pushDoubleMethod = nullptr;
    static jmethodID g_pushStringMethod = nullptr;
    static jmethodID g_pushMapMethod = nullptr;
}

// Initialize cached class references - must be called during JNI_OnLoad
bool initialize(JNIEnv* env);

// Cleanup cached class references - called during JNI_OnUnload
void cleanup(JNIEnv* env);

// WritableMap creation and manipulation
jobject createMap(JNIEnv *env);
void putString(JNIEnv *env, jobject map, const char *key, const char *value);
void putInt(JNIEnv *env, jobject map, const char *key, int value);
void putDouble(JNIEnv *env, jobject map, const char *key, double value);
void putBoolean(JNIEnv *env, jobject map, const char *key, bool value);
void putMap(JNIEnv *env, jobject map, const char *key, jobject value);
void putArray(JNIEnv *env, jobject map, const char *key, jobject value);

// WritableArray creation and manipulation
jobject createArray(JNIEnv *env);
void pushInt(JNIEnv *env, jobject arr, int value);
void pushDouble(JNIEnv *env, jobject arr, double value);
void pushString(JNIEnv *env, jobject arr, const char *value);
void pushMap(JNIEnv *env, jobject arr, jobject value);

} // namespace rnbridge

// ReadableMap utils

namespace readablearray {

inline int size(JNIEnv *env, jobject readableArray) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID sizeMethod = env->GetMethodID(arrayClass, "size", "()I");
    return env->CallIntMethod(readableArray, sizeMethod);
}

inline jobject getMap(JNIEnv *env, jobject readableArray, int index) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID getMapMethod = env->GetMethodID(arrayClass, "getMap", "(I)Lcom/facebook/react/bridge/ReadableMap;");
    return env->CallObjectMethod(readableArray, getMapMethod, index);
}

inline jstring getString(JNIEnv *env, jobject readableArray, int index) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID getStringMethod = env->GetMethodID(arrayClass, "getString", "(I)Ljava/lang/String;");
    return (jstring) env->CallObjectMethod(readableArray, getStringMethod, index);
}

// Other methods not used yet

}

namespace readablemap {

inline bool hasKey(JNIEnv *env, jobject readableMap, const char *key) {
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID hasKeyMethod = env->GetMethodID(mapClass, "hasKey", "(Ljava/lang/String;)Z");
    jstring jKey = env->NewStringUTF(key);
    jboolean result = env->CallBooleanMethod(readableMap, hasKeyMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline int getInt(JNIEnv *env, jobject readableMap, const char *key, jint defaultValue) {
    if (!hasKey(env, readableMap, key)) {
        return defaultValue;
    }
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getIntMethod = env->GetMethodID(mapClass, "getInt", "(Ljava/lang/String;)I");
    jstring jKey = env->NewStringUTF(key);
    jint result = env->CallIntMethod(readableMap, getIntMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline bool getBool(JNIEnv *env, jobject readableMap, const char *key, jboolean defaultValue) {
    if (!hasKey(env, readableMap, key)) {
        return defaultValue;
    }
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getBoolMethod = env->GetMethodID(mapClass, "getBoolean", "(Ljava/lang/String;)Z");
    jstring jKey = env->NewStringUTF(key);
    jboolean result = env->CallBooleanMethod(readableMap, getBoolMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline long getLong(JNIEnv *env, jobject readableMap, const char *key, jlong defaultValue) {
    if (!hasKey(env, readableMap, key)) {
        return defaultValue;
    }
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getLongMethod = env->GetMethodID(mapClass, "getLong", "(Ljava/lang/String;)J");
    jstring jKey = env->NewStringUTF(key);
    jlong result = env->CallLongMethod(readableMap, getLongMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline float getFloat(JNIEnv *env, jobject readableMap, const char *key, jfloat defaultValue) {
    if (!hasKey(env, readableMap, key)) {
        return defaultValue;
    }
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getFloatMethod = env->GetMethodID(mapClass, "getDouble", "(Ljava/lang/String;)D");
    jstring jKey = env->NewStringUTF(key);
    jfloat result = env->CallDoubleMethod(readableMap, getFloatMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline jstring getString(JNIEnv *env, jobject readableMap, const char *key, jstring defaultValue) {
    if (!hasKey(env, readableMap, key)) {
        return defaultValue;
    }
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getStringMethod = env->GetMethodID(mapClass, "getString", "(Ljava/lang/String;)Ljava/lang/String;");
    jstring jKey = env->NewStringUTF(key);
    jstring result = (jstring) env->CallObjectMethod(readableMap, getStringMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

inline jobject getArray(JNIEnv *env, jobject readableMap, const char *key) {
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID getArrayMethod = env->GetMethodID(mapClass, "getArray", "(Ljava/lang/String;)Lcom/facebook/react/bridge/ReadableArray;");
    jstring jKey = env->NewStringUTF(key);
    jobject result = env->CallObjectMethod(readableMap, getArrayMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

} // namespace readablemap

namespace writablemap {

// Convenience wrappers for rnbridge namespace functions
// These maintain backward compatibility with existing code
static inline jobject createWriteableMap(JNIEnv *env) {
  return rnbridge::createMap(env);
}

static inline void putString(JNIEnv *env, jobject map, const char *key, const char *value) {
  rnbridge::putString(env, map, key, value);
}

static inline void putInt(JNIEnv *env, jobject map, const char *key, int value) {
  rnbridge::putInt(env, map, key, value);
}

static inline void putDouble(JNIEnv *env, jobject map, const char *key, double value) {
  rnbridge::putDouble(env, map, key, value);
}

static inline void putBoolean(JNIEnv *env, jobject map, const char *key, bool value) {
  rnbridge::putBoolean(env, map, key, value);
}

static inline void putMap(JNIEnv *env, jobject map, const char *key, jobject value) {
  rnbridge::putMap(env, map, key, value);
}

static inline void putArray(JNIEnv *env, jobject map, const char *key, jobject value) {
  rnbridge::putArray(env, map, key, value);
}

}

namespace writablearray {

static inline jobject createWritableArray(JNIEnv *env) {
  return rnbridge::createArray(env);
}

static inline void pushInt(JNIEnv *env, jobject arr, int value) {
  rnbridge::pushInt(env, arr, value);
}

static inline void pushDouble(JNIEnv *env, jobject arr, double value) {
  rnbridge::pushDouble(env, arr, value);
}

static inline void pushString(JNIEnv *env, jobject arr, const char *value) {
  rnbridge::pushString(env, arr, value);
}

static inline void pushMap(JNIEnv *env, jobject arr, jobject value) {
  rnbridge::pushMap(env, arr, value);
}

}

#endif // JNI_UTILS_H
