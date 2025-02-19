#include <jni.h>

// ReadableMap utils

namespace readablearray {

int size(JNIEnv *env, jobject readableArray) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID sizeMethod = env->GetMethodID(arrayClass, "size", "()I");
    return env->CallIntMethod(readableArray, sizeMethod);
}

jobject getMap(JNIEnv *env, jobject readableArray, int index) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID getMapMethod = env->GetMethodID(arrayClass, "getMap", "(I)Lcom/facebook/react/bridge/ReadableMap;");
    return env->CallObjectMethod(readableArray, getMapMethod, index);
}

jstring getString(JNIEnv *env, jobject readableArray, int index) {
    jclass arrayClass = env->GetObjectClass(readableArray);
    jmethodID getStringMethod = env->GetMethodID(arrayClass, "getString", "(I)Ljava/lang/String;");
    return (jstring) env->CallObjectMethod(readableArray, getStringMethod, index);
}

// Other methods not used yet

}

namespace readablemap {

bool hasKey(JNIEnv *env, jobject readableMap, const char *key) {
    jclass mapClass = env->GetObjectClass(readableMap);
    jmethodID hasKeyMethod = env->GetMethodID(mapClass, "hasKey", "(Ljava/lang/String;)Z");
    jstring jKey = env->NewStringUTF(key);
    jboolean result = env->CallBooleanMethod(readableMap, hasKeyMethod, jKey);
    env->DeleteLocalRef(jKey);
    return result;
}

int getInt(JNIEnv *env, jobject readableMap, const char *key, jint defaultValue) {
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

bool getBool(JNIEnv *env, jobject readableMap, const char *key, jboolean defaultValue) {
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

long getLong(JNIEnv *env, jobject readableMap, const char *key, jlong defaultValue) {
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

float getFloat(JNIEnv *env, jobject readableMap, const char *key, jfloat defaultValue) {
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

jstring getString(JNIEnv *env, jobject readableMap, const char *key, jstring defaultValue) {
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

}
