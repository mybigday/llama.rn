LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE    := librnllama
include $(LOCAL_PATH)/RNLlama.mk
# include $(BUILD_SHARED_LIBRARY)

# ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
# 	include $(CLEAR_VARS)
# 	LOCAL_MODULE    := librnllama_vfpv4
# 	include $(LOCAL_PATH)/RNLlama.mk
# 	# Allow building NEON FMA code.
# 	# https://android.googlesource.com/platform/ndk/+/master/sources/android/cpufeatures/cpu-features.h
# 	LOCAL_CFLAGS += -mfpu=neon-vfpv4
# 	include $(BUILD_SHARED_LIBRARY)
# endif

ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
	include $(CLEAR_VARS)
	LOCAL_MODULE    := librnllama_arm64
	include $(LOCAL_PATH)/RNLlama.mk
	LOCAL_CFLAGS += -mcpu=native
  LOCAL_CPPFLAGS += -mcpu=native
	include $(BUILD_SHARED_LIBRARY)
endif

