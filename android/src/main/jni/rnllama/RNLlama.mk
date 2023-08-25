RNLLAMA_LIB_DIR := $(LOCAL_PATH)/../../../../../cpp
LOCAL_LDLIBS    := -landroid -llog

LOCAL_CFLAGS += -DLM_GGML_USE_K_QUANTS
LOCAL_CPPFLAGS += -DLM_GGML_USE_K_QUANTS

# NOTE: If you want to debug the native code, you can uncomment ifneq and endif
# ifneq ($(APP_OPTIM),debug)

# Make the final output library smaller by only keeping the symbols referenced from the app.
LOCAL_CFLAGS += -Ofast -DNDEBUG
LOCAL_CFLAGS += -fvisibility=hidden -fvisibility-inlines-hidden
LOCAL_CFLAGS += -ffunction-sections -fdata-sections
LOCAL_CFLAGS += -pthread
LOCAL_CPPFLAGS += -pthread
LOCAL_LDFLAGS += -Wl,--gc-sections
LOCAL_LDFLAGS += -Wl,--exclude-libs,ALL
LOCAL_LDFLAGS += -flto

# endif

LOCAL_CPP_FEATURES += exceptions

LOCAL_CFLAGS    += -DSTDC_HEADERS -std=c11 -I $(RNLLAMA_LIB_DIR)
LOCAL_CPPFLAGS  += -std=c++11 -I $(RNLLAMA_LIB_DIR)
LOCAL_SRC_FILES := $(RNLLAMA_LIB_DIR)/ggml-alloc.c \
                   $(RNLLAMA_LIB_DIR)/ggml.c \
                   $(RNLLAMA_LIB_DIR)/k_quants.c \
                   $(RNLLAMA_LIB_DIR)/common.cpp \
                   $(RNLLAMA_LIB_DIR)/grammar-parser.cpp \
                   $(RNLLAMA_LIB_DIR)/llama.cpp \
                   $(RNLLAMA_LIB_DIR)/rn-llama.hpp \
                   $(LOCAL_PATH)/jni.cpp
