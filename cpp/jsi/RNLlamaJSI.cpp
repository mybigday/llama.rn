#include "RNLlamaJSI.h"
#include "JSIContext.h"
#include "ThreadPool.h"
#include "JSIUtils.h"
#include "JSIParams.h"
#include "JSIHelpers.h"
#include "JSISession.h"
#include "JSICompletion.h"
#include "JSIRequestManager.h"
#include "JSITaskManager.h"
#include "JSINativeHeaders.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <cstring>
#endif

using namespace facebook;
using json = nlohmann::ordered_json;

// Consolidated logging function
enum class LogLevel { LOG_DEBUG, LOG_INFO, LOG_ERROR };

static void log(LogLevel level, const char* format, ...) {
    va_list args;
    va_start(args, format);

#if defined(__ANDROID__)
    int androidLevel = (level == LogLevel::LOG_DEBUG) ? ANDROID_LOG_DEBUG :
                      (level == LogLevel::LOG_INFO) ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR;
    __android_log_vprint(androidLevel, "RNWhisperJSI", format, args);
#else
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    const char* levelStr = (level == LogLevel::LOG_DEBUG) ? "DEBUG" :
                          (level == LogLevel::LOG_INFO) ? "INFO" : "ERROR";
    printf("RNWhisperJSI %s: %s\n", levelStr, buffer);
#endif

    va_end(args);
}

#define logInfo(format, ...) log(LogLevel::LOG_INFO, format, ##__VA_ARGS__)
#define logError(format, ...) log(LogLevel::LOG_ERROR, format, ##__VA_ARGS__)
#define logDebug(format, ...) log(LogLevel::LOG_DEBUG, format, ##__VA_ARGS__)

static std::once_flag backend_init_once;

#if defined(__ANDROID__)
static bool shouldExcludeHexagonDevice(lm_ggml_backend_dev_t dev) {
#if defined(LM_GGML_USE_HEXAGON)
    const char *dev_name = lm_ggml_backend_dev_name(dev);
    if (dev_name != nullptr && strncmp(dev_name, "HTP", 3) == 0) {
        return true;
    }
#else
    (void) dev;
#endif
    return false;
}

static std::vector<lm_ggml_backend_dev_t> getFilteredDefaultDevices() {
    std::vector<lm_ggml_backend_dev_t> rpc_servers;
    std::vector<lm_ggml_backend_dev_t> gpus;
    std::vector<lm_ggml_backend_dev_t> igpus;

    for (size_t i = 0; i < lm_ggml_backend_dev_count(); ++i) {
        lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
        if (shouldExcludeHexagonDevice(dev)) {
            continue;
        }

        switch (lm_ggml_backend_dev_type(dev)) {
            case LM_GGML_BACKEND_DEVICE_TYPE_CPU:
            case LM_GGML_BACKEND_DEVICE_TYPE_ACCEL:
                break;
            case LM_GGML_BACKEND_DEVICE_TYPE_GPU: {
                lm_ggml_backend_reg_t reg = lm_ggml_backend_dev_backend_reg(dev);
                const char *reg_name = reg ? lm_ggml_backend_reg_name(reg) : nullptr;
                if (reg_name != nullptr && strcmp(reg_name, "RPC") == 0) {
                    rpc_servers.push_back(dev);
                } else {
                    lm_ggml_backend_dev_props props;
                    lm_ggml_backend_dev_get_props(dev, &props);
                    auto it = std::find_if(gpus.begin(), gpus.end(), [&props](lm_ggml_backend_dev_t other) {
                        lm_ggml_backend_dev_props other_props;
                        lm_ggml_backend_dev_get_props(other, &other_props);
                        return props.device_id != nullptr &&
                               other_props.device_id != nullptr &&
                               strcmp(props.device_id, other_props.device_id) == 0;
                    });

                    if (it == gpus.end()) {
                        gpus.push_back(dev);
                    }
                }
                break;
            }
            case LM_GGML_BACKEND_DEVICE_TYPE_IGPU:
                igpus.push_back(dev);
                break;
        }
    }

    std::vector<lm_ggml_backend_dev_t> devices;
    devices.insert(devices.end(), rpc_servers.begin(), rpc_servers.end());
    devices.insert(devices.end(), gpus.begin(), gpus.end());

    if (devices.empty()) {
        devices.insert(devices.end(), igpus.begin(), igpus.end());
    }

    if (!devices.empty()) {
        devices.push_back(nullptr);
    }

    return devices;
}
#endif

static std::string stripFileScheme(const std::string& path) {
    const std::string prefix = "file://";
    if (path.rfind(prefix, 0) == 0) {
        return path.substr(prefix.size());
    }
    return path;
}

namespace rnllama_jsi {
    static std::atomic<int64_t> g_context_limit(-1);
#if defined(__ANDROID__)
    static std::string g_android_loaded_library;
#endif
    static std::mutex g_log_mutex;
    static std::weak_ptr<react::CallInvoker> g_log_invoker;
    static std::shared_ptr<jsi::Function> g_log_handler;
    static std::shared_ptr<jsi::Runtime> g_log_runtime;

    struct ProgressCallbackData {
        std::shared_ptr<jsi::Function> callback;
        std::weak_ptr<react::CallInvoker> callInvoker;
        std::shared_ptr<jsi::Runtime> runtime;
        int contextId;
        std::atomic<int> lastProgress{0};
        int progressEvery = 1;
    };

    void setContextLimit(int64_t limit) {
        g_context_limit.store(limit);
    }

#if defined(__ANDROID__)
    void setAndroidLoadedLibrary(const std::string& name) {
        g_android_loaded_library = name;
    }
#endif

    static bool isContextLimitReached() {
        int64_t limit = g_context_limit.load();
        if (limit < 0) {
            return false;
        }
        return g_llamaContexts.size() >= static_cast<size_t>(limit);
    }

    static void ensureBackendInitialized() {
        std::call_once(backend_init_once, []() {
            llama_backend_init();
        });
    }

    static void logToJsCallback(enum lm_ggml_log_level level, const char* text, void* /*data*/) {
        llama_log_callback_default(level, text, nullptr);

        std::shared_ptr<react::CallInvoker> invoker;
        std::shared_ptr<jsi::Function> handler;
        std::shared_ptr<jsi::Runtime> runtime;
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            invoker = g_log_invoker.lock();
            handler = g_log_handler;
            runtime = g_log_runtime;
        }

        if (!invoker || !handler || !runtime) {
            return;
        }

        std::string levelStr = "info";
        switch (level) {
            case LM_GGML_LOG_LEVEL_ERROR: levelStr = "error"; break;
            case LM_GGML_LOG_LEVEL_WARN: levelStr = "warn"; break;
            case LM_GGML_LOG_LEVEL_INFO: levelStr = "info"; break;
            default: break;
        }

        std::string message = text ? text : "";

        invoker->invokeAsync([handler, levelStr, message, runtime]() {
            auto& rt = *runtime;
            handler->call(
                rt,
                jsi::String::createFromUtf8(rt, levelStr),
                jsi::String::createFromUtf8(rt, message)
            );
        });
    }

    // Helper: convert vector<string> to JSI array
    static jsi::Array toJsStringArray(jsi::Runtime& runtime, const std::vector<std::string>& values) {
        jsi::Array arr(runtime, values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            arr.setValueAtIndex(runtime, i, jsi::String::createFromUtf8(runtime, values[i]));
        }
        return arr;
    }

    static jsi::Object createModelDetails(jsi::Runtime& runtime, rnllama::llama_rn_context* ctx) {
        jsi::Object model(runtime);

        char desc[1024];
        llama_model_desc(ctx->model, desc, sizeof(desc));
        model.setProperty(runtime, "desc", jsi::String::createFromUtf8(runtime, desc));
        model.setProperty(runtime, "size", (double)llama_model_size(ctx->model));
        model.setProperty(runtime, "nEmbd", (double)llama_model_n_embd(ctx->model));
        model.setProperty(runtime, "nParams", (double)llama_model_n_params(ctx->model));

        // Metadata
        jsi::Object metadata(runtime);
        int metaCount = llama_model_meta_count(ctx->model);
        for (int i = 0; i < metaCount; ++i) {
            char key[256];
            llama_model_meta_key_by_index(ctx->model, i, key, sizeof(key));
            char val[16384];
            llama_model_meta_val_str_by_index(ctx->model, i, val, sizeof(val));
            metadata.setProperty(runtime, key, jsi::String::createFromUtf8(runtime, val));
        }
        model.setProperty(runtime, "metadata", metadata);

        // Chat template capabilities
        jsi::Object chatTemplates(runtime);
        bool llamaChat = ctx->validateModelChatTemplate(false, nullptr);
        chatTemplates.setProperty(runtime, "llamaChat", llamaChat);

        jsi::Object minja(runtime);
        bool minjaDefault = ctx->validateModelChatTemplate(true, nullptr);
        minja.setProperty(runtime, "default", minjaDefault);

        jsi::Object defaultCaps(runtime);
        if (ctx->templates && ctx->templates->template_default) {
            auto caps = ctx->templates->template_default->original_caps();
            defaultCaps.setProperty(runtime, "tools", caps.supports_tools);
            defaultCaps.setProperty(runtime, "toolCalls", caps.supports_tool_calls);
            defaultCaps.setProperty(runtime, "parallelToolCalls", caps.supports_parallel_tool_calls);
            defaultCaps.setProperty(runtime, "toolResponses", caps.supports_tool_responses);
            defaultCaps.setProperty(runtime, "systemRole", caps.supports_system_role);
            defaultCaps.setProperty(runtime, "toolCallId", caps.supports_tool_call_id);
        } else {
            defaultCaps.setProperty(runtime, "tools", false);
            defaultCaps.setProperty(runtime, "toolCalls", false);
            defaultCaps.setProperty(runtime, "parallelToolCalls", false);
            defaultCaps.setProperty(runtime, "toolResponses", false);
            defaultCaps.setProperty(runtime, "systemRole", false);
            defaultCaps.setProperty(runtime, "toolCallId", false);
        }
        minja.setProperty(runtime, "defaultCaps", defaultCaps);

        bool toolUseSupported = ctx->validateModelChatTemplate(true, "tool_use");
        minja.setProperty(runtime, "toolUse", toolUseSupported);
        if (ctx->templates && ctx->templates->template_tool_use) {
            auto caps = ctx->templates->template_tool_use->original_caps();
            jsi::Object toolUseCaps(runtime);
            toolUseCaps.setProperty(runtime, "tools", caps.supports_tools);
            toolUseCaps.setProperty(runtime, "toolCalls", caps.supports_tool_calls);
            toolUseCaps.setProperty(runtime, "parallelToolCalls", caps.supports_parallel_tool_calls);
            toolUseCaps.setProperty(runtime, "systemRole", caps.supports_system_role);
            toolUseCaps.setProperty(runtime, "toolResponses", caps.supports_tool_responses);
            toolUseCaps.setProperty(runtime, "toolCallId", caps.supports_tool_call_id);
            minja.setProperty(runtime, "toolUseCaps", toolUseCaps);
        }

        chatTemplates.setProperty(runtime, "minja", minja);
        model.setProperty(runtime, "chatTemplates", chatTemplates);

        // Deprecated flag maintained for compatibility
        model.setProperty(runtime, "isChatTemplateSupported", llamaChat);

        return model;
    }

    static std::vector<lm_ggml_backend_dev_t> buildDeviceOverrides(
        const std::vector<std::string>& requestedDevices,
        bool skipGpuDevices,
        bool& anyGpuAvailable
    ) {
        std::vector<lm_ggml_backend_dev_t> selected;
        anyGpuAvailable = false;

        const size_t devCount = lm_ggml_backend_dev_count();
        for (size_t i = 0; i < devCount; ++i) {
            lm_ggml_backend_dev_t dev = lm_ggml_backend_dev_get(i);
            const auto type = lm_ggml_backend_dev_type(dev);
#if TARGET_OS_SIMULATOR
            if (type == LM_GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                continue;
            }
#endif
            const bool isGpuType = type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU;
            if (isGpuType) {
                anyGpuAvailable = true;
            }
            if (skipGpuDevices && isGpuType) {
                continue;
            }

            if (!requestedDevices.empty()) {
                const char* name = lm_ggml_backend_dev_name(dev);
                std::string nameStr = name ? name : "";
                auto it = std::find(requestedDevices.begin(), requestedDevices.end(), nameStr);
                if (it == requestedDevices.end()) {
                    continue;
                }
            }

            selected.push_back(dev);
        }

        if (!selected.empty()) {
            selected.push_back(nullptr);
        }

        return selected;
    }

    void addContext(int contextId, long contextPtr) {
        g_llamaContexts.add(contextId, contextPtr);
    }

    void removeContext(int contextId) {
        g_llamaContexts.remove(contextId);
    }

    rnllama::llama_rn_context* getContextOrThrow(int contextId) {
        long ctxPtr = g_llamaContexts.get(contextId);
        if (!ctxPtr) {
            throw std::runtime_error("Context not found");
        }
        return reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
    }

    void installJSIBindings(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker
    ) {
        auto initContext = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitContext"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                bool isModelAsset = getPropertyAsBool(runtime, params, "is_model_asset", false);

                bool useProgressCallback = getPropertyAsBool(runtime, params, "use_progress_callback", false);
                int progressCallbackEvery = getPropertyAsInt(runtime, params, "progress_callback_every", 1);
                std::shared_ptr<ProgressCallbackData> progressData;
                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    useProgressCallback = true;
                    progressData = std::make_shared<ProgressCallbackData>();
                    progressData->callback = std::make_shared<jsi::Function>(arguments[2].asObject(runtime).asFunction(runtime));
                    progressData->callInvoker = callInvoker;
                    progressData->runtime = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){});
                    progressData->contextId = contextId;
                    progressData->progressEvery = std::max(1, progressCallbackEvery);
                    progressData->lastProgress.store(0);
                } else if (useProgressCallback) {
                    // Progress requested but no callback provided
                    useProgressCallback = false;
                }

                ensureBackendInitialized();

                common_params cparams;
                parseCommonParams(runtime, params, cparams);

#if defined(__APPLE__)
                if (isModelAsset) {
                    cparams.model.path = resolveIosModelPath(cparams.model.path, true);
                }
#endif

                bool skipGpuDevices = getPropertyAsBool(runtime, params, "no_gpu_devices", false);
                if (skipGpuDevices) {
                    cparams.n_gpu_layers = 0;
                }

#if defined(__APPLE__)
                auto metalAvailability = getMetalAvailability(skipGpuDevices);
                std::string appleGpuReason = metalAvailability.available ? "" : metalAvailability.reason;
                if (!metalAvailability.available && !skipGpuDevices) {
                    skipGpuDevices = true;
                    cparams.n_gpu_layers = 0;
                }
#endif

                std::vector<std::string> requestedDevices;
                bool devicesProvided = false;
                if (params.hasProperty(runtime, "devices") && params.getProperty(runtime, "devices").isObject()) {
                    jsi::Array devicesArr = params.getProperty(runtime, "devices").asObject(runtime).asArray(runtime);
                    if (devicesArr.size(runtime) > 0) {
                        devicesProvided = true;
                        for (size_t i = 0; i < devicesArr.size(runtime); ++i) {
                            auto val = devicesArr.getValueAtIndex(runtime, i);
                            if (val.isString()) {
                                requestedDevices.push_back(val.asString(runtime).utf8(runtime));
                            }
                        }
                    }
                }
                bool anyGpuAvailable = false;
                std::vector<lm_ggml_backend_dev_t> overrideDevices;
                if (devicesProvided) {
                    overrideDevices = buildDeviceOverrides(requestedDevices, skipGpuDevices, anyGpuAvailable);
                    if (!overrideDevices.empty()) {
                        cparams.devices = overrideDevices;
                    }
                }
                if (overrideDevices.empty() && !skipGpuDevices) {
#if defined(__ANDROID__)
                    auto defaultDevices = getFilteredDefaultDevices();
                    if (!defaultDevices.empty()) {
                        cparams.devices = defaultDevices;
                        for (auto dev : defaultDevices) {
                            if (dev == nullptr) continue;
                            auto type = lm_ggml_backend_dev_type(dev);
                            if (type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                                anyGpuAvailable = true;
                                break;
                            }
                        }
                    }
#endif
                }

                // Track backend availability when no explicit override was applied
                if (overrideDevices.empty() && anyGpuAvailable == false) {
                    const size_t devCount = lm_ggml_backend_dev_count();
                    for (size_t i = 0; i < devCount; ++i) {
                        auto dev = lm_ggml_backend_dev_get(i);
                        auto type = lm_ggml_backend_dev_type(dev);
                        if (type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                            anyGpuAvailable = true;
                            break;
                        }
                    }
                }

                return createPromiseTask(runtime, callInvoker, [contextId, cparams, skipGpuDevices, anyGpuAvailable, useProgressCallback, progressData
#if defined(__APPLE__)
                    , appleGpuReason
#endif
                ]() mutable -> PromiseResultGenerator {
                    if (isContextLimitReached()) {
                        throw std::runtime_error("Context limit reached");
                    }

                    if (useProgressCallback && progressData && progressData->callback) {
                        cparams.progress_callback = [](float progress, void * user_data) {
                            auto *data = static_cast<ProgressCallbackData *>(user_data);
                            if (!data) {
                                return true;
                            }

                            int percentage = (int) (progress * 100.0f);
                            int last = data->lastProgress.load();
                            if (percentage < 100 && percentage - last < data->progressEvery) {
                                return true;
                            }
                            if (percentage <= last) {
                                return true;
                            }

                            data->lastProgress.store(percentage);

                            auto invoker = data->callInvoker.lock();
                            auto cb = data->callback;
                            auto runtime = data->runtime;
                            if (invoker && cb && runtime) {
                                invoker->invokeAsync([cb, percentage, runtime]() {
                                    auto& rt = *runtime;
                                    cb->call(rt, jsi::Value((double) percentage));
                                });
                            }

                            return true;
                        };
                        cparams.progress_callback_user_data = progressData.get();
                    }

                    auto ctx = new rnllama::llama_rn_context();
                    if (ctx->loadModel(cparams)) {
                         ctx->attachThreadpoolsIfAvailable();

                         if (ctx->params.embedding && llama_model_has_encoder(ctx->model) && llama_model_has_decoder(ctx->model)) {
                             delete ctx;
                             throw std::runtime_error("Embedding is not supported in encoder-decoder models");
                         }

                         if (!ctx->params.lora_adapters.empty()) {
                             int lora_result = ctx->applyLoraAdapters(ctx->params.lora_adapters);
                             if (lora_result != 0) {
                                 delete ctx;
                                 throw std::runtime_error("Failed to apply lora adapters");
                             }
                         }

                         std::vector<std::string> usedDevices;
                         bool gpuEnabled = false;
                         if (ctx->llama_init.model != nullptr) {
                             for (auto dev : ctx->llama_init.model->devices) {
                                 if (dev == nullptr) continue;
                                 const char* used_name = lm_ggml_backend_dev_name(dev);
                                 if (used_name != nullptr) {
                                     usedDevices.push_back(used_name);
                                 }
                                 auto devType = lm_ggml_backend_dev_type(dev);
                                 if (devType == LM_GGML_BACKEND_DEVICE_TYPE_GPU || devType == LM_GGML_BACKEND_DEVICE_TYPE_IGPU) {
                                     gpuEnabled = true;
                                 }
                             }
                         }

                         std::string reasonNoGPU;
#if defined(__APPLE__)
                         const std::string platformReason = appleGpuReason;
#endif
                         if (!gpuEnabled) {
#if defined(__APPLE__)
                             if (!platformReason.empty()) {
                                 reasonNoGPU = platformReason;
                             } else
#endif
                             if (skipGpuDevices) {
                                 reasonNoGPU = "GPU devices disabled by user";
                             } else if (anyGpuAvailable) {
                                 reasonNoGPU = "GPU backend is available but was not selected";
                             } else {
                                 reasonNoGPU = "GPU backend is not available";
                             }
                         }

                         addContext(contextId, (long)ctx);

                         std::string system_info = common_params_get_system_info(ctx->params);

                         return [gpuEnabled, reasonNoGPU, system_info, usedDevices, ctx](jsi::Runtime& rt) {
                             jsi::Object result(rt);
                             result.setProperty(rt, "gpu", gpuEnabled);
                             result.setProperty(rt, "reasonNoGPU", jsi::String::createFromUtf8(rt, reasonNoGPU));
                             result.setProperty(rt, "systemInfo", jsi::String::createFromUtf8(rt, system_info));

                             // Model metadata and chat template capabilities
                             result.setProperty(rt, "model", createModelDetails(rt, ctx));

                             // Maintain shape expected by TypeScript
                             result.setProperty(rt, "devices", toJsStringArray(rt, usedDevices));
                             std::string androidLibName = "";
                             #if defined(__ANDROID__)
                             androidLibName = g_android_loaded_library;
                             #endif
                             result.setProperty(rt, "androidLib", jsi::String::createFromUtf8(rt, androidLibName));
                             return result;
                         };
                    } else {
                        delete ctx;
                        throw std::runtime_error("Failed to load model");
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaInitContext", initContext);

        // ... (modelInfo, getBackendDevicesInfo, loadSession, saveSession, tokenize, detokenize, getFormattedChat from previous)
        auto modelInfo = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaModelInfo"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                std::string path = arguments[0].asString(runtime).utf8(runtime);
                std::vector<std::string> skip;
                if (count > 1 && arguments[1].isObject()) {
                    jsi::Array skipArr = arguments[1].asObject(runtime).asArray(runtime);
                    for (size_t i = 0; i < skipArr.size(runtime); i++) {
                        skip.push_back(skipArr.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                    }
                }

                return createPromiseTask(runtime, callInvoker, [path, skip]() -> PromiseResultGenerator {
                    return [path, skip](jsi::Runtime& rt) {
                        return createModelInfo(rt, path, skip);
                    };
                });
            }
        );
        runtime.global().setProperty(runtime, "llamaModelInfo", modelInfo);

        auto getBackendDevicesInfo = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetBackendDevicesInfo"),
            0,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 return createPromiseTask(runtime, callInvoker, [callInvoker]() -> PromiseResultGenerator {
                     ensureBackendInitialized();

                     std::string info = rnllama::get_backend_devices_info();

                     return [info](jsi::Runtime& rt) {
                         return jsi::String::createFromUtf8(rt, info);
                     };
                 });
            }
        );
        runtime.global().setProperty(runtime, "llamaGetBackendDevicesInfo", getBackendDevicesInfo);

        auto loadSession = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaLoadSession"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string path = arguments[1].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, path]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    return [ctx, path](jsi::Runtime& rt) {
                        return rnllama_jsi::loadSession(rt, ctx, path);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaLoadSession", loadSession);

        auto saveSession = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaSaveSession"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string path = arguments[1].asString(runtime).utf8(runtime);
                int size = (int)arguments[2].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId, path, size]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    int tokens_saved = rnllama_jsi::saveSession(ctx, path, size);
                    return [tokens_saved](jsi::Runtime& rt) {
                        return jsi::Value(tokens_saved);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaSaveSession", saveSession);

        auto tokenize = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaTokenize"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string text = arguments[1].asString(runtime).utf8(runtime);
                std::vector<std::string> mediaPaths;
                if (count > 2 && arguments[2].isObject()) {
                    jsi::Array paths = arguments[2].asObject(runtime).asArray(runtime);
                    for (size_t i = 0; i < paths.size(runtime); ++i) {
                         mediaPaths.push_back(paths.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                    }
                }

                return createPromiseTask(runtime, callInvoker, [contextId, text, mediaPaths]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    auto result = ctx->tokenize(text, mediaPaths);
                    return [result](jsi::Runtime& rt) {
                        return createTokenizeResult(rt, result);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaTokenize", tokenize);

        auto detokenize = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDetokenize"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::vector<llama_token> tokens;
                jsi::Array tokensArr = arguments[1].asObject(runtime).asArray(runtime);
                for (size_t i = 0; i < tokensArr.size(runtime); ++i) {
                    tokens.push_back((llama_token)tokensArr.getValueAtIndex(runtime, i).asNumber());
                }

                return createPromiseTask(runtime, callInvoker, [contextId, tokens]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    std::string text = rnllama::tokens_to_str(ctx->ctx, tokens.cbegin(), tokens.cend());
                    return [text](jsi::Runtime& rt) {
                        return jsi::String::createFromUtf8(rt, text);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDetokenize", detokenize);

        auto getFormattedChat = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetFormattedChat"),
            4,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 int contextId = (int)arguments[0].asNumber();
                 std::string messages = arguments[1].asString(runtime).utf8(runtime);
                 std::string chatTemplate = "";
                 if (count > 2 && arguments[2].isString()) {
                     chatTemplate = arguments[2].asString(runtime).utf8(runtime);
                 }

                 std::string jsonSchema = "";
                 std::string tools = "";
                 bool parallelToolCalls = false;
                 std::string toolChoice = "";
                 bool enableThinking = false;
                 bool addGenerationPrompt = true;
                 std::string nowStr = "";
                 std::map<std::string, std::string> chatTemplateKwargs;
                 bool useJinja = false;

                 if (count > 3 && arguments[3].isObject()) {
                     jsi::Object params = arguments[3].asObject(runtime);
                     useJinja = getPropertyAsBool(runtime, params, "jinja", false);

                     if (useJinja) {
                         jsonSchema = getPropertyAsString(runtime, params, "json_schema");
                         tools = getPropertyAsString(runtime, params, "tools");
                         parallelToolCalls = getPropertyAsBool(runtime, params, "parallel_tool_calls", false);
                         toolChoice = getPropertyAsString(runtime, params, "tool_choice");
                         enableThinking = getPropertyAsBool(runtime, params, "enable_thinking", false);
                         addGenerationPrompt = getPropertyAsBool(runtime, params, "add_generation_prompt", true);
                         nowStr = getPropertyAsString(runtime, params, "now");

                         std::string kwargsStr = getPropertyAsString(runtime, params, "chat_template_kwargs");
                          if (!kwargsStr.empty()) {
                              try {
                                  auto kwargs_json = json::parse(kwargsStr);
                                  for (auto& [key, value] : kwargs_json.items()) {
                                      if (value.is_string()) {
                                          chatTemplateKwargs[key] = value.get<std::string>();
                                      }
                                  }
                              } catch (...) { }
                          }
                     }
                 }

                 return createPromiseTask(runtime, callInvoker, [contextId, messages, chatTemplate, jsonSchema, tools, parallelToolCalls, toolChoice, enableThinking, addGenerationPrompt, nowStr, chatTemplateKwargs, useJinja]() -> PromiseResultGenerator {
                      auto ctx = getContextOrThrow(contextId);
                      if (useJinja) {
                          auto chatParams = ctx->getFormattedChatWithJinja(
                               messages, chatTemplate, jsonSchema, tools, parallelToolCalls,
                               toolChoice, enableThinking, addGenerationPrompt, nowStr, chatTemplateKwargs
                          );

                          return [chatParams](jsi::Runtime& rt) {
                              jsi::Object result(rt);
                              result.setProperty(rt, "prompt", jsi::String::createFromUtf8(rt, chatParams.prompt));
                              result.setProperty(rt, "chat_format", (int)chatParams.format);
                              result.setProperty(rt, "grammar", jsi::String::createFromUtf8(rt, chatParams.grammar));
                              result.setProperty(rt, "grammar_lazy", chatParams.grammar_lazy);
                              result.setProperty(rt, "thinking_forced_open", chatParams.thinking_forced_open);

                              // Preserve the same shape as legacy native bridge
                              result.setProperty(rt, "type", jsi::String::createFromUtf8(rt, "jinja"));

                              jsi::Array preserved(rt, chatParams.preserved_tokens.size());
                              for (size_t i = 0; i < chatParams.preserved_tokens.size(); i++) {
                                  preserved.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, chatParams.preserved_tokens[i]));
                              }
                              result.setProperty(rt, "preserved_tokens", preserved);

                              jsi::Array additionalStops(rt, chatParams.additional_stops.size());
                              for (size_t i = 0; i < chatParams.additional_stops.size(); i++) {
                                  additionalStops.setValueAtIndex(rt, i, jsi::String::createFromUtf8(rt, chatParams.additional_stops[i]));
                              }
                              result.setProperty(rt, "additional_stops", additionalStops);

                              jsi::Array triggers = jsi::Array(rt, chatParams.grammar_triggers.size());
                              for (size_t i = 0; i < chatParams.grammar_triggers.size(); i++) {
                                  jsi::Object trigger(rt);
                                  trigger.setProperty(rt, "type", (int)chatParams.grammar_triggers[i].type);
                                  trigger.setProperty(rt, "value", jsi::String::createFromUtf8(rt, chatParams.grammar_triggers[i].value));
                                  trigger.setProperty(rt, "token", (int)chatParams.grammar_triggers[i].token);
                                  triggers.setValueAtIndex(rt, i, trigger);
                              }
                              result.setProperty(rt, "grammar_triggers", triggers);

                              return result;
                          };
                      } else {
                          std::string prompt = ctx->getFormattedChat(messages, chatTemplate);
                          return [prompt](jsi::Runtime& rt) {
                              return jsi::String::createFromUtf8(rt, prompt);
                          };
                      }
                 }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetFormattedChat", getFormattedChat);

        auto embedding = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaEmbedding"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string text = arguments[1].asString(runtime).utf8(runtime);
                jsi::Object params = arguments[2].asObject(runtime);

                int embd_normalize = 0;
                bool has_embd_normalize = false;
                if (params.hasProperty(runtime, "embd_normalize")) {
                    embd_normalize = getPropertyAsInt(runtime, params, "embd_normalize", 2);
                    has_embd_normalize = true;
                }

                return createPromiseTask(runtime, callInvoker, [contextId, text, embd_normalize, has_embd_normalize]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (!ctx->completion) throw std::runtime_error("Completion not initialized");
                    if (ctx->params.embedding != true) throw std::runtime_error("Embedding is not enabled");
                    if (ctx->completion->is_predicting) throw std::runtime_error("Context is predicting");

                    common_params embdParams = ctx->params;
                    embdParams.embedding = true;
                    embdParams.embd_normalize = has_embd_normalize ? embd_normalize : ctx->params.embd_normalize;

                    ctx->params.prompt = text;
                    ctx->params.n_predict = 0;

                    std::vector<float> result = ctx->completion->embedding(embdParams);

                    return [result](jsi::Runtime& rt) {
                        jsi::Object resultDict(rt);
                        jsi::Array embeddingResult(rt, result.size());
                        for (size_t i = 0; i < result.size(); i++) {
                            embeddingResult.setValueAtIndex(rt, i, (double)result[i]);
                        }
                        resultDict.setProperty(rt, "embedding", embeddingResult);
                        return resultDict;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaEmbedding", embedding);

        auto rerank = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaRerank"),
            4,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string query = arguments[1].asString(runtime).utf8(runtime);
                jsi::Array documentsArr = arguments[2].asObject(runtime).asArray(runtime);
                std::vector<std::string> documents;
                for (size_t i = 0; i < documentsArr.size(runtime); i++) {
                    documents.push_back(documentsArr.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                }
                // params argument ignored for now as per iOS implementation logic (only checks context state)

                return createPromiseTask(runtime, callInvoker, [contextId, query, documents]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (!ctx->completion) throw std::runtime_error("Completion not initialized");
                    if (ctx->params.embedding != true) throw std::runtime_error("Embedding is not enabled");
                    if (ctx->completion->is_predicting) throw std::runtime_error("Context is predicting");

                    std::vector<float> scores = ctx->completion->rerank(query, documents);

                    return [scores](jsi::Runtime& rt) {
                        jsi::Array result(rt, scores.size());
                        for (size_t i = 0; i < scores.size(); i++) {
                            jsi::Object item(rt);
                            item.setProperty(rt, "score", (double)scores[i]);
                            item.setProperty(rt, "index", (int)i);
                            result.setValueAtIndex(rt, i, item);
                        }
                        return result;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaRerank", rerank);

        auto bench = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaBench"),
            5,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int pp = (int)arguments[1].asNumber();
                int tg = (int)arguments[2].asNumber();
                int pl = (int)arguments[3].asNumber();
                int nr = (int)arguments[4].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId, pp, tg, pl, nr]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->completion) return [](jsi::Runtime& rt) { return jsi::String::createFromUtf8(rt, ""); };

                    std::string res = ctx->completion->bench(pp, tg, pl, nr);

                    return [res](jsi::Runtime& rt) {
                        return jsi::String::createFromUtf8(rt, res);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaBench", bench);

        auto completion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaCompletion"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                std::shared_ptr<jsi::Function> onToken;

                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    onToken = std::make_shared<jsi::Function>(arguments[2].asObject(runtime).asFunction(runtime));
                }

                bool emitPartial = getPropertyAsBool(runtime, params, "emit_partial_completion", false);

                auto ctx = getContextOrThrow(contextId);
                if (ctx->completion && ctx->completion->is_predicting) {
                     throw std::runtime_error("Context is busy");
                }

                parseCompletionParams(runtime, params, ctx);

                std::vector<std::string> mediaPaths;
                if (params.hasProperty(runtime, "media_paths")) {
                    jsi::Array paths = params.getProperty(runtime, "media_paths").asObject(runtime).asArray(runtime);
                    for (size_t i = 0; i < paths.size(runtime); i++) {
                        mediaPaths.push_back(paths.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                    }
                }

                int chat_format = getPropertyAsInt(runtime, params, "chat_format", 0);
                std::string reasoningFormatStr = getPropertyAsString(runtime, params, "reasoning_format", "none");
                common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);
                bool thinking_forced_open = getPropertyAsBool(runtime, params, "thinking_forced_open", false);
                std::string prefill_text = getPropertyAsString(runtime, params, "prefill_text");
                std::vector<llama_token> guide_tokens;
                if (params.hasProperty(runtime, "guide_tokens")) {
                    auto guideVal = params.getProperty(runtime, "guide_tokens");
                    if (guideVal.isObject() && guideVal.asObject(runtime).isArray(runtime)) {
                        jsi::Array guideArr = guideVal.asObject(runtime).asArray(runtime);
                        guide_tokens.reserve(guideArr.size(runtime));
                        for (size_t i = 0; i < guideArr.size(runtime); i++) {
                            auto tokVal = guideArr.getValueAtIndex(runtime, i);
                            if (tokVal.isNumber()) {
                                guide_tokens.push_back((llama_token)tokVal.asNumber());
                            }
                        }
                    }
                }

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, onToken, emitPartial, mediaPaths, chat_format, reasoning_format, thinking_forced_open, prefill_text, guide_tokens, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (ctx->completion == nullptr) {
                        throw std::runtime_error("Completion not initialized");
                    }

                    if (!guide_tokens.empty() && ctx->tts_wrapper != nullptr) {
                        ctx->params.vocoder.use_guide_tokens = true;
                        ctx->tts_wrapper->setGuideTokens(guide_tokens);
                    }

                    ctx->completion->rewind();
                    if (!ctx->completion->initSampling()) {
                        throw std::runtime_error("Failed to initialize sampling");
                    }

                    ctx->completion->prefill_text = prefill_text;
                    ctx->completion->beginCompletion(chat_format, reasoning_format, thinking_forced_open);

                    try {
                        if (!mediaPaths.empty() && !ctx->isMultimodalEnabled()) {
                            throw std::runtime_error("Multimodal support not enabled. Call initMultimodal first.");
                        }
                        ctx->completion->loadPrompt(mediaPaths);
                    } catch (const std::exception &e) {
                        ctx->completion->endCompletion();
                        throw std::runtime_error(e.what());
                    }

                    if (ctx->completion->context_full) {
                        ctx->completion->endCompletion();
                        throw std::runtime_error("Context is full");
                    }

                    size_t sent_count = 0;

                    while (ctx->completion->has_next_token && !ctx->completion->is_interrupted) {
                        const rnllama::completion_token_output token_with_probs = ctx->completion->doCompletion();
                        if (token_with_probs.tok == -1 || ctx->completion->incomplete) {
                            continue;
                        }

                        const std::string token_text = common_token_to_piece(ctx->ctx, token_with_probs.tok);
                        size_t pos = std::min(sent_count, ctx->completion->generated_text.size());
                        const std::string str_test = ctx->completion->generated_text.substr(pos);

                        bool is_stop_full = false;
                        size_t stop_pos = ctx->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_FULL);
                        if (stop_pos != std::string::npos) {
                            is_stop_full = true;
                            ctx->completion->generated_text.erase(
                                ctx->completion->generated_text.begin() + pos + stop_pos,
                                ctx->completion->generated_text.end());
                            pos = std::min(sent_count, ctx->completion->generated_text.size());
                        } else {
                             stop_pos = ctx->completion->findStoppingStrings(str_test, token_text.size(), rnllama::STOP_PARTIAL);
                        }

                        if (stop_pos == std::string::npos || (!ctx->completion->has_next_token && !is_stop_full && stop_pos > 0)) {
                            const std::string to_send = ctx->completion->generated_text.substr(pos, std::string::npos);
                            sent_count += to_send.size();

                            if (emitPartial && onToken) {
                                rnllama::completion_token_output output_copy = token_with_probs;
                                output_copy.text = to_send;

                                rnllama::completion_chat_output partial_output;
                                bool has_partial_output = false;
                                try {
                                    partial_output = ctx->completion->parseChatOutput(true);
                                    has_partial_output = true;
                                } catch (...) {
                                    // ignore parse errors for partial output
                                }

                                auto runtime = runtimePtr;
                                if (runtime) {
                                    callInvoker->invokeAsync([onToken, output_copy, ctx, partial_output, has_partial_output, runtime]() {
                                        auto& rt = *runtime;
                                        jsi::Object res = createTokenResult(rt, ctx, output_copy);
                                        if (has_partial_output) {
                                            setChatOutputFields(rt, res, partial_output);
                                        }
                                        onToken->call(rt, res);
                                    });
                                }
                            }
                        }
                    }

                    common_perf_print(ctx->ctx, ctx->completion->ctx_sampling);
                    ctx->completion->endCompletion();

                    return [ctx](jsi::Runtime& rt) {
                        return createCompletionResult(rt, ctx);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaCompletion", completion);

        auto stopCompletion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaStopCompletion"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                auto ctx = getContextOrThrow(contextId);
                if (ctx->completion) {
                    ctx->completion->is_interrupted = true;
                }
                return jsi::Value::undefined();
            }
        );
        runtime.global().setProperty(runtime, "llamaStopCompletion", stopCompletion);

        auto toggleNativeLog = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaToggleNativeLog"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                bool enabled = count > 0 && arguments[0].isBool() ? arguments[0].getBool() : false;
                std::shared_ptr<jsi::Function> onLog;
                if (enabled && count > 1 && arguments[1].isObject() && arguments[1].asObject(runtime).isFunction(runtime)) {
                    onLog = std::make_shared<jsi::Function>(arguments[1].asObject(runtime).asFunction(runtime));
                }

                return createPromiseTask(runtime, callInvoker, [enabled, onLog, callInvoker, runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){})]() -> PromiseResultGenerator {
                    if (enabled && onLog) {
                        {
                            std::lock_guard<std::mutex> lock(g_log_mutex);
                            g_log_handler = onLog;
                            g_log_invoker = callInvoker;
                            g_log_runtime = runtimePtr;
                        }
                        llama_log_set(logToJsCallback, nullptr);
                    } else {
                        {
                            std::lock_guard<std::mutex> lock(g_log_mutex);
                            g_log_handler.reset();
                            g_log_invoker.reset();
                            g_log_runtime.reset();
                        }
                        llama_log_set(llama_log_callback_default, nullptr);
                    }
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                });
            }
        );
        runtime.global().setProperty(runtime, "llamaToggleNativeLog", toggleNativeLog);

        auto enableParallelMode = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaEnableParallelMode"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);

                bool enabled = getPropertyAsBool(runtime, params, "enabled", true);
                int nParallel = getPropertyAsInt(runtime, params, "n_parallel", 2);
                int nBatch = getPropertyAsInt(runtime, params, "n_batch", 512);

                return createPromiseTask(runtime, callInvoker, [contextId, enabled, nParallel, nBatch]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (enabled) {
                        ctx->enableParallelMode(nParallel, nBatch);
                        if (ctx->slot_manager) {
                            ctx->slot_manager->start_processing_loop();
                        }
                    } else {
                        if (ctx->slot_manager) {
                            ctx->slot_manager->stop_processing_loop();
                        }
                        ctx->disableParallelMode();
                    }
                    return [](jsi::Runtime& rt) { return jsi::Value(true); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaEnableParallelMode", enableParallelMode);

        auto queueCompletion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaQueueCompletion"),
            4,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);

                auto onToken = std::make_shared<jsi::Function>(arguments[2].asObject(runtime).asFunction(runtime));
                auto onComplete = std::make_shared<jsi::Function>(arguments[3].asObject(runtime).asFunction(runtime));

                auto ctxPtr = getContextOrThrow(contextId);
                auto originalParams = ctxPtr->params;
                parseCompletionParams(runtime, params, ctxPtr);
                common_params cparams = ctxPtr->params;
                ctxPtr->params = originalParams;

                std::vector<std::string> mediaPaths;
                if (params.hasProperty(runtime, "media_paths")) {
                    jsi::Array paths = params.getProperty(runtime, "media_paths").asObject(runtime).asArray(runtime);
                    for (size_t i = 0; i < paths.size(runtime); i++) {
                        mediaPaths.push_back(paths.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                    }
                }

                int chat_format = getPropertyAsInt(runtime, params, "chat_format", 0);
                std::string reasoningFormatStr = getPropertyAsString(runtime, params, "reasoning_format", "none");
                common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);
                bool thinking_forced_open = getPropertyAsBool(runtime, params, "thinking_forced_open", false);
                std::string prefill_text = getPropertyAsString(runtime, params, "prefill_text");
                std::string load_state_path = stripFileScheme(getPropertyAsString(runtime, params, "load_state_path"));
                std::string save_state_path = stripFileScheme(getPropertyAsString(runtime, params, "save_state_path"));
                int save_state_size = getPropertyAsInt(runtime, params, "save_state_size", -1);

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, cparams, mediaPaths, chat_format, reasoning_format, thinking_forced_open, prefill_text, load_state_path, save_state_path, save_state_size, onToken, onComplete, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    // TODO: guide_tokens support for queued completions (enable TTS guide tokens per request)

                    auto tokenizeResult = ctx->tokenize(cparams.prompt, mediaPaths);
                    std::vector<llama_token> tokens = tokenizeResult.tokens;

                    auto tokenCallback = [contextId, callInvoker, ctx, runtimePtr](const rnllama::completion_token_output& token) {
                        int requestId = token.request_id;
                        rnllama::completion_chat_output parsed_output;
                        bool has_parsed_output = false;
                        if (ctx->slot_manager) {
                            auto* slot = ctx->slot_manager->get_slot_by_request_id(requestId);
                            if (slot) {
                                try {
                                    parsed_output = slot->parseChatOutput(true);
                                    has_parsed_output = true;
                                } catch (...) {
                                    has_parsed_output = false;
                                }
                            }
                        }

                        auto callbacks = RequestManager::getInstance().getRequest(contextId, requestId);
                        if (callbacks.onToken) {
                            rnllama::completion_token_output tokenCopy = token;
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            callInvoker->invokeAsync([callbacks, contextId, tokenCopy, requestId, parsed_output, has_parsed_output, runtime]() {
                                long ctxPtr = g_llamaContexts.get(contextId);
                                if (ctxPtr) {
                                    auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                                    auto& rt = *runtime;
                                    jsi::Object res = createTokenResult(rt, ctx, tokenCopy);
                                    if (has_parsed_output) {
                                        setChatOutputFields(rt, res, parsed_output);
                                    }
                                    callbacks.onToken->call(rt, res, jsi::Value(requestId));
                                }
                            });
                        }
                    };

                    auto completeCallback = [contextId, callInvoker, runtimePtr](rnllama::llama_rn_slot* slot) {
                        int requestId = slot->request_id;
                        auto callbacks = RequestManager::getInstance().getRequest(contextId, requestId);
                        RequestManager::getInstance().removeRequest(contextId, requestId);
                        if (callbacks.onComplete) {
                            std::string text = slot->generated_text;
                            bool stopped_eos = slot->stopped_eos;
                            bool stopped_limit = slot->stopped_limit;
                            bool stopped_word = slot->stopped_word;
                            bool context_full = slot->context_full;
                            bool incomplete = slot->incomplete;
                            bool truncated = slot->truncated;
                            bool interrupted = slot->is_interrupted;
                            int32_t chat_format_val = slot->current_chat_format;
                            std::string stopping_word = slot->stopping_word;
                            size_t tokens_predicted = slot->num_tokens_predicted;
                            size_t tokens_evaluated = slot->num_prompt_tokens;
                            llama_pos tokens_cached = slot->n_past;
                            int32_t n_decoded = slot->n_decoded;
                            std::string error_message = slot->error_message;
                            auto timings = slot->get_timings();
                            auto token_probs = slot->generated_token_probs;
                            if (slot->parent_ctx && slot->ctx_sampling) {
                                common_perf_print(slot->parent_ctx->ctx, slot->ctx_sampling);
                            }

                            rnllama::completion_chat_output final_output;
                            bool has_final_output = false;
                            try {
                                final_output = slot->parseChatOutput(false);
                                has_final_output = true;
                            } catch (...) {
                                has_final_output = false;
                            }

                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            callInvoker->invokeAsync([callbacks, contextId, requestId, text, stopped_eos, stopped_limit, stopped_word, context_full, incomplete, truncated, interrupted, chat_format_val, stopping_word, tokens_predicted, tokens_evaluated, tokens_cached, n_decoded, error_message, timings, token_probs, final_output, has_final_output, runtime]() {
                                long ctxPtr = g_llamaContexts.get(contextId);
                                if (!ctxPtr) {
                                    return;
                                }
                                auto ctxVal = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                                auto& rt = *runtime;

                                jsi::Object res(rt);
                                res.setProperty(rt, "requestId", requestId);
                                res.setProperty(rt, "text", jsi::String::createFromUtf8(rt, text));
                                res.setProperty(rt, "chat_format", chat_format_val);
                                res.setProperty(rt, "stopped_eos", stopped_eos);
                                res.setProperty(rt, "stopped_limit", stopped_limit);
                                res.setProperty(rt, "stopped_word", stopped_word);
                                res.setProperty(rt, "context_full", context_full);
                                res.setProperty(rt, "incomplete", incomplete);
                                res.setProperty(rt, "truncated", truncated);
                                res.setProperty(rt, "interrupted", interrupted);
                                res.setProperty(rt, "stopping_word", jsi::String::createFromUtf8(rt, stopping_word));
                                res.setProperty(rt, "tokens_predicted", (double)tokens_predicted);
                                res.setProperty(rt, "tokens_evaluated", (double)tokens_evaluated);
                                res.setProperty(rt, "tokens_cached", (double)tokens_cached);
                                res.setProperty(rt, "n_decoded", (double)n_decoded);

                                res.setProperty(rt, "completion_probabilities", createCompletionProbabilities(rt, ctxVal, token_probs));

                                if (!error_message.empty()) {
                                    res.setProperty(rt, "error", jsi::String::createFromUtf8(rt, error_message));
                                }

                                if (has_final_output) {
                                    setChatOutputFields(rt, res, final_output);
                                }

                                jsi::Object timingsObj(rt);
                                timingsObj.setProperty(rt, "cache_n", (double)timings.cache_n);
                                timingsObj.setProperty(rt, "prompt_n", (double)timings.prompt_n);
                                timingsObj.setProperty(rt, "prompt_ms", (double)timings.prompt_ms);
                                timingsObj.setProperty(rt, "prompt_per_token_ms", (double)timings.prompt_per_token_ms);
                                timingsObj.setProperty(rt, "prompt_per_second", (double)timings.prompt_per_second);
                                timingsObj.setProperty(rt, "predicted_n", (double)timings.predicted_n);
                                timingsObj.setProperty(rt, "predicted_ms", (double)timings.predicted_ms);
                                timingsObj.setProperty(rt, "predicted_per_token_ms", (double)timings.predicted_per_token_ms);
                                timingsObj.setProperty(rt, "predicted_per_second", (double)timings.predicted_per_second);
                                res.setProperty(rt, "timings", timingsObj);

                                callbacks.onComplete->call(rt, res);
                            });
                        }
                    };

                    int requestId = ctx->slot_manager->queue_request(
                        cparams, tokens, mediaPaths, cparams.prompt, chat_format, reasoning_format, thinking_forced_open, prefill_text, load_state_path, save_state_path, save_state_size,
                        tokenCallback, completeCallback
                    );

                    RequestManager::getInstance().addRequest(contextId, requestId, {onToken, onComplete, nullptr});

                    return [requestId](jsi::Runtime& rt) {
                        jsi::Object res(rt);
                        res.setProperty(rt, "requestId", requestId);
                        return res;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaQueueCompletion", queueCompletion);

        auto cancelRequest = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaCancelRequest"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int requestId = (int)arguments[1].asNumber();

                auto ctx = getContextOrThrow(contextId);
                if (ctx->slot_manager) {
                    ctx->slot_manager->cancel_request(requestId);
                }
                RequestManager::getInstance().removeRequest(contextId, requestId);

                return jsi::Value::undefined();
            }
        );
        runtime.global().setProperty(runtime, "llamaCancelRequest", cancelRequest);

        auto queueEmbedding = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaQueueEmbedding"),
            4,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string text = arguments[1].asString(runtime).utf8(runtime);
                jsi::Object params = arguments[2].asObject(runtime);
                auto onResult = std::make_shared<jsi::Function>(arguments[3].asObject(runtime).asFunction(runtime));

                int embd_normalize = 0;
                bool has_embd_normalize = false;
                if (params.hasProperty(runtime, "embd_normalize")) {
                    embd_normalize = getPropertyAsInt(runtime, params, "embd_normalize", 2);
                    has_embd_normalize = true;
                }

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, text, embd_normalize, has_embd_normalize, onResult, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    const llama_vocab* vocab = llama_model_get_vocab(ctx->model);
                    const bool add_bos = llama_vocab_get_add_bos(vocab);
                    const bool is_enc_dec = llama_model_has_encoder(ctx->model);
                    std::vector<llama_token> tokens = common_tokenize(ctx->ctx, text, add_bos || is_enc_dec, true);

                    auto resultCallback = [contextId, callInvoker, runtimePtr](int32_t requestId, const std::vector<float>& embedding) {
                        auto callbacks = RequestManager::getInstance().getRequest(contextId, requestId);
                        RequestManager::getInstance().removeRequest(contextId, requestId);
                        if (callbacks.onResult) {
                            std::vector<float> embCopy = embedding;
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            callInvoker->invokeAsync([callbacks, embCopy, runtime]() {
                                auto& rt = *runtime;
                                jsi::Array res(rt, embCopy.size());
                                for (size_t i = 0; i < embCopy.size(); i++) {
                                    res.setValueAtIndex(rt, i, (double)embCopy[i]);
                                }
                                callbacks.onResult->call(rt, res);
                            });
                        }
                    };

                    const int normalize = has_embd_normalize ? embd_normalize : ctx->params.embd_normalize;
                    int requestId = ctx->slot_manager->queue_embedding_request(tokens, normalize, resultCallback);

                    RequestManager::getInstance().addRequest(contextId, requestId, {nullptr, nullptr, onResult});

                    return [requestId](jsi::Runtime& rt) {
                        jsi::Object res(rt);
                        res.setProperty(rt, "requestId", requestId);
                        return res;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaQueueEmbedding", queueEmbedding);

        auto queueRerank = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaQueueRerank"),
            5,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string query = arguments[1].asString(runtime).utf8(runtime);
                jsi::Array documentsArr = arguments[2].asObject(runtime).asArray(runtime);
                std::vector<std::string> documents;
                for (size_t i = 0; i < documentsArr.size(runtime); i++) {
                    documents.push_back(documentsArr.getValueAtIndex(runtime, i).asString(runtime).utf8(runtime));
                }
                jsi::Object params = arguments[3].asObject(runtime);
                auto onResult = std::make_shared<jsi::Function>(arguments[4].asObject(runtime).asFunction(runtime));

                int normalize = getPropertyAsInt(runtime, params, "normalize", 0);

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, query, documents, normalize, onResult, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    auto resultCallback = [contextId, callInvoker, runtimePtr](int32_t requestId, const std::vector<float>& scores) {
                        auto callbacks = RequestManager::getInstance().getRequest(contextId, requestId);
                        RequestManager::getInstance().removeRequest(contextId, requestId);
                        if (callbacks.onResult) {
                            std::vector<float> scoresCopy = scores;
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            callInvoker->invokeAsync([callbacks, scoresCopy, runtime]() {
                                auto& rt = *runtime;
                                jsi::Array res(rt, scoresCopy.size());
                                for (size_t i = 0; i < scoresCopy.size(); i++) {
                                    jsi::Object item(rt);
                                    item.setProperty(rt, "score", (double)scoresCopy[i]);
                                    item.setProperty(rt, "index", (int)i);
                                    res.setValueAtIndex(rt, i, item);
                                }
                                callbacks.onResult->call(rt, res);
                            });
                        }
                    };

                    int requestId = ctx->slot_manager->queue_rerank_request(query, documents, normalize, resultCallback);

                    RequestManager::getInstance().addRequest(contextId, requestId, {nullptr, nullptr, onResult});

                    return [requestId](jsi::Runtime& rt) {
                        jsi::Object res(rt);
                        res.setProperty(rt, "requestId", requestId);
                        return res;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaQueueRerank", queueRerank);

        auto releaseContext = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseContext"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 int contextId = (int)arguments[0].asNumber();
                 return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                     RequestManager::getInstance().clearContext(contextId);
                     long ctxPtr = g_llamaContexts.get(contextId);
                     if (ctxPtr) {
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         if (ctx->completion) {
                             ctx->completion->is_interrupted = true;
                         }
                         if (ctx->slot_manager) {
                             ctx->slot_manager->stop_processing_loop();
                         }
                     }

                     TaskManager::getInstance().waitForContext(contextId, 1);

                     if (ctxPtr) {
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         delete ctx;
                         removeContext(contextId);
                     }
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseContext", releaseContext);

        auto releaseAllContexts = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseAllContexts"),
            0,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                 return createPromiseTask(runtime, callInvoker, []() -> PromiseResultGenerator {
                     RequestManager::getInstance().clearAll();

                     auto contexts = g_llamaContexts.snapshot();
                     for (const auto& entry : contexts) {
                         long ctxPtr = entry.second;
                         if (!ctxPtr) {
                             continue;
                         }
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         if (ctx->completion) {
                             ctx->completion->is_interrupted = true;
                         }
                         if (ctx->slot_manager) {
                             ctx->slot_manager->stop_processing_loop();
                         }
                     }

                     TaskManager::getInstance().waitForAll(1);

                     g_llamaContexts.clear([](long ptr) {
                        if (ptr) {
                            auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                            delete ctx;
                        }
                     });
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 });
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseAllContexts", releaseAllContexts);

        auto setContextLimitFn = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaSetContextLimit"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int64_t limit = (int64_t)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [limit]() -> PromiseResultGenerator {
                    setContextLimit(limit);
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                });
            }
        );
        runtime.global().setProperty(runtime, "llamaSetContextLimit", setContextLimitFn);

        // LoRA Adapters
        auto applyLoraAdapters = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaApplyLoraAdapters"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Array loraList = arguments[1].asObject(runtime).asArray(runtime);
                std::vector<common_adapter_lora_info> lora_adapters;
                for (size_t i = 0; i < loraList.size(runtime); i++) {
                    jsi::Object item = loraList.getValueAtIndex(runtime, i).asObject(runtime);
                    common_adapter_lora_info la;
                    la.path = getPropertyAsString(runtime, item, "path");
                    la.scale = getPropertyAsFloat(runtime, item, "scaled", 1.0f);
                    if (!la.path.empty()) {
                        lora_adapters.push_back(la);
                    }
                }

                return createPromiseTask(runtime, callInvoker, [contextId, lora_adapters]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    int result = ctx->applyLoraAdapters(lora_adapters);
                    if (result != 0) throw std::runtime_error("Failed to apply lora adapters");
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaApplyLoraAdapters", applyLoraAdapters);

        auto removeLoraAdapters = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaRemoveLoraAdapters"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    ctx->removeLoraAdapters();
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaRemoveLoraAdapters", removeLoraAdapters);

        auto getLoadedLoraAdapters = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetLoadedLoraAdapters"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    auto adapters = ctx->getLoadedLoraAdapters();
                    return [adapters](jsi::Runtime& rt) {
                        jsi::Array res(rt, adapters.size());
                        for (size_t i = 0; i < adapters.size(); i++) {
                            jsi::Object item(rt);
                            item.setProperty(rt, "path", jsi::String::createFromUtf8(rt, adapters[i].path));
                            item.setProperty(rt, "scaled", (double)adapters[i].scale);
                            res.setValueAtIndex(rt, i, item);
                        }
                        return res;
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetLoadedLoraAdapters", getLoadedLoraAdapters);

        // Multimodal
        auto initMultimodal = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitMultimodal"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                std::string path = getPropertyAsString(runtime, params, "path");
                bool use_gpu = getPropertyAsBool(runtime, params, "use_gpu", true);

                return createPromiseTask(runtime, callInvoker, [contextId, path, use_gpu]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    bool result = ctx->initMultimodal(path, use_gpu);
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaInitMultimodal", initMultimodal);

        auto isMultimodalEnabled = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaIsMultimodalEnabled"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    bool result = ctx->isMultimodalEnabled();
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaIsMultimodalEnabled", isMultimodalEnabled);

        auto getMultimodalSupport = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetMultimodalSupport"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isMultimodalEnabled()) throw std::runtime_error("Multimodal is not enabled");
                    bool vision = ctx->isMultimodalSupportVision();
                    bool audio = ctx->isMultimodalSupportAudio();
                    return [vision, audio](jsi::Runtime& rt) {
                        jsi::Object res(rt);
                            res.setProperty(rt, "vision", vision);
                            res.setProperty(rt, "audio", audio);
                            return res;
                        };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetMultimodalSupport", getMultimodalSupport);

        auto releaseMultimodal = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseMultimodal"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    ctx->releaseMultimodal();
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseMultimodal", releaseMultimodal);

        // Vocoder
        auto initVocoder = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitVocoder"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Object params = arguments[1].asObject(runtime);
                std::string path = getPropertyAsString(runtime, params, "path");
                int n_batch = getPropertyAsInt(runtime, params, "n_batch", 512);

                return createPromiseTask(runtime, callInvoker, [contextId, path, n_batch]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                         throw std::runtime_error("Context is busy");
                    }
                    bool result = ctx->initVocoder(path, n_batch);
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaInitVocoder", initVocoder);

        auto isVocoderEnabled = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaIsVocoderEnabled"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    bool result = ctx->isVocoderEnabled();
                    return [result](jsi::Runtime& rt) { return jsi::Value(result); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaIsVocoderEnabled", isVocoderEnabled);

        auto getFormattedAudioCompletion = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetFormattedAudioCompletion"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string speakerJsonStr = arguments[1].asString(runtime).utf8(runtime);
                std::string textToSpeak = arguments[2].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, speakerJsonStr, textToSpeak]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_result = ctx->tts_wrapper->getFormattedAudioCompletion(ctx, speakerJsonStr, textToSpeak);
                        return [audio_result](jsi::Runtime& rt) {
                            jsi::Object res(rt);
                            res.setProperty(rt, "prompt", jsi::String::createFromUtf8(rt, audio_result.prompt));
                            res.setProperty(rt, "grammar", jsi::String::createFromUtf8(rt, audio_result.grammar));
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetFormattedAudioCompletion", getFormattedAudioCompletion);

        auto getAudioCompletionGuideTokens = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetAudioCompletionGuideTokens"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::string textToSpeak = arguments[1].asString(runtime).utf8(runtime);

                return createPromiseTask(runtime, callInvoker, [contextId, textToSpeak]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto guide_tokens = ctx->tts_wrapper->getAudioCompletionGuideTokens(ctx, textToSpeak);
                        return [guide_tokens](jsi::Runtime& rt) {
                            jsi::Array res(rt, guide_tokens.size());
                            for (size_t i = 0; i < guide_tokens.size(); i++) {
                                res.setValueAtIndex(rt, i, (double)guide_tokens[i]);
                            }
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetAudioCompletionGuideTokens", getAudioCompletionGuideTokens);

        auto decodeAudioTokens = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDecodeAudioTokens"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                jsi::Array tokensArr = arguments[1].asObject(runtime).asArray(runtime);
                std::vector<llama_token> tokens;
                for (size_t i = 0; i < tokensArr.size(runtime); i++) {
                    tokens.push_back((llama_token)tokensArr.getValueAtIndex(runtime, i).asNumber());
                }

                return createPromiseTask(runtime, callInvoker, [contextId, tokens]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_data = ctx->tts_wrapper->decodeAudioTokens(ctx, tokens);
                        return [audio_data](jsi::Runtime& rt) {
                            jsi::Array res(rt, audio_data.size());
                            for (size_t i = 0; i < audio_data.size(); i++) {
                                res.setValueAtIndex(rt, i, (double)audio_data[i]);
                            }
                            return res;
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDecodeAudioTokens", decodeAudioTokens);

        // Cache management
        auto clearCache = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaClearCache"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                bool clearData = count > 1 && arguments[1].isBool() ? arguments[1].asBool() : false;
                return createPromiseTask(runtime, callInvoker, [contextId, clearData]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (ctx->completion && ctx->completion->is_predicting) {
                        throw std::runtime_error("Context is busy");
                    }
                    ctx->clearCache(clearData);
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaClearCache", clearCache);

        auto releaseVocoder = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseVocoder"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    ctx->releaseVocoder();
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseVocoder", releaseVocoder);
    }

    void cleanupJSIBindings() {
        RequestManager::getInstance().clearAll();
        auto contexts = g_llamaContexts.snapshot();
        for (const auto& entry : contexts) {
            long ctxPtr = entry.second;
            if (!ctxPtr) {
                continue;
            }
            auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
            if (ctx->completion) {
                ctx->completion->is_interrupted = true;
            }
            if (ctx->slot_manager) {
                ctx->slot_manager->stop_processing_loop();
            }
        }

        llama_log_set(llama_log_callback_default, nullptr);
        TaskManager::getInstance().waitForAll();
        ThreadPool::getInstance().shutdown();

        g_llamaContexts.clear([](long ptr) {
            if (ptr) {
                auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                delete ctx;
            }
        });
        g_context_limit.store(-1);
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_log_handler.reset();
            g_log_invoker.reset();
        }
        g_log_runtime.reset();
    }
}
