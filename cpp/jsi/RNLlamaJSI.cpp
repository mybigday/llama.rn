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
#include "JSIJson.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
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
            case LM_GGML_BACKEND_DEVICE_TYPE_META:
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

    static bool isContextBusy(rnllama::llama_rn_context* ctx) {
        if (ctx == nullptr) {
            return false;
        }

        if (ctx->completion && ctx->completion->is_predicting) {
            return true;
        }

        return ctx->slot_manager && ctx->slot_manager->has_pending_work();
    }

    static void throwIfContextBusy(rnllama::llama_rn_context* ctx) {
        if (isContextBusy(ctx)) {
            throw std::runtime_error("Context is busy");
        }
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

    static bool isThinkingForcedOpen(const common_chat_params& chatParams) {
        if (!chatParams.supports_thinking || chatParams.thinking_start_tag.empty()) {
            return false;
        }

        const size_t lastStart = chatParams.generation_prompt.rfind(chatParams.thinking_start_tag);
        if (lastStart == std::string::npos) {
            return false;
        }

        for (const auto& endTag : chatParams.thinking_end_tags) {
            if (endTag.empty()) {
                continue;
            }
            const size_t lastEnd = chatParams.generation_prompt.rfind(endTag);
            if (lastEnd != std::string::npos && lastEnd >= lastStart) {
                return false;
            }
        }
        return true;
    }

    static json chatTemplateCapsJson(bool tools, bool toolCalls, bool parallelToolCalls, bool systemRole) {
        return json::object({
            {"tools", tools},
            {"toolCalls", toolCalls},
            {"parallelToolCalls", parallelToolCalls},
            {"systemRole", systemRole},
        });
    }

    static json modelDetailsJson(rnllama::llama_rn_context* ctx) {
        char desc[1024];
        llama_model_desc(ctx->model, desc, sizeof(desc));

        json metadata = json::object();
        const int metaCount = llama_model_meta_count(ctx->model);
        for (int i = 0; i < metaCount; ++i) {
            char key[256];
            llama_model_meta_key_by_index(ctx->model, i, key, sizeof(key));
            char val[16384];
            llama_model_meta_val_str_by_index(ctx->model, i, val, sizeof(val));
            metadata[key] = val;
        }

        // Chat template capabilities
        const bool llamaChat = ctx->validateModelChatTemplate(false, nullptr);

        json jinja = json::object({{"default", ctx->validateModelChatTemplate(true, nullptr)}});
        if (ctx->templates && common_chat_templates_has_variant(ctx->templates.get(), "")) {
            auto caps = common_chat_templates_get_caps(ctx->templates.get(), "");
            jinja["defaultCaps"] = chatTemplateCapsJson(caps.supports_tools, caps.supports_tool_calls,
                                                       caps.supports_parallel_tool_calls, caps.supports_system_role);
        } else {
            jinja["defaultCaps"] = chatTemplateCapsJson(false, false, false, false);
        }
        jinja["toolUse"] = ctx->validateModelChatTemplate(true, "tool_use");
        if (ctx->templates && common_chat_templates_has_variant(ctx->templates.get(), "tool_use")) {
            auto caps = common_chat_templates_get_caps(ctx->templates.get(), "tool_use");
            jinja["toolUseCaps"] = chatTemplateCapsJson(caps.supports_tools, caps.supports_tool_calls,
                                                       caps.supports_parallel_tool_calls, caps.supports_system_role);
        }

        return json::object({
            {"desc", desc},
            {"size", llama_model_size(ctx->model)},
            {"nEmbd", llama_model_n_embd(ctx->model)},
            {"nParams", llama_model_n_params(ctx->model)},
            {"is_recurrent", llama_model_is_recurrent(ctx->model)},
            {"is_hybrid", llama_model_is_hybrid(ctx->model)},
            {"metadata", metadata},
            {"chatTemplates", json::object({{"llamaChat", llamaChat}, {"jinja", jinja}})},
            // Deprecated flag maintained for compatibility
            {"isChatTemplateSupported", llamaChat},
        });
    }

    static json chatParamsJson(const common_chat_params& chatParams) {
        json result = json::object({
            {"prompt", chatParams.prompt},
            {"chat_format", (int) chatParams.format},
            {"grammar", chatParams.grammar},
            {"grammar_lazy", chatParams.grammar_lazy},
            {"generation_prompt", chatParams.generation_prompt},
            {"thinking_forced_open", isThinkingForcedOpen(chatParams)},
        });
        if (!chatParams.thinking_start_tag.empty()) {
            result["thinking_start_tag"] = chatParams.thinking_start_tag;
        }
        if (!chatParams.thinking_end_tags.empty()) {
            result["thinking_end_tag"] = chatParams.thinking_end_tags.front();
        }

        // Preserve the same shape as legacy native bridge
        result["type"] = "jinja";
        result["preserved_tokens"] = chatParams.preserved_tokens;
        result["additional_stops"] = chatParams.additional_stops;

        json triggers = json::array();
        for (const auto& trigger : chatParams.grammar_triggers) {
            triggers.push_back(json::object({
                {"type", (int) trigger.type},
                {"value", trigger.value},
                {"token", (int) trigger.token},
            }));
        }
        result["grammar_triggers"] = triggers;

        // Return the PEG parser string for COMMON_CHAT_FORMAT_PEG_* formats
        if (!chatParams.parser.empty()) {
            result["chat_parser"] = chatParams.parser;
        }
        return result;
    }

    static json rerankResultJson(const std::vector<float>& scores) {
        json result = json::array();
        for (size_t i = 0; i < scores.size(); i++) {
            result.push_back(json::object({{"score", (double) scores[i]}, {"index", (int) i}}));
        }
        return result;
    }

    static json parallelStatusJson(const rnllama::llama_rn_parallel_status& status) {
        json requests = json::array();
        for (const auto& req : status.requests) {
            requests.push_back(json::object({
                {"request_id", req.request_id},
                {"type", req.type},
                {"state", req.state},
                {"prompt_length", req.prompt_length},
                {"tokens_generated", req.tokens_generated},
                {"prompt_ms", req.prompt_ms},
                {"generation_ms", req.generation_ms},
                {"tokens_per_second", req.tokens_per_second},
            }));
        }
        return json::object({
            {"n_parallel", status.n_parallel},
            {"active_slots", status.active_slots},
            {"queued_requests", status.queued_requests},
            {"requests", requests},
        });
    }

    static std::vector<lm_ggml_backend_dev_t> buildDeviceOverrides(
        const std::vector<std::string>& requestedDevices,
        bool skipGpuDevices,
        bool& anyGpuAvailable
    ) {
        std::vector<lm_ggml_backend_dev_t> selected;
        anyGpuAvailable = false;
        bool cpuRequested = false;

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

            // The CPU device is always used as the fallback and must not be passed as an
            // offload target: llama.cpp would place layers in the plain CPU buffer and skip
            // the repacked (i8mm/dotprod) buffer types, which is several times slower.
            if (type == LM_GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpuRequested = true;
                continue;
            }

            selected.push_back(dev);
        }

        // A null-terminated empty list tells llama.cpp to use no offload devices, i.e. CPU
        // only with its extra buffer types. Only do this when the caller asked for the CPU
        // explicitly, so an unknown device name still falls back to the default selection.
        if (!selected.empty() || cpuRequested) {
            selected.push_back(nullptr);
        }

        return selected;
    }

    static bool isGpuDeviceType(enum lm_ggml_backend_dev_type type) {
        return type == LM_GGML_BACKEND_DEVICE_TYPE_GPU || type == LM_GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    static bool hasGpuBackendDevice() {
        const size_t devCount = lm_ggml_backend_dev_count();
        for (size_t i = 0; i < devCount; ++i) {
            auto dev = lm_ggml_backend_dev_get(i);
            if (isGpuDeviceType(lm_ggml_backend_dev_type(dev))) {
                return true;
            }
        }
        return false;
    }

    static void configureBackendDevices(
        common_params& cparams,
        const std::vector<std::string>& requestedDevices,
        bool devicesProvided,
        bool skipGpuDevices,
        bool& anyGpuAvailable
    ) {
        anyGpuAvailable = false;
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
                    if (isGpuDeviceType(lm_ggml_backend_dev_type(dev))) {
                        anyGpuAvailable = true;
                        break;
                    }
                }
            }
#endif
        }

        // Track backend availability when no explicit override was applied.
        if (overrideDevices.empty() && !anyGpuAvailable) {
            anyGpuAvailable = hasGpuBackendDevice();
        }
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
        rnllama::install_ggml_abort_handler();
        TaskManager::getInstance().reset();
        auto initContext = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaInitContext"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                // Convert the params object once; every lookup below and the parser
                // work on plain json (see JSIJson.h for the conversion rules).
                json params = toJson(runtime, arguments[1]);
                if (!params.is_object()) {
                    throw jsi::JSError(runtime, "llamaInitContext: params must be an object");
                }
                bool isModelAsset = getPropertyAsBool(params, "is_model_asset", false);
                bool isModelDraftAsset = getPropertyAsBool(params, "is_model_draft_asset", false);

                bool useProgressCallback = getPropertyAsBool(params, "use_progress_callback", false);
                int progressCallbackEvery = getPropertyAsInt(params, "progress_callback_every", 1);
                std::shared_ptr<ProgressCallbackData> progressData;
                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    useProgressCallback = true;
                    progressData = std::make_shared<ProgressCallbackData>();
                    progressData->callback = makeJsiFunction(runtime, arguments[2], callInvoker);
                    progressData->callInvoker = callInvoker;
                    progressData->runtime = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){});
                    progressData->contextId = contextId;
                    progressData->progressEvery = std::max(1, progressCallbackEvery);
                    progressData->lastProgress.store(0);
                } else if (useProgressCallback) {
                    // Progress requested but no callback provided
                    useProgressCallback = false;
                }

                common_params cparams;
                parseCommonParams(params, cparams);

#if defined(__APPLE__)
                if (isModelAsset) {
                    cparams.model.path = resolveIosModelPath(cparams.model.path, true);
                }
                if (isModelDraftAsset && !cparams.speculative.draft.mparams.path.empty()) {
                    cparams.speculative.draft.mparams.path =
                        resolveIosModelPath(cparams.speculative.draft.mparams.path, true);
                }
#endif

                bool skipGpuDevices = getPropertyAsBool(params, "no_gpu_devices", false);
                if (skipGpuDevices) {
                    cparams.n_gpu_layers = 0;
                }

                std::vector<std::string> requestedDevices;
                bool devicesProvided = false;
                if (auto it = params.find("devices"); it != params.end() && it->is_array() && !it->empty()) {
                    devicesProvided = true;
                    for (const auto& val : *it) {
                        if (val.is_string()) {
                            requestedDevices.push_back(val.get<std::string>());
                        }
                    }
                }

                int stateCacheBudgetMb =
                    getPropertyAsInt(params, "state_cache_budget_mb", 160);
                int stateCacheMaxCheckpoints =
                    getPropertyAsInt(params, "state_cache_max_checkpoints", 8);

                return createPromiseTask(runtime, callInvoker, [
                    contextId,
                    cparams,
                    skipGpuDevices,
                    requestedDevices,
                    devicesProvided,
                    useProgressCallback,
                    progressData,
                    stateCacheBudgetMb,
                    stateCacheMaxCheckpoints
                ]() mutable -> PromiseResultGenerator {
                    if (isContextLimitReached()) {
                        throw std::runtime_error("Context limit reached");
                    }

                    ensureBackendInitialized();

#if defined(__APPLE__)
                    auto metalAvailability = getMetalAvailability(skipGpuDevices);
                    std::string appleGpuReason = metalAvailability.available ? "" : metalAvailability.reason;
                    if (!metalAvailability.available && !skipGpuDevices) {
                        skipGpuDevices = true;
                        cparams.n_gpu_layers = 0;
                    }
#endif

                    bool anyGpuAvailable = false;
                    configureBackendDevices(
                        cparams,
                        requestedDevices,
                        devicesProvided,
                        skipGpuDevices,
                        anyGpuAvailable
                    );

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
                    // Prompt state cache tuning (multi-turn KV reuse on
                    // recurrent/hybrid/SWA models). Budget in MiB; 0 disables it.
                    {
                        ctx->state_cache_budget_bytes =
                            stateCacheBudgetMb > 0 ? (size_t) stateCacheBudgetMb * 1024 * 1024 : 0;
                        ctx->state_cache_max_checkpoints = stateCacheMaxCheckpoints;
                    }
                    if (ctx->loadModel(cparams)) {
                         ctx->attachThreadpoolsIfAvailable();

                         if (ctx->params.embedding && llama_model_has_encoder(ctx->model) && llama_model_has_decoder(ctx->model)) {
                             delete ctx;
                             throw std::runtime_error("Embedding is not supported in encoder-decoder models");
                         }

                         std::vector<std::string> usedDevices;
                         bool gpuEnabled = false;
                         if (ctx->llama_init->model() != nullptr) {
                             for (const auto & dev_info : ctx->llama_init->model()->devices) {
                                 auto dev = dev_info.dev;
                                 if (dev == nullptr) continue;
                                 const char* used_name = lm_ggml_backend_dev_name(dev);
                                 if (used_name != nullptr) {
                                     usedDevices.push_back(used_name);
                                 }
                                 if (isGpuDeviceType(lm_ggml_backend_dev_type(dev))) {
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

                         std::string androidLibName = "";
                         #if defined(__ANDROID__)
                         androidLibName = g_android_loaded_library;
                         #endif
                         json result = json::object({
                             {"gpu", gpuEnabled},
                             {"reasonNoGPU", reasonNoGPU},
                             {"systemInfo", common_params_get_system_info(ctx->params)},
                             // Model metadata and chat template capabilities
                             {"model", modelDetailsJson(ctx)},
                             {"devices", usedDevices},
                             {"androidLib", androidLibName},
                         });

                         return [result](jsi::Runtime& rt) {
                             return fromJson(rt, result);
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
                    json info = modelInfoJson(path, skip);
                    return [info](jsi::Runtime& rt) {
                        return fromJson(rt, info);
                    };
                }, -1, false);
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
                 }, -1, false);
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
                    throwIfContextBusy(ctx);
                    json result = rnllama_jsi::loadSession(ctx, path);
                    return [result](jsi::Runtime& rt) {
                        return fromJson(rt, result);
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
                    throwIfContextBusy(ctx);
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
                    json result = tokenizeResultJson(ctx->tokenize(text, mediaPaths));
                    return [result](jsi::Runtime& rt) {
                        return fromJson(rt, result);
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

        // llamaGetFormattedChat(ctxId, messages, chatTemplate?, opts?)
        //   messages: OpenAI-compatible messages as a JSON string. Kept as a
        //             string on purpose: llama.cpp's common_json can only be
        //             built via parse(), so an object here would just be
        //             re-serialized before reaching common_chat_msgs_parse_oaicompat.
        //   opts: { jinja, json_schema (JSON string), tools (JSON string),
        //           parallel_tool_calls, tool_choice, enable_thinking,
        //           reasoning_format, add_generation_prompt, now (string|number),
        //           chat_template_kwargs (object), force_pure_content }
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
                 // Convert the options object once on the JS thread; the json
                 // value is plain data and safe to hand to the worker.
                 json opts = (count > 3 && arguments[3].isObject())
                     ? toJson(runtime, arguments[3])
                     : json::object();

                 return createPromiseTask(runtime, callInvoker, [contextId, messages, chatTemplate, opts]() -> PromiseResultGenerator {
                      auto ctx = getContextOrThrow(contextId);

                      // Type-checked lookups that fall back to the default on a
                      // missing or mismatched value, matching the old
                      // getPropertyAs* behaviour (nlohmann's value() would throw).
                      auto getStr = [&opts](const char* key, const std::string& def = "") {
                          auto it = opts.find(key);
                          return (it != opts.end() && it->is_string()) ? it->get<std::string>() : def;
                      };
                      auto getBool = [&opts](const char* key, bool def) {
                          auto it = opts.find(key);
                          return (it != opts.end() && it->is_boolean()) ? it->get<bool>() : def;
                      };

                      const bool useJinja = getBool("jinja", false);
                      if (useJinja) {
                          // `now` is seconds since epoch, accepted as string or number.
                          std::string nowStr = getStr("now");
                          if (auto it = opts.find("now"); it != opts.end() && it->is_number()) {
                              nowStr = std::to_string(it->get<long long>());
                          }

                          // Template kwargs are passed to the jinja engine as JSON
                          // text per value (same as llama.cpp's server), so dump()
                          // each raw value instead of asking JS to pre-stringify.
                          std::map<std::string, std::string> chatTemplateKwargs;
                          if (auto it = opts.find("chat_template_kwargs"); it != opts.end() && it->is_object()) {
                              for (auto& [key, value] : it->items()) {
                                  chatTemplateKwargs[key] = value.dump();
                              }
                          }

                          auto chatParams = ctx->getFormattedChatWithJinja(
                               messages, chatTemplate,
                               getStr("json_schema"), getStr("tools"),
                               getBool("parallel_tool_calls", false),
                               getStr("tool_choice"),
                               getBool("enable_thinking", false),
                               getStr("reasoning_format", "none"),
                               getBool("add_generation_prompt", true),
                               nowStr, chatTemplateKwargs,
                               getBool("force_pure_content", false)
                          );

                          json result = chatParamsJson(chatParams);
                          return [result](jsi::Runtime& rt) {
                              return fromJson(rt, result);
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
                json params = toJson(runtime, arguments[2]);

                // Absent -> keep the context's embd_normalize
                int embd_normalize = 0;
                bool has_embd_normalize = false;
                if (auto it = params.find("embd_normalize"); it != params.end() && it->is_number()) {
                    embd_normalize = it->get<int>();
                    has_embd_normalize = true;
                }

                return createPromiseTask(runtime, callInvoker, [contextId, text, embd_normalize, has_embd_normalize]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (!ctx->completion) throw std::runtime_error("Completion not initialized");
                    if (ctx->params.embedding != true) throw std::runtime_error("Embedding is not enabled");
                    throwIfContextBusy(ctx);

                    common_params embdParams = ctx->params;
                    embdParams.embedding = true;
                    embdParams.embd_normalize = has_embd_normalize ? embd_normalize : ctx->params.embd_normalize;

                    ctx->params.prompt = text;
                    ctx->params.n_predict = 0;

                    std::vector<float> result = ctx->completion->embedding(embdParams);

                    return [result](jsi::Runtime& rt) {
                        jsi::Object resultDict(rt);
                        resultDict.setProperty(rt, "embedding", makeFloat32Array(rt, result));
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
                    throwIfContextBusy(ctx);

                    std::vector<float> scores = ctx->completion->rerank(query, documents);

                    json result = rerankResultJson(scores);
                    return [result](jsi::Runtime& rt) {
                        return fromJson(rt, result);
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
                // Convert the params object once; every lookup below and the parser
                // work on plain json (see JSIJson.h for the conversion rules).
                json params = toJson(runtime, arguments[1]);
                if (!params.is_object()) {
                    throw jsi::JSError(runtime, "llamaCompletion: params must be an object");
                }
                std::shared_ptr<jsi::Function> onToken;

                if (count > 2 && arguments[2].isObject() && arguments[2].asObject(runtime).isFunction(runtime)) {
                    onToken = makeJsiFunction(runtime, arguments[2], callInvoker);
                }

                bool emitPartial = getPropertyAsBool(params, "emit_partial_completion", false);

                auto ctx = getContextOrThrow(contextId);
                throwIfContextBusy(ctx);
                ctx->completion->rewind();

                parseCompletionParams(params, ctx);

                std::vector<std::string> mediaPaths;
                // media_paths: string | string[]
                if (auto it = params.find("media_paths"); it != params.end()) {
                    if (it->is_string()) {
                        mediaPaths.push_back(it->get<std::string>());
                    } else if (it->is_array()) {
                        for (const auto& path : *it) {
                            if (path.is_string()) {
                                mediaPaths.push_back(path.get<std::string>());
                            }
                        }
                    }
                }

                int chat_format = getPropertyAsInt(params, "chat_format", 0);
                std::string reasoningFormatStr = getPropertyAsString(params, "reasoning_format", "none");
                common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);
                std::string generation_prompt = getPropertyAsString(params, "generation_prompt");
                std::string chat_parser = getPropertyAsString(params, "chat_parser");
                std::string prefill_text = getPropertyAsString(params, "prefill_text");

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, onToken, emitPartial, mediaPaths, chat_format, reasoning_format, generation_prompt, chat_parser, prefill_text, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);

                    if (ctx->completion == nullptr) {
                        throw std::runtime_error("Completion not initialized");
                    }
                    throwIfContextBusy(ctx);

                    if (!mediaPaths.empty() && ctx->completion->shouldUseMTP()) {
                        throw std::runtime_error("MTP speculative decoding currently supports text-only completion");
                    }

                    // NOTE: no rewind() here — it already ran on the JS thread
                    // BEFORE parseCompletionParams. rewind() resets
                    // sampling.grammar / antiprompt, so calling it again at this
                    // point would wipe the grammar and stop sequences this
                    // completion's params just configured (that regression broke
                    // TTS grammar-forced output).
                    if (!ctx->completion->initSampling()) {
                        throw std::runtime_error("Failed to initialize sampling");
                    }

                    ctx->completion->prefill_text = rnllama::utf8_sanitize(prefill_text);
                    ctx->completion->beginCompletion(chat_format, reasoning_format, generation_prompt, chat_parser);

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
                                    json tokenResult = tokenResultJson(ctx, output_copy);
                                    if (has_partial_output) {
                                        addChatOutputFields(tokenResult, partial_output);
                                    }
                                    callInvoker->invokeAsync([onToken, tokenResult, contextId, runtime]() {
                                        // Skip the callback if the context was released meanwhile
                                        if (!g_llamaContexts.get(contextId)) {
                                            return;
                                        }
                                        auto& rt = *runtime;
                                        onToken->call(rt, fromJson(rt, tokenResult));
                                    });
                                }
                            }
                        }
                    }

                    common_perf_print(ctx->ctx, ctx->completion->ctx_sampling);
                    ctx->completion->endCompletion();

                    // Snapshot the result here, before another task can touch the context
                    CompletionResult result = completionResult(ctx);

                    return [contextId, result](jsi::Runtime& rt) -> jsi::Value {
                        if (!g_llamaContexts.get(contextId)) {
                            // Context was released, return minimal interrupted result
                            return fromJson(rt, json::object({
                                {"text", ""},
                                {"interrupted", true},
                                {"context_released", true},
                            }));
                        }
                        return result.toJsi(rt);
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
                    onLog = makeJsiFunction(runtime, arguments[1], callInvoker);
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
                }, -1, false);
            }
        );
        runtime.global().setProperty(runtime, "llamaToggleNativeLog", toggleNativeLog);

        auto enableParallelMode = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaEnableParallelMode"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                auto opts = optionsFromJson<ParallelModeOptions>(runtime, toJson(runtime, arguments[1]), "enableParallelMode");

                return createPromiseTask(runtime, callInvoker, [contextId, opts]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (opts.enabled) {
                        ctx->enableParallelMode(opts.n_parallel, opts.n_batch);
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
                // Convert the params object once; every lookup below and the parser
                // work on plain json (see JSIJson.h for the conversion rules).
                json params = toJson(runtime, arguments[1]);
                if (!params.is_object()) {
                    throw jsi::JSError(runtime, "llamaQueueCompletion: params must be an object");
                }

                auto onToken = makeJsiFunction(runtime, arguments[2], callInvoker);
                auto onComplete = makeJsiFunction(runtime, arguments[3], callInvoker);

                auto ctxPtr = getContextOrThrow(contextId);
                auto originalParams = ctxPtr->params;
                parseCompletionParams(params, ctxPtr);
                common_params cparams = ctxPtr->params;
                ctxPtr->params = originalParams;

                std::vector<std::string> mediaPaths;
                // media_paths: string | string[]
                if (auto it = params.find("media_paths"); it != params.end()) {
                    if (it->is_string()) {
                        mediaPaths.push_back(it->get<std::string>());
                    } else if (it->is_array()) {
                        for (const auto& path : *it) {
                            if (path.is_string()) {
                                mediaPaths.push_back(path.get<std::string>());
                            }
                        }
                    }
                }

                int chat_format = getPropertyAsInt(params, "chat_format", 0);
                std::string reasoningFormatStr = getPropertyAsString(params, "reasoning_format", "none");
                common_reasoning_format reasoning_format = common_reasoning_format_from_name(reasoningFormatStr);
                std::string generation_prompt = getPropertyAsString(params, "generation_prompt");
                std::string chat_parser = getPropertyAsString(params, "chat_parser");
                std::string prefill_text = getPropertyAsString(params, "prefill_text");
                std::string load_state_path = stripFileScheme(getPropertyAsString(params, "load_state_path"));
                std::string save_state_path = stripFileScheme(getPropertyAsString(params, "save_state_path"));
                std::string save_prompt_state_path = stripFileScheme(getPropertyAsString(params, "save_prompt_state_path"));
                int load_state_size = getPropertyAsInt(params, "load_state_size", -1);
                int save_state_size = getPropertyAsInt(params, "save_state_size", -1);

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, cparams, mediaPaths, chat_format, reasoning_format, generation_prompt, chat_parser, prefill_text, load_state_path, save_state_path, save_prompt_state_path, load_state_size, save_state_size, onToken, onComplete, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

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
                            json tokenResult = tokenResultJson(ctx, token);
                            if (has_parsed_output) {
                                addChatOutputFields(tokenResult, parsed_output);
                            }
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            invokeAsyncTracked(callInvoker, contextId, [callbacks, contextId, tokenResult, requestId, runtime](bool shouldProceed) {
                                if (!shouldProceed) return;
                                if (!g_llamaContexts.get(contextId)) return;
                                auto& rt = *runtime;
                                callbacks.onToken->call(rt, fromJson(rt, tokenResult), jsi::Value(requestId));
                            });
                        }
                    };

                    auto completeCallback = [contextId, callInvoker, runtimePtr](rnllama::llama_rn_slot* slot) {
                        int requestId = slot->request_id;
                        auto callbacks = RequestManager::getInstance().takeRequest(contextId, requestId);
                        if (callbacks.onComplete) {
                            if (slot->parent_ctx && slot->ctx_sampling) {
                                common_perf_print(slot->parent_ctx->ctx, slot->ctx_sampling);
                            }

                            json result = parallelCompletionResultJson(slot->parent_ctx, captureParallelCompletionResult(slot));
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            invokeAsyncTracked(callInvoker, contextId, [callbacks, contextId, result, runtime](bool shouldProceed) {
                                if (!shouldProceed) return;
                                if (!g_llamaContexts.get(contextId)) return;
                                auto& rt = *runtime;
                                callbacks.onComplete->call(rt, fromJson(rt, result));
                            });
                        }
                    };

                    int requestId = ctx->slot_manager->reserve_request_id();
                    RequestManager::getInstance().addRequest(contextId, requestId, {onToken, onComplete, nullptr});
                    try {
                        int queuedRequestId = ctx->slot_manager->queue_request(
                            cparams, tokens, mediaPaths, cparams.prompt, chat_format, reasoning_format, generation_prompt, chat_parser, prefill_text, load_state_path, save_state_path, save_prompt_state_path, load_state_size, save_state_size,
                            tokenCallback, completeCallback, requestId
                        );
                        if (queuedRequestId != requestId) {
                            RequestManager::getInstance().takeRequest(contextId, requestId);
                            throw std::runtime_error("Failed to queue completion request");
                        }
                    } catch (...) {
                        RequestManager::getInstance().takeRequest(contextId, requestId);
                        throw;
                    }

                    return [requestId](jsi::Runtime& rt) {
                        return fromJson(rt, json::object({{"requestId", requestId}}));
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaQueueCompletion", queueCompletion);

        auto cancelRequest = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaCancelRequest"),
            2,
            [](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int requestId = (int)arguments[1].asNumber();

                auto ctx = getContextOrThrow(contextId);
                if (ctx->slot_manager) {
                    auto result = ctx->slot_manager->cancel_request(requestId);
                    if (result == rnllama::llama_rn_cancel_result::QUEUED) {
                        auto callbacks = RequestManager::getInstance().takeRequest(contextId, requestId);
                        if (callbacks.onComplete) {
                            json response = parallelCompletionResultJson(ctx, createQueuedCancellationSnapshot(requestId));
                            callbacks.onComplete->call(runtime, fromJson(runtime, response));
                        }
                    }
                }

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
                json params = toJson(runtime, arguments[2]);
                auto onResult = makeJsiFunction(runtime, arguments[3], callInvoker);

                // Absent -> keep the context's embd_normalize
                int embd_normalize = 0;
                bool has_embd_normalize = false;
                if (auto it = params.find("embd_normalize"); it != params.end() && it->is_number()) {
                    embd_normalize = it->get<int>();
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
                        auto callbacks = RequestManager::getInstance().takeRequest(contextId, requestId);
                        if (callbacks.onResult) {
                            std::vector<float> embCopy = embedding;
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            invokeAsyncTracked(callInvoker, contextId, [callbacks, embCopy, runtime](bool shouldProceed) {
                                if (!shouldProceed) return;
                                auto& rt = *runtime;
                                callbacks.onResult->call(rt, makeFloat32Array(rt, embCopy));
                            });
                        }
                    };

                    const int normalize = has_embd_normalize ? embd_normalize : ctx->params.embd_normalize;
                    int requestId = ctx->slot_manager->reserve_request_id();
                    RequestManager::getInstance().addRequest(contextId, requestId, {nullptr, nullptr, onResult});
                    try {
                        int queuedRequestId = ctx->slot_manager->queue_embedding_request(
                            tokens, normalize, resultCallback, requestId
                        );
                        if (queuedRequestId != requestId) {
                            RequestManager::getInstance().takeRequest(contextId, requestId);
                            throw std::runtime_error("Failed to queue embedding request");
                        }
                    } catch (...) {
                        RequestManager::getInstance().takeRequest(contextId, requestId);
                        throw;
                    }

                    return [requestId](jsi::Runtime& rt) {
                        return fromJson(rt, json::object({{"requestId", requestId}}));
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
                json params = toJson(runtime, arguments[3]);
                auto onResult = makeJsiFunction(runtime, arguments[4], callInvoker);

                int normalize = getPropertyAsInt(params, "normalize", 0);

                return createPromiseTask(runtime, callInvoker, [runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){}), contextId, query, documents, normalize, onResult, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    auto resultCallback = [contextId, callInvoker, runtimePtr](int32_t requestId, const std::vector<float>& scores) {
                        auto callbacks = RequestManager::getInstance().takeRequest(contextId, requestId);
                        if (callbacks.onResult) {
                            json result = rerankResultJson(scores);
                            auto runtime = runtimePtr;
                            if (!runtime) {
                              return;
                            }
                            invokeAsyncTracked(callInvoker, contextId, [callbacks, result, runtime](bool shouldProceed) {
                                if (!shouldProceed) return;
                                auto& rt = *runtime;
                                callbacks.onResult->call(rt, fromJson(rt, result));
                            });
                        }
                    };

                    int requestId = ctx->slot_manager->reserve_request_id();
                    RequestManager::getInstance().addRequest(contextId, requestId, {nullptr, nullptr, onResult});
                    try {
                        int queuedRequestId = ctx->slot_manager->queue_rerank_request(
                            query, documents, normalize, resultCallback, requestId
                        );
                        if (queuedRequestId != requestId) {
                            RequestManager::getInstance().takeRequest(contextId, requestId);
                            throw std::runtime_error("Failed to queue rerank request");
                        }
                    } catch (...) {
                        RequestManager::getInstance().takeRequest(contextId, requestId);
                        throw;
                    }

                    return [requestId](jsi::Runtime& rt) {
                        return fromJson(rt, json::object({{"requestId", requestId}}));
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaQueueRerank", queueRerank);

        // Get parallel status (one-time snapshot)
        auto getParallelStatus = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetParallelStatus"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    json result = parallelStatusJson(ctx->slot_manager->get_status());

                    return [result](jsi::Runtime& rt) {
                        return fromJson(rt, result);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetParallelStatus", getParallelStatus);

        // Subscribe to parallel status changes
        auto subscribeParallelStatus = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaSubscribeParallelStatus"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                auto onStatus = makeJsiFunction(runtime, arguments[1], callInvoker);

                auto runtimePtr = std::make_shared<jsi::Runtime*>(&runtime);

                return createPromiseTask(runtime, callInvoker,
                    [contextId, onStatus, callInvoker, runtimePtr]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->parallel_mode_enabled || !ctx->slot_manager) {
                        throw std::runtime_error("Parallel mode not enabled");
                    }

                    auto statusCallback = [contextId, callInvoker, onStatus, runtimePtr](
                        const rnllama::llama_rn_parallel_status& status
                    ) {
                        json result = parallelStatusJson(status);

                        callInvoker->invokeAsync([onStatus, result, runtimePtr]() {
                            if (!runtimePtr || !*runtimePtr) return;
                            auto& rt = **runtimePtr;
                            onStatus->call(rt, fromJson(rt, result));
                        });
                    };

                    int32_t subscriberId = ctx->slot_manager->add_status_subscriber(statusCallback);

                    return [subscriberId](jsi::Runtime& rt) {
                        return fromJson(rt, json::object({{"subscriberId", subscriberId}}));
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaSubscribeParallelStatus", subscribeParallelStatus);

        // Unsubscribe from parallel status changes
        auto unsubscribeParallelStatus = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaUnsubscribeParallelStatus"),
            2,
            [](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int subscriberId = (int)arguments[1].asNumber();

                long ctxPtr = g_llamaContexts.get(contextId);
                if (ctxPtr) {
                    auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                    if (ctx->slot_manager) {
                        ctx->slot_manager->remove_status_subscriber(subscriberId);
                    }
                }

                return jsi::Value::undefined();
            }
        );
        runtime.global().setProperty(runtime, "llamaUnsubscribeParallelStatus", unsubscribeParallelStatus);

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

                     // Wait for ALL other tasks on this context to complete (including their
                     // invokeAsync callbacks) before deleting. This prevents race conditions
                     // where we delete the context while a completion's JS callback is still
                     // accessing ctx->completion.
                     TaskManager::getInstance().waitForContext(contextId, 0);
                     if (TaskManager::getInstance().isShuttingDown()) {
                         return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                     }

                     if (ctxPtr) {
                         auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ctxPtr);
                         // Remove from map FIRST, then delete.
                         // This ensures any concurrent lookups via g_llamaContexts.get()
                         // will return 0 (not found) rather than a dangling pointer.
                         removeContext(contextId);
                         delete ctx;
                     }
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 }, contextId, false);  // trackTask=false - release should not count itself
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

                     // Wait for ALL tasks to complete (including their invokeAsync callbacks)
                     TaskManager::getInstance().waitForAll(0);
                     if (TaskManager::getInstance().isShuttingDown()) {
                         return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                     }

                     g_llamaContexts.clear([](long ptr) {
                        if (ptr) {
                            auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                            delete ctx;
                        }
                     });
                     return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                 }, -1, false);  // contextId=-1 (not tracked), trackTask=false
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
                std::vector<common_adapter_lora_info> lora_adapters = parseLoraAdapters(toJson(runtime, arguments[1]));

                return createPromiseTask(runtime, callInvoker, [contextId, lora_adapters]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    ctx->applyLoraAdapters(lora_adapters);
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
                    throwIfContextBusy(ctx);
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
                    json adapters = json::array();
                    for (const auto& la : ctx->getLoadedLoraAdapters()) {
                        adapters.push_back(json::object({{"path", la.path}, {"scaled", (double) la.scale}}));
                    }
                    return [adapters](jsi::Runtime& rt) {
                        return fromJson(rt, adapters);
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
                auto opts = optionsFromJson<MultimodalInitOptions>(runtime, toJson(runtime, arguments[1]), "initMultimodal");

                return createPromiseTask(runtime, callInvoker, [contextId, opts]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    bool result = ctx->initMultimodal(opts.path, opts.use_gpu, opts.image_min_tokens, opts.image_max_tokens);
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
                    json support = json::object({
                        {"vision", ctx->isMultimodalSupportVision()},
                        {"audio", ctx->isMultimodalSupportAudio()},
                    });
                    return [support](jsi::Runtime& rt) {
                        return fromJson(rt, support);
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
                    throwIfContextBusy(ctx);
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
                json params = toJson(runtime, arguments[1]);
                auto opts = optionsFromJson<VocoderInitOptions>(runtime, params, "initVocoder");
                // use_gpu defaults to follow the main context's n_gpu_layers
                // (any > 0 means the backbone is GPU-offloaded — pair the
                // codec / codec_lm there too unless the caller overrides).
                if (!params.contains("use_gpu")) {
                    opts.use_gpu = getContextOrThrow(contextId)->params.n_gpu_layers > 0;
                }

                return createPromiseTask(runtime, callInvoker, [contextId, opts]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
                    bool result = ctx->initVocoder(opts.path, opts.n_batch, opts.use_gpu);
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
            4,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                // Speaker payload arrives as a JS object, or null when the
                // caller relies on a registered speaker id.
                json speaker = arguments[1].isObject() ? toJson(runtime, arguments[1]) : json(nullptr);
                std::string textToSpeak = arguments[2].asString(runtime).utf8(runtime);
                // Optional 4th arg: speakerId (registry id >= 0, or -1 for none).
                int speakerId = (count >= 4 && arguments[3].isNumber())
                    ? (int)arguments[3].asNumber()
                    : -1;

                return createPromiseTask(runtime, callInvoker, [contextId, speaker, textToSpeak, speakerId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_result = ctx->tts_wrapper->getFormattedAudioCompletion(ctx, speaker, textToSpeak, speakerId);
                        json res = json::object({{"prompt", audio_result.prompt}});
                        if (!audio_result.grammar.empty()) {
                            res["grammar"] = audio_result.grammar;
                        }
                        res["embedding"] = audio_result.embedding;
                        res["flow"] = audio_result.flow;
                        return [res](jsi::Runtime& rt) {
                            return fromJson(rt, res);
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetFormattedAudioCompletion", getFormattedAudioCompletion);

        auto getTTSCapabilities = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetTTSCapabilities"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");
                    auto cap = ctx->tts_wrapper->getTTSCapabilities(ctx);
                    json res = json::object({
                        {"type", cap.type},
                        {"promptKind", cap.prompt_kind},
                        {"family", cap.family},
                        {"requiresPhonemes", cap.requires_phonemes},
                        {"defaultLanguage", cap.default_language},
                    });
                    return [res](jsi::Runtime& rt) {
                        return fromJson(rt, res);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetTTSCapabilities", getTTSCapabilities);

        auto decodeAudioTokens = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDecodeAudioTokens"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                // Int32Array | number[]
                std::vector<llama_token> tokens = toInt32Vector(runtime, arguments[1]);

                return createPromiseTask(runtime, callInvoker, [contextId, tokens]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_data = ctx->tts_wrapper->decodeAudioTokens(ctx, tokens);
                        return [audio_data](jsi::Runtime& rt) {
                            return makeFloat32Array(rt, audio_data);
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDecodeAudioTokens", decodeAudioTokens);

        // generateAudioCodes — drives the backbone + codec_lm AR loop for
        // codec_lm-flow models (CSM, etc.).  Args:
        //   (contextId, opts, onFrame?)
        // opts: { prompt, maxFrames?, temperature?, topP?, topK?, seed? }
        // onFrame:  optional (step:number, codes:number[]) => void — fired
        //           per-frame as audio codes are produced.
        // Returns { codes:number[], nCodebook, nFrames, stoppedOnEos, aborted }.
        auto generateAudioCodes = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGenerateAudioCodes"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                auto opts = optionsFromJson<rnllama::llama_rn_audio_codes_options>(
                    runtime, toJson(runtime, arguments[1]), "generateAudioCodes");

                std::shared_ptr<jsi::Function> onFrame;
                if (count >= 3 && arguments[2].isObject() &&
                    arguments[2].asObject(runtime).isFunction(runtime)) {
                    onFrame = std::make_shared<jsi::Function>(
                        arguments[2].asObject(runtime).asFunction(runtime));
                }
                jsi::Runtime * runtimePtr = &runtime;

                return createPromiseTask(runtime, callInvoker, [contextId, opts, onFrame, runtimePtr, callInvoker]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");
                    if (opts.prompt.empty()) {
                        throw std::runtime_error("generateAudioCodes: prompt is empty");
                    }

                    rnllama::llama_rn_audio_codes_progress_cb cb;
                    if (onFrame) {
                        // Fire-and-forget per-frame notification.  We never
                        // block on the JS side, so the return value is
                        // always "continue"; aborting from JS isn't wired
                        // through here yet.
                        cb = [onFrame, runtimePtr, callInvoker](int step, const std::vector<int32_t> &codes) -> bool {
                            std::vector<int32_t> codes_copy = codes;
                            callInvoker->invokeAsync([onFrame, runtimePtr, step, codes_copy]() {
                                auto &rt = *runtimePtr;
                                jsi::Array arr(rt, codes_copy.size());
                                for (size_t i = 0; i < codes_copy.size(); ++i) {
                                    arr.setValueAtIndex(rt, i, (double) codes_copy[i]);
                                }
                                onFrame->call(rt, jsi::Value((double) step), arr);
                            });
                            return true;
                        };
                    }

                    try {
                        auto r = ctx->tts_wrapper->generateAudioCodes(ctx, opts, cb);
                        json res = json::object({
                            {"codes", r.codes},
                            {"nCodebook", r.n_codebook},
                            {"nFrames", r.n_frames},
                            {"stoppedOnEos", r.stopped_on_eos},
                            {"aborted", r.aborted},
                        });
                        return [res](jsi::Runtime& rt) {
                            return fromJson(rt, res);
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGenerateAudioCodes", generateAudioCodes);

        // llamaCreateSpeaker(ctxId, pcm, opts)
        //   pcm:  Float32Array | number[] (reference audio samples)
        //   opts: { inputSampleRate: number, refText?: string,
        //           bake?: boolean, emotion?: number }
        // Resolves: { id: number, family: string, rows: number, baked: boolean }
        auto createSpeaker = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaCreateSpeaker"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                std::vector<float> pcm = toFloatVector(runtime, arguments[1]);
                auto opts = optionsFromJson<SpeakerOptions>(
                    runtime, count > 2 ? toJson(runtime, arguments[2]) : json::object(), "createSpeaker");

                return createPromiseTask(runtime, callInvoker, [contextId, pcm, opts]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    auto cap = ctx->tts_wrapper->getTTSCapabilities(ctx);
                    std::string family = cap.family;

                    // Without an explicit emotion the speaker keeps rn_speaker's
                    // default; the encoder only reads it when has_emotion is set.
                    const bool has_emotion = opts.hasEmotion();
                    const int speakerId = ctx->tts_wrapper->createSpeaker(
                        ctx, pcm, opts.inputSampleRate, opts.refText,
                        has_emotion ? opts.emotion : 0.5f, has_emotion, opts.bake);

                    const rnllama::rn_speaker * spk = ctx->tts_wrapper->getSpeaker(speakerId);
                    json res = json::object({
                        {"id", speakerId},
                        {"family", family},
                        {"rows", spk ? spk->rows : 0},
                        {"baked", spk ? spk->baked : false},
                    });

                    return [res](jsi::Runtime& rt) {
                        return fromJson(rt, res);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaCreateSpeaker", createSpeaker);

        // llamaBakeSpeaker(ctxId, speakerId)
        // Resolves: { rows: number, baked: boolean }
        auto bakeSpeaker = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaBakeSpeaker"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int speakerId = (int)arguments[1].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId, speakerId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    ctx->tts_wrapper->bakeSpeaker(ctx, speakerId);

                    const rnllama::rn_speaker * spk = ctx->tts_wrapper->getSpeaker(speakerId);
                    if (!spk) throw std::runtime_error("bakeSpeaker: speaker id not found");

                    json res = json::object({{"rows", spk->rows}, {"baked", spk->baked}});
                    return [res](jsi::Runtime& rt) {
                        return fromJson(rt, res);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaBakeSpeaker", bakeSpeaker);

        // llamaReleaseSpeaker(ctxId, speakerId)
        // Resolves: void
        auto releaseSpeaker = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaReleaseSpeaker"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                int speakerId = (int)arguments[1].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId, speakerId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    ctx->tts_wrapper->releaseSpeaker(speakerId);

                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseSpeaker", releaseSpeaker);

        auto decodeAudioEmbeddings = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaDecodeAudioEmbeddings"),
            3,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                // Float32Array | number[]
                std::vector<float> embeddings = toFloatVector(runtime, arguments[1]);
                int embeddingDim = (int)arguments[2].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId, embeddings, embeddingDim]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    try {
                        auto audio_data = ctx->tts_wrapper->decodeAudioEmbeddings(ctx, embeddings, embeddingDim);
                        return [audio_data](jsi::Runtime& rt) {
                            return makeFloat32Array(rt, audio_data);
                        };
                    } catch (const std::exception &e) {
                        throw std::runtime_error(e.what());
                    }
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaDecodeAudioEmbeddings", decodeAudioEmbeddings);

        auto getAudioSampleRate = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaGetAudioSampleRate"),
            1,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();

                return createPromiseTask(runtime, callInvoker, [contextId]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    if (!ctx->isVocoderEnabled()) throw std::runtime_error("Vocoder is not enabled");

                    const int sample_rate = ctx->tts_wrapper->getAudioSampleRate();
                    return [sample_rate](jsi::Runtime& rt) {
                        return jsi::Value((double)sample_rate);
                    };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaGetAudioSampleRate", getAudioSampleRate);

        // Cache management
        auto clearCache = jsi::Function::createFromHostFunction(runtime,
            jsi::PropNameID::forAscii(runtime, "llamaClearCache"),
            2,
            [callInvoker](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments, size_t count) -> jsi::Value {
                int contextId = (int)arguments[0].asNumber();
                bool clearData = count > 1 && arguments[1].isBool() ? arguments[1].asBool() : false;
                return createPromiseTask(runtime, callInvoker, [contextId, clearData]() -> PromiseResultGenerator {
                    auto ctx = getContextOrThrow(contextId);
                    throwIfContextBusy(ctx);
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
                    throwIfContextBusy(ctx);
                    ctx->releaseVocoder();
                    return [](jsi::Runtime& rt) { return jsi::Value::undefined(); };
                }, contextId);
            }
        );
        runtime.global().setProperty(runtime, "llamaReleaseVocoder", releaseVocoder);
    }

    void cleanupJSIBindings() {
        TaskManager::getInstance().beginShutdown();
        {
            std::lock_guard<std::mutex> lock(g_log_mutex);
            g_log_handler.reset();
            g_log_invoker.reset();
            g_log_runtime.reset();
        }
        llama_log_set(llama_log_callback_default, nullptr);

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

        if (contexts.empty()) {
            g_context_limit.store(-1);
            return;
        }
        ThreadPool::getInstance().shutdown();

        g_llamaContexts.clear([](long ptr) {
            if (ptr) {
                auto ctx = reinterpret_cast<rnllama::llama_rn_context*>(ptr);
                delete ctx;
            }
        });
        g_context_limit.store(-1);
    }
}
