#include "JSIUtils.h"
#include "JSITaskManager.h"
#include "ThreadPool.h"

namespace rnllama_jsi {

    JsiFunctionPtr makeJsiFunction(
        jsi::Runtime& runtime,
        const jsi::Value& value,
        std::shared_ptr<react::CallInvoker> callInvoker
    ) {
        if (!value.isObject() || !value.asObject(runtime).isFunction(runtime)) {
            return nullptr;
        }
        auto fn = new jsi::Function(value.asObject(runtime).asFunction(runtime));
        std::weak_ptr<react::CallInvoker> weakInvoker = callInvoker;

        return JsiFunctionPtr(fn, [weakInvoker](jsi::Function* ptr) {
            if (!ptr) {
                return;
            }
            auto invoker = weakInvoker.lock();
            if (!invoker || TaskManager::getInstance().isShuttingDown()) {
                return;
            }
            try {
                invoker->invokeAsync([ptr]() {
                    delete ptr;
                });
            } catch (...) {
                // Runtime may be shutting down; leak to avoid crashing on a non-JS thread.
            }
        });
    }

    jsi::Value createPromiseTask(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker,
        PromiseTask task,
        int contextId,
        bool trackTask
    ) {
        auto PromiseConstructor = runtime.global().getPropertyAsObject(runtime, "Promise").asFunction(runtime);
        auto runtimePtr = std::shared_ptr<jsi::Runtime>(&runtime, [](jsi::Runtime*){});

        return PromiseConstructor.callAsConstructor(runtime,
            jsi::Function::createFromHostFunction(runtime,
                jsi::PropNameID::forAscii(runtime, "executor"),
                2,
                [callInvoker, task, contextId, trackTask, runtimePtr](jsi::Runtime& runtime,
                                  const jsi::Value& thisValue,
                                  const jsi::Value* arguments,
                                  size_t count) -> jsi::Value {

                    auto resolve = makeJsiFunction(runtime, arguments[0], callInvoker);
                    auto reject = makeJsiFunction(runtime, arguments[1], callInvoker);
                    bool shouldTrack = trackTask && !TaskManager::getInstance().isShuttingDown();

                    if (shouldTrack) {
                        TaskManager::getInstance().startTask(contextId);
                    }

                    ThreadPool::getInstance().enqueue([callInvoker, task, resolve, reject, contextId, shouldTrack, runtimePtr]() {
                        // Track whether we successfully scheduled the invokeAsync callback.
                        // The task should only be marked complete after the JS callback finishes,
                        // not when the thread pool work completes - this prevents race conditions
                        // where release() deletes the context before the JS callback runs.
                        bool invokeScheduled = false;

                        try {
                            if (TaskManager::getInstance().isShuttingDown()) {
                                if (shouldTrack) {
                                    TaskManager::getInstance().finishTask(contextId);
                                }
                                return;
                            }
                            auto resultGenerator = task();
                            if (TaskManager::getInstance().isShuttingDown()) {
                                if (shouldTrack) {
                                    TaskManager::getInstance().finishTask(contextId);
                                }
                                return;
                            }
                            try {
                                callInvoker->invokeAsync([resolve, resultGenerator, runtimePtr, contextId, shouldTrack]() {
                                    if (TaskManager::getInstance().isShuttingDown()) {
                                        TaskFinishGuard guard(contextId, shouldTrack);
                                        return;
                                    }
                                    // Finish task AFTER the JS callback completes (when resultGenerator runs)
                                    TaskFinishGuard guard(contextId, shouldTrack);
                                    auto& rt = *runtimePtr;
                                    resolve->call(rt, resultGenerator(rt));
                                });
                                invokeScheduled = true;
                            } catch (...) {
                                // invokeAsync failed (e.g., React Native shutting down)
                                // Fall through to finish task below
                            }
                        } catch (const std::exception& e) {
                            std::string msg = e.what();
                            try {
                                if (TaskManager::getInstance().isShuttingDown()) {
                                    if (shouldTrack) {
                                        TaskManager::getInstance().finishTask(contextId);
                                    }
                                    return;
                                }
                                callInvoker->invokeAsync([reject, msg, runtimePtr, contextId, shouldTrack]() {
                                    if (TaskManager::getInstance().isShuttingDown()) {
                                        TaskFinishGuard guard(contextId, shouldTrack);
                                        return;
                                    }
                                    TaskFinishGuard guard(contextId, shouldTrack);
                                    auto& rt = *runtimePtr;
                                    reject->call(rt, jsi::String::createFromUtf8(rt, msg));
                                });
                                invokeScheduled = true;
                            } catch (...) {
                                // invokeAsync failed
                            }
                        } catch (...) {
                            try {
                                if (TaskManager::getInstance().isShuttingDown()) {
                                    if (shouldTrack) {
                                        TaskManager::getInstance().finishTask(contextId);
                                    }
                                    return;
                                }
                                callInvoker->invokeAsync([reject, runtimePtr, contextId, shouldTrack]() {
                                    if (TaskManager::getInstance().isShuttingDown()) {
                                        TaskFinishGuard guard(contextId, shouldTrack);
                                        return;
                                    }
                                    TaskFinishGuard guard(contextId, shouldTrack);
                                    auto& rt = *runtimePtr;
                                    reject->call(rt, jsi::String::createFromUtf8(rt, "Unknown error"));
                                });
                                invokeScheduled = true;
                            } catch (...) {
                                // invokeAsync failed
                            }
                        }

                        // If we couldn't schedule invokeAsync, finish the task now to prevent
                        // waitForContext from hanging forever
                        if (!invokeScheduled && shouldTrack) {
                            TaskManager::getInstance().finishTask(contextId);
                        }
                    });

                    return jsi::Value::undefined();
                }
            )
        );
    }

    void invokeAsyncTracked(
        std::shared_ptr<react::CallInvoker> callInvoker,
        int contextId,
        std::function<void(bool shouldProceed)> callback
    ) {
        bool shouldTrack = !TaskManager::getInstance().isShuttingDown();
        if (shouldTrack) {
            TaskManager::getInstance().startTask(contextId);
        }
        try {
            callInvoker->invokeAsync([contextId, shouldTrack, callback]() {
                TaskFinishGuard guard(contextId, shouldTrack);
                bool shouldProceed = !TaskManager::getInstance().isShuttingDown();
                callback(shouldProceed);
            });
        } catch (...) {
            // invokeAsync failed - finish task to prevent waitForContext hanging
            if (shouldTrack) {
                TaskManager::getInstance().finishTask(contextId);
            }
        }
    }
}
