#include "JSIUtils.h"
#include "JSITaskManager.h"
#include "ThreadPool.h"

namespace rnllama_jsi {

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
                    
                    auto resolve = std::make_shared<jsi::Function>(arguments[0].asObject(runtime).asFunction(runtime));
                    auto reject = std::make_shared<jsi::Function>(arguments[1].asObject(runtime).asFunction(runtime));

                    if (trackTask) {
                        TaskManager::getInstance().startTask(contextId);
                    }

                    ThreadPool::getInstance().enqueue([callInvoker, task, resolve, reject, contextId, trackTask, runtimePtr]() {
                        // Track whether we successfully scheduled the invokeAsync callback.
                        // The task should only be marked complete after the JS callback finishes,
                        // not when the thread pool work completes - this prevents race conditions
                        // where release() deletes the context before the JS callback runs.
                        bool invokeScheduled = false;

                        try {
                            auto resultGenerator = task();
                            try {
                                callInvoker->invokeAsync([resolve, resultGenerator, runtimePtr, contextId, trackTask]() {
                                    // Finish task AFTER the JS callback completes (when resultGenerator runs)
                                    TaskFinishGuard guard(contextId, trackTask);
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
                                callInvoker->invokeAsync([reject, msg, runtimePtr, contextId, trackTask]() {
                                    TaskFinishGuard guard(contextId, trackTask);
                                    auto& rt = *runtimePtr;
                                    reject->call(rt, jsi::String::createFromUtf8(rt, msg));
                                });
                                invokeScheduled = true;
                            } catch (...) {
                                // invokeAsync failed
                            }
                        } catch (...) {
                            try {
                                callInvoker->invokeAsync([reject, runtimePtr, contextId, trackTask]() {
                                    TaskFinishGuard guard(contextId, trackTask);
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
                        if (!invokeScheduled && trackTask) {
                            TaskManager::getInstance().finishTask(contextId);
                        }
                    });

                    return jsi::Value::undefined();
                }
            )
        );
    }
}
