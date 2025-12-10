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
                        TaskFinishGuard guard(contextId, trackTask);
                        try {
                            auto resultGenerator = task();
                            callInvoker->invokeAsync([resolve, resultGenerator, runtimePtr]() {
                                auto& rt = *runtimePtr;
                                resolve->call(rt, resultGenerator(rt));
                            });
                        } catch (const std::exception& e) {
                            std::string msg = e.what();
                            callInvoker->invokeAsync([reject, msg, runtimePtr]() {
                                auto& rt = *runtimePtr;
                                reject->call(rt, jsi::String::createFromUtf8(rt, msg));
                            });
                        } catch (...) {
                            callInvoker->invokeAsync([reject, runtimePtr]() {
                                auto& rt = *runtimePtr;
                                reject->call(rt, jsi::String::createFromUtf8(rt, "Unknown error"));
                            });
                        }
                    });

                    return jsi::Value::undefined();
                }
            )
        );
    }
}
