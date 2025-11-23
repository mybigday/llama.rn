#include "JSIUtils.h"
#include "ThreadPool.h"

namespace rnllama_jsi {

    jsi::Value createPromiseTask(
        jsi::Runtime& runtime,
        std::shared_ptr<react::CallInvoker> callInvoker,
        PromiseTask task
    ) {
        auto PromiseConstructor = runtime.global().getPropertyAsObject(runtime, "Promise").asFunction(runtime);

        return PromiseConstructor.callAsConstructor(runtime,
            jsi::Function::createFromHostFunction(runtime,
                jsi::PropNameID::forAscii(runtime, "executor"),
                2,
                [callInvoker, task](jsi::Runtime& runtime,
                                  const jsi::Value& thisValue,
                                  const jsi::Value* arguments,
                                  size_t count) -> jsi::Value {
                    
                    auto resolve = std::make_shared<jsi::Function>(arguments[0].asObject(runtime).asFunction(runtime));
                    auto reject = std::make_shared<jsi::Function>(arguments[1].asObject(runtime).asFunction(runtime));

                    ThreadPool::getInstance().enqueue([callInvoker, task, resolve, reject]() {
                        try {
                            auto resultGenerator = task();
                            callInvoker->invokeAsync([resolve, resultGenerator](jsi::Runtime& rt) {
                                resolve->call(rt, resultGenerator(rt));
                            });
                        } catch (const std::exception& e) {
                            std::string msg = e.what();
                            callInvoker->invokeAsync([reject, msg](jsi::Runtime& rt) {
                                reject->call(rt, jsi::String::createFromUtf8(rt, msg));
                            });
                        } catch (...) {
                            callInvoker->invokeAsync([reject](jsi::Runtime& rt) {
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
