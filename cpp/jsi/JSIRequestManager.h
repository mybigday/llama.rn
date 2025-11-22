#pragma once

#include <jsi/jsi.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <memory>

using namespace facebook;

namespace rnllama_jsi {

    struct JSIRequestCallbacks {
        std::shared_ptr<jsi::Function> onToken;
        std::shared_ptr<jsi::Function> onComplete;
        std::shared_ptr<jsi::Function> onResult; // For embedding/rerank
    };

    class RequestManager {
    private:
        std::unordered_map<std::string, JSIRequestCallbacks> requests;
        std::mutex mutex;

        std::string getKey(int contextId, int requestId) {
            return std::to_string(contextId) + ":" + std::to_string(requestId);
        }

    public:
        void addRequest(int contextId, int requestId, JSIRequestCallbacks callbacks) {
            std::lock_guard<std::mutex> lock(mutex);
            requests[getKey(contextId, requestId)] = callbacks;
        }

        void removeRequest(int contextId, int requestId) {
            std::lock_guard<std::mutex> lock(mutex);
            requests.erase(getKey(contextId, requestId));
        }

        void clearContext(int contextId) {
            std::lock_guard<std::mutex> lock(mutex);
            const std::string prefix = std::to_string(contextId) + ":";
            for (auto it = requests.begin(); it != requests.end();) {
                if (it->first.rfind(prefix, 0) == 0) {
                    it = requests.erase(it);
                } else {
                    ++it;
                }
            }
        }

        void clearAll() {
            std::lock_guard<std::mutex> lock(mutex);
            requests.clear();
        }

        JSIRequestCallbacks getRequest(int contextId, int requestId) {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = requests.find(getKey(contextId, requestId));
            if (it != requests.end()) {
                return it->second;
            }
            return {nullptr, nullptr, nullptr};
        }
        
        static RequestManager& getInstance() {
            static RequestManager instance;
            return instance;
        }
    };

}
