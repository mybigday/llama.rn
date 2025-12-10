#pragma once
#include "JSINativeHeaders.h"
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rnllama_jsi {
    template<typename T>
    class ContextManager {
    private:
        std::unordered_map<int, long> contextMap;
        std::mutex contextMutex;

    public:
        void add(int contextId, long contextPtr) {
            std::lock_guard<std::mutex> lock(contextMutex);
            contextMap[contextId] = contextPtr;
        }

        void remove(int contextId) {
            std::lock_guard<std::mutex> lock(contextMutex);
            contextMap.erase(contextId);
        }

        long get(int contextId) {
            std::lock_guard<std::mutex> lock(contextMutex);
            auto it = contextMap.find(contextId);
            return (it != contextMap.end()) ? it->second : 0;
        }

        size_t size() {
            std::lock_guard<std::mutex> lock(contextMutex);
            return contextMap.size();
        }

        std::vector<std::pair<int, long>> snapshot() {
            std::lock_guard<std::mutex> lock(contextMutex);
            std::vector<std::pair<int, long>> items;
            items.reserve(contextMap.size());
            for (const auto& entry : contextMap) {
                items.push_back(entry);
            }
            return items;
        }
        
        void clear(std::function<void(long)> deleter = nullptr) {
            std::lock_guard<std::mutex> lock(contextMutex);
            if (deleter) {
                for (auto& pair : contextMap) {
                    deleter(pair.second);
                }
            }
            contextMap.clear();
        }
    };

    extern ContextManager<rnllama::llama_rn_context> g_llamaContexts;
}
