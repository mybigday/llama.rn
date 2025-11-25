#pragma once

#include <condition_variable>
#include <mutex>
#include <unordered_map>

namespace rnllama_jsi {

    class TaskManager {
    public:
        static TaskManager& getInstance();

        void startTask(int contextId);
        void finishTask(int contextId);

        // Wait until all tracked tasks for the given context complete.
        void waitForContext(int contextId, int targetCount = 0);

        // Wait until no tracked tasks remain.
        void waitForAll(int targetCount = 0);

    private:
        TaskManager() = default;

        std::mutex mutex;
        std::condition_variable cv;
        std::unordered_map<int, int> activeTasks;
        int totalTasks = 0;
    };

    // RAII helper to ensure finishTask is called even when exceptions are thrown.
    class TaskFinishGuard {
    public:
        TaskFinishGuard(int contextId, bool tracked);
        ~TaskFinishGuard();

    private:
        int contextId;
        bool tracked;
    };

} // namespace rnllama_jsi
