#include "JSITaskManager.h"
#include <chrono>

namespace rnllama_jsi {

// Timeout for waiting on tasks to complete.
// This prevents indefinite blocking if a task fails to call finishTask().
static constexpr auto TASK_WAIT_TIMEOUT = std::chrono::milliseconds(5000);

TaskManager& TaskManager::getInstance() {
    static TaskManager instance;
    return instance;
}

void TaskManager::startTask(int contextId) {
    std::lock_guard<std::mutex> lock(mutex);
    activeTasks[contextId] += 1;
    totalTasks += 1;
}

void TaskManager::finishTask(int contextId) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = activeTasks.find(contextId);
    if (it != activeTasks.end()) {
        it->second -= 1;
        if (it->second <= 0) {
            activeTasks.erase(it);
        }
    }

    if (totalTasks > 0) {
        totalTasks -= 1;
    }

    cv.notify_all();
}

void TaskManager::beginShutdown() {
    shuttingDown.store(true, std::memory_order_relaxed);
    cv.notify_all();
}

void TaskManager::reset() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        activeTasks.clear();
        totalTasks = 0;
    }
    shuttingDown.store(false, std::memory_order_relaxed);
    cv.notify_all();
}

bool TaskManager::isShuttingDown() const {
    return shuttingDown.load(std::memory_order_relaxed);
}

void TaskManager::waitForContext(int contextId, int targetCount) {
    if (contextId < 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait_for(lock, TASK_WAIT_TIMEOUT, [this, contextId, targetCount]() {
        if (shuttingDown.load(std::memory_order_relaxed)) {
            return true;
        }
        auto it = activeTasks.find(contextId);
        int count = it != activeTasks.end() ? it->second : 0;
        return count <= targetCount;
    });
}

void TaskManager::waitForAll(int targetCount) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait_for(lock, TASK_WAIT_TIMEOUT, [this, targetCount]() {
        if (shuttingDown.load(std::memory_order_relaxed)) {
            return true;
        }
        return totalTasks <= targetCount;
    });
}

TaskFinishGuard::TaskFinishGuard(int contextId, bool tracked)
    : contextId(contextId), tracked(tracked) {}

TaskFinishGuard::~TaskFinishGuard() {
    if (tracked) {
        TaskManager::getInstance().finishTask(contextId);
    }
}

} // namespace rnllama_jsi
