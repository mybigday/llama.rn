#include "JSITaskManager.h"

namespace rnllama_jsi {

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

void TaskManager::waitForContext(int contextId, int targetCount) {
    if (contextId < 0) {
        return;
    }

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this, contextId, targetCount]() {
        auto it = activeTasks.find(contextId);
        int count = it != activeTasks.end() ? it->second : 0;
        return count <= targetCount;
    });
}

void TaskManager::waitForAll(int targetCount) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this, targetCount]() {
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
