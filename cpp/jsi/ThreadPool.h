#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    static ThreadPool instance;

public:
    static ThreadPool& getInstance() {
        return instance;
    }

    // Ensure worker threads are active; lazily restarts if shut down.
    void ensureRunning();

    // Stop all workers and clear queued tasks.
    void shutdown();

    template<class F>
    void enqueue(F&& f) {
        ensureRunning();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ThreadPool(size_t threads = std::thread::hardware_concurrency());
    ~ThreadPool();

private:
    void startWorkers(size_t threads);
};
