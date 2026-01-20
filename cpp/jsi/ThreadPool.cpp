#include "ThreadPool.h"
#include <system_error>

ThreadPool ThreadPool::instance;

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    startWorkers(threads);
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::startWorkers(size_t threads) {
    if (threads == 0) {
        threads = 1;
    }
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty())
                        return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

void ThreadPool::ensureRunning() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (stop) {
        stop = false;
    }
    if (!workers.empty()) {
        return;
    }
    startWorkers(std::thread::hardware_concurrency());
}

void ThreadPool::shutdown() {
    std::unique_lock<std::mutex> shutdown_lock(shutdown_mutex);
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (stop && workers.empty()) {
            return;
        }
        stop = true;
    }
    condition.notify_all();
    const auto self_id = std::this_thread::get_id();
    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            if (worker.get_id() == self_id) {
                worker.detach();
                continue;
            }
            try {
                worker.join();
            } catch (const std::system_error&) {
                worker.detach();
            }
        }
    }
    workers.clear();

    // Clear queued tasks.
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        std::queue<std::function<void()>> empty;
        std::swap(tasks, empty);
    }
}
