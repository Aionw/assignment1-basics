#include "thread_pool.h"

#include <utility>

namespace Tokenizer {

ThreadPool::ThreadPool(size_t thread_count) {
    threads_.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        threads_.emplace_back(&ThreadPool::workerLoop, this);
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void ThreadPool::workerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this] { return stop_.load(std::memory_order_acquire) || !tasks_.empty(); });
            if (stop_.load(std::memory_order_acquire) && tasks_.empty()) {
                return;
            }
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }
        if (task) {
            ++active_;
            task();
            --active_;
            if (tasks_.empty() && active_.load() == 0) {
                completion_cv_.notify_all();
            }
        }
    }
}

void ThreadPool::wait() {
    std::unique_lock lock(mutex_);
    completion_cv_.wait(lock, [this] { return tasks_.empty() && active_.load() == 0; });
}

}  // namespace Tokenizer
