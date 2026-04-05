#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <vector>

namespace Tokenizer {

class ThreadPool {
public:
    explicit ThreadPool(size_t thread_count);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    template <typename F>
    void submit(F&& task);

    void wait();

    size_t threadCount() const { return threads_.size(); }

private:
    void workerLoop();

    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable completion_cv_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_{0};
};

template <typename F>
void ThreadPool::submit(F&& task) {
    {
        std::lock_guard lock(mutex_);
        tasks_.emplace(std::forward<F>(task));
    }
    cv_.notify_one();
}

}  // namespace Tokenizer
