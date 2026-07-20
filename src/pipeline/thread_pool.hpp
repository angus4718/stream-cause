#pragma once
#include <functional>
#include <future>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace sc {

// Fixed-size thread pool; submit() returns a future for task completion.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();

    std::future<void> submit(std::function<void()> task);
    void wait_all();
    int n_threads() const { return static_cast<int>(workers_.size()); }

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_tasks_{0};
};

}  // namespace sc
