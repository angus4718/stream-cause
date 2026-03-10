#include "thread_pool.hpp"

namespace sc {

ThreadPool::ThreadPool(int n_threads) {
    // Handle n_threads == 0 gracefully (use hardware_concurrency).
    workers_.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
    // task into queue; notify one worker.
    auto pt = std::make_shared<std::packaged_task<void()>>(std::move(task));
    std::future<void> fut = pt->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        ++active_tasks_;
        task_queue_.push([pt]() { (*pt)(); });
    }
    cv_.notify_one();
    return fut;
}

void ThreadPool::wait_all() {
    // Use a separate condition variable or spin with a small sleep.
    while (active_tasks_.load() > 0) {
        std::this_thread::yield();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this] { return stop_.load() || !task_queue_.empty(); });
            if (stop_.load() && task_queue_.empty()) return;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }
        task();
        --active_tasks_;
    }
}

} // namespace sc
