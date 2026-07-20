#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include "market_event.hpp"

namespace sc {

// Lock-free SPSC ring buffer with power-of-2 capacity (one producer, one consumer).
template <std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr std::size_t kMask = Capacity - 1;

public:
    RingBuffer() : head_(0), tail_(0) {}

    bool push(const MarketEvent& event) {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t current_head = head_.load(std::memory_order_acquire);
        if (current_tail - current_head >= Capacity) {
            return false;
        }
        buf_[current_tail & kMask] = event;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(MarketEvent& out) {
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);
        if (current_head == current_tail) {
            return false;
        }
        out = buf_[current_head & kMask];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    std::size_t size() const {
        return tail_.load(std::memory_order_relaxed) - head_.load(std::memory_order_relaxed);
    }

    bool empty() const { return size() == 0; }
    bool full() const { return size() >= Capacity; }

private:
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
    std::array<MarketEvent, Capacity> buf_;
};

using DefaultRingBuffer = RingBuffer<1u << 20>;

}  // namespace sc
