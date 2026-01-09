#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include "market_event.hpp"

namespace sc {

// Lock-free SPSC ring buffer using power-of-2 capacity.
// One producer thread (ingestion) and one consumer thread (Hawkes EM).
// Uses acquire/release semantics on head_/tail_ to avoid data races.
//
// wrap past capacity within a single epoch given 2^20 capacity and
// realistic event rates).
template <std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    static constexpr std::size_t kMask = Capacity - 1;

public:
    RingBuffer() : head_(0), tail_(0) {}

    // Push an event. Returns false if buffer is full (producer should back off).
    bool push(const MarketEvent& event) {
        // Hint: load tail_ relaxed, load head_ acquire, check full
        // write buf_[tail & mask], store tail_ release.
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        size_t current_head = head_.load(std::memory_order_acquire);
        if (current_tail - current_head >= Capacity){
            return false;
        }
        buf_[current_tail & kMask] = event;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Pop an event into out. Returns false if buffer is empty.
    bool pop(MarketEvent& out) {
        // Hint: load head_ relaxed, load tail_ acquire, check empty
        // read buf_[head & mask] into out, store head_ release.
        size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_tail = tail_.load(std::memory_order_acquire);
        if (current_head == current_tail){
            return false;
        }
        out = buf_[current_head & kMask];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    std::size_t size() const {
        return tail_.load(std::memory_order_relaxed) -
               head_.load(std::memory_order_relaxed);
    }

    bool empty() const { return size() == 0; }
    bool full() const { return size() >= Capacity; }

private:
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
    std::array<MarketEvent, Capacity> buf_;
};

// Default capacity: 2^20 events per instrument.
using DefaultRingBuffer = RingBuffer<1u << 20>;

} // namespace sc
