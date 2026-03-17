#include <catch2/catch_test_macros.hpp>
#include <thread>
#include "ingestion/ring_buffer.hpp"

using namespace sc;

// Use a small capacity for tests.
using SmallRingBuffer = RingBuffer<16>;

TEST_CASE("RingBuffer: push and pop single-threaded", "[ring_buffer]") {
    SmallRingBuffer rb;
    MarketEvent ev{};
    ev.instrument_id = 3;
    ev.price = 100'000'000'000LL;

    REQUIRE(rb.empty());
    REQUIRE(rb.push(ev));
    REQUIRE(!rb.empty());

    MarketEvent out{};
    REQUIRE(rb.pop(out));
    REQUIRE(out.instrument_id == 3);
    REQUIRE(out.price == 100'000'000'000LL);
    REQUIRE(rb.empty());
}

TEST_CASE("RingBuffer: overflow returns false", "[ring_buffer]") {
    SmallRingBuffer rb;
    MarketEvent ev{};

    // Fill the buffer.
    for (int i = 0; i < 16; ++i) {
        rb.push(ev);
    }
    // One more push should fail (buffer full).
    REQUIRE_FALSE(rb.push(ev));
}

TEST_CASE("RingBuffer: SPSC concurrent correctness", "[ring_buffer]") {
    constexpr int N = 10000;
    RingBuffer<1 << 14> rb; // 16k capacity

    std::atomic<int> consumed{0};
    std::thread producer([&] {
        MarketEvent ev{};
        for (int i = 0; i < N; ++i) {
            ev.instrument_id = i;
            while (!rb.push(ev)) std::this_thread::yield();
        }
    });
    std::thread consumer([&] {
        MarketEvent out{};
        for (int i = 0; i < N; ++i) {
            while (!rb.pop(out)) std::this_thread::yield();
            ++consumed;
        }
    });
    producer.join();
    consumer.join();
    REQUIRE(consumed.load() == N);
}
