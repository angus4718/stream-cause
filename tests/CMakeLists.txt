find_package(Catch2 CONFIG REQUIRED)

# Sources are added incrementally as each module is implemented.
# Add a module's .cpp here only after its stubs are filled in.
set(TEST_SOURCES
    test_ring_buffer.cpp
    test_hawkes_em.cpp
    test_ci_test.cpp
    test_ts_fci.cpp
    test_cusum.cpp
    test_bocpd.cpp
)

set(LIB_SOURCES
    # Step 4+: uncomment as each file is implemented
    ./src/ingestion/market_event.cpp
    ./src/ingestion/event_router.cpp
    ./src/hawkes/hawkes_pair.cpp
    ./src/hawkes/em_estimator.cpp
    ./src/hawkes/rolling_window.cpp
    ./src/causal/pag.cpp
    ./src/causal/ci_test.cpp
    ./src/causal/ts_fci.cpp
    ./src/causal/graph_store.cpp
    ./src/changepoint/cusum.cpp
    ./src/changepoint/bocpd.cpp
    ./src/pipeline/thread_pool.cpp
)

add_executable(streamcause_tests ${TEST_SOURCES} ${LIB_SOURCES})
target_include_directories(streamcause_tests PRIVATE ./src)
target_link_libraries(streamcause_tests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    TBB::tbb
    # Boost::lockfree # header-only, no linking needed
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    databento::databento
)

include(CTest)
include(Catch)
catch_discover_tests(streamcause_tests)
