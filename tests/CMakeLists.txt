find_package(Catch2 CONFIG REQUIRED)

set(TEST_SOURCES
    test_ring_buffer.cpp
    test_hawkes_em.cpp
)

set(LIB_SOURCES
    ${CMAKE_SOURCE_DIR}/src/ingestion/market_event.cpp
    ${CMAKE_SOURCE_DIR}/src/ingestion/event_router.cpp
    ${CMAKE_SOURCE_DIR}/src/hawkes/hawkes_pair.cpp
    ${CMAKE_SOURCE_DIR}/src/hawkes/em_estimator.cpp
    ${CMAKE_SOURCE_DIR}/src/hawkes/rolling_window.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/thread_pool.cpp
)

add_executable(streamcause_tests ${TEST_SOURCES} ${LIB_SOURCES})
target_include_directories(streamcause_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(streamcause_tests PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
    TBB::tbb
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    databento::databento
)

include(CTest)
include(Catch)
catch_discover_tests(streamcause_tests)
