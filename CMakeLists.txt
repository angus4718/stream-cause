cmake_minimum_required(VERSION 3.20)
project(streamcause LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# e.g. cmake -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake ..

find_package(Eigen3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

include(ExternalProject)

ExternalProject_Add(
    date_ext
    GIT_REPOSITORY https://github.com/HowardHinnant/date.git
    GIT_TAG        v3.0.1
    GIT_SHALLOW    FALSE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/date-src"
)

ExternalProject_Add(
    httplib_ext
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.14.1
    GIT_SHALLOW    FALSE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/httplib-src"
)

ExternalProject_Add(
    databento_ext
    GIT_REPOSITORY https://github.com/databento/databento-cpp.git
    GIT_TAG        v0.36.0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/databento-src"
    DEPENDS date_ext httplib_ext
)

find_package(OpenSSL REQUIRED)

find_package(zstd CONFIG QUIET)
if(NOT zstd_FOUND)
    find_library(ZSTD_LIB zstd PATHS "C:/Users/AngusCheung/anaconda3/Library/lib" NO_DEFAULT_PATH)
    if(ZSTD_LIB)
        message(STATUS "Found zstd library: ${ZSTD_LIB}")
        add_library(zstd::zstd UNKNOWN IMPORTED)
        set_target_properties(zstd::zstd PROPERTIES
            IMPORTED_LOCATION "${ZSTD_LIB}"
        )
    endif()
endif()

add_library(databento INTERFACE)
target_include_directories(databento INTERFACE
    "${CMAKE_BINARY_DIR}/_deps/databento-src/include"
    "${CMAKE_BINARY_DIR}/_deps/date-src/include"
    "${CMAKE_BINARY_DIR}/_deps/httplib-src"
)
target_link_libraries(databento INTERFACE OpenSSL::Crypto nlohmann_json::nlohmann_json)
if(TARGET zstd::zstd)
    target_link_libraries(databento INTERFACE zstd::zstd)
endif()
add_dependencies(databento databento_ext date_ext httplib_ext)

add_library(databento::databento ALIAS databento)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-mavx2 -march=native)
    add_compile_options($<$<CONFIG:Release>:-O3>)
elseif(MSVC)
    add_compile_options(/arch:AVX2)
    add_compile_options($<$<CONFIG:Release>:/O2>)
endif()

set(STREAMCAUSE_SOURCES
    src/main.cpp
    src/ingestion/market_event.cpp
    # src/ingestion/databento_client.cpp
    src/ingestion/file_replay.cpp
    src/ingestion/event_router.cpp
    src/hawkes/hawkes_pair.cpp
    src/hawkes/em_estimator.cpp
    src/hawkes/rolling_window.cpp
    src/hawkes/intensity_tracker.cpp
    src/causal/pag.cpp
    src/causal/ci_test.cpp
    src/causal/ts_fci.cpp
    src/causal/graph_store.cpp
    src/changepoint/cusum.cpp
    src/changepoint/bocpd.cpp
    src/pipeline/thread_pool.cpp
    src/pipeline/pipeline.cpp
)

add_executable(streamcause ${STREAMCAUSE_SOURCES})
target_include_directories(streamcause PRIVATE src)
target_link_libraries(streamcause PRIVATE
    Eigen3::Eigen
    TBB::tbb
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    # databento::databento
)
