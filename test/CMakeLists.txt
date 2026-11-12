find_package(GTest 1.17 CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.17.0
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googletest)
endif()
set(test_sources knn_test.cu renderer_test.cu solver_test.cu density_test.cu
    io_test.cpp gaussian_splatting_test.cpp)
set_source_files_properties(knn_test.cu renderer_test.cu solver_test.cu density_test.cu PROPERTIES LANGUAGE CXX)
add_executable(gs-native-tests ${test_sources})
target_link_libraries(gs-native-tests PRIVATE gs-native GTest::gtest_main)
if(MSVC)
    target_compile_options(gs-native-tests PRIVATE /utf-8)
endif()
if(WIN32)
    add_custom_command(TARGET gs-native-tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:gs-native-tests> $<TARGET_FILE_DIR:gs-native-tests>
        COMMAND_EXPAND_LISTS)
endif()
include(GoogleTest)
gtest_discover_tests(gs-native-tests DISCOVERY_MODE PRE_TEST
    PROPERTIES LABELS correctness RUN_SERIAL TRUE)
