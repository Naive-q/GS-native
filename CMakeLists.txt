cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(gs-native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    set(CMAKE_VS_GLOBALS "VcpkgEnabled=false")
endif()

include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(WIN32)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_DEBUG Release)
endif()

add_library(gs-native STATIC
    src/gaussian_splatting.cpp
    src/render/renderer.cu
    src/solver/solver.cu
    src/density/density_controller.cu
    src/io/data_io.cpp
    src/knn/knn.cu)
set_source_files_properties(src/knn/knn.cu PROPERTIES COMPILE_DEFINITIONS "EIGEN_NO_CUDA;FMT_UNICODE=0")
target_include_directories(gs-native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen>
    "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gs-native>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/gs-native/third_party/eigen>)
target_compile_features(gs-native PUBLIC cxx_std_17)
target_include_directories(gs-native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog/include)
target_link_libraries(gs-native PUBLIC ${OpenCV_LIBS} CUDA::cudart)
if(MSVC)
    target_compile_options(gs-native PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

add_subdirectory(demo)

option(BUILD_TESTING "Build correctness tests" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
option(GS_NATIVE_BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(GS_NATIVE_BUILD_BENCHMARKS)
    add_subdirectory(test/benchmark)
endif()

install(TARGETS gs-native EXPORT gs-nativeTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gs-native
    FILES_MATCHING PATTERN "*.h" PATTERN "util" EXCLUDE)
install(DIRECTORY third_party/eigen/Eigen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gs-native/third_party/eigen)
install(FILES LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native)
file(GLOB eigen_license_files "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/COPYING.*")
install(FILES ${eigen_license_files}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/eigen)
install(FILES third_party/spdlog/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/spdlog)
install(FILES third_party/spdlog/include/spdlog/fmt/bundled/fmt.license.rst
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/fmt)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/gs-nativeConfig.cmake" [=[
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(OpenCV COMPONENTS core imgproc imgcodecs)
if(WIN32)
    set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_DEBUG Release)
endif()
if(NOT TARGET gs-native::gs-native)
    include("${CMAKE_CURRENT_LIST_DIR}/gs-nativeTargets.cmake")
    set_property(TARGET gs-native::gs-native APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}")
endif()
]=])
install(EXPORT gs-nativeTargets
    FILE gs-nativeTargets.cmake
    NAMESPACE gs-native::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gs-native)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/gs-nativeConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gs-native)
