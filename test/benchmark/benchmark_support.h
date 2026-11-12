#pragma once

#include "test_support.h"

#include <opencv2/core/utility.hpp>
#include <functional>
#include <iomanip>
#include <iostream>

namespace gs_native::benchmark {
using namespace gs_native::test;

struct Options {
    int count_ = 4096;
    int width_ = 320;
    int height_ = 240;
    int repeats_ = 7;
    int warmups_ = 2;
    int degree_ = 3;
    int iterations_ = 10;
    std::string scene_;
    std::string points_;
}; // struct Options

inline bool ReadOptions(
    char** argv,
    Options& options,
    int argc) {
    cv::CommandLineParser parser(argc, argv,
        "{help h ? | | show benchmark options}"
        "{count |4096| synthetic Gaussian or point count}"
        "{width |320| image width} {height |240| image height}"
        "{repeats |7| measured repetitions} {warmups |2| warmup repetitions}"
        "{sh_degree |3| SH degree} {iterations |10| updates per whole-pipeline measurement}"
        "{scene | | optional COLMAP scene for the pipeline benchmark}"
        "{points | | optional COLMAP points3D.bin for the KNN benchmark}");
    if (parser.has("help")) { parser.printMessage(); return false; }
    options.count_ = std::max(1, parser.get<int>("count"));
    options.width_ = std::max(1, parser.get<int>("width"));
    options.height_ = std::max(1, parser.get<int>("height"));
    options.repeats_ = std::max(1, parser.get<int>("repeats"));
    options.warmups_ = std::max(0, parser.get<int>("warmups"));
    options.degree_ = std::clamp(parser.get<int>("sh_degree"), 0, 3);
    options.iterations_ = std::max(1, parser.get<int>("iterations"));
    options.scene_ = parser.get<std::string>("scene");
    options.points_ = parser.get<std::string>("points");
    if (!parser.check()) throw std::runtime_error("Invalid benchmark options");
    if (!HasCuda()) { std::cout << "SKIP: no CUDA device\n"; return false; }
    int device = 0;
    CheckCuda(cudaGetDevice(&device));
    cudaDeviceProp properties = {};
    CheckCuda(cudaGetDeviceProperties(&properties, device));
    std::cout << "GPU=" << properties.name << " repeats=" << options.repeats_
        << " warmups=" << options.warmups_ << '\n';
    return true;
}

template<class Prepare, class Operation>
void Measure(
    const std::string& name,
    const Options& options,
    Prepare prepare,
    Operation operation) {
    std::vector<double> times;
    for (int i = -options.warmups_; i < options.repeats_; ++i) {
        prepare();
        CheckCuda(cudaDeviceSynchronize());
        cv::TickMeter timer;
        timer.start();
        bool success = operation();
        CheckCuda(cudaDeviceSynchronize());
        timer.stop();
        if (!success) throw std::runtime_error("Benchmark operation failed");
        if (i >= 0) times.push_back(timer.getTimeMilli());
    }
    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    if (times.size() % 2 == 0) median = (median + times[times.size() / 2 - 1]) * 0.5;
    std::cout << std::fixed << std::setprecision(6) << name << " median_ms=" << median
        << " min_ms=" << times.front() << " max_ms=" << times.back() << '\n';
}

inline void CheckFinite(const std::vector<float>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite benchmark output");
    }
}
} // namespace gs_native::benchmark
