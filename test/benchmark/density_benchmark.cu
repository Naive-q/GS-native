#include "benchmark_support.h"

#include "density/density_controller.h"

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    float* output = nullptr;
    int* parents = nullptr;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        auto gaussians = MakeGaussians(options.count_);
        std::vector<float> statistics(size_t(options.count_) * 2, 1);
        int expected = 0;
        for (int i = 0; i < options.count_; ++i) {
            statistics[i] = i % 2 ? 0 : 1;
            if (i % 7 == 0) gaussians[i].opacity_logit_ = -12;
            else expected += i % 2 ? 1 : 2;
        }
        DeviceGaussians device(gaussians);
        DeviceBuffer<float> stats(statistics);
        DensityController controller;
        int stride = 0;
        CheckCuda(cudaMalloc(&parents, sizeof(int)));
        auto prepare = [&] {
            if (!controller.Initialize(options.count_, options.count_) ||
                !controller.AccumulateStatistics(stats.Get(), options.count_)) throw std::runtime_error("Density setup failed");
        };
        auto control = [&] { return controller.ControlDensity(device.buffers_.gaussians_, output, parents,
            options.count_, stride, 1, 0.0002, 0.01, 0.005, 0, 42); };
        prepare();
        if (!control()) throw std::runtime_error("Density validation failed");
        if (stride != expected || ReadDevice(controller.GetGaussianCount(), 1)[0] != expected) {
            throw std::runtime_error("Density count mismatch");
        }
        CheckFinite(ReadDevice(output, size_t(stride) * kMaxParamCount));
        std::cout << "scope=ControlDensity_scan_sampling_output count=" << options.count_ << '\n';
        Measure("density", options, prepare, control);
        if (stride != expected || ReadDevice(controller.GetGaussianCount(), 1)[0] != expected) {
            throw std::runtime_error("Density count mismatch");
        }
        CheckFinite(ReadDevice(output, size_t(stride) * kMaxParamCount));
    }
    catch (const std::exception& error) {
        cudaFree(output); cudaFree(parents);
        std::cerr << error.what() << '\n'; return 1;
    }
    cudaFree(output); cudaFree(parents);
}
