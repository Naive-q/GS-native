#include "benchmark_support.h"

#include "density/density_controller.h"

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    GaussianBuffers gaussians;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        auto initial_gaussians = MakeGaussians(options.count_);
        std::vector<float> statistics(size_t(options.count_) * 2, 1);
        int expected = 0;
        for (int i = 0; i < options.count_; ++i) {
            statistics[i] = i % 2 ? 0 : 1;
            if (i % 7 == 0) initial_gaussians[i].opacity_logit_ = -12;
            else expected += i % 2 ? 1 : 2;
        }
        DeviceGaussians device(initial_gaussians);
        DeviceBuffer<float> stats(statistics);
        DensityController controller;
        auto prepare = [&] {
            if (!gaussians.gaussians_ || gaussians.gaussian_count_ != options.count_) {
                CheckCuda(cudaFree(gaussians.gaussians_));
                gaussians.gaussians_ = nullptr;
                CheckCuda(cudaMalloc(&gaussians.gaussians_,
                    static_cast<size_t>(options.count_) * kMaxParamCount * sizeof(float)));
            }
            gaussians.gaussian_count_ = options.count_;
            CheckCuda(cudaMemcpy(gaussians.gaussians_, device.buffers_.gaussians_,
                static_cast<size_t>(options.count_) * kMaxParamCount * sizeof(float), cudaMemcpyDeviceToDevice));
            if (!controller.Initialize(options.count_, options.count_) ||
                !controller.AccumulateStatistics(stats.Get(), stats.Get(), options.count_)) throw std::runtime_error("Density setup failed");
        };
        auto control = [&] {
            return controller.ControlDensity(
                gaussians,
                1.0f,
                0.0002f,
                0.01f,
                0.005f,
                0.0f,
                42,
                nullptr,
                0.0008f);
        };
        prepare();
        if (!control()) throw std::runtime_error("Density validation failed");
        if (gaussians.gaussian_count_ != expected || ReadDevice(controller.GetGaussianCount(), 1)[0] != expected) {
            throw std::runtime_error("Density count mismatch");
        }
        CheckFinite(ReadDevice(gaussians.gaussians_, size_t(gaussians.gaussian_count_) * kMaxParamCount));
        std::cout << "scope=ControlDensity_scan_sampling_output count=" << options.count_ << '\n';
        Measure("density", options, prepare, control);
        if (gaussians.gaussian_count_ != expected || ReadDevice(controller.GetGaussianCount(), 1)[0] != expected) {
            throw std::runtime_error("Density count mismatch");
        }
        CheckFinite(ReadDevice(gaussians.gaussians_, size_t(gaussians.gaussian_count_) * kMaxParamCount));
    }
    catch (const std::exception& error) {
        cudaFree(gaussians.gaussians_);
        std::cerr << error.what() << '\n'; return 1;
    }
    cudaFree(gaussians.gaussians_);
}
