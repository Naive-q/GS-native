#include "benchmark_support.h"

#include "render/renderer.h"
#include "solver/solver.h"

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        auto gaussians = MakeGaussians(options.count_);
        auto parameters = PackGaussians(gaussians);
        DeviceGaussians device(gaussians);
        auto camera = MakeCamera(options.width_, options.height_);
        DeviceBuffer<float> image(size_t(camera.width_) * camera.height_ * 3);
        DeviceBuffer<uint8_t> target(std::vector<uint8_t>(size_t(camera.width_) * camera.height_ * 3, 180));
        Renderer renderer;
        Solver solver;
        float background[3] = {};
        auto prepare = [&] {
            device.parameters_.Upload(parameters);
            if (!renderer.Render(device.buffers_, camera, background, image.Get(), options.degree_)) {
                throw std::runtime_error("Rendering solver input failed");
            }
        };
        auto iterate = [&] { return solver.Iterate(camera, renderer, image.Get(), target.Get(), device.buffers_, 0.0001f, options.degree_); };
        prepare();
        if (!iterate()) return 1;
        CheckFinite(device.parameters_.Download());
        std::cout << "scope=Iterate_image_gradients_Gaussian_gradients_Adam count=" << options.count_ << '\n';
        Measure("solver", options, prepare, iterate);
        CheckFinite(device.parameters_.Download());
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
