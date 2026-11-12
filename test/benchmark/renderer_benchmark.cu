#include "benchmark_support.h"

#include "cpu_reference.h"
#include "render/renderer.h"

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        auto gaussians = MakeGaussians(options.count_);
        DeviceGaussians device(gaussians);
        auto camera = MakeCamera(options.width_, options.height_);
        DeviceBuffer<float> image(size_t(camera.width_) * camera.height_ * 3);
        Renderer renderer;
        std::array<float, 3> background = {0.05f, 0.1f, 0.15f};
        if (!renderer.Render(device.buffers_, camera, background.data(), image.Get(), options.degree_)) return 1;
        auto values = image.Download();
        CheckFinite(values);
        cv::Mat actual(camera.height_, camera.width_, CV_32FC3, values.data()), converted;
        actual.convertTo(converted, CV_64F);
        auto expected = RenderReference(gaussians, camera, background, options.degree_);
        if (!cv::checkRange(expected)) throw std::runtime_error("Non-finite renderer reference");
        double error = cv::norm(converted, expected, cv::NORM_INF);
        if (!std::isfinite(error) || error > 1e-4) {
            throw std::runtime_error("Renderer reference mismatch");
        }
        std::cout << "scope=Render_projection_sort_compositing count=" << options.count_
            << " width=" << options.width_ << " height=" << options.height_ << '\n';
        Measure("renderer", options, [] {}, [&] {
            return renderer.Render(device.buffers_, camera, background.data(), image.Get(), options.degree_);
        });
        CheckFinite(image.Download());
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
