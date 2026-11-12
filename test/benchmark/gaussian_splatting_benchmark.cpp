#include "benchmark_support.h"

#include "gaussian_splatting.h"
#include "io/data_io.h"

#include <memory>

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        Scene scene;
        if (options.scene_.empty()) scene = MakeScene();
        else if (!ReadColmapScene(options.scene_, 1, scene)) return 1;
        std::unique_ptr<GaussianSplatting> model;
        auto prepare = [&] {
            model = std::make_unique<GaussianSplatting>();
            if (!model->Initialize(scene)) throw std::runtime_error("Scene initialization failed");
        };
        std::cout << "scope=Train_excluding_initialization_save_evaluation updates_per_sample=" << options.iterations_ << '\n';
        Measure("gaussian_splatting", options, prepare, [&] { return model->Train(options.iterations_); });
        cv::Mat image;
        if (!model->Render(scene.train_cameras_[0], image) || image.empty() || !cv::checkRange(image)) {
            throw std::runtime_error("Pipeline output validation failed");
        }
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
