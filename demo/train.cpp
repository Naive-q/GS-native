#include "gaussian_splatting.h"

#include "io/data_io.h"

#include <opencv2/core/utility.hpp>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const char* keys = "{help h ?  |       | print this message}"
                       "{scene     | <none>| COLMAP scene directory}"
                       "{output    | <none>| model output directory}"
                       "{downscale | 1     | image downscale factor}"
                       "{iterations| 30000 | number of single-view updates}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("gs-native training");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string scene_dir = parser.get<std::string>("scene");
    std::string output_dir = parser.get<std::string>("output");
    int downscale = std::max(1, parser.get<int>("downscale"));
    int iterations = parser.get<int>("iterations");
    if (!parser.check() || iterations <= 0) {
        parser.printErrors();
        std::cerr << "Specify --scene, --output and positive --iterations.\n";
        return 1;
    }
    try {
        cv::TickMeter stage_timer;
        stage_timer.start();
        std::cout << "Reading scene: " << scene_dir << std::endl;
        gs_native::Scene scene;
        if (!gs_native::ReadColmapScene(scene_dir, downscale, scene)) {
            return 1;
        }
        stage_timer.stop();
        std::cout << "Scene read: " << stage_timer.getTimeSec() << " s" << std::endl;
        stage_timer.reset();
        stage_timer.start();
        std::cout << "Initializing Gaussians..." << std::endl;
        gs_native::GaussianSplatting gs;
        if (!gs.Initialize(scene)) {
            std::cerr << "Initialization failed.\n";
            return 1;
        }
        stage_timer.stop();
        std::cout << "Initialization: " << stage_timer.getTimeSec() << " s" << std::endl;
        std::cout << "Cameras: " << scene.train_cameras_.size()
                  << ", initial Gaussians: " << scene.point_cloud_.positions_.size()
                  << ", single-view updates: " << iterations << std::endl;
        cv::TickMeter total_timer;
        total_timer.start();
        if (!gs.Train(iterations)) {
            std::cerr << "Training failed.\n";
            return 1;
        }
        total_timer.stop();
        std::cout << "Training total: " << total_timer.getTimeSec() << " s"
                  << ", average: " << total_timer.getTimeMilli() / iterations << " ms/update" << std::endl;

        std::filesystem::path model_dir = std::filesystem::absolute(output_dir);
        std::filesystem::path point_cloud_dir = model_dir / "point_cloud" /
            ("iteration_" + std::to_string(iterations));
        std::filesystem::create_directories(point_cloud_dir);
        stage_timer.reset();
        stage_timer.start();
        std::cout << "Saving model..." << std::endl;
        if (!gs.Save((point_cloud_dir / "point_cloud.ply").string())) {
            std::cerr << "Saving model failed.\n";
            return 1;
        }
        std::ofstream config(model_dir / "cfg_args");
        config << "Namespace(sh_degree=" << std::min(3, iterations / 1000)
               << ", source_path=" << std::quoted(std::filesystem::absolute(scene_dir).generic_string(), '\'')
               << ", model_path=" << std::quoted(model_dir.generic_string(), '\'')
               << ", images='images', resolution=" << downscale
               << ", white_background=False, data_device='cuda', eval=True)\n";
        config.close();
        if (!config) {
            std::cerr << "Saving viewer configuration failed.\n";
            return 1;
        }
        stage_timer.stop();
        std::cout << "Saving: " << stage_timer.getTimeSec() << " s" << std::endl;
        std::cout << "Model: " << (point_cloud_dir / "point_cloud.ply").string() << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
