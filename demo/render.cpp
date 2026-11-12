#include "render/renderer.h"

#include "io/data_io.h"

#include <cuda_runtime.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const char* keys = "{help h ?  |       | print this message}"
                       "{scene     | <none>| COLMAP scene directory}"
                       "{model     | <none>| Gaussian PLY path}"
                       "{output    | <none>| rendering output directory}"
                       "{downscale | 1     | image downscale factor}"
                       "{sh_degree | 3     | SH degree used for rendering}"
                       "{split     | test  | test or train cameras}";
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("gs-native rendering");
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::string scene_dir = parser.get<std::string>("scene");
    std::string model_path = parser.get<std::string>("model");
    std::string output_dir = parser.get<std::string>("output");
    std::string split = parser.get<std::string>("split");
    int downscale = std::max(1, parser.get<int>("downscale"));
    int sh_degree = std::clamp(parser.get<int>("sh_degree"), 0, 3);
    if (!parser.check() || (split != "test" && split != "train")) {
        parser.printErrors();
        std::cerr << "Specify --scene, --model, --output and --split=test or train.\n";
        return 1;
    }
    gs_native::GaussianBuffers buffers;
    float* device_image = nullptr;
    int result = 0;
    try {
        gs_native::Scene scene;
        std::vector<gs_native::Gaussian> gaussians;
        if (!gs_native::ReadColmapScene(scene_dir, downscale, scene) ||
            !gs_native::ReadGaussiansPly(model_path, gaussians)) {
            return 1;
        }
        const std::vector<gs_native::Camera>& cameras = split == "test" ?
            scene.test_cameras_ :
            scene.train_cameras_;
        const std::vector<cv::Mat>& targets = split == "test" ?
            scene.test_images_ :
            scene.train_images_;
        if (cameras.empty() || gaussians.empty() ||
            gaussians.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("No cameras or no valid Gaussians.");
        }
        int count = static_cast<int>(gaussians.size());
        std::vector<float> parameters(static_cast<size_t>(count) * gs_native::kMaxParamCount);
        for (int id = 0; id < count; ++id) {
            const gs_native::Gaussian& gaussian = gaussians[id];
            for (int axis = 0; axis < 3; ++axis) {
                parameters[static_cast<size_t>(axis) * count + id] = gaussian.position_[axis];
                parameters[static_cast<size_t>(gs_native::kLogScaleIndex + axis) * count + id] = gaussian.log_scale_[axis];
                parameters[static_cast<size_t>(gs_native::kRotationIndex + axis) * count + id] = gaussian.rotation_[axis];
                for (int coefficient = 0; coefficient < 16; ++coefficient) {
                    parameters[static_cast<size_t>(gs_native::kShIndex + axis * 16 + coefficient) * count + id] =
                        gaussian.sh_(coefficient, axis);
                }
            }
            parameters[static_cast<size_t>(gs_native::kOpacityIndex) * count + id] = gaussian.opacity_logit_;
        }
        size_t pixel_capacity = 0;
        for (const cv::Mat& target : targets) {
            pixel_capacity = std::max(pixel_capacity, target.total());
        }
        size_t bytes = parameters.size() * sizeof(float);
        if (cudaMalloc(&buffers.gaussians_, bytes) != cudaSuccess ||
            cudaMalloc(&device_image, pixel_capacity * 3 * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(buffers.gaussians_, parameters.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("Allocating or uploading rendering buffers failed.");
        }
        buffers.gaussian_count_ = count;
        std::filesystem::path destination = std::filesystem::path(output_dir) / split / "gs-native";
        if (std::filesystem::exists(destination) && !std::filesystem::is_empty(destination)) {
            throw std::runtime_error("Rendering directory is not empty: " + destination.string());
        }
        for (const char* directory : {"renders", "gt", "comparison"}) {
            std::filesystem::create_directories(destination / directory);
        }
        std::ofstream report(destination / "psnr.csv");
        if (!report) {
            throw std::runtime_error("Opening PSNR report failed.");
        }
        report << "image,psnr_db\n" << std::setprecision(10);
        gs_native::Renderer renderer;
        float background[3] = {0.0f, 0.0f, 0.0f};
        double sum = 0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            const gs_native::Camera& source = cameras[i];
            gs_native::CameraParams camera = {};
            camera.width_ = source.image_size_.width;
            camera.height_ = source.image_size_.height;
            camera.fx_ = source.K_.at<float>(0, 0);
            camera.fy_ = source.K_.at<float>(1, 1);
            camera.cx_ = source.K_.at<float>(0, 2);
            camera.cy_ = source.K_.at<float>(1, 2);
            for (int row = 0; row < 3; ++row) {
                camera.t_c_w_[row] = source.t_c_w_.at<float>(row);
                for (int col = 0; col < 3; ++col) {
                    camera.R_c_w_[row * 3 + col] = source.R_c_w_.at<float>(row, col);
                }
            }
            camera.near_depth_ = 0.2f;
            camera.far_depth_ = std::numeric_limits<float>::max();
            camera.tile_count_x_ = (camera.width_ + 15) / 16;
            camera.tile_count_y_ = (camera.height_ + 15) / 16;
            cv::Mat image(camera.height_, camera.width_, CV_32FC3);
            if (!renderer.Render(buffers, camera, background, device_image, sh_degree) ||
                cudaMemcpy(image.data, device_image, image.total() * 3 * sizeof(float),
                    cudaMemcpyDeviceToHost) != cudaSuccess || !cv::checkRange(image)) {
                throw std::runtime_error("Rendering failed at " + source.image_name_);
            }

            cv::max(image, 0.0, image);
            cv::min(image, 1.0, image);
            cv::Mat image_8u, image_bgr, target_bgr, comparison;
            image.convertTo(image_8u, CV_8UC3, 255.0);
            double mse = cv::norm(image_8u, targets[i], cv::NORM_L2SQR) /
                (3.0 * image.total() * 255.0 * 255.0);
            double psnr = mse == 0 ?
                std::numeric_limits<double>::infinity() :
                -10.0 * std::log10(mse);
            sum += psnr;
            std::string name = cv::format("%05d.png", static_cast<int>(i));
            report << name << ',' << psnr << '\n';
            cv::cvtColor(image_8u, image_bgr, cv::COLOR_RGB2BGR);
            cv::cvtColor(targets[i], target_bgr, cv::COLOR_RGB2BGR);
            cv::hconcat(image_bgr, target_bgr, comparison);
            if (!cv::imwrite((destination / "renders" / name).string(), image_bgr) ||
                !cv::imwrite((destination / "gt" / name).string(), target_bgr) ||
                !cv::imwrite((destination / "comparison" / name).string(), comparison)) {
                throw std::runtime_error("Saving rendered image failed.");
            }
            std::cout << "View " << i << " (" << source.image_name_ << "): PSNR "
                      << std::fixed << std::setprecision(6) << psnr << " dB" << std::endl;
        }
        report << "mean," << sum / cameras.size() << '\n';
        report.close();
        if (!report) {
            throw std::runtime_error("Writing PSNR report failed.");
        }
        std::cout << "Mean PSNR: " << sum / cameras.size() << " dB\nOutput: "
                  << destination.string() << std::endl;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    cudaFree(device_image);
    cudaFree(buffers.gaussians_);
    return result;
}
