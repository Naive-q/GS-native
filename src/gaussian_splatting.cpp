#include "gaussian_splatting.h"

#include "knn/knn.h"
#include "io/data_io.h"
#include "util/log.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace gs_native {

GaussianSplatting::~GaussianSplatting() {
    cudaFree(gaussians_.gaussians_);
    cudaFree(image_);
    cudaFree(target_);
    cudaFree(density_output_);
    cudaFree(parent_ids_);
}

bool GaussianSplatting::Initialize(const Scene& scene) {
    GS_NATIVE_INFO("Initializing scene: training cameras {}, test cameras {}, points {}.",
        scene.train_cameras_.size(), scene.test_cameras_.size(), scene.point_cloud_.positions_.size());
    if (gaussians_.gaussians_ || scene.train_cameras_.empty() ||
        scene.train_cameras_.size() != scene.train_images_.size() ||
        scene.point_cloud_.positions_.empty() ||
        scene.point_cloud_.positions_.size() != scene.point_cloud_.colors_.size() ||
        scene.point_cloud_.positions_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    size_t pixels = 0;
    for (size_t i = 0; i < scene.train_cameras_.size(); ++i) {
        const Camera& camera = scene.train_cameras_[i];
        const cv::Mat& image = scene.train_images_[i];
        if (image.empty() || image.type() != CV_8UC3 || image.size() != camera.image_size_ ||
            camera.K_.type() != CV_32FC1 || camera.K_.rows != 3 || camera.K_.cols != 3 ||
            camera.R_c_w_.type() != CV_32FC1 || camera.R_c_w_.rows != 3 || camera.R_c_w_.cols != 3 ||
            camera.t_c_w_.type() != CV_32FC1 || camera.t_c_w_.rows != 3 || camera.t_c_w_.cols != 1) {
            return false;
        }
        pixels = std::max(pixels, image.total());
    }
    scene_ = scene;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for (const Camera& camera : scene_.train_cameras_) {
        for (int axis = 0; axis < 3; ++axis) {
            for (int row = 0; row < 3; ++row) {
                mean[axis] -= camera.R_c_w_.at<float>(row, axis) * camera.t_c_w_.at<float>(row);
            }
        }
    }
    mean /= static_cast<float>(scene_.train_cameras_.size());
    scene_extent_ = 0.0f;
    for (const Camera& camera : scene_.train_cameras_) {
        Eigen::Vector3f center = Eigen::Vector3f::Zero();
        for (int axis = 0; axis < 3; ++axis) {
            for (int row = 0; row < 3; ++row) {
                center[axis] -= camera.R_c_w_.at<float>(row, axis) * camera.t_c_w_.at<float>(row);
            }
        }
        scene_extent_ = std::max(scene_extent_, (center - mean).norm());
    }
    scene_extent_ *= 1.1f;
    if (scene_extent_ == 0.0f) {
        for (const Eigen::Vector3f& point : scene_.point_cloud_.positions_) {
            scene_extent_ = std::max(scene_extent_, (point - mean).norm());
        }
        scene_extent_ = std::max(scene_extent_, 1e-6f);
    }
    cudaFree(image_);
    cudaFree(target_);
    image_ = nullptr;
    target_ = nullptr;
    pixel_capacity_ = 0;
    if (cudaMalloc(&image_, pixels * 3 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&target_, pixels * 3) != cudaSuccess) {
        return false;
    }
    pixel_capacity_ = pixels;
    camera_order_.resize(scene_.train_cameras_.size());
    std::iota(camera_order_.begin(), camera_order_.end(), 0);
    camera_cursor_ = static_cast<int>(camera_order_.size());
    completed_ = 0;
    sh_degree_ = 0;
    if (!InitializeGaussians()) {
        GS_NATIVE_ERROR("Gaussian initialization failed.");
        return false;
    }
    GS_NATIVE_INFO("Initialization complete: gaussians {}, scene extent {}, pixel capacity {}.",
        gaussians_.gaussian_count_, scene_extent_, pixel_capacity_);
    return true;
}

bool GaussianSplatting::InitializeGaussians() {
    const PointCloud& cloud = scene_.point_cloud_;
    int count = static_cast<int>(cloud.positions_.size());
    std::vector<float> distances = ComputeMeanDist2Cuda(cloud.positions_);
    if (distances.size() != cloud.positions_.size()) {
        return false;
    }
    std::vector<float> parameters(static_cast<size_t>(count) * kMaxParamCount, 0.0f);
    for (int id = 0; id < count; ++id) {
        float scale = 0.5f * std::log(std::max(distances[id], 1e-7f));
        for (int axis = 0; axis < 3; ++axis) {
            parameters[static_cast<size_t>(axis) * count + id] = cloud.positions_[id][axis];
            parameters[static_cast<size_t>(kLogScaleIndex + axis) * count + id] = scale;
            parameters[static_cast<size_t>(kShIndex + axis * 16) * count + id] =
                (cloud.colors_[id][axis] - 0.5f) / 0.28209479177387814f;
        }
        parameters[static_cast<size_t>(kOpacityIndex) * count + id] = std::log(0.1f / 0.9f);
    }
    float* replacement = nullptr;
    size_t bytes = parameters.size() * sizeof(float);
    if (cudaMalloc(&replacement, bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(replacement, parameters.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        !density_.Initialize(count, count) || !solver_.RemapGaussians(nullptr, count)) {
        cudaFree(replacement);
        return false;
    }
    cudaFree(gaussians_.gaussians_);
    gaussians_.gaussians_ = replacement;
    gaussians_.gaussian_count_ = count;
    return true;
}

bool GaussianSplatting::Train(int iterations) {
    if (iterations < 0 || !gaussians_.gaussians_ || scene_.train_cameras_.empty()) {
        return false;
    }
    GS_NATIVE_INFO("Training started: requested iterations {}, completed {}, gaussians {}.",
        iterations, completed_, gaussians_.gaussian_count_);
    std::mt19937 generator(std::random_device{}());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (gaussians_.gaussian_count_ == 0) {
            return false;
        }
        if (camera_cursor_ >= static_cast<int>(camera_order_.size())) {
            std::shuffle(camera_order_.begin(), camera_order_.end(), generator);
            camera_cursor_ = 0;
        }
        int index = camera_order_[camera_cursor_];
        const Camera& source = scene_.train_cameras_[index];
        const cv::Mat& target = scene_.train_images_[index];
        CameraParams camera = {};
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
        int next = completed_ + 1;
        sh_degree_ = std::min(3, next / 1000);
        float fraction = std::min(1.0f, static_cast<float>(next) / 30000.0f);
        float learning_rate = scene_extent_ * std::exp(
            std::log(0.00016f) * (1.0f - fraction) + std::log(0.0000016f) * fraction);
        if (cudaMemcpy2D(target_, static_cast<size_t>(camera.width_) * 3, target.data, target.step,
                static_cast<size_t>(camera.width_) * 3, camera.height_, cudaMemcpyHostToDevice) != cudaSuccess) {
            GS_NATIVE_ERROR("Iteration {}: target upload failed for {}.", next, source.image_name_);
            return false;
        }
        if (!renderer_.Render(gaussians_, camera, background_, image_, sh_degree_)) {
            GS_NATIVE_ERROR("Iteration {}: rendering failed for {}.", next, source.image_name_);
            return false;
        }
        if (!solver_.Iterate(camera, renderer_, image_, target_, gaussians_, learning_rate, sh_degree_)) {
            GS_NATIVE_ERROR("Iteration {}: solver failed for {}.", next, source.image_name_);
            return false;
        }
        completed_ = next;
        ++camera_cursor_;
        if (completed_ < 15000) {
            if (!density_.AccumulateStatistics(solver_.GetProjectionStatistics(), gaussians_.gaussian_count_)) {
                return false;
            }
            if (completed_ > 500 && completed_ % 100 == 0 && !ControlDensity()) {
                return false;
            }
            if (completed_ % 3000 == 0) {
                if (!density_.ResetOpacity(gaussians_.gaussians_, gaussians_.gaussian_count_) ||
                    !solver_.ResetOpacityHistory()) {
                    return false;
                }
                GS_NATIVE_INFO("Iteration {}: opacity reset, gaussians {}.", completed_, gaussians_.gaussian_count_);
            }
        }
        GS_NATIVE_INFO("Iteration {}/{}, completed {}, SH {}, gaussians {}.",
            iteration + 1, iterations, completed_, sh_degree_, gaussians_.gaussian_count_);
    }
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        GS_NATIVE_ERROR("Training synchronization failed: {}.", cudaGetErrorString(result));
        return false;
    }
    GS_NATIVE_INFO("Training complete: completed {}, SH {}, gaussians {}.",
        completed_, sh_degree_, gaussians_.gaussian_count_);
    return true;
}

bool GaussianSplatting::Render(
    const Camera& source,
    cv::Mat& image) {
    if (!gaussians_.gaussians_ || source.image_size_.width <= 0 || source.image_size_.height <= 0 ||
        source.K_.type() != CV_32FC1 || source.K_.rows != 3 || source.K_.cols != 3 ||
        source.R_c_w_.type() != CV_32FC1 || source.R_c_w_.rows != 3 || source.R_c_w_.cols != 3 ||
        source.t_c_w_.type() != CV_32FC1 || source.t_c_w_.rows != 3 || source.t_c_w_.cols != 1) {
        return false;
    }
    CameraParams camera = {};
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
    size_t pixels = static_cast<size_t>(camera.width_) * camera.height_;
    if (pixels > pixel_capacity_) {
        float* replacement_image = nullptr;
        uint8_t* replacement_target = nullptr;
        if (cudaMalloc(&replacement_image, pixels * 3 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&replacement_target, pixels * 3) != cudaSuccess) {
            cudaFree(replacement_image);
            cudaFree(replacement_target);
            return false;
        }
        cudaFree(image_);
        cudaFree(target_);
        image_ = replacement_image;
        target_ = replacement_target;
        pixel_capacity_ = pixels;
    }
    if (!renderer_.Render(gaussians_, camera, background_, image_, sh_degree_)) {
        return false;
    }
    image.create(camera.height_, camera.width_, CV_32FC3);
    return cudaMemcpy2D(image.data, image.step, image_, static_cast<size_t>(camera.width_) * 3 * sizeof(float),
        static_cast<size_t>(camera.width_) * 3 * sizeof(float), camera.height_, cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool GaussianSplatting::Save(const std::string& path) {
    GS_NATIVE_INFO("Saving model: {}, gaussians {}.", path, gaussians_.gaussian_count_);
    int count = gaussians_.gaussian_count_;
    if (count <= 0 || !gaussians_.gaussians_) {
        return false;
    }
    std::vector<float> parameters(static_cast<size_t>(count) * kMaxParamCount);
    if (cudaMemcpy(parameters.data(), gaussians_.gaussians_, parameters.size() * sizeof(float),
            cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    std::vector<Gaussian> gaussians(count);
    for (int id = 0; id < count; ++id) {
        Gaussian& gaussian = gaussians[id];
        for (int axis = 0; axis < 3; ++axis) {
            gaussian.position_[axis] = parameters[static_cast<size_t>(axis) * count + id];
            gaussian.log_scale_[axis] = parameters[static_cast<size_t>(kLogScaleIndex + axis) * count + id];
            gaussian.rotation_[axis] = parameters[static_cast<size_t>(kRotationIndex + axis) * count + id];
            for (int coefficient = 0; coefficient < 16; ++coefficient) {
                gaussian.sh_(coefficient, axis) =
                    parameters[static_cast<size_t>(kShIndex + axis * 16 + coefficient) * count + id];
            }
        }
        gaussian.opacity_logit_ = parameters[static_cast<size_t>(kOpacityIndex) * count + id];
    }
    return WriteGaussiansPly(path, gaussians);
}

bool GaussianSplatting::ControlDensity() {
    if (!parent_ids_ && cudaMalloc(&parent_ids_, sizeof(int)) != cudaSuccess) {
        return false;
    }

    int changes[2] = {};
    int count = gaussians_.gaussian_count_;
    if (!density_.ControlDensity(gaussians_.gaussians_, density_output_, parent_ids_, count,
            density_output_stride_, scene_extent_, 0.0002, 0.01, 0.005,
            completed_ > 3000 ? 20.0 : 0.0, 0, changes)) {
        return false;
    }
    int next_count = count + changes[0] - changes[1];
    if (!solver_.RemapGaussians(parent_ids_, next_count)) {
        return false;
    }
    std::swap(gaussians_.gaussians_, density_output_);
    gaussians_.gaussian_count_ = next_count;
    density_output_stride_ = count;
    GS_NATIVE_INFO("Iteration {}: density control, gaussians {} -> {}, added {}, removed {}.",
        completed_, count, next_count, changes[0], changes[1]);
    return true;
}

}
