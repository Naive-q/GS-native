#pragma once

#include "types/buffers.h"
#include "types/scene.h"
#include "types/gaussian.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace gs_native::test {

inline void CheckCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

inline bool HasCuda() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { Resize(count); }
    explicit DeviceBuffer(const std::vector<T>& values) : DeviceBuffer(values.size()) {
        Upload(values);
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void Resize(size_t count) {
        if (count == count_) {
            return;
        }
        CheckCuda(cudaFree(data_));
        data_ = nullptr;
        count_ = 0;
        if (count) {
            CheckCuda(cudaMalloc(&data_, count * sizeof(T)));
        }
        count_ = count;
    }
    void Upload(const std::vector<T>& values) {
        Resize(values.size());
        if (count_) {
            CheckCuda(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    std::vector<T> Download() const {
        std::vector<T> values(count_);
        if (count_) {
            CheckCuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return values;
    }
    T* Get() { return data_; }
    const T* Get() const { return data_; }
private:
    T* data_ = nullptr;
    size_t count_ = 0;
}; // class DeviceBuffer

template<typename T>
std::vector<T> ReadDevice(
    const T* data,
    size_t count) {
    std::vector<T> result(count);
    if (count) {
        CheckCuda(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return result;
}

inline std::vector<float> PackGaussians(const std::vector<Gaussian>& gaussians) {
    size_t count = gaussians.size();
    std::vector<float> values(count * kMaxParamCount);
    for (size_t i = 0; i < count; ++i) {
        for (int axis = 0; axis < 3; ++axis) {
            values[axis * count + i] = gaussians[i].position_[axis];
            values[(kLogScaleIndex + axis) * count + i] = gaussians[i].log_scale_[axis];
            values[(kRotationIndex + axis) * count + i] = gaussians[i].rotation_[axis];
            for (int k = 0; k < 16; ++k) {
                values[(kShIndex + axis * 16 + k) * count + i] = gaussians[i].sh_(k, axis);
            }
        }
        values[kOpacityIndex * count + i] = gaussians[i].opacity_logit_;
    }
    return values;
}

struct DeviceGaussians {
    explicit DeviceGaussians(const std::vector<Gaussian>& gaussians) :
        parameters_(PackGaussians(gaussians)) {
        buffers_.gaussians_ = parameters_.Get();
        buffers_.gaussian_count_ = static_cast<int>(gaussians.size());
    }
    DeviceBuffer<float> parameters_;
    GaussianBuffers buffers_;
}; // struct DeviceGaussians

inline std::vector<Gaussian> MakeGaussians(int count) {
    std::vector<Gaussian> gaussians(count);
    for (int i = 0; i < count; ++i) {
        Gaussian& g = gaussians[i];
        g.position_ = Eigen::Vector3f((i % 16 - 7.5f) * 0.035f,
            ((i / 16) % 16 - 7.5f) * 0.035f, 3.0f + 0.001f * i);
        g.log_scale_ = Eigen::Vector3f(std::log(0.12f), std::log(0.16f), std::log(0.2f));
        g.rotation_ = Eigen::Vector3f(0.15f, -0.2f, 0.1f);
        g.opacity_logit_ = -1.0f;
        for (int c = 0; c < 3; ++c) {
            g.sh_(0, c) = (0.6f - 0.1f * c - 0.5f) / 0.28209479177387814f;
            for (int k = 1; k < 16; ++k) {
                g.sh_(k, c) = 0.02f * std::sin(float(k + c + i));
            }
        }
    }
    return gaussians;
}

inline CameraParams MakeCamera(
    int width,
    int height) {
    CameraParams camera = {};
    camera.width_ = width;
    camera.height_ = height;
    camera.fx_ = 20.0f;
    camera.fy_ = 21.0f;
    camera.cx_ = 0.5f * (width - 1);
    camera.cy_ = 0.5f * (height - 1);
    camera.R_c_w_[0] = camera.R_c_w_[4] = camera.R_c_w_[8] = 1;
    camera.near_depth_ = 0.2f;
    camera.far_depth_ = 100.0f;
    camera.tile_count_x_ = (width + 15) / 16;
    camera.tile_count_y_ = (height + 15) / 16;
    return camera;
}

inline Scene MakeScene() {
    Scene scene;
    Camera camera;
    camera.image_name_ = "synthetic.png";
    camera.image_size_ = cv::Size(9, 7);
    camera.K_ = (cv::Mat_<float>(3, 3) << 20, 0, 4, 0, 21, 3, 0, 0, 1);
    scene.train_cameras_.push_back(camera);
    scene.train_images_.emplace_back(7, 9, CV_8UC3, cv::Scalar(160, 170, 180));
    scene.point_cloud_.positions_ = {{-0.1f, -0.1f, 3}, {0.1f, -0.1f, 3},
        {-0.1f, 0.1f, 3}, {0.1f, 0.1f, 3}};
    scene.point_cloud_.colors_.assign(4, Eigen::Vector3f(0.6f, 0.5f, 0.4f));
    return scene;
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        static std::atomic<unsigned int> counter{0};
        path_ = std::filesystem::temp_directory_path() / ("gs-native-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(counter++));
        std::filesystem::create_directories(path_);
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }
    std::filesystem::path path_;
}; // class TemporaryDirectory

} // namespace gs_native::test
