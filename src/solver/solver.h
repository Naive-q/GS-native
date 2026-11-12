#pragma once

#include "types/buffers.h"

#include <cstdint>

namespace gs_native {

class Renderer;

class Solver {
public:
    Solver() = default;
    ~Solver();

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;

    bool Iterate(
        const CameraParams& camera,
        const Renderer& renderer,
        const float* image,
        const uint8_t* target,
        GaussianBuffers& gaussians,
        float position_learning_rate,
        int sh_degree);

    bool ResetOpacityHistory();

    bool RemapGaussians(
        const int* parent_ids,
        int gaussian_count);

    const float* GetProjectionStatistics() const {
        return projection_statistics_;
    }

private:
    bool ComputeImageGradients(
        const CameraParams& camera,
        const float* image,
        const uint8_t* target);
    bool ComputeGaussianGradients(
        const CameraParams& camera,
        const Renderer& renderer,
        const GaussianBuffers& gaussians,
        const float* image,
        int sh_degree);
    bool UpdateGaussians(
        GaussianBuffers& gaussians,
        float position_learning_rate);

    float* image_gradients_ = nullptr;
    int64_t pixel_capacity_ = 0;

    float* projection_statistics_ = nullptr;
    float* gaussian_gradients_ = nullptr;
    float* first_moments_ = nullptr;
    float* second_moments_ = nullptr;
    int gaussian_capacity_ = 0;

    uint64_t iteration_ = 0;
};

}
