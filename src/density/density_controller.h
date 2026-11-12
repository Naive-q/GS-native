#pragma once

#include "types/buffers.h"

#include <cstddef>

namespace gs_native {

class DensityController {
public:
    DensityController() = default;
    ~DensityController();
    DensityController(const DensityController&) = delete;
    DensityController& operator=(const DensityController&) = delete;

    bool Initialize(
        int gaussian_count,
        int capacity);

    bool
    AccumulateStatistics(const float* gradients, const float* radii, const int* gaussian_ids, int observation_count);

    bool AccumulateStatistics(
        const float* statistics,
        int gaussian_count);

    bool ResetOpacity(
        float* parameters,
        int stride);

    bool ControlDensity(
        const float* parameters,
        float*& output,
        int*& parent_ids,
        int input_stride,
        int& output_stride,
        double scene_extent,
        double gradient_threshold = 0.0002,
        double percent_dense = 0.01,
        double min_opacity = 0.005,
        double max_screen_size = 0.0,
        unsigned long long seed = 0,
        int* changes = nullptr);

    const int* GetGaussianCount() const {
        return control_;
    }
    const int* GetStatus() const {
        return control_ ? control_ + 2 : nullptr;
    }
    const int* GetParentIds() const {
        return samples_ ? reinterpret_cast<const int*>(samples_ + capacity_) : nullptr;
    }

private:
    bool Reserve(int capacity);

    float* gradient_sums_ = nullptr;
    float* samples_ = nullptr;
    unsigned int* observation_counts_ = nullptr;
    int* output_counts_ = nullptr;
    int* output_offsets_ = nullptr;
    int* control_ = nullptr;
    void* scan_storage_ = nullptr;
    size_t scan_storage_bytes_ = 0;
    int capacity_ = 0;
    int parameter_stride_ = 0;
    int random_blocks_ = 0;
    unsigned long long seed_ = 0;
};

}
