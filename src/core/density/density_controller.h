#pragma once

#include "core/types/buffers.h"

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

    bool AccumulateStatistics(
        const float* gradients,
        const float* radii,
        const int* gaussian_ids,
        int observation_count);

    bool AccumulateStatistics(
        const float* statistics,
        int gaussian_count);

    bool AccumulateL1Statistics(
        const unsigned int* tile_counts,
        int gaussian_count);

    bool ResetOpacity(
        float* parameters,
        int stride);

    bool ControlDensity(
        GaussianBuffers& gaussians,
        float scene_extent,
        float gradient_threshold = 0.0002f,
        float percent_dense = 0.01f,
        float min_opacity = 0.005f,
        float max_screen_size = 0.0f,
        unsigned long long seed = 0,
        int* changes = nullptr);

    const int* GetHistoryParentIds() const {
        return history_parent_ids_;
    }

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

    unsigned int* tile_counts_ = nullptr;
    float* spare_gaussian_parameters_ = nullptr;
    int* history_parent_ids_ = nullptr;
    int spare_gaussian_stride_ = 0;

    float* gradient_sums_ = nullptr;
    float* samples_ = nullptr;
    unsigned int* observation_counts_ = nullptr;
    int* density_counts_ = nullptr;
    int* density_offsets_ = nullptr;
    int* control_ = nullptr;
    void* scan_storage_ = nullptr;
    size_t scan_storage_bytes_ = 0;
    int capacity_ = 0;
    int parameter_stride_ = 0;
    int random_blocks_ = 0;
    unsigned long long seed_ = 0;
}; // class DensityController

} // namespace gs_native
