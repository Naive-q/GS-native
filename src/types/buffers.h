#pragma once

#include <cstdint>
#include <vector>

#define GS_NATIVE_TILE_CACHE_SIZE 64

namespace gs_native {

constexpr int kPositionIndex = 0;
constexpr int kLogScaleIndex = 3;
constexpr int kRotationIndex = 6;
constexpr int kOpacityIndex = 9;
constexpr int kShIndex = 10;
constexpr int kMaxParamCount = 58;

struct CameraParams {
    float fx_, fy_, cx_, cy_;
    float R_c_w_[9];
    float t_c_w_[3];
    float near_depth_, far_depth_;

    int width_, height_;
    int64_t pixel_offset_;

    int tile_count_x_, tile_count_y_;
    int tile_offset_;
};

struct GaussianBuffers {
    float* gaussians_ = nullptr;
    int gaussian_count_ = 0;
};

struct ViewBuffers {
    CameraParams* cameras_ = nullptr;
    int camera_count_ = 0;
    int tile_count_ = 0;
    uint8_t* images_ = nullptr;
};

struct TileListBuffers {
    int* gaussian_ids_ = nullptr;
    float* gaussian_depths_ = nullptr;
    int* gaussian_counts_ = nullptr;
    int64_t* gaussian_offsets_ = nullptr;
    int64_t capacity_ = 0;
};

struct ColorGroupBuffers {
    int* colors_ = nullptr;
    int* gaussian_ids_ = nullptr;
    std::vector<int> offsets_;

    int GetColorCount() const {
        return static_cast<int>(offsets_.size()) - 1;
    }
};

}
