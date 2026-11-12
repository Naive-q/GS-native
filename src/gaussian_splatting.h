#pragma once

#include "density/density_controller.h"
#include "render/renderer.h"
#include "solver/solver.h"
#include "types/scene.h"

#include <cstddef>
#include <string>
#include <vector>

namespace gs_native {

class GaussianSplatting {
public:
    GaussianSplatting() = default;
    ~GaussianSplatting();

    GaussianSplatting(const GaussianSplatting&) = delete;
    GaussianSplatting& operator=(const GaussianSplatting&) = delete;

    bool Initialize(const Scene& scene);

    bool Train(int iterations);

    bool Render(const Camera& camera, cv::Mat& image);

    bool Save(const std::string& path);

private:
    bool InitializeGaussians();

    bool ControlDensity();

    Scene scene_;
    GaussianBuffers gaussians_;

    Renderer renderer_;
    Solver solver_;
    DensityController density_;

    float* image_ = nullptr;
    uint8_t* target_ = nullptr;
    size_t pixel_capacity_ = 0;

    float* density_output_ = nullptr;
    int* parent_ids_ = nullptr;
    int density_output_stride_ = 0;

    std::vector<int> camera_order_;
    int camera_cursor_ = 0;

    int completed_ = 0;
    int sh_degree_ = 0;
    float scene_extent_ = 0.0f;
    float background_[3] = {0.0f, 0.0f, 0.0f};
};

}
