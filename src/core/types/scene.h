#pragma once

#include "core/types/camera.h"
#include "core/types/point_cloud.h"

#include <opencv2/core.hpp>

#include <vector>

namespace gs_native {

struct Scene {
    std::vector<Camera> train_cameras_;
    std::vector<Camera> test_cameras_;
    std::vector<cv::Mat> train_images_;
    std::vector<cv::Mat> test_images_;
    PointCloud point_cloud_;
}; // struct Scene

} // namespace gs_native
