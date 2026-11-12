#pragma once

#include <Eigen/Core>

#include <vector>

namespace gs_native {

struct PointCloud {
    std::vector<Eigen::Vector3f> positions_;
    std::vector<Eigen::Vector3f> colors_;
};

}
