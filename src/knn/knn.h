#pragma once

#include <Eigen/Core>

#include <vector>

namespace gs_native {

std::vector<float> ComputeMeanDist2Cuda(const std::vector<Eigen::Vector3f>& points);

}
