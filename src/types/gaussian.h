#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

namespace gs_native {

using SHCoeffs = Eigen::Matrix<float, 16, 3>;

struct Gaussian {
    Eigen::Vector3f position_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f log_scale_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f rotation_ = Eigen::Vector3f::Zero();
    float opacity_logit_ = 0.0f;
    SHCoeffs sh_ = SHCoeffs::Zero();

    Eigen::Vector3f GetScale() const {
        return log_scale_.array().exp();
    }

    float GetMaxScale() const {
        return std::exp(log_scale_.maxCoeff());
    }

    Eigen::Matrix3f GetRotationMatrix() const {
        float angle = rotation_.norm();
        if (angle < 1e-8f) {
            return Eigen::Matrix3f::Identity();
        }
        return Eigen::AngleAxisf(angle, rotation_ / angle).toRotationMatrix();
    }

    Eigen::Vector3f GetExtent(float sigma_multiplier) const {
        return sigma_multiplier * (GetRotationMatrix() * GetScale().asDiagonal()).rowwise().norm();
    }

    float GetOpacity() const {
        return 1.0f / (1.0f + std::exp(-opacity_logit_));
    }
};

}
