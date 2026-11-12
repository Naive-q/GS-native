#pragma once

#include "test_support.h"

#include <opencv2/imgproc.hpp>
#include <array>
#include <numeric>

namespace gs_native::test {

inline std::array<double, 16> EvaluateBasis(const Eigen::Vector3d& d) {
    double x = d.x(), y = d.y(), z = d.z();
    return {0.28209479177387814, -0.4886025119029199 * y, 0.4886025119029199 * z,
        -0.4886025119029199 * x, 1.0925484305920792 * x * y, -1.0925484305920792 * y * z,
        0.31539156525252005 * (2 * z * z - x * x - y * y), -1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y), -0.5900435899266435 * y * (3 * x * x - y * y),
        2.890611442640554 * x * y * z, -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
        0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
        -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
        1.445305721320277 * z * (x * x - y * y), -0.5900435899266435 * x * (x * x - 3 * y * y)};
}

inline cv::Mat RenderReference(
    const std::vector<Gaussian>& gaussians,
    const CameraParams& camera,
    const std::array<float, 3>& background,
    int degree) {
    Eigen::Matrix3d view_rotation;
    Eigen::Vector3d view_translation;
    for (int r = 0; r < 3; ++r) {
        view_translation[r] = camera.t_c_w_[r];
        for (int c = 0; c < 3; ++c) {
            view_rotation(r, c) = camera.R_c_w_[r * 3 + c];
        }
    }
    std::vector<int> ids(gaussians.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        return (view_rotation * gaussians[a].position_.cast<double>())[2] <
            (view_rotation * gaussians[b].position_.cast<double>())[2];
    });
    cv::Mat image(camera.height_, camera.width_, CV_64FC3, cv::Scalar(0, 0, 0));
    cv::Mat transmittance(camera.height_, camera.width_, CV_64F, cv::Scalar(1));
    cv::Mat done(camera.height_, camera.width_, CV_8U, cv::Scalar(0));
    for (int id : ids) {
        const Gaussian& g = gaussians[id];
        Eigen::Vector3d position = g.position_.cast<double>();
        Eigen::Vector3d center = view_rotation * position + view_translation;
        double z = center.z();
        if (z <= camera.near_depth_ || z > camera.far_depth_) {
            continue;
        }
        Eigen::Vector3d omega = g.rotation_.cast<double>();
        Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
        if (omega.norm() >= 1e-8) {
            rotation = Eigen::AngleAxisd(omega.norm(), omega.normalized()).toRotationMatrix();
        }
        Eigen::Matrix3d transform = view_rotation * rotation * g.log_scale_.cast<double>().array().exp().matrix().asDiagonal();
        Eigen::Matrix3d covariance = transform * transform.transpose();
        double x = std::clamp(center.x() / z, -0.65 * camera.width_ / camera.fx_, 0.65 * camera.width_ / camera.fx_) * z;
        double y = std::clamp(center.y() / z, -0.65 * camera.height_ / camera.fy_, 0.65 * camera.height_ / camera.fy_) * z;
        Eigen::Matrix<double, 2, 3> projection;
        projection << camera.fx_ / z, 0, -camera.fx_ * x / (z * z),
            0, camera.fy_ / z, -camera.fy_ * y / (z * z);
        Eigen::Matrix2d projected = projection * covariance * projection.transpose();
        projected.diagonal().array() += 0.3;
        double a = projected(0, 0), b = projected(0, 1), c = projected(1, 1);
        double determinant = a * c - b * b;
        Eigen::Matrix2d inverse;
        inverse << c / determinant, -b / determinant, -b / determinant, a / determinant;
        double middle = 0.5 * (a + c);
        double radius = std::ceil(3 * std::sqrt(middle + std::sqrt(std::max(0.1, middle * middle - determinant))));
        double u = camera.fx_ * center.x() / z + camera.cx_;
        double v = camera.fy_ * center.y() / z + camera.cy_;
        int x0 = int(std::clamp((u - radius) / 16, 0.0, double(camera.tile_count_x_)));
        int y0 = int(std::clamp((v - radius) / 16, 0.0, double(camera.tile_count_y_)));
        int x1 = int(std::clamp((u + radius + 15) / 16, 0.0, double(camera.tile_count_x_)));
        int y1 = int(std::clamp((v + radius + 15) / 16, 0.0, double(camera.tile_count_y_)));
        auto basis = EvaluateBasis((position + view_rotation.transpose() * view_translation).normalized());
        Eigen::Vector3d color = Eigen::Vector3d::Constant(0.5);
        for (int k = 0; k < (degree + 1) * (degree + 1); ++k) {
            color += basis[k] * g.sh_.row(k).transpose().cast<double>();
        }
        color = color.cwiseMax(0);
        double opacity = 1.0 / (1.0 + std::exp(-double(g.opacity_logit_)));
        for (int py = y0 * 16; py < std::min(y1 * 16, camera.height_); ++py) {
            for (int px = x0 * 16; px < std::min(x1 * 16, camera.width_); ++px) {
                if (done.at<uint8_t>(py, px)) {
                    continue;
                }
                Eigen::Vector2d delta(u - px, v - py);
                double power = -0.5 * delta.dot(inverse * delta);
                double alpha = std::min(0.99, opacity * std::exp(power));
                if (power > 0 || alpha < 1.0 / 255) {
                    continue;
                }
                double& t = transmittance.at<double>(py, px);
                if (t * (1 - alpha) < 1e-4) {
                    done.at<uint8_t>(py, px) = 1;
                    continue;
                }
                cv::Vec3d& pixel = image.at<cv::Vec3d>(py, px);
                for (int channel = 0; channel < 3; ++channel) {
                    pixel[channel] += t * alpha * color[channel];
                }
                t *= 1 - alpha;
            }
        }
    }
    for (int y = 0; y < camera.height_; ++y) {
        for (int x = 0; x < camera.width_; ++x) {
            for (int c = 0; c < 3; ++c) {
                image.at<cv::Vec3d>(y, x)[c] += transmittance.at<double>(y, x) * background[c];
            }
        }
    }
    return image;
}

inline double ComputeLoss(
    const cv::Mat& image,
    const cv::Mat& target) {
    cv::Mat prediction, reference;
    image.convertTo(prediction, CV_64F);
    target.convertTo(reference, CV_64F, 1.0 / 255);
    cv::Mat kernel = cv::getGaussianKernel(11, 1.5, CV_64F);
    auto filter = [&](const cv::Mat& input) {
        cv::Mat result;
        cv::sepFilter2D(input, result, CV_64F, kernel, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
        return result;
    };
    cv::Mat mean_a = filter(prediction), mean_b = filter(reference);
    cv::Mat aa = filter(prediction.mul(prediction));
    cv::Mat bb = filter(reference.mul(reference));
    cv::Mat ab = filter(prediction.mul(reference));
    double l1 = 0, ssim = 0;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                double a = mean_a.at<cv::Vec3d>(y, x)[c], b = mean_b.at<cv::Vec3d>(y, x)[c];
                double va = aa.at<cv::Vec3d>(y, x)[c] - a * a;
                double vb = bb.at<cv::Vec3d>(y, x)[c] - b * b;
                double covariance = ab.at<cv::Vec3d>(y, x)[c] - a * b;
                ssim += (2 * a * b + 0.0001) * (2 * covariance + 0.0009) /
                    ((a * a + b * b + 0.0001) * (va + vb + 0.0009));
                l1 += std::abs(prediction.at<cv::Vec3d>(y, x)[c] - reference.at<cv::Vec3d>(y, x)[c]);
            }
        }
    }
    double count = 3.0 * image.total();
    return 0.8 * l1 / count + 0.2 * (1 - ssim / count);
}

} // namespace gs_native::test
