#pragma once

#include "core/types/buffers.h"

#include <opencv2/core.hpp>

#include <limits>
#include <string>

namespace gs_native {

struct Camera {
    std::string image_name_;
    cv::Size image_size_;
    cv::Mat K_ = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat R_c_w_ = cv::Mat::eye(3, 3, CV_32F);
    cv::Mat t_c_w_ = cv::Mat::zeros(3, 1, CV_32F);
    float feature_depth_low_ = std::numeric_limits<float>::min();
    float feature_depth_high_ = std::numeric_limits<float>::max();

    CameraParams GetCameraParams() const {
        CameraParams params = {};
        params.width_ = image_size_.width;
        params.height_ = image_size_.height;
        params.fx_ = K_.at<float>(0, 0);
        params.fy_ = K_.at<float>(1, 1);
        params.cx_ = K_.at<float>(0, 2);
        params.cy_ = K_.at<float>(1, 2);
        for (int row = 0; row < 3; ++row) {
            params.t_c_w_[row] = t_c_w_.at<float>(row);
            for (int col = 0; col < 3; ++col) {
                params.R_c_w_[row * 3 + col] = R_c_w_.at<float>(row, col);
            }
        }
        params.near_depth_ = 0.2f;
        params.far_depth_ = std::numeric_limits<float>::max();
        params.tile_count_x_ = (params.width_ + 15) / 16;
        params.tile_count_y_ = (params.height_ + 15) / 16;
        return params;
    }
}; // struct Camera

} // namespace gs_native
