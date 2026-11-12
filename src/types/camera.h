#pragma once

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
};

}
