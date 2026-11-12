#include "io/data_io.h"

#include "util/log.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace gs_native {

namespace {

constexpr int32_t kPinholeModelId = 1;
constexpr std::streamsize kTrackElementBytes = sizeof(int32_t) * 2;

constexpr int kMinFeatureCount = 50;
constexpr float kFeatureDepthLowQuantile = 0.01f;
constexpr float kFeatureDepthHighQuantile = 0.99f;

constexpr int kShRestCount = 15;
constexpr int kPlyFloatCount = 3 + 3 + 3 + 3 * kShRestCount + 1 + 3 + 4;

struct ColmapCamera {
    int id_ = 0;
    int width_ = 0;
    int height_ = 0;
    float fx_ = 0.0f;
    float fy_ = 0.0f;
    float cx_ = 0.0f;
    float cy_ = 0.0f;
};

struct ColmapImage {
    int id_ = 0;
    int camera_id_ = 0;
    std::string name_;
    cv::Mat R_c_w_;
    cv::Mat t_c_w_;
    std::vector<int64_t> point3d_ids_;
};

template<typename T> T ReadValue(std::ifstream& file) {
    T value{};
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

cv::Mat QvecToR(
    float w,
    float x,
    float y,
    float z) {
    float norm = std::sqrt(w * w + x * x + y * y + z * z);
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    return (cv::Mat_<float>(3, 3) << 1.0f - 2.0f * (y * y + z * z),
        2.0f * (x * y - w * z),
        2.0f * (x * z + w * y),
        2.0f * (x * y + w * z),
        1.0f - 2.0f * (x * x + z * z),
        2.0f * (y * z - w * x),
        2.0f * (x * z - w * y),
        2.0f * (y * z + w * x),
        1.0f - 2.0f * (x * x + y * y));
}

bool ReadCamerasBin(
    const std::string& path,
    ColmapCamera& camera) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        GS_NATIVE_ERROR("cannot open {}", path);
        return false;
    }

    uint64_t num_cameras = ReadValue<uint64_t>(file);
    if (num_cameras != 1) {
        GS_NATIVE_ERROR("only single camera is supported, {} has {}", path, num_cameras);
        return false;
    }

    camera.id_ = ReadValue<int32_t>(file);
    int32_t model_id = ReadValue<int32_t>(file);
    if (model_id != kPinholeModelId) {
        GS_NATIVE_ERROR("only PINHOLE camera is supported, model id is {}", model_id);
        return false;
    }

    camera.width_ = static_cast<int>(ReadValue<uint64_t>(file));
    camera.height_ = static_cast<int>(ReadValue<uint64_t>(file));
    camera.fx_ = static_cast<float>(ReadValue<double>(file));
    camera.fy_ = static_cast<float>(ReadValue<double>(file));
    camera.cx_ = static_cast<float>(ReadValue<double>(file));
    camera.cy_ = static_cast<float>(ReadValue<double>(file));

    if (!file) {
        GS_NATIVE_ERROR("failed to read {}", path);
        return false;
    }
    return true;
}

bool ReadImagesBin(
    const std::string& path,
    std::vector<ColmapImage>& images) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        GS_NATIVE_ERROR("cannot open {}", path);
        return false;
    }

    uint64_t num_images = ReadValue<uint64_t>(file);
    images.clear();
    images.reserve(num_images);

    for (uint64_t i = 0; i < num_images; ++i) {
        ColmapImage image;
        image.id_ = ReadValue<int32_t>(file);

        float qw = static_cast<float>(ReadValue<double>(file));
        float qx = static_cast<float>(ReadValue<double>(file));
        float qy = static_cast<float>(ReadValue<double>(file));
        float qz = static_cast<float>(ReadValue<double>(file));
        image.R_c_w_ = QvecToR(qw, qx, qy, qz);

        float tx = static_cast<float>(ReadValue<double>(file));
        float ty = static_cast<float>(ReadValue<double>(file));
        float tz = static_cast<float>(ReadValue<double>(file));
        image.t_c_w_ = (cv::Mat_<float>(3, 1) << tx, ty, tz);

        image.camera_id_ = ReadValue<int32_t>(file);
        std::getline(file, image.name_, '\0');

        uint64_t num_points2d = ReadValue<uint64_t>(file);
        for (uint64_t k = 0; k < num_points2d; ++k) {
            file.ignore(sizeof(double) * 2);
            int64_t point3d_id = ReadValue<int64_t>(file);
            if (point3d_id >= 0) {
                image.point3d_ids_.push_back(point3d_id);
            }
        }

        if (!file) {
            GS_NATIVE_ERROR("failed to read {}", path);
            return false;
        }
        images.push_back(image);
    }
    return true;
}

bool ReadPoints3dBin(
    const std::string& path,
    PointCloud& cloud,
    std::unordered_map<int64_t, int>& point_indices) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        GS_NATIVE_ERROR("cannot open {}", path);
        return false;
    }

    uint64_t num_points = ReadValue<uint64_t>(file);
    cloud.positions_.clear();
    cloud.colors_.clear();
    cloud.positions_.reserve(num_points);
    cloud.colors_.reserve(num_points);
    point_indices.clear();
    point_indices.reserve(num_points);

    for (uint64_t i = 0; i < num_points; ++i) {
        point_indices[static_cast<int64_t>(ReadValue<uint64_t>(file))] = static_cast<int>(i);

        float x = static_cast<float>(ReadValue<double>(file));
        float y = static_cast<float>(ReadValue<double>(file));
        float z = static_cast<float>(ReadValue<double>(file));

        uint8_t r = ReadValue<uint8_t>(file);
        uint8_t g = ReadValue<uint8_t>(file);
        uint8_t b = ReadValue<uint8_t>(file);

        [[maybe_unused]] double error = ReadValue<double>(file);

        uint64_t track_length = ReadValue<uint64_t>(file);
        file.ignore(static_cast<std::streamsize>(track_length) * kTrackElementBytes);

        if (!file) {
            GS_NATIVE_ERROR("failed to read {}", path);
            return false;
        }
        cloud.positions_.emplace_back(x, y, z);
        cloud.colors_.emplace_back(r / 255.0f, g / 255.0f, b / 255.0f);
    }
    return true;
}

bool LoadImage(
    const std::string& path,
    int downscale,
    cv::Mat& image) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        GS_NATIVE_ERROR("cannot read image {}", path);
        return false;
    }

    cv::cvtColor(bgr, image, cv::COLOR_BGR2RGB);

    if (downscale > 1) {
        cv::Size size(static_cast<int>(std::lround(image.cols / static_cast<float>(downscale))),
            static_cast<int>(std::lround(image.rows / static_cast<float>(downscale))));
        cv::resize(image, image, size, 0.0, 0.0, cv::INTER_AREA);
    }
    return true;
}

cv::Mat ScaleIntrinsics(
    const ColmapCamera& camera,
    const cv::Size& image_size) {
    float sx = image_size.width / static_cast<float>(camera.width_);
    float sy = image_size.height / static_cast<float>(camera.height_);

    float fx = camera.fx_ * sx;
    float fy = camera.fy_ * sy;
    float cx = camera.cx_ * sx - 0.5f;
    float cy = camera.cy_ * sy - 0.5f;

    return (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
}

bool ComputeFeatureDepths(
    const ColmapImage& image,
    const PointCloud& cloud,
    const std::unordered_map<int64_t, int>& point_indices,
    Camera& camera) {
    std::vector<float> depths;
    depths.reserve(image.point3d_ids_.size());
    for (int64_t point3d_id : image.point3d_ids_) {
        auto it = point_indices.find(point3d_id);
        if (it == point_indices.end()) {
            continue;
        }
        const Eigen::Vector3f& position = cloud.positions_[it->second];
        float depth = image.R_c_w_.at<float>(2, 0) * position.x() + image.R_c_w_.at<float>(2, 1) * position.y() +
            image.R_c_w_.at<float>(2, 2) * position.z() + image.t_c_w_.at<float>(2);
        if (depth > 0.0f) {
            depths.push_back(depth);
        }
    }
    if (static_cast<int>(depths.size()) < kMinFeatureCount) {
        return false;
    }

    std::sort(depths.begin(), depths.end());
    float last_index = static_cast<float>(depths.size() - 1);
    camera.feature_depth_low_ = depths[static_cast<size_t>(kFeatureDepthLowQuantile * last_index)];
    camera.feature_depth_high_ = depths[static_cast<size_t>(kFeatureDepthHighQuantile * last_index)];
    return true;
}

void FillMissingFeatureDepths(
    std::vector<Camera*>& cameras,
    const std::vector<bool>& is_valid) {
    std::vector<float> lows;
    std::vector<float> highs;
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (is_valid[i]) {
            lows.push_back(cameras[i]->feature_depth_low_);
            highs.push_back(cameras[i]->feature_depth_high_);
        }
    }
    if (lows.empty()) {
        return;
    }
    std::nth_element(lows.begin(), lows.begin() + lows.size() / 2, lows.end());
    std::nth_element(highs.begin(), highs.begin() + highs.size() / 2, highs.end());
    for (size_t i = 0; i < cameras.size(); ++i) {
        if (!is_valid[i]) {
            cameras[i]->feature_depth_low_ = lows[lows.size() / 2];
            cameras[i]->feature_depth_high_ = highs[highs.size() / 2];
        }
    }
}

Eigen::Vector4f AxisAngleToQuaternion(const Eigen::Vector3f& omega) {
    float theta = omega.norm();
    float k = theta < 1e-8f ? 0.5f : std::sin(0.5f * theta) / theta;
    return Eigen::Vector4f(std::cos(0.5f * theta), k * omega.x(), k * omega.y(), k * omega.z());
}

void WritePlyHeader(
    std::ofstream& file,
    size_t count) {
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << count << "\n";
    for (const char* name : {"x", "y", "z", "nx", "ny", "nz"}) {
        file << "property float " << name << "\n";
    }
    for (int i = 0; i < 3; ++i) {
        file << "property float f_dc_" << i << "\n";
    }
    for (int i = 0; i < 3 * kShRestCount; ++i) {
        file << "property float f_rest_" << i << "\n";
    }
    file << "property float opacity\n";
    for (int i = 0; i < 3; ++i) {
        file << "property float scale_" << i << "\n";
    }
    for (int i = 0; i < 4; ++i) {
        file << "property float rot_" << i << "\n";
    }
    file << "end_header\n";
}

}

bool ReadColmapScene(
    const std::string& scene_dir,
    int downscale,
    Scene& scene) {
    GS_NATIVE_INFO("Reading COLMAP scene: {}, downscale {}.", scene_dir, downscale);
    std::string sparse_dir = scene_dir + "/sparse/0/";
    std::string image_dir = scene_dir + "/images/";

    ColmapCamera colmap_camera;
    if (!ReadCamerasBin(
            sparse_dir + "cameras.bin",
            colmap_camera)) {
        return false;
    }

    std::vector<ColmapImage> colmap_images;
    if (!ReadImagesBin(
            sparse_dir + "images.bin",
            colmap_images)) {
        return false;
    }
    std::sort(colmap_images.begin(), colmap_images.end(), [](const ColmapImage& a, const ColmapImage& b) {
        return a.name_ < b.name_;
    });

    std::unordered_map<int64_t, int> point_indices;
    if (!ReadPoints3dBin(
            sparse_dir + "points3D.bin",
            scene.point_cloud_,
            point_indices)) {
        return false;
    }

    scene.train_cameras_.clear();
    scene.test_cameras_.clear();
    scene.train_images_.clear();
    scene.test_images_.clear();
    std::vector<bool> train_is_valid;
    std::vector<bool> test_is_valid;

    for (size_t i = 0; i < colmap_images.size(); ++i) {
        const ColmapImage& colmap_image = colmap_images[i];
        if (colmap_image.camera_id_ != colmap_camera.id_) {
            GS_NATIVE_ERROR("image {} uses unknown camera id {}", colmap_image.name_, colmap_image.camera_id_);
            return false;
        }

        cv::Mat image;
        if (!LoadImage(
                image_dir + colmap_image.name_,
                downscale,
                image)) {
            return false;
        }

        Camera camera;
        camera.image_name_ = colmap_image.name_;
        camera.image_size_ = image.size();
        camera.K_ = ScaleIntrinsics(colmap_camera, image.size());
        camera.R_c_w_ = colmap_image.R_c_w_;
        camera.t_c_w_ = colmap_image.t_c_w_;
        bool is_valid = ComputeFeatureDepths(colmap_image, scene.point_cloud_, point_indices, camera);

        if (i % 8 == 0) {
            scene.test_cameras_.push_back(camera);
            scene.test_images_.push_back(image);
            test_is_valid.push_back(is_valid);
        }
        else {
            scene.train_cameras_.push_back(camera);
            scene.train_images_.push_back(image);
            train_is_valid.push_back(is_valid);
        }
    }

    std::vector<Camera*> cameras;
    std::vector<bool> is_valid;
    for (size_t i = 0; i < scene.train_cameras_.size(); ++i) {
        cameras.push_back(&scene.train_cameras_[i]);
        is_valid.push_back(train_is_valid[i]);
    }
    for (size_t i = 0; i < scene.test_cameras_.size(); ++i) {
        cameras.push_back(&scene.test_cameras_[i]);
        is_valid.push_back(test_is_valid[i]);
    }
    FillMissingFeatureDepths(cameras, is_valid);
    GS_NATIVE_INFO("Scene loaded: training cameras {}, test cameras {}, points {}.",
        scene.train_cameras_.size(), scene.test_cameras_.size(), scene.point_cloud_.positions_.size());
    return true;
}

bool ReadGaussiansPly(
    const std::string& path,
    std::vector<Gaussian>& gaussians) {
    std::ifstream file(path, std::ios::binary);
    std::string line;
    if (!file || !std::getline(file, line) || (line != "ply" && line != "ply\r")) {
        GS_NATIVE_ERROR("Cannot read PLY header: {}.", path);
        return false;
    }
    bool binary = false, header_complete = false, vertex = false;
    int64_t count = -1;
    std::vector<std::string> properties;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::string token;
        stream >> token;
        if (token == "end_header") {
            header_complete = true;
            break;
        }
        if (token == "format") {
            std::string format, version;
            stream >> format >> version;
            binary = format == "binary_little_endian" && version == "1.0";
        }
        else if (token == "element") {
            std::string element;
            int64_t elements = -1;
            stream >> element >> elements;
            if (!stream || elements < 0 || (count < 0 && element != "vertex") ||
                (element == "vertex" && count >= 0)) {
                GS_NATIVE_ERROR("Unsupported PLY element layout: {}.", path);
                return false;
            }
            vertex = element == "vertex";
            if (vertex) {
                count = elements;
            }
        }
        else if (token == "property" && vertex) {
            std::string type, name;
            stream >> type >> name;
            if (!stream || (type != "float" && type != "float32") ||
                std::find(properties.begin(), properties.end(), name) != properties.end()) {
                GS_NATIVE_ERROR("Unsupported PLY vertex property: {}.", line);
                return false;
            }
            properties.push_back(name);
        }
    }
    if (!binary || !header_complete || count <= 0 || count > std::numeric_limits<int>::max()) {
        GS_NATIVE_ERROR("Expected a nonempty binary little-endian Gaussian PLY: {}.", path);
        return false;
    }

    std::vector<std::string> names = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"};
    for (int c = 0; c < 3; ++c) {
        names.push_back("f_dc_" + std::to_string(c));
    }
    for (int j = 0; j < 3 * kShRestCount; ++j) {
        names.push_back("f_rest_" + std::to_string(j));
    }
    std::vector<size_t> indices;
    for (const std::string& name : names) {
        auto it = std::find(properties.begin(), properties.end(), name);
        if (it == properties.end()) {
            GS_NATIVE_ERROR("Missing PLY property {}: {}.", name, path);
            return false;
        }
        indices.push_back(static_cast<size_t>(it - properties.begin()));
    }
    std::streampos data_begin = file.tellg();
    file.seekg(0, std::ios::end);
    std::streamoff available = file.tellg() - data_begin;
    size_t row_bytes = properties.size() * sizeof(float);
    if (available < 0 || static_cast<uint64_t>(available) / row_bytes < static_cast<uint64_t>(count)) {
        GS_NATIVE_ERROR("Truncated PLY data: {}.", path);
        return false;
    }
    file.seekg(data_begin);
    std::vector<Gaussian> loaded(static_cast<size_t>(count));
    std::vector<float> row(properties.size());
    for (Gaussian& gaussian : loaded) {
        if (!file.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row_bytes))) {
            GS_NATIVE_ERROR("Reading PLY data failed: {}.", path);
            return false;
        }
        for (size_t index : indices) {
            if (!std::isfinite(row[index])) {
                GS_NATIVE_ERROR("Non-finite Gaussian parameter in {}.", path);
                return false;
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            gaussian.position_[axis] = row[indices[axis]];
            gaussian.log_scale_[axis] = row[indices[4 + axis]];
        }
        gaussian.opacity_logit_ = row[indices[3]];
        Eigen::Quaternionf quaternion(row[indices[7]], row[indices[8]], row[indices[9]], row[indices[10]]);
        float norm = quaternion.norm();
        if (!(norm > 0) || !std::isfinite(norm)) {
            GS_NATIVE_ERROR("Invalid Gaussian rotation in {}.", path);
            return false;
        }
        quaternion.normalize();
        Eigen::AngleAxisf rotation(quaternion);
        gaussian.rotation_ = rotation.axis() * rotation.angle();
        for (int c = 0; c < 3; ++c) {
            gaussian.sh_(0, c) = row[indices[11 + c]];
            for (int j = 1; j <= kShRestCount; ++j) {
                gaussian.sh_(j, c) = row[indices[14 + c * kShRestCount + j - 1]];
            }
        }
    }
    gaussians.swap(loaded);
    GS_NATIVE_INFO("Loaded {} gaussians from {}.", count, path);
    return true;
}

bool WriteGaussiansPly(
    const std::string& path,
    const std::vector<Gaussian>& gaussians) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        GS_NATIVE_ERROR("cannot open {}", path);
        return false;
    }

    WritePlyHeader(file, gaussians.size());

    std::array<float, kPlyFloatCount> row;
    for (const Gaussian& g : gaussians) {
        int n = 0;
        row[n++] = g.position_.x();
        row[n++] = g.position_.y();
        row[n++] = g.position_.z();
        row[n++] = 0.0f;
        row[n++] = 0.0f;
        row[n++] = 0.0f;

        for (int c = 0; c < 3; ++c) {
            row[n++] = g.sh_(0, c);
        }
        for (int c = 0; c < 3; ++c) {
            for (int k = 1; k <= kShRestCount; ++k) {
                row[n++] = g.sh_(k, c);
            }
        }

        row[n++] = g.opacity_logit_;

        row[n++] = g.log_scale_.x();
        row[n++] = g.log_scale_.y();
        row[n++] = g.log_scale_.z();

        Eigen::Vector4f q = AxisAngleToQuaternion(g.rotation_);
        for (int i = 0; i < 4; ++i) {
            row[n++] = q[i];
        }

        file.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * row.size());
    }

    if (!file) {
        GS_NATIVE_ERROR("failed to write {}", path);
        return false;
    }
    GS_NATIVE_INFO("saved {} gaussians to {}", gaussians.size(), path);
    return true;
}

}
