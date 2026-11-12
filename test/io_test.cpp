#include "test_support.h"

#include "io/data_io.h"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace gs_native::test {

template<typename T>
void WriteBinary(
    std::ofstream& file,
    T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void WriteColmapFixture(const std::filesystem::path& root) {
    auto sparse = root / "sparse" / "0";
    std::filesystem::create_directories(sparse);
    std::filesystem::create_directories(root / "images");
    std::ofstream cameras(sparse / "cameras.bin", std::ios::binary);
    WriteBinary(cameras, uint64_t(1));
    WriteBinary(cameras, int32_t(1)); WriteBinary(cameras, int32_t(1));
    WriteBinary(cameras, uint64_t(19)); WriteBinary(cameras, uint64_t(17));
    for (double value : {20.0, 21.0, 9.0, 8.0}) WriteBinary(cameras, value);
    std::ofstream images(sparse / "images.bin", std::ios::binary);
    WriteBinary(images, uint64_t(9));
    for (int i = 8; i >= 0; --i) {
        std::ostringstream name;
        name << "frame" << std::setw(2) << std::setfill('0') << i << ".png";
        WriteBinary(images, int32_t(i + 1));
        for (double value : {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}) WriteBinary(images, value);
        WriteBinary(images, int32_t(1));
        images.write(name.str().c_str(), name.str().size() + 1);
        WriteBinary(images, uint64_t(1));
        WriteBinary(images, 0.0); WriteBinary(images, 0.0); WriteBinary(images, int64_t(1));
        cv::Mat image(17, 19, CV_8UC3, cv::Scalar(10, 20, 30));
        if (!cv::imwrite((root / "images" / name.str()).string(), image)) throw std::runtime_error("Writing fixture failed");
    }
    std::ofstream points(sparse / "points3D.bin", std::ios::binary);
    WriteBinary(points, uint64_t(4));
    for (int i = 0; i < 4; ++i) {
        WriteBinary(points, uint64_t(i + 1));
        for (double value : {0.1 * (i % 2), 0.1 * (i / 2), 3.0}) WriteBinary(points, value);
        for (uint8_t value : {uint8_t(100), uint8_t(150), uint8_t(200)}) WriteBinary(points, value);
        WriteBinary(points, 0.0); WriteBinary(points, uint64_t(0));
    }
}

TEST(IoTest, LoadsColmapSplitRgbAndScaledIntrinsics) {
    TemporaryDirectory directory;
    WriteColmapFixture(directory.path_);
    Scene scene;
    ASSERT_TRUE(ReadColmapScene(directory.path_.string(), 2, scene));
    ASSERT_EQ(scene.train_cameras_.size(), 7u);
    ASSERT_EQ(scene.test_cameras_.size(), 2u);
    ASSERT_EQ(scene.train_images_.size(), 7u);
    EXPECT_EQ(scene.test_cameras_[0].image_name_, "frame00.png");
    EXPECT_EQ(scene.test_cameras_[1].image_name_, "frame08.png");
    EXPECT_EQ(scene.train_cameras_[0].image_size_, cv::Size(10, 9));
    EXPECT_NEAR(scene.train_cameras_[0].K_.at<float>(0, 0), 20.0 * 10 / 19, 1e-6);
    EXPECT_NEAR(scene.train_cameras_[0].K_.at<float>(1, 1), 21.0 * 9 / 17, 1e-6);
    EXPECT_EQ(scene.train_images_[0].at<cv::Vec3b>(0, 0), cv::Vec3b(30, 20, 10));
    ASSERT_EQ(scene.point_cloud_.positions_.size(), 4u);
    ASSERT_EQ(scene.point_cloud_.colors_.size(), 4u);
    EXPECT_NEAR(scene.point_cloud_.colors_[0].x(), 100.0 / 255, 1e-6);
}

TEST(IoTest, GaussianPlyRoundTripPreservesParametersAndRotation) {
    TemporaryDirectory directory;
    auto expected = MakeGaussians(3);
    expected[1].rotation_.setZero();
    expected[2].rotation_ = Eigen::Vector3f(1.0f, -0.7f, 0.2f);
    auto path = directory.path_ / "model.ply";
    ASSERT_TRUE(WriteGaussiansPly(path.string(), expected));
    std::vector<Gaussian> actual;
    ASSERT_TRUE(ReadGaussiansPly(path.string(), actual));
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_TRUE(actual[i].position_.isApprox(expected[i].position_, 1e-7f));
        EXPECT_TRUE(actual[i].log_scale_.isApprox(expected[i].log_scale_, 1e-7f));
        EXPECT_FLOAT_EQ(actual[i].opacity_logit_, expected[i].opacity_logit_);
        EXPECT_TRUE(actual[i].sh_.isApprox(expected[i].sh_, 1e-7f));
        EXPECT_TRUE(actual[i].GetRotationMatrix().isApprox(expected[i].GetRotationMatrix(), 2e-6f));
    }
}

TEST(IoTest, RejectsMissingAndTruncatedInputs) {
    TemporaryDirectory directory;
    Scene scene;
    EXPECT_FALSE(ReadColmapScene(directory.path_.string(), 1, scene));
    auto path = directory.path_ / "truncated.ply";
    ASSERT_TRUE(WriteGaussiansPly(path.string(), MakeGaussians(2)));
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 4);
    std::vector<Gaussian> result;
    EXPECT_FALSE(ReadGaussiansPly(path.string(), result));
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(ReadGaussiansPly((directory.path_ / "missing.ply").string(), result));
}
} // namespace gs_native::test
