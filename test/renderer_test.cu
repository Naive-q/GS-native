#include "gpu_test.h"

#include "cpu_reference.h"
#include "render/renderer.h"

namespace gs_native::test {
using RendererTest = GpuTest;

TEST_F(RendererTest, MatchesCpuReferenceAcrossShDegreesAndPartialTiles) {
    auto gaussians = MakeGaussians(7);
    DeviceGaussians device(gaussians);
    Renderer renderer;
    std::array<float, 3> background = {0.05f, 0.1f, 0.15f};
    for (int degree = 0; degree <= 3; ++degree) {
        for (auto size : {cv::Size(9, 7), cv::Size(19, 17), cv::Size(33, 31)}) {
            SCOPED_TRACE(::testing::Message() << "degree=" << degree << " width=" << size.width);
            auto camera = MakeCamera(size.width, size.height);
            DeviceBuffer<float> output(size.area() * 3);
            ASSERT_TRUE(renderer.Render(device.buffers_, camera, background.data(), output.Get(), degree));
            auto values = output.Download();
            cv::Mat actual(size.height, size.width, CV_32FC3, values.data()), actual_double;
            actual.convertTo(actual_double, CV_64F);
            auto expected = RenderReference(gaussians, camera, background, degree);
            ASSERT_TRUE(cv::checkRange(actual_double));
            ASSERT_TRUE(cv::checkRange(expected));
            EXPECT_LT(cv::norm(actual_double, expected, cv::NORM_INF), 3e-5);
        }
    }
}

TEST_F(RendererTest, EmptyAndCulledScenesProduceBackground) {
    Renderer renderer;
    auto camera = MakeCamera(19, 17);
    std::array<float, 3> background = {0.2f, 0.4f, 0.6f};
    DeviceBuffer<float> image(camera.width_ * camera.height_ * 3);
    auto gaussians = MakeGaussians(2);
    gaussians[0].position_.z() = 0.1f;
    gaussians[1].position_.z() = 101;
    for (int count : {2, 0}) {
        gaussians.resize(count);
        DeviceGaussians device(gaussians);
        ASSERT_TRUE(renderer.Render(device.buffers_, camera, background.data(), image.Get(), 0));
        auto values = image.Download();
        for (size_t i = 0; i < values.size(); ++i) EXPECT_FLOAT_EQ(values[i], background[i % 3]);
    }
}

TEST_F(RendererTest, DepthOrderAndEarlyTerminationMatchReference) {
    auto gaussians = MakeGaussians(8);
    for (int i = 0; i < 8; ++i) {
        gaussians[i].position_ = Eigen::Vector3f(0, 0, 3.0f + i * 0.1f);
        gaussians[i].log_scale_.setZero();
        gaussians[i].opacity_logit_ = std::log(49.0f);
        gaussians[i].sh_(0, 0) = (0.2f + i * 0.08f - 0.5f) / 0.28209479177387814f;
    }
    std::reverse(gaussians.begin(), gaussians.end());
    DeviceGaussians device(gaussians);
    auto camera = MakeCamera(9, 7);
    std::array<float, 3> background = {0.1f, 0.2f, 0.3f};
    DeviceBuffer<float> image(9 * 7 * 3);
    Renderer renderer;
    ASSERT_TRUE(renderer.Render(device.buffers_, camera, background.data(), image.Get(), 0));
    auto values = image.Download();
    cv::Mat actual(7, 9, CV_32FC3, values.data()), converted;
    actual.convertTo(converted, CV_64F);
    auto expected = RenderReference(gaussians, camera, background, 0);
    ASSERT_TRUE(cv::checkRange(converted));
    ASSERT_TRUE(cv::checkRange(expected));
    EXPECT_LT(cv::norm(converted, expected, cv::NORM_INF), 3e-5);
}

TEST_F(RendererTest, LowerColorClampDoesNotClampUpperValues) {
    auto gaussians = MakeGaussians(1);
    gaussians[0].position_ = Eigen::Vector3f(0, 0, 3);
    gaussians[0].opacity_logit_ = std::log(9.0f);
    gaussians[0].sh_(0, 0) = (-0.2f - 0.5f) / 0.28209479177387814f;
    gaussians[0].sh_(0, 1) = (1.5f - 0.5f) / 0.28209479177387814f;
    DeviceGaussians device(gaussians);
    auto camera = MakeCamera(9, 7);
    float background[3] = {};
    DeviceBuffer<float> image(9 * 7 * 3);
    Renderer renderer;
    ASSERT_TRUE(renderer.Render(device.buffers_, camera, background, image.Get(), 0));
    auto values = image.Download();
    int center = (3 * 9 + 4) * 3;
    EXPECT_FLOAT_EQ(values[center], 0);
    EXPECT_NEAR(values[center + 1], 1.35f, 2e-6f);
}
TEST_F(RendererTest, TileListsCrossMultipleGaussianBatches) {
    auto camera = MakeCamera(9, 7);
    std::array<float, 3> background = {0.1f, 0.2f, 0.3f};
    for (int count : {257, 513}) {
        SCOPED_TRACE(::testing::Message() << "count=" << count);
        auto gaussians = MakeGaussians(count);
        for (int i = 0; i < count; ++i) {
            gaussians[i].position_ = Eigen::Vector3f(0.1f, -0.1f, 3.0f + 0.001f * i);
            gaussians[i].log_scale_.setConstant(std::log(0.8f));
            gaussians[i].opacity_logit_ = -4.4f;
            gaussians[i].sh_(0, 0) += 0.1f * std::sin(float(i));
        }
        std::reverse(gaussians.begin(), gaussians.end());
        DeviceGaussians device(gaussians);
        Renderer renderer;
        DeviceBuffer<float> output(9 * 7 * 3);
        ASSERT_TRUE(renderer.Render(device.buffers_, camera, background.data(), output.Get(), 0));
        auto values = output.Download();
        cv::Mat actual(7, 9, CV_32FC3, values.data()), converted;
        actual.convertTo(converted, CV_64F);
        auto expected = RenderReference(gaussians, camera, background, 0);
        ASSERT_TRUE(cv::checkRange(converted));
        ASSERT_TRUE(cv::checkRange(expected));
        EXPECT_LT(cv::norm(converted, expected, cv::NORM_INF), 3e-5);
    }
}

} // namespace gs_native::test
