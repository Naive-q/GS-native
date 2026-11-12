#include "gpu_test.h"

#include "cpu_reference.h"
#include "gaussian_splatting.h"
#include "io/data_io.h"

namespace gs_native::test {
using GaussianSplattingTest = GpuTest;

TEST_F(GaussianSplattingTest, InitializesIteratesSavesAndReloads) {
    auto scene = MakeScene();
    GaussianSplatting model;
    ASSERT_TRUE(model.Initialize(scene));
    cv::Mat before, after;
    ASSERT_TRUE(model.Render(scene.train_cameras_[0], before));
    ASSERT_TRUE(model.Train(3));
    ASSERT_TRUE(model.Render(scene.train_cameras_[0], after));
    ASSERT_TRUE(cv::checkRange(before));
    ASSERT_TRUE(cv::checkRange(after));
    EXPECT_LT(ComputeLoss(after, scene.train_images_[0]), ComputeLoss(before, scene.train_images_[0]));
    TemporaryDirectory directory;
    auto path = directory.path_ / "model.ply";
    ASSERT_TRUE(model.Save(path.string()));
    std::vector<Gaussian> loaded;
    ASSERT_TRUE(ReadGaussiansPly(path.string(), loaded));
    ASSERT_EQ(loaded.size(), 4u);
    DeviceGaussians device(loaded);
    auto camera = MakeCamera(9, 7);
    Renderer renderer;
    DeviceBuffer<float> image(9 * 7 * 3);
    float background[3] = {};
    ASSERT_TRUE(renderer.Render(device.buffers_, camera, background, image.Get(), 0));
    auto values = image.Download();
    cv::Mat reloaded(7, 9, CV_32FC3, values.data());
    ASSERT_TRUE(cv::checkRange(reloaded));
    EXPECT_LT(cv::norm(after, reloaded, cv::NORM_INF), 2e-5);
}

TEST_F(GaussianSplattingTest, CrossesDensityControlBoundary) {
    auto scene = MakeScene();
    GaussianSplatting model;
    ASSERT_TRUE(model.Initialize(scene));
    ASSERT_TRUE(model.Train(600));
    cv::Mat image;
    ASSERT_TRUE(model.Render(scene.train_cameras_[0], image));
    EXPECT_TRUE(cv::checkRange(image));
    TemporaryDirectory directory;
    auto path = directory.path_ / "density.ply";
    ASSERT_TRUE(model.Save(path.string()));
    std::vector<Gaussian> gaussians;
    ASSERT_TRUE(ReadGaussiansPly(path.string(), gaussians));
    EXPECT_EQ(gaussians.size(), 2 * scene.point_cloud_.positions_.size());
    for (const auto& g : gaussians) {
        EXPECT_TRUE(g.position_.allFinite());
        EXPECT_TRUE(g.log_scale_.allFinite());
        EXPECT_TRUE(g.sh_.allFinite());
    }
}

TEST_F(GaussianSplattingTest, RejectsEmptySceneAndSupportsZeroIterations) {
    GaussianSplatting model;
    EXPECT_FALSE(model.Initialize(Scene{}));
    auto scene = MakeScene();
    ASSERT_TRUE(model.Initialize(scene));
    cv::Mat before, after;
    ASSERT_TRUE(model.Render(scene.train_cameras_[0], before));
    ASSERT_TRUE(model.Train(0));
    ASSERT_TRUE(model.Render(scene.train_cameras_[0], after));
    ASSERT_TRUE(cv::checkRange(before));
    ASSERT_TRUE(cv::checkRange(after));
    EXPECT_DOUBLE_EQ(cv::norm(before, after, cv::NORM_INF), 0);
}
} // namespace gs_native::test
