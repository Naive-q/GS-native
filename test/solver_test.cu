#include "gpu_test.h"

#include "cpu_reference.h"
#include "render/renderer.h"
#include "solver/solver.h"

namespace gs_native::test {
using SolverTest = GpuTest;

float GetParameter(
    const Gaussian& g,
    int field) {
    if (field < 3) return g.position_[field];
    if (field < 6) return g.log_scale_[field - 3];
    if (field < 9) return g.rotation_[field - 6];
    if (field == 9) return g.opacity_logit_;
    return g.sh_((field - 10) % 16, (field - 10) / 16);
}

void SetParameter(
    Gaussian& g,
    int field,
    float value) {
    if (field < 3) g.position_[field] = value;
    else if (field < 6) g.log_scale_[field - 3] = value;
    else if (field < 9) g.rotation_[field - 6] = value;
    else if (field == 9) g.opacity_logit_ = value;
    else g.sh_((field - 10) % 16, (field - 10) / 16) = value;
}

std::vector<Gaussian> MakeDifferentiableScene(CameraParams& camera) {
    auto gaussians = MakeGaussians(1);
    Eigen::Matrix3f rotation = (Eigen::AngleAxisf(0.55f, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(0.35f, Eigen::Vector3f::UnitX()) *
        Eigen::AngleAxisf(0.2f, Eigen::Vector3f::UnitZ())).toRotationMatrix();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) camera.R_c_w_[3 * r + c] = rotation(r, c);
    }
    gaussians[0].position_ = rotation.transpose() * Eigen::Vector3f(0.2f, -0.1f, 3);
    gaussians[0].log_scale_ = Eigen::Vector3f(std::log(0.6f), std::log(0.45f), std::log(0.35f));
    return gaussians;
}

void RunStep(
    const CameraParams& camera,
    const DeviceBuffer<uint8_t>& target,
    Solver& solver,
    Renderer& renderer,
    DeviceGaussians& gaussians,
    int degree) {
    float background[3] = {};
    DeviceBuffer<float> image(size_t(camera.width_) * camera.height_ * 3);
    ASSERT_TRUE(renderer.Render(gaussians.buffers_, camera, background, image.Get(), degree));
    ASSERT_TRUE(solver.Iterate(camera, renderer, image.Get(), target.Get(), gaussians.buffers_, 0.0001f, degree));
    CheckCuda(cudaDeviceSynchronize());
}

TEST_F(SolverTest, UpdateDirectionsMatchFiniteDifferencesForAllShDegrees) {
    auto camera = MakeCamera(9, 7);
    auto gaussians = MakeDifferentiableScene(camera);
    cv::Mat target(7, 9, CV_8UC3, cv::Scalar(180, 190, 200));
    DeviceBuffer<uint8_t> target_device(std::vector<uint8_t>(target.data, target.data + target.total() * 3));
    for (int degree = 0; degree <= 3; ++degree) {
        DeviceGaussians device(gaussians);
        auto before = device.parameters_.Download();
        Renderer renderer;
        Solver solver;
        ASSERT_NO_FATAL_FAILURE(RunStep(camera, target_device, solver, renderer, device, degree));
        auto after = device.parameters_.Download();
        for (size_t index = 0; index < after.size(); ++index) {
            ASSERT_TRUE(std::isfinite(after[index])) << "degree=" << degree << " index=" << index;
        }
        int verified = 0;
        for (int field = 0; field < kMaxParamCount; ++field) {
            SCOPED_TRACE(::testing::Message() << "degree=" << degree << " field=" << field);
            bool active = field < kShIndex || (field - kShIndex) % 16 < (degree + 1) * (degree + 1);
            if (!active) {
                EXPECT_FLOAT_EQ(after[field], before[field]);
                continue;
            }
            auto high = gaussians, low = gaussians;
            float value = GetParameter(gaussians[0], field);
            SetParameter(high[0], field, value + 0.0005f);
            SetParameter(low[0], field, value - 0.0005f);
            double derivative = (ComputeLoss(RenderReference(high, camera, {0, 0, 0}, degree), target) -
                ComputeLoss(RenderReference(low, camera, {0, 0, 0}, degree), target)) /
                (GetParameter(high[0], field) - GetParameter(low[0], field));
            ASSERT_TRUE(std::isfinite(derivative));
            ASSERT_GT(std::abs(derivative), 1e-9);
            EXPECT_LT(derivative * (after[field] - before[field]), 0);
            ++verified;
        }
        EXPECT_EQ(verified, 10 + 3 * (degree + 1) * (degree + 1));
    }
}

TEST_F(SolverTest, OverlappingGaussiansWithBackgroundMatchFiniteDifferences) {
    auto camera = MakeCamera(9, 7);
    auto base = MakeDifferentiableScene(camera);
    auto gaussians = MakeGaussians(3);
    Eigen::Matrix3f rotation;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) rotation(r, c) = camera.R_c_w_[3 * r + c];
    }
    for (int i = 0; i < 3; ++i) {
        gaussians[i].position_ = rotation.transpose() * Eigen::Vector3f(0.2f - 0.12f * i, -0.1f + 0.09f * i, 3.0f + 0.4f * i);
        gaussians[i].log_scale_ = base[0].log_scale_;
        gaussians[i].opacity_logit_ = -0.4f + 0.3f * i;
        for (int c = 0; c < 3; ++c) {
            gaussians[i].sh_(0, c) = (0.25f + 0.15f * ((i + c) % 3) - 0.5f) / 0.28209479177387814f;
        }
    }
    std::reverse(gaussians.begin(), gaussians.end());
    std::array<float, 3> background = {0.12f, 0.31f, 0.57f};
    cv::Mat target(7, 9, CV_8UC3, cv::Scalar(180, 190, 200));
    DeviceBuffer<uint8_t> target_device(std::vector<uint8_t>(target.data, target.data + target.total() * 3));
    for (int degree = 0; degree <= 3; ++degree) {
        DeviceGaussians device(gaussians);
        auto before = device.parameters_.Download();
        DeviceBuffer<float> image(9 * 7 * 3);
        Renderer renderer;
        Solver solver;
        ASSERT_TRUE(renderer.Render(device.buffers_, camera, background.data(), image.Get(), degree));
        ASSERT_TRUE(solver.Iterate(camera, renderer, image.Get(), target_device.Get(), device.buffers_, 0.0001f, degree));
        auto after = device.parameters_.Download();
        for (size_t index = 0; index < after.size(); ++index) {
            ASSERT_TRUE(std::isfinite(after[index])) << "degree=" << degree << " index=" << index;
        }
        int verified = 0;
        for (int i = 0; i < 3; ++i) {
            for (int field = 0; field < kMaxParamCount; ++field) {
                SCOPED_TRACE(::testing::Message() << "degree=" << degree << " gaussian=" << i << " field=" << field);
                int offset = field * 3 + i;
                bool active = field < kShIndex || (field - kShIndex) % 16 < (degree + 1) * (degree + 1);
                if (!active) {
                    EXPECT_FLOAT_EQ(after[offset], before[offset]);
                    continue;
                }
                auto high = gaussians, low = gaussians;
                float value = GetParameter(gaussians[i], field);
                SetParameter(high[i], field, value + 0.0005f);
                SetParameter(low[i], field, value - 0.0005f);
                double derivative = (ComputeLoss(RenderReference(high, camera, background, degree), target) -
                    ComputeLoss(RenderReference(low, camera, background, degree), target)) /
                    (GetParameter(high[i], field) - GetParameter(low[i], field));
                ASSERT_TRUE(std::isfinite(derivative));
                ASSERT_GT(std::abs(derivative), 1e-9);
                EXPECT_LT(derivative * (after[offset] - before[offset]), 0);
                ++verified;
            }
        }
        EXPECT_EQ(verified, 3 * (10 + 3 * (degree + 1) * (degree + 1)));
    }
}

TEST_F(SolverTest, ProjectionStatisticMatchesScreenTranslationDifference) {
    auto camera = MakeCamera(9, 7);
    auto gaussians = MakeDifferentiableScene(camera);
    cv::Mat target(7, 9, CV_8UC3, cv::Scalar(180, 190, 200));
    DeviceBuffer<uint8_t> target_device(std::vector<uint8_t>(target.data, target.data + target.total() * 3));
    DeviceGaussians device(gaussians);
    Renderer renderer;
    Solver solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target_device, solver, renderer, device, 3));
    auto stats = ReadDevice(solver.GetProjectionStatistics(), 2);
    double derivatives[2];
    for (int axis = 0; axis < 2; ++axis) {
        auto high = camera, low = camera;
        (axis == 0 ? high.cx_ : high.cy_) += 0.001f;
        (axis == 0 ? low.cx_ : low.cy_) -= 0.001f;
        double difference = axis == 0 ? high.cx_ - low.cx_ : high.cy_ - low.cy_;
        derivatives[axis] = (ComputeLoss(RenderReference(gaussians, high, {0, 0, 0}, 3), target) -
            ComputeLoss(RenderReference(gaussians, low, {0, 0, 0}, 3), target)) / difference;
    }
    double expected = std::hypot(derivatives[0] * camera.width_ * 0.5, derivatives[1] * camera.height_ * 0.5);
    EXPECT_NEAR(stats[0], expected, 2e-5 + expected * 0.002);
    EXPECT_FLOAT_EQ(stats[1], 1);
}

TEST_F(SolverTest, ParentPermutationPreservesSubsequentUpdates) {
    auto gaussians = MakeGaussians(2);
    auto camera = MakeCamera(19, 17);
    DeviceGaussians first(gaussians), second(gaussians);
    DeviceBuffer<uint8_t> target(std::vector<uint8_t>(19 * 17 * 3, 180));
    Renderer first_renderer, second_renderer;
    Solver first_solver, second_solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, first_solver, first_renderer, first, 1));
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, second_solver, second_renderer, second, 1));
    auto values = second.parameters_.Download();
    for (int field = 0; field < kMaxParamCount; ++field) std::swap(values[field * 2], values[field * 2 + 1]);
    second.parameters_.Upload(values);
    DeviceBuffer<int> parents(std::vector<int>{1, 0});
    ASSERT_TRUE(second_solver.RemapGaussians(parents.Get(), 2));
    target.Upload(std::vector<uint8_t>(19 * 17 * 3, 20));
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, first_solver, first_renderer, first, 1));
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, second_solver, second_renderer, second, 1));
    auto a = first.parameters_.Download(), b = second.parameters_.Download();
    for (int field = 0; field < kMaxParamCount; ++field) {
        EXPECT_NEAR(a[field * 2], b[field * 2 + 1], 2e-6f);
        EXPECT_NEAR(a[field * 2 + 1], b[field * 2], 2e-6f);
    }
}

TEST_F(SolverTest, OpacityHistoryResetUsesZeroMomentsAtCurrentStep) {
    auto camera = MakeCamera(9, 7);
    auto gaussians = MakeDifferentiableScene(camera);
    DeviceGaussians device(gaussians);
    DeviceBuffer<uint8_t> target(std::vector<uint8_t>(9 * 7 * 3, 200));
    Renderer renderer;
    Solver solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, solver, renderer, device, 0));
    auto values = device.parameters_.Download();
    values[kOpacityIndex] = std::log(0.01f / 0.99f);
    device.parameters_.Upload(values);
    ASSERT_TRUE(solver.ResetOpacityHistory());
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, solver, renderer, device, 0));
    auto actual = device.parameters_.Download();
    double factor = (0.1 / (1 - std::pow(0.9, 2))) / std::sqrt(0.001 / (1 - std::pow(0.999, 2)));
    EXPECT_NEAR(actual[kOpacityIndex] - values[kOpacityIndex], 0.025 * factor, 2e-6);
}
TEST_F(SolverTest, NewParentStartsWithZeroHistory) {
    auto camera = MakeCamera(9, 7);
    auto gaussians = MakeDifferentiableScene(camera);
    DeviceGaussians device(gaussians);
    DeviceBuffer<uint8_t> target(std::vector<uint8_t>(9 * 7 * 3, 200));
    Renderer renderer;
    Solver solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, solver, renderer, device, 0));
    auto before = device.parameters_.Download();
    DeviceBuffer<int> parents(std::vector<int>{-1});
    ASSERT_TRUE(solver.RemapGaussians(parents.Get(), 1));
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target, solver, renderer, device, 0));
    auto actual = device.parameters_.Download();
    double factor = (0.1 / (1 - std::pow(0.9, 2))) / std::sqrt(0.001 / (1 - std::pow(0.999, 2)));
    EXPECT_NEAR(actual[kOpacityIndex] - before[kOpacityIndex], 0.025 * factor, 2e-6);
}
TEST_F(SolverTest, GradientsCrossGaussianBatchBoundary) {
    auto camera = MakeCamera(9, 7);
    auto gaussians = MakeGaussians(257);
    for (int i = 0; i < 257; ++i) {
        gaussians[i].position_ = Eigen::Vector3f(0.1f, -0.1f, 3.0f + 0.001f * i);
        gaussians[i].log_scale_.setConstant(std::log(0.8f));
        gaussians[i].opacity_logit_ = -4.4f;
        gaussians[i].sh_(0, 0) += 0.1f * std::sin(float(i));
    }
    std::reverse(gaussians.begin(), gaussians.end());
    cv::Mat target(7, 9, CV_8UC3, cv::Scalar(180, 190, 200));
    DeviceBuffer<uint8_t> target_device(std::vector<uint8_t>(target.data, target.data + target.total() * 3));
    DeviceGaussians device(gaussians);
    auto before = device.parameters_.Download();
    Renderer renderer;
    Solver solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target_device, solver, renderer, device, 0));
    auto after = device.parameters_.Download();
    for (size_t index = 0; index < after.size(); ++index) {
        ASSERT_TRUE(std::isfinite(after[index])) << "index=" << index;
    }
    for (int i : {0, 1, 128, 255, 256}) {
        for (int field : {kOpacityIndex, kShIndex, kShIndex + 16, kShIndex + 32}) {
            SCOPED_TRACE(::testing::Message() << "gaussian=" << i << " field=" << field);
            auto high = gaussians, low = gaussians;
            float value = GetParameter(gaussians[i], field);
            SetParameter(high[i], field, value + 0.0005f);
            SetParameter(low[i], field, value - 0.0005f);
            double derivative = (ComputeLoss(RenderReference(high, camera, {0, 0, 0}, 0), target) -
                ComputeLoss(RenderReference(low, camera, {0, 0, 0}, 0), target)) /
                (GetParameter(high[i], field) - GetParameter(low[i], field));
            ASSERT_TRUE(std::isfinite(derivative));
            ASSERT_GT(std::abs(derivative), 1e-9);
            int offset = field * 257 + i;
            EXPECT_LT(derivative * (after[offset] - before[offset]), 0);
        }
    }
}

TEST_F(SolverTest, ProjectionGradientMatchesAcrossSsimBlocksAndPartialEdges) {
    auto camera = MakeCamera(33, 65);
    auto gaussians = MakeDifferentiableScene(camera);
    gaussians[0].log_scale_ = Eigen::Vector3f(std::log(3.0f), std::log(2.5f), std::log(1.8f));
    cv::Mat target(65, 33, CV_8UC3);
    for (int y = 0; y < target.rows; ++y) {
        for (int x = 0; x < target.cols; ++x) {
            target.at<cv::Vec3b>(y, x) = cv::Vec3b(120 + (3 * x + y) % 100,
                100 + (x + 2 * y) % 120, 140 + (2 * x + 3 * y) % 100);
        }
    }
    DeviceBuffer<uint8_t> target_device(std::vector<uint8_t>(target.data, target.data + target.total() * 3));
    DeviceGaussians device(gaussians);
    Renderer renderer;
    Solver solver;
    ASSERT_NO_FATAL_FAILURE(RunStep(camera, target_device, solver, renderer, device, 3));
    auto stats = ReadDevice(solver.GetProjectionStatistics(), 2);
    double derivatives[2];
    for (int axis = 0; axis < 2; ++axis) {
        auto high = camera, low = camera;
        (axis == 0 ? high.cx_ : high.cy_) += 0.001f;
        (axis == 0 ? low.cx_ : low.cy_) -= 0.001f;
        double difference = axis == 0 ? high.cx_ - low.cx_ : high.cy_ - low.cy_;
        derivatives[axis] = (ComputeLoss(RenderReference(gaussians, high, {0, 0, 0}, 3), target) -
            ComputeLoss(RenderReference(gaussians, low, {0, 0, 0}, 3), target)) / difference;
        ASSERT_TRUE(std::isfinite(derivatives[axis]));
    }
    double expected = std::hypot(derivatives[0] * camera.width_ * 0.5, derivatives[1] * camera.height_ * 0.5);
    ASSERT_GT(expected, 1e-6);
    EXPECT_NEAR(stats[0], expected, 2e-5 + expected * 0.002);
    EXPECT_FLOAT_EQ(stats[1], 1);
}

} // namespace gs_native::test
