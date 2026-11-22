#include "gpu_test.h"

#include "density/density_controller.h"

namespace gs_native::test {
using DensityTest = GpuTest;

struct DensityResult {
    DensityResult() = default;
    DensityResult(const DensityResult&) = delete;
    DensityResult& operator=(const DensityResult&) = delete;
    ~DensityResult() { cudaFree(parameters_); }

    bool Apply(
        DensityController& controller,
        const float* parameters,
        int gaussian_count,
        float threshold = 0.15f,
        float max_screen_size = 0.0f) {
        GaussianBuffers gaussians;
        gaussians.gaussian_count_ = gaussian_count;
        if (gaussian_count > 0) {
            size_t bytes = static_cast<size_t>(gaussian_count) * kMaxParamCount * sizeof(float);
            CheckCuda(cudaMalloc(&gaussians.gaussians_, bytes));
            cudaError_t result = cudaMemcpy(gaussians.gaussians_, parameters, bytes, cudaMemcpyDeviceToDevice);
            if (result != cudaSuccess) {
                cudaFree(gaussians.gaussians_);
                CheckCuda(result);
            }
        }
        bool success = controller.ControlDensity(
            gaussians,
            1.0f,
            threshold,
            0.01f,
            0.005f,
            max_screen_size,
            42,
            changes_);
        cudaFree(parameters_);
        parameters_ = gaussians.gaussians_;
        count_ = gaussians.gaussian_count_;
        parents_ = controller.GetHistoryParentIds();
        return success;
    }

    float* parameters_ = nullptr;
    const int* parents_ = nullptr;
    int count_ = 0;
    int changes_[2] = {};
}; // struct DensityResult

TEST_F(DensityTest, PreservesParametersAndRemovesLowOpacity) {
    auto gaussians = MakeGaussians(3);
    gaussians[1].opacity_logit_ = std::log(0.001f / 0.999f);
    DeviceGaussians original_gaussians(gaussians);
    DensityController controller;
    ASSERT_TRUE(controller.Initialize(3, 3));
    DensityResult density_result;
    ASSERT_TRUE(density_result.Apply(controller, original_gaussians.buffers_.gaussians_, 3));
    ASSERT_EQ(density_result.count_, 2);
    EXPECT_EQ(ReadDevice(density_result.parents_, 2), (std::vector<int>{0, 2}));
    EXPECT_EQ(density_result.changes_[0], 0);
    EXPECT_EQ(density_result.changes_[1], 1);
    auto values = ReadDevice(density_result.parameters_, size_t(2 * kMaxParamCount));
    auto original = PackGaussians(gaussians);
    for (int field = 0; field < kMaxParamCount; ++field) {
        EXPECT_FLOAT_EQ(values[field * 2], original[field * 3]);
        EXPECT_FLOAT_EQ(values[field * 2 + 1], original[field * 3 + 2]);
    }
}

TEST_F(DensityTest, ClonesVisibleGradientAndClearsAccumulatedStatistics) {
    auto gaussians = MakeGaussians(3);
    for (auto& g : gaussians) g.log_scale_.setConstant(std::log(0.005f));
    DeviceGaussians original_gaussians(gaussians);
    DensityController controller;
    ASSERT_TRUE(controller.Initialize(3, 3));
    DeviceBuffer<float> gradients(std::vector<float>{0.12f, 0.16f, 50, 50, 0, 0});
    DeviceBuffer<float> radii(std::vector<float>{1, 0, 1});
    DeviceBuffer<int> ids(std::vector<int>{0, 0, 1});
    ASSERT_TRUE(controller.AccumulateStatistics(gradients.Get(), radii.Get(), ids.Get(), 3));
    DensityResult first;
    ASSERT_TRUE(first.Apply(controller, original_gaussians.buffers_.gaussians_, 3));
    ASSERT_EQ(first.count_, 4);
    EXPECT_EQ(ReadDevice(first.parents_, 4), (std::vector<int>{0, 1, 2, -1}));
    auto values = ReadDevice(first.parameters_, size_t(4 * kMaxParamCount));
    for (int field = 0; field < kMaxParamCount; ++field) EXPECT_FLOAT_EQ(values[field * 4], values[field * 4 + 3]);
    DensityResult second;
    ASSERT_TRUE(second.Apply(controller, first.parameters_, first.count_));
    EXPECT_EQ(second.count_, 4);
    EXPECT_EQ(second.changes_[0], 0);
}

TEST_F(DensityTest, SplitsReproduciblyAndRescalesChildren) {
    auto gaussians = MakeGaussians(1);
    gaussians[0].log_scale_.setConstant(std::log(0.2f));
    DeviceGaussians original_gaussians(gaussians);
    DeviceBuffer<float> statistics(std::vector<float>{1, 1});
    DensityController first_controller, second_controller;
    ASSERT_TRUE(first_controller.Initialize(1, 1));
    ASSERT_TRUE(second_controller.Initialize(1, 1));
    ASSERT_TRUE(first_controller.AccumulateStatistics(statistics.Get(), 1));
    ASSERT_TRUE(second_controller.AccumulateStatistics(statistics.Get(), 1));
    DensityResult first, second;
    ASSERT_TRUE(first.Apply(first_controller, original_gaussians.buffers_.gaussians_, 1));
    ASSERT_TRUE(second.Apply(second_controller, original_gaussians.buffers_.gaussians_, 1));
    ASSERT_EQ(first.count_, 2);
    EXPECT_EQ(ReadDevice(first.parents_, 2), (std::vector<int>{-1, -1}));
    EXPECT_EQ(ReadDevice(first_controller.GetParentIds(), 2), (std::vector<int>{0, 0}));
    auto actual = ReadDevice(first.parameters_, size_t(2 * kMaxParamCount));
    EXPECT_EQ(actual, ReadDevice(second.parameters_, size_t(2 * kMaxParamCount)));
    auto original = PackGaussians(gaussians);
    for (int field = 0; field < kMaxParamCount; ++field) {
        for (int child = 0; child < 2; ++child) {
            float value = actual[field * 2 + child];
            EXPECT_TRUE(std::isfinite(value));
            if (field >= kLogScaleIndex && field < kRotationIndex) {
                EXPECT_NEAR(std::exp(value), 0.125f, 1e-7f);
            }
            else if (field >= kRotationIndex) {
                EXPECT_FLOAT_EQ(value, original[field]);
            }
        }
    }
    EXPECT_GT(std::abs(actual[0] - actual[1]) + std::abs(actual[2] - actual[3]) +
        std::abs(actual[4] - actual[5]), 1e-6f);
}

TEST_F(DensityTest, AveragesRoundStatisticsBeforeSelecting) {
    auto gaussians = MakeGaussians(1);
    gaussians[0].log_scale_.setConstant(std::log(0.005f));
    DeviceGaussians original_gaussians(gaussians);
    DeviceBuffer<float> first(std::vector<float>{0.1f, 1});
    DeviceBuffer<float> second(std::vector<float>{0.3f, 1});
    for (float threshold : {0.19f, 0.21f}) {
        SCOPED_TRACE(::testing::Message() << "threshold=" << threshold);
        DensityController controller;
        ASSERT_TRUE(controller.Initialize(1, 1));
        ASSERT_TRUE(controller.AccumulateStatistics(first.Get(), 1));
        ASSERT_TRUE(controller.AccumulateStatistics(second.Get(), 1));
        DensityResult density_result;
        ASSERT_TRUE(density_result.Apply(controller, original_gaussians.buffers_.gaussians_, 1, threshold));
        EXPECT_EQ(density_result.count_, threshold < 0.2 ? 2 : 1);
        EXPECT_EQ(density_result.changes_[0], threshold < 0.2 ? 1 : 0);
        EXPECT_EQ(density_result.changes_[1], 0);
    }
}

TEST_F(DensityTest, PrunesOversizedGaussiansWhenEnabled) {
    auto gaussians = MakeGaussians(2);
    gaussians[0].log_scale_.setConstant(std::log(0.2f));
    gaussians[1].log_scale_.setConstant(std::log(0.01f));
    DeviceGaussians original_gaussians(gaussians);
    DensityController controller;
    ASSERT_TRUE(controller.Initialize(2, 2));
    DensityResult density_result;
    ASSERT_TRUE(density_result.Apply(controller, original_gaussians.buffers_.gaussians_, 2, 0.15, 20));
    ASSERT_EQ(density_result.count_, 1);
    EXPECT_EQ(ReadDevice(density_result.parents_, 1), (std::vector<int>{1}));
}

TEST_F(DensityTest, ResetsOpacityWithoutChangingOtherFields) {
    auto gaussians = MakeGaussians(3);
    gaussians[1].opacity_logit_ = -9;
    gaussians[2].opacity_logit_ = std::log(0.01f / 0.99f);
    DeviceGaussians original_gaussians(gaussians);
    DensityController controller;
    ASSERT_TRUE(controller.Initialize(3, 3));
    ASSERT_TRUE(controller.ResetOpacity(original_gaussians.buffers_.gaussians_, 3));
    auto actual = original_gaussians.parameters_.Download(), original = PackGaussians(gaussians);
    for (int field = 0; field < kMaxParamCount; ++field) {
        for (int i = 0; i < 3; ++i) {
            float expected = field == kOpacityIndex ? std::min(original[field * 3 + i], std::log(0.01f / 0.99f)) : original[field * 3 + i];
            EXPECT_FLOAT_EQ(actual[field * 3 + i], expected);
        }
    }
}

TEST_F(DensityTest, SupportsAllPrunedAndEmptyInput) {
    auto gaussians = MakeGaussians(2);
    for (auto& g : gaussians) g.opacity_logit_ = -12;
    DeviceGaussians original_gaussians(gaussians);
    DensityController controller;
    ASSERT_TRUE(controller.Initialize(2, 2));
    DensityResult density_result;
    ASSERT_TRUE(density_result.Apply(controller, original_gaussians.buffers_.gaussians_, 2));
    EXPECT_EQ(density_result.count_, 0);
    EXPECT_EQ(density_result.parameters_, nullptr);
    DensityResult empty;
    ASSERT_TRUE(empty.Apply(controller, nullptr, 0));
    EXPECT_EQ(empty.count_, 0);
}
} // namespace gs_native::test
