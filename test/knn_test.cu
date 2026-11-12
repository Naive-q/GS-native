#include "gpu_test.h"

#include "knn/knn.h"

namespace gs_native::test {
using KnnTest = GpuTest;

TEST_F(KnnTest, MatchesExhaustiveSearchForBoundaryAndDegenerateClouds) {
    std::mt19937 generator(17);
    std::uniform_real_distribution<float> random(-10, 10);
    for (int mode = 0; mode < 4; ++mode) {
        for (int count : {0, 1, 2, 3, 4, 17, 256, 1025}) {
            SCOPED_TRACE(::testing::Message() << "mode=" << mode << " count=" << count);
            std::vector<Eigen::Vector3f> points(count);
            for (auto& point : points) {
                point = Eigen::Vector3f(random(generator), random(generator), random(generator));
                if (mode == 1) point.x() = 0;
                if (mode == 2) point.setZero();
                if (mode == 3) { point.y() = 0; point.z() = 0; }
            }
            auto actual = ComputeMeanDist2Cuda(points);
            ASSERT_EQ(actual.size(), points.size());
            for (int i = 0; i < count; ++i) {
                std::vector<double> distances;
                for (int j = 0; j < count; ++j) {
                    if (j != i) distances.push_back((points[i].cast<double>() - points[j].cast<double>()).squaredNorm());
                }
                std::sort(distances.begin(), distances.end());
                int neighbors = std::min(3, int(distances.size()));
                double expected = 0;
                for (int k = 0; k < neighbors; ++k) expected += distances[k];
                if (neighbors) expected /= neighbors;
                if (expected == 0) {
                    EXPECT_EQ(actual[i], 0.0f) << "point=" << i;
                }
                else {
                    EXPECT_NEAR(actual[i], expected, 2e-6 * std::max(1.0, expected)) << "point=" << i;
                }
            }
        }
    }
}

TEST_F(KnnTest, PreservesOriginalPointOrder) {
    std::vector<Eigen::Vector3f> points = {{100, 0, 0}, {0, 0, 0}, {1, 0, 0}, {3, 0, 0}, {7, 0, 0}};
    auto first = ComputeMeanDist2Cuda(points);
    ASSERT_EQ(first.size(), points.size());
    std::reverse(points.begin(), points.end());
    auto second = ComputeMeanDist2Cuda(points);
    ASSERT_EQ(second.size(), points.size());
    std::reverse(second.begin(), second.end());
    EXPECT_EQ(first, second);
}
} // namespace gs_native::test
