#include "benchmark_support.h"

#include "knn/knn.h"

#include <fstream>
#include <numeric>

int main(int argc, char** argv) {
    using namespace gs_native;
    using namespace gs_native::benchmark;
    try {
        Options options;
        if (!ReadOptions(argv, options, argc)) return 0;
        std::vector<Eigen::Vector3f> points;
        if (options.points_.empty()) {
            std::mt19937 generator(17);
            std::uniform_real_distribution<float> random(-10, 10);
            for (int i = 0; i < options.count_; ++i) points.emplace_back(random(generator), random(generator), random(generator));
        }
        else {
            std::ifstream file(options.points_, std::ios::binary);
            uint64_t count = 0;
            file.read(reinterpret_cast<char*>(&count), 8);
            if (!file) throw std::runtime_error("Cannot read point cloud header");
            for (uint64_t i = 0; i < count; ++i) {
                double xyz[3]; uint64_t tracks;
                file.seekg(8, std::ios::cur);
                file.read(reinterpret_cast<char*>(xyz), 24);
                file.seekg(11, std::ios::cur);
                file.read(reinterpret_cast<char*>(&tracks), 8);
                file.seekg(tracks * 8, std::ios::cur);
                if (!file) throw std::runtime_error("Truncated point cloud");
                points.emplace_back(float(xyz[0]), float(xyz[1]), float(xyz[2]));
            }
        }
        auto values = ComputeMeanDist2Cuda(points);
        if (values.size() != points.size()) throw std::runtime_error("KNN result size mismatch");
        CheckFinite(values);
        for (size_t i = 0; i < std::min<size_t>(32, points.size()); ++i) {
            std::vector<double> distances;
            for (size_t j = 0; j < points.size(); ++j) {
                if (i != j) distances.push_back((points[i].cast<double>() - points[j].cast<double>()).squaredNorm());
            }
            std::sort(distances.begin(), distances.end());
            size_t count = std::min<size_t>(3, distances.size());
            double expected = count ? std::accumulate(distances.begin(), distances.begin() + count, 0.0) / count : 0;
            if (!std::isfinite(expected)) throw std::runtime_error("Non-finite KNN reference");
            if (std::abs(values[i] - expected) > 2e-6 * std::max(1.0, expected)) throw std::runtime_error("KNN reference mismatch");
        }
        std::cout << "scope=complete_KNN_with_allocation_and_transfers points=" << points.size() << '\n';
        Measure("knn", options, [] {}, [&] {
            values = ComputeMeanDist2Cuda(points);
            return values.size() == points.size();
        });
        CheckFinite(values);
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
