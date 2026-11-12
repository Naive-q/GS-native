#include "knn/knn.h"

#include "util/log.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cfloat>
#include <exception>
#include <algorithm>
#include <cstdint>

namespace gs_native {
namespace {

__global__ void PreparePoints(
    const float3* points,
    uint64_t* keys,
    int* ids,
    float3 minimum,
    float3 inverse_extent,
    int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) {
        return;
    }
    float3 point = points[id];
    unsigned int x = static_cast<unsigned int>(fminf(fmaxf((point.x - minimum.x) * inverse_extent.x, 0.0f), 2097151.0f));
    unsigned int y = static_cast<unsigned int>(fminf(fmaxf((point.y - minimum.y) * inverse_extent.y, 0.0f), 2097151.0f));
    unsigned int z = static_cast<unsigned int>(fminf(fmaxf((point.z - minimum.z) * inverse_extent.z, 0.0f), 2097151.0f));
    uint64_t key = 0;
#pragma unroll
    for (int bit = 0; bit < 21; ++bit) {
        key |= static_cast<uint64_t>((x >> bit) & 1u) << (3 * bit);
        key |= static_cast<uint64_t>((y >> bit) & 1u) << (3 * bit + 1);
        key |= static_cast<uint64_t>((z >> bit) & 1u) << (3 * bit + 2);
    }
    keys[id] = key;
    ids[id] = id;
}

__global__ void BuildLeaves(
    const float3* points,
    const int* ids,
    float3* lower,
    float3* upper,
    int count,
    int leaves) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < leaves) {
        lower[leaves + id] = id < count ? points[ids[id]] : make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        upper[leaves + id] = id < count ? points[ids[id]] : make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }
}

__global__ void BuildLevel(
    float3* lower,
    float3* upper,
    int begin) {
    int node = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= 2 * begin) {
        return;
    }
    float3 a = lower[2 * node];
    float3 b = lower[2 * node + 1];
    lower[node] = make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    a = upper[2 * node];
    b = upper[2 * node + 1];
    upper[node] = make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ float GetLowerDistance(
    float3 point,
    float3 lower,
    float3 upper) {
    float x = fmaxf(fmaxf(lower.x - point.x, point.x - upper.x), 0.0f);
    float y = fmaxf(fmaxf(lower.y - point.y, point.y - upper.y), 0.0f);
    float z = fmaxf(fmaxf(lower.z - point.z, point.z - upper.z), 0.0f);
    return x * x + y * y + z * z;
}

__global__ void QueryNeighbors(
    const float3* points,
    const int* ids,
    const float3* lower,
    const float3* upper,
    float* result,
    int count,
    int leaves) {
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= count) {
        return;
    }
    float3 point = points[ids[rank]];
    float best0 = FLT_MAX;
    float best1 = FLT_MAX;
    float best2 = FLT_MAX;
    int seed_count = min(count, 4);
    int seed_begin = max(0, min(rank - 1, count - seed_count));
    for (int seed = seed_begin; seed < seed_begin + seed_count; ++seed) {
        if (seed == rank) {
            continue;
        }
        float3 other = points[ids[seed]];
        float x = point.x - other.x;
        float y = point.y - other.y;
        float z = point.z - other.z;
        float distance = x * x + y * y + z * z;
        if (distance < best0) {
            best2 = best1;
            best1 = best0;
            best0 = distance;
        }
        else if (distance < best1) {
            best2 = best1;
            best1 = distance;
        }
        else if (distance < best2) {
            best2 = distance;
        }
    }
    int stack[32];
    int size = 0;
    int node = 1;
    while (true) {
        if (GetLowerDistance(point, lower[node], upper[node]) <= best2) {
            if (node >= leaves) {
                int other_rank = node - leaves;
                if (other_rank < count && (other_rank < seed_begin || other_rank >= seed_begin + seed_count)) {
                    float3 other = points[ids[other_rank]];
                    float x = point.x - other.x;
                    float y = point.y - other.y;
                    float z = point.z - other.z;
                    float distance = x * x + y * y + z * z;
                    if (distance < best0) {
                        best2 = best1;
                        best1 = best0;
                        best0 = distance;
                    }
                    else if (distance < best1) {
                        best2 = best1;
                        best1 = distance;
                    }
                    else if (distance < best2) {
                        best2 = distance;
                    }
                }
            }
            else {
                int left = 2 * node;
                int right = left + 1;
                float dl = GetLowerDistance(point, lower[left], upper[left]);
                float dr = GetLowerDistance(point, lower[right], upper[right]);
                if (fmaxf(dl, dr) <= best2) {
                    stack[size++] = dl <= dr ? right : left;
                }
                node = dl <= dr ? left : right;
                continue;
            }
        }
        if (size == 0) {
            break;
        }
        node = stack[--size];
    }
    if (count >= 4) {
        result[ids[rank]] = (best0 + best1 + best2) / 3.0f;
    }
    else {
        float sum = best0;
        if (count == 3) {
            sum += best1;
        }
        result[ids[rank]] = sum / (count - 1);
    }
}

}

std::vector<float> ComputeMeanDist2Cuda(const std::vector<Eigen::Vector3f>& points) {
    int count = static_cast<int>(points.size());
    if (count < 2) {
        return std::vector<float>(points.size(), 0.0f);
    }
    Eigen::Vector3f minimum = points[0];
    Eigen::Vector3f maximum = points[0];
    for (const Eigen::Vector3f& point : points) {
        minimum = minimum.cwiseMin(point);
        maximum = maximum.cwiseMax(point);
    }
    Eigen::Vector3f extent = maximum - minimum;
    float3 origin = make_float3(minimum.x(), minimum.y(), minimum.z());
    float3 inverse_extent = make_float3(
        extent.x() > 0.0f ? 2097151.0f / extent.x() : 0.0f,
        extent.y() > 0.0f ? 2097151.0f / extent.y() : 0.0f,
        extent.z() > 0.0f ? 2097151.0f / extent.z() : 0.0f);
    int leaves = 1;
    while (leaves < count) {
        leaves *= 2;
    }
    std::vector<float> result(count, 0.0f);
    float3* device_points = nullptr;
    float3* lower = nullptr;
    float3* upper = nullptr;
    uint64_t* keys = nullptr;
    float* distances = nullptr;
    int* ids = nullptr;
    cudaError_t error = cudaMalloc(&device_points, sizeof(float3) * count);
    if (error == cudaSuccess) {
        error = cudaMalloc(&lower, sizeof(float3) * 2 * leaves);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(&upper, sizeof(float3) * 2 * leaves);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(&keys, sizeof(uint64_t) * count);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(&distances, sizeof(float) * count);
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(&ids, sizeof(int) * count);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpy(device_points, points.data(), sizeof(float3) * count, cudaMemcpyHostToDevice);
    }
    if (error == cudaSuccess) {
        try {
            PreparePoints<<<(count + 255) / 256, 256>>>(device_points, keys, ids, origin, inverse_extent, count);
            thrust::sort_by_key(thrust::device_pointer_cast(keys), thrust::device_pointer_cast(keys + count),
                thrust::device_pointer_cast(ids));
            BuildLeaves<<<(leaves + 255) / 256, 256>>>(device_points, ids, lower, upper, count, leaves);
            for (int begin = leaves / 2; begin > 0; begin /= 2) {
                BuildLevel<<<(begin + 255) / 256, 256>>>(lower, upper, begin);
            }
            QueryNeighbors<<<(count + 127) / 128, 128>>>(device_points, ids, lower, upper, distances, count, leaves);
            error = cudaMemcpy(result.data(), distances, sizeof(float) * count, cudaMemcpyDeviceToHost);
        }
        catch (const std::exception& exception) {
            GS_NATIVE_ERROR("KNN failed: {}", exception.what());
            error = cudaErrorUnknown;
        }
    }
    cudaFree(ids);
    cudaFree(distances);
    cudaFree(keys);
    cudaFree(upper);
    cudaFree(lower);
    cudaFree(device_points);
    if (error != cudaSuccess) {
        GS_NATIVE_ERROR("KNN failed: {}", cudaGetErrorString(error));
        return {};
    }
    return result;
}

}
