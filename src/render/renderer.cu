#include "render/renderer.h"

#include <algorithm>
#include <climits>
#include <cub/cub.cuh>

namespace gs_native {

namespace {

constexpr int kTileSize = 16;
constexpr float kLowPassVariance = 0.3f;
constexpr float kFovClampScale = 1.3f;
constexpr float kMinNearDepth = 1e-6f;
constexpr int kShCoeffCount = 16;

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShC1 = 0.4886025119029199f;
constexpr float kShC2[5] = {1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constexpr float kShC3[7] = {-0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};

__constant__ float c_sh_c2[5] = {kShC2[0], kShC2[1], kShC2[2], kShC2[3], kShC2[4]};
__constant__ float c_sh_c3[7] = {kShC3[0], kShC3[1], kShC3[2], kShC3[3], kShC3[4], kShC3[5], kShC3[6]};

__device__ void GetRotationMatrix(
    float3 rotation,
    float R[9]) {
    float angle = sqrtf(rotation.x * rotation.x + rotation.y * rotation.y + rotation.z * rotation.z);
    if (angle < 1e-8f) {
        R[0] = 1.0f;
        R[1] = 0.0f;
        R[2] = 0.0f;
        R[3] = 0.0f;
        R[4] = 1.0f;
        R[5] = 0.0f;
        R[6] = 0.0f;
        R[7] = 0.0f;
        R[8] = 1.0f;
        return;
    }
    float x = rotation.x / angle;
    float y = rotation.y / angle;
    float z = rotation.z / angle;
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;
    R[0] = c + t * x * x;
    R[1] = t * x * y - s * z;
    R[2] = t * x * z + s * y;
    R[3] = t * x * y + s * z;
    R[4] = c + t * y * y;
    R[5] = t * y * z - s * x;
    R[6] = t * x * z - s * y;
    R[7] = t * y * z + s * x;
    R[8] = c + t * z * z;
}

__device__ void EvaluateShBasis(
    float3 d,
    float basis[kShCoeffCount],
    int sh_degree) {
    basis[0] = kShC0;
    if (sh_degree < 1) {
        return;
    }
    float x = d.x;
    float y = d.y;
    float z = d.z;
    basis[1] = -kShC1 * y;
    basis[2] = kShC1 * z;
    basis[3] = -kShC1 * x;
    if (sh_degree < 2) {
        return;
    }
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    basis[4] = c_sh_c2[0] * xy;
    basis[5] = c_sh_c2[1] * yz;
    basis[6] = c_sh_c2[2] * (2.0f * zz - xx - yy);
    basis[7] = c_sh_c2[3] * xz;
    basis[8] = c_sh_c2[4] * (xx - yy);
    if (sh_degree < 3) {
        return;
    }
    basis[9] = c_sh_c3[0] * y * (3.0f * xx - yy);
    basis[10] = c_sh_c3[1] * xy * z;
    basis[11] = c_sh_c3[2] * y * (4.0f * zz - xx - yy);
    basis[12] = c_sh_c3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    basis[13] = c_sh_c3[4] * x * (4.0f * zz - xx - yy);
    basis[14] = c_sh_c3[5] * z * (xx - yy);
    basis[15] = c_sh_c3[6] * x * (xx - 3.0f * yy);
}

__global__ void ProjectGaussiansKernel(
    const float* gaussians,
    CameraParams camera,
    float2* means,
    float4* conic_opacities,
    float3* colors,
    float* depths,
    int4* tile_rects,
    int* record_offsets,
    int count,
    int sh_degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        record_offsets[count] = 0;
    }
    if (i >= count) {
        return;
    }
    record_offsets[i] = 0;
    float3 position = make_float3(gaussians[kPositionIndex * count + i],
        gaussians[(kPositionIndex + 1) * count + i],
        gaussians[(kPositionIndex + 2) * count + i]);
    float3 log_scale = make_float3(gaussians[kLogScaleIndex * count + i],
        gaussians[(kLogScaleIndex + 1) * count + i],
        gaussians[(kLogScaleIndex + 2) * count + i]);
    float3 rotation = make_float3(gaussians[kRotationIndex * count + i],
        gaussians[(kRotationIndex + 1) * count + i],
        gaussians[(kRotationIndex + 2) * count + i]);
    const float* R_c_w = camera.R_c_w_;
    const float* t_c_w = camera.t_c_w_;
    float px = R_c_w[0] * position.x + R_c_w[1] * position.y + R_c_w[2] * position.z + t_c_w[0];
    float py = R_c_w[3] * position.x + R_c_w[4] * position.y + R_c_w[5] * position.z + t_c_w[1];
    float z = R_c_w[6] * position.x + R_c_w[7] * position.y + R_c_w[8] * position.z + t_c_w[2];

    float near_depth = fmaxf(camera.near_depth_, kMinNearDepth);
    if (z <= near_depth || z > fmaxf(camera.far_depth_, near_depth)) {
        return;
    }

    float R_g[9];
    GetRotationMatrix(rotation, R_g);
    float scale[3] = {expf(log_scale.x), expf(log_scale.y), expf(log_scale.z)};
    float M[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M[r * 3 + c] =
                (R_c_w[r * 3] * R_g[c] + R_c_w[r * 3 + 1] * R_g[3 + c] + R_c_w[r * 3 + 2] * R_g[6 + c]) * scale[c];
        }
    }
    float cov_3d[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            cov_3d[r * 3 + c] = M[r * 3] * M[c * 3] + M[r * 3 + 1] * M[c * 3 + 1] + M[r * 3 + 2] * M[c * 3 + 2];
        }
    }

    float limit_x = kFovClampScale * 0.5f * camera.width_ / camera.fx_;
    float limit_y = kFovClampScale * 0.5f * camera.height_ / camera.fy_;
    float x = fminf(fmaxf(px / z, -limit_x), limit_x) * z;
    float y = fminf(fmaxf(py / z, -limit_y), limit_y) * z;
    float J[6] = {camera.fx_ / z, 0.0f, -camera.fx_ * x / (z * z), 0.0f, camera.fy_ / z, -camera.fy_ * y / (z * z)};
    float JS[6];
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            JS[r * 3 + c] = J[r * 3] * cov_3d[c] + J[r * 3 + 1] * cov_3d[3 + c] + J[r * 3 + 2] * cov_3d[6 + c];
        }
    }
    float a = JS[0] * J[0] + JS[1] * J[1] + JS[2] * J[2] + kLowPassVariance;
    float b = JS[0] * J[3] + JS[1] * J[4] + JS[2] * J[5];
    float c = JS[3] * J[3] + JS[4] * J[4] + JS[5] * J[5] + kLowPassVariance;
    float det = a * c - b * b;
    if (!(det > 0.0f)) {
        return;
    }
    float middle = 0.5f * (a + c);
    float radius = ceilf(3.0f * sqrtf(middle + sqrtf(fmaxf(0.1f, middle * middle - det))));
    float u = camera.fx_ * px / z + camera.cx_;
    float v = camera.fy_ * py / z + camera.cy_;
    if (!isfinite(radius) || !isfinite(u) || !isfinite(v)) {
        return;
    }
    int4 rect;
    rect.x = static_cast<int>(fminf(camera.tile_count_x_, fmaxf(0.0f, (u - radius) / kTileSize)));
    rect.y = static_cast<int>(fminf(camera.tile_count_y_, fmaxf(0.0f, (v - radius) / kTileSize)));
    rect.z = static_cast<int>(fminf(camera.tile_count_x_, fmaxf(0.0f, (u + radius + kTileSize - 1) / kTileSize)));
    rect.w = static_cast<int>(fminf(camera.tile_count_y_, fmaxf(0.0f, (v + radius + kTileSize - 1) / kTileSize)));
    if (rect.x >= rect.z || rect.y >= rect.w) {
        return;
    }
    depths[i] = z;
    tile_rects[i] = rect;
    record_offsets[i] = (rect.z - rect.x) * (rect.w - rect.y);
    float det_inv = 1.0f / det;
    float opacity = 1.0f / (1.0f + expf(-gaussians[kOpacityIndex * count + i]));
    means[i] = make_float2(camera.fx_ * px / z + camera.cx_, camera.fy_ * py / z + camera.cy_);
    conic_opacities[i] = make_float4(c * det_inv, -b * det_inv, a * det_inv, opacity);

    float3 d = make_float3(position.x + R_c_w[0] * t_c_w[0] + R_c_w[3] * t_c_w[1] + R_c_w[6] * t_c_w[2],
        position.y + R_c_w[1] * t_c_w[0] + R_c_w[4] * t_c_w[1] + R_c_w[7] * t_c_w[2],
        position.z + R_c_w[2] * t_c_w[0] + R_c_w[5] * t_c_w[1] + R_c_w[8] * t_c_w[2]);
    float length_inv = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    float basis[kShCoeffCount];
    EvaluateShBasis(make_float3(d.x * length_inv, d.y * length_inv, d.z * length_inv), basis, sh_degree);
    int coeff_count = (sh_degree + 1) * (sh_degree + 1);
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
        float value = 0.0f;
        for (int j = 0; j < coeff_count; ++j) {
            value += basis[j] * gaussians[(kShIndex + c * kShCoeffCount + j) * count + i];
        }
        rgb[c] = fmaxf(value + 0.5f, 0.0f);
    }
    colors[i] = make_float3(rgb[0], rgb[1], rgb[2]);
}

__global__ void WriteTileRecordsKernel(
    const int4* tile_rects,
    const float* depths,
    const int* record_offsets,
    uint64_t* keys,
    int* gaussian_ids,
    int count,
    int tile_count_x) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count || record_offsets[id] == record_offsets[id + 1]) {
        return;
    }
    int4 rect = tile_rects[id];
    int record = record_offsets[id];
    uint64_t depth = __float_as_uint(depths[id]);
    for (int y = rect.y; y < rect.w; ++y) {
        for (int x = rect.x; x < rect.z; ++x) {
            uint64_t tile = static_cast<uint64_t>(y) * tile_count_x + x;
            keys[record] = (tile << 32) | depth;
            gaussian_ids[record++] = id;
        }
    }
}

__global__ void IdentifyTileOffsetsKernel(
    const uint64_t* keys,
    int64_t* offsets,
    float* depths,
    int record_count,
    int tile_count) {
    int record = blockIdx.x * blockDim.x + threadIdx.x;
    if (record >= record_count) {
        return;
    }
    uint64_t key = keys[record];
    int tile = static_cast<int>(key >> 32);
    depths[record] = __uint_as_float(static_cast<unsigned int>(key));
    int previous = record == 0 ? -1 : static_cast<int>(keys[record - 1] >> 32);
    if (tile != previous) {
        for (int empty = previous + 1; empty <= tile; ++empty) {
            offsets[empty] = record;
        }
    }
    if (record == record_count - 1) {
        for (int empty = tile + 1; empty <= tile_count; ++empty) {
            offsets[empty] = record_count;
        }
    }
}

__global__ void RenderTilesKernel(
    const int* gaussian_ids,
    const int64_t* gaussian_offsets,
    const float2* means,
    const float4* conic_opacities,
    const float3* colors,
    CameraParams camera,
    float* image,
    float3 background) {
    __shared__ float2 shared_means[256];
    __shared__ float4 shared_conic_opacities[256];
    __shared__ float3 shared_colors[256];
    int t = threadIdx.x;
    int tile = blockIdx.x;
    int x = tile % camera.tile_count_x_ * kTileSize + t % kTileSize;
    int y = tile / camera.tile_count_x_ * kTileSize + t / kTileSize;
    bool inside = x < camera.width_ && y < camera.height_;
    bool done = !inside;
    int64_t begin = gaussian_offsets[tile];
    int64_t end = gaussian_offsets[tile + 1];
    float transmittance = 1.0f;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    for (int64_t base = begin; base < end; base += 256) {
        if (__syncthreads_count(done) == 256) {
            break;
        }
        if (base + t < end) {
            int id = gaussian_ids[base + t];
            shared_means[t] = means[id];
            shared_conic_opacities[t] = conic_opacities[id];
            shared_colors[t] = colors[id];
        }
        __syncthreads();
        int batch = static_cast<int>(min(static_cast<int64_t>(256), end - base));
        for (int j = 0; !done && j < batch; ++j) {
            float dx = shared_means[j].x - x;
            float dy = shared_means[j].y - y;
            float4 conic = shared_conic_opacities[j];
            float power = -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
            if (power > 0.0f) {
                continue;
            }
            float alpha = fminf(0.99f, conic.w * expf(power));
            if (alpha < 1.0f / 255.0f) {
                continue;
            }
            float next_transmittance = transmittance * (1.0f - alpha);
            if (next_transmittance < 1e-4f) {
                done = true;
                continue;
            }
            float weight = alpha * transmittance;
            color.x += shared_colors[j].x * weight;
            color.y += shared_colors[j].y * weight;
            color.z += shared_colors[j].z * weight;
            transmittance = next_transmittance;
        }
        __syncthreads();
    }
    if (inside) {
        float* pixel = image + (static_cast<int64_t>(y) * camera.width_ + x) * 3;
        pixel[0] = color.x + transmittance * background.x;
        pixel[1] = color.y + transmittance * background.y;
        pixel[2] = color.z + transmittance * background.z;
    }
}

}

Renderer::~Renderer() {
    cudaFree(means_);
    cudaFree(conic_opacities_);
    cudaFree(colors_);
    cudaFree(depths_);
    cudaFree(tile_rects_);
    cudaFree(record_offsets_);
    cudaFree(tile_lists_.gaussian_ids_);
    cudaFree(tile_lists_.gaussian_depths_);
    cudaFree(tile_lists_.gaussian_counts_);
    cudaFree(tile_lists_.gaussian_offsets_);
    cudaFree(keys_);
    cudaFree(sorted_keys_);
    cudaFree(gaussian_ids_);
    cudaFree(temp_storage_);
}

bool Renderer::Render(
    const GaussianBuffers& gaussians,
    const CameraParams& camera,
    const float* background,
    float* image,
    int sh_degree) {
    if (!ProjectGaussians(gaussians, camera, sh_degree)) {
        return false;
    }
    if (!BuildTileLists(gaussians, camera)) {
        return false;
    }
    return RenderTiles(camera, background, image);
}

bool Renderer::ProjectGaussians(
    const GaussianBuffers& gaussians,
    const CameraParams& camera,
    int sh_degree) {
    int count = gaussians.gaussian_count_;
    if (count < 0 || (count > 0 && !gaussians.gaussians_)) {
        return false;
    }
    sh_degree = std::clamp(sh_degree, 0, 3);
    if (count > gaussian_capacity_ || !record_offsets_) {
        cudaFree(means_);
        cudaFree(conic_opacities_);
        cudaFree(colors_);
        cudaFree(depths_);
        cudaFree(tile_rects_);
        cudaFree(record_offsets_);
        means_ = nullptr;
        conic_opacities_ = nullptr;
        colors_ = nullptr;
        depths_ = nullptr;
        tile_rects_ = nullptr;
        record_offsets_ = nullptr;
        gaussian_capacity_ = 0;
        cudaError_t result = cudaMalloc(&record_offsets_, (static_cast<size_t>(count) + 1) * sizeof(int));
        if (count > 0 && result == cudaSuccess) {
            result = cudaMalloc(&means_, static_cast<size_t>(count) * sizeof(float2));
        }
        if (count > 0 && result == cudaSuccess) {
            result = cudaMalloc(&conic_opacities_, static_cast<size_t>(count) * sizeof(float4));
        }
        if (count > 0 && result == cudaSuccess) {
            result = cudaMalloc(&colors_, static_cast<size_t>(count) * sizeof(float3));
        }
        if (count > 0 && result == cudaSuccess) {
            result = cudaMalloc(&depths_, static_cast<size_t>(count) * sizeof(float));
        }
        if (count > 0 && result == cudaSuccess) {
            result = cudaMalloc(&tile_rects_, static_cast<size_t>(count) * sizeof(int4));
        }
        if (result != cudaSuccess) {
            cudaFree(means_);
            cudaFree(conic_opacities_);
            cudaFree(colors_);
            cudaFree(depths_);
            cudaFree(tile_rects_);
            cudaFree(record_offsets_);
            means_ = nullptr;
            conic_opacities_ = nullptr;
            colors_ = nullptr;
            depths_ = nullptr;
            tile_rects_ = nullptr;
            record_offsets_ = nullptr;
            return false;
        }
        gaussian_capacity_ = count;
    }
    if (count == 0) {
        return cudaMemsetAsync(record_offsets_, 0, sizeof(int)) == cudaSuccess;
    }
    ProjectGaussiansKernel<<<(count + 255ll) / 256, 256>>>(gaussians.gaussians_,
        camera,
        means_,
        conic_opacities_,
        colors_,
        depths_,
        tile_rects_,
        record_offsets_,
        count,
        sh_degree);
    return cudaGetLastError() == cudaSuccess;
}

bool Renderer::BuildTileLists(
    const GaussianBuffers& gaussians,
    const CameraParams& camera) {
    int count = gaussians.gaussian_count_;
    int64_t tile_count_long = static_cast<int64_t>(camera.tile_count_x_) * camera.tile_count_y_;
    if (count < 0 || count == INT_MAX || count > gaussian_capacity_ || !record_offsets_ ||
        camera.tile_count_x_ <= 0 || camera.tile_count_y_ <= 0 || tile_count_long >= INT_MAX) {
        return false;
    }
    int tile_count = static_cast<int>(tile_count_long);
    if (tile_count > tile_capacity_) {
        cudaFree(tile_lists_.gaussian_offsets_);
        tile_lists_.gaussian_offsets_ = nullptr;
        tile_capacity_ = 0;
        if (cudaMalloc(&tile_lists_.gaussian_offsets_, (static_cast<size_t>(tile_count) + 1) * sizeof(int64_t)) != cudaSuccess) {
            return false;
        }
        tile_capacity_ = tile_count;
    }
    size_t scan_bytes = 0;
    cudaError_t result = cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        record_offsets_, record_offsets_, count + 1);
    if (result != cudaSuccess) {
        return false;
    }
    if (scan_bytes > temp_storage_bytes_) {
        cudaFree(temp_storage_);
        temp_storage_ = nullptr;
        temp_storage_bytes_ = 0;
        if (cudaMalloc(&temp_storage_, scan_bytes) != cudaSuccess) {
            return false;
        }
        temp_storage_bytes_ = scan_bytes;
    }
    if (cub::DeviceScan::ExclusiveSum(temp_storage_, scan_bytes,
            record_offsets_, record_offsets_, count + 1) != cudaSuccess) {
        return false;
    }
    int records = 0;
    if (cudaMemcpy(&records, record_offsets_ + count, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || records < 0) {
        return false;
    }
    if (records == 0) {
        return cudaMemsetAsync(tile_lists_.gaussian_offsets_, 0,
            (static_cast<size_t>(tile_count) + 1) * sizeof(int64_t)) == cudaSuccess;
    }
    if (records > record_capacity_) {
        cudaFree(keys_);
        cudaFree(sorted_keys_);
        cudaFree(gaussian_ids_);
        cudaFree(tile_lists_.gaussian_ids_);
        cudaFree(tile_lists_.gaussian_depths_);
        keys_ = nullptr;
        sorted_keys_ = nullptr;
        gaussian_ids_ = nullptr;
        tile_lists_.gaussian_ids_ = nullptr;
        tile_lists_.gaussian_depths_ = nullptr;
        record_capacity_ = 0;
        tile_lists_.capacity_ = 0;
        result = cudaMalloc(&keys_, static_cast<size_t>(records) * sizeof(uint64_t));
        if (result == cudaSuccess) {
            result = cudaMalloc(&sorted_keys_, static_cast<size_t>(records) * sizeof(uint64_t));
        }
        if (result == cudaSuccess) {
            result = cudaMalloc(&gaussian_ids_, static_cast<size_t>(records) * sizeof(int));
        }
        if (result == cudaSuccess) {
            result = cudaMalloc(&tile_lists_.gaussian_ids_, static_cast<size_t>(records) * sizeof(int));
        }
        if (result == cudaSuccess) {
            result = cudaMalloc(&tile_lists_.gaussian_depths_, static_cast<size_t>(records) * sizeof(float));
        }
        if (result != cudaSuccess) {
            cudaFree(keys_);
            cudaFree(sorted_keys_);
            cudaFree(gaussian_ids_);
            cudaFree(tile_lists_.gaussian_ids_);
            cudaFree(tile_lists_.gaussian_depths_);
            keys_ = nullptr;
            sorted_keys_ = nullptr;
            gaussian_ids_ = nullptr;
            tile_lists_.gaussian_ids_ = nullptr;
            tile_lists_.gaussian_depths_ = nullptr;
            return false;
        }
        record_capacity_ = records;
        tile_lists_.capacity_ = records;
    }
    WriteTileRecordsKernel<<<(count + 255ll) / 256, 256>>>(tile_rects_,
        depths_,
        record_offsets_,
        keys_,
        gaussian_ids_,
        count,
        camera.tile_count_x_);
    int end_bit = 32;
    for (unsigned int tile = static_cast<unsigned int>(tile_count - 1); tile; tile >>= 1) {
        ++end_bit;
    }
    size_t sort_bytes = 0;
    result = cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys_, sorted_keys_,
        gaussian_ids_, tile_lists_.gaussian_ids_, records, 0, end_bit);
    if (result != cudaSuccess) {
        return false;
    }
    if (sort_bytes > temp_storage_bytes_) {
        cudaFree(temp_storage_);
        temp_storage_ = nullptr;
        temp_storage_bytes_ = 0;
        if (cudaMalloc(&temp_storage_, sort_bytes) != cudaSuccess) {
            return false;
        }
        temp_storage_bytes_ = sort_bytes;
    }
    if (cub::DeviceRadixSort::SortPairs(temp_storage_, sort_bytes, keys_, sorted_keys_,
            gaussian_ids_, tile_lists_.gaussian_ids_, records, 0, end_bit) != cudaSuccess) {
        return false;
    }
    IdentifyTileOffsetsKernel<<<(records + 255ll) / 256, 256>>>(sorted_keys_,
        tile_lists_.gaussian_offsets_,
        tile_lists_.gaussian_depths_,
        records,
        tile_count);
    return cudaGetLastError() == cudaSuccess;
}

bool Renderer::RenderTiles(
    const CameraParams& camera,
    const float* background,
    float* image) {
    int64_t tile_count = static_cast<int64_t>(camera.tile_count_x_) * camera.tile_count_y_;
    if (!background || !image || !tile_lists_.gaussian_offsets_ || camera.width_ <= 0 || camera.height_ <= 0 ||
        camera.tile_count_x_ <= 0 || camera.tile_count_y_ <= 0 || tile_count > tile_capacity_) {
        return false;
    }
    RenderTilesKernel<<<static_cast<unsigned int>(tile_count), 256>>>(tile_lists_.gaussian_ids_,
        tile_lists_.gaussian_offsets_,
        means_,
        conic_opacities_,
        colors_,
        camera,
        image,
        make_float3(background[0], background[1], background[2]));
    return cudaGetLastError() == cudaSuccess;
}

}
