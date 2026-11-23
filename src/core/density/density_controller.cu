#include "core/density/density_controller.h"

#include "core/types/buffers.h"
#include "util/cuda_util.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <climits>
#include <cmath>

namespace gs_native {

namespace {

struct AddDensityCounts {
    __host__ __device__ int4 operator()(
        int4 first,
        int4 second) const {
        return make_int4(first.x + second.x, first.y + second.y, first.z + second.z, first.w + second.w);
    }
}; // struct AddDensityCounts

__global__ void InitializeDensityKernel(
    int* control,
    int gaussian_count) {
    for (int i = 0; i < 8; ++i) {
        control[i] = 0;
    }
    control[0] = gaussian_count;
    control[1] = gaussian_count;
    unsigned long long* offsets = reinterpret_cast<unsigned long long*>(control + 8);
    offsets[0] = 0;
    offsets[1] = 0;
}

__global__ void AccumulateDensityStatisticsKernel(
    const float* gradients,
    const float* radii,
    const int* gaussian_ids,
    const int* control,
    float* gradient_sums,
    unsigned int* observation_counts,
    int observation_count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= observation_count || !(radii[index] > 0)) {
        return;
    }
    int id = gaussian_ids ? gaussian_ids[index] : index;
    if (id < 0 || id >= control[0]) {
        return;
    }
    float x = gradients[2ll * index], y = gradients[2ll * index + 1];
    atomicAdd(gradient_sums + id, sqrtf(x * x + y * y));
    atomicAdd(observation_counts + id, 1u);
}

__global__ void CountDensityChangesKernel(
    const float* parameters,
    const float* gradient_sums,
    const float* absolute_gradient_sums,
    const unsigned int* observation_counts,
    const int* control,
    int4* density_counts,
    int gaussian_count,
    float gradient_threshold,
    float split_threshold,
    float min_opacity,
    float max_world_size,
    bool prune_large,
    float absolute_gradient_threshold) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id > gaussian_count) {
        return;
    }
    if (id == gaussian_count || id >= control[0]) {
        density_counts[id] = make_int4(0, 0, 0, 0);
        return;
    }
    float opacity = 1.0f / (1.0f + expf(-parameters[static_cast<int64_t>(kOpacityIndex) * gaussian_count + id]));
    float gradient = observation_counts[id] ? gradient_sums[id] / observation_counts[id] : 0;
    if (isnan(gradient)) {
        gradient = 0;
    }
    float scale[3];
    for (int axis = 0; axis < 3; ++axis) {
        scale[axis] = expf(parameters[static_cast<int64_t>(kLogScaleIndex + axis) * gaussian_count + id]);
    }
    float max_scale = fmaxf(scale[0], fmaxf(scale[1], scale[2]));
    if (isnan(scale[0]) || isnan(scale[1]) || isnan(scale[2])) {
        max_scale = nanf("");
    }
    bool selected = gradient >= gradient_threshold;
    bool clone = selected && max_scale <= split_threshold;
    float absolute_gradient = observation_counts[id] ? absolute_gradient_sums[id] / observation_counts[id] : 0;
    bool split_selected = absolute_gradient_threshold > 0.0f ?
        absolute_gradient >= absolute_gradient_threshold :
        selected;
    bool split = split_selected && max_scale > split_threshold;

    float remaining_max_scale = max_scale;
    if (split && prune_large) {
        float x = expf(logf(scale[0] / 1.6f));
        float y = expf(logf(scale[1] / 1.6f));
        float z = expf(logf(scale[2] / 1.6f));
        remaining_max_scale = fmaxf(x, fmaxf(y, z));
        if (isnan(x) || isnan(y) || isnan(z)) {
            remaining_max_scale = nanf("");
        }
    }
    bool remove = opacity < min_opacity || (prune_large && remaining_max_scale > max_world_size);

    density_counts[id] = make_int4(!split && !remove, clone && !remove, split, split && !remove);
}

__global__ void PrepareDensityControlKernel(
    const int4* density_offsets,
    int* control,
    int gaussian_count,
    int random_blocks) {
    int4 totals = density_offsets[gaussian_count];
    control[7] = control[0];
    control[1] = totals.x + totals.y + 2 * totals.w;
    control[2] = 0;
    control[3] = totals.z;
    control[4] = totals.x;
    control[5] = totals.y;
    control[6] = totals.w;
    unsigned long long* offsets = reinterpret_cast<unsigned long long*>(control + 8);
    unsigned long long numel = 6ull * totals.z;
    unsigned long long threads =
        min(static_cast<unsigned long long>(random_blocks), (numel + kBlockSize - 1) / kBlockSize) * kBlockSize;
    offsets[1] = offsets[0] + (numel ? ((numel - 1) / (4ull * threads) + 1) * 4ull : 0ull);
}

__global__ void GenerateDensitySamplesKernel(
    const int* control,
    float* samples,
    int random_blocks,
    unsigned long long seed) {
    int64_t numel = 6ll * control[3];
    if (numel == 0) {
        return;
    }
    int blocks = static_cast<int>(min(static_cast<int64_t>(random_blocks), (numel + kBlockSize - 1) / kBlockSize));
    if (blockIdx.x >= blocks) {
        return;
    }
    int64_t threads = static_cast<int64_t>(blocks) * kBlockSize;
    int64_t index = static_cast<int64_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    int64_t rounded = ((numel - 1) / (4 * threads) + 1) * 4 * threads;
    const unsigned long long* offsets = reinterpret_cast<const unsigned long long*>(control + 8);
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(index), offsets[0], &state);
    for (int64_t base = index; base < rounded; base += 4 * threads) {
        float4 values = curand_normal4(&state);
        for (int component = 0; component < 4; ++component) {
            int64_t destination = base + component * threads;
            if (destination < numel) {
                samples[destination] = (&values.x)[component];
            }
        }
    }
}

__global__ void WriteDensityGaussiansKernel(
    const float* parameters,
    const int4* density_counts,
    const int4* density_offsets,
    const int* control,
    const float* samples,
    float* gaussian_parameters,
    int* history_parent_ids,
    int gaussian_count,
    int next_gaussian_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= control[0]) {
        return;
    }
    int4 counts = density_counts[id], offsets = density_offsets[id];
    bool split = counts.w != 0;
    if (!counts.x && !counts.y && !split) {
        return;
    }
    float scale[3], rotation[9];
    if (split) {
        for (int axis = 0; axis < 3; ++axis) {
            scale[axis] = expf(parameters[static_cast<int64_t>(kLogScaleIndex + axis) * gaussian_count + id]);
        }
        float x = parameters[static_cast<int64_t>(kRotationIndex) * gaussian_count + id];
        float y = parameters[static_cast<int64_t>(kRotationIndex + 1) * gaussian_count + id];
        float z = parameters[static_cast<int64_t>(kRotationIndex + 2) * gaussian_count + id];
        float angle = sqrtf(x * x + y * y + z * z);
        float a = 1.0f, b = 0.5f;
        if (angle > 1e-4f) {
            a = sinf(angle) / angle;
            float half_sinc = sinf(angle * 0.5f) / (angle * 0.5f);
            b = half_sinc * half_sinc * 0.5f;
        }
        else {
            float square = angle * angle;
            a -= square / 6.0f;
            b -= square / 24.0f;
        }
        rotation[0] = 1 - b * (y * y + z * z);
        rotation[1] = b * x * y - a * z;
        rotation[2] = b * x * z + a * y;
        rotation[3] = b * x * y + a * z;
        rotation[4] = 1 - b * (x * x + z * z);
        rotation[5] = b * y * z - a * x;
        rotation[6] = b * x * z - a * y;
        rotation[7] = b * y * z + a * x;
        rotation[8] = 1 - b * (x * x + y * y);
    }
    int count = split ? 2 : counts.x + counts.y;
    for (int child = 0; child < count; ++child) {
        int destination = split ?
            control[4] + control[5] + child * control[6] + offsets.w :
            (child == 0 && counts.x ? offsets.x : control[4] + offsets.y);
        for (int parameter = 0; parameter < kMaxParamCount; ++parameter) {
            float value = parameters[static_cast<int64_t>(parameter) * gaussian_count + id];
            if (split && parameter < kPositionIndex + 3) {
                float delta = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    int64_t sample = (static_cast<int64_t>(child) * control[3] + offsets.z) * 3 + axis;
                    float local = samples[sample] * scale[axis];
                    delta += rotation[parameter * 3 + axis] * local;
                }
                value += delta;
            }
            else if (split && parameter >= kLogScaleIndex && parameter < kLogScaleIndex + 3) {
                value = logf(scale[parameter - kLogScaleIndex] / 1.6f);
            }
            gaussian_parameters[static_cast<int64_t>(parameter) * next_gaussian_count + destination] = value;
        }
        if (history_parent_ids) {

            history_parent_ids[destination] = !split && child == 0 && counts.x ? id : -1;
        }
    }
}

__global__ void FinishDensityControlKernel(
    float* gradient_sums,
    float* absolute_gradient_sums,
    unsigned int* observation_counts,
    const int4* density_counts,
    const int4* density_offsets,
    float* samples,
    int* control,
    int capacity) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < __ldg(control + 7)) {
        gradient_sums[id] = 0;
        absolute_gradient_sums[id] = 0;
        observation_counts[id] = 0;
        int4 counts = __ldg(density_counts + id), offsets = __ldg(density_offsets + id);
        int survivor_count = __ldg(control + 4);
        int clone_count = __ldg(control + 5);
        int split_count = __ldg(control + 6);
        bool split = counts.w != 0;
        int count = split ? 2 : counts.x + counts.y;
        for (int child = 0; child < count; ++child) {
            int destination = split ?
                survivor_count + clone_count + child * split_count + offsets.w :
                (child == 0 && counts.x ? offsets.x : survivor_count + offsets.y);
            samples[destination] = counts.y || counts.z ? 1.0f : 0.0f;
            reinterpret_cast<int*>(samples + capacity)[destination] = id;
        }
    }
    if (id == 0) {
        control[0] = control[1];
        unsigned long long* offsets = reinterpret_cast<unsigned long long*>(control + 8);
        offsets[0] = offsets[1];
    }
}

__global__ void AccumulateRoundStatisticsKernel(
    const float* statistics,
    const float* absolute_statistics,
    const int* control,
    float* gradient_sums,
    float* absolute_gradient_sums,
    unsigned int* observation_counts,
    int gaussian_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gaussian_count || id >= control[0]) {
        return;
    }
    gradient_sums[id] += statistics[id];
    absolute_gradient_sums[id] += absolute_statistics[id];
    observation_counts[id] += static_cast<unsigned int>(statistics[gaussian_count + id]);
}

__global__ void ResetDensityOpacityKernel(
    const int* control,
    float* parameters,
    int stride,
    float max_opacity_parameter) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= control[0]) {
        return;
    }
    int64_t index = static_cast<int64_t>(kOpacityIndex) * stride + id;
    parameters[index] = fminf(parameters[index], max_opacity_parameter);
}

} // namespace

DensityController::~DensityController() {
    cudaFree(spare_gaussian_parameters_);
    cudaFree(history_parent_ids_);
    cudaFree(gradient_sums_);
    cudaFree(absolute_gradient_sums_);
    cudaFree(samples_);
    cudaFree(observation_counts_);
    cudaFree(density_counts_);
    cudaFree(density_offsets_);
    cudaFree(control_);
    cudaFree(scan_storage_);
}

bool DensityController::Reserve(int capacity) {
    if (capacity <= capacity_) {
        return true;
    }
    float* gradient_sums = nullptr;
    float* absolute_gradient_sums = nullptr;
    float* samples = nullptr;
    unsigned int* observation_counts = nullptr;
    int* density_counts = nullptr;
    int* density_offsets = nullptr;
    size_t bytes = static_cast<size_t>(capacity) * sizeof(int);
    if (cudaMalloc(&gradient_sums, bytes) != cudaSuccess ||
        cudaMalloc(&absolute_gradient_sums, bytes) != cudaSuccess ||
        cudaMalloc(&observation_counts, bytes) != cudaSuccess ||
        cudaMalloc(&density_counts, 4 * (bytes + sizeof(int))) != cudaSuccess ||
        cudaMalloc(&density_offsets, 4 * (bytes + sizeof(int))) != cudaSuccess ||
        cudaMalloc(&samples, 6 * bytes) != cudaSuccess) {
        cudaFree(gradient_sums);
        cudaFree(absolute_gradient_sums);
        cudaFree(samples);
        cudaFree(observation_counts);
        cudaFree(density_counts);
        cudaFree(density_offsets);
        return false;
    }
    cudaError_t result = cudaSuccess;
    if (capacity_ > 0) {
        result = cudaMemcpyAsync(gradient_sums, gradient_sums_, sizeof(float) * capacity_, cudaMemcpyDeviceToDevice);
        if (result == cudaSuccess) {
            result = cudaMemcpyAsync(absolute_gradient_sums, absolute_gradient_sums_,
                sizeof(float) * capacity_, cudaMemcpyDeviceToDevice);
        }
        if (result == cudaSuccess) {
            result = cudaMemcpyAsync(observation_counts,
                observation_counts_,
                sizeof(unsigned int) * capacity_,
                cudaMemcpyDeviceToDevice);
        }
        if (result == cudaSuccess) {
            result = cudaMemcpyAsync(density_counts,
                density_counts_,
                sizeof(int4) * (static_cast<size_t>(capacity_) + 1),
                cudaMemcpyDeviceToDevice);
        }
        if (result == cudaSuccess) {
            result = cudaMemcpyAsync(density_offsets,
                density_offsets_,
                sizeof(int4) * (static_cast<size_t>(capacity_) + 1),
                cudaMemcpyDeviceToDevice);
        }
    }
    if (result == cudaSuccess) {
        result = cudaMemsetAsync(gradient_sums + capacity_, 0, sizeof(float) * (capacity - capacity_));
    }
    if (result == cudaSuccess) {
        result = cudaMemsetAsync(absolute_gradient_sums + capacity_, 0, sizeof(float) * (capacity - capacity_));
    }
    if (result == cudaSuccess) {
        result = cudaMemsetAsync(observation_counts + capacity_, 0, sizeof(unsigned int) * (capacity - capacity_));
    }
    if (result != cudaSuccess) {
        cudaFree(gradient_sums);
        cudaFree(absolute_gradient_sums);
        cudaFree(samples);
        cudaFree(observation_counts);
        cudaFree(density_counts);
        cudaFree(density_offsets);
        return false;
    }
    cudaFree(gradient_sums_);
    cudaFree(absolute_gradient_sums_);
    cudaFree(observation_counts_);
    cudaFree(density_counts_);
    cudaFree(density_offsets_);
    cudaFree(samples_);
    samples_ = samples;
    gradient_sums_ = gradient_sums;
    absolute_gradient_sums_ = absolute_gradient_sums;
    observation_counts_ = observation_counts;
    density_counts_ = density_counts;
    density_offsets_ = density_offsets;
    capacity_ = capacity;
    return true;
}

bool DensityController::Initialize(
    int gaussian_count,
    int capacity) {
    if (gaussian_count < 0 || capacity < gaussian_count || capacity < 0 || capacity > INT_MAX / 6) {
        return false;
    }
    if (!control_ && cudaMalloc(&control_, 8 * sizeof(int) + 2 * sizeof(unsigned long long)) != cudaSuccess) {
        return false;
    }
    if (!Reserve(std::max(capacity, 1))) {
        return false;
    }
    if (cudaMemsetAsync(gradient_sums_, 0, sizeof(float) * capacity_) != cudaSuccess ||
        cudaMemsetAsync(absolute_gradient_sums_, 0, sizeof(float) * capacity_) != cudaSuccess ||
        cudaMemsetAsync(observation_counts_, 0, sizeof(unsigned int) * capacity_) != cudaSuccess) {
        return false;
    }
    int device = 0, sm_count = 0, threads_per_sm = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device) != cudaSuccess) {
        return false;
    }
    random_blocks_ = std::max(1, sm_count * (threads_per_sm / kBlockSize));
    InitializeDensityKernel<<<1, 1>>>(control_, gaussian_count);
    parameter_stride_ = capacity;
    seed_ = 0;
    return cudaGetLastError() == cudaSuccess;
}

bool DensityController::AccumulateStatistics(
    const float* gradients,
    const float* radii,
    const int* gaussian_ids,
    int observation_count) {
    if (observation_count == 0) {
        return true;
    }
    if (!control_ || !gradients || !radii || observation_count < 0) {
        return false;
    }
    AccumulateDensityStatisticsKernel<<<GetGridSize(observation_count, kBlockSize), kBlockSize>>>(gradients,
        radii,
        gaussian_ids,
        control_,
        gradient_sums_,
        observation_counts_,
        observation_count);
    return cudaGetLastError() == cudaSuccess;
}

bool DensityController::ControlDensity(
    GaussianBuffers& gaussians,
    float scene_extent,
    float gradient_threshold,
    float percent_dense,
    float min_opacity,
    float max_screen_size,
    unsigned long long seed,
    int* changes,
    float absolute_gradient_threshold) {
    int gaussian_count = gaussians.gaussian_count_;
    const float* parameters = gaussians.gaussians_;
    if (!control_ || (!parameters && gaussian_count > 0) ||
        (parameters && parameters == spare_gaussian_parameters_) || gaussian_count < 0 ||
        gaussian_count != parameter_stride_ || spare_gaussian_stride_ < 0 || spare_gaussian_stride_ > INT_MAX / 6 ||
        !std::isfinite(scene_extent) || scene_extent < 0 || !std::isfinite(gradient_threshold) ||
        gradient_threshold < 0 || !std::isfinite(percent_dense) || percent_dense < 0 || !std::isfinite(min_opacity) ||
        min_opacity < 0 || min_opacity > 1 || !std::isfinite(max_screen_size) || max_screen_size < 0 ||
        !std::isfinite(absolute_gradient_threshold) || absolute_gradient_threshold < 0) {
        return false;
    }
    size_t scan_bytes = 0;
    int4* counts = reinterpret_cast<int4*>(density_counts_);
    int4* offsets = reinterpret_cast<int4*>(density_offsets_);
    if (cub::DeviceScan::ExclusiveScan(nullptr,
            scan_bytes,
            counts,
            offsets,
            AddDensityCounts{},
            make_int4(0, 0, 0, 0),
            gaussian_count + 1) != cudaSuccess) {
        return false;
    }
    if (scan_bytes > scan_storage_bytes_) {
        void* storage = nullptr;
        if (cudaMalloc(&storage, scan_bytes) != cudaSuccess) {
            return false;
        }
        cudaFree(scan_storage_);
        scan_storage_ = storage;
        scan_storage_bytes_ = scan_bytes;
    }
    CountDensityChangesKernel<<<GetGridSize(gaussian_count + 1, kBlockSize), kBlockSize>>>(parameters,
        gradient_sums_,
        absolute_gradient_sums_,
        observation_counts_,
        control_,
        counts,
        gaussian_count,
        gradient_threshold,
        percent_dense * scene_extent,
        min_opacity,
        0.1f * scene_extent,
        max_screen_size > 0,
        absolute_gradient_threshold);
    if (cub::DeviceScan::ExclusiveScan(scan_storage_,
            scan_bytes,
            counts,
            offsets,
            AddDensityCounts{},
            make_int4(0, 0, 0, 0),
            gaussian_count + 1) != cudaSuccess) {
        return false;
    }
    PrepareDensityControlKernel<<<1, 1>>>(offsets, control_, gaussian_count, random_blocks_);

    int required = 0;
    if (changes) {
        int totals[8];
        if (cudaMemcpy(totals, control_, sizeof(totals), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        required = totals[1];
        changes[0] = totals[5] + 2 * totals[6];
        changes[1] = totals[7] - totals[4];
    }
    else {
        if (cudaMemcpy(&required, control_ + 1, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
    }
    int next_stride = required;
    if (!Reserve(std::max(gaussian_count, std::max(next_stride, spare_gaussian_stride_)))) {
        return false;
    }
    if (required == 0) {
        cudaFree(spare_gaussian_parameters_);
        cudaFree(history_parent_ids_);
        spare_gaussian_parameters_ = nullptr;
        history_parent_ids_ = nullptr;
        spare_gaussian_stride_ = 0;
    }
    else if (!spare_gaussian_parameters_ || next_stride > spare_gaussian_stride_) {
        float* replacement = nullptr;
        if (cudaMalloc(&replacement, sizeof(float) * kMaxParamCount * static_cast<size_t>(next_stride)) !=
            cudaSuccess) {
            return false;
        }
        cudaFree(spare_gaussian_parameters_);
        spare_gaussian_parameters_ = replacement;
        spare_gaussian_stride_ = next_stride;
    }

    if (required > 0) {
        int* history_parent_ids = nullptr;
        if (cudaMalloc(&history_parent_ids, static_cast<size_t>(required) * sizeof(int)) != cudaSuccess) {
            return false;
        }
        cudaFree(history_parent_ids_);
        history_parent_ids_ = history_parent_ids;
    }

    spare_gaussian_stride_ = next_stride;
    if (seed != seed_) {
        if (cudaMemsetAsync(control_ + 8, 0, 2 * sizeof(unsigned long long)) != cudaSuccess) {
            return false;
        }
        seed_ = seed;

        PrepareDensityControlKernel<<<1, 1>>>(reinterpret_cast<int4*>(density_offsets_),
            control_,
            gaussian_count,
            random_blocks_);
    }
    GenerateDensitySamplesKernel<<<random_blocks_, kBlockSize>>>(control_, samples_, random_blocks_, seed);
    WriteDensityGaussiansKernel<<<GetGridSize(gaussian_count, kBlockSize), kBlockSize>>>(parameters,
        reinterpret_cast<int4*>(density_counts_),
        reinterpret_cast<int4*>(density_offsets_),
        control_,
        samples_,
        spare_gaussian_parameters_,
        history_parent_ids_,
        gaussian_count,
        spare_gaussian_stride_);
    FinishDensityControlKernel<<<GetGridSize(gaussian_count, kBlockSize), kBlockSize>>>(gradient_sums_,
        absolute_gradient_sums_,
        observation_counts_,
        reinterpret_cast<int4*>(density_counts_),
        reinterpret_cast<int4*>(density_offsets_),
        samples_,
        control_,
        capacity_);
    bool submitted = cudaGetLastError() == cudaSuccess;
    if (submitted) {
        parameter_stride_ = next_stride;
        std::swap(gaussians.gaussians_, spare_gaussian_parameters_);
        gaussians.gaussian_count_ = next_stride;
        spare_gaussian_stride_ = gaussian_count;
    }
    return submitted;
}

bool DensityController::AccumulateStatistics(
    const float* statistics,
    const float* absolute_statistics,
    int gaussian_count) {
    if (gaussian_count == 0) {
        return true;
    }
    if (!control_ || !statistics || !absolute_statistics || gaussian_count < 0 || gaussian_count > capacity_) {
        return false;
    }
    AccumulateRoundStatisticsKernel<<<GetGridSize(gaussian_count, kBlockSize), kBlockSize>>>(statistics,
        absolute_statistics,
        control_,
        gradient_sums_,
        absolute_gradient_sums_,
        observation_counts_,
        gaussian_count);
    return cudaGetLastError() == cudaSuccess;
}

bool DensityController::ResetOpacity(
    float* parameters,
    int stride,
    float max_opacity) {
    if (!control_ || stride != parameter_stride_ || (!parameters && stride > 0) ||
        !std::isfinite(max_opacity) || max_opacity <= 0.0f || max_opacity >= 1.0f) {
        return false;
    }
    float max_opacity_parameter = std::log(max_opacity / (1.0f - max_opacity));
    ResetDensityOpacityKernel<<<GetGridSize(stride, kBlockSize), kBlockSize>>>(control_,
        parameters,
        stride,
        max_opacity_parameter);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace gs_native
