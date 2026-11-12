#include "solver/solver.h"

#include "render/renderer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace gs_native {

namespace {

__constant__ float c_ssim_kernel[11] = {1.028380120e-03f,
    7.598758210e-03f,
    3.600077330e-02f,
    1.093606870e-01f,
    2.130055430e-01f,
    2.660117150e-01f,
    2.130055430e-01f,
    1.093606870e-01f,
    3.600077330e-02f,
    7.598758210e-03f,
    1.028380120e-03f};

__global__ __launch_bounds__(512) void ComputeImageGradientsKernel(
    const float* image,
    const uint8_t* target,
    float* gradients,
    int width,
    int height) {
    constexpr int tile_width = 16;
    constexpr int tile_height = 32;
    constexpr int block_size = tile_width * tile_height;
    constexpr int input_width = tile_width + 20;
    constexpr int input_height = tile_height + 20;
    constexpr int moment_width = tile_width + 10;
    constexpr int center_height = tile_height + 10;
    constexpr int horizontal_slots = (input_height * moment_width + block_size - 1) / block_size;
    constexpr int statistic_slots = (center_height * moment_width + block_size - 1) / block_size;
    constexpr int gradient_slots = (center_height * tile_width + block_size - 1) / block_size;

    __shared__ float scratch[5 * input_height * moment_width];
    int t = threadIdx.x;
    int channel = blockIdx.z;
    int origin_x = blockIdx.x * tile_width;
    int origin_y = blockIdx.y * tile_height;
    int x = origin_x + t % tile_width;
    int y = origin_y + t / tile_width;
    bool inside = x < width && y < height;
    int64_t pixel = (static_cast<int64_t>(y) * width + x) * 3 + channel;
    float render = inside ? image[pixel] : 0;
    float reference = inside ? target[pixel] * (1.0f / 255.0f) : 0;

    for (int j = t; j < input_height * input_width; j += block_size) {
        int hx = origin_x + j % input_width - 10;
        int hy = origin_y + j / input_width - 10;
        bool valid = hx >= 0 && hy >= 0 && hx < width && hy < height;
        int64_t index = (static_cast<int64_t>(hy) * width + hx) * 3 + channel;
        scratch[j] = valid ? image[index] : 0;
        scratch[input_height * input_width + j] = valid ? target[index] * (1.0f / 255.0f) : 0;
    }
    __syncthreads();
    float horizontal[horizontal_slots][5] = {};
    for (int slot = 0; slot < horizontal_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < input_height * moment_width) {
            for (int dx = 0; dx < 11; ++dx) {
                float weight = c_ssim_kernel[dx];
                int index = (j / moment_width) * input_width + j % moment_width + dx;
                float a = scratch[index];
                float b = scratch[input_height * input_width + index];
                horizontal[slot][0] += weight * a;
                horizontal[slot][1] += weight * b;
                horizontal[slot][2] += weight * a * a;
                horizontal[slot][3] += weight * b * b;
                horizontal[slot][4] += weight * a * b;
            }
        }
    }
    __syncthreads();
    for (int slot = 0; slot < horizontal_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < input_height * moment_width) {
            for (int moment = 0; moment < 5; ++moment) {
                scratch[moment * input_height * moment_width + j] = horizontal[slot][moment];
            }
        }
    }
    __syncthreads();
    float statistics[statistic_slots][5] = {};
    for (int slot = 0; slot < statistic_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < center_height * moment_width) {
            for (int dy = 0; dy < 11; ++dy) {
                float weight = c_ssim_kernel[dy];
                int index = (j / moment_width + dy) * moment_width + j % moment_width;
                for (int moment = 0; moment < 5; ++moment) {
                    statistics[slot][moment] += weight * scratch[moment * input_height * moment_width + index];
                }
            }
        }
    }
    __syncthreads();
    for (int slot = 0; slot < statistic_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < center_height * moment_width) {
            int hx = origin_x + j % moment_width - 5;
            int hy = origin_y + j / moment_width - 5;
            float a = 0, b = 0, c = 0;

            if (hx >= 0 && hy >= 0 && hx < width && hy < height) {
                float* moments = statistics[slot];
                float m = moments[0], n = moments[1];
                float inverse_d = 1.0f / (m * m + n * n + 0.0001f);
                float inverse_e = 1.0f / (moments[2] - m * m + moments[3] - n * n + 0.0009f);
                float luminance = (2 * m * n + 0.0001f) * inverse_d;
                float contrast = (2 * (moments[4] - m * n) + 0.0009f) * inverse_e;
                float mean_derivative = 2 * (n - m * luminance) * inverse_d * contrast;
                b = 2 * luminance * inverse_e;
                c = -luminance * contrast * inverse_e;
                a = mean_derivative - n * b - 2 * m * c;
            }
            scratch[j] = a;
            scratch[center_height * moment_width + j] = b;
            scratch[2 * center_height * moment_width + j] = c;
        }
    }
    __syncthreads();
    float gradient_horizontal[gradient_slots][3] = {};
    for (int slot = 0; slot < gradient_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < center_height * tile_width) {
            for (int dx = 0; dx < 11; ++dx) {
                float weight = c_ssim_kernel[dx];
                int index = (j / tile_width) * moment_width + j % tile_width + dx;
                for (int k = 0; k < 3; ++k) {
                    gradient_horizontal[slot][k] += weight * scratch[k * center_height * moment_width + index];
                }
            }
        }
    }
    __syncthreads();
    for (int slot = 0; slot < gradient_slots; ++slot) {
        int j = t + slot * block_size;
        if (j < center_height * tile_width) {
            for (int k = 0; k < 3; ++k) {
                scratch[k * center_height * tile_width + j] = gradient_horizontal[slot][k];
            }
        }
    }
    __syncthreads();
    if (inside) {
        float filtered[3] = {};
        for (int dy = 0; dy < 11; ++dy) {
            float weight = c_ssim_kernel[dy];
            int index = (t / tile_width + dy) * tile_width + t % tile_width;
            for (int k = 0; k < 3; ++k) {
                filtered[k] += weight * scratch[k * center_height * tile_width + index];
            }
        }
        float residual = render - reference;
        float l1_gradient = static_cast<float>(residual > 0) - static_cast<float>(residual < 0);
        float ssim_gradient = -(filtered[0] + reference * filtered[1] + 2 * render * filtered[2]);
        float normalization = 1.0f / (3.0f * static_cast<float>(width) * height);
        gradients[pixel] = (0.8f * l1_gradient + 0.2f * ssim_gradient) * normalization;
    }
}

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

__device__ void ComputeEquationRotation(
    const float* parameters,
    float* rotation,
    int id,
    int count) {
    float x = parameters[(kRotationIndex + 0) * count + id];
    float y = parameters[(kRotationIndex + 1) * count + id];
    float z = parameters[(kRotationIndex + 2) * count + id];
    float angle = sqrtf(x * x + y * y + z * z);
    if (angle < 1e-8f) {
        rotation[0] = 1.0f;
        rotation[1] = 0.0f;
        rotation[2] = 0.0f;
        rotation[3] = 0.0f;
        rotation[4] = 1.0f;
        rotation[5] = 0.0f;
        rotation[6] = 0.0f;
        rotation[7] = 0.0f;
        rotation[8] = 1.0f;
        return;
    }
    x /= angle;
    y /= angle;
    z /= angle;
    float c = cosf(angle);
    float s = sinf(angle);
    float t = 1.0f - c;
    rotation[0] = c + t * x * x;
    rotation[1] = t * x * y - s * z;
    rotation[2] = t * x * z + s * y;
    rotation[3] = t * x * y + s * z;
    rotation[4] = c + t * y * y;
    rotation[5] = t * y * z - s * x;
    rotation[6] = t * x * z - s * y;
    rotation[7] = t * y * z + s * x;
    rotation[8] = c + t * z * z;
}

template<int Degree>
__device__ void ComputeEquationSh(
    const float* parameters,
    const CameraParams& camera,
    float* basis,
    float* color,
    float* color_derivatives,
    int id,
    int count) {
    constexpr int s = (Degree + 1) * (Degree + 1);
    float direction[3];
    for (int a = 0; a < 3; ++a) {
        direction[a] = parameters[a * count + id];
        for (int b = 0; b < 3; ++b) {
            direction[a] += camera.R_c_w_[b * 3 + a] * camera.t_c_w_[b];
        }
    }
    float inverse_length =
        rsqrtf(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    float x = direction[0] * inverse_length;
    float y = direction[1] * inverse_length;
    float z = direction[2] * inverse_length;
    float derivatives[s * 3] = {};
    basis[0] = kShC0;
    if (Degree >= 1) {
        basis[1] = -kShC1 * y;
        basis[2] = kShC1 * z;
        basis[3] = -kShC1 * x;
        derivatives[4] = -kShC1;
        derivatives[8] = kShC1;
        derivatives[9] = -kShC1;
    }
    if (Degree >= 2) {
        constexpr float c = kShC2[0], d = kShC2[2], e = kShC2[4];
        basis[4] = c * x * y;
        basis[5] = -c * y * z;
        basis[6] = d * (2 * z * z - x * x - y * y);
        basis[7] = -c * x * z;
        basis[8] = e * (x * x - y * y);
        derivatives[12] = c * y;
        derivatives[13] = c * x;
        derivatives[16] = -c * z;
        derivatives[17] = -c * y;
        derivatives[18] = -2 * d * x;
        derivatives[19] = -2 * d * y;
        derivatives[20] = 4 * d * z;
        derivatives[21] = -c * z;
        derivatives[23] = -c * x;
        derivatives[24] = 2 * e * x;
        derivatives[25] = -2 * e * y;
    }
    if (Degree >= 3) {
        constexpr float a = kShC3[0], b = kShC3[1], c = kShC3[2], d = kShC3[3], e = kShC3[5];
        basis[9] = a * y * (3 * x * x - y * y);
        basis[10] = b * x * y * z;
        basis[11] = c * y * (4 * z * z - x * x - y * y);
        basis[12] = d * z * (2 * z * z - 3 * x * x - 3 * y * y);
        basis[13] = c * x * (4 * z * z - x * x - y * y);
        basis[14] = e * z * (x * x - y * y);
        basis[15] = a * x * (x * x - 3 * y * y);
        derivatives[27] = 6 * a * x * y;
        derivatives[28] = 3 * a * (x * x - y * y);
        derivatives[30] = b * y * z;
        derivatives[31] = b * x * z;
        derivatives[32] = b * x * y;
        derivatives[33] = -2 * c * x * y;
        derivatives[34] = c * (4 * z * z - x * x - 3 * y * y);
        derivatives[35] = 8 * c * y * z;
        derivatives[36] = -6 * d * x * z;
        derivatives[37] = -6 * d * y * z;
        derivatives[38] = d * (6 * z * z - 3 * x * x - 3 * y * y);
        derivatives[39] = c * (4 * z * z - 3 * x * x - y * y);
        derivatives[40] = -2 * c * x * y;
        derivatives[41] = 8 * c * x * z;
        derivatives[42] = 2 * e * x * z;
        derivatives[43] = -2 * e * y * z;
        derivatives[44] = e * (x * x - y * y);
        derivatives[45] = 3 * a * (x * x - y * y);
        derivatives[46] = -6 * a * x * y;
    }
    for (int channel = 0; channel < 3; ++channel) {
        float value = 0.0f;
        float derivative[3] = {};
        for (int j = 0; j < s; ++j) {
            float coefficient = parameters[(kShIndex + channel * 16 + j) * count + id];
            value += coefficient * basis[j];
            if (color_derivatives) {
                for (int a = 0; a < 3; ++a) {
                    derivative[a] += coefficient * derivatives[j * 3 + a];
                }
            }
        }
        color[channel] = value + 0.5f;
        if (color_derivatives) {
            float radial = derivative[0] * x + derivative[1] * y + derivative[2] * z;
            color_derivatives[channel * 3] = (derivative[0] - radial * x) * inverse_length;
            color_derivatives[channel * 3 + 1] = (derivative[1] - radial * y) * inverse_length;
            color_derivatives[channel * 3 + 2] = (derivative[2] - radial * z) * inverse_length;
        }
    }
}

__global__ void ComputeSplatGradientsKernel(
    const int* ids,
    const int64_t* offsets,
    const float2* means,
    const float4* conic_opacities,
    const float3* colors,
    const float* image,
    const float* image_gradients,
    CameraParams camera,
    float* gradients,
    int count) {
    __shared__ int shared_ids[256];
    __shared__ float2 shared_means[256];
    __shared__ float4 shared_conics[256];
    __shared__ float3 shared_colors[256];
    int t = threadIdx.x;
    int tile = blockIdx.x;
    int x = tile % camera.tile_count_x_ * 16 + t % 16;
    int y = tile / camera.tile_count_x_ * 16 + t / 16;
    bool inside = x < camera.width_ && y < camera.height_;
    bool done = !inside;
    int64_t pixel = (static_cast<int64_t>(y) * camera.width_ + x) * 3;
    float final_color[3] = {}, image_gradient[3] = {}, prefix_color[3] = {};
    if (inside) {
        for (int c = 0; c < 3; ++c) {
            final_color[c] = image[pixel + c];
            image_gradient[c] = image_gradients[pixel + c];
        }
    }
    float transmittance = 1.0f;
    int64_t begin = offsets[tile], end = offsets[tile + 1];
    for (int64_t base = begin; base < end; base += 256) {
        if (__syncthreads_count(done) == 256) {
            break;
        }
        if (base + t < end) {
            int id = ids[base + t];
            shared_ids[t] = id;
            shared_means[t] = means[id];
            shared_conics[t] = conic_opacities[id];
            shared_colors[t] = colors[id];
        }
        __syncthreads();
        int batch = static_cast<int>(min(static_cast<int64_t>(256), end - base));
        for (int j = 0; j < batch; ++j) {
            float contribution[9] = {};
            bool has_contribution = false;
            if (!done) {
                float dx = shared_means[j].x - x, dy = shared_means[j].y - y;
                float4 conic = shared_conics[j];
                float power = -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
                if (power <= 0.0f) {
                    float raw_alpha = conic.w * expf(power);
                    float alpha = fminf(0.99f, raw_alpha);
                    if (alpha >= 1.0f / 255.0f) {
                        float next_transmittance = transmittance * (1.0f - alpha);
                        if (next_transmittance < 1e-4f) {
                            done = true;
                        }
                        else {
                            has_contribution = true;
                            float color[3] = {shared_colors[j].x, shared_colors[j].y, shared_colors[j].z};
                            float alpha_gradient = 0;
                            float weight = alpha * transmittance;
                            for (int c = 0; c < 3; ++c) {

                                alpha_gradient += image_gradient[c] *
                                    (prefix_color[c] + transmittance * color[c] - final_color[c]) / (1.0f - alpha);
                                contribution[6 + c] = weight * image_gradient[c];
                            }

                            if (raw_alpha < 0.99f) {
                                float common = alpha_gradient * raw_alpha;
                                float qx = conic.x * dx + conic.y * dy;
                                float qy = conic.y * dx + conic.z * dy;
                                contribution[0] = -common * qx;
                                contribution[1] = -common * qy;
                                contribution[2] = 0.5f * common * qx * qx;
                                contribution[3] = common * qx * qy;
                                contribution[4] = 0.5f * common * qy * qy;
                                contribution[5] = common * (1.0f - conic.w);
                            }
                            for (int c = 0; c < 3; ++c) {
                                prefix_color[c] += color[c] * weight;
                            }
                            transmittance = next_transmittance;
                        }
                    }
                }
            }

            if (__any_sync(0xffffffffu, has_contribution)) {
                for (int field = 0; field < 9; ++field) {
                    float sum = contribution[field];
                    for (int stride = 16; stride > 0; stride /= 2) {
                        sum += __shfl_down_sync(0xffffffffu, sum, stride);
                    }
                    if (t % 32 == 0 && sum != 0.0f) {
                        atomicAdd(gradients + static_cast<int64_t>(field) * count + shared_ids[j], sum);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void CollectProjectionStatisticsKernel(
    const float* gradients,
    const int* record_offsets,
    float* statistics,
    int count,
    int width,
    int height) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) {
        return;
    }
    float x = gradients[id] * (0.5f * width);
    float y = gradients[count + id] * (0.5f * height);
    statistics[id] = sqrtf(x * x + y * y);
    statistics[count + id] = record_offsets[id + 1] > record_offsets[id] ? 1.0f : 0.0f;
}

template<int Degree>
__global__ void MapGaussianGradientsKernel(
    const float* parameters,
    CameraParams camera,
    float* gradients,
    int count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) {
        return;
    }
    float input[9];
    bool active = false;
    for (int j = 0; j < 9; ++j) {
        input[j] = gradients[static_cast<int64_t>(j) * count + id];
        active = active || input[j] != 0.0f;
    }

    if (!active) {
        return;
    }
    float rotation[9], camera_rotation[9], covariance[9], position[3], scale_squared[3];
    ComputeEquationRotation(parameters, rotation, id, count);
    for (int r = 0; r < 3; ++r) {
        position[r] = camera.t_c_w_[r];
        for (int c = 0; c < 3; ++c) {
            position[r] += camera.R_c_w_[r * 3 + c] * parameters[static_cast<int64_t>(c) * count + id];
            camera_rotation[r * 3 + c] = camera.R_c_w_[r * 3] * rotation[c] +
                camera.R_c_w_[r * 3 + 1] * rotation[3 + c] + camera.R_c_w_[r * 3 + 2] * rotation[6 + c];
        }
        float scale = expf(parameters[static_cast<int64_t>(kLogScaleIndex + r) * count + id]);
        scale_squared[r] = scale * scale;
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            covariance[r * 3 + c] = 0;
            for (int j = 0; j < 3; ++j) {
                covariance[r * 3 + c] += camera_rotation[r * 3 + j] * scale_squared[j] * camera_rotation[c * 3 + j];
            }
        }
    }
    float inverse_z = 1.0f / position[2];
    float nx = position[0] * inverse_z, ny = position[1] * inverse_z;
    float limit_x = 1.3f * 0.5f * camera.width_ / camera.fx_;
    float limit_y = 1.3f * 0.5f * camera.height_ / camera.fy_;
    float jx = fminf(fmaxf(nx, -limit_x), limit_x);
    float jy = fminf(fmaxf(ny, -limit_y), limit_y);
    float projection[6] = {camera.fx_ * inverse_z, 0, -camera.fx_ * jx * inverse_z,
        0, camera.fy_ * inverse_z, -camera.fy_ * jy * inverse_z};
    float product[6] = {}, local_projection[6] = {};
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) {
            for (int j = 0; j < 3; ++j) {
                product[r * 3 + c] += projection[r * 3 + j] * covariance[j * 3 + c];
                local_projection[r * 3 + c] += projection[r * 3 + j] * camera_rotation[j * 3 + c];
            }
        }
    }
    constexpr int s = (Degree + 1) * (Degree + 1);
    float basis[s], color[3], color_derivatives[9];
    ComputeEquationSh<Degree>(parameters, camera, basis, color, color_derivatives, id, count);
    float right_gradient[3];
    for (int axis = 0; axis < 3; ++axis) {
        float dx = camera.R_c_w_[axis], dy = camera.R_c_w_[3 + axis], dz = camera.R_c_w_[6 + axis];
        float du = (dx - nx * dz) * inverse_z;
        float dv = (dy - ny * dz) * inverse_z;
        float djx = nx >= -limit_x && nx <= limit_x ? du : 0;
        float djy = ny >= -limit_y && ny <= limit_y ? dv : 0;
        float delta[6] = {-camera.fx_ * dz * inverse_z * inverse_z, 0,
            -camera.fx_ * (djx - jx * dz * inverse_z) * inverse_z,
            0, -camera.fy_ * dz * inverse_z * inverse_z,
            -camera.fy_ * (djy - jy * dz * inverse_z) * inverse_z};
        float daa = 0, dab = 0, ddd = 0;
        for (int j = 0; j < 3; ++j) {
            daa += 2 * delta[j] * product[j];
            dab += delta[j] * product[3 + j] + delta[3 + j] * product[j];
            ddd += 2 * delta[3 + j] * product[3 + j];
        }
        float center_gradient = input[0] * camera.fx_ * du + input[1] * camera.fy_ * dv +
            input[2] * daa + input[3] * dab + input[4] * ddd;
        for (int c = 0; c < 3; ++c) {
            if (color[c] >= 0) {
                center_gradient += input[6 + c] * color_derivatives[c * 3 + axis];
            }
        }
        gradients[static_cast<int64_t>(axis) * count + id] = center_gradient;
        float a = local_projection[axis], b = local_projection[3 + axis];
        gradients[static_cast<int64_t>(kLogScaleIndex + axis) * count + id] =
            2 * scale_squared[axis] * (input[2] * a * a + input[3] * a * b + input[4] * b * b);
        int j = (axis + 1) % 3, k = (axis + 2) % 3;
        float difference = scale_squared[j] - scale_squared[k];
        right_gradient[axis] = difference *
            (2 * input[2] * local_projection[j] * local_projection[k] +
                input[3] * (local_projection[j] * local_projection[3 + k] +
                    local_projection[k] * local_projection[3 + j]) +
                2 * input[4] * local_projection[3 + j] * local_projection[3 + k]);
    }

    float omega[3];
    float theta_squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
        omega[axis] = parameters[static_cast<int64_t>(kRotationIndex + axis) * count + id];
        theta_squared += omega[axis] * omega[axis];
    }
    float a = 0.5f, b = 1.0f / 6.0f;
    if (theta_squared > 1e-4f) {
        float theta = sqrtf(theta_squared);
        float half_sinc = sinf(0.5f * theta) / (0.5f * theta);
        a = 0.5f * half_sinc * half_sinc;
        b = (theta - sinf(theta)) / (theta_squared * theta);
    }
    else {
        a -= theta_squared / 24.0f;
        b -= theta_squared / 120.0f;
    }
    float cross[3];
    for (int axis = 0; axis < 3; ++axis) {
        int j = (axis + 1) % 3, k = (axis + 2) % 3;
        cross[axis] = omega[j] * right_gradient[k] - omega[k] * right_gradient[j];
    }
    for (int axis = 0; axis < 3; ++axis) {
        int j = (axis + 1) % 3, k = (axis + 2) % 3;
        gradients[static_cast<int64_t>(kRotationIndex + axis) * count + id] =
            right_gradient[axis] + a * cross[axis] + b * (omega[j] * cross[k] - omega[k] * cross[j]);
    }
    gradients[static_cast<int64_t>(kOpacityIndex) * count + id] = input[5];
    for (int c = 0; c < 3; ++c) {
        for (int coefficient = 0; coefficient < s; ++coefficient) {
            gradients[static_cast<int64_t>(kShIndex + c * 16 + coefficient) * count + id] =
                color[c] >= 0 ? input[6 + c] * basis[coefficient] : 0;
        }
    }
}

__global__ void RemapGaussianHistoryKernel(
    const int* parent_ids,
    const float* previous_first,
    const float* previous_second,
    float* first,
    float* second,
    int previous_count,
    int gaussian_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gaussian_count) {
        return;
    }
    int parameter = blockIdx.y;
    int parent = parent_ids ? parent_ids[id] : -1;
    int64_t index = static_cast<int64_t>(parameter) * gaussian_count + id;
    if (parent >= 0 && parent < previous_count) {
        int64_t previous_index = static_cast<int64_t>(parameter) * previous_count + parent;
        first[index] = previous_first[previous_index];
        second[index] = previous_second[previous_index];
    }
    else {
        first[index] = 0.0f;
        second[index] = 0.0f;
    }
}

__global__ void UpdateGaussiansKernel(
    const float* gradients,
    float* parameters,
    float* first_moments,
    float* second_moments,
    int gaussian_count,
    float position_learning_rate,
    float first_correction,
    float second_correction) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= gaussian_count) {
        return;
    }
    int parameter = blockIdx.y;
    int64_t index = static_cast<int64_t>(parameter) * gaussian_count + id;
    float learning_rate;
    if (parameter < kLogScaleIndex) {
        learning_rate = position_learning_rate;
    }
    else if (parameter < kRotationIndex) {
        learning_rate = 0.005f;
    }
    else if (parameter < kOpacityIndex) {
        learning_rate = 0.001f;
    }
    else if (parameter < kShIndex) {
        learning_rate = 0.025f;
    }
    else {
        learning_rate = (parameter - kShIndex) % 16 == 0 ? 0.0025f : 0.0025f / 20.0f;
    }
    float gradient = gradients[index];
    float first = 0.9f * first_moments[index] + 0.1f * gradient;
    float second = 0.999f * second_moments[index] + 0.001f * gradient * gradient;
    first_moments[index] = first;
    second_moments[index] = second;
    parameters[index] -= learning_rate * (first * first_correction) /
        (sqrtf(second * second_correction) + 1e-15f);
}

}

Solver::~Solver() {
    cudaFree(image_gradients_);
    cudaFree(projection_statistics_);
    cudaFree(gaussian_gradients_);
    cudaFree(first_moments_);
    cudaFree(second_moments_);
}

bool Solver::Iterate(
    const CameraParams& camera,
    const Renderer& renderer,
    const float* image,
    const uint8_t* target,
    GaussianBuffers& gaussians,
    float position_learning_rate,
    int sh_degree) {
    if (!ComputeImageGradients(camera, image, target)) {
        return false;
    }
    if (!ComputeGaussianGradients(camera, renderer, gaussians, image, sh_degree)) {
        return false;
    }
    if (!UpdateGaussians(gaussians, position_learning_rate)) {
        return false;
    }
    return true;
}

bool Solver::ResetOpacityHistory() {
    if (gaussian_capacity_ == 0) {
        return true;
    }
    int64_t offset = static_cast<int64_t>(kOpacityIndex) * gaussian_capacity_;
    size_t bytes = static_cast<size_t>(gaussian_capacity_) * sizeof(float);
    if (cudaMemsetAsync(first_moments_ + offset, 0, bytes) != cudaSuccess ||
        cudaMemsetAsync(second_moments_ + offset, 0, bytes) != cudaSuccess) {
        return false;
    }
    return true;
}

bool Solver::ComputeImageGradients(
    const CameraParams& camera,
    const float* image,
    const uint8_t* target) {
    if (!image || !target || camera.width_ <= 0 || camera.height_ <= 0) {
        return false;
    }
    int64_t pixels = static_cast<int64_t>(camera.width_) * camera.height_;
    if (pixels > pixel_capacity_) {
        cudaFree(image_gradients_);
        image_gradients_ = nullptr;
        pixel_capacity_ = 0;
        if (cudaMalloc(&image_gradients_, static_cast<size_t>(pixels) * 3 * sizeof(float)) != cudaSuccess) {
            return false;
        }
        pixel_capacity_ = pixels;
    }
    dim3 grid((camera.width_ + 15ll) / 16, (camera.height_ + 31ll) / 32, 3);
    ComputeImageGradientsKernel<<<grid, 512>>>(image,
        target,
        image_gradients_,
        camera.width_,
        camera.height_);
    return cudaGetLastError() == cudaSuccess;
}

bool Solver::RemapGaussians(
    const int* parent_ids,
    int gaussian_count) {
    if (gaussian_count < 0) {
        return false;
    }
    float* statistics = nullptr;
    float* gradients = nullptr;
    float* first = nullptr;
    float* second = nullptr;
    if (gaussian_count > 0) {
        size_t bytes = static_cast<size_t>(gaussian_count) * kMaxParamCount * sizeof(float);
        cudaError_t result = cudaMalloc(&gradients, bytes);
        if (result == cudaSuccess) {
            result = cudaMalloc(&first, bytes);
        }
        if (result == cudaSuccess) {
            result = cudaMalloc(&second, bytes);
        }
        if (result == cudaSuccess) {
            result = cudaMalloc(&statistics, static_cast<size_t>(gaussian_count) * 2 * sizeof(float));
        }
        if (result == cudaSuccess) {
            dim3 grid((gaussian_count + 255) / 256, kMaxParamCount);
            RemapGaussianHistoryKernel<<<grid, 256>>>(parent_ids, first_moments_, second_moments_,
                first, second, gaussian_capacity_, gaussian_count);
            result = cudaGetLastError();
            if (result == cudaSuccess) {
                result = cudaStreamSynchronize(nullptr);
            }
        }
        if (result != cudaSuccess) {
            cudaFree(statistics);
            cudaFree(gradients);
            cudaFree(first);
            cudaFree(second);
            return false;
        }
    }
    cudaFree(projection_statistics_);
    cudaFree(gaussian_gradients_);
    cudaFree(first_moments_);
    cudaFree(second_moments_);
    projection_statistics_ = statistics;
    gaussian_gradients_ = gradients;
    first_moments_ = first;
    second_moments_ = second;
    gaussian_capacity_ = gaussian_count;
    return true;
}

bool Solver::ComputeGaussianGradients(
    const CameraParams& camera,
    const Renderer& renderer,
    const GaussianBuffers& gaussians,
    const float* image,
    int sh_degree) {
    int count = gaussians.gaussian_count_;
    if (count == 0) {
        return true;
    }
    int64_t pixels = static_cast<int64_t>(camera.width_) * camera.height_;
    int64_t tiles = static_cast<int64_t>(camera.tile_count_x_) * camera.tile_count_y_;
    if (count < 0 || !gaussians.gaussians_ || !image || !image_gradients_ ||
        camera.width_ <= 0 || camera.height_ <= 0 || pixels > pixel_capacity_ ||
        camera.tile_count_x_ <= 0 || camera.tile_count_y_ <= 0 || tiles > renderer.tile_capacity_ ||
        count > renderer.gaussian_capacity_ || !renderer.tile_lists_.gaussian_offsets_ ||
        !renderer.means_ || !renderer.conic_opacities_ || !renderer.colors_) {
        return false;
    }
    if (count != gaussian_capacity_) {

        if (iteration_ != 0 || !RemapGaussians(nullptr, count)) {
            return false;
        }
    }
    if (cudaMemsetAsync(gaussian_gradients_, 0,
            static_cast<size_t>(count) * kMaxParamCount * sizeof(float)) != cudaSuccess) {
        return false;
    }
    ComputeSplatGradientsKernel<<<static_cast<unsigned int>(tiles), 256>>>(renderer.tile_lists_.gaussian_ids_,
        renderer.tile_lists_.gaussian_offsets_,
        renderer.means_,
        renderer.conic_opacities_,
        renderer.colors_,
        image,
        image_gradients_,
        camera,
        gaussian_gradients_,
        count);
    CollectProjectionStatisticsKernel<<<static_cast<unsigned int>((count + 255ll) / 256), 256>>>(
        gaussian_gradients_, renderer.record_offsets_, projection_statistics_,
        count, camera.width_, camera.height_);
    sh_degree = std::clamp(sh_degree, 0, 3);
    int grid = static_cast<int>((count + 255ll) / 256);
    switch (sh_degree) {
    case 0:
        MapGaussianGradientsKernel<0><<<grid, 256>>>(gaussians.gaussians_, camera, gaussian_gradients_, count);
        break;
    case 1:
        MapGaussianGradientsKernel<1><<<grid, 256>>>(gaussians.gaussians_, camera, gaussian_gradients_, count);
        break;
    case 2:
        MapGaussianGradientsKernel<2><<<grid, 256>>>(gaussians.gaussians_, camera, gaussian_gradients_, count);
        break;
    case 3:
        MapGaussianGradientsKernel<3><<<grid, 256>>>(gaussians.gaussians_, camera, gaussian_gradients_, count);
        break;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool Solver::UpdateGaussians(
    GaussianBuffers& gaussians,
    float position_learning_rate) {
    int count = gaussians.gaussian_count_;
    if (count == 0) {
        return true;
    }
    if (count < 0 || count > gaussian_capacity_ || !gaussians.gaussians_ || !gaussian_gradients_ ||
        !first_moments_ || !second_moments_ || !std::isfinite(position_learning_rate) || position_learning_rate < 0) {
        return false;
    }
    uint64_t iteration = iteration_ + 1;
    float first_correction = static_cast<float>(1.0 / (1.0 - std::pow(0.9, static_cast<double>(iteration))));
    float second_correction = static_cast<float>(1.0 / (1.0 - std::pow(0.999, static_cast<double>(iteration))));
    dim3 grid((count + 255ll) / 256, kMaxParamCount);
    UpdateGaussiansKernel<<<grid, 256>>>(gaussian_gradients_,
        gaussians.gaussians_,
        first_moments_,
        second_moments_,
        count,
        position_learning_rate,
        first_correction,
        second_correction);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    iteration_ = iteration;
    return true;
}

}
