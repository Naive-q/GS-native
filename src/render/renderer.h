#pragma once

#include "types/buffers.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace gs_native {

class Renderer {
public:
    Renderer() = default;
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    bool Render(
        const GaussianBuffers& gaussians,
        const CameraParams& camera,
        const float* background,
        float* image,
        int sh_degree);

private:
    friend class Solver;

    bool ProjectGaussians(
        const GaussianBuffers& gaussians,
        const CameraParams& camera,
        int sh_degree);

    bool BuildTileLists(
        const GaussianBuffers& gaussians,
        const CameraParams& camera);

    bool RenderTiles(
        const CameraParams& camera,
        const float* background,
        float* image);

    float2* means_ = nullptr;
    float4* conic_opacities_ = nullptr;
    float3* colors_ = nullptr;
    float* depths_ = nullptr;
    int4* tile_rects_ = nullptr;
    int* record_offsets_ = nullptr;
    int gaussian_capacity_ = 0;

    TileListBuffers tile_lists_;
    int tile_capacity_ = 0;

    uint64_t* keys_ = nullptr;
    uint64_t* sorted_keys_ = nullptr;
    int* gaussian_ids_ = nullptr;
    int record_capacity_ = 0;
    void* temp_storage_ = nullptr;
    size_t temp_storage_bytes_ = 0;
};

}
