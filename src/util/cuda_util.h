#pragma once

#include <algorithm>
#include <cstdint>

namespace gs_native {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

inline int GetGridSize(
    int64_t count,
    int block_size) {
    return static_cast<int>(std::max<int64_t>((count + block_size - 1) / block_size, 1));
}

}
