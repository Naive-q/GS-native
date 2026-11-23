#pragma once

#include "core/types/gaussian.h"
#include "core/types/scene.h"

#include <string>
#include <vector>

namespace gs_native {

bool ReadColmapScene(
    const std::string& scene_dir,
    int downscale,
    Scene& scene);

bool WriteGaussiansPly(
    const std::string& path,
    const std::vector<Gaussian>& gaussians);

bool ReadGaussiansPly(
    const std::string& path,
    std::vector<Gaussian>& gaussians);

} // namespace gs_native
