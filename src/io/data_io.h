#pragma once

#include "types/gaussian.h"
#include "types/scene.h"

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

}
