// ROCm/HIP compatibility header: map CUDA <cub/cub.cuh> onto hipCUB.
#pragma once
#include <hipcub/hipcub.hpp>
namespace cub = ::hipcub;
