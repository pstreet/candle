// ROCm/HIP compatibility header: map CUDA <cub/block/block_radix_sort.cuh> onto hipCUB.
#pragma once
#include <hipcub/block/block_radix_sort.hpp>
namespace cub = ::hipcub;
