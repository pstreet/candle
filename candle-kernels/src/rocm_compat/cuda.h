// ROCm/HIP compatibility header: map CUDA <cuda.h> (driver API) minimally onto
// HIP. candle's kernels only rely on the runtime device surface here.
#pragma once
#include <hip/hip_runtime.h>

// Host-side stream handle used by the statically linked moe/mmq kernels.
typedef hipStream_t cudaStream_t;
