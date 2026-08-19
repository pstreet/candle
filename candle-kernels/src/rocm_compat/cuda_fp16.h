// ROCm/HIP compatibility header: map CUDA <cuda_fp16.h> onto HIP.
#pragma once
#include <hip/hip_fp16.h>

// candle's fp8 helpers call __half2float on a __half_raw produced by the fp8
// conversion intrinsic; HIP only provides the __half overload.
__device__ inline float __half2float(__half_raw hr) {
  return __half2float(__ushort_as_half(hr.x));
}
