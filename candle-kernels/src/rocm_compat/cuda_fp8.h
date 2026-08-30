// ROCm/HIP compatibility header: map CUDA <cuda_fp8.h> onto HIP.
#pragma once
#include <hip/hip_fp8.h>

#define __nv_fp8_e4m3 __hip_fp8_e4m3
#define __nv_fp8_e5m2 __hip_fp8_e5m2
#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2
#define __NV_SATFINITE __HIP_SATFINITE
#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_halfraw_to_fp8 __hip_cvt_halfraw_to_fp8
#define __nv_cvt_bfloat16raw_to_fp8 __hip_cvt_bfloat16raw_to_fp8
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
