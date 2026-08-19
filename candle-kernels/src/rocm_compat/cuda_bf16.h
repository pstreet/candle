// ROCm/HIP compatibility header: map CUDA <cuda_bf16.h> onto HIP.
#pragma once
#include <hip/hip_bf16.h>

#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
// Some sources use the non-mangled alias.
#define nv_bfloat16 __hip_bfloat16
#define nv_bfloat162 __hip_bfloat162

// HIP's amd_hip_fp16 only provides the __half overloads of these; the CUDA
// sources call the __nv_bfloat16 variants. Implement them via float.
__device__ inline __nv_bfloat16 __hmax_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
  const float fa = __bfloat162float(a);
  const float fb = __bfloat162float(b);
  const float r = (fa != fa) ? fa : ((fb != fb) ? fb : fmaxf(fa, fb));
  return __float2bfloat16(r);
}

__device__ inline __nv_bfloat16 __hmin_nan(__nv_bfloat16 a, __nv_bfloat16 b) {
  const float fa = __bfloat162float(a);
  const float fb = __bfloat162float(b);
  const float r = (fa != fa) ? fa : ((fb != fb) ? fb : fminf(fa, fb));
  return __float2bfloat16(r);
}
