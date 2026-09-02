// ROCm/HIP compatibility header: CUDA <cuda_pipeline.h> has no HIP equivalent.
// gfx1250 (RDNA4.5) has a true async global->LDS copy (the cp.async equivalent);
// other targets (incl. gfx1151/RDNA3) fall back to a synchronous copy. commit is
// a no-op; wait_prior flushes the outstanding async LDS loads where supported.
#pragma once
#include <cstddef>

#if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx1250__) && __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
#define GDN_ROCM_ASYNC_LDS 1
#else
#define GDN_ROCM_ASYNC_LDS 0
#endif
#endif

#if GDN_ROCM_ASYNC_LDS
typedef int gdn_rocm_vint4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void gdn_rocm_async_load_b128(void *dst, const void *src) {
  __builtin_amdgcn_global_load_async_to_lds_b128(
      ((__attribute__((address_space(1))) gdn_rocm_vint4*)static_cast<const char *>(src)),
      ((__attribute__((address_space(3))) gdn_rocm_vint4*)static_cast<char *>(dst)), 0, 0);
}
__device__ __forceinline__ void gdn_rocm_wait_async_lds() {
  __builtin_amdgcn_s_wait_asynccnt(0);
}
__device__ inline void __pipeline_memcpy_async(void *dst, const void *src, size_t size) {
  for (size_t i = 0; i < size; i += 16) {
    gdn_rocm_async_load_b128(static_cast<char *>(dst) + i, static_cast<const char *>(src) + i);
  }
}
__device__ inline void __pipeline_commit() {}
__device__ inline void __pipeline_wait_prior(int) { gdn_rocm_wait_async_lds(); }

#elif defined(USE_ROCM)
// No async global->LDS on this target: a synchronous copy is always complete by
// the time it returns, so commit/wait are no-ops and correctness relies on the
// __syncthreads() the kernels already issue after the wait.
__device__ __forceinline__ void gdn_rocm_async_load_b128(void *dst, const void *src) {
  *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(src);
}
__device__ __forceinline__ void gdn_rocm_wait_async_lds() {}
__device__ inline void __pipeline_memcpy_async(void *dst, const void *src, size_t size) {
  char *d = static_cast<char *>(dst);
  const char *s = static_cast<const char *>(src);
  for (size_t i = 0; i + 16 <= size; i += 16) {
    *reinterpret_cast<float4 *>(d + i) = *reinterpret_cast<const float4 *>(s + i);
  }
  for (size_t i = size & ~size_t(15); i < size; i++) d[i] = s[i];
}
__device__ inline void __pipeline_commit() {}
__device__ inline void __pipeline_wait_prior(int) {}
#endif
