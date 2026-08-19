// ROCm/HIP compatibility header: map CUDA <cuda_runtime.h> onto HIP.
#pragma once
#include <hip/hip_runtime.h>
// Pre-include the fp16/bf16/fp8 detail headers so the __shfl_*_sync macros
// below are only active *after* HIP's own __shfl_*_sync declarations are parsed
// (otherwise the macros would corrupt those declarations).
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// `__CUDA_ARCH__` is a device-code arch tag used by the CUDA sources' feature
// gates (e.g. #if __CUDA_ARCH__ >= 750). Mirror CUDA: define it only on the
// device pass, leaving it undefined for host code so host/device dispatch in the
// sources works. gfx1151 (RDNA3) supports fp16/bf16/fp8; the non-Turing MMQ
// fallback is selected by passing CANDLE_ROCM_CUDA_ARCH=600.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  #ifndef CANDLE_ROCM_CUDA_ARCH
  #define CANDLE_ROCM_CUDA_ARCH 1030
  #endif
  #define __CUDA_ARCH__ CANDLE_ROCM_CUDA_ARCH
#endif

// BF16 type aliases: the CUDA <cuda_bf16.h> names for HIP's __hip_bfloat16.
// Defined here (not only in cuda_bf16.h) so any TU that only includes
// <cuda_runtime.h> still resolves nv_bfloat16 / __nv_bfloat16.
#ifndef __nv_bfloat16
#define __nv_bfloat16 __hip_bfloat16
#define __nv_bfloat162 __hip_bfloat162
#endif
#ifndef nv_bfloat16
#define nv_bfloat16 __hip_bfloat16
#define nv_bfloat162 __hip_bfloat162
#endif

// CUDA DP4A (byte-wise dot product with carry); not provided by HIP. Matches
// the manual fallback used elsewhere in the kernels.
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
  const int8_t *a8 = (const int8_t *) & a;
  const int8_t *b8 = (const int8_t *) & b;
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

// Byte-wise (4x8-bit) unsigned subtract with no cross-byte borrow, as CUDA's
// __vsub4. Each result byte is (a[i] - b[i]) reduced mod 256.
__device__ __forceinline__ int __vsub4(int a, int b) {
  const unsigned char *u8a = (const unsigned char *) & a;
  const unsigned char *u8b = (const unsigned char *) & b;
  int r = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r |= (int)(u8a[i] - u8b[i]) << (8 * i);
  }
  return r;
}

// Byte-wise not-equal compare, as CUDA's __vcmpne4: each result byte is 0xFF
// where the two input bytes differ, 0x00 where they are equal.
__device__ __forceinline__ int __vcmpne4(int a, int b) {
  const unsigned char *u8a = (const unsigned char *) & a;
  const unsigned char *u8b = (const unsigned char *) & b;
  int r = 0;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    r |= (u8a[i] != u8b[i]) ? (0xFF << (8 * i)) : 0;
  }
  return r;
}

// Device trap: CUDA's __trap() raises an asynchronous exception / hard fault.
// A hardware trap instruction is the closest analogue and never returns. It may
// also be reached through host error paths, so mark it for both.
__host__ __device__ __forceinline__ void __trap() { __builtin_trap(); }

// Packed 8-bit SIMD subtract (CUDA __vsubss4), used by the quantized mmq
// kernels to dequant 4-bit/8-bit values. Computed per byte with no borrow.
__device__ __forceinline__ int __vsubss4(int a, int b) {
  const unsigned ua = (unsigned) a;
  const unsigned ub = (unsigned) b;
  int r = 0;
  r |= (int) (((ua >>  0) & 0xFF) - ((ub >>  0) & 0xFF)) & 0xFF;
  r |= (int) ((((ua >>  8) & 0xFF) - ((ub >>  8) & 0xFF)) & 0xFF) <<  8;
  r |= (int) ((((ua >> 16) & 0xFF) - ((ub >> 16) & 0xFF)) & 0xFF) << 16;
  r |= (int) ((((ua >> 24) & 0xFF) - ((ub >> 24) & 0xFF)) & 0xFF) << 24;
  return r;
}

// HIP requires a 64-bit mask for the *_sync warp primitives, but the CUDA
// sources pass a 32-bit all-lanes mask. Promote the mask in a single self
// expansion to HIP's expected width (numerically identical).
#define __shfl_sync(M, ...)   __shfl_sync((unsigned long long)(M), __VA_ARGS__)
#define __shfl_up_sync(M, ...)   __shfl_up_sync((unsigned long long)(M), __VA_ARGS__)
#define __shfl_down_sync(M, ...) __shfl_down_sync((unsigned long long)(M), __VA_ARGS__)
#define __shfl_xor_sync(M, ...)  __shfl_xor_sync((unsigned long long)(M), __VA_ARGS__)
#define __ballot_sync(M, ...)    __ballot_sync((unsigned long long)(M), __VA_ARGS__)
#define __any_sync(M, ...)       __any_sync((unsigned long long)(M), __VA_ARGS__)
#define __all_sync(M, ...)       __all_sync((unsigned long long)(M), __VA_ARGS__)

// Map the CUDA host/runtime API the moe/mmq/mmvq launcher wrappers call onto the
// equivalent HIP entry points.
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAlloc hipHostMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemsetD32Async hipMemsetD32Async
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamQuery hipStreamQuery
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
// Cast the kernel symbol so a `__global__` function coerces to `const void*`
// (HIP does not perform the function->pointer->void* decay implicitly here).
#define cudaFuncSetAttribute(func, attr, val) \
    hipFuncSetAttribute((const void *) (func), (attr), (val))
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaDeviceProp hipDeviceProp
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaMemcpyKind hipMemcpyKind
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaSuccess hipSuccess
#define cudaHostAllocDefault hipHostAllocDefault
