// ROCm/HIP compatibility header: provide a CUDA-Thrust-compatible
// `thrust::cuda::par` execution-policy entry point for sources that use the
// `thrust::cuda::par.on(stream)` idiom (CUDA's <thrust/cuda/par.h>).
//
// ROCm's Thrust ships the device backend under `thrust::hip_rocprim::` (including
// `execute_on_stream`, which has an `.on(stream)` method), but it does not expose
// the public `thrust::cuda::par` object. We surface it here so the moe kernels keep
// compiling and run on the requested device stream.
#pragma once
#include <hip/hip_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/system/hip/detail/par.h>

#if defined(__HIP_PLATFORM_AMD__)
namespace thrust {
    namespace cuda {
        // Reuse THRUST's HIP device-backend stream-bound execution policy. Its
        // `.on` method sets the stream (overriding the global policy), so
        // `thrust::cuda::par.on(stream)` forwards to that stream.
        using par_type = thrust::hip_rocprim::execute_on_stream;
        static const par_type par{};
    } // namespace cuda
} // namespace thrust
#endif
