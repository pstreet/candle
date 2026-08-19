// ROCm/HIP compatibility header for CUDA <mma.h> (wmma/mma).
//
// NVIDIA `nvcuda::wmma` is exposed on ROCm through the rocwmma library. Its API
// is nearly a drop-in for `nvcuda::wmma`: the same `fragment<matrix_a/
// matrix_b/accumulator, M, N, K, T, layout>` type, `fill_fragment`,
// `load_matrix_sync`, `store_matrix_sync` and `mem_row_major`/`mem_col_major`
// tags. The one naming difference is the matrix-multiply-accumulate entry point:
// CUDA calls it `matrix_sync_mul_acc`, ROCm calls it `mma_sync`.
//
// We re-export just the public wmma API surface into the `nvcuda::wmma`
// namespace rather than doing `using namespace ::rocwmma`, which would leak the
// whole rocwmma namespace (including the internal `VecT` alias template and all
// the `VReg*` types) into global scope and collide with the kernels' own local
// `using VecT = float4;`.
#pragma once
#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <rocwmma/rocwmma.hpp>
namespace nvcuda {
    namespace wmma {
        using ::rocwmma::fragment;
        using ::rocwmma::fill_fragment;
        using ::rocwmma::load_matrix_sync;
        using ::rocwmma::store_matrix_sync;
        using ::rocwmma::mma_sync;
        using ::rocwmma::matrix_a;
        using ::rocwmma::matrix_b;
        using ::rocwmma::accumulator;
        using ::rocwmma::row_major;
        using ::rocwmma::col_major;
        using ::rocwmma::mem_row_major;
        using ::rocwmma::mem_col_major;
        // CUDA's matrix-multiply-accumulate maps 1:1 onto rocwmma::mma_sync.
        template <class FragOut, class FragA, class FragB, class FragIn>
        ROCWMMA_DEVICE void matrix_sync_mul_acc(FragOut &c_out, const FragA &a, const FragB &b, FragIn &c_in) {
            ::rocwmma::mma_sync(c_out, a, b, c_in);
        }
    } // namespace wmma
} // namespace nvcuda
#endif
