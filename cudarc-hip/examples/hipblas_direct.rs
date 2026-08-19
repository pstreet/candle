use cudarc_hip::cublas::result;
use cudarc_hip::cublas::sys;
use cudarc_hip::cublas::CudaBlas;
use cudarc_hip::driver::{CudaContext, DevicePtr, DevicePtrMut, DeviceRepr};

/// Minimal smoke test: one `CudaStream`, one `CudaBlas`, a strided-batched GEMM
/// and then full teardown of every slice/stream/module. Regression guard for the
/// ROCm double-`hipStreamDestroy` heap-corruption bug (each slice shares the same
/// underlying stream, so the stream must be destroyed exactly once).
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let blas = CudaBlas::new(stream.clone())?;

    let a = unsafe { stream.alloc::<f32>(3)? };
    let b = unsafe { stream.alloc::<f32>(18)? };
    let mut c = unsafe { stream.alloc_zeros::<f32>(6)? };

    let (a_ptr, _ga) = a.device_ptr(&stream);
    let (b_ptr, _gb) = b.device_ptr(&stream);
    let (c_ptr, _gc) = c.device_ptr_mut(&stream);

    let alpha = 1.0f32;
    let beta = 0.0f32;
    unsafe {
        result::gemm_strided_batched_ex(
            *blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            1,
            6,
            3,
            &alpha as *const f32 as *const _,
            a_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            1,
            3,
            b_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            3,
            18,
            &beta as *const f32 as *const _,
            c_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            1,
            6,
            1,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }
    stream.synchronize()?;
    unsafe {
        if cudarc_hip::driver::sys::hipDeviceSynchronize() != 0 {
            eprintln!("device sync failed");
        }
    }
    eprintln!("hipblas_direct: gemm ok, tearing down");
    Ok(())
}
