use crate::cublas::result as res;
use crate::cublas::result::CublasError;
use crate::cublas::sys;
use crate::driver::{DevicePtr, DevicePtrMut};
use std::ffi::c_int;
use std::os::raw::c_longlong;
use std::sync::Arc;

/// GEMM configuration, mirroring `cudarc::cublas::GemmConfig`.
#[derive(Debug, Copy, Clone)]
pub struct GemmConfig<T> {
    pub transa: sys::cublasOperation_t,
    pub transb: sys::cublasOperation_t,
    pub m: c_int,
    pub n: c_int,
    pub k: c_int,
    pub alpha: T,
    pub lda: c_int,
    pub ldb: c_int,
    pub beta: T,
    pub ldc: c_int,
}

/// Strided-batched GEMM configuration.
#[derive(Debug, Copy, Clone)]
pub struct StridedBatchedConfig<T> {
    pub gemm: GemmConfig<T>,
    pub batch_size: c_int,
    pub stride_a: c_longlong,
    pub stride_b: c_longlong,
    pub stride_c: c_longlong,
}

/// A cuBLAS-like handle bound to a stream, over hipBLAS.
#[derive(Debug)]
pub struct CudaBlas {
    pub(crate) handle: sys::cublasHandle_t,
    pub(crate) stream: Arc<crate::driver::CudaStream>,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let _ = self.stream.inner.ctx.bind_to_thread();
        let _ = unsafe { res::destroy(self.handle) };
    }
}

impl CudaBlas {
    /// Creates a new handle bound to `stream`.
    pub fn new(stream: Arc<crate::driver::CudaStream>) -> Result<Self, CublasError> {
        let ctx = &stream.inner.ctx;
        ctx.record_err(ctx.bind_to_thread());
        let mut handle: sys::cublasHandle_t = std::ptr::null_mut();
        unsafe {
            res::create(&mut handle)?;
            res::set_stream(handle, stream.inner.cu_stream)?;
        }
        Ok(Self { handle, stream })
    }

    /// The underlying handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// # Safety
    /// Sets the stream associated with this handle.
    pub unsafe fn set_stream(
        &mut self,
        stream: Arc<crate::driver::CudaStream>,
    ) -> Result<(), CublasError> {
        self.stream = stream.clone();
        res::set_stream(self.handle, stream.inner.cu_stream)
    }
}

/// Matrix-matrix multiplication interface, mirroring `cudarc::cublas::Gemm`.
pub trait Gemm<T> {
    /// Matrix-matrix multiplication. See the cuBLAS docs for the argument layout.
    ///
    /// # Safety
    /// Improper arguments may lead to invalid memory accesses.
    unsafe fn gemm<A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
        &self,
        cfg: GemmConfig<T>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError>;

    /// Strided-batched matrix-matrix multiplication.
    ///
    /// # Safety
    /// Improper arguments may lead to invalid memory accesses.
    unsafe fn gemm_strided_batched<A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
        &self,
        cfg: StridedBatchedConfig<T>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError>;
}

impl Gemm<f32> for CudaBlas {
    unsafe fn gemm<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        cfg: GemmConfig<f32>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.alpha;
        let beta: f32 = cfg.beta;
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_ex(
            self.handle,
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.lda,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.ldb,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.ldc,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }

    unsafe fn gemm_strided_batched<A: DevicePtr<f32>, B: DevicePtr<f32>, C: DevicePtrMut<f32>>(
        &self,
        cfg: StridedBatchedConfig<f32>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.gemm.alpha;
        let beta: f32 = cfg.gemm.beta;
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_strided_batched_ex(
            self.handle,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldb,
            cfg.stride_b,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

impl Gemm<f64> for CudaBlas {
    unsafe fn gemm<A: DevicePtr<f64>, B: DevicePtr<f64>, C: DevicePtrMut<f64>>(
        &self,
        cfg: GemmConfig<f64>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f64 = cfg.alpha;
        let beta: f64 = cfg.beta;
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_ex(
            self.handle,
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            (&alpha) as *const f64 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.lda,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.ldb,
            (&beta) as *const f64 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.ldc,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_64F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }

    unsafe fn gemm_strided_batched<A: DevicePtr<f64>, B: DevicePtr<f64>, C: DevicePtrMut<f64>>(
        &self,
        cfg: StridedBatchedConfig<f64>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f64 = cfg.gemm.alpha;
        let beta: f64 = cfg.gemm.beta;
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_strided_batched_ex(
            self.handle,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            (&alpha) as *const f64 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.gemm.ldb,
            cfg.stride_b,
            (&beta) as *const f64 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_64F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_64F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

impl Gemm<half::f16> for CudaBlas {
    unsafe fn gemm<A: DevicePtr<half::f16>, B: DevicePtr<half::f16>, C: DevicePtrMut<half::f16>>(
        &self,
        cfg: GemmConfig<half::f16>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.alpha.to_f32();
        let beta: f32 = cfg.beta.to_f32();
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_ex(
            self.handle,
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.lda,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.ldb,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.ldc,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }

    unsafe fn gemm_strided_batched<
        A: DevicePtr<half::f16>,
        B: DevicePtr<half::f16>,
        C: DevicePtrMut<half::f16>,
    >(
        &self,
        cfg: StridedBatchedConfig<half::f16>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.gemm.alpha.to_f32();
        let beta: f32 = cfg.gemm.beta.to_f32();
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_strided_batched_ex(
            self.handle,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldb,
            cfg.stride_b,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}

impl Gemm<half::bf16> for CudaBlas {
    unsafe fn gemm<
        A: DevicePtr<half::bf16>,
        B: DevicePtr<half::bf16>,
        C: DevicePtrMut<half::bf16>,
    >(
        &self,
        cfg: GemmConfig<half::bf16>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.alpha.to_f32();
        let beta: f32 = cfg.beta.to_f32();
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_ex(
            self.handle,
            cfg.transa,
            cfg.transb,
            cfg.m,
            cfg.n,
            cfg.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.lda,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.ldb,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.ldc,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }

    unsafe fn gemm_strided_batched<
        A: DevicePtr<half::bf16>,
        B: DevicePtr<half::bf16>,
        C: DevicePtrMut<half::bf16>,
    >(
        &self,
        cfg: StridedBatchedConfig<half::bf16>,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), CublasError> {
        let alpha: f32 = cfg.gemm.alpha.to_f32();
        let beta: f32 = cfg.gemm.beta.to_f32();
        let (a, _ra) = a.device_ptr(&self.stream);
        let (b, _rb) = b.device_ptr(&self.stream);
        let (c, _rc) = c.device_ptr_mut(&self.stream);
        res::gemm_strided_batched_ex(
            self.handle,
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            (&alpha) as *const f32 as *const _,
            a as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.lda,
            cfg.stride_a,
            b as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldb,
            cfg.stride_b,
            (&beta) as *const f32 as *const _,
            c as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
}
