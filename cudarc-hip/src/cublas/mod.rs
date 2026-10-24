//! cuBLAS-compatible matrix routines over hipBLAS.

pub mod result {
    use std::os::raw::c_int;

    /// Error from a hipBLAS call, mirroring `cudarc::cublas::result::CublasError`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CublasError(pub i32);

    impl std::fmt::Display for CublasError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
	    write!(f, "hipblas error {}", self.0)
	}
    }

    impl std::error::Error for CublasError {}

    pub unsafe fn check(err: i32) -> Result<(), CublasError> {
	// HIPBLAS_STATUS_SUCCESS == 0
	if err == 0 {
	    Ok(())
	} else {
	    Err(CublasError(err))
	}
    }

    #[cfg(feature = "dynamic-loading")]
    fn load<F: Copy>(name: &str) -> F {
	// SAFETY: loading a symbol is not itself unsafe; the `unsafe fn`
	// wrappers below own the safety contract of each call.
	unsafe {
	    *cublaslib()
		.get::<F>(name.as_bytes())
		.unwrap_or_else(|e| panic!("Missing symbol {name}: {e}"))
	}
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn cublaslib() -> &'static ::libloading::Library {
	static LIB: std::sync::OnceLock<::libloading::Library> = std::sync::OnceLock::new();
	LIB.get_or_init(|| {
	    let choices = crate::get_lib_name_candidates("hipblas");
	    for choice in choices.iter() {
		if let Ok(lib) = ::libloading::Library::new(choice) {
		    return lib;
		}
	    }
	    crate::panic_no_lib_found("hipblas", &choices);
	})
    }

    #[cfg(not(feature = "dynamic-loading"))]
    extern "C" {
	fn hipblas_create(handle: *mut *mut super::sys::cublasHandle_t) -> i32;
	fn hipblas_destroy(handle: super::sys::cublasHandle_t) -> i32;
	fn hipblas_set_stream(
	    handle: super::sys::cublasHandle_t,
	    stream: super::sys::CUstream,
	) -> i32;
	fn hipblas_gemm_ex(
	    handle: super::sys::cublasHandle_t,
	    transa: super::sys::cublasOperation_t,
	    transb: super::sys::cublasOperation_t,
	    m: c_int,
	    n: c_int,
	    k: c_int,
	    alpha: *const core::ffi::c_void,
	    a: *const core::ffi::c_void,
	    a_type: super::sys::cudaDataType_t,
	    lda: c_int,
	    b: *const core::ffi::c_void,
	    b_type: super::sys::cudaDataType_t,
	    ldb: c_int,
	    beta: *const core::ffi::c_void,
	    c: *mut core::ffi::c_void,
	    c_type: super::sys::cudaDataType_t,
	    ldc: c_int,
	    compute_type: super::sys::cublasComputeType_t,
	    algo: super::sys::cublasGemmAlgo_t,
	) -> i32;
	fn hipblas_gemm_strided_batched_ex(
	    handle: super::sys::cublasHandle_t,
	    transa: super::sys::cublasOperation_t,
	    transb: super::sys::cublasOperation_t,
	    m: c_int,
	    n: c_int,
	    k: c_int,
	    alpha: *const core::ffi::c_void,
	    a: *const core::ffi::c_void,
	    a_type: super::sys::cudaDataType_t,
	    lda: c_int,
	    stride_a: i64,
	    b: *const core::ffi::c_void,
	    b_type: super::sys::cudaDataType_t,
	    ldb: c_int,
	    stride_b: i64,
	    beta: *const core::ffi::c_void,
	    c: *mut core::ffi::c_void,
	    c_type: super::sys::cudaDataType_t,
	    ldc: c_int,
	    stride_c: i64,
	    batch_count: c_int,
	    compute_type: super::sys::cublasComputeType_t,
	    algo: super::sys::cublasGemmAlgo_t,
	) -> i32;
	fn hipblas_hgemm(
	    handle: super::sys::cublasHandle_t,
	    transa: super::sys::cublasOperation_t,
	    transb: super::sys::cublasOperation_t,
	    m: c_int,
	    n: c_int,
	    k: c_int,
	    alpha: *const half::f16,
	    a: *const half::f16,
	    lda: c_int,
	    b: *const half::f16,
	    ldb: c_int,
	    beta: *const half::f16,
	    c: *mut half::f16,
	    ldc: c_int,
	) -> i32;
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_create(handle: *mut *mut super::sys::cublasHandle_t) -> i32 {
	let hipblas_create: extern "C" fn(*mut *mut super::sys::cublasHandle_t) -> i32 =
	    load("hipblas_create");
	hipblas_create(handle)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_destroy(handle: super::sys::cublasHandle_t) -> i32 {
	let hipblas_destroy: extern "C" fn(super::sys::cublasHandle_t) -> i32 =
	    load("hipblas_destroy");
	hipblas_destroy(handle)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_set_stream(
	handle: super::sys::cublasHandle_t,
	stream: super::sys::CUstream,
    ) -> i32 {
	let hipblas_set_stream: extern "C" fn(
	    super::sys::cublasHandle_t,
	    super::sys::CUstream,
	) -> i32 = load("hipblas_set_stream");
	hipblas_set_stream(handle, stream)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_gemm_ex(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const core::ffi::c_void,
	a: *const core::ffi::c_void,
	a_type: super::sys::cudaDataType_t,
	lda: c_int,
	b: *const core::ffi::c_void,
	b_type: super::sys::cudaDataType_t,
	ldb: c_int,
	beta: *const core::ffi::c_void,
	c: *mut core::ffi::c_void,
	c_type: super::sys::cudaDataType_t,
	ldc: c_int,
	compute_type: super::sys::cublasComputeType_t,
	algo: super::sys::cublasGemmAlgo_t,
    ) -> i32 {
	let hipblas_gemm_ex: extern "C" fn(
	    super::sys::cublasHandle_t,
	    super::sys::cublasOperation_t,
	    super::sys::cublasOperation_t,
	    c_int,
	    c_int,
	    c_int,
	    *const core::ffi::c_void,
	    *const core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    *const core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    *const core::ffi::c_void,
	    *mut core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    super::sys::cublasComputeType_t,
	    super::sys::cublasGemmAlgo_t,
	) -> i32 = load("hipblas_gemm_ex");
	hipblas_gemm_ex(
	    handle,
	    transa,
	    transb,
	    m,
	    n,
	    k,
	    alpha,
	    a,
	    a_type,
	    lda,
	    b,
	    b_type,
	    ldb,
	    beta,
	    c,
	    c_type,
	    ldc,
	    compute_type,
	    algo,
	)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_gemm_strided_batched_ex(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const core::ffi::c_void,
	a: *const core::ffi::c_void,
	a_type: super::sys::cudaDataType_t,
	lda: c_int,
	stride_a: i64,
	b: *const core::ffi::c_void,
	b_type: super::sys::cudaDataType_t,
	ldb: c_int,
	stride_b: i64,
	beta: *const core::ffi::c_void,
	c: *mut core::ffi::c_void,
	c_type: super::sys::cudaDataType_t,
	ldc: c_int,
	stride_c: i64,
	batch_count: c_int,
	compute_type: super::sys::cublasComputeType_t,
	algo: super::sys::cublasGemmAlgo_t,
    ) -> i32 {
	let hipblas_gemm_strided_batched_ex: extern "C" fn(
	    super::sys::cublasHandle_t,
	    super::sys::cublasOperation_t,
	    super::sys::cublasOperation_t,
	    c_int,
	    c_int,
	    c_int,
	    *const core::ffi::c_void,
	    *const core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    i64,
	    *const core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    i64,
	    *const core::ffi::c_void,
	    *mut core::ffi::c_void,
	    super::sys::cudaDataType_t,
	    c_int,
	    i64,
	    c_int,
	    super::sys::cublasComputeType_t,
	    super::sys::cublasGemmAlgo_t,
	) -> i32 = load("hipblas_gemm_strided_batched_ex");
	hipblas_gemm_strided_batched_ex(
	    handle,
	    transa,
	    transb,
	    m,
	    n,
	    k,
	    alpha,
	    a,
	    a_type,
	    lda,
	    stride_a,
	    b,
	    b_type,
	    ldb,
	    stride_b,
	    beta,
	    c,
	    c_type,
	    ldc,
	    stride_c,
	    batch_count,
	    compute_type,
	    algo,
	)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_hgemm(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const half::f16,
	a: *const half::f16,
	lda: c_int,
	b: *const half::f16,
	ldb: c_int,
	beta: *const half::f16,
	c: *mut half::f16,
	ldc: c_int,
    ) -> i32 {
	let hipblas_hgemm: extern "C" fn(
	    super::sys::cublasHandle_t,
	    super::sys::cublasOperation_t,
	    super::sys::cublasOperation_t,
	    c_int,
	    c_int,
	    c_int,
	    *const half::f16,
	    *const half::f16,
	    c_int,
	    *const half::f16,
	    c_int,
	    *const half::f16,
	    *mut half::f16,
	    c_int,
	) -> i32 = load("hipblas_hgemm");
	hipblas_hgemm(
	    handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
	)
    }

    pub unsafe fn create(handle: *mut super::sys::cublasHandle_t) -> Result<(), CublasError> {
	check(hipblas_create(handle as _))
    }

    pub unsafe fn destroy(handle: super::sys::cublasHandle_t) -> Result<(), CublasError> {
	check(hipblas_destroy(handle))
    }

    pub unsafe fn set_stream(
	handle: super::sys::cublasHandle_t,
	stream: super::sys::CUstream,
    ) -> Result<(), CublasError> {
	check(hipblas_set_stream(handle, stream))
    }

    pub unsafe fn gemm_ex(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const core::ffi::c_void,
	a: *const core::ffi::c_void,
	a_type: super::sys::cudaDataType_t,
	lda: c_int,
	b: *const core::ffi::c_void,
	b_type: super::sys::cudaDataType_t,
	ldb: c_int,
	beta: *const core::ffi::c_void,
	c: *mut core::ffi::c_void,
	c_type: super::sys::cudaDataType_t,
	ldc: c_int,
	compute_type: super::sys::cublasComputeType_t,
	algo: super::sys::cublasGemmAlgo_t,
    ) -> Result<(), CublasError> {
	check(hipblas_gemm_ex(
	    handle,
	    transa,
	    transb,
	    m,
	    n,
	    k,
	    alpha,
	    a,
	    a_type,
	    lda,
	    b,
	    b_type,
	    ldb,
	    beta,
	    c,
	    c_type,
	    ldc,
	    compute_type,
	    algo,
	))
    }

    pub unsafe fn gemm_strided_batched_ex(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const core::ffi::c_void,
	a: *const core::ffi::c_void,
	a_type: super::sys::cudaDataType_t,
	lda: c_int,
	stride_a: i64,
	b: *const core::ffi::c_void,
	b_type: super::sys::cudaDataType_t,
	ldb: c_int,
	stride_b: i64,
	beta: *const core::ffi::c_void,
	c: *mut core::ffi::c_void,
	c_type: super::sys::cudaDataType_t,
	ldc: c_int,
	stride_c: i64,
	batch_count: c_int,
	compute_type: super::sys::cublasComputeType_t,
	algo: super::sys::cublasGemmAlgo_t,
    ) -> Result<(), CublasError> {
	check(hipblas_gemm_strided_batched_ex(
	    handle,
	    transa,
	    transb,
	    m,
	    n,
	    k,
	    alpha,
	    a,
	    a_type,
	    lda,
	    stride_a,
	    b,
	    b_type,
	    ldb,
	    stride_b,
	    beta,
	    c,
	    c_type,
	    ldc,
	    stride_c,
	    batch_count,
	    compute_type,
	    algo,
	))
    }

    pub unsafe fn hgemm(
	handle: super::sys::cublasHandle_t,
	transa: super::sys::cublasOperation_t,
	transb: super::sys::cublasOperation_t,
	m: c_int,
	n: c_int,
	k: c_int,
	alpha: *const half::f16,
	a: *const half::f16,
	lda: c_int,
	b: *const half::f16,
	ldb: c_int,
	beta: *const half::f16,
	c: *mut half::f16,
	ldc: c_int,
    ) -> Result<(), CublasError> {
	check(hipblas_hgemm(
	    handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
	))
    }
}

pub mod sys {
    //! cuBLAS-compatible enumerations (values matching the CUDA/cuBLAS ABI where
    //! hipBLAS agrees, otherwise mapped from the hipBLAS constants).
    // FFI surface: C-ABI names mirroring cuBLAS.
    #![allow(non_camel_case_types, non_snake_case)]
    use std::os::raw::c_void;

    pub type cublasHandle_t = *mut c_void;
    pub type CUstream = crate::driver::sys::CUstream;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasOperation_t {
	CUBLAS_OP_N = 111,
	CUBLAS_OP_T = 112,
	CUBLAS_OP_C = 113,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasComputeType_t {
	CUBLAS_COMPUTE_16F = 0,
	CUBLAS_COMPUTE_16F_PEDANTIC = 1,
	CUBLAS_COMPUTE_32F = 2,
	CUBLAS_COMPUTE_32F_PEDANTIC = 3,
	CUBLAS_COMPUTE_32F_FAST_16F = 4,
	CUBLAS_COMPUTE_32F_FAST_16BF = 5,
	CUBLAS_COMPUTE_32F_FAST_TF32 = 6,
	CUBLAS_COMPUTE_64F = 7,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasGemmAlgo_t {
	CUBLAS_GEMM_DEFAULT = 160,
    }

    impl cublasGemmAlgo_t {
	pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: cublasGemmAlgo_t =
	    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT;
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cudaDataType_t {
	CUDA_R_32F = 0,
	CUDA_R_64F = 1,
	CUDA_R_16F = 2,
	CUDA_R_8I = 3,
	CUDA_R_16BF = 14,
    }
}

mod safe;
pub use safe::*;
