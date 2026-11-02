//! cuBLASLt-compatible matrix routines over hipBLASLt.

pub mod result {
    use std::mem::MaybeUninit;
    use std::os::raw::{c_int, c_void};

    /// Error from a hipBLASLt call, mirroring `cudarc::cublaslt::result::CublasError`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CublasError(pub i32);

    impl std::fmt::Display for CublasError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
	    write!(f, "hipblasLt error {}", self.0)
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
	    *cublasltlib()
		.get::<F>(name.as_bytes())
		.unwrap_or_else(|e| panic!("Missing symbol {name}: {e}"))
	}
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn cublasltlib() -> &'static ::libloading::Library {
	static LIB: std::sync::OnceLock<::libloading::Library> = std::sync::OnceLock::new();
	LIB.get_or_init(|| {
	    let choices = crate::get_lib_name_candidates("hipblaslt");
	    for choice in choices.iter() {
		if let Ok(lib) = ::libloading::Library::new(choice) {
		    return lib;
		}
	    }
	    crate::panic_no_lib_found("hipblaslt", &choices);
	})
    }

    #[cfg(not(feature = "dynamic-loading"))]
    extern "C" {
	fn hipblas_lt_create(handle: *mut super::sys::cublasLtHandle_t) -> i32;
	fn hipblas_lt_destroy(handle: super::sys::cublasLtHandle_t) -> i32;
	fn hipblas_lt_matmul_desc_create(
	    op: *mut super::sys::cublasLtMatmulDesc_t,
	    compute_type: super::sys::cublasComputeType_t,
	    scale_type: super::sys::cudaDataType_t,
	) -> i32;
	fn hipblas_lt_matmul_desc_set_attribute(
	    op: super::sys::cublasLtMatmulDesc_t,
	    attr: super::sys::cublasLtMatmulDescAttributes_t,
	    buf: *const c_void,
	    size: usize,
	) -> i32;
	fn hipblas_lt_matmul_desc_destroy(op: super::sys::cublasLtMatmulDesc_t) -> i32;
	fn hipblas_lt_matrix_layout_create(
	    layout: *mut super::sys::cublasLtMatrixLayout_t,
	    dtype: super::sys::cudaDataType_t,
	    rows: u64,
	    cols: u64,
	    ld: i64,
	) -> i32;
	fn hipblas_lt_matrix_layout_set_attribute(
	    layout: super::sys::cublasLtMatrixLayout_t,
	    attr: super::sys::cublasLtMatrixLayoutAttribute_t,
	    buf: *const c_void,
	    size: usize,
	) -> i32;
	fn hipblas_lt_matrix_layout_destroy(layout: super::sys::cublasLtMatrixLayout_t) -> i32;
	fn hipblas_lt_matmul_preference_create(
	    pref: *mut super::sys::cublasLtMatmulPreference_t,
	) -> i32;
	fn hipblas_lt_matmul_preference_set_attribute(
	    pref: super::sys::cublasLtMatmulPreference_t,
	    attr: super::sys::cublasLtMatmulPreferenceAttributes_t,
	    buf: *const c_void,
	    size: usize,
	) -> i32;
	fn hipblas_lt_matmul_preference_destroy(pref: super::sys::cublasLtMatmulPreference_t) -> i32;
	fn hipblas_lt_matmul_algo_get_heuristic(
	    handle: super::sys::cublasLtHandle_t,
	    compute_desc: super::sys::cublasLtMatmulDesc_t,
	    a_layout: super::sys::cublasLtMatrixLayout_t,
	    b_layout: super::sys::cublasLtMatrixLayout_t,
	    c_layout: super::sys::cublasLtMatrixLayout_t,
	    d_layout: super::sys::cublasLtMatrixLayout_t,
	    pref: super::sys::cublasLtMatmulPreference_t,
	    requested: c_int,
	    heur: *mut super::sys::cublasLtMatmulHeuristicResult_t,
	    count: *mut c_int,
	) -> i32;
	fn hipblas_lt_matmul(
	    handle: super::sys::cublasLtHandle_t,
	    compute_desc: super::sys::cublasLtMatmulDesc_t,
	    alpha: *const c_void,
	    a: *const c_void,
	    a_layout: super::sys::cublasLtMatrixLayout_t,
	    b: *const c_void,
	    b_layout: super::sys::cublasLtMatrixLayout_t,
	    beta: *const c_void,
	    c: *const c_void,
	    c_layout: super::sys::cublasLtMatrixLayout_t,
	    d: *mut c_void,
	    d_layout: super::sys::cublasLtMatrixLayout_t,
	    algo: *const super::sys::cublasLtMatmulAlgo_t,
	    workspace: *mut c_void,
	    workspace_size: usize,
	    stream: crate::driver::sys::CUstream,
	) -> i32;
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_create(handle: *mut super::sys::cublasLtHandle_t) -> i32 {
	let hipblas_lt_create: extern "C" fn(*mut super::sys::cublasLtHandle_t) -> i32 =
	    load("hipblas_lt_create");
	hipblas_lt_create(handle)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_destroy(handle: super::sys::cublasLtHandle_t) -> i32 {
	let hipblas_lt_destroy: extern "C" fn(super::sys::cublasLtHandle_t) -> i32 =
	    load("hipblas_lt_destroy");
	hipblas_lt_destroy(handle)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_desc_create(
	op: *mut super::sys::cublasLtMatmulDesc_t,
	compute_type: super::sys::cublasComputeType_t,
	scale_type: super::sys::cudaDataType_t,
    ) -> i32 {
	let hipblas_lt_matmul_desc_create: extern "C" fn(
	    *mut super::sys::cublasLtMatmulDesc_t,
	    super::sys::cublasComputeType_t,
	    super::sys::cudaDataType_t,
	) -> i32 = load("hipblas_lt_matmul_desc_create");
	hipblas_lt_matmul_desc_create(op, compute_type, scale_type)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_desc_set_attribute(
	op: super::sys::cublasLtMatmulDesc_t,
	attr: super::sys::cublasLtMatmulDescAttributes_t,
	buf: *const c_void,
	size: usize,
    ) -> i32 {
	let hipblas_lt_matmul_desc_set_attribute: extern "C" fn(
	    super::sys::cublasLtMatmulDesc_t,
	    super::sys::cublasLtMatmulDescAttributes_t,
	    *const c_void,
	    usize,
	) -> i32 = load("hipblas_lt_matmul_desc_set_attribute");
	hipblas_lt_matmul_desc_set_attribute(op, attr, buf, size)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_desc_destroy(op: super::sys::cublasLtMatmulDesc_t) -> i32 {
	let hipblas_lt_matmul_desc_destroy: extern "C" fn(super::sys::cublasLtMatmulDesc_t) -> i32 =
	    load("hipblas_lt_matmul_desc_destroy");
	hipblas_lt_matmul_desc_destroy(op)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matrix_layout_create(
	layout: *mut super::sys::cublasLtMatrixLayout_t,
	dtype: super::sys::cudaDataType_t,
	rows: u64,
	cols: u64,
	ld: i64,
    ) -> i32 {
	let hipblas_lt_matrix_layout_create: extern "C" fn(
	    *mut super::sys::cublasLtMatrixLayout_t,
	    super::sys::cudaDataType_t,
	    u64,
	    u64,
	    i64,
	) -> i32 = load("hipblas_lt_matrix_layout_create");
	hipblas_lt_matrix_layout_create(layout, dtype, rows, cols, ld)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matrix_layout_set_attribute(
	layout: super::sys::cublasLtMatrixLayout_t,
	attr: super::sys::cublasLtMatrixLayoutAttribute_t,
	buf: *const c_void,
	size: usize,
    ) -> i32 {
	let hipblas_lt_matrix_layout_set_attribute: extern "C" fn(
	    super::sys::cublasLtMatrixLayout_t,
	    super::sys::cublasLtMatrixLayoutAttribute_t,
	    *const c_void,
	    usize,
	) -> i32 = load("hipblas_lt_matrix_layout_set_attribute");
	hipblas_lt_matrix_layout_set_attribute(layout, attr, buf, size)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matrix_layout_destroy(layout: super::sys::cublasLtMatrixLayout_t) -> i32 {
	let hipblas_lt_matrix_layout_destroy: extern "C" fn(super::sys::cublasLtMatrixLayout_t) -> i32 =
	    load("hipblas_lt_matrix_layout_destroy");
	hipblas_lt_matrix_layout_destroy(layout)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_preference_create(
	pref: *mut super::sys::cublasLtMatmulPreference_t,
    ) -> i32 {
	let hipblas_lt_matmul_preference_create: extern "C" fn(
	    *mut super::sys::cublasLtMatmulPreference_t,
	) -> i32 = load("hipblas_lt_matmul_preference_create");
	hipblas_lt_matmul_preference_create(pref)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_preference_set_attribute(
	pref: super::sys::cublasLtMatmulPreference_t,
	attr: super::sys::cublasLtMatmulPreferenceAttributes_t,
	buf: *const c_void,
	size: usize,
    ) -> i32 {
	let hipblas_lt_matmul_preference_set_attribute: extern "C" fn(
	    super::sys::cublasLtMatmulPreference_t,
	    super::sys::cublasLtMatmulPreferenceAttributes_t,
	    *const c_void,
	    usize,
	) -> i32 = load("hipblas_lt_matmul_preference_set_attribute");
	hipblas_lt_matmul_preference_set_attribute(pref, attr, buf, size)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_preference_destroy(
	pref: super::sys::cublasLtMatmulPreference_t,
    ) -> i32 {
	let hipblas_lt_matmul_preference_destroy: extern "C" fn(
	    super::sys::cublasLtMatmulPreference_t,
	) -> i32 = load("hipblas_lt_matmul_preference_destroy");
	hipblas_lt_matmul_preference_destroy(pref)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul_algo_get_heuristic(
	handle: super::sys::cublasLtHandle_t,
	compute_desc: super::sys::cublasLtMatmulDesc_t,
	a_layout: super::sys::cublasLtMatrixLayout_t,
	b_layout: super::sys::cublasLtMatrixLayout_t,
	c_layout: super::sys::cublasLtMatrixLayout_t,
	d_layout: super::sys::cublasLtMatrixLayout_t,
	pref: super::sys::cublasLtMatmulPreference_t,
	requested: c_int,
	heur: *mut super::sys::cublasLtMatmulHeuristicResult_t,
	count: *mut c_int,
    ) -> i32 {
	let hipblas_lt_matmul_algo_get_heuristic: extern "C" fn(
	    super::sys::cublasLtHandle_t,
	    super::sys::cublasLtMatmulDesc_t,
	    super::sys::cublasLtMatrixLayout_t,
	    super::sys::cublasLtMatrixLayout_t,
	    super::sys::cublasLtMatrixLayout_t,
	    super::sys::cublasLtMatrixLayout_t,
	    super::sys::cublasLtMatmulPreference_t,
	    c_int,
	    *mut super::sys::cublasLtMatmulHeuristicResult_t,
	    *mut c_int,
	) -> i32 = load("hipblas_lt_matmul_algo_get_heuristic");
	hipblas_lt_matmul_algo_get_heuristic(
	    handle,
	    compute_desc,
	    a_layout,
	    b_layout,
	    c_layout,
	    d_layout,
	    pref,
	    requested,
	    heur,
	    count,
	)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hipblas_lt_matmul(
	handle: super::sys::cublasLtHandle_t,
	compute_desc: super::sys::cublasLtMatmulDesc_t,
	alpha: *const c_void,
	a: *const c_void,
	a_layout: super::sys::cublasLtMatrixLayout_t,
	b: *const c_void,
	b_layout: super::sys::cublasLtMatrixLayout_t,
	beta: *const c_void,
	c: *const c_void,
	c_layout: super::sys::cublasLtMatrixLayout_t,
	d: *mut c_void,
	d_layout: super::sys::cublasLtMatrixLayout_t,
	algo: *const super::sys::cublasLtMatmulAlgo_t,
	workspace: *mut c_void,
	workspace_size: usize,
	stream: crate::driver::sys::CUstream,
    ) -> i32 {
	let hipblas_lt_matmul: extern "C" fn(
	    super::sys::cublasLtHandle_t,
	    super::sys::cublasLtMatmulDesc_t,
	    *const c_void,
	    *const c_void,
	    super::sys::cublasLtMatrixLayout_t,
	    *const c_void,
	    super::sys::cublasLtMatrixLayout_t,
	    *const c_void,
	    *const c_void,
	    super::sys::cublasLtMatrixLayout_t,
	    *mut c_void,
	    super::sys::cublasLtMatrixLayout_t,
	    *const super::sys::cublasLtMatmulAlgo_t,
	    *mut c_void,
	    usize,
	    crate::driver::sys::CUstream,
	) -> i32 = load("hipblas_lt_matmul");
	hipblas_lt_matmul(
	    handle,
	    compute_desc,
	    alpha,
	    a,
	    a_layout,
	    b,
	    b_layout,
	    beta,
	    c,
	    c_layout,
	    d,
	    d_layout,
	    algo,
	    workspace,
	    workspace_size,
	    stream,
	)
    }

    pub fn create_handle() -> Result<super::sys::cublasLtHandle_t, CublasError> {
	let mut handle = MaybeUninit::uninit();
	unsafe {
	    check(hipblas_lt_create(handle.as_mut_ptr()))?;
	    Ok(handle.assume_init())
	}
    }

    pub unsafe fn destroy_handle(handle: super::sys::cublasLtHandle_t) -> Result<(), CublasError> {
	check(hipblas_lt_destroy(handle))
    }

    pub fn create_matmul_desc(
	compute_type: super::sys::cublasComputeType_t,
	scale_type: super::sys::cudaDataType_t,
    ) -> Result<super::sys::cublasLtMatmulDesc_t, CublasError> {
	let mut op = MaybeUninit::uninit();
	unsafe {
	    check(hipblas_lt_matmul_desc_create(
		op.as_mut_ptr(),
		compute_type,
		scale_type,
	    ))?;
	    Ok(op.assume_init())
	}
    }

    pub unsafe fn set_matmul_desc_attribute(
	op: super::sys::cublasLtMatmulDesc_t,
	attr: super::sys::cublasLtMatmulDescAttributes_t,
	buf: *const c_void,
	size: usize,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matmul_desc_set_attribute(op, attr, buf, size))
    }

    pub unsafe fn destroy_matmul_desc(
	op: super::sys::cublasLtMatmulDesc_t,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matmul_desc_destroy(op))
    }

    pub fn create_matrix_layout(
	dtype: super::sys::cudaDataType_t,
	rows: u64,
	cols: u64,
	ld: i64,
    ) -> Result<super::sys::cublasLtMatrixLayout_t, CublasError> {
	let mut layout = MaybeUninit::uninit();
	unsafe {
	    check(hipblas_lt_matrix_layout_create(
		layout.as_mut_ptr(),
		dtype,
		rows,
		cols,
		ld,
	    ))?;
	    Ok(layout.assume_init())
	}
    }

    pub unsafe fn set_matrix_layout_attribute(
	layout: super::sys::cublasLtMatrixLayout_t,
	attr: super::sys::cublasLtMatrixLayoutAttribute_t,
	buf: *const c_void,
	size: usize,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matrix_layout_set_attribute(layout, attr, buf, size))
    }

    pub unsafe fn destroy_matrix_layout(
	layout: super::sys::cublasLtMatrixLayout_t,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matrix_layout_destroy(layout))
    }

    pub fn create_matmul_pref() -> Result<super::sys::cublasLtMatmulPreference_t, CublasError> {
	let mut pref = MaybeUninit::uninit();
	unsafe {
	    check(hipblas_lt_matmul_preference_create(pref.as_mut_ptr()))?;
	    Ok(pref.assume_init())
	}
    }

    pub unsafe fn set_matmul_pref_attribute(
	pref: super::sys::cublasLtMatmulPreference_t,
	attr: super::sys::cublasLtMatmulPreferenceAttributes_t,
	buf: *const c_void,
	size: usize,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matmul_preference_set_attribute(pref, attr, buf, size))
    }

    pub unsafe fn destroy_matmul_pref(
	pref: super::sys::cublasLtMatmulPreference_t,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matmul_preference_destroy(pref))
    }

    pub unsafe fn get_matmul_algo_heuristic(
	handle: super::sys::cublasLtHandle_t,
	compute_desc: super::sys::cublasLtMatmulDesc_t,
	a_layout: super::sys::cublasLtMatrixLayout_t,
	b_layout: super::sys::cublasLtMatrixLayout_t,
	c_layout: super::sys::cublasLtMatrixLayout_t,
	d_layout: super::sys::cublasLtMatrixLayout_t,
	pref: super::sys::cublasLtMatmulPreference_t,
    ) -> Result<super::sys::cublasLtMatmulHeuristicResult_t, CublasError> {
	let mut heur = MaybeUninit::uninit();
	let mut count = 0;
	check(hipblas_lt_matmul_algo_get_heuristic(
	    handle,
	    compute_desc,
	    a_layout,
	    b_layout,
	    c_layout,
	    d_layout,
	    pref,
	    1,
	    heur.as_mut_ptr(),
	    &mut count,
	))?;
	if count == 0 {
	    return Err(CublasError(-1));
	}
	Ok(heur.assume_init())
    }

    pub unsafe fn matmul(
	handle: super::sys::cublasLtHandle_t,
	compute_desc: super::sys::cublasLtMatmulDesc_t,
	alpha: *mut c_void,
	a: *mut c_void,
	a_layout: super::sys::cublasLtMatrixLayout_t,
	b: *mut c_void,
	b_layout: super::sys::cublasLtMatrixLayout_t,
	beta: *mut c_void,
	c: *mut c_void,
	c_layout: super::sys::cublasLtMatrixLayout_t,
	d: *mut c_void,
	d_layout: super::sys::cublasLtMatrixLayout_t,
	algo: *const super::sys::cublasLtMatmulAlgo_t,
	workspace: *mut c_void,
	workspace_size: usize,
	stream: crate::driver::sys::CUstream,
    ) -> Result<(), CublasError> {
	check(hipblas_lt_matmul(
	    handle,
	    compute_desc,
	    alpha,
	    a,
	    a_layout,
	    b,
	    b_layout,
	    beta,
	    c,
	    c_layout,
	    d,
	    d_layout,
	    algo,
	    workspace,
	    workspace_size,
	    stream,
	))
    }
}

pub mod sys {
    // FFI surface: C-ABI names mirroring cuBLASLt.
    #![allow(non_camel_case_types, non_snake_case)]
    use std::os::raw::c_void;

    pub type cublasLtHandle_t = *mut c_void;
    pub type cublasLtMatmulDesc_t = *mut c_void;
    pub type cublasLtMatmulPreference_t = *mut c_void;
    pub type cublasLtMatrixLayout_t = *mut c_void;

    // Reuse the cublas enums (values match the hipBLAS ABI).
    pub type cublasComputeType_t = crate::cublas::sys::cublasComputeType_t;
    pub type cudaDataType_t = crate::cublas::sys::cudaDataType_t;

    // Layout must match the C `hipblasLtMatmulAlgo_t`: uint8_t data[16] + size_t
    // max_workspace_bytes (24 bytes total).
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub struct cublasLtMatmulAlgo_t {
	pub data: [u8; 16],
	pub max_workspace_bytes: usize,
    }

    // Layout must match the C `hipblasLtMatmulHeuristicResult_t`.
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub struct cublasLtMatmulHeuristicResult_t {
	pub algo: cublasLtMatmulAlgo_t,
	pub workspace_size: usize,
	pub state: i32,
	pub waves_count: f32,
	pub reserved: [i32; 4],
    }

    // NOTE: hipBLASLt attribute values differ from CUDA cublasLt. These are the HIP values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasLtMatrixLayoutAttribute_t {
	CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0,
	CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasLtMatmulDescAttributes_t {
	CUBLASLT_MATMUL_DESC_TRANSA = 0,
	CUBLASLT_MATMUL_DESC_TRANSB = 1,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(i32)]
    pub enum cublasLtMatmulPreferenceAttributes_t {
	CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    }
}
