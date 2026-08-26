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

    extern "C" {
        fn hipblasLtCreate(handle: *mut super::sys::cublasLtHandle_t) -> i32;
        fn hipblasLtDestroy(handle: super::sys::cublasLtHandle_t) -> i32;
        fn hipblasLtMatmulDescCreate(
            op: *mut super::sys::cublasLtMatmulDesc_t,
            compute_type: super::sys::cublasComputeType_t,
            scale_type: super::sys::cudaDataType_t,
        ) -> i32;
        fn hipblasLtMatmulDescSetAttribute(
            op: super::sys::cublasLtMatmulDesc_t,
            attr: super::sys::cublasLtMatmulDescAttributes_t,
            buf: *const c_void,
            size: usize,
        ) -> i32;
        fn hipblasLtMatmulDescDestroy(op: super::sys::cublasLtMatmulDesc_t) -> i32;
        fn hipblasLtMatrixLayoutCreate(
            layout: *mut super::sys::cublasLtMatrixLayout_t,
            dtype: super::sys::cudaDataType_t,
            rows: u64,
            cols: u64,
            ld: i64,
        ) -> i32;
        fn hipblasLtMatrixLayoutSetAttribute(
            layout: super::sys::cublasLtMatrixLayout_t,
            attr: super::sys::cublasLtMatrixLayoutAttribute_t,
            buf: *const c_void,
            size: usize,
        ) -> i32;
        fn hipblasLtMatrixLayoutDestroy(layout: super::sys::cublasLtMatrixLayout_t) -> i32;
        fn hipblasLtMatmulPreferenceCreate(
            pref: *mut super::sys::cublasLtMatmulPreference_t,
        ) -> i32;
        fn hipblasLtMatmulPreferenceSetAttribute(
            pref: super::sys::cublasLtMatmulPreference_t,
            attr: super::sys::cublasLtMatmulPreferenceAttributes_t,
            buf: *const c_void,
            size: usize,
        ) -> i32;
        fn hipblasLtMatmulPreferenceDestroy(pref: super::sys::cublasLtMatmulPreference_t) -> i32;
        fn hipblasLtMatmulAlgoGetHeuristic(
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
        fn hipblasLtMatmul(
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

    pub fn create_handle() -> Result<super::sys::cublasLtHandle_t, CublasError> {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            check(hipblasLtCreate(handle.as_mut_ptr()))?;
            Ok(handle.assume_init())
        }
    }

    pub unsafe fn destroy_handle(handle: super::sys::cublasLtHandle_t) -> Result<(), CublasError> {
        check(hipblasLtDestroy(handle))
    }

    pub fn create_matmul_desc(
        compute_type: super::sys::cublasComputeType_t,
        scale_type: super::sys::cudaDataType_t,
    ) -> Result<super::sys::cublasLtMatmulDesc_t, CublasError> {
        let mut op = MaybeUninit::uninit();
        unsafe {
            check(hipblasLtMatmulDescCreate(
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
        check(hipblasLtMatmulDescSetAttribute(op, attr, buf, size))
    }

    pub unsafe fn destroy_matmul_desc(
        op: super::sys::cublasLtMatmulDesc_t,
    ) -> Result<(), CublasError> {
        check(hipblasLtMatmulDescDestroy(op))
    }

    pub fn create_matrix_layout(
        dtype: super::sys::cudaDataType_t,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> Result<super::sys::cublasLtMatrixLayout_t, CublasError> {
        let mut layout = MaybeUninit::uninit();
        unsafe {
            check(hipblasLtMatrixLayoutCreate(
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
        check(hipblasLtMatrixLayoutSetAttribute(layout, attr, buf, size))
    }

    pub unsafe fn destroy_matrix_layout(
        layout: super::sys::cublasLtMatrixLayout_t,
    ) -> Result<(), CublasError> {
        check(hipblasLtMatrixLayoutDestroy(layout))
    }

    pub fn create_matmul_pref() -> Result<super::sys::cublasLtMatmulPreference_t, CublasError> {
        let mut pref = MaybeUninit::uninit();
        unsafe {
            check(hipblasLtMatmulPreferenceCreate(pref.as_mut_ptr()))?;
            Ok(pref.assume_init())
        }
    }

    pub unsafe fn set_matmul_pref_attribute(
        pref: super::sys::cublasLtMatmulPreference_t,
        attr: super::sys::cublasLtMatmulPreferenceAttributes_t,
        buf: *const c_void,
        size: usize,
    ) -> Result<(), CublasError> {
        check(hipblasLtMatmulPreferenceSetAttribute(pref, attr, buf, size))
    }

    pub unsafe fn destroy_matmul_pref(
        pref: super::sys::cublasLtMatmulPreference_t,
    ) -> Result<(), CublasError> {
        check(hipblasLtMatmulPreferenceDestroy(pref))
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
        check(hipblasLtMatmulAlgoGetHeuristic(
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
        check(hipblasLtMatmul(
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
