//! CUDA-compatible API surface over the HIP (ROCm) runtime.
//!
//! This crate re-implements the subset of the `cudarc` API that candle uses, on
//! top of the HIP runtime, so that candle's `cuda_backend` can run on AMD GPUs.

pub mod cublas;
pub mod cublaslt;
pub mod curand;
pub mod driver;
pub mod nvrtc;

/// Candidate shared-object names for a HIP library, mirroring
/// `cudarc::get_lib_name_candidates`. HIP sonames are stable across ROCm
/// releases, so unlike CUDA's per-driver-version zoo this is a short list:
/// the unversioned symlink plus the common major versions, so
/// versioned-only installs resolve too.
#[cfg(feature = "dynamic-loading")]
pub(crate) fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    let majors: &[&str] = match lib_name {
        "amdhip64" => &["", ".7", ".6"],
        "hipblas" => &["", ".3", ".1", ".0"],
        "hipblaslt" => &["", ".1", ".0"],
        "hiprand" => &["", ".1", ".0"],
        "hiprtc" => &["", ".7", ".1", ".0"],
        _ => &[""],
    };
    majors
        .iter()
        .map(|m| format!("lib{lib_name}.so{m}"))
        .collect()
}

#[cfg(feature = "dynamic-loading")]
pub(crate) fn panic_no_lib_found<S: std::fmt::Debug>(lib_name: &str, choices: &[S]) -> ! {
    panic!(
        "Unable to dynamically load the \"{lib_name}\" shared library - searched for library names: {choices:?}. Ensure LD_LIBRARY_PATH contains the ROCm lib directory."
    );
}
