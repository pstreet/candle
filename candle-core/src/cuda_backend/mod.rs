//! NVIDIA role shell: binds the shared backend body (`body/mod.rs`) to
//! the vendored `cudarc-nvidia` driver crate in every build that links
//! the CUDA feature (cuda-only and dual builds alike). This module is
//! a ROLE DEFINITION, not a shape switch:
//!   - kernels: the PTX twin crate (`candle-kernels-cuda`), loaded as
//!     PTX source;
//!   - graph-capture arena: present (ported to the vendored fork);
//!   - hipBLASLt fast-path tier and managed uploads: absent on this
//!     role's bound crate; GEMMs serve through the plain path and
//!     weight uploads through the plain copy path (see BackendCaps).
//!
//! Legacy exception until the solo-rocm cutover (role-identity phase
//! 3, journal 2026-10-06): in `--features rocm` alone this shell also
//! serves the AMD role through the crate-root `cudarc` alias
//! (= `cudarc-hip`, full fork API, binary kernel images). After the
//! cutover this module compiles only for the NVIDIA role.
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub use candle_kernels as kernels;
#[cfg(feature = "cuda")]
pub use candle_kernels_cuda as kernels;
pub use cudarc;

/// Fork graph-capture arena API: present on this shell's bound crate in
/// every shape (cudarc-nvidia carries the arena port; the rocm-only
/// alias binds cudarc-hip, which has it natively) - unconditional.
macro_rules! gpu_fork_graphs {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

/// PTX loading for the shared body, role-keyed. PTX text must reach
/// cuModuleLoadData NUL-terminated (Ptx::from -> Src -> CString); the
/// rocm-only alias consumes the embedded binary image instead (the
/// hipcc path has no PTX text to JIT). A shared cfg(feature="rocm")
/// keyed the old choice, which fed raw PTX bytes as a binary image on
/// the dual NVIDIA role and read past the unterminated string into
/// garbage (CUDA_ERROR_INVALID_PTX).
#[cfg(feature = "cuda")]
macro_rules! gpu_ptx_load {
    ($mdl:expr) => {
        cudarc::nvrtc::Ptx::from($mdl.ptx())
    };
}
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
macro_rules! gpu_ptx_load {
    ($mdl:expr) => {
        cudarc::nvrtc::Ptx::from_binary($mdl.ptx_bytes().to_vec())
    };
}

/// Fork hipBLASLt fast-path API: present only on the rocm-only alias
/// (this shell bound to cudarc-hip). The NVIDIA role's bound crate
/// does not carry the fork Lt API. Retires together with the alias in
/// role-identity phase 3.
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
macro_rules! gpu_fork_lt {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(feature = "cuda")]
macro_rules! gpu_fork_lt {
    ($($tokens:tt)*) => {};
}

/// Companion to `gpu_fork_lt!`: expands to its input exactly where the
/// Lt API is ABSENT (the NVIDIA role in any cuda shape), providing the
/// stub definition the role still needs by name. No-op on the alias.
#[cfg(feature = "cuda")]
macro_rules! gpu_fork_lt_stub {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
macro_rules! gpu_fork_lt_stub {
    ($($tokens:tt)*) => {};
}

/// Inverted companion to the `gpu_fork_lt!` dispatch gate: expands to
/// the plain-GEMM statement where the Lt branch above is ABSENT (the
/// NVIDIA role in any cuda shape), and to nothing where that branch
/// handles it. This keeps the fast path on roles with the fork Lt API
/// and a correct plain path everywhere without duplicating the GEMM
/// logic beyond these three lines.
#[cfg(feature = "cuda")]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {};
}

/// Fork managed-memory upload API (`clone_htod_managed*`), present only
/// on the rocm-only alias. Roles without it serve weight uploads through
/// the plain copy path (see `load_quantized_managed`'s fallback).
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(feature = "cuda")]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {};
}

/// Role-owned device-location variant: the shared body cannot name it
/// (both roles share the feature set), so each role shell defines which
/// `DeviceLocation` variant its devices report.
macro_rules! gpu_backend_location {
    ($id:expr) => {
        crate::DeviceLocation::Cuda { gpu_id: $id }
    };
}

include!("body/mod.rs");
