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
//! Solo-`rocm` serves the AMD role through `hip_backend` (Phase 3
//! cutover); this module is NVIDIA-role only.
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
/// cuModuleLoadData NUL-terminated (Ptx::from -> Src -> CString).
#[cfg(feature = "cuda")]
macro_rules! gpu_ptx_load {
    ($mdl:expr) => {
        cudarc::nvrtc::Ptx::from($mdl.ptx())
    };
}
/// Fork hipBLASLt fast-path API: absent on the NVIDIA role's bound
/// crate in every shape.
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

/// Fork managed-memory upload API (`clone_htod_managed*`), absent on
/// the NVIDIA role. Roles without it serve weight uploads through
/// the plain copy path (see `load_quantized_managed`'s fallback).
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
