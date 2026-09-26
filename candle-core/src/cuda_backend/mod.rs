//! Implementation of Backend traits for the CUDA (NVIDIA) device, and the
//! AMD/HIP device in single-rocm builds via the crate-root `cudarc` alias.
//! In dual cuda+rocm builds this module is the NVIDIA role; the identical
//! body serves the AMD role through `hip_backend`.
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
pub use candle_kernels as kernels;
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub use candle_kernels_cuda as kernels;
pub use cudarc;
// Per-role capability switches for the shared backend body. Both roles
// compile the same feature set, so cfg cannot distinguish them; the role
// shells define these instead. In single-vendor builds (fork's cudarc-hip,
// which has the graph-capture and Lt fast-path API) they expand to their
// input; in a dual build the NVIDIA role (upstream cudarc, which lacks
// that fork API) drops them. The dropped functionality is exactly the
// BackendCaps degradation: that role serves GEMMs through the plain path
// and exposes no graph-capture methods until the fork API is ported.
/// Fork graph-capture arena API: present - expand to the input tokens.
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_graphs {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
/// Fork graph-capture arena API: absent - drop the input tokens.
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_graphs {
    ($($tokens:tt)*) => {};
}
// Per-role capability switches for the shared backend body. Both roles
// compile the same feature set, so cfg cannot distinguish them; the role
// shells define these instead. In single-vendor builds (fork's cudarc-hip,
// which has the graph-capture and Lt fast-path API) they expand to their
// input; in a dual build the NVIDIA role (upstream cudarc, which lacks
// that fork API) drops them. The dropped functionality is exactly the
// BackendCaps degradation: that role serves GEMMs through the plain path
// and exposes no graph-capture methods until the fork API is ported.
/// Fork hipBLASLt fast-path API: present - expand to the input tokens.
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_lt {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
/// Fork hipBLASLt fast-path API: absent - drop the input tokens.
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_lt {
    ($($tokens:tt)*) => {};
}

/// Companion to `gpu_fork_lt!`: expands to its input exactly where the Lt
/// API is ABSENT (the NVIDIA role of a dual build), providing the stub
/// definition the role still needs by name. No-op on single-vendor builds.
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_lt_stub {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_lt_stub {
    ($($tokens:tt)*) => {};
}

/// Inverted companion to the `gpu_fork_lt!` dispatch gate: expands to the
/// plain-GEMM statement where the Lt branch above is ABSENT (NVIDIA role of
/// a dual build), and to nothing where that branch handles it. This keeps
/// the fast path on roles with the fork Lt API and a correct plain path
/// everywhere without duplicating the GEMM logic beyond these three lines.
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {};
}

/// Fork managed-memory upload API (`clone_htod_managed*`), present only on
/// the fork's cudarc-hip. Roles without it serve weight uploads through the
/// plain copy path (see `load_quantized_managed`'s existing fallback).
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {};
}

/// Role-owned device-location variant: the shared body cannot name it (both
/// roles share the feature set), so each role shell defines which
/// `DeviceLocation` variant its devices report. Single-vendor builds report
/// `Cuda` exactly as before.
macro_rules! gpu_backend_location {
    ($id:expr) => {
        crate::DeviceLocation::Cuda { gpu_id: $id }
    };
}

include!("body/mod.rs");
