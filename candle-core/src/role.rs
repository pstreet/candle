//! Serving-role backend selection for multi-vendor builds.
//!
//! All downstream C kernels in this fork are hipcc-built, so every consumer
//! serves the AMD role in a dual build. Single-vendor builds keep today's
//! meaning: `cuda_backend` bound to the compiled vendor. (A future NVIDIA-
//! serving consumer would bind the other role here instead - the mechanism
//! is intentionally one alias.)
#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub use super::cuda_backend as backend;
#[cfg(feature = "rocm")]
pub use super::hip_backend as backend;

/// The serving role's JIT-kernel modules.
pub use backend::kernels as role_kernels;
/// The serving role's device wrapper.
pub use backend::CudaDevice as RoleDevice;
/// The serving role's storage.
pub use backend::CudaStorage as RoleStorage;
