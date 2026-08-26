//! CUDA-compatible API surface over the HIP (ROCm) runtime.
//!
//! This crate re-implements the subset of the `cudarc` API that candle uses, on
//! top of the HIP runtime, so that candle's `cuda_backend` can run on AMD GPUs.

pub mod cublas;
pub mod cublaslt;
pub mod curand;
pub mod driver;
pub mod nvrtc;
