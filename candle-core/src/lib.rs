//! ML framework for Rust
//!
//! ```rust
//! use candle_core::{Tensor, DType, Device};
//! # use candle_core::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and feels like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Candle?
//!
//! Candle stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)
//!
//! ## Other Crates
//!
//! Candle consists of a number of crates. This crate holds core the common data structures but you may wish
//! to look at the docs for the other crates which can be found here:
//!
//! - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
//! - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
//! - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
//! - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
//! - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
//! - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
//! - [candle-transformers](https://docs.rs/candle-transformers/). Candle implementation of many published transformer models.
//!

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod backprop;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod cuda_backend;
mod custom_op;
mod device;
pub mod display;
mod dtype;
pub mod dummy_cuda_backend;
pub mod dummy_dtype;
mod dummy_metal_backend;
pub mod error;
pub mod gpu;
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub mod hip_backend;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub mod role;
mod indexer;
pub mod layout;
#[cfg(feature = "metal")]
pub mod metal_backend;
#[cfg(feature = "mkl")]
mod mkl;
pub mod nditer;
pub mod npy;
pub mod op;
pub mod pickle;
pub mod quantized;
pub mod safetensors;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
pub mod streaming;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod test_utils;
pub mod utils;
mod variable;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
#[cfg(feature = "ug")]
pub use custom_op::UgIOp1;
pub use custom_op::{CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use gpu::GpuArch;
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use nditer::NdIter;
pub use shape::{Shape, D};
pub use storage::{Storage, StorageMutRef, StorageRef};
pub use streaming::{StreamTensor, StreamingBinOp, StreamingModule};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};
pub use variable::Var;

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use cuda_backend as cuda;

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
pub use dummy_cuda_backend as cuda;

pub use cuda::{CudaDevice, CudaStorage};
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub use hip_backend::{CudaDevice as HipDevice, CudaStorage as HipStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(not(feature = "metal"))]
pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

// Feature matrix:
// - default / `--no-default-features`-style CPU builds: no GPU backend,
//   `GpuArch` exists but `resolve` is not compiled.
// - `cuda`: NVIDIA only. `rocm`: AMD only.
// - `cuda+rocm`: the multi-vendor build - both backends in one binary.
//   `cuda_backend` is the NVIDIA role (real `cudarc`, `candle-kernels-cuda`
//   PTX), `hip_backend` is the AMD role (`cudarc-hip`, hipcc-built
//   `candle-kernels`). NVIDIA driver access is runtime dlopen and must be
//   gated on `cudarc::driver::sys::is_culib_present()`.
// - plain `cuda` + `rocm` together is still rejected: opt into `cuda+rocm`
//   explicitly so the feature set is unambiguous.
#[cfg(all(feature = "cuda", feature = "rocm", not(feature = "cuda+rocm")))]
compile_error!(
    "building with both `cuda` and `rocm` requires the multi-vendor feature: \
     use `--features cuda+rocm` (both backends in one binary) or build one \
     backend per binary."
);

// Under plain `rocm`, `cudarc-hip` (`cudarc_hip`) re-implements the `cudarc`
// surface over HIP; alias it so the single-vendor GPU backend needs no source
// changes. In a `cuda+rocm` build the name `cudarc` is the real NVIDIA crate
// and the AMD role binds it per-module instead (see `hip_backend`).
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub extern crate cudarc_hip as cudarc;
// Under `cuda` (single or dual) the name `cudarc` is the real NVIDIA crate -
// needed by the backend bodies and GpuArch resolution.
#[cfg(feature = "cuda")]
pub extern crate cudarc;

/// Push zero or more kernel-launch arguments onto a
/// [`cudarc::driver::LaunchArgs`] builder. Lives at the crate root (not in
/// the backend body): the body is instantiated once per GPU role in a dual
/// build, so a body-local definition would export it twice. Downstream
/// crates (e.g. candle-nn) invoke it as `candle::builder_arg!`.
#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

/// A single forward method using a single tensor argument and a flag to
/// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}
