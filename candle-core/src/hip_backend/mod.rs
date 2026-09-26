//! Implementation of Backend traits for the HIP (AMD) device in dual
//! cuda+rocm builds: the shared cuda_backend body bound to `cudarc-hip`
//! and the hipcc-built `candle-kernels` PTX-equivalent modules.
pub use candle_kernels as kernels;
pub use cudarc_hip as cudarc;
// The AMD role always has the fork API (cudarc-hip): both macros expand to
// their input, serving the full-fidelity backend in every feature shape.
macro_rules! gpu_fork_graphs {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
macro_rules! gpu_fork_lt {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
// The AMD role always has the fork Lt API: the stub variant expands to
// nothing here.
macro_rules! gpu_fork_lt_stub {
    ($($tokens:tt)*) => {};
}

// The AMD role always has the Lt branch: the inverted companion expands to
// nothing here.
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {};
}

// The AMD role always has the fork managed-upload API.
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

include!("../cuda_backend/body/mod.rs");

/// Dual-build (`cuda+rocm`) tests: the AMD role must work end-to-end (arch
/// resolution, host/device roundtrip, real kernel-module load from the
/// hipcc-built kernels crate) while the NVIDIA side stays probe-gated on a
/// box with no NVIDIA driver. No upstream-cudarc driver call may run here.
#[cfg(all(test, feature = "cuda", feature = "rocm"))]
mod dual_backend_tests {
    use super::{kernels, CudaDevice};
    use crate::backend::BackendDevice;
    use crate::{Device, GpuArch, Tensor};

    #[test]
    fn nvidia_side_fails_gracefully_without_driver() {
        // In link mode the binary needs a libcuda.so.1 to start (the toolkit
        // stub on this driver-less box). Init must return a proper stub error
        // - never panic and never touch the AMD side.
        let r = cudarc::driver::safe::CudaContext::new(0);
        assert!(r.is_err(), "expected a graceful stub-library error");
    }

    #[test]
    fn cpu_backend_coexists() {
        // CPU is unconditional (no feature): in a cuda+rocm build all three
        // backends live in one binary. Prove it with a real CPU op next to
        // the GPU roles.
        let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &Device::Cpu).unwrap();
        let c = (&a + &b).unwrap();
        let v: Vec<Vec<f32>> = c.to_vec2().unwrap();
        assert_eq!(v, vec![vec![6., 8.], vec![10., 12.]]);
    }

    #[test]
    fn hip_role_resolves_arch() {
        let dev = CudaDevice::new(0).expect("hip device 0");
        match GpuArch::resolve_hip(&dev).expect("arch resolves") {
            GpuArch::Rocm { gfx } => assert!(gfx.0 > 0, "{gfx:?}"),
            arch => panic!("expected an AMD arch, got {arch:?}"),
        }
    }

    #[test]
    fn hip_role_roundtrip_and_kernel_load() {
        let dev = CudaDevice::new(0).expect("hip device 0");
        let stream = dev.cuda_stream();
        let host: Vec<u32> = (0..4096u32).collect();
        let buf = stream.clone_htod(&host).expect("htod");
        let back: Vec<u32> = stream.clone_dtoh(&buf).expect("dtoh");
        assert_eq!(host, back);
        // Real kernel-module load from the hipcc-built kernels crate through
        // the role's `kernels` binding (proves the binding, not just alloc).
        let _func = dev
            .get_or_load_func(&super::kernel_name::<f32>("affine"), &kernels::AFFINE)
            .expect("affine module loads");
    }
}
