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

/// AMD role: kernels reach the driver as the embedded binary image, not
/// PTX source (the hipcc path has no PTX text to JIT).
macro_rules! gpu_ptx_load {
    ($mdl:expr) => {
        cudarc::nvrtc::Ptx::from_binary($mdl.ptx_bytes().to_vec())
    };
}

/// AMD role: devices report the `Hip` location variant (per-API index, so
/// `hip:0` and `cuda:0` may address the same card).
macro_rules! gpu_backend_location {
    ($id:expr) => {
        crate::DeviceLocation::Hip { gpu_id: $id }
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
    fn nvidia_side_probe_gated_without_driver() {
        // The serving pattern: probe BEFORE any driver call - with
        // dynamic-loading cudarc panics (not errors) on the first call when
        // the library is absent, so is_culib_present() is mandatory. On this
        // driver-less box the probe must be false; with a driver present the
        // init may succeed and the test skips its assertions.
        if unsafe { cudarc::driver::sys::is_culib_present() } {
            return;
        }
        assert!(
            !unsafe { cudarc::driver::sys::is_culib_present() },
            "probe must be false without an NVIDIA runtime"
        );
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
    fn hip_device_enum_dispatch() {
        // Tensor-level proof of the Device::Hip / Storage::Hip dispatch
        // layer: zeros + add on hip, dtoh via to_vec, hip->cpu transfer,
        // and location reporting the Hip variant.
        use crate::DType;
        let dev = Device::new_hip(0).expect("hip:0");
        assert!(dev.is_hip());
        assert!(!dev.is_cuda());
        assert!(matches!(
            dev.location(),
            crate::DeviceLocation::Hip { gpu_id: 0 }
        ));
        let a = Tensor::zeros((2, 2), DType::F32, &dev).unwrap();
        let b = Tensor::ones((2, 2), DType::F32, &dev).unwrap();
        let c = (&a + &b).unwrap();
        assert!(c.device().is_hip());
        let v: Vec<Vec<f32>> = c.to_vec2().unwrap();
        assert_eq!(v, vec![vec![1., 1.], vec![1., 1.]]);
        let back = c.to_device(&Device::Cpu).unwrap();
        assert!(back.device().is_cpu());
        let v: Vec<Vec<f32>> = back.to_vec2().unwrap();
        assert_eq!(v, vec![vec![1., 1.], vec![1., 1.]]);
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

    #[test]
    fn cuda_role_constructor_refuses_cleanly_without_driver() {
        // Hot-pluggable: on this driver-less box the NVIDIA probe is false,
        // so the constructor must return a clean Err (never the cudarc
        // first-driver-call panic), availability must report false, and the
        // _if_available helper must fall back to CPU. With a driver present
        // the probe passes and the test skips its assertions.
        if unsafe { crate::cudarc::driver::sys::is_culib_present() } {
            return;
        }
        assert!(!crate::utils::cuda_is_available());
        let err = Device::new_cuda(0).expect_err("no NVIDIA runtime here");
        assert!(err.to_string().contains("not loadable"), "{err}");
        assert!(Device::cuda_if_available(0).expect("cpu fallback").is_cpu());
    }

    #[test]
    fn hip_role_constructor_gated_on_probe() {
        // Mirror image on the present runtime: the probe is true, the
        // availability helper agrees, and construction succeeds.
        assert!(crate::utils::hip_is_available());
        assert!(Device::new_hip(0).is_ok());
    }
}
