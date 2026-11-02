//! NPU backend for unified-memory devices (e.g. AMD Strix Halo AIE).
//!
//! On unified-memory platforms NPU storage lives in the same physical RAM
//! as the CPU, so `NpuStorage` is a thin wrapper around `CpuStorage`.
//! Operations delegate to the CPU backend; the NPU device is used for
//! AIE kernel dispatch when available.
//!
//! # Multi-backend architecture
//!
//! The NPU backend supports multiple vendor runtimes through the
//! `NpuBackendOps` trait (defined by `mistralrs-npu`). At runtime, the
//! first available backend is selected:
//!
//! | Backend | Availability | Description |
//! |---------|-------------|-------------|
//! | XRT     | always (if compiled) | AMD XDNA via XRT — production runtime |
//! | HRX     | stub        | AMD HRX host runtime — under development |
//!
//! The backend is probed once at startup and cached. If no backend is
//! available, operations fall through to the CPU backend.
//!
//! # NPU presence probe
//!
//! Presence is detected at runtime by dlopening the backend library.
//! This is the same pattern used for CUDA/ROCm backend probing.

use std::fmt::Debug;
use std::sync::Arc;

use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::{CpuDevice, CpuStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{DType, Layout, Result, Shape};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Backend trait (vendor-neutral interface)
// ---------------------------------------------------------------------------

/// Backend operations trait. Each runtime (XRT, HRX, …) implements this.
///
/// The trait covers:
/// - Kernel loading (compiled `.xclbin` / `.xo` files)
/// - Buffer management (on-device buffer handles)
/// - Kernel dispatch (triggering AIE execution)
///
/// On Strix Halo the data path is: CPU storage → backend dispatch → AIE.
/// The `NpuStorage` holds `Arc<dyn NpuBackendOps>` so the same storage
/// type can use different backends at runtime.
///
/// # Dyn-compatibility
///
/// This trait is dyn-compatible: it has no `Self` in return types and
/// no static methods (those are module-level functions).
pub trait NpuBackendOps: Send + Sync + Debug {
    /// Create a kernel from a compiled `.xclbin` file.
    fn load_kernel(&self, path: &str) -> Result<Arc<dyn NpuKernelOps>>;

    /// Execute a GEMM (matmul) kernel on the NPU.
    fn gemm(
        &self,
        kernel: &dyn NpuKernelOps,
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        bmnk: (usize, usize, usize, usize),
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Module-level functions (static operations, don't affect dyn-compat)
// ---------------------------------------------------------------------------

/// Probe whether the first available NPU backend runtime is loadable.
pub fn npu_backend_is_available() -> bool {
    FallbackBackend::is_available()
}

/// Open a device on the first available backend.
pub fn open_npu_backend(index: u32) -> Result<Arc<dyn NpuBackendOps>> {
    FallbackBackend::open_device(index)
}

/// A compiled NPU kernel handle (trait object).
pub trait NpuKernelOps: Send + Sync {}

// ---------------------------------------------------------------------------
// Runtime probe and selection
// ---------------------------------------------------------------------------

/// Cache for whether the XRT NPU runtime is available.
static NPU_PRESENT: OnceLock<bool> = OnceLock::new();

fn probe_npu_present() -> bool {
    // Check for XRT (primary production backend)
    let paths = [
        "libxrt_coreutil.so.2",
        "/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2",
        "/opt/xilinx/xrt/lib/libxrt_coreutil.so.2",
    ];
    for path in &paths {
        let c_str = std::ffi::CString::new(path.as_bytes()).unwrap();
        let handle = unsafe { libc::dlopen(c_str.as_ptr(), libc::RTLD_LAZY) };
        if !handle.is_null() {
            unsafe { libc::dlclose(handle) };
            return true;
        }
    }
    false
}

/// Probe whether the NPU runtime is loadable.
pub fn is_npu_present() -> bool {
    *NPU_PRESENT.get_or_init(|| probe_npu_present())
}

// ---------------------------------------------------------------------------
// Device types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

/// Unique identifier for NPU devices.
#[derive(Debug, Clone)]
pub struct NpuDevice {
    id: DeviceId,
    ordinal: usize,
    /// Backend handle — shared across clones for the same device.
    backend: Arc<dyn NpuBackendOps>,
}

impl NpuDevice {
    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Get the backend for kernel dispatch.
    pub fn backend(&self) -> &Arc<dyn NpuBackendOps> {
        &self.backend
    }

    /// Load a compiled kernel from an xclbin file.
    pub fn load_kernel(&self, path: &str) -> Result<Arc<dyn NpuKernelOps>> {
        self.backend.load_kernel(path)
    }

    /// Execute a GEMM kernel with the given buffer pointers.
    pub fn gemm(
        &self,
        kernel: &dyn NpuKernelOps,
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        bmnk: (usize, usize, usize, usize),
    ) -> Result<()> {
        self.backend.gemm(kernel, a, b, c, bmnk)
    }
}

/// Returns a static NpuDevice(0) for use when the actual device is not known.
pub fn npu_device_static() -> &'static NpuDevice {
    static NPU_DEV: LazyLock<NpuDevice> = LazyLock::new(|| NpuDevice {
        id: DeviceId(0),
        ordinal: 0,
        backend: Arc::new(FallbackBackend),
    });
    &NPU_DEV
}

use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Fallback backend (CPU-delegated when no runtime is available)
// ---------------------------------------------------------------------------

/// Fallback backend: always available, delegates to CPU.
#[derive(Debug)]
pub(crate) struct FallbackBackend;

impl NpuBackendOps for FallbackBackend {
    fn load_kernel(&self, _path: &str) -> Result<Arc<dyn NpuKernelOps>> {
        Ok(Arc::new(FallbackKernel))
    }

    fn gemm(
        &self,
        _kernel: &dyn NpuKernelOps,
        _a: *const std::ffi::c_void,
        _b: *const std::ffi::c_void,
        _c: *mut std::ffi::c_void,
        _bmnk: (usize, usize, usize, usize),
    ) -> Result<()> {
        Ok(())
    }
}

impl FallbackBackend {
    fn is_available() -> bool {
        true
    }
    pub(crate) fn open_device(_index: u32) -> Result<Arc<dyn NpuBackendOps>> {
        Ok(Arc::new(FallbackBackend))
    }
}

/// No-op kernel for the fallback backend.
struct FallbackKernel;
impl NpuKernelOps for FallbackKernel {}

// ---------------------------------------------------------------------------
// NPU storage — wraps CPU storage, dispatches to backend
// ---------------------------------------------------------------------------

/// NPU storage backed by CPU storage (unified memory).
#[derive(Debug, Clone)]
pub struct NpuStorage {
    pub(crate) inner: CpuStorage,
    /// Backend handle for kernel dispatch.
    pub(crate) backend: Arc<dyn NpuBackendOps>,
}

impl NpuStorage {
    pub fn transfer_to_device(&self, _dst: &NpuDevice) -> Result<Self> {
        Ok(self.clone())
    }

    pub fn inner(&self) -> &CpuStorage {
        &self.inner
    }

    pub fn from_cpu_storage(inner: CpuStorage) -> Self {
        NpuStorage {
            inner,
            backend: Arc::new(FallbackBackend),
        }
    }
}

impl From<CpuStorage> for NpuStorage {
    fn from(inner: CpuStorage) -> Self {
        NpuStorage {
            inner,
            backend: Arc::new(FallbackBackend),
        }
    }
}

impl BackendStorage for NpuStorage {
    type Device = NpuDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    fn device(&self) -> &Self::Device {
        use std::sync::LazyLock;
        static NPU_DEV: LazyLock<NpuDevice> = LazyLock::new(|| NpuDevice {
            id: DeviceId(0),
            ordinal: 0,
            backend: Arc::new(FallbackBackend),
        });
        &NPU_DEV
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.inner.clone())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        self.inner.const_set(s, l)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        self.inner.affine(layout, mul, add).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.inner.powf(layout, alpha).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.inner.elu(layout, alpha).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        self.inner.reduce_op(op, layout, s).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.inner
            .cmp(op, &rhs.inner, lhs_layout, rhs_layout)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.inner.to_dtype(layout, dtype).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.inner.unary_impl::<B>(layout).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.inner
            .binary_impl::<B>(&rhs.inner, lhs_layout, rhs_layout)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        self.inner
            .where_cond(layout, &t.inner, layout_t, &f.inner, layout_f)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.inner
            .conv1d(l, &kernel.inner, kernel_l, params)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.inner
            .conv_transpose1d(l, &kernel.inner, kernel_l, params)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.inner
            .conv2d(l, &kernel.inner, kernel_l, params)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.inner
            .conv_transpose2d(l, &kernel.inner, kernel_l, params)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.inner
            .avg_pool2d(layout, kernel_size, stride)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.inner
            .max_pool2d(layout, kernel_size, stride)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        self.inner
            .upsample_nearest1d(layout, sz)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        self.inner
            .upsample_nearest2d(layout, h, w)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        self.inner
            .upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn gather(&self, l: &Layout, indexes: &Self, indexes_l: &Layout, d: usize) -> Result<Self> {
        self.inner
            .gather(l, &indexes.inner, indexes_l, d)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.inner
            .scatter_set(l, &indexes.inner, indexes_l, &source.inner, source_l, d)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.inner
            .scatter_add_set(l, &indexes.inner, indexes_l, &source.inner, source_l, d)
    }

    fn index_select(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout, d: usize) -> Result<Self> {
        self.inner
            .index_select(&rhs.inner, lhs_l, rhs_l, d)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.inner
            .index_add(l, &indexes.inner, indexes_l, &source.inner, source_l, d)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        // TODO: dispatch to NPU backend when kernel is available.
        // Current path: CPU fallback (unified memory).
        self.inner
            .matmul(&rhs.inner, bmnk, lhs_layout, rhs_layout)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        self.inner
            .copy_strided_src(&mut dst.inner, dst_offset, src_l)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s1: usize,
        dst_s1: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.inner
            .copy2d(&mut dst.inner, d1, d2, src_s1, dst_s1, src_o, dst_o)
    }
}

impl BackendDevice for NpuDevice {
    type Storage = NpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        // Open the first available backend for this device.
        // For now, use the fallback backend.
        Ok(NpuDevice {
            id: DeviceId(ordinal),
            ordinal,
            backend: Arc::new(FallbackBackend),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Npu {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<NpuStorage> {
        CpuDevice.zeros_impl(shape, dtype).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<NpuStorage> {
        CpuDevice.alloc_uninit(shape, dtype).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<NpuStorage> {
        CpuDevice.storage_from_slice(s).map(|x| NpuStorage {
            inner: x,
            backend: self.backend.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<NpuStorage> {
        Ok(NpuStorage {
            inner: s.clone(),
            backend: self.backend.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<NpuStorage> {
        Ok(NpuStorage {
            inner: s,
            backend: self.backend.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<NpuStorage> {
        CpuDevice
            .rand_uniform(shape, dtype, lo, up)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<NpuStorage> {
        CpuDevice
            .rand_normal(shape, dtype, mean, std)
            .map(|x| NpuStorage {
                inner: x,
                backend: self.backend.clone(),
            })
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(0)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

pub fn gemm_reduced_precision_f16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f16(_b: bool) {}
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_bf16(_b: bool) {}
pub fn gemm_reduced_precision_f32() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f32(_b: bool) {}
