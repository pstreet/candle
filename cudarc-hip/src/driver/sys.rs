//! Raw FFI to the HIP runtime, with CUDA-driver-API-compatible type aliases.
#![allow(non_camel_case_types, non_snake_case)]
// FFI surface: C-ABI names and docless unsafe wrappers mirroring cudarc.
#![allow(clippy::missing_safety_doc)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type CUstream = *mut c_void;
pub type CUcontext = *mut c_void;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUmemPool = *mut c_void;
pub type CUstreamCaptureMode = c_uint;

// `hipStreamCaptureMode` values (identical to CUDA's `cudaStreamCaptureMode`).
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: CUstreamCaptureMode = 0;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: CUstreamCaptureMode = 1;
pub const CU_STREAM_CAPTURE_MODE_RELAXED: CUstreamCaptureMode = 2;

// `hipEventFlags` values (identical to CUDA's `CUevent_flags`), exposed as an enum so
// downstream code can address them as `CUevent_flags::CU_EVENT_*` like cudarc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CUevent_flags {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}

// `hipHostAlloc` flag values (identical to CUDA's `cuMemHostAllocFlags`).
pub const CU_MEMHOSTALLOC_DEFAULT: c_uint = 0x0;
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x1;
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x2;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: c_uint = 0x4;

// `hipMemAttach` flag values for `hipMallocManaged`.
pub const HIP_MEM_ATTACH_GLOBAL: c_uint = 0x1;
pub const HIP_MEM_ATTACH_HOST: c_uint = 0x2;

// Driver-API result codes, matching `hipError_t` discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CUresult {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_UNKNOWN = 999,
    CUDA_ERROR_NOT_SUPPORTED = 801,
}

pub type CUmemoryPool = *mut c_void;

// `hipMemPoolAttr` values (identical to CUDA's `CUmemPool_attribute`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CUmemPool_attribute {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 0x1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 0x2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 0x3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 0x4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 0x5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 0x6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 0x7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 0x8,
}

// `hipGraphMemAttributeType` values; these differ from CUDA's, so the
// discriminants carry the HIP runtime values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum CUgraphMem_attribute {
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0,
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3,
}

pub const HIP_STREAM_NON_BLOCKING: c_uint = 0x1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CUstreamCaptureStatus {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_GLOBAL = 2,
    CU_STREAM_CAPTURE_STATUS_MERGED = 3,
}

// Device attribute. This mirrors the CUDA `CUdevice_attribute` enum (which is how
// candle's CUDA backend addresses it, e.g. `sys::CUdevice_attribute::...`) while the
// discriminants carry the `hipDeviceAttribute*` values of the HIP runtime that backs
// this crate. The numeric value is what is passed to `hipDeviceGetAttribute`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 9,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 23,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 61,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 63,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 74,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 87,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTI_PROCESSOR = 25,
}

// hipMemcpyKind values (identical to CUDA's cudaMemcpyKind).
pub const CU_MEMCPY_HOST_TO_HOST: c_int = 0;
pub const CU_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CU_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CU_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;
pub const CU_MEMCPY_DEFAULT: c_int = 4;

pub const HIP_SUCCESS: c_int = 0;
pub const HIP_ERROR_INVALID_VALUE: c_int = 1;
pub const HIP_ERROR_OUT_OF_MEMORY: c_int = 2;

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

// `hipGetDeviceProperties` writes the full 1472-byte `hipDeviceProp_t`; only the
// leading name field is read back here.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct CUdeviceProp {
    pub bytes: [u8; 1472],
}

#[cfg(feature = "dynamic-loading")]
fn load<F: Copy>(name: &str) -> F {
    // SAFETY: loading a symbol is not itself unsafe; the `unsafe fn`
    // wrappers below own the safety contract of each call.
    unsafe {
        *culib()
            .get::<F>(name.as_bytes())
            .unwrap_or_else(|e| panic!("Missing symbol {name}: {e}"))
    }
}

#[cfg(feature = "dynamic-loading")]
fn culib_once() -> &'static Result<&'static ::libloading::Library, Vec<String>> {
    static LIB: std::sync::OnceLock<Result<&'static ::libloading::Library, Vec<String>>> =
        std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let choices = crate::get_lib_name_candidates("amdhip64");
        for choice in choices.iter() {
            // SAFETY: the candidate name comes from our own list; loading a
            // library executes no init code we do not already trust.
            if let Ok(lib) = unsafe { ::libloading::Library::new(choice) } {
                // The handle must never be dropped: the HIP runtime's static
                // initializers cannot survive a dlclose/reload cycle (verified
                // by segfault in the coexistence spike), and every later call
                // resolves through this same library.
                return Ok(&*Box::leak(Box::new(lib)));
            }
        }
        Err(choices)
    })
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn culib() -> &'static ::libloading::Library {
    match culib_once() {
        Ok(lib) => lib,
        Err(choices) => crate::panic_no_lib_found("amdhip64", choices),
    }
}

#[cfg(feature = "dynamic-loading")]
/// Mirrors `cudarc::driver::sys::is_culib_present`: check for the HIP runtime
/// so a caller can gate GPU-backend selection on a machine with no ROCm
/// installed. Shares the loader's handle, so probing never dlclose()s the
/// runtime (which would poison a later real init) - whichever of probe or
/// loader runs first keeps the library loaded for the process lifetime.
pub unsafe fn is_culib_present() -> bool {
    culib_once().is_ok()
}

#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn hipSetDevice(dev: i32) -> c_int;
    pub fn hipGetDevice(dev: *mut i32) -> c_int;
    pub fn hipGetDeviceCount(count: *mut i32) -> c_int;
    pub fn hipDeviceGet(dev: *mut CUdevice, ordinal: i32) -> c_int;
    pub fn hipDeviceGetAttribute(pi: *mut i32, attr: i32, deviceId: i32) -> c_int;
    pub fn hipGetDeviceProperties(prop: *mut CUdeviceProp, deviceId: i32) -> c_int;
    // Modern ROCm 6.0 ABI: the header maps hipGetDeviceProperties to this,
    // and the unversioned symbol only fills the legacy (pre-gcnArchName)
    // struct. Takes an opaque pointer so callers can pass an oversized
    // buffer - the modern struct layout is not stable across versions.
    pub fn hipGetDevicePropertiesR0600(prop: *mut c_void, deviceId: i32) -> c_int;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;

    pub fn hipMalloc(ptr: *mut CUdeviceptr, sizeInBytes: usize) -> c_int;
    pub fn hipMallocAsync(dev_ptr: *mut CUdeviceptr, size: usize, stream: CUstream) -> c_int;
    pub fn hipMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub fn hipMemPrefetchAsync(
        dev_ptr: *const c_void,
        count: usize,
        dst_device: c_int,
        stream: CUstream,
    ) -> c_int;
    pub fn hipFree(ptr: CUdeviceptr) -> c_int;
    pub fn hipFreeAsync(ptr: CUdeviceptr, stream: CUstream) -> c_int;
    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        sizeInBytes: usize,
        kind: c_int,
    ) -> c_int;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        sizeInBytes: usize,
        kind: c_int,
        stream: CUstream,
    ) -> c_int;
    pub fn hipMemsetAsync(
        dst: *mut c_void,
        value: c_int,
        sizeInBytes: usize,
        stream: CUstream,
    ) -> c_int;

    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> c_int;
    pub fn hipStreamCreate(stream: *mut CUstream) -> c_int;
    pub fn hipStreamCreateWithFlags(stream: *mut CUstream, flags: c_uint) -> c_int;
    pub fn hipStreamDestroy(stream: CUstream) -> c_int;
    pub fn hipStreamSynchronize(stream: CUstream) -> c_int;
    pub fn hipStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> c_int;
    pub fn hipStreamIsCapturing(stream: CUstream, capture_status: *mut i32) -> c_int;

    // Stream capture / graphs.
    pub fn hipStreamBeginCapture(stream: CUstream, mode: CUstreamCaptureMode) -> c_int;
    pub fn hipStreamEndCapture(stream: CUstream, pGraph: *mut CUgraph) -> c_int;
    pub fn hipGraphDestroy(graph: CUgraph) -> c_int;
    pub fn hipGraphInstantiateWithFlags(
        pGraphExec: *mut CUgraphExec,
        graph: CUgraph,
        flags: u64,
    ) -> c_int;
    pub fn hipGraphLaunch(graphExec: CUgraphExec, stream: CUstream) -> c_int;
    pub fn hipGraphExecDestroy(graphExec: CUgraphExec) -> c_int;
    pub fn hipGraphUpload(graphExec: CUgraphExec, stream: CUstream) -> c_int;

    pub fn hipDeviceGetMemPool(mem_pool: *mut CUmemPool, device: c_int) -> c_int;
    pub fn hipMemPoolSetAttribute(pool: CUmemPool, attr: c_uint, value: *mut c_void) -> c_int;
    pub fn hipMemPoolGetAttribute(pool: CUmemPool, attr: c_uint, value: *mut c_void) -> c_int;
    pub fn hipDeviceGraphMemTrim(device: c_int) -> c_int;
    pub fn hipDeviceGetGraphMemAttribute(device: c_int, attr: c_uint, value: *mut c_void) -> c_int;
    pub fn hipDeviceSetGraphMemAttribute(device: c_int, attr: c_uint, value: *mut c_void) -> c_int;
    pub fn hipHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int;
    pub fn hipHostFree(ptr: *mut c_void) -> c_int;
    pub fn hipEventElapsedTime(ms: *mut f32, start: CUevent, stop: CUevent) -> c_int;

    pub fn hipEventCreate(event: *mut CUevent) -> c_int;
    pub fn hipEventCreateWithFlags(event: *mut CUevent, flags: c_uint) -> c_int;
    pub fn hipEventRecord(event: CUevent, stream: CUstream) -> c_int;
    pub fn hipEventSynchronize(event: CUevent) -> c_int;
    pub fn hipEventDestroy(event: CUevent) -> c_int;

    pub fn hipModuleLoadData(module: *mut CUmodule, image: *const c_void) -> c_int;
    pub fn hipModuleGetFunction(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> c_int;
    pub fn hipModuleUnload(module: CUmodule) -> c_int;
    pub fn hipModuleLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        stream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> c_int;

    pub fn hipDeviceSynchronize() -> c_int;
    pub fn hipGetLastError() -> c_int;
    pub fn hipGetErrorString(error: c_int) -> *const c_char;
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipSetDevice(dev: i32) -> c_int {
    let hipSetDevice: extern "C" fn(i32) -> c_int = load("hipSetDevice");
    hipSetDevice(dev)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetDevice(dev: *mut i32) -> c_int {
    let hipGetDevice: extern "C" fn(*mut i32) -> c_int = load("hipGetDevice");
    hipGetDevice(dev)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetDeviceCount(count: *mut i32) -> c_int {
    let hipGetDeviceCount: extern "C" fn(*mut i32) -> c_int = load("hipGetDeviceCount");
    hipGetDeviceCount(count)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGet(dev: *mut CUdevice, ordinal: i32) -> c_int {
    let hipDeviceGet: extern "C" fn(*mut CUdevice, i32) -> c_int = load("hipDeviceGet");
    hipDeviceGet(dev, ordinal)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGetAttribute(pi: *mut i32, attr: i32, deviceId: i32) -> c_int {
    let hipDeviceGetAttribute: extern "C" fn(*mut i32, i32, i32) -> c_int =
        load("hipDeviceGetAttribute");
    hipDeviceGetAttribute(pi, attr, deviceId)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetDeviceProperties(prop: *mut CUdeviceProp, deviceId: i32) -> c_int {
    let hipGetDeviceProperties: extern "C" fn(*mut CUdeviceProp, i32) -> c_int =
        load("hipGetDeviceProperties");
    hipGetDeviceProperties(prop, deviceId)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetDevicePropertiesR0600(prop: *mut c_void, deviceId: i32) -> c_int {
    let hipGetDevicePropertiesR0600: extern "C" fn(*mut c_void, i32) -> c_int =
        load("hipGetDevicePropertiesR0600");
    hipGetDevicePropertiesR0600(prop, deviceId)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int {
    let hipMemGetInfo: extern "C" fn(*mut usize, *mut usize) -> c_int = load("hipMemGetInfo");
    hipMemGetInfo(free, total)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMalloc(ptr: *mut CUdeviceptr, sizeInBytes: usize) -> c_int {
    let hipMalloc: extern "C" fn(*mut CUdeviceptr, usize) -> c_int = load("hipMalloc");
    hipMalloc(ptr, sizeInBytes)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMallocAsync(dev_ptr: *mut CUdeviceptr, size: usize, stream: CUstream) -> c_int {
    let hipMallocAsync: extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> c_int =
        load("hipMallocAsync");
    hipMallocAsync(dev_ptr, size, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int {
    let hipMallocManaged: extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int =
        load("hipMallocManaged");
    hipMallocManaged(dev_ptr, size, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemPrefetchAsync(
    dev_ptr: *const c_void,
    count: usize,
    dst_device: c_int,
    stream: CUstream,
) -> c_int {
    let hipMemPrefetchAsync: extern "C" fn(*const c_void, usize, c_int, CUstream) -> c_int =
        load("hipMemPrefetchAsync");
    hipMemPrefetchAsync(dev_ptr, count, dst_device, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipFree(ptr: CUdeviceptr) -> c_int {
    let hipFree: extern "C" fn(CUdeviceptr) -> c_int = load("hipFree");
    hipFree(ptr)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipFreeAsync(ptr: CUdeviceptr, stream: CUstream) -> c_int {
    let hipFreeAsync: extern "C" fn(CUdeviceptr, CUstream) -> c_int = load("hipFreeAsync");
    hipFreeAsync(ptr, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    sizeInBytes: usize,
    kind: c_int,
) -> c_int {
    let hipMemcpy: extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int =
        load("hipMemcpy");
    hipMemcpy(dst, src, sizeInBytes, kind)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    sizeInBytes: usize,
    kind: c_int,
    stream: CUstream,
) -> c_int {
    let hipMemcpyAsync: extern "C" fn(*mut c_void, *const c_void, usize, c_int, CUstream) -> c_int =
        load("hipMemcpyAsync");
    hipMemcpyAsync(dst, src, sizeInBytes, kind, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemsetAsync(
    dst: *mut c_void,
    value: c_int,
    sizeInBytes: usize,
    stream: CUstream,
) -> c_int {
    let hipMemsetAsync: extern "C" fn(*mut c_void, c_int, usize, CUstream) -> c_int =
        load("hipMemsetAsync");
    hipMemsetAsync(dst, value, sizeInBytes, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> c_int {
    let hipDeviceGetName: extern "C" fn(*mut c_char, c_int, c_int) -> c_int =
        load("hipDeviceGetName");
    hipDeviceGetName(name, len, device)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamCreate(stream: *mut CUstream) -> c_int {
    let hipStreamCreate: extern "C" fn(*mut CUstream) -> c_int = load("hipStreamCreate");
    hipStreamCreate(stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamCreateWithFlags(stream: *mut CUstream, flags: c_uint) -> c_int {
    let hipStreamCreateWithFlags: extern "C" fn(*mut CUstream, c_uint) -> c_int =
        load("hipStreamCreateWithFlags");
    hipStreamCreateWithFlags(stream, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamDestroy(stream: CUstream) -> c_int {
    let hipStreamDestroy: extern "C" fn(CUstream) -> c_int = load("hipStreamDestroy");
    hipStreamDestroy(stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamSynchronize(stream: CUstream) -> c_int {
    let hipStreamSynchronize: extern "C" fn(CUstream) -> c_int = load("hipStreamSynchronize");
    hipStreamSynchronize(stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamWaitEvent(stream: CUstream, event: CUevent, flags: c_uint) -> c_int {
    let hipStreamWaitEvent: extern "C" fn(CUstream, CUevent, c_uint) -> c_int =
        load("hipStreamWaitEvent");
    hipStreamWaitEvent(stream, event, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamIsCapturing(stream: CUstream, capture_status: *mut i32) -> c_int {
    let hipStreamIsCapturing: extern "C" fn(CUstream, *mut i32) -> c_int =
        load("hipStreamIsCapturing");
    hipStreamIsCapturing(stream, capture_status)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamBeginCapture(stream: CUstream, mode: CUstreamCaptureMode) -> c_int {
    let hipStreamBeginCapture: extern "C" fn(CUstream, CUstreamCaptureMode) -> c_int =
        load("hipStreamBeginCapture");
    hipStreamBeginCapture(stream, mode)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipStreamEndCapture(stream: CUstream, pGraph: *mut CUgraph) -> c_int {
    let hipStreamEndCapture: extern "C" fn(CUstream, *mut CUgraph) -> c_int =
        load("hipStreamEndCapture");
    hipStreamEndCapture(stream, pGraph)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGraphDestroy(graph: CUgraph) -> c_int {
    let hipGraphDestroy: extern "C" fn(CUgraph) -> c_int = load("hipGraphDestroy");
    hipGraphDestroy(graph)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGraphInstantiateWithFlags(
    pGraphExec: *mut CUgraphExec,
    graph: CUgraph,
    flags: u64,
) -> c_int {
    let hipGraphInstantiateWithFlags: extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> c_int =
        load("hipGraphInstantiateWithFlags");
    hipGraphInstantiateWithFlags(pGraphExec, graph, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGraphLaunch(graphExec: CUgraphExec, stream: CUstream) -> c_int {
    let hipGraphLaunch: extern "C" fn(CUgraphExec, CUstream) -> c_int = load("hipGraphLaunch");
    hipGraphLaunch(graphExec, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGraphExecDestroy(graphExec: CUgraphExec) -> c_int {
    let hipGraphExecDestroy: extern "C" fn(CUgraphExec) -> c_int = load("hipGraphExecDestroy");
    hipGraphExecDestroy(graphExec)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGraphUpload(graphExec: CUgraphExec, stream: CUstream) -> c_int {
    let hipGraphUpload: extern "C" fn(CUgraphExec, CUstream) -> c_int = load("hipGraphUpload");
    hipGraphUpload(graphExec, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGetMemPool(mem_pool: *mut CUmemPool, device: c_int) -> c_int {
    let hipDeviceGetMemPool: extern "C" fn(*mut CUmemPool, c_int) -> c_int =
        load("hipDeviceGetMemPool");
    hipDeviceGetMemPool(mem_pool, device)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemPoolSetAttribute(pool: CUmemPool, attr: c_uint, value: *mut c_void) -> c_int {
    let hipMemPoolSetAttribute: extern "C" fn(CUmemPool, c_uint, *mut c_void) -> c_int =
        load("hipMemPoolSetAttribute");
    hipMemPoolSetAttribute(pool, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipMemPoolGetAttribute(pool: CUmemPool, attr: c_uint, value: *mut c_void) -> c_int {
    let hipMemPoolGetAttribute: extern "C" fn(CUmemPool, c_uint, *mut c_void) -> c_int =
        load("hipMemPoolGetAttribute");
    hipMemPoolGetAttribute(pool, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGraphMemTrim(device: c_int) -> c_int {
    let hipDeviceGraphMemTrim: extern "C" fn(c_int) -> c_int = load("hipDeviceGraphMemTrim");
    hipDeviceGraphMemTrim(device)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceGetGraphMemAttribute(
    device: c_int,
    attr: c_uint,
    value: *mut c_void,
) -> c_int {
    let hipDeviceGetGraphMemAttribute: extern "C" fn(c_int, c_uint, *mut c_void) -> c_int =
        load("hipDeviceGetGraphMemAttribute");
    hipDeviceGetGraphMemAttribute(device, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceSetGraphMemAttribute(
    device: c_int,
    attr: c_uint,
    value: *mut c_void,
) -> c_int {
    let hipDeviceSetGraphMemAttribute: extern "C" fn(c_int, c_uint, *mut c_void) -> c_int =
        load("hipDeviceSetGraphMemAttribute");
    hipDeviceSetGraphMemAttribute(device, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int {
    let hipHostAlloc: extern "C" fn(*mut *mut c_void, usize, c_uint) -> c_int =
        load("hipHostAlloc");
    hipHostAlloc(ptr, size, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipHostFree(ptr: *mut c_void) -> c_int {
    let hipHostFree: extern "C" fn(*mut c_void) -> c_int = load("hipHostFree");
    hipHostFree(ptr)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventElapsedTime(ms: *mut f32, start: CUevent, stop: CUevent) -> c_int {
    let hipEventElapsedTime: extern "C" fn(*mut f32, CUevent, CUevent) -> c_int =
        load("hipEventElapsedTime");
    hipEventElapsedTime(ms, start, stop)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventCreate(event: *mut CUevent) -> c_int {
    let hipEventCreate: extern "C" fn(*mut CUevent) -> c_int = load("hipEventCreate");
    hipEventCreate(event)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventCreateWithFlags(event: *mut CUevent, flags: c_uint) -> c_int {
    let hipEventCreateWithFlags: extern "C" fn(*mut CUevent, c_uint) -> c_int =
        load("hipEventCreateWithFlags");
    hipEventCreateWithFlags(event, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventRecord(event: CUevent, stream: CUstream) -> c_int {
    let hipEventRecord: extern "C" fn(CUevent, CUstream) -> c_int = load("hipEventRecord");
    hipEventRecord(event, stream)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventSynchronize(event: CUevent) -> c_int {
    let hipEventSynchronize: extern "C" fn(CUevent) -> c_int = load("hipEventSynchronize");
    hipEventSynchronize(event)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipEventDestroy(event: CUevent) -> c_int {
    let hipEventDestroy: extern "C" fn(CUevent) -> c_int = load("hipEventDestroy");
    hipEventDestroy(event)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipModuleLoadData(module: *mut CUmodule, image: *const c_void) -> c_int {
    let hipModuleLoadData: extern "C" fn(*mut CUmodule, *const c_void) -> c_int =
        load("hipModuleLoadData");
    hipModuleLoadData(module, image)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipModuleGetFunction(
    function: *mut CUfunction,
    module: CUmodule,
    name: *const c_char,
) -> c_int {
    let hipModuleGetFunction: extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> c_int =
        load("hipModuleGetFunction");
    hipModuleGetFunction(function, module, name)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipModuleUnload(module: CUmodule) -> c_int {
    let hipModuleUnload: extern "C" fn(CUmodule) -> c_int = load("hipModuleUnload");
    hipModuleUnload(module)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipModuleLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    stream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> c_int {
    let hipModuleLaunchKernel: extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int = load("hipModuleLaunchKernel");
    hipModuleLaunchKernel(
        f,
        gridDimX,
        gridDimY,
        gridDimZ,
        blockDimX,
        blockDimY,
        blockDimZ,
        sharedMemBytes,
        stream,
        kernelParams,
        extra,
    )
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipDeviceSynchronize() -> c_int {
    let hipDeviceSynchronize: extern "C" fn() -> c_int = load("hipDeviceSynchronize");
    hipDeviceSynchronize()
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetLastError() -> c_int {
    let hipGetLastError: extern "C" fn() -> c_int = load("hipGetLastError");
    hipGetLastError()
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn hipGetErrorString(error: c_int) -> *const c_char {
    let hipGetErrorString: extern "C" fn(c_int) -> *const c_char = load("hipGetErrorString");
    hipGetErrorString(error)
}

// Driver-API-style aliases used by downstream code; each links onto the matching
// HIP runtime entry point and returns a `CUresult`.
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    #[link_name = "hipMemPoolGetAttribute"]
    pub fn cuMemPoolGetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut c_void,
    ) -> CUresult;

    #[link_name = "hipMemPoolSetAttribute"]
    pub fn cuMemPoolSetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut c_void,
    ) -> CUresult;

    #[link_name = "hipDeviceGetMemPool"]
    pub fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, device: c_int) -> CUresult;

    #[link_name = "hipMemPoolTrimTo"]
    pub fn cuMemPoolTrimTo(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult;

    #[link_name = "hipDeviceGraphMemTrim"]
    pub fn cuDeviceGraphMemTrim(device: c_int) -> CUresult;

    #[link_name = "hipDeviceGetGraphMemAttribute"]
    pub fn cuDeviceGetGraphMemAttribute(
        device: c_int,
        attr: CUgraphMem_attribute,
        value: *mut c_void,
    ) -> CUresult;

    #[link_name = "hipDeviceSetGraphMemAttribute"]
    pub fn cuDeviceSetGraphMemAttribute(
        device: c_int,
        attr: CUgraphMem_attribute,
        value: *mut c_void,
    ) -> CUresult;

    #[link_name = "hipHostAlloc"]
    pub fn cuMemHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> CUresult;

    #[link_name = "hipHostFree"]
    pub fn cuMemFreeHost(ptr: *mut c_void) -> CUresult;

    #[link_name = "hipEventElapsedTime"]
    pub fn cuEventElapsedTime(ms: *mut f32, start: CUevent, stop: CUevent) -> CUresult;
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuMemPoolGetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    let cuMemPoolGetAttribute: extern "C" fn(
        CUmemoryPool,
        CUmemPool_attribute,
        *mut c_void,
    ) -> CUresult = load("hipMemPoolGetAttribute");
    cuMemPoolGetAttribute(pool, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuMemPoolSetAttribute(
    pool: CUmemoryPool,
    attr: CUmemPool_attribute,
    value: *mut c_void,
) -> CUresult {
    let cuMemPoolSetAttribute: extern "C" fn(
        CUmemoryPool,
        CUmemPool_attribute,
        *mut c_void,
    ) -> CUresult = load("hipMemPoolSetAttribute");
    cuMemPoolSetAttribute(pool, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, device: c_int) -> CUresult {
    let cuDeviceGetMemPool: extern "C" fn(*mut CUmemoryPool, c_int) -> CUresult =
        load("hipDeviceGetMemPool");
    cuDeviceGetMemPool(pool, device)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuMemPoolTrimTo(pool: CUmemoryPool, min_bytes_to_keep: usize) -> CUresult {
    let cuMemPoolTrimTo: extern "C" fn(CUmemoryPool, usize) -> CUresult = load("hipMemPoolTrimTo");
    cuMemPoolTrimTo(pool, min_bytes_to_keep)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuDeviceGraphMemTrim(device: c_int) -> CUresult {
    let cuDeviceGraphMemTrim: extern "C" fn(c_int) -> CUresult = load("hipDeviceGraphMemTrim");
    cuDeviceGraphMemTrim(device)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuDeviceGetGraphMemAttribute(
    device: c_int,
    attr: CUgraphMem_attribute,
    value: *mut c_void,
) -> CUresult {
    let cuDeviceGetGraphMemAttribute: extern "C" fn(
        c_int,
        CUgraphMem_attribute,
        *mut c_void,
    ) -> CUresult = load("hipDeviceGetGraphMemAttribute");
    cuDeviceGetGraphMemAttribute(device, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuDeviceSetGraphMemAttribute(
    device: c_int,
    attr: CUgraphMem_attribute,
    value: *mut c_void,
) -> CUresult {
    let cuDeviceSetGraphMemAttribute: extern "C" fn(
        c_int,
        CUgraphMem_attribute,
        *mut c_void,
    ) -> CUresult = load("hipDeviceSetGraphMemAttribute");
    cuDeviceSetGraphMemAttribute(device, attr, value)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuMemHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> CUresult {
    let cuMemHostAlloc: extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult =
        load("hipHostAlloc");
    cuMemHostAlloc(ptr, size, flags)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuMemFreeHost(ptr: *mut c_void) -> CUresult {
    let cuMemFreeHost: extern "C" fn(*mut c_void) -> CUresult = load("hipHostFree");
    cuMemFreeHost(ptr)
}

#[cfg(feature = "dynamic-loading")]
pub unsafe fn cuEventElapsedTime(ms: *mut f32, start: CUevent, stop: CUevent) -> CUresult {
    let cuEventElapsedTime: extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult =
        load("hipEventElapsedTime");
    cuEventElapsedTime(ms, start, stop)
}

// `cuMemcpyHtoDAsync_v2` has no 1:1 HIP symbol (the copy kind is implicit), so wrap it.
#[inline]
pub unsafe fn cuMemcpyHtoDAsync_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    size: usize,
    stream: CUstream,
) -> CUresult {
    let code = hipMemcpyAsync(
        dst as *mut c_void,
        src,
        size,
        CU_MEMCPY_HOST_TO_DEVICE,
        stream,
    );
    unsafe { std::mem::transmute::<c_int, CUresult>(code) }
}

// Same for `cuMemcpyDtoHAsync_v2`.
#[inline]
pub unsafe fn cuMemcpyDtoHAsync_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    size: usize,
    stream: CUstream,
) -> CUresult {
    let code = hipMemcpyAsync(
        dst,
        src as *const c_void,
        size,
        CU_MEMCPY_DEVICE_TO_HOST,
        stream,
    );
    unsafe { std::mem::transmute::<c_int, CUresult>(code) }
}
