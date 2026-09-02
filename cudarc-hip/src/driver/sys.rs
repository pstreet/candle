//! Raw FFI to the HIP runtime, with CUDA-driver-API-compatible type aliases.
#![allow(non_camel_case_types, non_snake_case)]

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

extern "C" {
    pub fn hipSetDevice(dev: i32) -> c_int;
    pub fn hipGetDevice(dev: *mut i32) -> c_int;
    pub fn hipGetDeviceCount(count: *mut i32) -> c_int;
    pub fn hipDeviceGet(dev: *mut CUdevice, ordinal: i32) -> c_int;
    pub fn hipDeviceGetAttribute(pi: *mut i32, attr: i32, deviceId: i32) -> c_int;
    pub fn hipGetDeviceProperties(prop: *mut CUdeviceProp, deviceId: i32) -> c_int;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;

    pub fn hipMalloc(ptr: *mut CUdeviceptr, sizeInBytes: usize) -> c_int;
    pub fn hipMallocAsync(dev_ptr: *mut CUdeviceptr, size: usize, stream: CUstream) -> c_int;
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

// Driver-API-style aliases used by downstream code; each links onto the matching
// HIP runtime entry point and returns a `CUresult`.
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
