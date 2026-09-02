use super::sys;
use std::os::raw::{c_char, c_int, c_uint, c_void};

/// Error from a HIP runtime call, mirroring `cudarc::driver::DriverError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub sys::CUresult);

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let code = self.0 as i32;
        let msg = unsafe { std::ffi::CStr::from_ptr(sys::hipGetErrorString(code)) };
        write!(f, "hip error {code}: {msg}", msg = msg.to_string_lossy())
    }
}

impl std::error::Error for DriverError {}

fn cu_result(code: i32) -> sys::CUresult {
    unsafe { std::mem::transmute::<i32, sys::CUresult>(code) }
}

pub fn check(err: i32) -> Result<(), DriverError> {
    if err == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(cu_result(err)))
    }
}

pub unsafe fn alloc(size_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    let mut ptr: sys::CUdeviceptr = 0;
    check(sys::hipMalloc(&mut ptr, size_bytes))?;
    Ok(ptr)
}

pub unsafe fn alloc_async(
    size_bytes: usize,
    stream: sys::CUstream,
) -> Result<sys::CUdeviceptr, DriverError> {
    let mut ptr: sys::CUdeviceptr = 0;
    check(sys::hipMallocAsync(&mut ptr, size_bytes, stream))?;
    Ok(ptr)
}

pub unsafe fn free_sync(ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    check(sys::hipFree(ptr))
}

pub unsafe fn free_async(ptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    check(sys::hipFreeAsync(ptr, stream))
}

pub unsafe fn malloc_host(size_bytes: usize, flags: c_uint) -> Result<*mut c_void, DriverError> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    check(sys::hipHostAlloc(&mut ptr, size_bytes, flags))?;
    Ok(ptr)
}

pub unsafe fn free_host(ptr: *mut c_void) -> Result<(), DriverError> {
    check(sys::hipHostFree(ptr))
}

// -- Stream capture / graphs -------------------------------------------------

pub unsafe fn begin_capture(
    stream: sys::CUstream,
    mode: sys::CUstreamCaptureMode,
) -> Result<(), DriverError> {
    check(sys::hipStreamBeginCapture(stream, mode))
}

pub unsafe fn end_capture(
    stream: sys::CUstream,
    graph: *mut sys::CUgraph,
) -> Result<(), DriverError> {
    check(sys::hipStreamEndCapture(stream, graph))
}

pub unsafe fn graph_destroy(graph: sys::CUgraph) -> Result<(), DriverError> {
    check(sys::hipGraphDestroy(graph))
}

pub unsafe fn graph_instantiate(
    exec: *mut sys::CUgraphExec,
    graph: sys::CUgraph,
) -> Result<(), DriverError> {
    check(sys::hipGraphInstantiateWithFlags(exec, graph, 0))
}

pub unsafe fn graph_launch(
    exec: sys::CUgraphExec,
    stream: sys::CUstream,
) -> Result<(), DriverError> {
    check(sys::hipGraphLaunch(exec, stream))
}

pub unsafe fn graph_exec_destroy(exec: sys::CUgraphExec) -> Result<(), DriverError> {
    check(sys::hipGraphExecDestroy(exec))
}

pub unsafe fn launch_kernel(
    func: sys::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: sys::CUstream,
    args: &mut Vec<*mut c_void>,
) -> Result<(), DriverError> {
    check(sys::hipModuleLaunchKernel(
        func,
        grid_dim.0,
        grid_dim.1,
        grid_dim.2,
        block_dim.0,
        block_dim.1,
        block_dim.2,
        shared_mem_bytes,
        stream,
        args.as_mut_ptr(),
        std::ptr::null_mut(),
    ))
}

pub fn mem_get_info() -> Result<(usize, usize), DriverError> {
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { sys::hipMemGetInfo(&mut free, &mut total) })?;
    Ok((free, total))
}

pub mod device {
    use super::{check, DriverError};
    use crate::driver::sys;
    use std::os::raw::{c_char, c_int};

    pub fn get(ordinal: i32) -> Result<sys::CUdevice, DriverError> {
        let mut dev: sys::CUdevice = 0;
        check(unsafe { sys::hipDeviceGet(&mut dev, ordinal) })?;
        Ok(dev)
    }

    pub unsafe fn get_attribute(
        dev: sys::CUdevice,
        attr: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut val = 0i32;
        check(sys::hipDeviceGetAttribute(&mut val, attr as i32, dev))?;
        Ok(val)
    }

    pub unsafe fn get_name(dev: sys::CUdevice) -> Result<String, DriverError> {
        let mut buf = [0 as c_char; 256];
        check(sys::hipDeviceGetName(
            buf.as_mut_ptr(),
            buf.len() as c_int,
            dev,
        ))?;
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Ok(
            String::from_utf8_lossy(&buf[..end].iter().map(|&c| c as u8).collect::<Vec<_>>())
                .into_owned(),
        )
    }
}

pub mod module {
    use super::{check, DriverError};
    use crate::driver::sys;
    use std::ffi::CString;
    use std::os::raw::c_void;

    pub unsafe fn load_data(image: *const c_void) -> Result<sys::CUmodule, DriverError> {
        let mut m = std::ptr::null_mut();
        check(sys::hipModuleLoadData(&mut m, image))?;
        Ok(m)
    }

    pub unsafe fn get_function(
        m: sys::CUmodule,
        name: CString,
    ) -> Result<sys::CUfunction, DriverError> {
        let mut f = std::ptr::null_mut();
        check(sys::hipModuleGetFunction(&mut f, m, name.as_ptr()))?;
        Ok(f)
    }

    pub unsafe fn unload(m: sys::CUmodule) -> Result<(), DriverError> {
        check(sys::hipModuleUnload(m))
    }
}
