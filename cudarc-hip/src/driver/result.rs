use super::sys;
use std::os::raw::c_void;

/// Error from a HIP runtime call, mirroring `cudarc::driver::DriverError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverError(pub i32);

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = unsafe { std::ffi::CStr::from_ptr(sys::hipGetErrorString(self.0)) };
        write!(f, "hip error {}: {}", self.0, msg.to_string_lossy())
    }
}

impl std::error::Error for DriverError {}

pub fn check(err: i32) -> Result<(), DriverError> {
    if err == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(DriverError(err))
    }
}

pub unsafe fn alloc(size_bytes: usize) -> Result<sys::CUdeviceptr, DriverError> {
    let mut ptr: sys::CUdeviceptr = 0;
    check(sys::hipMalloc(&mut ptr, size_bytes))?;
    Ok(ptr)
}

pub unsafe fn free_sync(ptr: sys::CUdeviceptr) -> Result<(), DriverError> {
    check(sys::hipFree(ptr))
}

pub unsafe fn free_async(ptr: sys::CUdeviceptr, stream: sys::CUstream) -> Result<(), DriverError> {
    check(sys::hipFreeAsync(ptr, stream))
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

pub mod device {
    use super::{check, DriverError};
    use crate::driver::sys;

    pub unsafe fn get_attribute(
        dev: sys::CUdevice,
        attr: sys::CUdevice_attribute,
    ) -> Result<i32, DriverError> {
        let mut val = 0i32;
        check(sys::hipDeviceGetAttribute(&mut val, attr as i32, dev))?;
        Ok(val)
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
