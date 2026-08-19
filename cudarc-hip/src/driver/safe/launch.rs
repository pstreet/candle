use crate::driver::result::{self, DriverError};
use crate::driver::safe::core::{
    CudaEvent, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr,
};
use crate::driver::sys;
use std::os::raw::c_void;
use std::sync::Arc;
use std::vec::Vec;

/// Configuration for [result::launch_kernel]
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// Creates a [LaunchConfig] with:
    /// - block_dim == `1024`
    /// - grid_dim == `(n + 1023) / 1024`
    /// - shared_mem_bytes == `0`
    pub fn for_num_elems(n: u32) -> Self {
        const NUM_THREADS: u32 = 1024;
        let num_blocks = n.div_ceil(NUM_THREADS);
        Self {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (NUM_THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

/// The kernel launch builder. Instantiate with [CudaStream::launch_builder()], and
/// then launch the kernel with [LaunchArgs::launch()]
///
/// Anything added as a kernel argument with [LaunchArgs::arg()] must either:
/// 1. Implement [DeviceRepr]
/// 2. Add a custom implementation of `impl<'a> PushKernelArg<T> for LaunchArgs<'a>`
#[derive(Debug)]
pub struct LaunchArgs<'a> {
    stream: &'a CudaStream,
    func: &'a CudaFunction,
    waits: Vec<&'a CudaEvent>,
    records: Vec<&'a CudaEvent>,
    args: Vec<*mut c_void>,
    flags: Option<sys::CUevent_flags>,
}

impl<'a> LaunchArgs<'a> {
    pub fn new(stream: &'a CudaStream, func: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs {
            stream,
            func,
            waits: Vec::new(),
            records: Vec::new(),
            args: Vec::new(),
            flags: None,
        }
    }

    /// Calling this will make [LaunchArgs::launch()] record events before and after
    /// the kernel is submitted.
    pub fn record_kernel_launch(&mut self, flags: sys::CUevent_flags) -> &mut Self {
        self.flags = Some(flags);
        self
    }

    /// Submits the configured [CudaFunction] to execute asynchronously on the
    /// configured device stream.
    ///
    /// # Safety
    /// The arguments must be valid for the configured [CudaFunction], and the
    /// referenced device memory must outlive the kernel execution.
    #[inline(always)]
    pub unsafe fn launch(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.stream.inner.ctx.bind_to_thread()?;
        for event in self.waits.iter() {
            self.stream.wait(*event)?;
        }
        if let Some(flags) = self.flags {
            let _ = self.stream.record_event(flags)?;
        }
        result::launch_kernel(
            self.func.cu_function,
            cfg.grid_dim,
            cfg.block_dim,
            cfg.shared_mem_bytes,
            self.stream.inner.cu_stream,
            &mut self.args,
        )?;
        for event in self.records.iter() {
            (*event).record(self.stream)?;
        }
        if let Some(flags) = self.flags {
            let _ = self.stream.record_event(flags)?;
        }
        Ok(())
    }
}

/// Something that can be copied to device memory and turned into a parameter for
/// [result::launch_kernel].
///
/// # Safety
/// `T` must be representable in device memory, and references to it can be
/// properly passed to the launcher.
pub unsafe trait PushKernelArg<T> {
    fn arg(&mut self, arg: T) -> &mut Self;
}

unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b T) -> &mut Self {
        self.args.push(arg as *const T as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaSlice<T>) -> &mut Self {
        add_events(self, &arg.read, &arg.write, true);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, T> PushKernelArg<&'b mut CudaSlice<T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaSlice<T>) -> &mut Self {
        add_events(self, &arg.read, &arg.write, false);
        self.args
            .push((&arg.cu_device_ptr) as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b CudaView<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b CudaView<'c, T>) -> &mut Self {
        add_events(self, &arg.read, &arg.write, true);
        self.args
            .push((&arg.ptr) as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

unsafe impl<'a, 'b: 'a, 'c: 'b, T> PushKernelArg<&'b mut CudaViewMut<'c, T>> for LaunchArgs<'a> {
    #[inline(always)]
    fn arg(&mut self, arg: &'b mut CudaViewMut<'c, T>) -> &mut Self {
        add_events(self, &arg.read, &arg.write, false);
        self.args
            .push((&arg.ptr) as *const sys::CUdeviceptr as *mut c_void);
        self
    }
}

#[inline(always)]
fn add_events<'a>(
    args: &mut LaunchArgs<'a>,
    read: &'a Option<Arc<CudaEvent>>,
    write: &'a Option<Arc<CudaEvent>>,
    immutable: bool,
) {
    if args.stream.inner.ctx.is_managing_stream_synchronization() {
        if let Some(write) = write {
            args.waits.push(&**write);
        }
        if immutable {
            if let Some(read) = read {
                args.records.push(&**read);
            }
        } else if let Some(write) = write {
            args.records.push(&**write);
        }
    }
}
