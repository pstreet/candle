use crate::driver::result::{self, DriverError};
use crate::driver::safe::launch::LaunchArgs;
use crate::driver::sys;
use crate::driver::sys::CUstreamCaptureStatus;
use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Marker traits (mirroring cudarc::driver)
// ---------------------------------------------------------------------------

/// Marker trait to indicate that the type is valid when all of its bits are set to 0.
///
/// # Safety
/// Not all types are valid when all bits are set to 0.
pub unsafe trait ValidAsZeroBits {}
unsafe impl ValidAsZeroBits for bool {}
unsafe impl ValidAsZeroBits for i8 {}
unsafe impl ValidAsZeroBits for i16 {}
unsafe impl ValidAsZeroBits for i32 {}
unsafe impl ValidAsZeroBits for i64 {}
unsafe impl ValidAsZeroBits for i128 {}
unsafe impl ValidAsZeroBits for isize {}
unsafe impl ValidAsZeroBits for u8 {}
unsafe impl ValidAsZeroBits for u16 {}
unsafe impl ValidAsZeroBits for u32 {}
unsafe impl ValidAsZeroBits for u64 {}
unsafe impl ValidAsZeroBits for u128 {}
unsafe impl ValidAsZeroBits for usize {}
unsafe impl ValidAsZeroBits for f32 {}
unsafe impl ValidAsZeroBits for f64 {}
unsafe impl ValidAsZeroBits for half::f16 {}
unsafe impl ValidAsZeroBits for half::bf16 {}
unsafe impl ValidAsZeroBits for float8::F8E4M3 {}
unsafe impl ValidAsZeroBits for float8::F8E5M2 {}
unsafe impl<T: ValidAsZeroBits, const M: usize> ValidAsZeroBits for [T; M] {}

macro_rules! impl_tuples {
    ($t:tt) => {
        impl_tuples!(@ $t);
    };
    ($l:tt $(,$t:tt)+) => {
        impl_tuples!($($t),+);
        impl_tuples!(@ $l $(,$t)+);
    };
    (@ $($t:tt),+) => {
        unsafe impl<$($t: ValidAsZeroBits,)+> ValidAsZeroBits for ($($t,)+) {}
    };
}
impl_tuples!(A, B, C, D, E, F, G, H, I, J, K, L);

/// Something that can be copied to device memory and turned into a kernel parameter.
///
/// # Safety
/// The type must be `#[repr(C)]`-representable and valid on the device.
pub unsafe trait DeviceRepr {}
unsafe impl DeviceRepr for bool {}
unsafe impl DeviceRepr for i8 {}
unsafe impl DeviceRepr for i16 {}
unsafe impl DeviceRepr for i32 {}
unsafe impl DeviceRepr for i64 {}
unsafe impl DeviceRepr for i128 {}
unsafe impl DeviceRepr for isize {}
unsafe impl DeviceRepr for u8 {}
unsafe impl DeviceRepr for u16 {}
unsafe impl DeviceRepr for u32 {}
unsafe impl DeviceRepr for u64 {}
unsafe impl DeviceRepr for u128 {}
unsafe impl DeviceRepr for usize {}
unsafe impl DeviceRepr for f32 {}
unsafe impl DeviceRepr for f64 {}
unsafe impl DeviceRepr for half::f16 {}
unsafe impl DeviceRepr for half::bf16 {}
unsafe impl DeviceRepr for float8::F8E4M3 {}
unsafe impl DeviceRepr for float8::F8E5M2 {}
unsafe impl<const N: usize, T> DeviceRepr for [T; N] where T: DeviceRepr {}

// ---------------------------------------------------------------------------
// CudaContext
// ---------------------------------------------------------------------------

/// A CUDA-like context; on HIP this maps to a device ordinal (the HIP runtime
/// manages a single implicit context per device).
#[derive(Debug)]
pub struct CudaContext {
    pub(crate) ordinal: usize,
    event_tracking: AtomicU8,
    stream_synchronization: AtomicU8,
    /// Pre-allocated buffer that graph-capture allocations are carved out of,
    /// instead of issuing `hipMallocAsync` nodes (which ROCm graphs cannot
    /// re-launch). Set for the duration of a capture via
    /// [CudaStream::begin_capture_arena].
    capture_arena: Mutex<Option<Arc<CudaSlice<u8>>>>,
    capture_arena_offset: AtomicUsize,
    /// Running total of bytes handed out by [CudaStream::alloc], useful for
    /// sizing the capture arena from a dry run.
    alloc_bytes: AtomicUsize,
}

unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

const TRACKING_OFF: u8 = 0;
const TRACKING_ON: u8 = 1;

impl CudaContext {
    /// Create a context bound to the device with the given `ordinal`.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        let count = Self::device_count()?;
        if ordinal as i32 >= count {
            return Err(DriverError(sys::HIP_ERROR_INVALID_VALUE));
        }
        let err = unsafe { sys::hipSetDevice(ordinal as i32) };
        result::check(err)?;
        Ok(Arc::new(Self {
            ordinal,
            event_tracking: AtomicU8::new(TRACKING_ON),
            stream_synchronization: AtomicU8::new(TRACKING_ON),
            capture_arena: Mutex::new(None),
            capture_arena_offset: AtomicUsize::new(0),
            alloc_bytes: AtomicUsize::new(0),
        }))
    }

    pub fn new_non_primary(ordinal: usize, _flags: u32) -> Result<Arc<Self>, DriverError> {
        Self::new(ordinal)
    }

    pub fn is_primary(&self) -> bool {
        true
    }

    /// HIP supports stream-ordered (async) allocation and freeing; using them
    /// keeps frees on the stream instead of synchronizing it, which matters a
    /// lot for inference performance.
    pub fn has_async_alloc(&self) -> bool {
        true
    }

    /// Running total of bytes handed out by [CudaStream::alloc] since context
    /// creation. Reset between phases to size the capture arena from a dry run.
    pub fn alloc_bytes(&self) -> usize {
        self.alloc_bytes.load(Ordering::Acquire)
    }

    pub fn reset_alloc_bytes(&self) {
        self.alloc_bytes.store(0, Ordering::Release);
    }

    /// Bump-allocate `bytes` from the active capture arena (256-byte aligned).
    /// Returns the device offset to add to the arena base, or `None` when no
    /// arena is active. Errors only on overflow of the arena.
    fn capture_arena_alloc(&self, bytes: usize) -> Result<Option<usize>, DriverError> {
        let arena = self.capture_arena.lock().unwrap();
        let arena = match &*arena {
            Some(a) => a,
            None => return Ok(None),
        };
        let need = (bytes + 255) & !255usize;
        let offset = self.capture_arena_offset.fetch_add(need, Ordering::AcqRel);
        if offset + need > arena.len() {
            return Err(DriverError(sys::HIP_ERROR_OUT_OF_MEMORY));
        }
        Ok(Some(offset as u64 as usize))
    }

    fn capture_arena_base(&self) -> Result<sys::CUdeviceptr, DriverError> {
        let arena = self.capture_arena.lock().unwrap();
        match &*arena {
            Some(a) => Ok(a.cu_device_ptr),
            None => Err(DriverError(sys::HIP_ERROR_INVALID_VALUE)),
        }
    }

    pub fn device_count() -> Result<i32, DriverError> {
        let mut count = 0;
        unsafe { result::check(sys::hipGetDeviceCount(&mut count)) }?;
        Ok(count)
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    pub fn name(&self) -> Result<String, DriverError> {
        let mut prop = std::mem::MaybeUninit::<sys::CUdeviceProp>::uninit();
        unsafe {
            result::check(sys::hipGetDeviceProperties(
                prop.as_mut_ptr(),
                self.ordinal as i32,
            ))?;
            let prop = prop.assume_init();
            let name = &prop.bytes[..256];
            let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            Ok(String::from_utf8_lossy(&name[..len]).into_owned())
        }
    }

    pub fn uuid(&self) -> Result<sys::CUuuid, DriverError> {
        Ok(sys::CUuuid::default())
    }

    pub fn compute_capability(&self) -> Result<(i32, i32), DriverError> {
        let major = unsafe {
            result::device::get_attribute(
                self.cu_device(),
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )?
        };
        let minor = unsafe {
            result::device::get_attribute(
                self.cu_device(),
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )?
        };
        Ok((major, minor))
    }

    pub fn total_mem(&self) -> Result<usize, DriverError> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { result::check(sys::hipMemGetInfo(&mut free, &mut total)) }?;
        Ok(total)
    }

    pub fn mem_get_info(&self) -> Result<(usize, usize), DriverError> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { result::check(sys::hipMemGetInfo(&mut free, &mut total)) }?;
        Ok((free, total))
    }

    pub fn cu_device(&self) -> sys::CUdevice {
        self.ordinal as i32
    }

    pub fn cu_ctx(&self) -> sys::CUcontext {
        std::ptr::null_mut()
    }

    /// The HIP runtime binds the current device on a per-thread basis; make sure
    /// this context's device is current before issuing further calls.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        unsafe { result::check(sys::hipSetDevice(self.ordinal as i32)) }
    }

    pub fn attribute(&self, attrib: sys::CUdevice_attribute) -> Result<i32, DriverError> {
        unsafe { result::device::get_attribute(self.cu_device(), attrib) }
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        unsafe { result::check(sys::hipDeviceSynchronize()) }
    }

    pub fn is_event_tracking(&self) -> bool {
        self.event_tracking.load(Ordering::Relaxed) == TRACKING_ON
    }

    pub fn is_managing_stream_synchronization(&self) -> bool {
        self.stream_synchronization.load(Ordering::Relaxed) == TRACKING_ON
    }

    /// # Safety
    /// Disabling event tracking may lead to use-after-free.
    pub unsafe fn enable_event_tracking(&self) {
        self.event_tracking.store(TRACKING_ON, Ordering::Relaxed);
        self.stream_synchronization
            .store(TRACKING_ON, Ordering::Relaxed);
    }

    /// # Safety
    /// Disabling event tracking may lead to use-after-free.
    pub unsafe fn disable_event_tracking(&self) {
        self.event_tracking.store(TRACKING_OFF, Ordering::Relaxed);
        self.stream_synchronization
            .store(TRACKING_OFF, Ordering::Relaxed);
    }

    pub fn check_err(&self) -> Result<(), DriverError> {
        unsafe { result::check(sys::hipGetLastError()) }
    }

    /// Best-effort error propagation for internal events; errors are reported but
    /// not fatal, mirroring cudarc's `record_err`.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(e) = result {
            eprintln!("cudarc-hip: recorded hip error: {e}");
        }
    }

    pub fn new_event(
        self: &Arc<Self>,
        flags: sys::CUevent_flags,
    ) -> Result<Arc<CudaEvent>, DriverError> {
        let mut ev = std::ptr::null_mut();
        self.bind_to_thread()?;
        unsafe {
            result::check(if flags == 0 {
                sys::hipEventCreate(&mut ev)
            } else {
                sys::hipEventCreateWithFlags(&mut ev, flags)
            })?;
        }
        Ok(Arc::new(CudaEvent {
            cu_event: ev,
            ctx: self.clone(),
        }))
    }

    /// A new non-blocking stream.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        let mut s = std::ptr::null_mut();
        self.bind_to_thread()?;
        unsafe {
            result::check(sys::hipStreamCreateWithFlags(
                &mut s,
                sys::HIP_STREAM_NON_BLOCKING,
            ))?;
        }
        Ok(Arc::new(CudaStream {
            inner: Arc::new(StreamInner {
                ctx: self.clone(),
                cu_stream: s,
                capturing: AtomicU8::new(0),
            }),
        }))
    }

    pub fn per_thread_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        thread_local! {
            static PER_THREAD_STREAMS: std::cell::RefCell<std::collections::HashMap<usize, Arc<CudaStream>>> =
                std::cell::RefCell::new(std::collections::HashMap::new());
        }
        PER_THREAD_STREAMS.with(|streams| {
            let mut streams = streams.borrow_mut();
            if let Some(s) = streams.get(&self.ordinal) {
                return s.clone();
            }
            let stream = self.new_stream().unwrap_or_else(|_| self.default_stream());
            streams.insert(self.ordinal, stream.clone());
            stream
        })
    }

    /// The HIP legacy stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            inner: Arc::new(StreamInner {
                ctx: self.clone(),
                cu_stream: std::ptr::null_mut(),
                capturing: AtomicU8::new(0),
            }),
        })
    }

    /// Dynamically load a compiled kernel image (ELF or PTX bytes) into this
    /// context.
    pub fn load_module(
        self: &Arc<Self>,
        ptx: crate::nvrtc::Ptx,
    ) -> Result<Arc<CudaModule>, DriverError> {
        self.bind_to_thread()?;
        let data: Vec<u8> = match ptx.0 {
            crate::nvrtc::PtxKind::Image(image) => unsafe {
                std::slice::from_raw_parts(image.as_ptr().cast(), image.len()).to_vec()
            },
            crate::nvrtc::PtxKind::Src(src) => src.into_bytes(),
            crate::nvrtc::PtxKind::File(path) => {
                std::fs::read(&path).map_err(|_| DriverError(sys::HIP_ERROR_INVALID_VALUE))?
            }
            crate::nvrtc::PtxKind::Binary(data) => data,
        };
        let cu_module = unsafe { result::module::load_data(data.as_ptr().cast()) }?;
        // Keep the image bytes alive for the lifetime of the module.
        Ok(Arc::new(CudaModule {
            cu_module,
            ctx: self.clone(),
            keep: data.into_boxed_slice(),
        }))
    }
}

// ---------------------------------------------------------------------------
// CudaEvent
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) cu_event: sys::CUevent,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        self.ctx
            .record_err(unsafe { result::check(sys::hipEventDestroy(self.cu_event)) });
    }
}

impl CudaEvent {
    pub fn cu_event(&self) -> sys::CUevent {
        self.cu_event
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn record(&self, stream: &CudaStream) -> Result<(), DriverError> {
        stream.inner.ctx.bind_to_thread()?;
        unsafe { result::check(sys::hipEventRecord(self.cu_event, stream.inner.cu_stream)) }
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        unsafe { result::check(sys::hipEventSynchronize(self.cu_event)) }
    }

    pub fn is_complete(&self) -> bool {
        let rc = unsafe { sys::hipEventSynchronize(self.cu_event) };
        rc == sys::HIP_SUCCESS
    }
}

// ---------------------------------------------------------------------------
// CudaStream
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct StreamInner {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) cu_stream: sys::CUstream,
    /// Whether this stream is currently in a graph capture. Allocations made
    /// while capturing are owned by the resulting graph (their memory is
    /// released when the graph exec is destroyed), not by the host.
    pub(crate) capturing: AtomicU8,
}

impl Drop for StreamInner {
    fn drop(&mut self) {
        if !self.cu_stream.is_null() {
            // Make sure this context is current on the dropping thread before we
            // tear the underlying HIP stream down (it may drop on a thread that is
            // not the one that created it).
            let _ = self.ctx.bind_to_thread();
            self.ctx
                .record_err(unsafe { result::check(sys::hipStreamDestroy(self.cu_stream)) });
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaStream {
    pub(crate) inner: Arc<StreamInner>,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn fork(&self) -> Result<Arc<Self>, DriverError> {
        self.inner.ctx.new_stream()
    }

    pub fn cu_stream(&self) -> sys::CUstream {
        self.inner.cu_stream
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.inner.ctx
    }

    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.inner.ctx.bind_to_thread()?;
        unsafe { result::check(sys::hipStreamSynchronize(self.inner.cu_stream)) }
    }

    pub fn record_event(&self, flags: sys::CUevent_flags) -> Result<Arc<CudaEvent>, DriverError> {
        let ev = self.inner.ctx.new_event(flags)?;
        ev.record(self)?;
        Ok(ev)
    }

    pub fn wait(&self, event: &CudaEvent) -> Result<(), DriverError> {
        self.inner.ctx.record_err(unsafe {
            result::check(sys::hipStreamWaitEvent(
                self.inner.cu_stream,
                event.cu_event,
                0,
            ))
        });
        Ok(())
    }

    pub fn join(&self, other: &CudaStream) -> Result<(), DriverError> {
        let ev = other.record_event(0)?;
        self.wait(&ev)
    }

    // Non-owning view over an existing device allocation; drop must not free it.
    pub unsafe fn upgrade_device_ptr<T: DeviceRepr>(&self, ptr: u64, len: usize) -> CudaSlice<T> {
        CudaSlice {
            cu_device_ptr: ptr as sys::CUdeviceptr,
            len: len / std::mem::size_of::<T>(),
            read: None,
            write: None,
            stream: Arc::new(self.clone()),
            graph_owned: true,
            marker: PhantomData,
        }
    }

    pub fn capture_status(&self) -> Result<CUstreamCaptureStatus, DriverError> {
        let mut status = 0i32;
        self.inner.ctx.bind_to_thread()?;
        unsafe { result::check(sys::hipStreamIsCapturing(self.inner.cu_stream, &mut status)) }?;
        Ok(match status {
            0 => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            2 => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_MERGED,
            _ => CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE,
        })
    }

    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        LaunchArgs::new(self, func)
    }

    // -- Memory allocation & copies -----------------------------------------

    /// True while this stream is in a graph capture (local, not HIP, state).
    pub fn is_capturing(&self) -> bool {
        self.inner.capturing.load(Ordering::Relaxed) != 0
    }

    /// Begin capturing this stream into a graph. All kernel launches, copies
    /// and (stream-ordered) allocations issued on this stream afterwards are
    /// recorded. [Self::end_capture] turns the result into a graph.
    ///
    /// In this (memory-pool) mode allocations inside the capture become
    /// `hipMallocAsync` graph nodes. ROCm graphs containing such nodes can
    /// only be launched once, so for graphs that must be replayed use
    /// [Self::begin_capture_arena] instead.
    pub fn begin_capture(&self) -> Result<(), DriverError> {
        *self.inner.ctx.capture_arena.lock().unwrap() = None;
        self.inner
            .ctx
            .capture_arena_offset
            .store(0, Ordering::Relaxed);
        self.begin_capture_common()
    }

    /// Begin capturing with a pre-allocated arena. All allocations made while
    /// capturing are carved out of `arena` (a plain `hipMalloc` buffer created
    /// before the capture), so the resulting graph contains no allocation
    /// nodes and can be re-launched any number of times. `arena` must stay
    /// valid until the returned graph exec is dropped.
    pub fn begin_capture_arena(&self, arena: Arc<CudaSlice<u8>>) -> Result<(), DriverError> {
        let mut g = self.inner.ctx.capture_arena.lock().unwrap();
        *g = Some(arena);
        drop(g);
        self.inner
            .ctx
            .capture_arena_offset
            .store(0, Ordering::Relaxed);
        self.begin_capture_common()
    }

    fn begin_capture_common(&self) -> Result<(), DriverError> {
        self.end_capture_abort_if_active()?;
        self.inner.ctx.bind_to_thread()?;
        unsafe {
            result::begin_capture(self.inner.cu_stream, sys::CU_STREAM_CAPTURE_MODE_GLOBAL)?;
        }
        self.inner.capturing.store(1, Ordering::Relaxed);
        Ok(())
    }

    fn end_capture_abort_if_active(&self) -> Result<(), DriverError> {
        if self.inner.capturing.load(Ordering::Relaxed) != 0 {
            // A capture is in flight but we lost track of it; stop capturing so
            // that `hipStreamEndCapture` below does not return a stale graph.
            self.inner.capturing.store(0, Ordering::Relaxed);
        }
        Ok(())
    }

    /// End the capture begun by [Self::begin_capture], instantiating the
    /// captured work into an executable graph bound to this stream.
    pub fn end_capture(&self) -> Result<Arc<CudaGraphExec>, DriverError> {
        if self.inner.capturing.load(Ordering::Relaxed) == 0 {
            return Err(DriverError(sys::HIP_ERROR_INVALID_VALUE));
        }
        self.inner.capturing.store(0, Ordering::Relaxed);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        self.inner.ctx.bind_to_thread()?;
        unsafe { result::end_capture(self.inner.cu_stream, &mut graph) }?;
        let arena = self.inner.ctx.capture_arena.lock().unwrap().take();
        let g = CudaGraph {
            graph,
            ctx: self.inner.ctx.clone(),
        };
        g.instantiate(Arc::new(self.clone()), arena)
    }

    pub unsafe fn alloc<T: DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>, DriverError> {
        if len == 0 {
            let mut empty = CudaSlice::new_empty(Arc::new(self.clone()))?;
            empty.read = Some(CudaEvent::new_internal(&self.inner.ctx, 0)?);
            empty.write = Some(CudaEvent::new_internal(&self.inner.ctx, 0)?);
            return Ok(empty);
        }
        self.inner.ctx.bind_to_thread()?;
        let bytes = len * std::mem::size_of::<T>();
        self.inner
            .ctx
            .alloc_bytes
            .fetch_add(bytes, Ordering::Release);
        let ptr = if self.is_capturing() {
            match self.inner.ctx.capture_arena_alloc(bytes)? {
                Some(offset) => self.inner.ctx.capture_arena_base()? + offset as u64,
                None => result::alloc_async(bytes, self.inner.cu_stream)?,
            }
        } else if self.inner.ctx.has_async_alloc() {
            result::alloc_async(bytes, self.inner.cu_stream)?
        } else {
            result::alloc(bytes)?
        };
        let events = self.new_slice_events()?;
        Ok(CudaSlice {
            cu_device_ptr: ptr,
            len,
            read: events.read,
            write: events.write,
            // Memory allocated while capturing belongs to the graph; the host
            // must not free it, so mark the slice as graph-owned (no-op drop).
            graph_owned: self.is_capturing(),
            stream: Arc::new(self.clone()),
            marker: PhantomData,
        })
    }

    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut slice = unsafe { self.alloc::<T>(len) }?;
        if !slice.is_empty() {
            unsafe {
                self.inner.ctx.bind_to_thread()?;
                result::check(sys::hipMemsetAsync(
                    slice.cu_device_ptr as *mut c_void,
                    0,
                    slice.num_bytes(),
                    self.inner.cu_stream,
                ))?;
            }
            event_record(&slice.write, self)?;
        }
        Ok(slice)
    }

    pub fn memset_zeros<T: DeviceRepr + ValidAsZeroBits, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        dst_len: usize,
    ) -> Result<(), DriverError> {
        if dst_len == 0 {
            return Ok(());
        }
        self.inner.ctx.bind_to_thread()?;
        let (dst_ptr, _record_dst) = dst.device_ptr_mut(self);
        unsafe {
            result::check(sys::hipMemsetAsync(
                dst_ptr as *mut c_void,
                0,
                dst_len * std::mem::size_of::<T>(),
                self.inner.cu_stream,
            ))?;
        }
        Ok(())
    }

    pub fn clone_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc::<T>(src.len()) }?;
        self.memcpy_htod(src, &mut dst)?;
        Ok(dst)
    }

    pub fn memcpy_htod<T: DeviceRepr, Src: HostSlice<T> + ?Sized, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        if src.is_empty() {
            return Ok(());
        }
        self.inner.ctx.bind_to_thread()?;
        let (src_, _record_src) = unsafe { src.stream_synced_slice(self) };
        let (dst_, _record_dst) = dst.device_ptr_mut(self);
        unsafe {
            result::check(sys::hipMemcpyAsync(
                dst_ as *mut c_void,
                src_.as_ptr().cast::<c_void>(),
                src_.len() * std::mem::size_of::<T>(),
                sys::CU_MEMCPY_HOST_TO_DEVICE,
                self.inner.cu_stream,
            ))?;
        }
        Ok(())
    }

    pub fn clone_dtoh<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>, DriverError> {
        let mut dst = Vec::with_capacity(src.len());
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len())
        };
        self.memcpy_dtoh(src, &mut dst)?;
        Ok(dst)
    }

    pub fn memcpy_dtoh<T: DeviceRepr, Src: DevicePtr<T>, Dst: HostSlice<T> + ?Sized>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        if src.is_empty() {
            return Ok(());
        }
        self.inner.ctx.bind_to_thread()?;
        let (src_, _record_src) = src.device_ptr(self);
        let (dst_, _record_dst) = unsafe { dst.stream_synced_mut_slice(self) };
        unsafe {
            result::check(sys::hipMemcpyAsync(
                dst_.as_mut_ptr().cast::<c_void>(),
                src_ as *const c_void,
                dst_.len() * std::mem::size_of::<T>(),
                sys::CU_MEMCPY_DEVICE_TO_HOST,
                self.inner.cu_stream,
            ))?;
        }
        // The caller reads the host slice immediately: block on the transfer.
        self.synchronize()?;
        Ok(())
    }

    pub fn memcpy_dtod<T, Src: DevicePtr<T>, Dst: DevicePtrMut<T>>(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<(), DriverError> {
        if src.is_empty() {
            return Ok(());
        }
        self.inner.ctx.bind_to_thread()?;
        let (src_, _record_src) = src.device_ptr(self);
        let (dst_, _record_dst) = dst.device_ptr_mut(self);
        unsafe {
            result::check(sys::hipMemcpyAsync(
                dst_ as *mut c_void,
                src_ as *const c_void,
                src.len() * std::mem::size_of::<T>(),
                sys::CU_MEMCPY_DEVICE_TO_DEVICE,
                self.inner.cu_stream,
            ))?;
        }
        Ok(())
    }

    pub fn clone_dtod<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<CudaSlice<T>, DriverError> {
        let mut dst = unsafe { self.alloc::<T>(src.len()) }?;
        self.memcpy_dtod(src, &mut dst)?;
        Ok(dst)
    }

    fn new_slice_events(&self) -> Result<SliceEvents, DriverError> {
        if !self.inner.ctx.is_event_tracking() {
            return Ok(SliceEvents {
                read: None,
                write: None,
            });
        }
        Ok(SliceEvents {
            read: Some(self.record_event(0)?),
            write: Some(self.record_event(0)?),
        })
    }
}

// ---------------------------------------------------------------------------
// CudaSlice / views
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CudaSlice<T> {
    pub(crate) cu_device_ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: Option<Arc<CudaEvent>>,
    pub(crate) write: Option<Arc<CudaEvent>>,
    pub(crate) stream: Arc<CudaStream>,
    /// True for memory allocated during a graph capture: it is owned by the
    /// graph (freed when the graph exec is destroyed), so the host drop is a no-op.
    pub(crate) graph_owned: bool,
    pub(crate) marker: PhantomData<*const T>,
}

unsafe impl<T> Send for CudaSlice<T> {}
unsafe impl<T> Sync for CudaSlice<T> {}

#[derive(Debug)]
struct SliceEvents {
    read: Option<Arc<CudaEvent>>,
    write: Option<Arc<CudaEvent>>,
}

impl<T> Drop for CudaSlice<T> {
    fn drop(&mut self) {
        // Memory allocated inside a graph capture is released with the graph;
        // the host must not free it again.
        if self.graph_owned {
            return;
        }
        let ctx = &self.stream.inner.ctx;
        if let Some(read) = self.read.as_ref() {
            ctx.record_err(self.stream.wait(read));
        }
        if let Some(write) = self.write.as_ref() {
            ctx.record_err(self.stream.wait(write));
        }
        if ctx.has_async_alloc() {
            ctx.record_err(unsafe {
                result::free_async(self.cu_device_ptr, self.stream.inner.cu_stream)
            });
        } else {
            ctx.record_err(self.stream.synchronize());
            ctx.record_err(unsafe { result::free_sync(self.cu_device_ptr) });
        }
    }
}

impl<T> CudaSlice<T> {
    fn new_empty(stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        Ok(Self {
            cu_device_ptr: 0,
            len: 0,
            read: None,
            write: None,
            graph_owned: false,
            stream,
            marker: PhantomData,
        })
    }

    /// The number of elements of `T` in this object.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes in this object.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// True if there are no elements in the object.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device ordinal this belongs to
    pub fn ordinal(&self) -> usize {
        self.stream.inner.ctx.ordinal
    }

    /// The context this belongs to
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.inner.ctx
    }

    /// The stream this object was allocated on and later will be dropped on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn as_view(&self) -> CudaView<'_, T> {
        CudaView {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    pub fn as_view_mut(&mut self) -> CudaViewMut<'_, T> {
        CudaViewMut {
            ptr: self.cu_device_ptr,
            len: self.len,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        }
    }

    /// Get a new [CudaView] that is a slice of this one. Panics if out of bounds.
    pub fn slice(&self, bounds: impl RangeBounds<usize>) -> CudaView<'_, T> {
        self.as_view().slice(bounds)
    }

    /// Fallible version of [Self::slice].
    pub fn try_slice(&self, bounds: impl RangeBounds<usize>) -> Option<CudaView<'_, T>> {
        self.as_view().try_slice(bounds)
    }

    /// Get a new [CudaViewMut] that is a slice of this one. Panics if out of bounds.
    pub fn slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> CudaViewMut<'_, T> {
        self.try_slice_mut(bounds).unwrap()
    }

    /// Fallible version of [Self::slice_mut].
    pub fn try_slice_mut(&mut self, bounds: impl RangeBounds<usize>) -> Option<CudaViewMut<'_, T>> {
        let (start, end) = range_to_offsets_opt(&bounds, self.len)?;
        Some(CudaViewMut {
            ptr: self.cu_device_ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: &self.read,
            write: &self.write,
            stream: &self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the memory as a slice of `S`, without copying.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'_, S>> {
        self.as_view().transmute(len)
    }
}

impl<T: DeviceRepr> CudaSlice<T> {
    /// Allocates a copy of self and schedules a device-to-device copy of the memory.
    pub fn try_clone(&self) -> Result<Self, DriverError> {
        self.stream.clone_dtod(self)
    }
}

impl<T: DeviceRepr> Clone for CudaSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: Clone + Default + DeviceRepr> TryFrom<CudaSlice<T>> for Vec<T> {
    type Error = DriverError;
    fn try_from(value: CudaSlice<T>) -> Result<Self, Self::Error> {
        value.stream.clone_dtoh(&value)
    }
}

/// `&[T]` on a gpu device. An immutable sub-view into a [CudaSlice].
#[derive(Debug)]
pub struct CudaView<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a Option<Arc<CudaEvent>>,
    pub(crate) write: &'a Option<Arc<CudaEvent>>,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a [T]>,
}

impl<'a, T> CudaView<'a, T> {
    /// The number of elements of `T` in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes in this view.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// True if there are no elements in the view.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device ordinal this belongs to.
    pub fn ordinal(&self) -> usize {
        self.stream.inner.ctx.ordinal
    }

    /// The context this belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.inner.ctx
    }

    /// The stream this was allocated on and later will be dropped on.
    pub fn stream(&self) -> &'a Arc<CudaStream> {
        self.stream
    }

    #[inline]
    fn clone_view(&self) -> CudaView<'a, T> {
        CudaView {
            ptr: self.ptr,
            len: self.len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Reinterpret this view as a mutable one, allowing in-place writes
    /// (e.g. `memcpy_htod`) through an immutable borrow.
    ///
    /// # Safety
    /// The caller must guarantee that no other view writes concurrently and
    /// that ordering on the stream is otherwise respected.
    pub unsafe fn as_mut_view(&self) -> CudaViewMut<'a, T> {
        CudaViewMut {
            ptr: self.ptr,
            len: self.len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Get a new [CudaView] that is a slice of this one. Panics if out of bounds.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let (start, end) = range_to_offsets(&range, self.len);
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Get a new [CudaView] that is a slice of this one. Returns `None` if the
    /// range is out of bounds.
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<Self> {
        let (start, end) = range_to_offsets_opt(&range, self.len)?;
        Some(Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the memory as a slice of `S`, without copying.
    ///
    /// # Safety
    /// `S` must have the same total byte size for the given lengths.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<CudaView<'a, S>> {
        if self.len * std::mem::size_of::<T>() != len * std::mem::size_of::<S>() {
            return None;
        }
        Some(CudaView {
            ptr: self.ptr,
            len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Splits the view at the given index, returning two views.
    pub fn split_at(&self, mid: usize) -> (CudaView<'a, T>, CudaView<'a, T>) {
        (self.slice(..mid), self.slice(mid..))
    }

    pub fn try_split_at(&self, mid: usize) -> Option<(CudaView<'a, T>, CudaView<'a, T>)> {
        Some((self.try_slice(..mid)?, self.try_slice(mid..)?))
    }

    /// Splits the view into `chunk_size` sized chunks.
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'a, T> {
        assert!(chunk_size > 0);
        ChunksExact {
            view: self.clone_view(),
            pos: 0,
            chunk_size,
            total: self.len / chunk_size,
        }
    }

    /// Identical behavior to [DevicePtr::device_ptr()], but the lifetime of the
    /// returned [SyncOnDrop] matches the lifetime of the view.
    pub fn view_ptr(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

/// `&mut [T]` on a gpu device. A mutable sub-view into a [CudaSlice].
#[derive(Debug)]
pub struct CudaViewMut<'a, T> {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) len: usize,
    pub(crate) read: &'a Option<Arc<CudaEvent>>,
    pub(crate) write: &'a Option<Arc<CudaEvent>>,
    pub(crate) stream: &'a Arc<CudaStream>,
    marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> CudaViewMut<'a, T> {
    /// The number of elements of `T` in this view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The number of bytes in this view.
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// True if there are no elements in the view.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device ordinal this belongs to.
    pub fn ordinal(&self) -> usize {
        self.stream.inner.ctx.ordinal
    }

    /// The context this belongs to.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.stream.inner.ctx
    }

    /// The stream this was allocated on and later will be dropped on.
    pub fn stream(&self) -> &'a Arc<CudaStream> {
        self.stream
    }

    /// Get a new [CudaViewMut] that is a slice of this one. Panics if out of bounds.
    pub fn slice(&mut self, range: impl RangeBounds<usize>) -> Self {
        let (start, end) = range_to_offsets(&range, self.len);
        Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        }
    }

    /// Get a new [CudaViewMut] that is a slice of this one. Returns `None` if the
    /// range is out of bounds.
    pub fn try_slice(&mut self, range: impl RangeBounds<usize>) -> Option<Self> {
        let (start, end) = range_to_offsets_opt(&range, self.len)?;
        Some(Self {
            ptr: self.ptr + (start * std::mem::size_of::<T>()) as u64,
            len: end - start,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    /// Reinterprets the memory as a slice of `S`, without copying.
    ///
    /// # Safety
    /// `S` must have the same total byte size for the given lengths.
    pub unsafe fn transmute<S>(&mut self, len: usize) -> Option<CudaViewMut<'a, S>> {
        if self.len * std::mem::size_of::<T>() != len * std::mem::size_of::<S>() {
            return None;
        }
        Some(CudaViewMut {
            ptr: self.ptr,
            len,
            read: self.read,
            write: self.write,
            stream: self.stream,
            marker: PhantomData,
        })
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (CudaViewMut<'a, T>, CudaViewMut<'a, T>) {
        (self.slice(..mid), self.slice(mid..))
    }

    pub fn try_split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(CudaViewMut<'a, T>, CudaViewMut<'a, T>)> {
        Some((self.try_slice(..mid)?, self.try_slice(mid..)?))
    }

    pub fn chunks_exact_mut(self, chunk_size: usize) -> ChunksExactMut<'a, T> {
        assert!(chunk_size > 0);
        let total = self.len / chunk_size;
        ChunksExactMut {
            view: self,
            pos: 0,
            chunk_size,
            total,
        }
    }

    /// Identical behavior to [DevicePtrMut::device_ptr_mut()], but the lifetime of
    /// the returned [SyncOnDrop] matches the lifetime of the view.
    pub fn view_ptr_mut(self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.inner.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        let ptr = self.ptr;
        (ptr, SyncOnDrop::record_event(self.write, stream))
    }
}

// ---------------------------------------------------------------------------
// Device pointer/slice traits
// ---------------------------------------------------------------------------

/// Base trait for abstracting over [CudaSlice]/[CudaView]/[CudaViewMut].
pub trait DeviceSlice<T> {
    fn len(&self) -> usize;
    fn num_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn stream(&self) -> &Arc<CudaStream>;
}

impl<T> DeviceSlice<T> for CudaSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DeviceSlice<T> for CudaView<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

impl<T> DeviceSlice<T> for CudaViewMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        self.stream
    }
}

/// A synchronization primitive to enable stream & event synchronization.
#[derive(Debug)]
#[must_use]
pub enum SyncOnDrop<'a> {
    /// Will record the stream's workload to the event on drop.
    Record(Option<(&'a CudaEvent, &'a CudaStream)>),
    /// Will call stream synchronize on drop.
    Sync(Option<&'a CudaStream>),
}

impl<'a> SyncOnDrop<'a> {
    /// Construct a [SyncOnDrop::Record] variant
    pub fn record_event(event: &'a Option<Arc<CudaEvent>>, stream: &'a CudaStream) -> Self {
        SyncOnDrop::Record(event.as_deref().map(|e| (e, stream)))
    }
    /// Construct a [SyncOnDrop::Sync] variant
    pub fn sync_stream(stream: &'a CudaStream) -> Self {
        SyncOnDrop::Sync(Some(stream))
    }
}

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        match self {
            SyncOnDrop::Record(target) => {
                if let Some((event, stream)) = std::mem::take(target) {
                    stream.inner.ctx.record_err(event.record(stream));
                }
            }
            SyncOnDrop::Sync(target) => {
                if let Some(stream) = std::mem::take(target) {
                    stream.inner.ctx.record_err(stream.synchronize());
                }
            }
        }
    }
}

/// Abstraction over [CudaSlice]/[CudaView]
pub trait DevicePtr<T>: DeviceSlice<T> {
    /// Retrieve the device pointer with the intent to read the device memory
    /// associated with it.
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>);
}

impl<T> DevicePtr<T> for CudaSlice<T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        (
            self.cu_device_ptr,
            SyncOnDrop::record_event(&self.read, stream),
        )
    }
}

impl<T> DevicePtr<T> for CudaView<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

impl<T> DevicePtr<T> for CudaViewMut<'_, T> {
    fn device_ptr<'a>(&'a self, stream: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        (self.ptr, SyncOnDrop::record_event(self.read, stream))
    }
}

/// Abstraction over [CudaSlice]/[CudaViewMut]
pub trait DevicePtrMut<T>: DeviceSlice<T> {
    /// Retrieve the device pointer with the intent to modify the device memory
    /// associated with it.
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>);

    /// Access to the event tracking the last write; used to re-record the event
    /// after scheduling a write.
    fn write_slot(&self) -> &Option<Arc<CudaEvent>>;
}

impl<T> DevicePtrMut<T> for CudaSlice<T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.inner.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        (
            self.cu_device_ptr,
            SyncOnDrop::record_event(&self.write, stream),
        )
    }

    fn write_slot(&self) -> &Option<Arc<CudaEvent>> {
        &self.write
    }
}

impl<T> DevicePtrMut<T> for CudaViewMut<'_, T> {
    fn device_ptr_mut<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        if self.stream.context().is_managing_stream_synchronization() {
            if let Some(read) = self.read.as_ref() {
                stream.inner.ctx.record_err(stream.wait(read));
            }
            if let Some(write) = self.write.as_ref() {
                stream.inner.ctx.record_err(stream.wait(write));
            }
        }
        let ptr = self.ptr;
        (ptr, SyncOnDrop::record_event(self.write, stream))
    }

    fn write_slot(&self) -> &Option<Arc<CudaEvent>> {
        self.write
    }
}

/// Host memory that can be copied to the device.
pub trait HostSlice<T>: AsRef<[T]> + AsMut<[T]> {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`.
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self.as_ref(), SyncOnDrop::Sync(None))
    }

    /// # Safety
    /// This is **only** safe if the resulting slice is used with `stream`.
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self.as_mut(), SyncOnDrop::Sync(None))
    }
}

impl<T> HostSlice<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T> HostSlice<T> for Vec<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T, const N: usize> HostSlice<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

// ---------------------------------------------------------------------------
// CudaModule / CudaFunction
// ---------------------------------------------------------------------------

/// Wrapper around a loaded HIP module.
#[derive(Debug)]
pub struct CudaModule {
    pub(crate) cu_module: sys::CUmodule,
    pub(crate) ctx: Arc<CudaContext>,
    keep: Box<[u8]>,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { result::check(sys::hipDeviceSynchronize()) });
        self.ctx
            .record_err(unsafe { result::module::unload(self.cu_module) });
    }
}

impl CudaModule {
    /// Loads a function from the loaded module with the given name.
    pub fn load_function(self: &Arc<Self>, fn_name: &str) -> Result<CudaFunction, DriverError> {
        let fn_name_c = std::ffi::CString::new(fn_name).unwrap();
        let cu_function = unsafe { result::module::get_function(self.cu_module, fn_name_c) }?;
        Ok(CudaFunction {
            cu_function,
            module: self.clone(),
        })
    }
}

/// Wrapper around a HIP function within a module.
#[derive(Debug, Clone)]
pub struct CudaFunction {
    pub(crate) cu_function: sys::CUfunction,
    pub(crate) module: Arc<CudaModule>,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}

impl CudaFunction {}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

impl CudaEvent {
    fn new_internal(ctx: &Arc<CudaContext>, flags: u32) -> Result<Arc<Self>, DriverError> {
        let mut ev = std::ptr::null_mut();
        unsafe {
            result::check(if flags == 0 {
                sys::hipEventCreate(&mut ev)
            } else {
                sys::hipEventCreateWithFlags(&mut ev, flags)
            })?;
        }
        Ok(Arc::new(Self {
            cu_event: ev,
            ctx: ctx.clone(),
        }))
    }
}

fn event_record(event: &Option<Arc<CudaEvent>>, stream: &CudaStream) -> Result<(), DriverError> {
    if let Some(e) = event {
        e.record(stream)?;
    }
    Ok(())
}

fn range_to_offsets<R: RangeBounds<usize>>(range: &R, len: usize) -> (usize, usize) {
    match range_to_offsets_opt(range, len) {
        Some(o) => o,
        None => panic!("index out of range"),
    }
}

fn range_to_offsets_opt<R: RangeBounds<usize>>(range: &R, len: usize) -> Option<(usize, usize)> {
    use std::ops::Bound::*;
    let start = match range.start_bound() {
        Included(&i) => i,
        Excluded(&i) => i.checked_add(1)?,
        Unbounded => 0,
    };
    let end = match range.end_bound() {
        Included(&i) => i.checked_add(1)?,
        Excluded(&i) => i,
        Unbounded => len,
    };
    if start > end || end > len {
        return None;
    }
    Some((start, end))
}

pub struct ChunksExact<'a, T> {
    view: CudaView<'a, T>,
    pos: usize,
    chunk_size: usize,
    total: usize,
}

impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = CudaView<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.total {
            return None;
        }
        let v = self.view.slice(self.pos..self.pos + self.chunk_size);
        self.pos += self.chunk_size;
        Some(v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.total - self.pos;
        (n, Some(n))
    }
}
impl<'a, T> ExactSizeIterator for ChunksExact<'a, T> {}

pub struct ChunksExactMut<'a, T> {
    view: CudaViewMut<'a, T>,
    pos: usize,
    chunk_size: usize,
    total: usize,
}

impl<'a, T> Iterator for ChunksExactMut<'a, T> {
    type Item = CudaViewMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.total {
            return None;
        }
        let v = self.view.slice(self.pos..self.pos + self.chunk_size);
        self.pos += self.chunk_size;
        Some(v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.total - self.pos;
        (n, Some(n))
    }
}
impl<'a, T> ExactSizeIterator for ChunksExactMut<'a, T> {}

// ---------------------------------------------------------------------------
// Graph capture
// ---------------------------------------------------------------------------

/// A graph template produced by [CudaStream::end_capture]. Instantiate it once
/// to get a [CudaGraphExec] that can be replayed any number of times.
#[derive(Debug)]
pub struct CudaGraph {
    graph: sys::CUgraph,
    ctx: Arc<CudaContext>,
}

impl CudaGraph {
    /// Compile the captured work into an executable graph bound to `stream`.
    ///
    /// The template is consumed and destroyed by this call; the executable
    /// owns the captured allocations and releases them when dropped.
    pub fn instantiate(
        self,
        stream: Arc<CudaStream>,
        arena: Option<Arc<CudaSlice<u8>>>,
    ) -> Result<Arc<CudaGraphExec>, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let res = unsafe { result::graph_instantiate(&mut exec, self.graph) };
        let destroy_template = || unsafe {
            let _ = result::graph_destroy(self.graph);
        };
        if let Err(e) = res {
            self.ctx.bind_to_thread().ok();
            destroy_template();
            return Err(e);
        }
        self.ctx.bind_to_thread().ok();
        destroy_template();
        Ok(Arc::new(CudaGraphExec {
            exec,
            stream,
            ctx: self.ctx.clone(),
            arena,
        }))
    }
}

/// An executable graph captured from a stream. Replaying it re-runs the exact
/// captured kernels and stream-ordered allocations with the same device
/// pointers, which collapses thousands of per-op launches into one.
#[derive(Debug, Clone)]
pub struct CudaGraphExec {
    exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    /// Arena the captured allocations were carved out of (arena-capture mode);
    /// must outlive every replay and is released when the last exec handle
    /// drops.
    arena: Option<Arc<CudaSlice<u8>>>,
}

impl CudaGraphExec {
    /// The stream this graph was captured from and is replayed on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Launch the captured work on the owning stream.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::graph_launch(self.exec, self.stream.inner.cu_stream) }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        self.ctx.bind_to_thread().ok();
        // All memory allocated inside the captured region is released here.
        unsafe {
            let _ = result::graph_exec_destroy(self.exec);
        }
    }
}
