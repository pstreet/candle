use std::sync::Arc;

use crate::driver::{result, sys};

use super::{core::CudaSlice, CudaStream, DriverError};

/// Represents a replay-able Cuda Graph. Create with [CudaStream::begin_capture()] and [CudaStream::end_capture()].
///
/// Once created you can replay with [CudaGraph::launch()].
///
/// # On Thread safety
///
/// This object is **NOT** thread safe.
///
/// From official docs:
///
/// > Graph objects (cudaGraph_t, CUgraph) are not internally synchronized and must not be accessed concurrently from multiple threads. API calls accessing the same graph object must be serialized externally.
/// >
/// > Note that this includes APIs which may appear to be read-only, such as cudaGraphClone() (cuGraphClone()) and cudaGraphInstantiate() (cuGraphInstantiate()). No API or pair of APIs is guaranteed to be safe to call on the same graph object from two different threads without serialization.
///
/// <https://docs.nvidia.com/cuda/cuda-driver-api/graphs-thread-safety.html#graphs-thread-safety>
pub struct CudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let ctx = &self.stream.ctx;

        let cu_graph_exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !cu_graph_exec.is_null() {
            ctx.record_err(unsafe { result::graph::exec_destroy(cu_graph_exec) });
        }

        let cu_graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !cu_graph.is_null() {
            ctx.record_err(unsafe { result::graph::destroy(cu_graph) });
        }
    }
}

impl CudaStream {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g767167da0bbf07157dc20b6c258a2143)
    pub fn begin_capture_with_mode(&self, mode: sys::CUstreamCaptureMode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::begin_capture(self.cu_stream, mode) }
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g03dab8b2ba76b00718955177a929970c)
    ///
    /// `flags` is passed to [cuGraphInstantiate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gb53b435e178cccfa37ac87285d2c3fa1)
    pub fn end_capture_with_flags(
        self: &Arc<Self>,
        flags: sys::CUgraphInstantiate_flags,
    ) -> Result<Option<CudaGraph>, DriverError> {
        self.ctx.bind_to_thread()?;
        let cu_graph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        if cu_graph.is_null() {
            return Ok(None);
        }
        let cu_graph_exec = unsafe { result::graph::instantiate(cu_graph, flags) }?;
        Ok(Some(CudaGraph {
            cu_graph,
            cu_graph_exec,
            stream: self.clone(),
        }))
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g37823c49206e3704ae23c7ad78560bca)
    pub fn capture_status(&self) -> Result<sys::CUstreamCaptureStatus, DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::stream::is_capturing(self.cu_stream) }
    }
}

impl CudaGraph {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471)
    pub fn launch(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Pre-uploads the graph's resources to the device so that the
    /// first [CudaGraph::launch()] does not incur setup overhead.
    ///
    /// See [cuda docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb81438b083d42a26693f6f2bce150cd)
    pub fn upload(&self) -> Result<(), DriverError> {
        self.stream.ctx.bind_to_thread()?;
        unsafe { result::graph::upload(self.cu_graph_exec, self.stream.cu_stream) }
    }

    /// Get the underlying [sys::CUgraph].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the graph**, as it is still
    /// owned by the [CudaGraph].
    pub fn cu_graph(&self) -> sys::CUgraph {
        self.cu_graph
    }

    /// Get the underlying [sys::CUgraphExec].
    ///
    /// # Safety
    /// While this function is marked as safe, actually using the
    /// returned object is unsafe.
    ///
    /// **You must not destroy the graph exec**, as it is still
    /// owned by the [CudaGraph].
    pub fn cu_graph_exec(&self) -> sys::CUgraphExec {
        self.cu_graph_exec
    }
}

/// Template of captured work, produced by the fork-style
/// [CudaStream::begin_capture] / [CudaStream::end_capture] pair. Ported from
/// cudarc-hip: instantiate consumes the template, so each capture yields
/// exactly one exec.
pub struct CudaGraphTemplate {
    graph: sys::CUgraph,
    ctx: std::sync::Arc<super::core::CudaContext>,
}

impl CudaGraphTemplate {
    /// Compile the captured work into an executable graph bound to `stream`.
    /// Instantiated with flags 0: arena-captured graphs hold no mempool
    /// nodes, so there is nothing for AUTO_FREE to manage. The template is
    /// consumed and destroyed by this call.
    pub fn instantiate(
        self,
        stream: std::sync::Arc<CudaStream>,
        arena: Option<std::sync::Arc<CudaSlice<u8>>>,
    ) -> Result<std::sync::Arc<CudaGraphExec>, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let code = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut exec as *mut sys::CUgraphExec,
                self.graph,
                0,
            )
        };
        let res: Result<(), super::DriverError> =
            if code == sys::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(super::DriverError(code))
            };
        let destroy_template = || unsafe {
            let _ = result::graph::destroy(self.graph);
        };
        if let Err(e) = res {
            self.ctx.bind_to_thread().ok();
            destroy_template();
            return Err(e);
        }
        self.ctx.bind_to_thread().ok();
        destroy_template();
        Ok(std::sync::Arc::new(CudaGraphExec {
            exec,
            stream,
            ctx: self.ctx.clone(),
            arena,
        }))
    }
}

/// An executable graph captured from a stream. Ported from cudarc-hip.
pub struct CudaGraphExec {
    exec: sys::CUgraphExec,
    stream: std::sync::Arc<CudaStream>,
    ctx: std::sync::Arc<super::core::CudaContext>,
    /// Arena the captured allocations were carved out of (arena-capture
    /// mode); must outlive every replay and is released when the last exec
    /// handle drops.
    arena: Option<std::sync::Arc<CudaSlice<u8>>>,
}

impl CudaGraphExec {
    /// The stream this graph was captured from and is replayed on.
    pub fn stream(&self) -> &std::sync::Arc<CudaStream> {
        &self.stream
    }

    /// Launch the captured work on the owning stream.
    pub fn launch(&self) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { result::graph::launch(self.exec, self.stream.cu_stream) }
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        self.ctx.bind_to_thread().ok();
        unsafe {
            let _ = result::graph::exec_destroy(self.exec);
        }
    }
}

impl CudaStream {
    /// Begin capturing this stream into a graph (fork-style, no arena).
    /// Prefer [CudaStream::begin_capture_arena] for graphs that replay.
    pub fn begin_capture(&self) -> Result<(), DriverError> {
        *self.ctx.capture_arena.lock().unwrap() = None;
        self.ctx.capture_arena_offset.store(0, std::sync::atomic::Ordering::Relaxed);
        self.begin_capture_common()
    }

    /// End the capture begun by [CudaStream::begin_capture] or
    /// [CudaStream::begin_capture_arena], instantiating the captured work
    /// into an executable graph bound to this stream.
    pub fn end_capture(
        self: &std::sync::Arc<Self>,
    ) -> Result<std::sync::Arc<CudaGraphExec>, DriverError> {
        if self.capturing.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            return Err(DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE));
        }
        self.capturing.store(0, std::sync::atomic::Ordering::Relaxed);
        self.ctx.bind_to_thread()?;
        let graph: sys::CUgraph = unsafe { result::stream::end_capture(self.cu_stream) }?;
        let arena = self.ctx.capture_arena.lock().unwrap().take();
        let template = CudaGraphTemplate {
            graph,
            ctx: self.ctx.clone(),
        };
        template.instantiate(self.clone(), arena)
    }
}
