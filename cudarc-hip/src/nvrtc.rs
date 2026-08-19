//! NVRTC-compatible module compilation over HIPRTC.

use std::ffi::c_char;
use std::path::PathBuf;

/// An error compiling a module source with HIPRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileError(pub i32);

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hiprtc error {}", self.0)
    }
}

impl std::error::Error for CompileError {}

/// Options for [safe::compile_ptx_with_opts].
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    pub use_fast_math: Option<bool>,
    pub include_paths: Vec<PathBuf>,
    pub macros: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub enum PtxKind {
    /// An image produced by [safe::compile_ptx_with_opts].
    Image(Vec<c_char>),
    /// The content of a pre-compiled image file (PTX or ELF bytes).
    Src(String),
    /// Path to a compiled image file.
    File(PathBuf),
    /// Raw binary (ELF) data.
    Binary(Vec<u8>),
}

/// A compiled module image, mirroring `cudarc::nvrtc::Ptx`.
#[derive(Debug, Clone)]
pub struct Ptx(pub PtxKind);

impl Ptx {
    /// Creates a `Ptx` from a pre-compiled image file path.
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        Self(PtxKind::File(path.into()))
    }

    /// Creates a `Ptx` from the source string of a pre-compiled image.
    pub fn from_src<S: Into<String>>(src: S) -> Self {
        Self(PtxKind::Src(src.into()))
    }

    /// Creates a `Ptx` from raw binary (ELF/CUBIN) data.
    pub fn from_binary(data: Vec<u8>) -> Self {
        Self(PtxKind::Binary(data))
    }
}

impl From<&str> for Ptx {
    fn from(value: &str) -> Self {
        Self::from_src(value)
    }
}

impl From<&'static [u8]> for Ptx {
    fn from(value: &'static [u8]) -> Self {
        Self::from_binary(value.to_vec())
    }
}

/// Compiles `src` (a CUDA C++ / HIP source string) into a module image.
///
/// On ROCm this uses HIPRTC; the produced PTX is JIT-compiled by the HIP runtime
/// when the module is loaded.
pub mod safe {
    use super::{CompileError, CompileOptions, Ptx, PtxKind};
    use std::ffi::{c_char, CString};
    use std::os::raw::{c_int, c_void};

    /// Compiles the given CUDA/HIP source and returns the module image (PTX).
    pub fn compile_ptx_with_opts(src: String, opt: CompileOptions) -> Result<Ptx, CompileError> {
        compile(src.as_str(), &opt)
    }

    /// Compiles the given CUDA/HIP source with default options.
    pub fn compile_ptx(src: String) -> Result<Ptx, CompileError> {
        compile(src.as_str(), &CompileOptions::default())
    }

    fn compile(src: &str, opt: &CompileOptions) -> Result<Ptx, CompileError> {
        extern "C" {
            fn hiprtcCreateProgram(
                prog: *mut *mut c_void,
                src: *const c_char,
                name: *const c_char,
                numHeaders: c_int,
                headers: *const *const c_char,
                includeNames: *const *const c_char,
            ) -> i32;
            fn hiprtcDestroyProgram(prog: *mut *mut c_void) -> i32;
            fn hiprtcCompileProgram(
                prog: *mut c_void,
                numOpts: c_int,
                opts: *const *const c_char,
            ) -> i32;
            fn hiprtcGetPTXSize(prog: *mut c_void, size: *mut usize) -> i32;
            fn hiprtcGetPTX(prog: *mut c_void, ptx: *mut c_char) -> i32;
        }

        let c_src = CString::new(src).map_err(|_| CompileError(7))?;
        let c_name = CString::new("candle_module").unwrap();
        let mut prog: *mut c_void = std::ptr::null_mut();
        let rc = unsafe {
            hiprtcCreateProgram(
                &mut prog,
                c_src.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if rc != 0 {
            return Err(CompileError(rc));
        }
        let mut opts: Vec<CString> = Vec::new();
        let arch = std::env::var("CANDLE_ROCM_ARCH")
            .ok()
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| "gfx1151".to_string());
        opts.push(CString::new(format!("--offload-arch={arch}")).unwrap());
        if opt.use_fast_math == Some(true) {
            opts.push(CString::new("--use_fast_math").unwrap());
        }
        for (name, value) in &opt.macros {
            opts.push(CString::new(format!("-D{name}={value}")).unwrap());
        }
        for path in &opt.include_paths {
            opts.push(CString::new(format!("-I{}", path.display())).unwrap());
        }
        let c_opts: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        let rc = unsafe { hiprtcCompileProgram(prog, c_opts.len() as c_int, c_opts.as_ptr()) };
        if rc != 0 {
            unsafe { hiprtcDestroyProgram(&mut prog) };
            return Err(CompileError(rc));
        }
        let mut size = 0usize;
        let rc = unsafe { hiprtcGetPTXSize(prog, &mut size) };
        if rc != 0 {
            unsafe { hiprtcDestroyProgram(&mut prog) };
            return Err(CompileError(rc));
        }
        let mut buf = vec![0 as c_char; size];
        let rc = unsafe { hiprtcGetPTX(prog, buf.as_mut_ptr()) };
        if rc != 0 {
            unsafe { hiprtcDestroyProgram(&mut prog) };
            return Err(CompileError(rc));
        }
        unsafe { hiprtcDestroyProgram(&mut prog) };
        Ok(Ptx(PtxKind::Image(buf)))
    }
}
