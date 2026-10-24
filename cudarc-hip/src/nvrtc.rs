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

    #[cfg(not(feature = "dynamic-loading"))]
    extern "C" {
	fn hiprtc_create_program(
	    prog: *mut *mut c_void,
	    src: *const c_char,
	    name: *const c_char,
	    num_headers: c_int,
	    headers: *const *const c_char,
	    include_names: *const *const c_char,
	) -> i32;
	fn hiprtc_destroy_program(prog: *mut *mut c_void) -> i32;
	fn hiprtc_compile_program(
	    prog: *mut c_void,
	    num_opts: c_int,
	    opts: *const *const c_char,
	) -> i32;
	fn hiprtc_get_ptxsize(prog: *mut c_void, size: *mut usize) -> i32;
	fn hiprtc_get_ptx(prog: *mut c_void, ptx: *mut c_char) -> i32;
    }

    #[cfg(feature = "dynamic-loading")]
    fn load<F: Copy>(name: &str) -> F {
	// SAFETY: loading a symbol is not itself unsafe; the `unsafe fn`
	// wrappers below own the safety contract of each call.
	unsafe {
	    *nvrtclib()
		.get::<F>(name.as_bytes())
		.unwrap_or_else(|e| panic!("Missing symbol {name}: {e}"))
	}
    }

    #[cfg(feature = "dynamic-loading")]
    fn nvrtclib() -> &'static ::libloading::Library {
	static LIB: std::sync::OnceLock<::libloading::Library> = std::sync::OnceLock::new();
	LIB.get_or_init(|| {
	    let choices = crate::get_lib_name_candidates("hiprtc");
	    for choice in choices.iter() {
		// SAFETY: the candidate name comes from our own list; loading
		// a library executes no init code we do not already trust.
		if let Ok(lib) = unsafe { ::libloading::Library::new(choice) } {
		    return lib;
		}
	    }
	    crate::panic_no_lib_found("hiprtc", &choices);
	})
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hiprtc_create_program(
	prog: *mut *mut c_void,
	src: *const c_char,
	name: *const c_char,
	num_headers: c_int,
	headers: *const *const c_char,
	include_names: *const *const c_char,
    ) -> i32 {
	let hiprtc_create_program: extern "C" fn(
	    *mut *mut c_void,
	    *const c_char,
	    *const c_char,
	    c_int,
	    *const *const c_char,
	    *const *const c_char,
	) -> i32 = load("hiprtc_create_program");
	hiprtc_create_program(prog, src, name, num_headers, headers, include_names)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hiprtc_destroy_program(prog: *mut *mut c_void) -> i32 {
	let hiprtc_destroy_program: extern "C" fn(*mut *mut c_void) -> i32 =
	    load("hiprtc_destroy_program");
	hiprtc_destroy_program(prog)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hiprtc_compile_program(
	prog: *mut c_void,
	num_opts: c_int,
	opts: *const *const c_char,
    ) -> i32 {
	let hiprtc_compile_program: extern "C" fn(*mut c_void, c_int, *const *const c_char) -> i32 =
	    load("hiprtc_compile_program");
	hiprtc_compile_program(prog, num_opts, opts)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hiprtc_get_ptxsize(prog: *mut c_void, size: *mut usize) -> i32 {
	let hiprtc_get_ptxsize: extern "C" fn(*mut c_void, *mut usize) -> i32 =
	    load("hiprtc_get_ptxsize");
	hiprtc_get_ptxsize(prog, size)
    }

    #[cfg(feature = "dynamic-loading")]
    unsafe fn hiprtc_get_ptx(prog: *mut c_void, ptx: *mut c_char) -> i32 {
	let hiprtc_get_ptx: extern "C" fn(*mut c_void, *mut c_char) -> i32 = load("hiprtc_get_ptx");
	hiprtc_get_ptx(prog, ptx)
    }

    fn compile(src: &str, opt: &CompileOptions) -> Result<Ptx, CompileError> {
	let c_src = CString::new(src).map_err(|_| CompileError(7))?;
	let c_name = CString::new("candle_module").unwrap();
	let mut prog: *mut c_void = std::ptr::null_mut();
	let rc = unsafe {
	    hiprtc_create_program(
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
	let rc = unsafe { hiprtc_compile_program(prog, c_opts.len() as c_int, c_opts.as_ptr()) };
	if rc != 0 {
	    unsafe { hiprtc_destroy_program(&mut prog) };
	    return Err(CompileError(rc));
	}
	let mut size = 0usize;
	let rc = unsafe { hiprtc_get_ptxsize(prog, &mut size) };
	if rc != 0 {
	    unsafe { hiprtc_destroy_program(&mut prog) };
	    return Err(CompileError(rc));
	}
	let mut buf = vec![0 as c_char; size];
	let rc = unsafe { hiprtc_get_ptx(prog, buf.as_mut_ptr()) };
	if rc != 0 {
	    unsafe { hiprtc_destroy_program(&mut prog) };
	    return Err(CompileError(rc));
	}
	unsafe { hiprtc_destroy_program(&mut prog) };
	Ok(Ptx(PtxKind::Image(buf)))
    }
}
