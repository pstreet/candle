// The NVIDIA kernels twin for multi-vendor builds. The .cu sources are NOT
// duplicated here: the build script compiles them in place from
// ../candle-kernels/src (see build.rs), keeping a single source of truth.
// This manifest is the CUDA-semantics shape of candle-kernels' lib: the PTX
// blobs produced by the twin's own build, plus the Id/Module surface the
// backend consumes. The statically linked ffi module is deliberately absent
// (see build.rs for the collision rationale).

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 11] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Indexing,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }

    pub fn ptx_bytes(&self) -> &'static [u8] {
        self.ptx.as_bytes()
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

// FFI declarations shared with candle-kernels (single source of truth via
// #[path], no symlink needed). Declarations only: link-inert unless
// referenced. The dual build never links the twin's static archive (it is
// not built), so any reference to these symbols on the NVIDIA role fails
// loudly at link time instead of silently misbehaving.
#[path = "../../candle-kernels/src/ffi.rs"]
pub mod ffi;

// The declaration macro travels with ffi.rs (same single-source rule);
// the dual branch strips out here (this crate never sets hip-plugin),
// leaving plain extern declarations exactly as before.
#[path = "../../candle-kernels/src/kernel_decl.rs"]
mod kernel_decl;
