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

#[cfg(not(feature = "rocm"))]
pub struct Module {
    index: usize,
    ptx: &'static str,
}

#[cfg(not(feature = "rocm"))]
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

#[cfg(feature = "rocm")]
pub struct Module {
    index: usize,
    data: &'static [u8],
}

#[cfg(feature = "rocm")]
impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    /// The compiled amdgcn ELF image for this kernel.
    pub fn ptx_bytes(&self) -> &'static [u8] {
        self.data
    }

    /// ELF bytes are not textual; provided for call-site compatibility.
    pub fn ptx(&self) -> &'static str {
        std::str::from_utf8(self.data).unwrap_or("")
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

#[cfg(not(feature = "rocm"))]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

#[cfg(feature = "rocm")]
macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            data: ptx::$cst,
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

pub mod ffi;
