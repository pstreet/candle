//! Stub NPU backend when NPU support has not been compiled in.
#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct NpuDevice;

#[derive(Debug)]
pub struct NpuStorage;

impl NpuStorage {
    pub fn transfer_to_device(&self, _dst: &NpuDevice) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
}

macro_rules! fail {
    () => {
        unimplemented!("npu support has not been enabled, add `npu` feature to enable.")
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl NpuDevice {
    pub fn new_with_stream(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    pub fn id(&self) -> DeviceId {
        DeviceId(0)
    }
}

impl crate::backend::BackendStorage for NpuStorage {
    type Device = NpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn dtype(&self) -> DType {
        fail!()
    }
    fn device(&self) -> &Self::Device {
        fail!()
    }
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
}

impl crate::backend::BackendDevice for NpuDevice {
    type Storage = NpuStorage;
    fn new(_: usize) -> Result<Self> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn set_seed(&self, _: u64) -> Result<()> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn get_current_seed(&self) -> Result<u64> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }
    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }
    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        Err(Error::NotCompiledWithNpuSupport)
    }
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

pub fn gemm_reduced_precision_f16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f16(_: bool) {}
pub fn gemm_reduced_precision_bf16() -> bool {
    true
}
pub fn set_gemm_reduced_precision_bf16(_: bool) {}
pub fn gemm_reduced_precision_f32() -> bool {
    true
}
pub fn set_gemm_reduced_precision_f32(_: bool) {}
