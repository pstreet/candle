// cuRAND-compatible random number generation over HIPRAND.

use crate::curand::result::CurandError;
use crate::driver::{CudaStream, DevicePtrMut};
use std::sync::Arc;

pub mod sys {
    use std::os::raw::{c_double, c_float, c_ulonglong, c_void};

    pub type curandGenerator_t = *mut c_void;
    pub type curandRngType_t = i32;
    pub type curandStatus_t = i32;

    // Mirrors CUDA's CURAND_RNG_PSEUDO_PHILOX4_32_10.
    pub const CURAND_RNG_PSEUDO_PHILOX4_32_10: curandRngType_t = 405;
    pub const CURAND_RNG_PSEUDO_DEFAULT: curandRngType_t = 400;
    pub const CURAND_STATUS_SUCCESS: curandStatus_t = 0;

    extern "C" {
        pub fn hiprandCreateGenerator(
            generator: *mut curandGenerator_t,
            rng_type: curandRngType_t,
        ) -> curandStatus_t;
        pub fn hiprandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;
        pub fn hiprandSetPseudoRandomGeneratorSeed(
            generator: curandGenerator_t,
            seed: c_ulonglong,
        ) -> curandStatus_t;
        pub fn hiprandSetPseudoRandomGeneratorOffset(
            generator: curandGenerator_t,
            offset: c_ulonglong,
        ) -> curandStatus_t;
        pub fn hiprandSetStream(
            generator: curandGenerator_t,
            stream: crate::driver::sys::CUstream,
        ) -> curandStatus_t;
        pub fn hiprandGenerateUniform(
            generator: curandGenerator_t,
            output: *mut c_float,
            n: usize,
        ) -> curandStatus_t;
        pub fn hiprandGenerateUniformDouble(
            generator: curandGenerator_t,
            output: *mut c_double,
            n: usize,
        ) -> curandStatus_t;
        pub fn hiprandGenerateNormal(
            generator: curandGenerator_t,
            output: *mut c_float,
            n: usize,
            mean: c_float,
            stddev: c_float,
        ) -> curandStatus_t;
        pub fn hiprandGenerateNormalDouble(
            generator: curandGenerator_t,
            output: *mut c_double,
            n: usize,
            mean: c_double,
            stddev: c_double,
        ) -> curandStatus_t;
    }
}

pub mod result {
    use crate::curand::sys;
    use crate::driver::sys::CUstream;
    use std::os::raw::c_ulonglong;

    /// Error from a HIPRAND call, mirroring `cudarc::curand::result::CurandError`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CurandError(pub i32);

    impl std::fmt::Display for CurandError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "hiprand error {}", self.0)
        }
    }

    impl std::error::Error for CurandError {}

    pub unsafe fn check(err: i32) -> Result<(), CurandError> {
        if err == sys::CURAND_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(CurandError(err))
        }
    }

    pub unsafe fn create_generator() -> Result<sys::curandGenerator_t, CurandError> {
        let mut gen = std::ptr::null_mut();
        check(sys::hiprandCreateGenerator(
            &mut gen,
            sys::CURAND_RNG_PSEUDO_PHILOX4_32_10,
        ))?;
        Ok(gen)
    }

    pub unsafe fn destroy(gen: sys::curandGenerator_t) -> Result<(), CurandError> {
        check(sys::hiprandDestroyGenerator(gen))
    }

    pub unsafe fn set_seed(gen: sys::curandGenerator_t, seed: u64) -> Result<(), CurandError> {
        check(sys::hiprandSetPseudoRandomGeneratorSeed(
            gen,
            seed as c_ulonglong,
        ))
    }

    pub unsafe fn set_offset(gen: sys::curandGenerator_t, offset: u64) -> Result<(), CurandError> {
        check(sys::hiprandSetPseudoRandomGeneratorOffset(
            gen,
            offset as c_ulonglong,
        ))
    }

    pub unsafe fn set_stream(
        gen: sys::curandGenerator_t,
        stream: CUstream,
    ) -> Result<(), CurandError> {
        check(sys::hiprandSetStream(gen, stream))
    }

    pub trait UniformFill<T> {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut T,
            n: usize,
        ) -> Result<(), CurandError>;
    }

    pub trait NormalFill<T> {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut T,
            n: usize,
            mean: T,
            std: T,
        ) -> Result<(), CurandError>;
    }

    impl UniformFill<f32> for sys::curandGenerator_t {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut f32,
            n: usize,
        ) -> Result<(), CurandError> {
            check(sys::hiprandGenerateUniform(gen, dst as _, n))
        }
    }

    impl UniformFill<f64> for sys::curandGenerator_t {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut f64,
            n: usize,
        ) -> Result<(), CurandError> {
            check(sys::hiprandGenerateUniformDouble(gen, dst as _, n))
        }
    }

    impl NormalFill<f32> for sys::curandGenerator_t {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut f32,
            n: usize,
            mean: f32,
            std: f32,
        ) -> Result<(), CurandError> {
            check(sys::hiprandGenerateNormal(gen, dst as _, n, mean, std))
        }
    }

    impl NormalFill<f64> for sys::curandGenerator_t {
        unsafe fn fill(
            gen: sys::curandGenerator_t,
            dst: *mut f64,
            n: usize,
            mean: f64,
            std: f64,
        ) -> Result<(), CurandError> {
            check(sys::hiprandGenerateNormalDouble(
                gen, dst as _, n, mean, std,
            ))
        }
    }
}

use result::{NormalFill, UniformFill};

/// A random number generator, mirroring `cudarc::curand::CudaRng`.
#[derive(Debug)]
pub struct CudaRng {
    pub(crate) gen: sys::curandGenerator_t,
    pub(crate) stream: Arc<CudaStream>,
}

impl Drop for CudaRng {
    fn drop(&mut self) {
        let _ = unsafe { result::destroy(self.gen) };
    }
}

impl CudaRng {
    /// Constructs the RNG with the given `seed`. All calls run on `stream`.
    pub fn new(seed: u64, stream: Arc<CudaStream>) -> Result<Self, CurandError> {
        let ctx = &stream.inner.ctx;
        if let Err(e) = ctx.bind_to_thread() {
            return Err(CurandError(e.0 as i32));
        }
        let gen = unsafe { result::create_generator()? };
        unsafe { result::set_stream(gen, stream.inner.cu_stream)? };
        let mut rng = Self { gen, stream };
        rng.set_seed(seed)?;
        Ok(rng)
    }

    /// # Safety
    /// Users must ensure the stream is properly synchronized.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CurandError> {
        self.stream = stream.clone();
        result::set_stream(self.gen, stream.inner.cu_stream)
    }

    /// Re-seed the RNG.
    pub fn set_seed(&mut self, seed: u64) -> Result<(), CurandError> {
        unsafe { result::set_seed(self.gen, seed) }
    }

    /// Set the offset of the RNG.
    pub fn set_offset(&mut self, offset: u64) -> Result<(), CurandError> {
        unsafe { result::set_offset(self.gen, offset) }
    }

    /// Fill with data from a `Uniform` distribution.
    pub fn fill_with_uniform<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
    ) -> Result<(), CurandError>
    where
        sys::curandGenerator_t: UniformFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe { <sys::curandGenerator_t as UniformFill<T>>::fill(self.gen, dst as *mut T, num) }
    }

    /// Fill with data from a `Normal(mean, std)` distribution.
    pub fn fill_with_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), CurandError>
    where
        sys::curandGenerator_t: NormalFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe {
            <sys::curandGenerator_t as NormalFill<T>>::fill(self.gen, dst as *mut T, num, mean, std)
        }
    }

    /// Fill with data from a `LogNormal(mean, std)` distribution.
    pub fn fill_with_log_normal<T, Dst: DevicePtrMut<T>>(
        &self,
        dst: &mut Dst,
        mean: T,
        std: T,
    ) -> Result<(), CurandError>
    where
        T: LogNormalFill + Copy,
        sys::curandGenerator_t: NormalFill<T>,
    {
        let num = dst.len();
        let (dst, _record_dst) = dst.device_ptr_mut(&self.stream);
        unsafe {
            <sys::curandGenerator_t as NormalFill<T>>::fill(
                self.gen,
                dst as *mut T,
                num,
                mean,
                std,
            )?;
            let p = dst as *mut T;
            for i in 0..num {
                *p.add(i) = T::exp_log(*p.add(i));
            }
        }
        Ok(())
    }
}

/// Helper trait for log-normal exponentiation.
pub trait LogNormalFill: NormalFillHelper {}

pub trait NormalFillHelper {
    fn exp_log(v: Self) -> Self;
}
impl NormalFillHelper for f32 {
    #[inline]
    fn exp_log(v: f32) -> f32 {
        v.exp()
    }
}
impl NormalFillHelper for f64 {
    #[inline]
    fn exp_log(v: f64) -> f64 {
        v.exp()
    }
}
impl LogNormalFill for f32 {}
impl LogNormalFill for f64 {}
