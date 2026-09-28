//! AMD-role shell of the quantized GPU storage in dual (`cuda+rocm`)
//! builds: the shared body bound to the hip backend's types and the
//! hipcc-built `candle-kernels` (whose FFI entries ride the companion
//! plugin; see candle-kernels/src/hip_plugin.rs).

pub use crate::hip_backend::cudarc;
use crate::hip_backend::set_gemm_reduced_precision_f16;
use crate::hip_backend::{
    kernels, CudaDevice, CudaStorage, CudaStorageSlice, WrapErr, WrapErr as _,
};

// The AMD role always has the fork managed-upload API (cudarc-hip).
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}

// Companion of the managed macro above: nothing to expand here.
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {};
}

// QStorage variant this role constructs: the `Hip` variant.
macro_rules! gpu_qstorage {
    ($e:expr) => {
        crate::quantized::QStorage::Hip($e)
    };
}

// Fused MMVQ/MMQ dispatch attempt: not active on the AMD role of a dual
// build yet; the dequant fallback serves and mistral.rs's dispatch owns
// fused routing (launcher-port follow-up in the fork journal).
macro_rules! gpu_quant_fused {
    ($($tokens:tt)*) => {};
}

include!("cuda/body.rs");

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn hip_qstorage_managed_upload_roundtrip() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let vs: Vec<f32> = (0..1024).map(|v| v as f32 * 0.5).collect();
        let (managed, baseline) = match (
            load_quantized_managed(&dev, &vs)?,
            load_quantized(&dev, &vs)?,
        ) {
            (
                crate::quantized::QStorage::Hip(managed),
                crate::quantized::QStorage::Hip(baseline),
            ) => (managed, baseline),
            _ => unreachable!("hip managed upload test requires the hip storage backend"),
        };
        assert_eq!(managed.data()?, baseline.data()?);
        let raw: Vec<u8> = vs.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&managed.data()?[..raw.len()], &raw[..]);
        let dq = managed.dequantize(vs.len())?;
        let back: Vec<f32> = dev.clone_dtoh(dq.as_cuda_slice::<f32>()?)?;
        assert_eq!(back, vs);
        Ok(())
    }

    #[test]
    fn hip_qtensor_quantize_onto_and_dequantize() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let vs: Vec<f32> = (0..1024).map(|v| v as f32 * 0.25).collect();
        let mut storage = QCudaStorage::zeros(&dev, vs.len(), GgmlDType::F32)?;
        storage.quantize_onto(&crate::CpuStorage::F32(vs.clone()))?;
        let qtensor = crate::quantized::QTensor::new(
            crate::quantized::QStorage::Hip(storage),
            crate::Shape::from_dims(&[vs.len()]),
        )?;
        let dq = qtensor.dequantize(&crate::Device::Hip(dev.clone()))?;
        let back: Vec<f32> = dq.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(back, vs);
        Ok(())
    }
}
