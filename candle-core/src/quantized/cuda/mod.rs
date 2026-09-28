//! NVIDIA-role (and single-vendor) shell of the quantized GPU storage.
//! In a dual cuda+rocm build this module is the NVIDIA role; the identical
//! body serves the AMD role through `quantized::hip`. Vendor bindings and
//! the role macros below are what the shared `body.rs` resolves against.

use crate::cuda_backend::set_gemm_reduced_precision_f16;
use crate::cuda_backend::{
    kernels, CudaDevice, CudaStorage, CudaStorageSlice, WrapErr, WrapErr as _,
};
pub use crate::cudarc;

// Fork managed-upload API: present in every single-vendor build (the fork's
// cudarc surface), dropped for the NVIDIA role of a dual build (upstream
// cudarc lacks it).
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_managed {
    ($($tokens:tt)*) => {};
}

// Inverted companion of the fork-API macro: expands only for the NVIDIA
// role of a dual build (there the managed path is dropped and the plain
// copy below serves).
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_fork_lt_else {
    ($($tokens:tt)*) => {};
}

// QStorage variant this role constructs: the `Cuda` variant in every
// shape this shell serves.
macro_rules! gpu_qstorage {
    ($e:expr) => {
        crate::quantized::QStorage::Cuda($e)
    };
}

// Fused MMVQ/MMQ dispatch attempt: active in single-vendor builds only.
// Dual builds route through the dequant fallback (see body.rs).
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
macro_rules! gpu_quant_fused {
    ($($tokens:tt)*) => {
        $($tokens)*
    };
}
#[cfg(all(feature = "cuda", feature = "rocm"))]
macro_rules! gpu_quant_fused {
    ($($tokens:tt)*) => {};
}

include!("body.rs");

// Tests run in single-vendor shapes only: the dual build's NVIDIA role is
// driver-gated and its fused paths are dispatched by mistral.rs instead.
// The AMD role carries its own coverage in quantized/hip.rs.
#[cfg(test)]
#[cfg(not(all(feature = "cuda", feature = "rocm")))]
mod test {
    use super::*;

    use super::*;

    #[test]
    fn cuda_managed_upload_roundtrip() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let vs: Vec<f32> = (0..1024).map(|v| v as f32 * 0.5).collect();
        let (managed, baseline) = match (
            load_quantized_managed(&dev, &vs)?,
            load_quantized(&dev, &vs)?,
        ) {
            (QStorage::Cuda(managed), QStorage::Cuda(baseline)) => (managed, baseline),
            _ => unreachable!("managed upload test requires the cuda storage backend"),
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
    fn cuda_managed_isq_upload_matches_copy() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let vs: Vec<f32> = (0..1024).map(|v| v as f32 * 0.25).collect();
        let cpu = crate::CpuStorage::F32(vs.clone());
        std::env::set_var("CANDLE_MANAGED_WEIGHTS", "1");
        let mut managed = QCudaStorage::zeros(&dev, vs.len(), GgmlDType::Q4_0)?;
        managed.quantize_onto(&cpu)?;
        std::env::remove_var("CANDLE_MANAGED_WEIGHTS");
        let mut baseline = QCudaStorage::zeros(&dev, vs.len(), GgmlDType::Q4_0)?;
        baseline.quantize_onto(&cpu)?;
        assert_eq!(managed.data()?, baseline.data()?);
        let dq_managed = managed.dequantize(vs.len())?;
        let dq_baseline = baseline.dequantize(vs.len())?;
        let back_managed: Vec<f32> = dev.clone_dtoh(dq_managed.as_cuda_slice::<f32>()?)?;
        let back_baseline: Vec<f32> = dev.clone_dtoh(dq_baseline.as_cuda_slice::<f32>()?)?;
        assert_eq!(back_managed, back_baseline);
        Ok(())
    }

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }
}
