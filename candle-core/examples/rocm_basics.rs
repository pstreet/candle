#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};

/// Maximum absolute error between two equal-shape float tensors (both moved to CPU).
fn max_abs_err(ref_t: &Tensor, got: &Tensor) -> Result<f64> {
    let a = ref_t
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    let b = got
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1::<f32>()?;
    if a.len() != b.len() {
        bail!("shape mismatch: {} vs {} elements", a.len(), b.len());
    }
    let mut err = 0f64;
    for i in 0..a.len() {
        let av = a[i];
        let bv = b[i];
        let d = (av - bv).abs() as f64;
        if !d.is_finite() {
            bail!("non-finite value at {i}: {av} vs {bv}");
        }
        if d > err {
            err = d;
        }
    }
    Ok(err)
}

fn check(name: &str, ref_t: Tensor, got: Tensor, eps: f64) -> Result<()> {
    let err = max_abs_err(&ref_t, &got)?;
    if err > eps {
        bail!("{name}: max abs err {err:.3e} exceeds tolerance {eps}");
    }
    println!("{name:14} OK  (max abs err {err:.2e})");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cpu = Device::Cpu;
    let eps = 1e-4;

    // --- Strided/batched matmul: (batch=1, 6x3) x (batch=1, 3x1) -> (1, 6x1) ---
    // Deterministic data so we can compare against the CPU reference.
    let col_data: Vec<f32> = (0..18).map(|i| (i as f32) * 0.1234 - 0.5).collect();
    let ker_data: Vec<f32> = vec![1., 2., 3.];
    let col = Tensor::from_vec(col_data.clone(), (1, 6, 3), &device)?;
    let ker = Tensor::from_vec(ker_data.clone(), (1, 3, 1), &device)?;
    let bm = col.matmul(&ker)?;
    device.synchronize()?;
    let col_ref = Tensor::from_vec(col_data, (1, 6, 3), &cpu)?;
    let ker_ref = Tensor::from_vec(ker_data, (1, 3, 1), &cpu)?;
    let bm_ref = col_ref.matmul(&ker_ref)?;
    check("batched matmul", bm_ref, bm.clone(), eps)?;
    drop((col, ker, bm));
    device.synchronize()?;

    // --- Plain 2D matmul: (2x3) x (3x4) -> (2x4) ---
    let a_ref = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &cpu)?;
    let b_ref = Tensor::new(
        &[
            [7f32, 8., 9., 10.],
            [11., 12., 13., 14.],
            [15., 16., 17., 18.],
        ],
        &cpu,
    )?;
    let c_ref = a_ref.matmul(&b_ref)?;
    let a = a_ref.to_device(&device)?;
    let b = b_ref.to_device(&device)?;
    let c = a.matmul(&b)?;
    device.synchronize()?;
    check("2d matmul", c_ref, c, eps)?;
    drop((a, b));
    device.synchronize()?;

    // --- conv1d (im2col + strided batched matmul): in (1,1,8) ker (1,1,3) -> (1,1,6) ---
    let x_ref = Tensor::from_vec((0..8).map(|i| i as f32 * 0.5).collect(), (1, 1, 8), &cpu)?;
    let k_ref = Tensor::from_vec(vec![1f32, 2., -1.], (1, 1, 3), &cpu)?;
    let y_ref = x_ref.conv1d(&k_ref, 0, 1, 1, 1)?;
    let x = x_ref.to_device(&device)?;
    let k = k_ref.to_device(&device)?;
    let y = x.conv1d(&k, 0, 1, 1, 1)?;
    device.synchronize()?;
    check("conv1d im2col", y_ref, y, eps)?;
    drop((x, k));
    device.synchronize()?;

    // --- Random normal (curand path) just needs to run and produce finite output ---
    let r = Tensor::randn(0f32, 1.0, (1, 4, 4), &device)?;
    device.synchronize()?;
    let r = r.to_device(&cpu)?;
    for v in r.flatten_all()?.to_vec1::<f32>()? {
        if !v.is_finite() {
            bail!("randn produced non-finite value");
        }
    }
    println!("randn          OK  (all finite)");

    println!("rocm_basics: all checks passed");
    Ok(())
}
