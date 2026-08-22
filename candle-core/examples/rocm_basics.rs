#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use half::bf16;

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

/// `b (M,K) @ (a * mul + add)` on CPU, with `a` shaped `(K,N)`, for the graph
/// cross-checks. A well-posed `(M,K)@(K,N)` gemm (N>1) mirrors the model's
/// capture and avoids the degenerate `n=1` gemv path.
fn b_ref_matmul(
    b_data: &[f32],
    a: &[f32],
    m: usize,
    k: usize,
    n: usize,
    mul: f64,
    add: f64,
    cpu: &Device,
) -> Result<Tensor> {
    let a_ref = Tensor::from_vec(a.to_vec(), (k, n), cpu)?;
    let out = a_ref.affine(mul, add)?;
    let b_ref = Tensor::from_vec(b_data.to_vec(), (m, k), cpu)?;
    Ok(b_ref.matmul(&out)?)
}

fn check(name: &str, ref_t: Tensor, got: Tensor, eps: f64) -> Result<()> {
    let err = max_abs_err(&ref_t, &got)?;
    if err > eps {
        bail!("{name}: max abs err {err:.3e} exceeds tolerance {eps}");
    }
    println!("{name:14} OK  (max abs err {err:.2e})");
    Ok(())
}

/// Relative-tolerance check for low-precision (bf16) results: the per-element
/// error must stay below `tol * max(|ref|, 1)`.
fn check_rel(name: &str, ref_v: &[f32], got_v: &[f32], tol: f64) -> Result<()> {
    if ref_v.len() != got_v.len() {
        bail!("{name}: length mismatch {} vs {}", ref_v.len(), got_v.len());
    }
    let mut err = 0f64;
    for (i, (r, g)) in ref_v.iter().zip(got_v.iter()).enumerate() {
        if !g.is_finite() {
            bail!("{name}: non-finite value at {i}");
        }
        let scale = (*r).abs().max(1.0) as f64;
        err = err.max((*g - r).abs() as f64 / scale);
    }
    if err > tol {
        bail!("{name}: max rel err {err:.3e} exceeds tolerance {tol}");
    }
    println!("{name:14} OK  (max rel err {err:.2e})");
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

    // --- Graph capture/replay: affine + matmul on a stable input buffer.
    //     The captured graph is replayed several times after overwriting the
    //     input buffer in place; each replay must reflect the new data.
    //     Arena capture is used because ROCm graphs containing
    //     hipMallocAsync nodes cannot reliably be re-instantiated or
    //     re-launched. bf16 is used because the f32 hipBLAS path is not
    //     stream-capture safe on ROCm. The captured matmul is a well-posed
    //     `(M,K)@(K,N)` gemm (N>1) so it mirrors the model's capture and does
    //     not fall onto the degenerate single-column gemv path.
    const M: usize = 16;
    const K: usize = 8;
    const N: usize = 4;
    let b_f32: Vec<f32> = (0..(M * K)).map(|i| i as f32 * 0.25 + 1.0).collect();
    let b_data: Vec<bf16> = b_f32.iter().map(|v| bf16::from_f32(*v)).collect();
    let a_f32: Vec<f32> = (0..(K * N)).map(|i| i as f32 + 1.0).collect();
    let dev = device.as_cuda_device()?;
    let a = Tensor::from_vec(
        a_f32.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
        (K, N),
        &device,
    )?;
    let b = Tensor::from_vec(b_data, (M, K), &device)?;
    let h2d = |buf: &Tensor, vals: &[bf16]| -> Result<()> {
        let (s, _) = buf.storage_and_layout();
        match &*s {
            candle_core::Storage::Cuda(c) => {
                let slice = c.as_cuda_slice::<bf16>()?;
                let mut view = unsafe { slice.as_view().as_mut_view() };
                c.device.memcpy_htod(vals, &mut view)?;
                Ok(())
            }
            _ => bail!("expected a cuda buffer"),
        }
    };
    // Stage kernel parameter vectors into persistent device buffers on warm-up
    // so the capture itself performs no host->device copies (a staged copy of
    // a temporary host buffer baked into the graph would dangle on replay).
    let _htod_guard = dev.enable_cuda_graph_htod_cache();
    // Measure the allocation budget with one eager run, then capture with a
    // pre-allocated arena so the graph has no allocation nodes.
    dev.reset_alloc_counter();
    {
        let o = a.affine(2.0, 1.0)?;
        let mm_ref = b.matmul(&o)?;
        device.synchronize()?;
        drop(mm_ref);
    }
    let budget = dev.alloc_counter();
    let arena = dev.graph_capture_arena(budget + 4 * 1024 * 1024)?;
    dev.start_graph_capture_arena(&arena)?;
    let out = a.affine(2.0, 1.0)?;
    let mm = b.matmul(&out)?;
    let exec = dev.end_graph_capture()?;
    let cases: Vec<Vec<f32>> = vec![
        (0..(K * N)).map(|i| i as f32 + 1.0).collect(),
        vec![10.0; K * N],
        (0..(K * N)).map(|i| i as f32 * -1.5 + 3.0).collect(),
    ];
    for (i, vals) in cases.iter().enumerate() {
        h2d(
            &a,
            &vals.iter().map(|v| bf16::from_f32(*v)).collect::<Vec<_>>(),
        )?;
        dev.replay_graph(&exec)?;
        device.synchronize()?;
        // bf16 has an 8-bit mantissa: compare against the f32 reference with a
        // relative tolerance scaled by the magnitude of the reference values.
        let got_out: Vec<f32> = out
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let ref_out: Vec<f32> = vals.iter().map(|v| v * 2.0 + 1.0).collect();
        check_rel(&format!("graph affine {i}"), &ref_out, &got_out, 4e-2)?;
        let got_mm: Vec<f32> = mm
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        let ref_mm = b_ref_matmul(&b_f32, vals, M, K, N, 2.0, 1.0, &cpu)?;
        let ref_mm_v: Vec<f32> = ref_mm.flatten_all()?.to_vec1()?;
        check_rel(&format!("graph matmul {i}"), &ref_mm_v, &got_mm, 4e-2)?;
    }
    drop(exec);
    drop((a, b));

    println!("rocm_basics: all checks passed");
    Ok(())
}
