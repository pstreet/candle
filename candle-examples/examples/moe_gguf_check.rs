use anyhow::{bail, Result};
use candle::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle::Module;
use candle::{DType, Device, Tensor};
use candle_nn::moe;

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

fn check(name: &str, ref_t: &Tensor, got: &Tensor, eps: f64) -> Result<()> {
    let err = max_abs_err(ref_t, got)?;
    if err > eps {
        bail!("{name}: max abs err {err:.3e} exceeds tolerance {eps}");
    }
    println!("{name:34} OK  (max abs err {err:.2e})");
    Ok(())
}

fn h3(i: u64, m: u64, salt: u64) -> f32 {
    let h = i.wrapping_mul(2654435761).wrapping_add(salt) % m;
    h as f32 / (m as f32 / 2.0) - 1.0
}

// Verifies the fast_mmq (m>1) / fast_mmvq (m==1) FFI path: quantized weight (N,K)
// dequantized on CPU as reference, matmul done on the GPU.
fn check_qmatmul(
    device: &Device,
    cpu: &Device,
    name: &str,
    dtype: GgmlDType,
    n: usize,
    k: usize,
    m: usize,
) -> Result<()> {
    let w3: Vec<f32> = (0..(n * k)).map(|i| h3(i as u64, 1000, 987654)).collect();
    let w = Tensor::from_vec(w3, (n, k), cpu)?;
    let wq = QTensor::quantize(&w, dtype)?;
    let wref = wq.dequantize(cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let wq_dev = QTensor::new(QStorage::from_data(wq.data()?, device, dtype)?, (n, k))?;
    let qmm = QMatMul::from_qtensor(wq_dev)?;

    let x3: Vec<f32> = (0..(m * k)).map(|i| h3(i as u64, 500, 123456)).collect();
    let x = Tensor::from_vec(x3.clone(), (m, k), device)?;
    let y = qmm.forward(&x)?;
    device.synchronize()?;

    let mut ref_y = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for t in 0..k {
                acc += x3[i * k + t] * wref[j * k + t];
            }
            ref_y[i * n + j] = acc;
        }
    }
    let ref_t = Tensor::from_vec(ref_y, (m, n), cpu)?;
    let got = y.to_device(cpu)?;
    check(name, &ref_t, &got, 0.3)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cpu = Device::Cpu;

    const E: usize = 4;
    const N: usize = 128;
    const K: usize = 256;
    const TOPK: usize = 2;
    const M: usize = 1;

    let w3: Vec<f32> = (0..(E * N * K))
        .map(|i| h3(i as u64, 1000, 2246822519))
        .collect();
    let w = Tensor::from_vec(w3, (E, N, K), &cpu)?;
    let wq = QTensor::quantize(&w, GgmlDType::Q4K)?;
    let wref = wq.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let wq_dev = QTensor::new(
        QStorage::from_data(wq.data()?, &device, GgmlDType::Q4K)?,
        (E, N, K),
    )?;

    // x: [M, K] physical input rows
    let x3: Vec<f32> = (0..(M * K)).map(|i| h3(i as u64, 500, 42127)).collect();
    let x = Tensor::from_vec(x3.clone(), (M, K), &device)?;

    // ---- decode, no topk weights (gate/up path) ----
    // Virtual rows = M*TOPK. sorted is a permutation of [0, M*TOPK); physical input
    // row = virtual // TOPK; output row = virtual (token_id); expert = exps[slot].
    let slots = M * TOPK;
    let sorted_v: Vec<u32> = vec![0, 1]; // permutation of [0,2)
    let exps_v: Vec<u32> = vec![0, 3];
    let sorted_t = Tensor::from_vec(sorted_v.clone(), (slots,), &device)?;
    let exps_t = Tensor::from_vec(exps_v.clone(), (slots,), &device)?;
    let out = moe::moe_gemm_gguf(
        &x,
        &wq_dev,
        &None,
        &sorted_t,
        &exps_t,
        TOPK,
        false,
        DType::BF16,
    )?;
    device.synchronize()?;

    // reference driven by the kernel contract
    let mut ref_rows = vec![0f32; slots * N];
    for (slot, token_id) in sorted_v.iter().enumerate() {
        let r = *token_id as usize;
        let e = exps_v[slot] as usize;
        let input_row = r / TOPK; // physical row
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += x3[input_row * K + k] * wref[(e * N + n) * K + k];
            }
            ref_rows[r * N + n] = acc;
        }
    }
    let ref_t = Tensor::from_vec(ref_rows, (slots, N), &cpu)?;
    let got = out.to_device(&cpu)?;
    check("moe_gemm_gguf q4k decode", &ref_t, &got, 0.3)?;
    drop(out);
    device.synchronize()?;

    // ---- decode with topk weights (down path): physical input = M*TOPK rows ----
    let d3: Vec<f32> = (0..(slots * K)).map(|i| h3(i as u64, 600, 1337)).collect();
    let d = Tensor::from_vec(d3.clone(), (slots, K), &device)?;
    let sorted2_v: Vec<u32> = vec![0, 1];
    let exps2_v: Vec<u32> = vec![1, 2];
    let sorted2_t = Tensor::from_vec(sorted2_v.clone(), (slots,), &device)?;
    let exps2_t = Tensor::from_vec(exps2_v.clone(), (slots,), &device)?;
    let tw_v: Vec<f32> = vec![0.7, 0.3]; // [M, TOPK] flattened = [M*TOPK]
    let tw_t = Tensor::from_vec(tw_v.clone(), (M, TOPK), &device)?;
    let out2 = moe::moe_gemm_gguf(
        &d,
        &wq_dev,
        &Some(tw_t),
        &sorted2_t,
        &exps2_t,
        TOPK,
        false,
        DType::BF16,
    )?;
    device.synchronize()?;

    // input_index = token_id (topk weights present); scale = topk_weights[token_id]
    let mut ref_rows2 = vec![0f32; slots * N];
    for (slot, token_id) in sorted2_v.iter().enumerate() {
        let r = *token_id as usize;
        let e = exps2_v[slot] as usize;
        let scale = tw_v[r];
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += d3[r * K + k] * wref[(e * N + n) * K + k];
            }
            ref_rows2[r * N + n] = acc * scale;
        }
    }
    let ref_t2 = Tensor::from_vec(ref_rows2, (slots, N), &cpu)?;
    let got2 = out2.to_device(&cpu)?;
    check("moe_gemm_gguf q4k decode+topk", &ref_t2, &got2, 0.3)?;
    drop(out2);
    device.synchronize()?;

    // ---- Q8_0 weights ----
    let wq8 = QTensor::quantize(&w, GgmlDType::Q8_0)?;
    let wref8 = wq8.dequantize(&cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let wq8_dev = QTensor::new(
        QStorage::from_data(wq8.data()?, &device, GgmlDType::Q8_0)?,
        (E, N, K),
    )?;
    let sorted3_v: Vec<u32> = vec![0, 1];
    let exps3_v: Vec<u32> = vec![2, 0];
    let sorted3_t = Tensor::from_vec(sorted3_v.clone(), (slots,), &device)?;
    let exps3_t = Tensor::from_vec(exps3_v.clone(), (slots,), &device)?;
    let out3 = moe::moe_gemm_gguf(
        &x,
        &wq8_dev,
        &None,
        &sorted3_t,
        &exps3_t,
        TOPK,
        false,
        DType::BF16,
    )?;
    device.synchronize()?;

    let mut ref_rows3 = vec![0f32; slots * N];
    for (slot, token_id) in sorted3_v.iter().enumerate() {
        let r = *token_id as usize;
        let e = exps3_v[slot] as usize;
        let input_row = r / TOPK;
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += x3[input_row * K + k] * wref8[(e * N + n) * K + k];
            }
            ref_rows3[r * N + n] = acc;
        }
    }
    let ref_t3 = Tensor::from_vec(ref_rows3, (slots, N), &cpu)?;
    let got3 = out3.to_device(&cpu)?;
    check("moe_gemm_gguf q80 decode", &ref_t3, &got3, 0.3)?;
    drop(out3);
    device.synchronize()?;

    // ---- prefill (WMMA) path: M physical tokens, size_m = M*TOPK virtual ----
    const PM: usize = 3;
    let px3: Vec<f32> = (0..(PM * K)).map(|i| h3(i as u64, 700, 99)).collect();
    let px = Tensor::from_vec(px3.clone(), (PM, K), &device)?;
    let ps_v: Vec<u32> = vec![0, 3, 1, 4, 2, 5]; // permutation of [0, 6)
    let pe_v: Vec<u32> = vec![0, 0, 1, 1, 2, 3]; // non-decreasing (kernel requires sorted experts)
    let ps_t = Tensor::from_vec(ps_v.clone(), (PM * TOPK,), &device)?;
    let pe_t = Tensor::from_vec(pe_v.clone(), (PM * TOPK,), &device)?;
    let po = moe::moe_gemm_gguf(&px, &wq_dev, &None, &ps_t, &pe_t, TOPK, true, DType::BF16)?;
    device.synchronize()?;

    let mut pref = vec![0f32; PM * TOPK * N];
    for (slot, token_id) in ps_v.iter().enumerate() {
        let r = *token_id as usize;
        let e = pe_v[slot] as usize;
        let input_row = r / TOPK;
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += px3[input_row * K + k] * wref[(e * N + n) * K + k];
            }
            pref[r * N + n] = acc;
        }
    }
    let pref_t = Tensor::from_vec(pref, (PM * TOPK, N), &cpu)?;
    let pgot = po.to_device(&cpu)?;
    check("moe_gemm_gguf q4k prefill", &pref_t, &pgot, 0.5)?;
    drop(po);
    device.synchronize()?;

    // prefill down path (topk weights present)
    let pd3: Vec<f32> = (0..(PM * TOPK * K))
        .map(|i| h3(i as u64, 800, 31337))
        .collect();
    let pd = Tensor::from_vec(pd3.clone(), (PM * TOPK, K), &device)?;
    let ptw: Vec<f32> = vec![0.5, 0.2, 0.8, 0.1, 0.3, 0.9];
    let ptw_t = Tensor::from_vec(ptw.clone(), (PM, TOPK), &device)?;
    let pdo = moe::moe_gemm_gguf(
        &pd,
        &wq8_dev,
        &Some(ptw_t),
        &ps_t,
        &pe_t,
        TOPK,
        true,
        DType::BF16,
    )?;
    device.synchronize()?;

    let mut pref2 = vec![0f32; PM * TOPK * N];
    for (slot, token_id) in ps_v.iter().enumerate() {
        let r = *token_id as usize;
        let e = pe_v[slot] as usize;
        let scale = ptw[r];
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..K {
                acc += pd3[r * K + k] * wref8[(e * N + n) * K + k];
            }
            pref2[r * N + n] = acc * scale;
        }
    }
    let pref2_t = Tensor::from_vec(pref2, (PM * TOPK, N), &cpu)?;
    let pgot2 = pdo.to_device(&cpu)?;
    check("moe_gemm_gguf q80 prefill+topk", &pref2_t, &pgot2, 0.5)?;
    drop(pdo);
    device.synchronize()?;

    // ---- fast_mmq / fast_mmvq (attention-style qmatmul) ----
    for (dt, dname) in [
        (GgmlDType::Q6K, "q6k"),
        (GgmlDType::Q4K, "q4k"),
        (GgmlDType::Q8_0, "q80"),
    ] {
        let n16 = format!("qmatmul mmq {dname} m=16");
        check_qmatmul(&device, &cpu, &n16, dt, 128, 256, 16)?;
        device.synchronize()?;
        let n1 = format!("qmatmul mmvq {dname} m=1");
        check_qmatmul(&device, &cpu, &n1, dt, 128, 256, 1)?;
        device.synchronize()?;
    }

    println!("moe_gguf_check: all checks passed");
    Ok(())
}
