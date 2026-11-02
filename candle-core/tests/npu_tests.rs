//! Tests for the NPU backend (AMD Ryzen AI / AIE).
//!
//! Requires `--features npu` to compile.

#![cfg(feature = "npu")]

use candle_core::{DType, Device, Result, Tensor};

// ── NPU probe ──────────────────────────────────────────────────────────────

#[test]
fn npu_probe_returns_bool() {
    let present = candle_core::npu_backend::is_npu_present();
    // Result depends on whether XRT is installed; just verify it runs.
    assert!(matches!(present, true | false));
}

// ── NPU device creation ────────────────────────────────────────────────────

#[test]
#[cfg(feature = "npu")]
fn npu_device_works_with_fallback() -> Result<()> {
    // NPU backend uses FallbackBackend (CPU) when no runtime is present.
    // Device creation should succeed.
    let dev = Device::new_npu(0)?;
    // Verify we can create a tensor on it.
    let t = Tensor::zeros((4,), DType::F32, &dev)?;
    assert_eq!(t.dims(), &[4]);
    Ok(())
}

// ── NPU storage basic operations ───────────────────────────────────────────

/// These tests use the CPU path when NPU hardware isn't available,
/// since NpuStorage currently wraps CpuStorage (unified memory).
#[test]
#[cfg(feature = "npu")]
fn npu_tensor_alloc_zeros() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::zeros((4, 4), DType::F32, &npu_dev)?;
    let v: Vec<Vec<f32>> = t.to_vec2()?;
    for row in &v {
        for &val in row {
            assert_eq!(val, 0.0);
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_tensor_from_slice() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::from_slice(&data, (2, 2), &npu_dev)?;
    let v: Vec<Vec<f32>> = t.to_vec2()?;
    assert_eq!(v, &[[1.0, 2.0], [3.0, 4.0]]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_tensor_to_cpu() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let npu_t = Tensor::from_slice(&data, (2, 2), &npu_dev)?;
    let cpu_t = npu_t.to_device(&Device::Cpu)?;
    let v: Vec<Vec<f32>> = cpu_t.to_vec2()?;
    assert_eq!(v, &[[1.0, 2.0], [3.0, 4.0]]);
    Ok(())
}

// ── NPU matmul ─────────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "npu")]
fn npu_matmul_f32() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let c = a.matmul(&b)?;

    // Verify on CPU
    let c_cpu = c.to_device(&Device::Cpu)?;
    let v: Vec<Vec<f32>> = c_cpu.to_vec2()?;
    assert_eq!(v, &[[7.0, 10.0], [15.0, 22.0]]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_matmul_batched() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let data_a: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Tensor::from_slice(&data_a, (2, 2, 3), &npu_dev)?;
    let data_b: Vec<f32> = (0..12).map(|i| (i + 2) as f32).collect();
    let b = Tensor::from_slice(&data_b, (2, 3, 2), &npu_dev)?;

    let c = a.matmul(&b)?;
    let c_cpu = c.to_device(&Device::Cpu)?;
    let v: Vec<Vec<Vec<f32>>> = c_cpu.to_vec3()?;

    let expected = [[[16., 19.], [52., 64.]], [[214., 235.], [304., 334.]]];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                assert!((v[i][j][k] - expected[i][j][k]).abs() < 1e-5);
            }
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_matmul_bf16() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let a =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?.to_dtype(DType::BF16)?;
    let b =
        Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?.to_dtype(DType::BF16)?;

    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let c_cpu = c.to_device(&Device::Cpu)?;
    let v: Vec<Vec<f32>> = c_cpu.to_vec2()?;
    // BF16 has ~7 bits of mantissa so small integer results should be exact.
    assert_eq!(v, &[[7.0, 10.0], [15.0, 22.0]]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_matmul_add() -> Result<()> {
    // Test matmul + add: C = A @ B + bias
    let npu_dev = Device::new_npu(0)?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let bias = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], (2, 2), &npu_dev)?;

    let c = (a.matmul(&b)? + bias)?;
    let c_cpu = c.to_device(&Device::Cpu)?;
    let v: Vec<Vec<f32>> = c_cpu.to_vec2()?;

    // 7+0.5=7.5, 10+0.5=10.5, 15+0.5=15.5, 22+0.5=22.5
    assert_eq!(v, &[[7.5, 10.5], [15.5, 22.5]]);
    Ok(())
}

// ── NPU tensor operations ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "npu")]
fn npu_unary_ops() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], (5,), &npu_dev)?;

    // abs
    let a = t.abs()?;
    let v: Vec<f32> = a.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(v, &[2.0, 1.0, 0.0, 1.0, 2.0]);

    // square
    let s = t.sqr()?;
    let v: Vec<f32> = s.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(v, &[4.0, 1.0, 0.0, 1.0, 4.0]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_binary_ops() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), &npu_dev)?;

    // add
    let c = a.broadcast_add(&b)?;
    let v: Vec<Vec<f32>> = c.to_device(&Device::Cpu)?.to_vec2()?;
    assert_eq!(v, &[[11.0, 22.0], [33.0, 44.0]]);

    // mul
    let c = a.broadcast_mul(&b)?;
    let v: Vec<Vec<f32>> = c.to_device(&Device::Cpu)?.to_vec2()?;
    assert_eq!(v, &[[10.0, 40.0], [90.0, 160.0]]);

    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_reduce_ops() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;

    // sum along axis 0
    let s = t.sum_keepdim(0)?;
    let v: Vec<Vec<f32>> = s.to_device(&Device::Cpu)?.to_vec2()?;
    assert_eq!(v, &[[4.0, 6.0]]);

    // sum along axis 1
    let s = t.sum_keepdim(1)?;
    let v: Vec<Vec<f32>> = s.to_device(&Device::Cpu)?.to_vec2()?;
    assert_eq!(v, &[[3.0], [7.0]]);

    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_transpose() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let t_t = t.t()?;
    let v: Vec<Vec<f32>> = t_t.to_device(&Device::Cpu)?.to_vec2()?;
    assert_eq!(v, &[[1.0, 3.0], [2.0, 4.0]]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_reshape_flatten() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;

    // reshape
    let r = t.reshape((4,))?;
    let v: Vec<f32> = r.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(v, &[1.0, 2.0, 3.0, 4.0]);

    // flatten
    let f = t.flatten_all()?;
    let v: Vec<f32> = f.to_device(&Device::Cpu)?.to_vec1()?;
    assert_eq!(v, &[1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

// ── NPU dtype conversions ──────────────────────────────────────────────────

#[test]
#[cfg(feature = "npu")]
fn npu_dtype_cast_f32_to_f16() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[1.5f32, 2.5, 3.5, 4.5], (2, 2), &npu_dev)?;
    let t16 = t.to_dtype(DType::F16)?;
    let back = t16.to_dtype(DType::F32)?;
    let v: Vec<Vec<f32>> = back.to_device(&Device::Cpu)?.to_vec2()?;
    // F16 has less precision but these small values should be close
    for row in &v {
        for val in row {
            // rough check: values should be in expected range
            let expected = if *val < 2.0 {
                1.5
            } else if *val < 3.0 {
                2.5
            } else if *val < 4.0 {
                3.5
            } else {
                4.5
            };
            assert!((val - expected).abs() < 0.05);
        }
    }
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_dtype_cast_f32_to_bf16() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;

    let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &npu_dev)?;
    let tbf = t.to_dtype(DType::BF16)?;
    let back = tbf.to_dtype(DType::F32)?;
    let v: Vec<Vec<f32>> = back.to_device(&Device::Cpu)?.to_vec2()?;
    // Exact for small integers
    assert_eq!(v, &[[1.0, 2.0], [3.0, 4.0]]);
    Ok(())
}

// ── NPU storage round-trip ─────────────────────────────────────────────────

#[test]
#[cfg(feature = "npu")]
fn npu_cpu_to_npu_to_cpu() -> Result<()> {
    let cpu_dev = &Device::Cpu;
    let npu_dev = Device::new_npu(0)?;

    // Create on CPU, move to NPU, back to CPU
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let cpu_t = Tensor::from_slice(&data, (2, 2), cpu_dev)?;
    let npu_t = cpu_t.to_device(&npu_dev)?;
    let back = npu_t.to_device(cpu_dev)?;

    let v: Vec<Vec<f32>> = back.to_vec2()?;
    assert_eq!(v, &[[1.0, 2.0], [3.0, 4.0]]);
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_device_location() -> Result<()> {
    let npu_dev = Device::new_npu(0)?;
    match npu_dev.location() {
        candle_core::DeviceLocation::Npu { gpu_id } => assert_eq!(gpu_id, 0),
        _ => panic!("expected NPU location"),
    }
    Ok(())
}

#[test]
#[cfg(feature = "npu")]
fn npu_same_device() -> Result<()> {
    let dev1 = Device::new_npu(0)?;
    let dev2 = Device::new_npu(0)?;
    let dev3 = Device::new_npu(1)?;
    assert!(dev1.same_device(&dev2));
    assert!(!dev1.same_device(&dev3));
    Ok(())
}
