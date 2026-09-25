use candle_core::{DType, Device, Result, Tensor};

fn sweep(device: &Device, dt: DType, shapes: &[(usize, usize, usize, usize)]) -> Result<()> {
    for &(b, m, k, n) in shapes {
        let a = Tensor::from_vec(
            (0..b * m * k)
                .map(|i| (i % 97) as f32 * 0.01)
                .collect::<Vec<_>>(),
            (b, m, k),
            device,
        )?
        .to_dtype(dt)?;
        let w = Tensor::from_vec(
            (0..b * k * n)
                .map(|i| (i % 89) as f32 * -0.01)
                .collect::<Vec<_>>(),
            (b, k, n),
            device,
        )?
        .to_dtype(dt)?;
        let _ = a.matmul(&w)?;
        let _ = a.matmul(&w)?; // warmup
        device.synchronize()?;
        let iters = 6;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = a.matmul(&w)?;
        }
        device.synchronize()?;
        let sec = t0.elapsed().as_secs_f64() / iters as f64;
        let tflops = 2.0 * (b as f64) * (m as f64) * (k as f64) * (n as f64) / sec / 1e12;
        let ms = sec * 1e3;
        let tag = match dt {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => "??",
        };
        println!("{tag} b={b:2} m={m:5} k={k:5} n={n:5}  {ms:9.3} ms  {tflops:7.1} TFLOPS");
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut shapes: Vec<(usize, usize, usize, usize)> = Vec::new();
    for &m in &[256usize, 512, 1024, 2048, 4096, 8192] {
        for &k in &[2048usize, 4096, 8192] {
            for &n in &[2048usize, 4096, 8192, 16384] {
                shapes.push((1, m, k, n));
            }
        }
    }
    // Odd shapes: historical Lt heuristic failures cluster on non-multiples.
    shapes.extend([(1, 257, 2049, 333), (1, 1023, 4097, 1025)]);
    // Batched strided GEMMs (b, m, k, n): the serving shape under concurrency.
    shapes.extend([
        (4, 256, 2048, 2048),
        (8, 512, 4096, 2048),
        (16, 256, 2048, 8192),
        (3, 1024, 4096, 4096),
    ]);

    println!(
        "lt_probe: compute={} lt_min_tokens={} rocblas_use_hipblaslt={}",
        std::env::var("CANDLE_LT_COMPUTE").unwrap_or_else(|_| "unset(16F)".to_string()),
        std::env::var("CANDLE_LT_MIN_TOKENS").unwrap_or_else(|_| "256".to_string()),
        std::env::var("ROCBLAS_USE_HIPBLASLT").unwrap_or_else(|_| "unset".to_string()),
    );
    sweep(&device, DType::F16, &shapes)?;
    sweep(&device, DType::BF16, &shapes)?;
    Ok(())
}
