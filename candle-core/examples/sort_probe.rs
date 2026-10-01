//! Probe: sort_last_dim and the topk fallback path on the CUDA role of a
//! dual build vs CPU. The MoE router and the CPU-fallback sampling both
//! rely on this after the fused topk was role-gated off.
use candle_core::{Device, Tensor};

fn main() {
    let n = 151_936usize;
    // Sane logit-like distribution with a few large values.
    let data: Vec<f32> = (0..n)
        .map(|i| {
            if i == 12345 {
                12.9
            } else if i % 977 == 0 {
                (i % 13) as f32 * 0.7
            } else {
                (i as f32 * 0.001) % 1.0 - 0.5
            }
        })
        .collect();

    let run = |dev: &Device| -> (Vec<f32>, Vec<u32>) {
        let t = Tensor::from_vec(data.clone(), (1, n), dev)
            .unwrap()
            .to_dtype(candle_core::DType::BF16)
            .unwrap();
        // The TopKLastDimOp::topk fallback, verbatim.
        let (values, sorted_indices) = t.sort_last_dim(false).unwrap();
        let topk_indices = sorted_indices
            .narrow(candle_core::D::Minus1, 0, 8)
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        let topk_values = values
            .narrow(candle_core::D::Minus1, 0, 8)
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        (topk_values, topk_indices)
    };

    let (cpu_vals, cpu_idx) = run(&Device::Cpu);
    println!("CPU  topk vals {cpu_vals:?} idx {cpu_idx:?}");

    let dev = Device::new_cuda(0).expect("cuda device");
    let (gpu_vals, gpu_idx) = run(&dev);
    println!("CUDA topk vals {gpu_vals:?} idx {gpu_idx:?}");

    // Also a direct full-sort comparison.
    let t_cpu = Tensor::from_vec(data.clone(), (1, n), &Device::Cpu).unwrap();
    let t_gpu = t_cpu.to_device(&dev).unwrap();
    let s_cpu = t_cpu
        .sort_last_dim(false)
        .unwrap()
        .0
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let s_gpu = t_gpu
        .sort_last_dim(false)
        .unwrap()
        .0
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mut worst = 0f32;
    let mut worst_i = 0;
    for (i, (g, c)) in s_gpu.iter().zip(s_cpu.iter()).enumerate() {
        let d = (g - c).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    println!(
        "sort probe: len {} worst diff {:.6} at {worst_i} (gpu {:?} cpu {:?})",
        s_gpu.len(),
        worst,
        s_gpu.get(worst_i),
        s_cpu.get(worst_i),
    );
}
