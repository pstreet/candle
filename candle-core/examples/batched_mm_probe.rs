//! Probe: the MoE expert gather matmul shape on the CUDA role of a dual
//! build vs CPU. Replicates the exact op sequence of the eager expert path
//! (`mistralrs-quant` gather): broadcast the input to (n*k, 1, hidden),
//! index_select the expert weights, transpose the view, batched matmul.
use candle_core::{Device, Tensor};

fn amax(t: &Tensor) -> f32 {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::abs)
        .fold(0f32, f32::max)
}

fn main() {
    let (tokens, top_k, hidden, inter) = (1usize, 8, 2048, 768);
    let batch = tokens * top_k;

    // Known-pattern data: magnitudes like the real activations/weights.
    let a_data: Vec<f32> = (0..tokens * hidden)
        .map(|i| ((i % 17) as f32 * 0.31 - 1.0) * 0.25)
        .collect();
    let w_data: Vec<f32> = (0..64 * inter * hidden)
        .map(|i| ((i % 23) as f32 * 0.37 - 1.1) * 0.05)
        .collect();
    let ids: Vec<u32> = [36usize, 57, 47, 5, 12, 24, 30, 32]
        .iter()
        .map(|&e| e as u32)
        .collect();

    let run = |dev: &Device| -> (Tensor, Tensor, Tensor) {
        let x = Tensor::from_vec(a_data.clone(), (tokens, hidden), dev).unwrap();
        let w = Tensor::from_vec(w_data.clone(), (64, inter, hidden), dev).unwrap();
        let idx = Tensor::from_vec(ids.clone(), (tokens, top_k), dev).unwrap();

        // The eager gather path, verbatim.
        let a_expanded = x
            .broadcast_as((tokens, top_k, hidden))
            .unwrap()
            .reshape((batch, hidden))
            .unwrap()
            .unsqueeze(1)
            .unwrap();
        let flat_indices = idx.reshape((batch,)).unwrap();
        let selected_w = w.index_select(&flat_indices, 0).unwrap();
        let out = a_expanded
            .matmul(&selected_w.transpose(1, 2).unwrap())
            .unwrap()
            .squeeze(1)
            .unwrap();
        (out, selected_w, a_expanded)
    };

    let (cpu_out, cpu_sel, cpu_a) = run(&Device::Cpu);
    println!("CPU: out absmax {:.4}", amax(&cpu_out));

    let dev = Device::new_cuda(0).expect("cuda device");
    let (gpu_out, gpu_sel, gpu_a) = run(&dev);
    println!(
        "CUDA: sel absmax {:.4} / cpu {:.4} | a absmax {:.4} / cpu {:.4}",
        amax(&gpu_sel),
        amax(&cpu_sel),
        amax(&gpu_a),
        amax(&cpu_a),
    );
    println!("CUDA: out absmax {:.4}", amax(&gpu_out));

    let gpu_v = gpu_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let cpu_v = cpu_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut worst = 0f32;
    let mut worst_i = 0;
    for (i, (g, c)) in gpu_v.iter().zip(cpu_v.iter()).enumerate() {
        let d = (g - c).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    println!(
        "batched mm probe: len {} worst diff {:.6} at {worst_i} (gpu {:?} cpu {:?})",
        gpu_v.len(),
        worst,
        gpu_v.get(worst_i),
        cpu_v.get(worst_i),
    );
}
