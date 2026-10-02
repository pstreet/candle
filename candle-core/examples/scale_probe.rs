//! Probe: prefill-scale attention pieces on the CUDA role -
//! (1) softmax over 1138 keys, (2) KV-cache growth copies (slice_set into
//! a preallocated buffer at large offsets).
use candle_core::{DType, Device, Tensor};

fn main() {
    let dev = Device::new_cuda(0).expect("cuda device");

    // (1) Softmax over 1138 keys, 8 heads, BF16 - like decode/prefill attention.
    let n = 1138usize;
    let data: Vec<f32> = (0..8 * n)
        .map(|i| ((i % 29) as f32 * 0.13 - 1.9) * 0.5)
        .collect();
    let run_softmax = |d: &Device| -> Vec<f32> {
        let t = Tensor::from_vec(data.clone(), (8, n), d)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        // softmax_last_dim via max-sub/exp/normalize (what the eager path does).
        let f = t.to_dtype(DType::F32).unwrap();
        let m = f.max_keepdim(candle_core::D::Minus1).unwrap();
        let e = f.broadcast_sub(&m).unwrap().exp().unwrap();
        let s = e.sum_keepdim(candle_core::D::Minus1).unwrap();
        e.broadcast_div(&s)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let g = run_softmax(&dev);
    let c = run_softmax(&Device::Cpu);
    let mut worst = 0f32;
    for (a, b) in g.iter().zip(c.iter()) {
        worst = worst.max((a - b).abs());
    }
    let row0: f32 = g[..n].iter().sum();
    println!("softmax1138: worst diff {worst:.6} gpu row0 sum {row0:.4} (want 1.0)");

    // (2) KV growth: zeros (1,8,2048,128), slice_set two chunks, narrow, compare.
    let run_growth = |d: &Device| -> Vec<f32> {
        let mut buf = Tensor::zeros((1usize, 8, 2048, 128), DType::BF16, d).unwrap();
        for (off, len) in [(0usize, 512usize), (512, 626)] {
            let chunk: Vec<f32> = (0..1 * 8 * len * 128)
                .map(|i| ((i % 31) as f32 * 0.07 - 0.9) * 0.4)
                .collect();
            let t = Tensor::from_vec(chunk, (1, 8, len, 128), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
                .to_device(d)
                .unwrap();
            buf.slice_set(&t, 2, off).unwrap();
        }
        buf.narrow(2, 0, 1138)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let gg = run_growth(&dev);
    let gc = run_growth(&Device::Cpu);
    let mut worst2 = 0f32;
    let mut worst_i = 0;
    for (i, (a, b)) in gg.iter().zip(gc.iter()).enumerate() {
        let d = (a - b).abs();
        if d > worst2 {
            worst2 = d;
            worst_i = i;
        }
    }
    println!(
        "growth1138: len {} worst diff {worst2:.6} at {worst_i}",
        gg.len()
    );
}
