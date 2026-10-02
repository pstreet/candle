//! Probe: does rms_norm (or neighbors) stomp an adjacent small buffer?
//! Upload a U32 canary, run the exact layer-0 prefill ops around it
//! (rms_norm on [1,10,2048] BF16), read the canary back.
use candle_core::{DType, Device, Tensor};

fn canary(dev: &Device) -> Tensor {
    Tensor::from_vec((0u32..40).collect::<Vec<_>>(), (40,), dev).unwrap()
}
fn check(c: &Tensor, label: &str) {
    let v: Vec<u32> = c.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    let bad = v
        .iter()
        .enumerate()
        .filter(|(i, x)| **x != *i as u32)
        .count();
    println!("{label}: bad={bad} head={:?}", &v[..4.min(v.len())]);
}

fn main() {
    let dev = Device::new_cuda(0).expect("cuda device");
    let c = canary(&dev);
    check(&c, "canary baseline");

    // q-like tensor [1,10,2048] bf16.
    let data: Vec<f32> = (0..10 * 2048)
        .map(|i| ((i % 17) as f32 * 0.31 - 1.0) * 0.25)
        .collect();
    let x = Tensor::from_vec(data, (1, 10, 2048), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    let w_data: Vec<f32> = (0..2048).map(|_| 1.0).collect();
    let w = Tensor::from_vec(w_data, (2048,), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    // rms_norm inline: y = x / sqrt(mean(x^2) + eps) * w
    let xc = x.contiguous().unwrap();
    let xf = xc.to_dtype(DType::F32).unwrap();
    let msd = xf
        .sqr()
        .unwrap()
        .mean_keepdim(candle_core::D::Minus1)
        .unwrap();
    let normed = xf
        .broadcast_div(&(msd + 1e-6).unwrap().sqrt().unwrap())
        .unwrap();
    let y = normed
        .to_dtype(DType::BF16)
        .unwrap()
        .broadcast_mul(&w)
        .unwrap();
    check(&c, "canary after rms_norm");
    let yv: Vec<f32> = y
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    println!(
        "rms out absmax {:.4}",
        yv.iter().map(|f| f.abs()).fold(0f32, f32::max)
    );

    // F32 matmul like q@kT scores [1,8,10,10] then softmax.
    let q = Tensor::from_vec(
        (0..8 * 10 * 128)
            .map(|i| (i % 13) as f32 * 0.01)
            .collect::<Vec<_>>(),
        (1, 8, 10, 128),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap()
    .to_device(&dev)
    .unwrap();
    let k = q.clone();
    let scores = q
        .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
        .unwrap();
    check(&c, "canary after scores gemm");
    // softmax inline.
    let max = scores
        .to_dtype(DType::F32)
        .unwrap()
        .max_keepdim(candle_core::D::Minus1)
        .unwrap();
    let shifted = scores
        .to_dtype(DType::F32)
        .unwrap()
        .broadcast_sub(&max)
        .unwrap();
    let exp = shifted.exp().unwrap();
    let denom = exp.sum_keepdim(candle_core::D::Minus1).unwrap();
    let probs = exp.broadcast_div(&denom).unwrap();
    check(&c, "canary after softmax");
    let pv: Vec<f32> = probs
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let row0: f32 = pv[..10].iter().sum();
    println!("softmax row0 sum {row0:.4} (want 1.0)");
}
