//! Probe: masked_fill / where_cond at prefill scales on the CUDA role.
//! Replicates CausalMasker::make_causal_mask's tail: host U8 mask upload,
//! broadcast, where_cond with -inf.
use candle_core::{DType, Device, Tensor};

fn masked_fill(xs: &Tensor, mask: &Tensor, value: f32) -> candle_core::Result<Tensor> {
    let on_true = Tensor::new(value, xs.device())?
        .to_dtype(xs.dtype())?
        .broadcast_as(xs.shape())?;
    let res = mask.broadcast_as(xs.shape())?.where_cond(&on_true, xs)?;
    Ok(res)
}

fn one(n: usize, dev: &Device) -> (f32, f32) {
    // causal U8 mask [n,n] built on host, like make_mask.
    let mask: Vec<u8> = (0..n)
        .flat_map(|i| (0..n).map(move |j| u8::from(j > i)))
        .collect();
    let zero = Tensor::new(0.0f32, dev).unwrap();
    let causal = Tensor::from_vec(mask, (n, n), dev).unwrap();
    let m = masked_fill(
        &zero
            .to_dtype(DType::BF16)
            .unwrap()
            .broadcast_as((n, n))
            .unwrap(),
        &causal,
        f32::NEG_INFINITY,
    )
    .unwrap();
    let v: Vec<f32> = m
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let neginf = v.iter().filter(|x| x.is_infinite() && **x < 0.0).count();
    let zero = v.iter().filter(|&&x| x == 0.0).count();
    // Upper triangle (j>i) must be -inf: n*(n-1)/2 entries.
    let want_inf = n * (n - 1) / 2;
    println!(
        "n={n}: -inf={neginf} (want {want_inf}) zeros={zero} (want {})",
        n * n - want_inf
    );
    (neginf as f32, zero as f32)
}

fn main() {
    let dev = Device::new_cuda(0).expect("cuda device");
    for n in [165usize, 1138] {
        print!("CUDA: ");
        one(n, &dev);
    }
    print!("CPU : ");
    one(1138, &Device::Cpu);
}
