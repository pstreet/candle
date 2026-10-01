//! Probe: isolate which basic candle CUDA-role op fails. Steps up through
//! upload, cast (both directions), sort at increasing sizes.
use candle_core::{DType, Device, Tensor};

fn step(label: &str, r: candle_core::Result<Tensor>) {
    match r {
        Ok(t) => {
            let v = t
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let m = v.iter().map(|f| f.abs()).fold(0f32, f32::max);
            println!("OK   {label} (absmax {m:.3})");
        }
        Err(e) => println!("FAIL {label}: {e}"),
    }
}

fn main() {
    let dev = Device::new_cuda(0).expect("cuda device");
    for n in [64usize, 4096, 151_936] {
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.31 - 1.0).collect();
        println!("--- n={n}");
        let cpu = Tensor::from_vec(data.clone(), (1, n), &Device::Cpu).unwrap();
        step("upload f32", cpu.to_device(&dev));
        let g = cpu.to_device(&dev).unwrap();
        step("cast f32->bf16", g.to_dtype(DType::BF16));
        let b = g.to_dtype(DType::BF16).unwrap();
        step("cast bf16->f32", b.to_dtype(DType::F32));
        step("sort bf16", b.sort_last_dim(false).map(|t| t.0));
        let sorted = b.sort_last_dim(false).unwrap().0;
        step("cast sorted->f32", sorted.to_dtype(DType::F32));
        let sf = cpu.sort_last_dim(false).unwrap().0;
        step("upload presorted f32", sf.to_device(&dev));
        step(
            "sort f32 on gpu",
            sf.to_device(&dev)
                .unwrap()
                .sort_last_dim(false)
                .map(|t| t.0),
        );
    }
}
