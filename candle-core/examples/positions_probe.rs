//! Probe: U32 positions-tensor lifecycle on the CUDA role of a dual build:
//! host upload (from_vec), D2H readback, u32->i64 cast, and index_select
//! with the resulting ids - the exact tensor_rope_rows sequence.
use candle_core::{DType, Device, Tensor};

fn main() {
    let dev = Device::new_cuda(0).expect("cuda device");
    // 10-token prefill positions [0..10), like the failing request.
    let positions: Vec<u32> = (0..10).collect();
    let t = Tensor::from_vec(positions.clone(), (10,), &dev).unwrap();
    let back = t.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    println!("u32 roundtrip: {back:?} (want [0..10))");

    let idx = t
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_dtype(DType::I64)
        .unwrap();
    let back64 = idx.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    println!("u32->i64 cast: {back64:?}");

    // cos rows (10, 64) bf16, gather rows 0..10.
    let cos_data: Vec<f32> = (0..10 * 64).map(|i| (i % 17) as f32 * 0.031).collect();
    let cos = Tensor::from_vec(cos_data, (10, 64), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&dev)
        .unwrap();
    match cos.index_select(&idx, 0) {
        Ok(g) => {
            let v: Vec<f32> = g
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            println!(
                "index_select OK absmax {:.4} head {:?}",
                v.iter().map(|f| f.abs()).fold(0f32, f32::max),
                &v[..4]
            );
        }
        Err(e) => println!("index_select FAILED: {e}"),
    }

    // Same but with ids built directly on device (arange path).
    let ar = Tensor::arange(0u32, 10u32, &dev).unwrap();
    println!(
        "arange u32: {:?}",
        ar.flatten_all().unwrap().to_vec1::<u32>().unwrap()
    );
}
