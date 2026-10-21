//! Probe: load each Candle CUDA kernel module through the same
//! `get_or_load_func` path the runtime uses, reporting per-module success
//! or the driver error. Diagnoses PTX-vs-driver JIT failures per blob.
#[cfg(feature = "cuda")]
mod probe {
    use candle_core::cuda_backend::kernels;
    use candle_core::Device;

    pub fn main() {
        let dev = Device::new_cuda(0).expect("cuda device");
        let Device::Cuda(d) = &dev else {
            panic!("not cuda");
        };
        let modules: &[(&str, &kernels::Module)] = &[
            ("AFFINE", &kernels::AFFINE),
            ("BINARY", &kernels::BINARY),
            ("CAST", &kernels::CAST),
            ("CONV", &kernels::CONV),
            ("FILL", &kernels::FILL),
            ("INDEXING", &kernels::INDEXING),
            ("QUANTIZED", &kernels::QUANTIZED),
            ("REDUCE", &kernels::REDUCE),
            ("SORT", &kernels::SORT),
            ("TERNARY", &kernels::TERNARY),
            ("UNARY", &kernels::UNARY),
        ];
        for (name, module) in modules {
            let ptx = module.ptx();
            let first_entry = ptx
                .lines()
                .find_map(|l| l.trim_start().strip_prefix(".visible .entry "))
                .and_then(|e| e.split_once('(').map(|(n, _)| n.trim().to_string()));
            let Some(entry) = first_entry else {
                println!("{name}: NO ENTRIES in blob (len {})", ptx.len());
                continue;
            };
            match d.get_or_load_func(&entry, module) {
                Ok(_) => println!("{name}: OK ({entry})"),
                Err(e) => println!("{name}: FAILED ({entry}): {e}"),
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn main() {
    probe::main()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("ptx_probe requires the cuda feature");
}
