use std::env;
use std::path::{Path, PathBuf};

// The NVIDIA twin of candle-kernels for multi-vendor builds: the SAME .cu
// sources, compiled in place from ../candle-kernels/src by cudaforge/nvcc
// into PTX (no symlink, no duplication - single source of truth). Only the
// runtime-loaded JIT kernels are built; the statically linked FFI archive
// (moe/mmq/mmvq) is deliberately absent: both vendors' archives export the
// same C symbols, so linking both into one binary is a collision. The
// NVIDIA side of a dual build serves those ops through the BackendCaps
// degradation path (dequant GEMM) until symbol prefixing lands.
fn main() {
    let src = PathBuf::from("../candle-kernels/src");
    if !src.join("affine.cu").exists() {
        panic!(
            "candle-kernels-cuda: kernel sources not found at {}",
            src.display()
        );
    }
    rerun_changed(&src);

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    build_ptx(&src, &ptx_path).expect("candle-kernels-cuda PTX build failed");
}

fn rerun_changed(src: &Path) {
    println!("cargo::rerun-if-changed=build.rs");
    // Watch the shared sources: the JIT kernel .cu files plus the headers
    // they include. The rocm_compat dir is HIP-only and not compiled here.
    for entry in std::fs::read_dir(src).expect("read kernel source dir") {
        let entry = entry.expect("read dir entry");
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.ends_with(".cu") || name.ends_with(".cuh") {
            println!("cargo::rerun-if-changed={}", path.display());
        }
    }
}

fn build_ptx(src: &Path, ptx_path: &PathBuf) -> cudaforge::Result<()> {
    use cudaforge::KernelBuilder;

    let mut builder = KernelBuilder::new()
        .source_dir(src.display().to_string()) // compile in place, no copy
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // statically compiled kernels, not built here
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // PTX is JIT-compiled by the driver, so a build box without an NVIDIA
    // GPU is fine - fall back to the sm_80 (Ampere) baseline, which covers
    // every bf16-capable arch and JIT-compiles on anything newer. With a
    // GPU present (or CUDA_COMPUTE_CAP set), use the detected arch.
    match cudaforge::detect_compute_cap() {
        Ok(cap) => builder = builder.compute_cap_arch(&cap.to_nvcc_arch()),
        Err(_) => builder = builder.compute_cap(80),
    }

    let bindings = builder.build_ptx()?;
    bindings.write(ptx_path)
}
