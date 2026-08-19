use std::env;
use std::path::PathBuf;

fn main() {
    let rocm = env::var("CANDLE_ROCM_PATH")
        .ok()
        .or_else(|| env::var("ROCM_HOME").ok())
        .or_else(|| env::var("ROCM_PATH").ok())
        .filter(|p| !p.is_empty());

    match &rocm {
        Some(dir) => {
            println!("cargo::rerun-if-env-changed=CANDLE_ROCM_PATH");
            println!("cargo::rerun-if-env-changed=ROCM_HOME");
            println!("cargo::rerun-if-env-changed=ROCM_PATH");
            println!("cargo:rustc-link-search=native={}/lib", dir);
        }
        None => {
            println!("cargo:warning=cudarc-hip: no ROCm installation location found (set CANDLE_ROCM_PATH/ROCM_HOME/ROCM_PATH); relying on the default linker/loader search paths");
        }
    }
    let _ = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rustc-link-lib=dylib=hiprand");
    println!("cargo:rustc-link-lib=dylib=hiprtc");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
