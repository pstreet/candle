use std::env;
use std::path::PathBuf;

fn main() {
    // With dynamic-loading the HIP libs are resolved at runtime, so emit no
    // link-time dependencies at all - that mode exists precisely so binaries
    // start on machines without ROCm installed.
    let dynamic_loading = env::var_os("CARGO_FEATURE_DYNAMIC_LOADING").is_some();

    let rocm = env::var("CANDLE_ROCM_PATH")
        .ok()
        .or_else(|| env::var("ROCM_HOME").ok())
        .or_else(|| env::var("ROCM_PATH").ok())
        .filter(|p| !p.is_empty());

    if !dynamic_loading {
        match &rocm {
            Some(dir) => {
                println!("cargo::rerun-if-env-changed=CANDLE_ROCM_PATH");
                println!("cargo::rerun-if-env-changed=ROCM_HOME");
                println!("cargo::rerun-if-env-changed=ROCM_PATH");
                println!("cargo::rustc-link-search=native={}/lib", dir);
            }
            None => {
                println!("cargo::warning=cudarc-hip: no ROCm installation location found (set CANDLE_ROCM_PATH/ROCM_HOME/ROCM_PATH); relying on the default linker/loader search paths");
            }
        }
    }
    let _ = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if !dynamic_loading {
        println!("cargo::rustc-link-lib=dylib=amdhip64");
        println!("cargo::rustc-link-lib=dylib=hipblas");
        println!("cargo::rustc-link-lib=dylib=hipblaslt");
        println!("cargo::rustc-link-lib=dylib=hiprand");
        println!("cargo::rustc-link-lib=dylib=hiprtc");
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
}
