use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Kernels compiled into the JIT module blob (loaded at runtime via
// `load_module`). The moe/mmq/mmvq kernels are linked separately and are not
// part of this set.
const JIT_KERNELS: &[(&str, &str)] = &[
    ("affine", "AFFINE"),
    ("binary", "BINARY"),
    ("cast", "CAST"),
    ("conv", "CONV"),
    ("fill", "FILL"),
    ("indexing", "INDEXING"),
    ("quantized", "QUANTIZED"),
    ("reduce", "REDUCE"),
    ("sort", "SORT"),
    ("ternary", "TERNARY"),
    ("unary", "UNARY"),
];

fn rerun_changed() {
    println!("cargo::rerun-if-changed=build.rs");
    for f in [
        "src/compatibility.cuh",
        "src/cuda_utils.cuh",
        "src/binary_op_macros.cuh",
        "src/rocm_compat",
    ] {
        if Path::new(f).exists() {
            println!("cargo::rerun-if-changed={f}");
        }
    }
    for (id, _) in JIT_KERNELS {
        println!("cargo::rerun-if-changed=src/{id}.cu");
    }
    // Statically-linked FFI kernels (moe/mmvq/mmq).
    for f in ["src/moe", "src/mmq_gguf", "src/mmvq_gguf.cu"] {
        if Path::new(f).exists() {
            println!("cargo::rerun-if-changed={f}");
        }
    }
}

fn main() {
    rerun_changed();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    if env::var("CARGO_FEATURE_ROCM").is_ok() {
        build_rocm(&out_dir, &ptx_path);
    } else {
        build_cuda(&ptx_path).expect("candle-kernels CUDA build failed");
    }
}

fn build_cuda(ptx_path: &PathBuf) -> cudaforge::Result<()> {
    use cudaforge::KernelBuilder;
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .exclude(&["moe_*.cu", "mmvq_gguf.cu", "mmq_*.cu"]) // Exclude statically compiled kernels from ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
            "src/mmvq_gguf.cu",
            "src/mmq_gguf/mmq_quantize.cu",
            "src/mmq_gguf/mmq_instance_q4_0.cu",
            "src/mmq_gguf/mmq_instance_q4_1.cu",
            "src/mmq_gguf/mmq_instance_q5_0.cu",
            "src/mmq_gguf/mmq_instance_q5_1.cu",
            "src/mmq_gguf/mmq_instance_q8_0.cu",
            "src/mmq_gguf/mmq_instance_q2_k.cu",
            "src/mmq_gguf/mmq_instance_q3_k.cu",
            "src/mmq_gguf/mmq_instance_q4_k.cu",
            "src/mmq_gguf/mmq_instance_q5_k.cu",
            "src/mmq_gguf/mmq_instance_q6_k.cu",
        ])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}

fn rocm_root() -> PathBuf {
    for var in ["CANDLE_ROCM_PATH", "ROCM_HOME", "ROCM_PATH"] {
        if let Ok(p) = env::var(var) {
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    // Fall back to locating hipcc on PATH.
    if let Ok(path) = env::var("PATH") {
        for dir in path.split(':') {
            let dir = PathBuf::from(dir);
            if dir.join("hipcc").exists() {
                if let Some(parent) = dir.parent() {
                    return parent.to_path_buf();
                }
            }
        }
    }
    PathBuf::from("/opt/rocm")
}

fn rocm_arch() -> String {
    env::var("CANDLE_ROCM_ARCH")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "gfx1151".to_string())
}

// AMD cc encoding used for host-side launch decisions (mmq_x, mmq_y,
// granularity). Matches GGML_CUDA_CC_OFFSET_AMD | gfxNNNN in the kernel sources.
fn amd_host_cc(arch: &str) -> Option<String> {
    let num = arch.strip_prefix("gfx")?;
    let gfx: u32 = u32::from_str_radix(num, 16).ok()?;
    Some(format!("{:#x}", 0x0100_0000 | gfx))
}

// Statically-linked FFI kernels (moe/mmvq/mmq). Each entry is (source,
// __CUDA_ARCH__, whether to force-include <mma.h>, whether to define
// NO_BF16_KERNEL). The `CANDLE_ROCM_CUDA_ARCH` values select the device
// codepath in the sources: the dense GGUF MMQ kernels keep the non-Turing
// device arch (600, so `TURING_MMA_AVAILABLE` stays off) but run the RDNA
// WMMA tile paths, which gate on RDNA3/RDNA4 family tags; the WMMA MoE
// kernels use the default (1030).
const FFI_KERNELS: &[(&str, &str, bool, bool)] = &[
    ("src/moe/moe_wmma.cu", "1030", true, true),
    ("src/moe/moe_wmma_gguf.cu", "1030", true, true),
    ("src/moe/moe_gguf.cu", "1030", true, true),
    ("src/mmvq_gguf.cu", "1030", false, false),
    ("src/mmq_gguf/mmq_quantize.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q2_k.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q3_k.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q4_0.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q4_1.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q4_k.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q5_0.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q5_1.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q5_k.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q6_k.cu", "600", false, false),
    ("src/mmq_gguf/mmq_instance_q8_0.cu", "600", false, false),
];

fn run(args: &[String], what: &str) {
    let mut cmd = std::process::Command::new(&args[0]);
    cmd.args(&args[1..]);
    match cmd.output() {
        Ok(out) if out.status.success() => {}
        Ok(out) => panic!(
            "\n-- what: {what}\n-- program: {:?}\n-- status: {:?}\n-- stderr:\n{}",
            args[0],
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => panic!("failed to run {what} ({:?}): {e}", args[0]),
    }
}

/// Compile the statically-linked FFI kernels (moe/mmvq/mmq) into objects and
/// archive them into `OUT_DIR/libmoe.a`. Emits the `cargo:rustc-link-*`
/// directives so the binary links them.
fn build_rocm_ffis(out_dir: &PathBuf, rocm: &PathBuf, arch: &str) {
    let hipcc = rocm.join("bin/hipcc");
    let obj_root = out_dir.join("moe_obj");
    std::fs::create_dir_all(&obj_root).expect("create moe_obj dir");

    let mut objects: Vec<PathBuf> = Vec::new();
    for (src, cu_arch, mma, no_bf16) in FFI_KERNELS {
        let cu = PathBuf::from(src);
        if !cu.exists() {
            panic!("missing FFI kernel source {}", cu.display());
        }
        let obj = obj_root.join(format!("{}.o", src.replace('/', "_")));
        let mut args: Vec<String> = vec![
            hipcc.display().to_string(),
            cu.display().to_string(),
            format!("-I{}", "src/rocm_compat"),
            "-include".to_string(),
            "cuda_runtime.h".to_string(),
            "-include".to_string(),
            "cuda_thrust.h".to_string(),
        ];
        if *mma {
            args.push("-include".to_string());
            args.push("mma.h".to_string());
        }
        args.push("-DUSE_ROCM".to_string());
        args.push(format!("-DCANDLE_ROCM_CUDA_ARCH={cu_arch}"));
        if src.starts_with("src/mmq_gguf/") {
            // MMQ kernels: HIP codepath semantics + AMD-encoded host cc so the
            // host predicates agree with the WMMA device path on RDNA3/4.
            args.push("-DGGML_USE_HIP".to_string());
            if let Some(host_cc) = amd_host_cc(arch) {
                args.push(format!("-DCANDLE_ROCM_HOST_CC={host_cc}"));
            }
        }
        if *no_bf16 {
            args.push("-DNO_BF16_KERNEL".to_string());
        }
        args.extend([
            format!("--offload-arch={arch}"),
            "-std=c++17".to_string(),
            "-O3".to_string(),
            "-fPIC".to_string(),
            "-c".to_string(),
            "-o".to_string(),
            obj.display().to_string(),
        ]);
        run(&args, &format!("hipcc {src}"));
        objects.push(obj);
    }

    let lib = out_dir.join("libmoe.a");
    if lib.exists() {
        std::fs::remove_file(&lib).expect("remove stale libmoe.a");
    }
    let ar = if rocm.join("lib/llvm/bin/llvm-ar").exists() {
        rocm.join("lib/llvm/bin/llvm-ar").display().to_string()
    } else {
        "ar".to_string()
    };
    let mut ar_args: Vec<String> = vec![ar, "cr".to_string(), lib.display().to_string()];
    for o in &objects {
        ar_args.push(o.display().to_string());
    }
    run(&ar_args, "llvm-ar archive of libmoe.a");

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=stdc++");
}

fn build_rocm(out_dir: &PathBuf, ptx_path: &PathBuf) {
    let rocm = rocm_root();
    let hipcc = rocm.join("bin/hipcc");
    let bundler = rocm.join("lib/llvm/bin/clang-offload-bundler");
    let arch = rocm_arch();

    let mut buf = String::new();
    for (id, const_name) in JIT_KERNELS {
        let cu = PathBuf::from(format!("src/{id}.cu"));
        if !cu.exists() {
            panic!("missing kernel source {}", cu.display());
        }
        let o = out_dir.join(format!("{id}.o"));
        let elf = out_dir.join(format!("{id}.cubin"));

        let status = Command::new(&hipcc)
            .arg(&cu)
            .arg("-I")
            .arg("src/rocm_compat")
            .arg("-include")
            .arg("cuda_runtime.h")
            .arg(format!("--offload-arch={arch}"))
            .arg("--offload-device-only")
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-c")
            .arg("-o")
            .arg(&o)
            .status()
            .unwrap_or_else(|e| panic!("failed to run hipcc for {id}: {e}"));
        if !status.success() {
            panic!("hipcc failed for {id} ({}): {}", hipcc.display(), status);
        }

        let status = Command::new(&bundler)
            .arg("--unbundle")
            .arg("--type=o")
            .arg("--input")
            .arg(&o)
            .arg(format!("--targets=hipv4-amdgcn-amd-amdhsa--{arch}"))
            .arg("--output")
            .arg(&elf)
            .status()
            .unwrap_or_else(|e| panic!("failed to run clang-offload-bundler for {id}: {e}"));
        if !status.success() {
            panic!(
                "clang-offload-bundler failed for {id} ({}): {}",
                bundler.display(),
                status
            );
        }

        buf.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{id}.cubin\"));\n"
        ));
    }

    std::fs::write(ptx_path, buf).expect("failed to write ptx.rs");
    println!(
        "cargo:warning=candle-kernels: built {} ROCm kernels for {arch}",
        JIT_KERNELS.len()
    );

    build_rocm_ffis(&out_dir, &rocm, &arch);
}
