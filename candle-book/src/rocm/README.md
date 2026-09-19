# Advanced ROCm usage

The `rocm` feature runs Candle on AMD GPUs via HIP. Three crates make up
the backend:

- `cudarc-hip`: a HIP port of the `cudarc` driver
  (device, streams, stream-ordered allocations, rocBLAS, stream
  capture/graphs).
- `candle-kernels`: custom kernels (MoE, MMVQ,
  MMQ, GDN helpers) compiled as HIP FFI entry points. CUDA sources are
  reused through a `rocm_compat` shim (`MISTRALRS_ROCM_COMPAT_INCLUDE`
  overrides its location); the `CANDLE_ROCM_CUDA_ARCH` values in the
  per-kernel table select the device codepath, not your GPU.
- `candle-core` `rocm` backend: GEMM dispatch (hipBLASLt with rocBLAS
  fallback), managed-weight uploads, graph capture arenas.

## Building

```bash
export ROCM_PATH=/opt/rocm          # or your local ROCm install
export CANDLE_ROCM_ARCH=gfx1151     # your GPU's gfx target
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
cargo run --example quantized-qwen3-moe --release --features rocm -- \
  --model model.gguf --prompt "Hello" -n 64 --graph
```

`CANDLE_ROCM_ARCH` is baked in at compile time: rebuilding is required
when moving to a different GPU. The full knob list (GEMM thresholds,
MMQ selection, managed weights, graph capture) is in the main
knobs, graph capture) is in the main repo README (ROCm sections) and its
troubleshooting entries.

## Writing and porting kernels

New kernels go in `candle-kernels/src/` next to the CUDA sources and are
registered in the `FFI_KERNELS` table in `candle-kernels/build.rs` with
their device codepath value. Porting an existing CUDA kernel usually
means compiling it through the `rocm_compat` headers, which map the
CUDA runtime/driver API to HIP, and checking the WMMA-family gates for
RDNA codepaths. Host-side launch decisions keyed off the GPU use the
`amd_host_cc` encoding derived from `CANDLE_ROCM_ARCH`.
