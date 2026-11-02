# AMD Ryzen AI NPU (Strix Halo / XDNA 2)

The `npu` feature in Candle provides a NPU-aware backend for AMD Strix Halo
devices. On unified-memory platforms the NPU and CPU share physical RAM, so
`NpuStorage` wraps `CpuStorage` and all tensor operations delegate to the CPU
backend. The NPU device is used for dispatch and synchronization.

## Device selection

```rust
use candle_core::Device;

let device = Device::new_npu(0)?;
```

## Unified memory model

```
Tensor allocation  →  CpuStorage (malloc/mmap in unified RAM)
Matmul/attention   →  CpuStorage (BLAS/GEMM on CPU cores)
Synchronization    →  NpuStorage fence (for future AIE dispatch)
```

There is no PCIe-bound VRAM to copy data across. Device selection routes
tensors through the NPU path even though compute runs on CPU.

## MLIR-AIE + Peano (AIE kernel compilation)

To compile kernels that actually run on the AIE engine, install the
open-source toolchain:

```bash
# Create a venv (once)
python3 -m venv ~/.local/share/ironenv
source ~/.local/share/ironenv/bin/activate
pip install --upgrade pip

# MLIR-AIE 1.4.4 — compiler driver, Python API, MLIR dialects
pip install mlir_aie==1.4.4 \
  -f https://github.com/Xilinx/mlir-aie/releases/download/v1.4.4/

# Peano (llvm-aie) — per-core AIE compiler
pip install -f \
  https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  "llvm-aie==22.0.0.2026092101+0006955e"

# Verify
aiecc --version
```

## Compilation workflow

```
AIE C++ kernel  →  Peano (per-core clang)  →  .xo object
MLIR-AIE Python →  aiecc                   →  .xclbin
.xo + .xclbin   →  xclbinutil              →  linked executable
```

The compiled `.xclbin` is loaded by the XRT runtime at inference time. The
`mistralrs-npu` crate provides the XRT bridge.

## Kernel acceleration plan

The first kernel to port is GEMM (matrix multiplication), which covers:

- All linear/FC layers in transformer models
- Attention QKV projections
- Output projections

Subsequent kernels: RMSNorm, RoPE, Softmax.

See [AIE_KERNEL_PLAN.md](../mistral.rs/mistralrs-npu/AIE_KERNEL_PLAN.md)
for the full development plan.

## Troubleshooting

- **`libxrt_coreutil.so.2` not found**: Install XRT packages
  (`libxrt-dev libxrt-npu2`) or set `LD_LIBRARY_PATH`.
- **`/dev/accel/accel0` not found**: Load the `amdxdna` kernel module
  (`sudo modprobe amdxdna`). Kernel 6.17+ required.
- **Model still uses CPU**: Expected — the NPU backend currently
  delegates to CPU. Layer mapping will show `npu[0]` in the logs.
