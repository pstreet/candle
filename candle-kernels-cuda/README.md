# candle-kernels-cuda

PTX twin of `candle-kernels` for `cuda+rocm` dual-vendor builds: the SAME
`.cu` sources, compiled in place from `../candle-kernels/src` through
cudaforge/nvcc to PTX. No symlink, no duplication - single source of truth
(symlinks do not survive git ZIP or Windows checkouts).

Only the runtime-loaded JIT kernels are built. The statically linked FFI
archive (moe/mmq/mmvq) is deliberately absent: both vendors' archives export
the same C symbols, so linking both into one binary is a collision. The
NVIDIA side of a dual build serves those ops through the BackendCaps
degradation path (dequant GEMM) until symbol prefixing lands.

On build boxes with no NVIDIA GPU set `CUDA_COMPUTE_CAP` (e.g. `80`): the
default capability detection needs a driver, so GPU-less builds fall back to
the sm_80 (Ampere) baseline. With a driver present the detected arch is used.
