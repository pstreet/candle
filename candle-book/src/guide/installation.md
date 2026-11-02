# Installation

## 1. Create a new rust app or library

```bash
cargo new myapp
cd myapp
```

## 2. Add the correct candle version

### Standard

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core
```

### CUDA

First, make sure that Cuda is correctly installed.
- `nvcc --version` should print information about your Cuda compiler driver.
- `nvidia-smi --query-gpu=compute_cap --format=csv` should print your GPUs compute capability, e.g. something
like:

```bash
compute_cap
8.9
```

You can also compile the Cuda kernels for a specific compute cap using the 
`CUDA_COMPUTE_CAP=<compute cap>` environment variable.

If any of the above commands errors out, please make sure to update your Cuda version.

Add the `candle-core` crate with the cuda feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "cuda"
```

### MKL

You can also see the `mkl` feature which can get faster inference on CPU.

Add the `candle-core` crate with the mkl feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "mkl"
```

### Metal

Metal is exclusive to MacOS.

Add the `candle-core` crate with the metal feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "metal"
```

### AMD Ryzen AI NPU (Strix Halo)

The `npu` feature builds a CPU-delegated NPU backend for unified-memory
devices. On Strix Halo the NPU and CPU share the same physical RAM so tensor
ops delegate to the CPU backend; the NPU device is used for dispatch and
synchronization. AIE kernel acceleration is under development.

System dependencies:

```bash
sudo apt-get install -y libxrt-dev libxrt2 libxrt-npu2 python3-xrt linux-firmware-amd-misc
```

Verify the kernel driver:

```bash
lsmod | grep amdxdna          # must show loaded
ls /dev/dri/card0 /dev/dri/card1  # device nodes
```

Add the `candle-core` crate with the npu feature:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-core --features "npu"
```

## 3. Building

Run `cargo build` to make sure everything can be correctly built.

```bash
cargo build
```
