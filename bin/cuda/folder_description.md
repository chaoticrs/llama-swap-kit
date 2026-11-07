# bin/cuda/

Shared NVIDIA CUDA 12 runtime used by every GPU backend (llama.cpp, sd.cpp, audio.cpp). `start.ps1`
puts this folder on PATH, so the backend folders don't each carry ~750 MB of copies. Only installed
with the CUDA build (NVIDIA GPUs); `scripts/setup.ps1` skips it for Vulkan/CPU.

- `cudart64_12.dll`: CUDA 12 runtime.
- `cublas64_12.dll`, `cublasLt64_12.dll`: cuBLAS 12 (matrix multiplication).
- `cufft64_11.dll`: cuFFT 11 (from audio.cpp's CUDA 12.4 runtime archive). audio.cpp's server will not start
  without it (Windows error 0xC0000135, no message), even for CPU-backend models.

Source: `scripts/setup.ps1` installs llama.cpp's `cudart-llama-bin-win-cuda-12.4-x64.zip` (CUDA 12.4, what the
backends are built against). Any newer 12.x runtime also works (CUDA minor-version compatibility). Needs an
NVIDIA driver with CUDA 12 support (R528 or newer; update it if the backends fail to start). No CUDA toolkit
install is needed.
