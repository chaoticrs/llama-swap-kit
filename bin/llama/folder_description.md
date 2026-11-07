# bin/llama/

llama.cpp release b12119 (`llama-b12119-bin-win-cuda-12.4-x64.zip`), unmodified.

- `llama-server.exe`: OpenAI-compatible LLM server. llama-swap starts it per model with
  `-m <gguf> [--mmproj <projector>] -c 8192`; `--fit` (on by default) picks how many layers fit in VRAM
  and leaves the rest in RAM.
- `llama-quantize.exe`: used by `scripts/manage.py` to quantize converted HF models (Q4_K_M / Q8_0).
- `llama-cli.exe`, `llama-mtmd-cli.exe`, `llama-bench.exe`, ...: standalone tools, handy for testing a model directly.
- `ggml*.dll`, `llama*.dll`, `mtmd.dll`: runtime libraries; `ggml-cpu-*.dll` are per-CPU-generation kernels picked at load.

Needs the CUDA runtime from `bin/cuda` on PATH (`start.ps1` does this).

On a PC without NVIDIA, `scripts/setup.ps1` installs the Vulkan or CPU build of the same release here
(`ggml-vulkan.dll` instead of `ggml-cuda.dll`, no `bin/cuda` needed); this file is kept across that switch.
