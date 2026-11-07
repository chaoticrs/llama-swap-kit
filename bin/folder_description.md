# bin/

Prebuilt backend binaries, one folder per project so their `ggml*.dll` files never overwrite
each other (that collision broke the previous layout). Versions are pinned in `scripts/setup.ps1`;
to upgrade one, bump its URL there, delete the folder, and re-run setup.

`llama/`, `sd/` and `audiocpp/` come in one of three builds, chosen by `setup.ps1` for this PC: CUDA (NVIDIA;
listed below, what this machine has), Vulkan (AMD/Intel GPUs) or CPU. The build shows as `ggml-cuda.dll` /
`ggml-vulkan.dll` (neither = CPU); `manage.py` reads the one in `llama/` to set audio.cpp's backend.
`cuda/` exists only with the CUDA build.

| Folder | Project / version | Role |
|---|---|---|
| `llama/` | llama.cpp b12119, CUDA 12.4 | `llama-server.exe` for text + vision LLMs (GGUF); `llama-quantize.exe` used by HF conversion. |
| `sd/` | stable-diffusion.cpp master-1050-58bb3c1, CUDA 12 | `sd-server.exe` for image and video diffusion (SD1/2/XL/3, Flux, Qwen-Image, Wan, LTX...). |
| `whisper/` | whisper.cpp b5454 (v1.9.5), CPU + OpenBLAS | `whisper-server.exe` for speech-to-text, kept on CPU so it never competes for VRAM. |
| `audiocpp/` | audio.cpp v0.9.3, CUDA 12.4 (also runs `--backend cpu`) | `audiocpp_server.exe` for TTS and heavy ASR (Kokoro, Chatterbox, Qwen3-TTS, Parakeet...). |
| `llama-swap/` | llama-swap v269 | The router/process manager listening on :8080. |
| `uv/` | uv 0.13.7 | Python/venv manager for workers and the HF converter (portable, not installed system-wide). |
| `espeak-ng/` | eSpeak-ng 1.52.0 | Text-to-phoneme library + data that Kokoro (audio.cpp) needs; located via env vars set in `start.ps1`. |
| `cuda/` | CUDA runtime 12.8 (cudart, cuBLAS, cuBLASLt) + cuFFT 11 | Shared by every GPU backend via PATH (set in `start.ps1`). |
