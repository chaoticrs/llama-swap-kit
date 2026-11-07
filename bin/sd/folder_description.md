# bin/sd/

stable-diffusion.cpp master-1050-58bb3c1 (`sd-master-58bb3c1-bin-win-cuda12-x64.zip`), unmodified.

- `sd-server.exe`: image/video server. Endpoints: `POST /v1/images/generations` (OpenAI),
  `/sdapi/v1/txt2img|img2img` (A1111), `POST /sdcpp/v1/vid_gen` (async video jobs), web UI at `/`.
  llama-swap starts it with `-m <checkpoint>` or `--diffusion-model <file>` plus any `--vae/--clip_l/--t5xxl/--llm`
  given at `manage.py add --args`. `--auto-fit` (default on) spreads weights over GPU, RAM and disk,
  which is what lets Flux/Wan-class models run in 4 GB VRAM (slowly).
- `sd-cli.exe`: one-shot CLI; also converts/quantizes checkpoints (`-M convert`).
- `stable-diffusion.dll`, `ggml*.dll`, `libwebp*.dll`, `webm.dll`: runtime libraries.

On a PC without NVIDIA, `scripts/setup.ps1` installs the Vulkan or CPU build of the same release here
(`ggml-vulkan.dll` instead of `ggml-cuda.dll`, no `bin/cuda` needed); this file is kept across that switch.
