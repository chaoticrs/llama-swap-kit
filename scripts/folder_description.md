# scripts/

- `setup.ps1 [-Variant auto|cuda|vulkan|cpu]`: downloads and extracts every backend into `bin/<name>/` (pinned
  versions at the top) and the default STT/TTS models. `auto` (default) asks `manage.py hardware`: NVIDIA ->
  CUDA build (+ `bin/cuda` runtime), AMD/Intel with >= 3 GB dedicated VRAM -> Vulkan, otherwise CPU. llama.cpp,
  sd.cpp and audio.cpp always get the same variant; a folder holding a different one (told apart by its
  `ggml-cuda.dll` / `ggml-vulkan.dll`) is replaced, keeping its folder_description.md. whisper.cpp is the CPU
  build everywhere. On a fresh install it registers `whisper-small` and `kokoro`, then runs `manage.py detect`.
  Idempotent; delete a `bin/` folder and re-run to reinstall/upgrade it. Windows x64 only.
- `manage.py`: model manager, standard-library Python only. Run it via `..\manage.ps1` (uv-managed Python 3.13)
  or any Python 3.10+.
  - `inspect <repo|file>`: identify a model without downloading (reads GGUF/safetensors headers over HTTP Range).
  - `add <repo|file> [--quant Q --file F --id X --cpu --always-on --ttl N --args "..." --mmproj P --worker]`:
    routes it, downloads/converts if needed, registers it in `models/registry.json`, regenerates config.
    Routes: GGUF LLM -> llama.cpp; HF transformers LLM -> converted to GGUF (llama.cpp's converter in
    `tools/`) -> llama.cpp; diffusion -> sd.cpp; whisper ggml -> whisper.cpp; audio.cpp GGUF -> audio.cpp;
    text-to-speech repos with their own Python code -> TTS worker from `workers/_template_tts` (repo downloaded into `models/tts/<id>/`, its `requirements.txt` installed, registered as kind `tts` with `--voices`); anything else -> Python worker scaffold from `workers/_template`. For audio repos without a GGUF it suggests
    `audio-cpp/audio.cpp-gguf` packages whose name contains every word of the repo name.
    Local sources: a `.gguf`/whisper `.bin`/diffusion file is registered in place. A transformers LLM must be a
    whole folder (config.json + tokenizer + *.safetensors) or a safetensors file with config.json beside it; it
    is converted into `models/llm/<id>/` and the original folder is left untouched. A lone safetensors LLM
    file is refused (nothing to tell llama.cpp the architecture or tokenizer).
  - `list`, `remove <id> [--delete-files]`, `render`. `render` leaves out (with a warning) models whose files
    are missing, e.g. a registry copied from another PC.
  - `role [<name> <model-id> | <name> --unset]`: lists roles (with the standard ones: agent, chat, vision,
    reasoner, embed, stt, tts, image, video), or points one at a model. Roles become llama-swap `aliases`, so
    clients request `"model": "agent"`. Names must not collide with model ids; `remove` unsets a model's roles.
  - `check [--all]`: tests `CONTRACT.md` against the running stack, through roles only: health, roles listed in
    `/v1/models`, a chat per LLM role, a real tool call for `agent`, TTS -> STT round trip, embeddings, and
    with `--all` an image generation (slow, evicts the GPU LLM). Exits 1 on any failure.
  - `hardware`: prints the detected GPU (nvidia-smi, else the display drivers' registry entries), total RAM,
    the suitable build variant and the installed one, as JSON. Changes nothing.
  - `detect`: stores that in the registry, sets `pools` to this PC (VRAM total - 250 MiB; RAM total minus
    max(4 GiB, 30%) for Windows and apps), re-estimates every model's VRAM/RAM from its file size, warns about
    models too big for this PC, and renders. `start.ps1` runs it on every start. Set `"pools_auto": false` in
    the registry to keep hand-set pools.
  - Memory estimates feed llama-swap's capacity router; override with `--vram/--ram` if a model OOMs or is
    evicted too eagerly. Overridden entries are marked `mem_manual` and `detect` never changes them.
