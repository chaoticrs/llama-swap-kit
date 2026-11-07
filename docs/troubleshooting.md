# Troubleshooting

Start with these two, which find most problems:

```powershell
.\manage.ps1 check        # which role fails, and how (stack must be running)
.\manage.ps1 detect       # hardware, budget, and warnings about missing or too-big models
```

The web UI at http://127.0.0.1:8080/ui shows live logs for every engine.

## Setup

**"running scripts is disabled on this system"**
Run scripts with `powershell -ExecutionPolicy Bypass -File <script>`, or allow local scripts once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**A download fails halfway**
Re-run `scripts\setup.ps1`; finished parts are skipped. Downloads are kept in `_downloads/` (safe to delete
once everything works).

**Wrong build picked, or you changed GPU**
Force one with `scripts\setup.ps1 -Variant cuda` (or `vulkan` / `cpu`). The engines are swapped and your
models stay. `.\manage.ps1 hardware` shows what was detected.

## Starting

**Port 8080 is already in use**
Set `"listen": "127.0.0.1:8090"` in `models/registry.json`. To reach the stack from other devices on your
network, use `"0.0.0.0:8080"`; there is no authentication, so only do this on a network you trust.

**`warning: '<id>' skipped, ... not found`**
The registry lists a model whose file isn't on this PC. Re-add it, or `.\manage.ps1 remove <id>`.

## Requests

**The first request takes very long**
The model is loading: about 20 s for a 4B LLM, minutes for big image models on a small GPU. Later requests
are fast until it's unloaded after its idle `ttl`. Long-running apps should send one warm-up request.

**Chat was fast, now it's slow again after making an image**
On a small GPU, image models need all the VRAM, so the LLM was unloaded and reloads on the next chat.
That's expected. With more VRAM, both stay loaded.

**"upstream command exited prematurely" / a model never becomes healthy**
- Out of memory: raise its estimate (`add <source> --vram <MiB> --force`), use a smaller quant, or `--cpu`.
- NVIDIA: update the GPU driver (CUDA 12 support needed).
- audio.cpp (CUDA build) silently exits if `bin/cuda/cufft64_11.dll` is missing: re-run setup.

**TTS fails with "Could not load eSpeak-ng"**
`bin/espeak-ng/` is missing or the stack wasn't started with `start.ps1` (it sets the paths). Re-run setup,
then start with `start.ps1`.

**`agent` doesn't return `tool_calls`**
- Give it room: reasoning models spend tokens thinking first. Use `max_tokens` of 1024 or more.
- Some models or chat templates don't support tools. `.\manage.ps1 check` shows whether the current one
  does; point `agent` at a model that passes (Qwen3 and Gemma 4 do).

**`curl` in PowerShell mangles JSON**
Windows PowerShell's `curl` is `Invoke-WebRequest`, and `curl.exe` loses JSON quotes. Use
`Invoke-RestMethod` (see the README) or Python.

## Models

**`add` says a file can't be served on its own**
It's a component (VAE, text encoder) or a projector. Attach it to the model that uses it: `--args "--vae
<file>"` for image models, `--mmproj <file>` for vision LLMs.

**A safetensors LLM file is refused**
llama.cpp needs the whole model folder (`config.json` + tokenizer + weights) to convert it. Download the
full repo (`add <owner/repo>` does that).

**Gated repo: 401 / 403**
Accept the license on the Hugging Face model page, then `$env:HF_TOKEN = 'hf_...'` and run `add` again.

## Starting over

- Engines: delete `bin\<name>` and re-run setup.
- Config: it's regenerated on every start, so delete `config\llama-swap.yaml` freely.
- Everything about your models: `models\registry.json`. Back it up before experimenting.
