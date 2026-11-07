# Adding models

Everything goes through one command:

```powershell
.\manage.ps1 add <huggingface-repo | local file or folder> [options]
```

`add` reads the model's header (GGUF metadata or safetensors tensor names) over HTTP before downloading
anything, decides which engine runs it, downloads it into `models/<kind>/<id>/`, estimates its VRAM/RAM, and
registers it. If llama-swap is running it reloads the config by itself; no restart needed.

To see what would happen without downloading anything, use `inspect`:

```powershell
.\manage.ps1 inspect unsloth/Qwen3-4B-Instruct-2507-GGUF
```

## Which engine runs what

| You give it | Engine | Endpoint |
|---|---|---|
| GGUF LLM (text or vision) | llama.cpp `llama-server` | `/v1/chat/completions` |
| Hugging Face Transformers LLM (safetensors) | converted once to GGUF, then llama.cpp | `/v1/chat/completions` |
| Diffusion checkpoint (SD 1.x/2.x/XL/3, Flux, Qwen-Image, ...) | stable-diffusion.cpp `sd-server` | `/v1/images/generations` |
| Video diffusion (Wan, LTX, ...) | stable-diffusion.cpp | `/upstream/<id>/sdcpp/v1/vid_gen` |
| Whisper ggml `.bin` | whisper.cpp | `/v1/audio/transcriptions` |
| audio.cpp GGUF (Kokoro, Chatterbox, Qwen3-TTS, Parakeet, ...) | audio.cpp | `/v1/audio/speech`, `/v1/audio/transcriptions` |
| Anything else | Python worker (`workers/<id>/`) | `/upstream/<id>/run` |

## LLMs

```powershell
.\manage.ps1 add unsloth/Qwen3-4B-Instruct-2507-GGUF                 # picks Q4_K_M
.\manage.ps1 add unsloth/Qwen3-4B-Instruct-2507-GGUF --quant Q8_0    # bigger, better
.\manage.ps1 add <repo> --file <exact-file.gguf>                     # pick the file yourself
```

- **Quant choice:** without `--quant`, the first available of Q4_K_M, Q4_K_S, IQ4_XS, Q5_K_M, Q4_0, Q6_K, Q8_0
  is used. Rough guide: model file size + about 0.7 GB should fit in your VRAM. Bigger models still run, but
  the part that doesn't fit runs from RAM and is slower.
- **Context length:** 8192 tokens by default. Change it with `--args "-c 16384"` (uses more memory).
- **Vision:** if the repo has an `mmproj` projector file, it is downloaded and attached automatically. For a
  local model pass `--mmproj <file>`. Then point the `vision` role at it: `.\manage.ps1 role vision <id>`.
- **Transformers repos** (no GGUF) are converted with llama.cpp's own converter. The first time, this sets
  up the converter in `tools/` (a few hundred MB). Models above about 3B parameters are quantized to Q4_K_M,
  smaller ones to Q8_0 (override with `--quant`). The original download is deleted afterwards unless you
  pass `--keep-source`.
- **Gated repos** (Llama, Gemma, ...): accept the license on the model page, then
  `$env:HF_TOKEN = 'hf_...'` before running `add`.

## Image and video

```powershell
.\manage.ps1 add stabilityai/stable-diffusion-xl-base-1.0
```

Single-file checkpoints (SD 1.x/2.x/XL) just work. Newer models that ship the diffusion model separately
(Flux, SD3, Wan, ...) also need their VAE and text encoders. `add` tells you when, and you pass them with
`--args`, using full paths (the engine doesn't run from your current folder):

```powershell
.\manage.ps1 add <repo> --file <model>.gguf --args "--vae C:\ai\ae.safetensors --clip_l C:\ai\clip_l.safetensors --t5xxl C:\ai\t5xxl_fp16.safetensors" --force
```

On small GPUs, sd.cpp spreads weights over VRAM, RAM and disk automatically. Big models run, just slowly.

## Speech

```powershell
.\manage.ps1 add ggerganov/whisper.cpp --file ggml-large-v3-turbo-q5_0.bin --cpu    # better STT
.\manage.ps1 role stt ggml-large-v3-turbo-q5-0
```

For text to speech, voice cloning and heavier ASR, audio.cpp has ready GGUF packages in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf). If you `add` a speech repo that
has no GGUF, `add` lists matching packages from there.

`--cpu` keeps a model off the GPU, so it never competes with your LLM. The default Whisper and Kokoro
models run on CPU and stay loaded (`--always-on`).

## Local files

```powershell
.\manage.ps1 add D:\models\my-model.gguf                 # registered in place (not copied)
.\manage.ps1 add D:\models\some-llm-folder               # Transformers folder -> converted to GGUF
```

A Transformers LLM must be the whole folder (`config.json`, tokenizer files and `*.safetensors`). A lone
`.safetensors` LLM file is refused, because llama.cpp needs the architecture and tokenizer.

## Python workers (anything else)

```powershell
.\manage.ps1 add some-org/some-research-model --worker
```

This copies `workers/_template` to `workers/<id>/`, an isolated uv project. Then:

1. add the model's packages to `workers/<id>/pyproject.toml`;
2. fill in `load()` and `run()` in `workers/<id>/server.py`;
3. run `bin\uv\uv.exe sync --project workers\<id>`.

llama-swap starts the worker when it's called and stops it when idle, so PyTorch only uses memory while you
use the model. Call it with `POST http://127.0.0.1:8080/upstream/<id>/run` (JSON in and out; files as base64).

## Memory estimates and tuning

`list` shows each model's estimated VRAM and RAM. llama-swap keeps any set of models loaded that fits the
budget (`pools` in `models/registry.json`, set by `detect`), and unloads the cheapest ones otherwise.

- A model **crashes with out of memory**: raise its estimate, e.g. `add <same source> --vram 3500 --force`.
- A model **unloads others too eagerly**: lower its estimate the same way.
- Estimates you set with `--vram/--ram` are never changed by `detect`.
- To set the budget yourself, edit `pools` and add `"pools_auto": false` to `models/registry.json`.

## Roles

The first model of each kind fills that kind's default roles: the first LLM becomes `chat` and `agent`, the
first STT model `stt`, and so on. After that you choose:

```powershell
.\manage.ps1 role                       # what's assigned
.\manage.ps1 role reasoner qwen3-14b    # use a bigger model for hard questions
.\manage.ps1 role agent --unset
```

Then run `.\manage.ps1 check` to confirm the new model meets the role's contract (for example, that `agent`
really does tool calls).
