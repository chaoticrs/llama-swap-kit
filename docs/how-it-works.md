# How it works

## The pieces

| Piece | Job |
|---|---|
| `scripts/setup.ps1` | Detects the GPU, downloads the matching engine builds into `bin/`, registers the default speech models. |
| `models/registry.json` | **The only source of truth**: every model, its engine, file, memory estimate, roles, the memory budget and the listen address. |
| `scripts/manage.py` (via `manage.ps1`) | Adds and removes models, detects hardware, and generates llama-swap's config from the registry. Standard-library Python only. |
| `config/llama-swap.yaml` | Generated. Never edit it by hand. |
| `start.ps1` | Re-detects hardware, regenerates the config, starts llama-swap. |
| llama-swap | The front door on `:8080`: routes each request by its `model` field, starts the engine, proxies, unloads. |
| Engines | llama.cpp, stable-diffusion.cpp, whisper.cpp, audio.cpp and Python workers, each started on demand on a local port. |

```
setup.ps1 ──> bin/ (engines for this GPU)
                │
manage.ps1 add ─┼─> models/registry.json ──render──> config/llama-swap.yaml
                │                                       │
start.ps1 ──detect──────────────────────────────────────┘
          └─> llama-swap :8080 ──> llama-server / sd-server / whisper-server / audiocpp_server / workers
```

## A request's path

1. Your app sends `POST /v1/chat/completions` with `"model": "chat"`.
2. llama-swap resolves `chat` (a role, stored as an alias) to the model that fills it.
3. If that model isn't running, llama-swap checks the **capacity router**: every model declares the VRAM and
   RAM it needs, and if the new one doesn't fit next to what's loaded, the cheapest-to-evict models are
   stopped first. Always-on models have a high eviction cost, so they stay.
4. It starts the engine on a free port and waits for its health check (up to 600 s).
5. It forwards the request and streams the answer back.
6. After the model's idle `ttl` (15 min for LLMs, 5 min for image), it's stopped to free memory.

## Hardware detection

`manage.py detect` (run by `setup.ps1` and every `start.ps1`):

- **GPU:** `nvidia-smi` on NVIDIA. Otherwise the display drivers' registry entries, which give dedicated
  VRAM correctly above 4 GB, unlike WMI.
- **Build:** NVIDIA -> CUDA. Another GPU with 3+ GB dedicated VRAM -> Vulkan (integrated GPUs report less).
  Otherwise CPU. llama.cpp, sd.cpp and audio.cpp always use the same build; whisper.cpp is always the CPU
  build, so speech-to-text never competes for VRAM.
- **Budget:** VRAM = total - 250 MB (rounded down to 100). RAM = total minus the larger of 4 GB or 30%,
  left for Windows and your apps.
- **Re-estimates** every model's memory for the new budget (except ones you set by hand), and warns about
  models too big for this PC.
- **Skips** registered models whose files are missing, with a warning, so a registry copied from another PC
  still starts.

## Memory estimates

From the model file size `S` (MB):

| Kind | VRAM | RAM |
|---|---|---|
| LLM on GPU, fits whole (S + 1350 ≤ budget) | S + 1350 | 600 |
| LLM on GPU, too big | min(budget, S + 700) | overflow + 600 |
| Image / video / worker | whole budget (diffusion uses whatever is free) | S + 1500 |
| audio.cpp on GPU | min(budget, 1.4·S + 500) | 600 |
| audio.cpp on CPU | 0 | 1.2·S + 1500 (its runtime alone is ~1.3 GB) |
| anything else on CPU | 0 | 1.2·S + 400 |

These are deliberately simple. Override any model with `--vram/--ram`.

An LLM that fits whole is started with `-ngl 99`: llama.cpp's own automatic fit is cautious and left 1.8 GB
of a 4 GB card unused (30 tokens/s against 57 with every layer on the GPU, measured with a 4B Q4 model). Every
LLM is also started with `-np 1`, one slot holding the whole context, since the kit serves one user. Passing
`-ngl` or `-np` in a model's `--args` overrides either.

## Roles

Roles (`agent`, `chat`, `tts`, ...) live in the registry as `role -> model id`. The generated config turns
them into llama-swap `aliases`, lists them in `/v1/models` (`includeAliasesInList`), and sets
`useModelName` so engines that check the model name (audio.cpp) receive the real id. Clients only ever see
the role, which is what lets you swap models without breaking them. The public guarantees are in
[CONTRACT.md](../CONTRACT.md).

## Folder layout

```
bin/        engines (downloaded by setup; not in git)
config/     generated config (not in git)
docs/       these docs
models/     weights + registry.json (not in git: it's your PC's state)
scripts/    setup.ps1, manage.py
workers/    _template/ for Python workers; your workers sit next to it (not in git)
tools/      created on demand: HF->GGUF converter, uv-managed Python
start.ps1, manage.ps1, CONTRACT.md
```

Every folder has a `folder_description.md` explaining what is inside.
