# models/

Model weights, grouped by what they do, plus the registry that drives the whole stack.

| Item | Contents |
|---|---|
| `registry.json` | Source of truth: every served model's id, backend, file, device, ttl, VRAM/RAM estimate, extra args. Written by `scripts/manage.py`; hand edits are fine, then run `manage.py render`. Also holds the memory `pools` (VRAM/RAM MiB the router may use; rewritten for this PC by `manage.py detect` unless `"pools_auto": false`), the detected `hardware`, the `roles` map (role name -> model id, set with `manage.py role`), and the `listen` address. Per model, `size_mb` is what estimates are recomputed from and `mem_manual` marks hand-set `--vram/--ram`. |
| `llm/` | Text / vision LLMs (GGUF) for llama.cpp. |
| `image/` | Image diffusion models for sd.cpp. |
| `video/` | Video diffusion models (Wan, LTX, ...) for sd.cpp. |
| `stt/` | Speech-to-text models (whisper.cpp ggml `.bin`, or audio.cpp ASR GGUFs). |
| `tts/` | Text-to-speech models (audio.cpp GGUFs). |
| `unknown/` | Files that no native backend can run, parked until their source repo is known. |
| `hf-cache/` | Created on demand: Hugging Face cache used by Python workers (`HF_HOME`). |

`manage.py add` downloads into `<kind>/<id>/`; files you drop in yourself can live anywhere and are
registered in place with `manage.py add <path>`.
