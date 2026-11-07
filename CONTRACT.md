# Inference layer contract

What other layers (MCP tools, voice daemon, overlays, scripts) may rely on. Everything not listed here
is internal and can change without notice. `.\manage.ps1 check` tests this contract against the running stack.

## Address
- Base URL: `http://<listen>` from `models/registry.json` (default `http://127.0.0.1:8080`). Read it from
  there or make it configurable; don't hardcode another port.
- Started by `start.ps1`. `GET /health` returns 200 once it accepts requests.

## Ask for roles, never model ids
Send a **role** as `"model"`. The registry maps each role to whichever model fills it now, so swapping in a
better model is one `.\manage.ps1 role <role> <model>` and no client changes.

| Role | Endpoint | Guarantee |
|---|---|---|
| `agent` | `POST /v1/chat/completions` | OpenAI tool calling (`tools`, `tool_choice`) returns `message.tool_calls`. |
| `chat` | `POST /v1/chat/completions` | Plain chat, streaming (`"stream": true`) supported. |
| `vision` | `POST /v1/chat/completions` | Accepts `image_url` content parts. |
| `reasoner` | `POST /v1/chat/completions` | Strongest LLM available; may be slow to load. |
| `embed` | `POST /v1/embeddings` | Returns `data[].embedding`. Vectors change when the model changes: re-index. |
| `stt` | `POST /v1/audio/transcriptions` | Multipart `file` (WAV) + `model`; returns `{"text": ...}`. |
| `tts` | `POST /v1/audio/speech` | JSON `input`, optional `voice`; returns WAV bytes. Voice list: `GET /v1/audio/voices?model=tts`. |
| `image` | `POST /v1/images/generations` | Returns `data[]` (base64 images). |
| `video` | `POST /upstream/<model-id>/sdcpp/v1/vid_gen` | Async sd.cpp job. |

- `GET /v1/models` lists every role next to the real ids, with `capabilities` (input/output modalities,
  TTS voices). Use it to find out which roles exist on this PC. A role that isn't set isn't listed.
- Custom roles are allowed: lowercase letters, digits and `-`, starting with a letter, never equal to a
  model id.
- Python workers (no standard endpoint) are called as `POST /upstream/<model-id>/run`.

## Timing and memory (plan for these)
- **Cold start:** the first request to a role that isn't loaded blocks while it loads, up to 600 s. On the RTX
  3050 that's about 20 s for a 4B LLM and minutes for SDXL. Long-running clients should send one warm-up request.
- **Eviction:** GPU roles share VRAM. On a small GPU, requesting `image` unloads `chat`/`agent`, and the next
  chat request reloads them. `stt` and `tts` are always loaded on CPU and never evicted.
- **Idle unload:** models unload after their `ttl` (LLMs 15 min, image 5 min).
- **Reasoning models** may spend tokens on hidden `reasoning_content` before answering or calling a tool.
  Don't set `max_tokens` too low (the check uses 1024 for tool calls).
- Unload everything: `POST /api/models/unload`.

## Errors
- Unknown role or model: non-200 with a JSON `error` body. Clients should surface it, not retry in a loop.
- Backend crash or OOM during load: non-200. The next request starts it again.

## Not part of the contract
Model ids, backend ports, file paths, which backend serves a role, `config/` contents, and llama-swap
itself (it could be replaced by anything that keeps this file true).

## Changing this contract
Only add things (new roles, new optional fields). If something must change, keep the old behaviour working
until every client is updated, note it here, and run `.\manage.ps1 check`.
