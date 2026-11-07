# bin/llama-swap/

llama-swap v269 (`llama-swap_269_windows_amd64.zip`): single Go binary that is the stack's front door.

- `llama-swap.exe`: listens on 127.0.0.1:8080, reads `config/llama-swap.yaml` (`-watch-config` reloads it on
  change), starts the backend process for the requested `model`, proxies the request, and unloads models
  by idle `ttl` or when the capacity router needs their VRAM/RAM. Web UI with logs and loaded models at `/ui`.
  Also exposes `/upstream/<model>/...` for backend-specific endpoints (workers, sd.cpp video jobs).
- `README.md`, `CHANGELOG.md`, `LICENSE.md`: upstream docs.
