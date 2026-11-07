# workers/_template/

Starting point copied into `workers/<id>/` by `scripts/manage.py`. Not served itself.

- `server.py`: stdlib HTTP server; `GET /health`, `POST /run` (JSON in/out, files as base64). Fill in `load()`
  (runs once before the port opens, so llama-swap waits for it) and `run()`. As shipped it echoes requests,
  which is handy for testing the plumbing. Single-threaded so GPU work is serialized.
- `pyproject.toml`: dependencies for the worker. Pre-wired so `torch`/`torchvision` resolve from the CUDA 12.4
  wheel index; `package = false` because the worker is a script, not a library.
