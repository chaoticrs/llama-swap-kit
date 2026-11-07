# workers/

Python HTTP wrappers for models that need their own code (custom architectures, research repos such as
image-to-3D). Each `workers/<id>/` is an independent uv project with its own `.venv`, started and killed by
llama-swap like any native backend, so PyTorch is only in memory while that model is in use.

- `_template/`: copied by `manage.py add <repo> --worker` (or automatically for unsupported repos).

Your own workers live next to `_template/` and stay out of git (see `.gitignore`); `manage.ps1 list`
shows the registered ones. Call a worker through llama-swap: `POST http://127.0.0.1:8080/upstream/<id>/run`.
