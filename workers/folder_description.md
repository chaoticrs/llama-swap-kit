# workers/

Python HTTP wrappers for models that need their own code (custom architectures, research repos such as
image-to-3D). Each `workers/<id>/` is an independent uv project with its own `.venv`, started and killed by
llama-swap like any native backend, so PyTorch is only in memory while that model is in use.

- `_template/`: generic worker (`POST /run`, JSON in/out). Copied by `manage.py add <repo> --worker` (or
  automatically for unsupported repos). Call it through llama-swap: `POST http://127.0.0.1:8080/upstream/<id>/run`.
- `_template_tts/`: text-to-speech worker serving the standard `POST /v1/audio/speech`, so it can fill the `tts`
  role. Used automatically when the repo's pipeline is text-to-speech; the repo is downloaded into
  `models/tts/<id>/` and the worker gets `--model-dir` pointing there.

Your own workers live next to the templates and stay out of git (see `.gitignore`); `manage.ps1 list`
shows the registered ones.
