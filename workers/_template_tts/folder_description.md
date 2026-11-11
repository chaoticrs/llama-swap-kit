# workers/_template_tts/

Template for **text-to-speech** models that need their own Python code. `manage.py add <tts-repo> --worker`
copies it to `workers/<id>/` (instead of the generic `_template/`), downloads the repo into `models/tts/<id>/`,
adds the repo's `requirements.txt` to the worker's dependencies, and registers it as a `tts` model. So it can
fill the `tts` role like Kokoro does.

- `server.py`: serves `POST /v1/audio/speech` (OpenAI shape, WAV out), `GET /v1/audio/voices` and `/health`.
  Fill in `load()`: return an object with `voices`, `sample_rate` and `synth(text, voice)`. As shipped it
  answers with a short tone, so the plumbing can be tested first.
- `pyproject.toml`: the worker's dependencies (uv project).
