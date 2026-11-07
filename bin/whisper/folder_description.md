# bin/whisper/

whisper.cpp release b5454 / v1.9.5, OpenBLAS CPU build (`whisper-blas-bin-x64.zip`), unmodified.

- `whisper-server.exe`: speech-to-text server. llama-swap runs it with
  `--inference-path /v1/audio/transcriptions -l auto -t 4`, so it answers the OpenAI transcription API.
  Input must be WAV unless ffmpeg is on PATH and `--convert` is added via `manage.py add --args`.
- `whisper-cli.exe`, `whisper-stream.exe`, `whisper-quantize.exe`: standalone tools.
- `parakeet-*.exe`, `test-*.exe`, `wchess.exe`, ...: extras shipped in the release; unused by the stack.
- `libopenblas.dll`, `ggml*.dll`, `SDL2.dll`: runtime libraries.

CPU-only on purpose: STT stays loaded without taking VRAM from the LLM / image models.
For GPU Whisper use an audio.cpp ASR model instead (e.g. CrisperWhisper, Parakeet).
