# bin/uv/

uv 0.13.7 (`uv-x86_64-pc-windows-msvc.zip`), portable (not on the system PATH; `start.ps1` adds it).

- `uv.exe`: creates/syncs the per-worker venvs (`workers/<id>/.venv`) and the HF->GGUF converter venv
  (`tools/convert-venv`), downloading Python itself into `tools/python` when needed. All packages come from
  one cache (`.uv-cache/`) and are hardlinked, so several workers sharing torch cost one copy on disk.
- `uvx.exe`, `uvw.exe`: run-tool and windowless variants; unused.
