# Starts the whole stack: llama-swap on http://127.0.0.1:8080 (OpenAI-compatible API + web UI),
# which launches each backend on demand and unloads it per config\llama-swap.yaml.
#   powershell -ExecutionPolicy Bypass -File start.ps1        (Ctrl+C stops everything)
# Run scripts\setup.ps1 once first on a new PC.

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot

if (-not (Test-Path "$Root\bin\llama-swap\llama-swap.exe")) {
    throw "bin\ is not set up yet; run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1"
}

# Shared CUDA 12 runtime for llama.cpp / sd.cpp / audio.cpp (only present with the CUDA build);
# uv + HF caches stay inside this folder
$env:PATH = "$Root\bin\cuda;$Root\bin\uv;$env:PATH"
$env:UV_CACHE_DIR = "$Root\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$Root\tools\python"
$env:HF_HOME = "$Root\models\hf-cache"
# eSpeak-ng (text -> phonemes) for Kokoro and other audio.cpp TTS; children of llama-swap inherit these
$env:AUDIOCPP_ESPEAK_LIBRARY = "$Root\bin\espeak-ng\libespeak-ng.dll"
$env:AUDIOCPP_ESPEAK_DATA = "$Root\bin\espeak-ng\espeak-ng-data"

# Re-detect GPU/RAM and regenerate the config on every start, so it always has this PC's
# memory limits and this folder's current location (the folder can be moved or copied).
& "$Root\manage.ps1" detect
if ($LASTEXITCODE -ne 0) { throw 'manage.py detect failed' }

$config = "$Root\config\llama-swap.yaml"
$listen = (Get-Content "$Root\models\registry.json" -Raw | ConvertFrom-Json).listen
if (-not $listen) { $listen = '127.0.0.1:8080' }

Write-Host "LocalAI on http://$listen  (UI: http://$listen/ui)"
& "$Root\bin\llama-swap\llama-swap.exe" -config $config -listen $listen -watch-config
