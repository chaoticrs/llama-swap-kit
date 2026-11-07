# Downloads and extracts every backend binary and the default STT/TTS models, picking the build
# that fits this PC: CUDA (NVIDIA), Vulkan (AMD/Intel with 3+ GB dedicated VRAM) or CPU.
# Then sizes the VRAM/RAM pools to this PC (manage.py detect) and writes config\llama-swap.yaml.
# Idempotent: anything already present is skipped, so re-run it after deleting a bin\<name>
# folder to upgrade that component (bump the version below first). Windows x64 only.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1                 (auto-detect)
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Variant cpu    (force a build)

param([ValidateSet('auto', 'cuda', 'vulkan', 'cpu')] [string]$Variant = 'auto')

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path $PSScriptRoot -Parent
$Bin = Join-Path $Root 'bin'
$Dl = Join-Path $Root '_downloads'
New-Item -ItemType Directory -Force $Bin, $Dl | Out-Null

$gh = 'https://github.com'
$llama = "$gh/ggml-org/llama.cpp/releases/download/b12119/llama-b12119-bin-win"
$sd = "$gh/leejet/stable-diffusion.cpp/releases/download/master-1050-58bb3c1/sd-master-58bb3c1-bin-win"
$audio = "$gh/0xShug0/audio.cpp/releases/download/v0.9.3/audio-v0.9.3-bin-windows-x64"

# Same on every PC
$Common = @(
    @{ dir = 'uv';         check = 'uv.exe';             url = "$gh/astral-sh/uv/releases/download/0.13.7/uv-x86_64-pc-windows-msvc.zip" },
    @{ dir = 'llama-swap'; check = 'llama-swap.exe';     url = "$gh/mostlygeek/llama-swap/releases/download/v269/llama-swap_269_windows_amd64.zip" },
    @{ dir = 'whisper';    check = 'whisper-server.exe'; url = "$gh/ggml-org/whisper.cpp/releases/download/b5454/whisper-blas-bin-x64.zip" }
)
# GPU backends: one build per variant. Vulkan needs no extra runtime (it comes with the GPU driver).
$GpuBuilds = @{
    cuda   = @{ llama = "$llama-cuda-12.4-x64.zip"; sd = "$sd-cuda12-x64.zip"; audiocpp = "$audio-cuda12.4.zip" }
    vulkan = @{ llama = "$llama-vulkan-x64.zip";    sd = "$sd-vulkan-x64.zip"; audiocpp = "$audio-vulkan.zip" }
    cpu    = @{ llama = "$llama-cpu-x64.zip";       sd = "$sd-cpu-x64.zip";    audiocpp = "$audio-cpu.zip" }
}
$GpuChecks = @{ llama = 'llama-server.exe'; sd = 'sd-server.exe'; audiocpp = 'audiocpp_server.exe' }

$Models = @(
    @{ path = 'models\stt\ggml-small-q5_1.bin';    url = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin' },
    @{ path = 'models\tts\kokoro-82m-q8_0.gguf';   url = 'https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf' }
)

function Get-File($url, $dest) {
    Write-Host "  downloading $url"
    & curl.exe -L --fail --retry 3 -o "$dest.part" $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
    Move-Item -Force "$dest.part" $dest
}

# Build variant of an installed backend folder, from the GPU DLL it ships with.
function Get-InstalledVariant($target) {
    if (Test-Path (Join-Path $target 'ggml-cuda.dll')) { return 'cuda' }
    if (Test-Path (Join-Path $target 'ggml-vulkan.dll')) { return 'vulkan' }
    return 'cpu'
}

function Install-Component($dir, $check, $url) {
    $target = Join-Path $Bin $dir
    $zip = Join-Path $Dl ([IO.Path]::GetFileName($url))
    if (-not (Test-Path $zip)) { Get-File $url $zip }
    # keep our folder_description.md across a reinstall / variant switch
    $desc = Join-Path $target 'folder_description.md'
    $saved = if (Test-Path $desc) { Get-Content $desc -Raw } else { $null }
    if (Test-Path $target) { Remove-Item -Recurse -Force $target }
    New-Item -ItemType Directory -Force $target | Out-Null
    & tar.exe -xf $zip -C $target
    # Some archives wrap everything in one subfolder (e.g. Release\); flatten it.
    $items = @(Get-ChildItem $target)
    if ($items.Count -eq 1 -and $items[0].PSIsContainer) {
        Get-ChildItem $items[0].FullName | Move-Item -Destination $target
        Remove-Item $items[0].FullName
    }
    if (-not (Test-Path (Join-Path $target $check))) { throw "$check not found after extracting $zip" }
    if ($saved) { [IO.File]::WriteAllText($desc, $saved) }  # UTF-8 without BOM
}

foreach ($c in $Common) {
    if (Test-Path (Join-Path $Bin "$($c.dir)\$($c.check)")) { Write-Host "[skip] bin\$($c.dir)"; continue }
    Write-Host "[get]  bin\$($c.dir)"
    Install-Component $c.dir $c.check $c.url
}

# manage.ps1 runs manage.py with uv's own Python, so the PC needs no Python install.
$manage = Join-Path $Root 'manage.ps1'
if ($Variant -eq 'auto') {
    $hw = (& $manage hardware | Out-String) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'hardware detection failed' }
    $Variant = $hw.variant
    Write-Host "detected: $($hw.gpu) ($($hw.vram_total_mb) MiB VRAM), $($hw.ram_total_mb) MiB RAM -> $Variant build"
}

foreach ($dir in $GpuChecks.Keys) {
    $target = Join-Path $Bin $dir
    $exe = Join-Path $target $GpuChecks[$dir]
    if ((Test-Path $exe) -and (Get-InstalledVariant $target) -eq $Variant) { Write-Host "[skip] bin\$dir ($Variant)"; continue }
    Write-Host "[get]  bin\$dir ($Variant)"
    Install-Component $dir $GpuChecks[$dir] $GpuBuilds[$Variant][$dir]
}

if ($Variant -eq 'cuda') {
    # CUDA 12 runtime shared by every CUDA backend (bin\cuda is put on PATH by start.ps1)
    if (Test-Path (Join-Path $Bin 'cuda\cudart64_12.dll')) { Write-Host '[skip] bin\cuda' } else {
        Write-Host '[get]  bin\cuda'
        Install-Component 'cuda' 'cudart64_12.dll' "$gh/ggml-org/llama.cpp/releases/download/b12119/cudart-llama-bin-win-cuda-12.4-x64.zip"
    }
    # audio.cpp's CUDA build also links cuFFT, which llama.cpp's runtime zip lacks. Take only that DLL
    # from audio.cpp's runtime archive.
    $cufft = Join-Path $Bin 'cuda\cufft64_11.dll'
    if (Test-Path $cufft) { Write-Host '[skip] bin\cuda\cufft64_11.dll' } else {
        Write-Host '[get]  bin\cuda\cufft64_11.dll'
        $zip = Join-Path $Dl 'audio-v0.9.3-cudart-windows-x64-cuda12.4.zip'
        if (-not (Test-Path $zip)) { Get-File "$gh/0xShug0/audio.cpp/releases/download/v0.9.3/audio-v0.9.3-cudart-windows-x64-cuda12.4.zip" $zip }
        $tmp = Join-Path $Dl 'cufft-extract'
        New-Item -ItemType Directory -Force $tmp | Out-Null
        & tar.exe -xf $zip -C $tmp '*cufft64_11.dll'
        Get-ChildItem $tmp -Recurse -Filter cufft64_11.dll | Select-Object -First 1 | Move-Item -Destination $cufft
        Remove-Item -Recurse -Force $tmp
        if (-not (Test-Path $cufft)) { throw "cufft64_11.dll not found in $zip" }
    }
}

# eSpeak-ng 1.52.0 (text -> phonemes) for Kokoro and other audio.cpp TTS models. Only an MSI is
# published; an administrative extract (msiexec /a) copies its files out without installing anything.
$espeak = Join-Path $Bin 'espeak-ng'
if (Test-Path (Join-Path $espeak 'libespeak-ng.dll')) { Write-Host '[skip] bin\espeak-ng' } else {
    Write-Host '[get]  bin\espeak-ng'
    $msi = Join-Path $Dl 'espeak-ng-1.52.0.msi'
    if (-not (Test-Path $msi)) { Get-File "$gh/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi" $msi }
    $tmp = Join-Path $Dl 'espeak-extract'
    Start-Process msiexec.exe -ArgumentList "/a `"$msi`" /qn TARGETDIR=`"$tmp`"" -Wait
    $dll = Get-ChildItem $tmp -Recurse -Filter libespeak-ng.dll | Select-Object -First 1
    if (-not $dll) { throw "libespeak-ng.dll not found in $msi" }
    New-Item -ItemType Directory -Force $espeak | Out-Null
    Copy-Item $dll.FullName $espeak
    Copy-Item (Join-Path $dll.DirectoryName 'espeak-ng-data') $espeak -Recurse
    Remove-Item -Recurse -Force $tmp
}

foreach ($m in $Models) {
    $dest = Join-Path $Root $m.path
    if (Test-Path $dest) { Write-Host "[skip] $($m.path)"; continue }
    Write-Host "[get]  $($m.path)"
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Get-File $m.url $dest
}

# Register the default speech models on a fresh install (always loaded, on CPU so they never take VRAM).
$registry = Join-Path $Root 'models\registry.json'
$known = if (Test-Path $registry) { (Get-Content $registry -Raw | ConvertFrom-Json).models.PSObject.Properties.Name } else { @() }
if ($known -notcontains 'whisper-small') { & $manage add (Join-Path $Root $Models[0].path) --id whisper-small --always-on }
if ($known -notcontains 'kokoro') { & $manage add (Join-Path $Root $Models[1].path) --id kokoro --always-on --cpu }

# Size the VRAM/RAM pools to this PC, re-estimate every model, write config\llama-swap.yaml.
& $manage detect
if ($LASTEXITCODE -ne 0) { throw 'manage.py detect failed' }

Write-Host 'setup done. Start the stack with: powershell -ExecutionPolicy Bypass -File start.ps1'
