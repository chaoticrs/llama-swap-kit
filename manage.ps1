# Runs scripts\manage.py with the uv-managed Python (bin\uv), so no system Python is needed.
#   .\manage.ps1 list | add <repo|file> [opts] | remove <id> | role [<role> <model>] | check [--all]
#                | render | detect | hardware | inspect <repo|file>
# First run downloads Python 3.13 into tools\python (about 30 MB).

$Root = $PSScriptRoot
$env:UV_CACHE_DIR = "$Root\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$Root\tools\python"
$env:HF_HOME = "$Root\models\hf-cache"
& "$Root\bin\uv\uv.exe" run --quiet --no-project --python 3.13 "$Root\scripts\manage.py" @args
exit $LASTEXITCODE
