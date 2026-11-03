# First-run installer, started by "Writing Studio.bat" when something is missing. Safe to run again.
#   1. uv (a small Python installer, kept in .tools\; nothing is installed system-wide)
#   2. Python 3.12 and .venv
#   3. PyTorch (GPU build if there is an NVIDIA card, else CPU) and the other packages
#   4. The models (about 5 GB), each checked against models.json
#   5. A Desktop shortcut
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is very slow with its progress bar
Set-Location $PSScriptRoot
$Host.UI.RawUI.WindowTitle = "Writing Studio - installing"
Write-Host "Writing Studio setup`n" -ForegroundColor Cyan

$uv = ".tools\uv.exe"
if (-not (Test-Path $uv)) {
    Write-Host "[1/5] Getting uv..."
    New-Item -ItemType Directory -Force .tools | Out-Null
    $zip = ".tools\uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    Expand-Archive $zip .tools -Force
    Remove-Item $zip
} else { Write-Host "[1/5] uv already there" }

$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[2/5] Setting up Python 3.12..."
    & $uv venv .venv --python 3.12
    if ($LASTEXITCODE) { throw "Could not set up Python." }
} else { Write-Host "[2/5] Python already there" }

$gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
Write-Host ("[3/5] Installing packages (PyTorch for " + $(if ($gpu) { "NVIDIA GPU" } else { "CPU - no NVIDIA GPU found, writing will be slow" }) + ")...")
if ($gpu) { & $uv pip install --python $py "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128 }
else { & $uv pip install --python $py "torch==2.11.0" --index-url https://download.pytorch.org/whl/cpu }
if ($LASTEXITCODE) { throw "Could not install PyTorch." }
& $uv pip install --python $py -r requirements.txt
if ($LASTEXITCODE) { throw "Could not install packages." }

Write-Host "[4/5] Downloading models..."
& $py install_models.py
if ($LASTEXITCODE) { throw "Could not download the models." }

Write-Host "[5/5] Adding a Desktop shortcut"
$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath("Desktop") + "\Writing Studio.lnk")
$lnk.TargetPath = "$PSScriptRoot\.venv\Scripts\pythonw.exe"
$lnk.Arguments = "`"$PSScriptRoot\app.py`""
$lnk.WorkingDirectory = $PSScriptRoot
$lnk.IconLocation = "$PSScriptRoot\app.ico"
$lnk.Save()

Write-Host "`nDone. Starting Writing Studio..." -ForegroundColor Green
