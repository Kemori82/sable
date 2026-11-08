# Builds the release: dist\Sable\ with Sable.exe and the DLLs it needs next to it, and dist\Sable-<version>.zip.
#   DirectML.dll           ONNX Runtime's graphics-card backend (detector, corrector)
#   vulkan-1.dll           the Vulkan loader (writer, humanizer); on a PC without a Vulkan driver they use the CPU
#   msvcp140*, vcruntime*  the Microsoft C++ runtime, so nothing has to be installed first
# Needs Rust, the Visual Studio C++ Build Tools and the Vulkan SDK (see README, Building).
#   powershell -ExecutionPolicy Bypass -File tools\package.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "$root\target" }
$version = (Select-String -Path "$root\Cargo.toml" -Pattern '^version = "(.*)"').Matches[0].Groups[1].Value
$dist = "$root\dist\Sable"
Remove-Item -Recurse -Force "$root\dist" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dist | Out-Null

Copy-Item "$target\release\Sable.exe", "$target\release\DirectML.dll" $dist
Copy-Item "$env:SystemRoot\System32\vulkan-1.dll" $dist
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
$crt = Get-ChildItem "$vs\VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT" | Sort-Object FullName | Select-Object -Last 1
foreach ($f in "msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll") { Copy-Item "$crt\$f" $dist }
Copy-Item "$root\README.md" $dist

Compress-Archive -Path "$dist\*" -DestinationPath "$root\dist\Sable-$version.zip"
Get-ChildItem $dist | Format-Table Name, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
"dist\Sable-$version.zip: {0:N0} MB" -f ((Get-Item "$root\dist\Sable-$version.zip").Length / 1MB)
