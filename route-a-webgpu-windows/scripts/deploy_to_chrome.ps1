# Deploy the built accelerator DLL into a Chromium output dir.

param(
    [Parameter(Mandatory=$true)] [string]$ChromeOutDir,
    [ValidateSet("dbg","opt")]   [string]$Mode = "dbg",
    [string]$ChromiumSrc = "C:\Users\awx_localadmin\workspace\chromium\src"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ChromeOutDir)) { throw "Chrome out dir not found: $ChromeOutDir" }

$LITERT = Join-Path $ChromiumSrc "third_party\litert\src"
$cfg    = if ($Mode -eq "dbg") { "x64_windows-dbg" } else { "x64_windows-opt" }

$dllSrc = Join-Path $LITERT "bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll"
$pdbSrc = Join-Path $LITERT "bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.pdb"

if (-not (Test-Path $dllSrc)) { throw "Source missing: $dllSrc" }

function Copy-Overwrite($src, $tgt) {
    if (Test-Path $tgt) {
        # Rename existing (allowed even if the file is opened by Chrome).
        try { Rename-Item $tgt "$tgt.old" -Force -ErrorAction Stop } catch {}
    }
    Copy-Item -Force $src $tgt
    $size = (Get-Item $tgt).Length
    Write-Host ("  {0}  {1:N0} bytes" -f (Split-Path $tgt -Leaf), $size)
}

Write-Host "Deploying to $ChromeOutDir"
Copy-Overwrite $dllSrc  (Join-Path $ChromeOutDir "libLiteRtWebGpuAccelerator.dll")
if (Test-Path $pdbSrc) {
    Copy-Overwrite $pdbSrc  (Join-Path $ChromeOutDir "libLiteRtWebGpuAccelerator.pdb")
}

Write-Host "OK. Now launch: $ChromeOutDir\chrome.exe --no-sandbox --enable-features=WebMachineLearningNeuralNetwork <test-url>"
