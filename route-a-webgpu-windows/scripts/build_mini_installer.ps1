# One-shot: stage Bazel-built accelerator DLL + prebuilt Dawn DLL into a
# Chromium out dir, then build mini_installer.exe.
#
# Usage (typical):
#   .\build_mini_installer.ps1 -ChromeOutDir "C:\Users\junweifu\workspace\chromium\src\out\Release"
#
# Prereqs:
#   - `build_accelerator_dll.ps1 -Mode opt` succeeded → libLiteRtWebGpuAccelerator.dll in bazel-bin
#   - `build_dawn.ps1` succeeded → webgpu_dawn.dll in $WebnnDir\_dawn_prebuilt_win\lib\
#   - Chromium out dir is `is_official_build = true`-ish (uncompressed_archive off) or the debug
#     assert in mini_installer/BUILD.gn will refuse to build (see BUILD.gn line ~140).
#   - `chrome.release` already lists `webgpu_dawn.dll` (patched in Chromium source tree).
#
# Output: $ChromeOutDir\mini_installer.exe (a few hundred MB)

param(
    [Parameter(Mandatory=$true)] [string]$ChromeOutDir,
    [string]$ChromiumSrc = "C:\Users\junweifu\workspace\chromium\src",
    [string]$WebnnDir    = "C:\Users\junweifu\workspace\webnn",
    [switch]$SkipStage,        # skip DLL copy step (use whatever is already in out dir)
    [switch]$SkipBuild         # only stage, don't run autoninja
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ChromeOutDir)) { throw "Chrome out dir not found: $ChromeOutDir" }

$LITERT = Join-Path $ChromiumSrc "third_party\litert\src"
$dllSrc = Join-Path $LITERT "bazel-bin\litert\runtime\accelerators\gpu\libLiteRtWebGpuAccelerator.dll"
$dawnSrc = Join-Path $WebnnDir "_dawn_prebuilt_win\lib\webgpu_dawn.dll"

if (-not $SkipStage) {
    foreach ($f in @($dllSrc, $dawnSrc)) {
        if (-not (Test-Path $f)) { throw "Source missing: $f" }
    }

    function Copy-Overwrite($src, $tgt) {
        if (Test-Path $tgt) {
            try { Rename-Item $tgt "$tgt.old" -Force -ErrorAction Stop } catch {}
        }
        Copy-Item -Force $src $tgt
        $size = (Get-Item $tgt).Length
        Write-Host ("  staged {0}  ({1:N0} bytes)" -f (Split-Path $tgt -Leaf), $size)
    }

    Write-Host "Staging DLLs into $ChromeOutDir"
    Copy-Overwrite $dllSrc  (Join-Path $ChromeOutDir "libLiteRtWebGpuAccelerator.dll")
    Copy-Overwrite $dawnSrc (Join-Path $ChromeOutDir "webgpu_dawn.dll")

    # Clean up any dangling .old renamed during a previous run.
    Get-ChildItem $ChromeOutDir -Filter "*.dll.old" -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

# Verify chrome.release manifest lists webgpu_dawn.dll.
$manifest = Join-Path $ChromiumSrc "chrome\installer\mini_installer\chrome.release"
if (-not (Select-String -Path $manifest -Pattern "^webgpu_dawn\.dll:" -Quiet)) {
    Write-Warning "chrome.release does not list webgpu_dawn.dll — mini_installer archive will NOT include it."
    Write-Warning "  Apply the chrome.release edit before building. See route-a-webgpu-windows README."
}
if (-not (Select-String -Path $manifest -Pattern "^libLiteRtWebGpuAccelerator\.dll:" -Quiet)) {
    Write-Warning "chrome.release does not list libLiteRtWebGpuAccelerator.dll (upstream regression?)."
}

if ($SkipBuild) {
    Write-Host "SkipBuild set — done after staging."
    exit 0
}

# Locate autoninja (depot_tools) or ninja.
$ninja = $null
foreach ($cand in @("autoninja.bat", "autoninja", "ninja.exe", "ninja")) {
    $c = Get-Command $cand -ErrorAction SilentlyContinue
    if ($c) { $ninja = $c.Source; break }
}
if (-not $ninja) { throw "autoninja/ninja not found in PATH (add depot_tools)" }

Write-Host ""
Write-Host "Building mini_installer with $ninja"
Set-Location $ChromiumSrc
& $ninja -C $ChromeOutDir mini_installer
$rc = $LASTEXITCODE
if ($rc -ne 0) { throw "ninja failed with exit code $rc" }

$installer = Join-Path $ChromeOutDir "mini_installer.exe"
if (Test-Path $installer) {
    $mb = [math]::Round((Get-Item $installer).Length / 1MB, 1)
    Write-Host ""
    Write-Host "OK: $installer ($mb MB)"
    Write-Host "Copy to test machine, run to install Chrome; the accelerator + Dawn DLLs"
    Write-Host "will be extracted into the versioned dir (…\Chromium\Application\<version>\)."
} else {
    throw "mini_installer.exe did not appear in $ChromeOutDir"
}
