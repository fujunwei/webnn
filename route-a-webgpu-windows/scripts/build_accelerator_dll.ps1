# Build the Chromium-libc++-ABI-compatible LiteRtWebGpuAccelerator.dll.
#
# Prereqs (see README §Setup):
#   1. Chromium checkout built once in Release
#   2. libc++.lib packaged via scripts/build_libcxx_lib.ps1
#   3. Dawn prebuilt tree via scripts/build_dawn.ps1
#   4. All patches applied to $LITERT and $ML_DRIFT (see README §Apply patches)
#   5. First-time-only: run scripts/patch_bazel_toolchain.py after the FIRST
#      failed build (which will have created the local_config_cc/BUILD file).

param(
    [ValidateSet("dbg","opt")] [string]$Mode = "dbg",
    [string]$ChromiumSrc = "C:\Users\fujun\workspace\chromium\src",
    [string]$WebnnDir    = "C:\Users\fujun\workspace\webnn",
    [string]$MlDrift     = "C:\Users\fujun\workspace\chromium\src\third_party\ml-drift"
)

$ErrorActionPreference = "Continue"

# Refresh PATH from the registry (Machine + User) so freshly installed tools
# (bazel, python) are found even when the invoking shell has a stale env.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$LITERT = Join-Path $ChromiumSrc "third_party\litert\src"
if (-not (Test-Path $LITERT)) { throw "litert src not found at $LITERT" }

# Shell INCLUDE must contain libc++ header dirs so windows_cc_configure.bzl
# picks them up during Bazel's cc_toolchain auto-detection. (Any MSVC INCLUDE
# paths from vcvars are appended by Bazel automatically.)
$env:INCLUDE = @(
    (Join-Path $ChromiumSrc "third_party\libc++\src\include"),
    (Join-Path $ChromiumSrc "third_party\libc++abi\src\include"),
    (Join-Path $ChromiumSrc "buildtools\third_party\libc++")
) -join ";"
$env:BAZEL_LLVM    = (Join-Path $ChromiumSrc "third_party\llvm-build\Release+Asserts").Replace("\","/")
$env:USE_CLANG_CL  = "1"
$env:DAWN_PREBUILT_DIR = (Join-Path $WebnnDir "_dawn_prebuilt_win")

Set-Location $LITERT
Write-Host "Building //litert/runtime/accelerators/gpu:libLiteRtWebGpuAccelerator (-c $Mode)"
Write-Host "  Chromium src: $ChromiumSrc"
Write-Host "  ml-drift:     $MlDrift"
Write-Host "  DAWN dir:     $env:DAWN_PREBUILT_DIR"
Write-Host ""

$logPath = Join-Path $WebnnDir "build_crcxx_$Mode.log"
# NOTE: ml_drift is wired via local_repository in WORKSPACE (with repo_mapping
# @fp16 -> @FP16). Do NOT pass --override_repository here: it would replace the
# local_repository definition and drop the repo_mapping, breaking @fp16 deps.
bazel build --config=windows --config=crcxx_win --check_visibility=false `
    "--shell_executable=C:/Program Files/Git/bin/bash.exe" `
    -c $Mode `
    //litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dll 2>&1 |
    Tee-Object -FilePath $logPath

$rc = $LASTEXITCODE
Write-Host "Bazel exit code: $rc  (log: $logPath)"
if ($rc -ne 0) {
    Write-Host "Hint: if this is the first-ever build attempt, run scripts\patch_bazel_toolchain.py"
    Write-Host "      then re-run this script. See README §Setup for details."
}
exit $rc
